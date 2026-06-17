/*
 * kraskov_gpu.cu — CUDA implementation of the KSG mutual-information estimator.
 *
 * The CPU implementation (kraskov.cpp / kraskov_wrap) uses a bucket-sort
 * spatial index to achieve O(N log N) per atom pair.  On GPU the index
 * structures are impractical to port because each pair would need its own
 * independent bucket arrays and the dynamic linked-list traversals serialise
 * threads.  Instead we use a brute-force O(N²) scan per pair: for N ≤ ~5000
 * the arithmetic throughput of the GPU more than compensates for the extra
 * work, and the N² pairs are embarrassingly parallel across blocks.
 *
 * Algorithm implemented: KSG Algorithm 2 (mir_xnyn / Kraskov et al. 2004).
 * This matches what kraskov_wrap() calls on the CPU.
 *
 * Each CUDA block handles one unique atom pair (a1, a2).
 * Threads within the block divide the N query frames.
 * Three passes over the N reference frames per query:
 *   1. Find eps_k — the k-th smallest Chebyshev distance in the 6-D joint
 *      space — using a per-thread max-buffer of K elements.
 *   2. Scan all j ≤ eps_k in joint space to extract per-marginal radii
 *      epsx and epsy.
 *   3. Count marginal neighbours nx2, ny2 within epsx / epsy respectively.
 *
 * The GPU k limit is 128.  For k > 128, kraskov_gpu_probe() still returns
 * true but the caller (kraskov_corrmatrix in correlation_core.cpp) will
 * fall back to CPU when k > 128 is requested.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kraskov_gpu.h"

/* ------------------------------------------------------------------ */
/* Device helpers                                                       */
/* ------------------------------------------------------------------ */

static constexpr int BLOCK_DIM = 128;
static constexpr int K_MAX     = 128;

/* Inline Chebyshev helpers operate on interleaved component data. */

__device__ __forceinline__
double joint_chebyshev(const double* __restrict__ dx,
                        int nframes,
                        const double qx[3], const double qy[3],
                        int b1, int b2, int j)
{
    double dd = 0.0;
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        double tx = fabs(qx[d] - dx[(b1 + d) * nframes + j]);
        double ty = fabs(qy[d] - dx[(b2 + d) * nframes + j]);
        if (tx > dd) dd = tx;
        if (ty > dd) dd = ty;
    }
    return dd;
}

__device__ __forceinline__
double x_chebyshev(const double* __restrict__ dx,
                    int nframes,
                    const double qx[3],
                    int b1, int j)
{
    double dd = 0.0;
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        double t = fabs(qx[d] - dx[(b1 + d) * nframes + j]);
        if (t > dd) dd = t;
    }
    return dd;
}

__device__ __forceinline__
double y_chebyshev(const double* __restrict__ dx,
                    int nframes,
                    const double qy[3],
                    int b2, int j)
{
    double dd = 0.0;
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        double t = fabs(qy[d] - dx[(b2 + d) * nframes + j]);
        if (t > dd) dd = t;
    }
    return dd;
}

/* ------------------------------------------------------------------ */
/* Main kernel                                                          */
/* ------------------------------------------------------------------ */

/*
 * d_x      : normalised coordinates, layout [(natoms*3) * nframes], component-major.
 *            d_x[comp * nframes + frame] — same layout as kr->x[comp][frame].
 * d_psi    : digamma table, 1-indexed, size nframes+1.  d_psi[0] == 0 (from calloc).
 * phi_K    : psi[K] - 1/K  (precomputed on host).
 * psi_N    : psi[nframes]  (precomputed on host).
 * d_a1/a2  : atom index arrays, one entry per pair.
 * npairs   : total number of unique pairs.
 * d_out    : MI output per pair.
 */
__global__ void kraskov_mi_kernel(
        const double* __restrict__ d_x,
        const double* __restrict__ d_psi,
        double        phi_K,
        double        psi_N,
        const int*    d_a1,
        const int*    d_a2,
        int           nframes,
        int           K,
        double*       d_out)
{
    extern __shared__ double smem[];   /* blockDim.x doubles for reduction */

    const int pair = blockIdx.x;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    const int b1 = d_a1[pair] * 3;
    const int b2 = d_a2[pair] * 3;

    double partial = 0.0;

    for (int qi = tid; qi < nframes; qi += bdim) {

        /* Load query coordinates */
        double qx[3], qy[3];
        for (int d = 0; d < 3; ++d) {
            qx[d] = d_x[(b1 + d) * nframes + qi];
            qy[d] = d_x[(b2 + d) * nframes + qi];
        }

        /* ---- Pass 1: k-th smallest joint Chebyshev distance ---- *
         * Keep a max-buffer of K smallest distances seen so far.    *
         * buf_max is the current worst (largest) element: when a    *
         * new distance beats it, replace it and recompute buf_max.  */
        double buf[K_MAX];
        for (int k = 0; k < K; ++k) buf[k] = 1e30;
        double buf_max     = 1e30;
        int    buf_max_idx = 0;

        for (int j = 0; j < nframes; ++j) {
            if (j == qi) continue;
            const double dd = joint_chebyshev(d_x, nframes, qx, qy, b1, b2, j);
            if (dd < buf_max) {
                buf[buf_max_idx] = dd;
                buf_max = buf[0]; buf_max_idx = 0;
                for (int k = 1; k < K; ++k) {
                    if (buf[k] > buf_max) { buf_max = buf[k]; buf_max_idx = k; }
                }
            }
        }
        const double eps_k = buf_max;   /* k-th order statistic */

        /* ---- Pass 2: marginal radii from neighbours ≤ eps_k ---- */
        double epsx = 0.0, epsy = 0.0;
        for (int j = 0; j < nframes; ++j) {
            if (j == qi) continue;
            const double dd = joint_chebyshev(d_x, nframes, qx, qy, b1, b2, j);
            if (dd <= eps_k) {
                const double ddx = x_chebyshev(d_x, nframes, qx, b1, j);
                const double ddy = y_chebyshev(d_x, nframes, qy, b2, j);
                if (ddx > epsx) epsx = ddx;
                if (ddy > epsy) epsy = ddy;
            }
        }

        /* ---- Pass 3: count marginal neighbours ---- */
        int nx2 = 0, ny2 = 0;
        for (int j = 0; j < nframes; ++j) {
            if (j == qi) continue;
            if (x_chebyshev(d_x, nframes, qx, b1, j) <= epsx) ++nx2;
            if (y_chebyshev(d_x, nframes, qy, b2, j) <= epsy) ++ny2;
        }

        /* psi[0] == 0.0 (calloc); handles the degenerate zero-count case. */
        partial += d_psi[nx2] + d_psi[ny2];
    }

    /* ---- Block-level reduction ---- */
    smem[tid] = partial;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const double dxy2 = smem[0] / nframes;
        d_out[pair] = psi_N + phi_K - dxy2;
    }
}

/* ------------------------------------------------------------------ */
/* Host-side helpers                                                    */
/* ------------------------------------------------------------------ */

static void cuda_check(cudaError_t err, const char* ctx)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", ctx, cudaGetErrorString(err));
        exit(1);
    }
}

bool kraskov_gpu_probe()
{
    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err != cudaSuccess || ndev == 0) return false;
    /* Check that the first device has compute capability ≥ 3.5 */
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return (prop.major > 3) || (prop.major == 3 && prop.minor >= 5);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void kraskov_corrmatrix_gpu(const t_kraskov* kr, int natoms, double* mat, int k)
{
    const int N      = kr->N;
    const int ncomp  = natoms * 3;       /* total components = natoms * DIM */
    const int npairs = natoms * (natoms - 1) / 2;

    /* Print device info once */
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        fprintf(stderr, "GPU: %s  (sm_%d%d, %.0f MB)\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1e6);
    }

    /* ---- Flatten kr->x[comp][frame] → host_x[comp*N + frame] ---- */
    const size_t x_bytes = (size_t)ncomp * N * sizeof(double);
    double* host_x = (double*)malloc(x_bytes);
    if (!host_x) { fprintf(stderr, "kraskov_gpu: malloc failed\n"); exit(1); }
    for (int c = 0; c < ncomp; ++c)
        memcpy(host_x + (size_t)c * N, kr->x[c], N * sizeof(double));

    /* ---- Build pair lists ---- */
    int* host_a1 = (int*)malloc(npairs * sizeof(int));
    int* host_a2 = (int*)malloc(npairs * sizeof(int));
    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < a1; ++a2) {
                host_a1[idx] = a1;
                host_a2[idx] = a2;
                ++idx;
            }
    }

    /* ---- Precompute phi_K and psi_N on host ---- */
    const double phi_K = kr->psi[k] - 1.0 / k;   /* psi[K] - 1/K  (mir_xnyn formula) */
    const double psi_N = kr->psi[N];

    /* ---- Allocate device memory ---- */
    double *d_x = nullptr, *d_psi = nullptr, *d_out = nullptr;
    int    *d_a1 = nullptr, *d_a2 = nullptr;

    cuda_check(cudaMalloc(&d_x,   x_bytes),                           "malloc d_x");
    cuda_check(cudaMalloc(&d_psi, (N + 1) * sizeof(double)),          "malloc d_psi");
    cuda_check(cudaMalloc(&d_out, npairs * sizeof(double)),            "malloc d_out");
    cuda_check(cudaMalloc(&d_a1,  npairs * sizeof(int)),               "malloc d_a1");
    cuda_check(cudaMalloc(&d_a2,  npairs * sizeof(int)),               "malloc d_a2");

    cuda_check(cudaMemcpy(d_x,   host_x,      x_bytes,                cudaMemcpyHostToDevice), "copy x");
    cuda_check(cudaMemcpy(d_psi, kr->psi,    (N + 1) * sizeof(double), cudaMemcpyHostToDevice), "copy psi");
    cuda_check(cudaMemcpy(d_a1,  host_a1,    npairs * sizeof(int),    cudaMemcpyHostToDevice), "copy a1");
    cuda_check(cudaMemcpy(d_a2,  host_a2,    npairs * sizeof(int),    cudaMemcpyHostToDevice), "copy a2");

    /* ---- Launch kernel ---- */
    const size_t smem_bytes = BLOCK_DIM * sizeof(double);
    fprintf(stderr, "GPU Kraskov: %d pairs, N=%d, k=%d, blocks=%d threads=%d\n",
            npairs, N, k, npairs, BLOCK_DIM);

    kraskov_mi_kernel<<<npairs, BLOCK_DIM, smem_bytes>>>(
            d_x, d_psi, phi_K, psi_N, d_a1, d_a2, N, k, d_out);

    cuda_check(cudaGetLastError(),        "kernel launch");
    cuda_check(cudaDeviceSynchronize(),   "kernel sync");

    /* ---- Copy results back and fill matrix ---- */
    double* host_out = (double*)malloc(npairs * sizeof(double));
    cuda_check(cudaMemcpy(host_out, d_out, npairs * sizeof(double), cudaMemcpyDeviceToHost), "copy out");

    /* Diagonal = 2000 (sentinel for pearsify, matching CPU convention) */
    for (int i = 0; i < natoms; ++i)
        mat[i * natoms + i] = 2000.0;

    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < a1; ++a2, ++idx)
                mat[a1 * natoms + a2] = mat[a2 * natoms + a1] = host_out[idx];
    }

    /* ---- Cleanup ---- */
    cudaFree(d_x); cudaFree(d_psi); cudaFree(d_out); cudaFree(d_a1); cudaFree(d_a2);
    free(host_x); free(host_a1); free(host_a2); free(host_out);
}
