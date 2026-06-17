/*
 * kraskov_metal.mm — Metal (macOS GPU) implementation of the KSG estimator.
 *
 * The Metal Shading Language kernel below implements the same three-pass
 * brute-force k-NN algorithm as kraskov_gpu.cu:
 *
 *   Pass 1 — Find eps_k: the k-th smallest Chebyshev distance in the 6-D
 *             joint space (atom-pair coordinates) using a per-thread max-buffer
 *             of K elements.
 *   Pass 2 — Scan all frames within eps_k to extract per-marginal radii
 *             epsx and epsy.
 *   Pass 3 — Count marginal neighbours nx2 (within epsx) and ny2 (within
 *             epsy) and accumulate psi[nx2] + psi[ny2].
 *
 * Each Metal threadgroup handles one atom pair; threads within the group
 * divide the N query frames.
 *
 * Precision note:
 *   Apple Silicon GPUs do not support float64 in Metal shaders.  All
 *   per-GPU arithmetic uses float32.  Input coordinates and the digamma
 *   table are converted from double to float on the host before transfer.
 *   The resulting MI values agree with CPU double to roughly 1e-4 — within
 *   the same range as the legacy/new CPU inter-implementation spread.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kraskov_metal.h"

/* ------------------------------------------------------------------ */
/* Metal Shading Language kernel — embedded as a raw string literal    */
/* The device compiles this at runtime via MTLDevice.makeLibrary.      */
/* ------------------------------------------------------------------ */

static const char* kKraskovMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

/*
 * Buffer indices:
 *   0 — d_x     : float[(natoms*3) * nframes], component-major
 *   1 — d_psi   : float[nframes+1], 1-indexed (index 0 == 0.0)
 *   2 — d_a1    : int[npairs]
 *   3 — d_a2    : int[npairs]
 *   4 — d_out   : float[npairs] (output MI per pair)
 *   5 — params  : int[2]  = {nframes, K}
 *   6 — scalars : float[2] = {phi_K, psi_N}
 * Threadgroup(0): float[threads_per_threadgroup] for reduction scratch
 */
kernel void kraskov_mi(
    device const float* d_x      [[buffer(0)]],
    device const float* d_psi    [[buffer(1)]],
    device const int*   d_a1     [[buffer(2)]],
    device const int*   d_a2     [[buffer(3)]],
    device float*       d_out    [[buffer(4)]],
    device const int*   params   [[buffer(5)]],
    device const float* scalars  [[buffer(6)]],
    threadgroup float*  smem     [[threadgroup(0)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint bdim [[threads_per_threadgroup]])
{
    const int   nframes = params[0];
    const int   K       = params[1];
    const float phi_K   = scalars[0];
    const float psi_N   = scalars[1];

    const int b1 = d_a1[pair] * 3;
    const int b2 = d_a2[pair] * 3;

    float partial = 0.0f;

    for (int qi = (int)tid; qi < nframes; qi += (int)bdim) {

        /* Load query coordinates (6-D: 3 from atom a1, 3 from atom a2) */
        float qx[3], qy[3];
        for (int d = 0; d < 3; ++d) {
            qx[d] = d_x[(b1 + d) * nframes + qi];
            qy[d] = d_x[(b2 + d) * nframes + qi];
        }

        /* ---- Pass 1: k-th smallest joint Chebyshev distance ---- *
         * Max-buffer of K smallest distances seen.  buf_max is the  *
         * current worst element; when a new distance beats it we    *
         * replace that slot and recompute buf_max.                  */
        float buf[128];      /* K ≤ 128 */
        for (int k = 0; k < K; ++k) buf[k] = 1e30f;
        float buf_max     = 1e30f;
        int   buf_max_idx = 0;

        for (int j = 0; j < nframes; ++j) {
            if (j == qi) continue;
            float dd = 0.0f;
            for (int d = 0; d < 3; ++d) {
                float tx = abs(qx[d] - d_x[(b1 + d) * nframes + j]);
                float ty = abs(qy[d] - d_x[(b2 + d) * nframes + j]);
                dd = max(dd, max(tx, ty));
            }
            if (dd < buf_max) {
                buf[buf_max_idx] = dd;
                buf_max = buf[0]; buf_max_idx = 0;
                for (int k = 1; k < K; ++k)
                    if (buf[k] > buf_max) { buf_max = buf[k]; buf_max_idx = k; }
            }
        }
        const float eps_k = buf_max;

        /* ---- Pass 2: per-marginal radii from neighbours ≤ eps_k ---- */
        float epsx = 0.0f, epsy = 0.0f;
        for (int j = 0; j < nframes; ++j) {
            if (j == qi) continue;
            float dd = 0.0f, ddx = 0.0f, ddy = 0.0f;
            for (int d = 0; d < 3; ++d) {
                float tx = abs(qx[d] - d_x[(b1 + d) * nframes + j]);
                float ty = abs(qy[d] - d_x[(b2 + d) * nframes + j]);
                ddx = max(ddx, tx);
                ddy = max(ddy, ty);
                dd  = max(dd,  max(tx, ty));
            }
            if (dd <= eps_k) {
                epsx = max(epsx, ddx);
                epsy = max(epsy, ddy);
            }
        }

        /* ---- Pass 3: count marginal neighbours ---- */
        int nx2 = 0, ny2 = 0;
        for (int j = 0; j < nframes; ++j) {
            if (j == qi) continue;
            float ddx = 0.0f, ddy = 0.0f;
            for (int d = 0; d < 3; ++d) {
                ddx = max(ddx, abs(qx[d] - d_x[(b1 + d) * nframes + j]));
                ddy = max(ddy, abs(qy[d] - d_x[(b2 + d) * nframes + j]));
            }
            if (ddx <= epsx) ++nx2;
            if (ddy <= epsy) ++ny2;
        }

        /* d_psi[0] == 0.0 (safe default for degenerate zero-count case) */
        partial += d_psi[nx2] + d_psi[ny2];
    }

    /* ---- Block-level reduction ---- */
    smem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float dxy2 = smem[0] / (float)nframes;
        d_out[pair] = psi_N + phi_K - dxy2;
    }
}
)MSL";

/* ------------------------------------------------------------------ */
/* Host implementation                                                  */
/* ------------------------------------------------------------------ */

static constexpr int METAL_THREADS = 128;

bool kraskov_metal_probe()
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
}

void kraskov_corrmatrix_metal(const t_kraskov* kr, int natoms, double* mat, int k)
{
    const int N      = kr->N;
    const int ncomp  = natoms * 3;
    const int npairs = natoms * (natoms - 1) / 2;

    /* ---- Acquire device ---- */
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
        fprintf(stderr, "kraskov_metal: no Metal device found\n");
        return;
    }
    fprintf(stderr, "Metal GPU: %s\n", [[dev name] UTF8String]);

    /* ---- Compile kernel from embedded source ---- */
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kKraskovMetalSource];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_15_0
    opts.mathMode = MTLMathModeFast;
#else
    opts.fastMathEnabled = YES;
#endif

    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "Metal compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }
    id<MTLFunction>             fn  = [lib  newFunctionWithName:@"kraskov_mi"];
    id<MTLComputePipelineState> pso = [dev  newComputePipelineStateWithFunction:fn
                                                                          error:&err];
    if (!pso) {
        fprintf(stderr, "Metal pipeline error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }

    id<MTLCommandQueue> queue = [dev newCommandQueue];

    /* ---- Convert kr->x and kr->psi to float ---- */
    std::vector<float> host_x((size_t)ncomp * N);
    for (int c = 0; c < ncomp; ++c)
        for (int f = 0; f < N; ++f)
            host_x[(size_t)c * N + f] = (float)kr->x[c][f];

    std::vector<float> host_psi(N + 1);
    for (int i = 0; i <= N; ++i)
        host_psi[i] = (float)kr->psi[i];

    /* ---- Build pair index arrays ---- */
    std::vector<int> host_a1(npairs), host_a2(npairs);
    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < a1; ++a2)
                { host_a1[idx] = a1; host_a2[idx] = a2; ++idx; }
    }

    /* ---- Precompute scalars ---- */
    const float phi_K = (float)(kr->psi[k] - 1.0 / k);
    const float psi_N = (float)kr->psi[N];

    int   params[2]  = { N, k };
    float scalars[2] = { phi_K, psi_N };

    /* ---- Allocate Metal buffers (shared CPU/GPU on Apple Silicon) ---- */
    auto mkbuf = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
        id<MTLBuffer> buf = [dev newBufferWithBytes:data
                                            length:bytes
                                           options:MTLResourceStorageModeShared];
        return buf;
    };

    id<MTLBuffer> buf_x      = mkbuf(host_x.data(),   host_x.size()   * sizeof(float));
    id<MTLBuffer> buf_psi    = mkbuf(host_psi.data(), host_psi.size() * sizeof(float));
    id<MTLBuffer> buf_a1     = mkbuf(host_a1.data(),  host_a1.size()  * sizeof(int));
    id<MTLBuffer> buf_a2     = mkbuf(host_a2.data(),  host_a2.size()  * sizeof(int));
    id<MTLBuffer> buf_out    = [dev newBufferWithLength:npairs * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_params  = mkbuf(params,   sizeof(params));
    id<MTLBuffer> buf_scalars = mkbuf(scalars, sizeof(scalars));

    /* ---- Encode and dispatch ---- */
    fprintf(stderr, "  Dispatching Metal kernel: %d threadgroups × %d threads\n",
            npairs, METAL_THREADS);

    using Clock = std::chrono::steady_clock;
    const auto t_dispatch = Clock::now();

    id<MTLCommandBuffer>        cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd  computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_x       offset:0 atIndex:0];
    [enc setBuffer:buf_psi     offset:0 atIndex:1];
    [enc setBuffer:buf_a1      offset:0 atIndex:2];
    [enc setBuffer:buf_a2      offset:0 atIndex:3];
    [enc setBuffer:buf_out     offset:0 atIndex:4];
    [enc setBuffer:buf_params  offset:0 atIndex:5];
    [enc setBuffer:buf_scalars offset:0 atIndex:6];

    [enc setThreadgroupMemoryLength:METAL_THREADS * sizeof(float) atIndex:0];

    MTLSize tgSize   = MTLSizeMake(METAL_THREADS, 1, 1);
    MTLSize gridSize = MTLSizeMake(npairs, 1, 1);
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    const double kernel_s =
        std::chrono::duration<double>(Clock::now() - t_dispatch).count();

    if ([cmd status] == MTLCommandBufferStatusError) {
        fprintf(stderr, "Metal command buffer error: %s\n",
                [[[cmd error] localizedDescription] UTF8String]);
        return;
    }
    fprintf(stderr, "  Kernel execution: %.2f s\n", kernel_s);

    /* ---- Read results and fill matrix ---- */
    const float* out = (const float*)[buf_out contents];

    for (int i = 0; i < natoms; ++i)
        mat[i * natoms + i] = 2000.0;   /* diagonal sentinel */

    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < a1; ++a2, ++idx)
                mat[a1 * natoms + a2] = mat[a2 * natoms + a1] = (double)out[idx];
    }
}
