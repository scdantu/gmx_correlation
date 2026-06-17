// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * te_metal.mm — Metal GPU backend for transfer entropy (Frenzel-Pompe CMI).
 *
 * Each threadgroup handles one ordered pair (a1, a2), a2->a1 TE.
 * 9D Chebyshev k-NN: q = [Xf(3D), Xp(3D), Yp(3D)]
 *   Xf = x[a1][i+lag], Xp = x[a1][i], Yp = x[a2][i]
 *
 * Per-frame contribution: psi(k) + psi(n_xp+1) - psi(n_xfxp+1) - psi(n_ypxp+1)
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "te_metal.h"

static const char* kTeMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

/*
 * Buffers:
 *   0 — d_x      : float[ncomp * N_total], component-major (ncomp = natoms*3)
 *   1 — d_psi    : float[N_valid+2], psi[0]=0, psi[1]=-gamma, psi[n+1]=psi[n]+1/n
 *   2 — d_a1     : int[npairs_ordered]
 *   3 — d_a2     : int[npairs_ordered]
 *   4 — d_out    : float[npairs_ordered]
 *   5 — params   : int[3] = {N_valid, K, N_total}
 * Threadgroup: float[bdim] for reduction
 */
kernel void te_cmi(
    device const float* d_x    [[buffer(0)]],
    device const float* d_psi  [[buffer(1)]],
    device const int*   d_a1   [[buffer(2)]],
    device const int*   d_a2   [[buffer(3)]],
    device float*       d_out  [[buffer(4)]],
    device const int*   params [[buffer(5)]],
    threadgroup float*  smem   [[threadgroup(0)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint bdim [[threads_per_threadgroup]])
{
    const int N_valid = params[0];
    const int K       = params[1];
    const int N_total = params[2];
    const int b1      = d_a1[pair] * 3;
    const int b2      = d_a2[pair] * 3;

    float partial = 0.0f;
    const float psi_k = d_psi[K];

    for(int qi=(int)tid; qi<N_valid; qi+=(int)bdim){
        /* Build 9D query */
        float qxf[3], qxp[3], qyp[3];
        for(int d=0;d<3;d++){
            qxf[d] = d_x[(b1+d)*N_total + qi + (N_total - N_valid)];
            /* lag = N_total - N_valid */
            /* Actually: qi+lag where lag=N_total-N_valid. But frames go 0..N_total-1
             * and N_valid = N_total - lag, so qi+lag = qi + (N_total - N_valid). */
            qxp[d] = d_x[(b1+d)*N_total + qi];
            qyp[d] = d_x[(b2+d)*N_total + qi];
        }

        /* Pass 1: k-NN in 9D Chebyshev */
        float buf[128];
        for(int ki=0;ki<K;ki++) buf[ki]=1e30f;
        float buf_max=1e30f; int buf_max_idx=0;

        for(int j=0;j<N_valid;j++){
            if(j==qi) continue;
            float dist=0.0f;
            int jlag = j + (N_total - N_valid);
            for(int d=0;d<3;d++){
                dist = max(dist, abs(qxf[d] - d_x[(b1+d)*N_total + jlag]));
                dist = max(dist, abs(qxp[d] - d_x[(b1+d)*N_total + j]));
                dist = max(dist, abs(qyp[d] - d_x[(b2+d)*N_total + j]));
            }
            if(dist < buf_max){
                buf[buf_max_idx]=dist;
                buf_max=buf[0]; buf_max_idx=0;
                for(int ki=1;ki<K;ki++) if(buf[ki]>buf_max){buf_max=buf[ki];buf_max_idx=ki;}
            }
        }
        const float eps_k = buf_max;

        /* Pass 2: count subspace neighbours strictly < eps_k */
        int n_xfxp=0, n_ypxp=0, n_xp=0;
        for(int j=0;j<N_valid;j++){
            if(j==qi) continue;
            int jlag = j + (N_total - N_valid);
            float d_xfxp=0.0f, d_ypxp=0.0f, d_xp=0.0f;
            for(int d=0;d<3;d++){
                d_xfxp = max(d_xfxp, abs(qxf[d] - d_x[(b1+d)*N_total + jlag]));
                d_xfxp = max(d_xfxp, abs(qxp[d] - d_x[(b1+d)*N_total + j]));
                d_ypxp = max(d_ypxp, abs(qyp[d] - d_x[(b2+d)*N_total + j]));
                d_ypxp = max(d_ypxp, abs(qxp[d] - d_x[(b1+d)*N_total + j]));
                d_xp   = max(d_xp,   abs(qxp[d] - d_x[(b1+d)*N_total + j]));
            }
            if(d_xfxp < eps_k) n_xfxp++;
            if(d_ypxp < eps_k) n_ypxp++;
            if(d_xp   < eps_k) n_xp++;
        }

        /* Clamp indices to psi table range */
        int idx_xfxp = min(n_xfxp, N_valid);
        int idx_ypxp = min(n_ypxp, N_valid);
        int idx_xp   = min(n_xp,   N_valid);

        partial += psi_k + d_psi[idx_xp+1] - d_psi[idx_xfxp+1] - d_psi[idx_ypxp+1];
    }

    smem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=bdim>>1; s>0; s>>=1){
        if(tid<s) smem[tid]+=smem[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0){
        d_out[pair] = smem[0] / (float)N_valid;
    }
}
)MSL";

static constexpr int TE_THREADS = 128;

bool te_metal_probe()
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
}

void te_corrmatrix_metal(const t_kraskov* kr, int natoms, double* mat, int k, int lag)
{
    const int N_total  = kr->N;
    const int N_valid  = N_total - lag;
    const int ncomp    = natoms * 3;
    /* All ordered pairs excluding diagonal */
    const int npairs   = natoms * (natoms - 1);

    if (N_valid <= k) {
        fprintf(stderr, "te_metal: N_valid=%d <= k=%d; skipping GPU path.\n", N_valid, k);
        return;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
        fprintf(stderr, "te_metal: no Metal device found\n");
        return;
    }
    fprintf(stderr, "Metal GPU (TE): %s\n", [[dev name] UTF8String]);

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kTeMetalSource];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_15_0
    opts.mathMode = MTLMathModeFast;
#else
    opts.fastMathEnabled = YES;
#endif

    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "te_metal compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }
    id<MTLFunction>             fn  = [lib  newFunctionWithName:@"te_cmi"];
    id<MTLComputePipelineState> pso = [dev  newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "te_metal pipeline error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    /* Convert kr->x (component-major) to float */
    std::vector<float> host_x((size_t)ncomp * N_total);
    for (int c = 0; c < ncomp; ++c)
        for (int f = 0; f < N_total; ++f)
            host_x[(size_t)c * N_total + f] = (float)kr->x[c][f];

    /* Digamma table for N_valid */
    std::vector<float> host_psi(N_valid + 2, 0.0f);
    host_psi[1] = -0.57721566490153f;
    for (int i = 1; i <= N_valid; ++i)
        host_psi[i + 1] = host_psi[i] + 1.0f / i;

    /* Build ordered pair index arrays */
    std::vector<int> host_a1(npairs), host_a2(npairs);
    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < natoms; ++a2)
                if (a1 != a2) { host_a1[idx] = a1; host_a2[idx] = a2; ++idx; }
    }

    int params[3] = { N_valid, k, N_total };

    auto mkbuf = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
        return [dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> buf_x      = mkbuf(host_x.data(),   host_x.size()   * sizeof(float));
    id<MTLBuffer> buf_psi    = mkbuf(host_psi.data(), host_psi.size() * sizeof(float));
    id<MTLBuffer> buf_a1     = mkbuf(host_a1.data(),  host_a1.size()  * sizeof(int));
    id<MTLBuffer> buf_a2     = mkbuf(host_a2.data(),  host_a2.size()  * sizeof(int));
    id<MTLBuffer> buf_out    = [dev newBufferWithLength:npairs * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_params = mkbuf(params, sizeof(params));

    fprintf(stderr, "  Dispatching TE Metal kernel: %d threadgroups x %d threads\n",
            npairs, TE_THREADS);

    id<MTLCommandBuffer>         cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd   computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_x      offset:0 atIndex:0];
    [enc setBuffer:buf_psi    offset:0 atIndex:1];
    [enc setBuffer:buf_a1     offset:0 atIndex:2];
    [enc setBuffer:buf_a2     offset:0 atIndex:3];
    [enc setBuffer:buf_out    offset:0 atIndex:4];
    [enc setBuffer:buf_params offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:TE_THREADS * sizeof(float) atIndex:0];

    MTLSize tgSize   = MTLSizeMake(TE_THREADS, 1, 1);
    MTLSize gridSize = MTLSizeMake(npairs, 1, 1);
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusError) {
        fprintf(stderr, "te_metal command error: %s\n",
                [[[cmd error] localizedDescription] UTF8String]);
        return;
    }

    const float* out = (const float*)[buf_out contents];
    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < natoms; ++a2)
                if (a1 != a2) { mat[a1 * natoms + a2] = (double)out[idx++]; }
    }
}
