/*
 * gcmi_metal.mm — Metal GPU backend for the GCMI estimator.
 *
 * Each threadgroup handles one atom pair.
 * 64 threads accumulate partial 6x6 outer-product sums, then thread 0
 * reduces, computes determinants (Sarrus for 3x3, LU for 6x6), and
 * writes MI = 0.5*(log det Sigma_a1 + log det Sigma_a2 - log det Sigma_joint).
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "gcmi_metal.h"

/* ------------------------------------------------------------------ */
/* Metal Shading Language kernel                                        */
/* ------------------------------------------------------------------ */

static const char* kGcmiMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

/*
 * Buffers:
 *   0 — d_z      : float[ncomp * N], component-major copula z-scores
 *   1 — d_a1     : int[npairs]
 *   2 — d_a2     : int[npairs]
 *   3 — d_out    : float[npairs]
 *   4 — params   : int[2] = {N, natoms}
 *
 * Threadgroup memory: float[64 * 36] — 64 threads x 36 elements (6x6 matrix)
 */

float det3_gcmi(float m0,float m1,float m2,
                float m3,float m4,float m5,
                float m6,float m7,float m8) {
    return m0*(m4*m8 - m5*m7) - m1*(m3*m8 - m5*m6) + m2*(m3*m7 - m4*m6);
}

float det6_gcmi(thread float* a) {
    float m[36];
    for(int i=0;i<36;i++) m[i]=a[i];
    float det=1.0f;
    for(int c=0;c<6;c++){
        int piv=c; float best=abs(m[c*6+c]);
        for(int r=c+1;r<6;r++){float v=abs(m[r*6+c]);if(v>best){best=v;piv=r;}}
        if(best<1e-10f) return 0.0f;
        if(piv!=c){
            for(int j=0;j<6;j++){float t=m[c*6+j];m[c*6+j]=m[piv*6+j];m[piv*6+j]=t;}
            det=-det;
        }
        det*=m[c*6+c]; float d=m[c*6+c];
        for(int r=c+1;r<6;r++){
            float f=m[r*6+c]/d;
            for(int j=c+1;j<6;j++) m[r*6+j]-=f*m[c*6+j];
        }
    }
    return det;
}

kernel void gcmi_cov(
    device const float* d_z    [[buffer(0)]],
    device const int*   d_a1   [[buffer(1)]],
    device const int*   d_a2   [[buffer(2)]],
    device float*       d_out  [[buffer(3)]],
    device const int*   params [[buffer(4)]],
    threadgroup float*  smem   [[threadgroup(0)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint bdim [[threads_per_threadgroup]])
{
    const int N      = params[0];
    const int b1     = d_a1[pair] * 3;
    const int b2     = d_a2[pair] * 3;

    /* Each thread accumulates a partial 6x6 covariance (row-major, 36 floats) */
    float partial[36];
    for(int i=0;i<36;i++) partial[i]=0.0f;

    for(int f=(int)tid; f<N; f+=(int)bdim){
        float v[6];
        for(int d=0;d<3;d++){
            v[d]   = d_z[(b1+d)*N + f];
            v[d+3] = d_z[(b2+d)*N + f];
        }
        for(int r=0;r<6;r++)
            for(int c=0;c<6;c++)
                partial[r*6+c] += v[r]*v[c];
    }

    /* Store partial sums into threadgroup memory: layout [tid*36 + elem] */
    threadgroup float* mySlot = smem + tid * 36;
    for(int i=0;i<36;i++) mySlot[i] = partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Thread 0 reduces */
    if(tid == 0){
        float cov[36];
        for(int i=0;i<36;i++) cov[i]=0.0f;
        for(uint t=0;t<bdim;t++){
            threadgroup float* slot = smem + t*36;
            for(int i=0;i<36;i++) cov[i]+=slot[i];
        }
        float invN = 1.0f/N;
        for(int i=0;i<36;i++) cov[i]*=invN;

        /* 3x3 block a1 (rows/cols 0..2) */
        float d1 = det3_gcmi(cov[0],cov[1],cov[2],
                              cov[6],cov[7],cov[8],
                              cov[12],cov[13],cov[14]);
        /* 3x3 block a2 (rows/cols 3..5) */
        float d2 = det3_gcmi(cov[21],cov[22],cov[23],
                              cov[27],cov[28],cov[29],
                              cov[33],cov[34],cov[35]);
        float dj = det6_gcmi(cov);

        float mi = 0.0f;
        if(d1>0.0f && d2>0.0f && dj>0.0f)
            mi = 0.5f*(log(d1)+log(d2)-log(dj));

        d_out[pair] = mi;
    }
}
)MSL";

/* ------------------------------------------------------------------ */
/* Host implementation                                                  */
/* ------------------------------------------------------------------ */

static constexpr int GCMI_THREADS = 64;

bool gcmi_metal_probe()
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
}

void gcmi_corrmatrix_metal(const double* z, int natoms, int N, double* mat)
{
    const int ncomp  = natoms * 3;
    const int npairs = natoms * (natoms - 1) / 2;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
        fprintf(stderr, "gcmi_metal: no Metal device found\n");
        return;
    }
    fprintf(stderr, "Metal GPU (GCMI): %s\n", [[dev name] UTF8String]);

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kGcmiMetalSource];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_15_0
    opts.mathMode = MTLMathModeFast;
#else
    opts.fastMathEnabled = YES;
#endif

    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "gcmi_metal compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }
    id<MTLFunction>             fn  = [lib  newFunctionWithName:@"gcmi_cov"];
    id<MTLComputePipelineState> pso = [dev  newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "gcmi_metal pipeline error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    /* Convert z to float */
    std::vector<float> host_z((size_t)ncomp * N);
    for (size_t i = 0; i < host_z.size(); ++i)
        host_z[i] = (float)z[i];

    /* Build symmetric pair index arrays */
    std::vector<int> host_a1(npairs), host_a2(npairs);
    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < a1; ++a2)
                { host_a1[idx] = a1; host_a2[idx] = a2; ++idx; }
    }

    int params[2] = { N, natoms };

    auto mkbuf = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
        return [dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> buf_z      = mkbuf(host_z.data(), host_z.size() * sizeof(float));
    id<MTLBuffer> buf_a1     = mkbuf(host_a1.data(), host_a1.size() * sizeof(int));
    id<MTLBuffer> buf_a2     = mkbuf(host_a2.data(), host_a2.size() * sizeof(int));
    id<MTLBuffer> buf_out    = [dev newBufferWithLength:npairs * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_params = mkbuf(params, sizeof(params));

    fprintf(stderr, "  Dispatching GCMI Metal kernel: %d threadgroups x %d threads\n",
            npairs, GCMI_THREADS);

    id<MTLCommandBuffer>         cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd   computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_z      offset:0 atIndex:0];
    [enc setBuffer:buf_a1     offset:0 atIndex:1];
    [enc setBuffer:buf_a2     offset:0 atIndex:2];
    [enc setBuffer:buf_out    offset:0 atIndex:3];
    [enc setBuffer:buf_params offset:0 atIndex:4];

    /* Threadgroup memory: GCMI_THREADS * 36 floats */
    [enc setThreadgroupMemoryLength:GCMI_THREADS * 36 * sizeof(float) atIndex:0];

    MTLSize tgSize   = MTLSizeMake(GCMI_THREADS, 1, 1);
    MTLSize gridSize = MTLSizeMake(npairs, 1, 1);
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusError) {
        fprintf(stderr, "gcmi_metal command error: %s\n",
                [[[cmd error] localizedDescription] UTF8String]);
        return;
    }

    /* Read results */
    const float* out = (const float*)[buf_out contents];

    for (int i = 0; i < natoms; ++i)
        mat[i * natoms + i] = 2000.0;

    {
        int idx = 0;
        for (int a1 = 0; a1 < natoms; ++a1)
            for (int a2 = 0; a2 < a1; ++a2, ++idx)
                mat[a1 * natoms + a2] = mat[a2 * natoms + a1] = (double)out[idx];
    }
}
