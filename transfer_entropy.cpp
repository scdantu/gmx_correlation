/*
 * transfer_entropy.cpp — Frenzel-Pompe CMI-based transfer entropy estimator.
 *
 * TE(a2->a1, lag) = I(X_{a1,t+lag} ; X_{a2,t} | X_{a1,t})
 *
 * Estimated via the Frenzel & Pompe (2007) KSG extension:
 *   Build 9D joint vectors q_i = [Xf(3D), Xp(3D), Yp(3D)]
 *   Xf = x[a1][i+lag], Xp = x[a1][i], Yp = x[a2][i].
 *   Use Chebyshev (L-inf) k-NN in 9D to get eps_k per query.
 *   Count subspace neighbours (strictly < eps_k):
 *     n_xfxp: 6D subspace (Xf,Xp), components 0..5
 *     n_ypxp: 6D subspace (Yp,Xp), components 3..8 (Xp,Yp)
 *     n_xp:   3D subspace (Xp),    components 3..5
 *   TE_i = psi(k) + psi(n_xp+1) - psi(n_xfxp+1) - psi(n_ypxp+1)
 *   Average over i in 0..N_valid-1.
 */

#include "transfer_entropy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GMX_CORRELATION_USE_METAL
#include "te_metal.h"
#endif

/* ------------------------------------------------------------------ */
/* Digamma table                                                        */
/* ------------------------------------------------------------------ */

/* psi[n] for n=0..N.
 * psi[1] = -EULER_GAMMA, psi[n+1] = psi[n] + 1/n  (n>=1).
 * psi[0] is set to 0 as a safe default (unused in practice). */
static std::vector<double> digamma_table(int N)
{
    std::vector<double> psi(N + 1, 0.0);
    psi[1] = -0.57721566490153;
    for (int i = 1; i < N; ++i)
        psi[i + 1] = psi[i] + 1.0 / i;
    return psi;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

bool te_gpu_available()
{
#ifdef GMX_CORRELATION_USE_METAL
    return te_metal_probe();
#else
    return false;
#endif
}

void te_matrix(const t_traj* traj, double* mat, int k, int lag,
               bool use_gpu, int nthreads)
{
    const int natoms  = traj->natoms;
    const int N       = traj->nframes;
    const int N_valid = N - lag;

    std::fill(mat, mat + (size_t)natoms * natoms, 0.0);

    if (N_valid <= k)
    {
        fprintf(stderr, "Warning: TE: N_valid=%d <= k=%d; skipping.\n", N_valid, k);
        return;
    }

#ifdef GMX_CORRELATION_USE_METAL
    if (use_gpu)
    {
        if (!te_metal_probe())
            fprintf(stderr, "Note: no Metal device found — falling back to CPU.\n");
        else
        {
            t_traj mutableTraj = *traj;
            t_kraskov kr;
            kraskov_prepare(&mutableTraj, &kr);
            te_corrmatrix_metal(&kr, natoms, mat, k, lag);
            kraskov_done(&kr);
            return;
        }
    }
#else
    (void)use_gpu;
#endif

    /* Precompute digamma table indexed by count (0..N_valid) */
    std::vector<double> psi = digamma_table(N_valid);
    const double psi_k = psi[k];

#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
    fprintf(stderr, "  CPU threads: %d\n", omp_get_max_threads());
#else
    (void)nthreads;
    fprintf(stderr, "  CPU threads: 1\n");
#endif

    /* Total ordered pairs (excluding diagonal) */
    const long npairs  = (long)natoms * (natoms - 1);
    long       done    = 0;
    int        last_pct = -1;

#pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (int a1 = 0; a1 < natoms; ++a1)
    {
        for (int a2 = 0; a2 < natoms; ++a2)
        {
            if (a1 == a2) continue;

            double te_sum = 0.0;

            for (int qi = 0; qi < N_valid; ++qi)
            {
                /* Build 9D query:
                 *   q[0..2] = Xf = x[a1][qi+lag]
                 *   q[3..5] = Xp = x[a1][qi]
                 *   q[6..8] = Yp = x[a2][qi]
                 */
                double q[9];
                for (int d = 0; d < 3; ++d)
                {
                    q[d]     = traj->x[a1][qi + lag][d];
                    q[d + 3] = traj->x[a1][qi][d];
                    q[d + 6] = traj->x[a2][qi][d];
                }

                /* Pass 1: find k-th nearest neighbour in 9D Chebyshev.
                 * Use a max-buffer of k elements. */
                std::vector<double> buf(k, 1e30);
                double buf_max     = 1e30;
                int    buf_max_idx = 0;

                for (int j = 0; j < N_valid; ++j)
                {
                    if (j == qi) continue;
                    double dist = 0.0;
                    for (int d = 0; d < 3; ++d)
                    {
                        dist = std::max(dist, std::fabs(q[d]     - traj->x[a1][j + lag][d]));
                        dist = std::max(dist, std::fabs(q[d + 3] - traj->x[a1][j][d]));
                        dist = std::max(dist, std::fabs(q[d + 6] - traj->x[a2][j][d]));
                    }
                    if (dist < buf_max)
                    {
                        buf[buf_max_idx] = dist;
                        buf_max     = buf[0];
                        buf_max_idx = 0;
                        for (int ki = 1; ki < k; ++ki)
                            if (buf[ki] > buf_max) { buf_max = buf[ki]; buf_max_idx = ki; }
                    }
                }
                const double eps_k = buf_max;

                /* Pass 2: count subspace neighbours strictly < eps_k */
                int n_xfxp = 0, n_ypxp = 0, n_xp = 0;
                for (int j = 0; j < N_valid; ++j)
                {
                    if (j == qi) continue;

                    /* 6D (Xf,Xp) distance — components 0..5 */
                    double d_xfxp = 0.0;
                    for (int d = 0; d < 3; ++d)
                    {
                        d_xfxp = std::max(d_xfxp, std::fabs(q[d]     - traj->x[a1][j + lag][d]));
                        d_xfxp = std::max(d_xfxp, std::fabs(q[d + 3] - traj->x[a1][j][d]));
                    }
                    /* 6D (Yp,Xp) distance — components 3..8 */
                    double d_ypxp = 0.0;
                    for (int d = 0; d < 3; ++d)
                    {
                        d_ypxp = std::max(d_ypxp, std::fabs(q[d + 3] - traj->x[a1][j][d]));
                        d_ypxp = std::max(d_ypxp, std::fabs(q[d + 6] - traj->x[a2][j][d]));
                    }
                    /* 3D (Xp) distance — components 3..5 */
                    double d_xp = 0.0;
                    for (int d = 0; d < 3; ++d)
                        d_xp = std::max(d_xp, std::fabs(q[d + 3] - traj->x[a1][j][d]));

                    if (d_xfxp < eps_k) ++n_xfxp;
                    if (d_ypxp < eps_k) ++n_ypxp;
                    if (d_xp   < eps_k) ++n_xp;
                }

                /* Clamp indices into digamma table range */
                int idx_xfxp = std::min(n_xfxp, N_valid);
                int idx_ypxp = std::min(n_ypxp, N_valid);
                int idx_xp   = std::min(n_xp,   N_valid);

                te_sum += psi_k + psi[idx_xp + 1] - psi[idx_xfxp + 1] - psi[idx_ypxp + 1];
            }

            /* mat[a1*natoms + a2] = TE(a2->a1) */
            mat[a1 * natoms + a2] = te_sum / N_valid;

#pragma omp critical(te_prog)
            {
                const int pct = (int)(++done * 100 / npairs);
                if (pct > last_pct)
                {
                    last_pct = pct;
                    fprintf(stderr, "\r  Progress: %3d%%", pct);
                    fflush(stderr);
                }
            }
        }
    }

    fprintf(stderr, "\r  Progress: 100%%\n");
    fflush(stderr);
}
