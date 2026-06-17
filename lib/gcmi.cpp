// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * gcmi.cpp — Gaussian Copula Mutual Information estimator.
 *
 * Implements the Ince et al. (2017) GCMI method:
 *   1. Van der Waerden copula transform (rank -> probit) per component.
 *   2. Build 6x6 covariance of the copula z-scores for each atom pair.
 *   3. MI = 0.5*(log det Sigma_a1 + log det Sigma_a2 - log det Sigma_joint).
 *
 * Reference: Ince et al., eLife 6:e18401, 2017.
 */

#include "gcmi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

/* ------------------------------------------------------------------ */
/* erfinv: inverse error function via rational approximation           */
/* Minimax approximation from Abramowitz & Stegun 26.2.23 extended.   */
/* Accurate to ~1e-9 for |x| < 0.999.                                 */
/* ------------------------------------------------------------------ */
static double gcmi_erfinv(double x)
{
    /* Based on the Peter J. Acklam algorithm for inverse normal CDF
     * combined with the identity erfinv(x) = Phi_inv((x+1)/2) / sqrt(2). */
    /* Use Newton iteration starting from a rational seed. */
    if (x >= 1.0)  return  1e30;
    if (x <= -1.0) return -1e30;
    if (x == 0.0)  return  0.0;

    /* Rational approximation coefficients (Peter Acklam, adapted for erfinv) */
    const double a[] = {-3.969683028665376e+01,  2.209460984245205e+02,
                        -2.759285104469687e+02,  1.383577518672690e+02,
                        -3.066479806614716e+01,  2.506628277459239e+00};
    const double b[] = {-5.447609879822406e+01,  1.615858368580409e+02,
                        -1.556989798598866e+02,  6.680131188771972e+01,
                        -1.328068155288572e+01};
    const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                        -2.400758277161838e+00, -2.549732539343734e+00,
                         4.374664141464968e+00,  2.938163982698783e+00};
    const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,
                         2.445134137142996e+00,  3.754408661907416e+00};

    /* Convert erfinv(x) to inverse normal: Phi_inv((x+1)/2) / sqrt(2) */
    double p = (x + 1.0) * 0.5;   /* map [-1,1] -> [0,1] */

    double r, q;
    if (p < 0.02425)
    {
        q = std::sqrt(-2.0 * std::log(p));
        r = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    else if (p <= 0.97575)
    {
        q = p - 0.5;
        double s = q * q;
        r = (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*q /
            (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1.0);
    }
    else
    {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        r = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }

    /* r is now Phi_inv(p); erfinv(x) = r / sqrt(2) */
    r /= std::sqrt(2.0);

    /* One step of Newton refinement: erfinv'(x) = sqrt(pi)/2 * exp(erfinv(x)^2) */
    r -= (std::erf(r) - x) * std::sqrt(M_PI) * 0.5 * std::exp(r * r);

    return r;
}

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GMX_CORRELATION_USE_METAL
#include "gcmi_metal.h"
#endif

/* ------------------------------------------------------------------ */
/* Copula transform                                                     */
/* ------------------------------------------------------------------ */

/* Fill z[comp*N + frame] with van der Waerden (probit) scores.
 * comp = atom*3 + dim, sorted by value then transformed via erfinv. */
static void copula_transform(const t_traj* traj, std::vector<double>& z)
{
    const int N      = traj->nframes;
    const int natoms = traj->natoms;
    const int ncomp  = natoms * 3;

    z.resize((size_t)ncomp * N);

    /* Scratch index array for sorting */
    std::vector<int> idx(N);

    for (int atom = 0; atom < natoms; ++atom)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            const int comp = atom * 3 + dim;

            /* Build sort order */
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return traj->x[atom][a][dim] < traj->x[atom][b][dim];
            });

            /* Assign van der Waerden scores with midrank tie handling.
             * Tied frames receive the average probit of their shared rank
             * range so that the copula transform is well-defined regardless
             * of the order std::sort leaves equal elements in. */
            int r = 0;
            while (r < N)
            {
                /* Find the end of the run of equal values starting at r */
                int s = r + 1;
                while (s < N && traj->x[atom][idx[s]][dim] == traj->x[atom][idx[r]][dim])
                    ++s;

                /* Average the probit scores for ranks r, r+1, …, s-1 */
                double avg_z = 0.0;
                for (int t = r; t < s; ++t)
                {
                    const double p = (t + 0.5) / N;
                    avg_z += std::sqrt(2.0) * gcmi_erfinv(2.0 * p - 1.0);
                }
                avg_z /= (s - r);

                for (int t = r; t < s; ++t)
                    z[(size_t)comp * N + idx[t]] = avg_z;

                r = s;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* LU determinant (copy-by-value)                                      */
/* ------------------------------------------------------------------ */

static double lu_det(std::vector<double> m, int n)
{
    double sign = 1.0;
    for (int col = 0; col < n; ++col)
    {
        /* Partial pivot */
        int    pivot = col;
        double best  = std::fabs(m[col * n + col]);
        for (int row = col + 1; row < n; ++row)
        {
            const double v = std::fabs(m[row * n + col]);
            if (v > best) { best = v; pivot = row; }
        }
        if (best == 0.0) return 0.0;
        if (pivot != col)
        {
            for (int j = 0; j < n; ++j)
                std::swap(m[col * n + j], m[pivot * n + j]);
            sign = -sign;
        }
        const double diag = m[col * n + col];
        for (int row = col + 1; row < n; ++row)
        {
            const double factor = m[row * n + col] / diag;
            for (int j = col + 1; j < n; ++j)
                m[row * n + j] -= factor * m[col * n + j];
        }
    }
    double det = sign;
    for (int i = 0; i < n; ++i)
        det *= m[i * n + i];
    return det;
}

/* ------------------------------------------------------------------ */
/* Per-pair GCMI                                                        */
/* ------------------------------------------------------------------ */

static double gcmi_pair(const std::vector<double>& z, int N, int a1, int a2)
{
    const int b1 = a1 * 3;
    const int b2 = a2 * 3;

    /* Build 6x6 covariance: rows/cols 0..2 = a1 components, 3..5 = a2 */
    double cov6[36] = {};
    const double invN = 1.0 / N;

    for (int f = 0; f < N; ++f)
    {
        double v[6];
        for (int d = 0; d < 3; ++d)
        {
            v[d]     = z[(size_t)(b1 + d) * N + f];
            v[d + 3] = z[(size_t)(b2 + d) * N + f];
        }
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                cov6[r * 6 + c] += v[r] * v[c];
    }
    for (int i = 0; i < 36; ++i) cov6[i] *= invN;

    /* Extract 3x3 diagonal blocks */
    std::vector<double> cov_a1(9), cov_a2(9);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
        {
            cov_a1[r * 3 + c] = cov6[r * 6 + c];
            cov_a2[r * 3 + c] = cov6[(r + 3) * 6 + (c + 3)];
        }

    std::vector<double> cov_joint(cov6, cov6 + 36);

    const double det1 = lu_det(cov_a1,    3);
    const double det2 = lu_det(cov_a2,    3);
    const double detJ = lu_det(cov_joint, 6);

    if (det1 <= 0.0 || det2 <= 0.0 || detJ <= 0.0) return 0.0;

    return 0.5 * (std::log(det1) + std::log(det2) - std::log(detJ));
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

bool gcmi_gpu_available()
{
#ifdef GMX_CORRELATION_USE_METAL
    return gcmi_metal_probe();
#else
    return false;
#endif
}

void gcmi_corrmatrix(const t_traj* traj, double* mat, bool use_gpu, int nthreads)
{
    const int natoms = traj->natoms;
    const int N      = traj->nframes;

    if (natoms < 2)
        throw std::runtime_error("gcmi_corrmatrix: need at least 2 atoms");
    if (N < 6)
        throw std::runtime_error("gcmi_corrmatrix: need at least 6 frames for a non-singular 6x6 covariance");

    std::fill(mat, mat + (size_t)natoms * natoms, 0.0);

    /* Copula transform */
    std::vector<double> z;
    copula_transform(traj, z);

#ifdef GMX_CORRELATION_USE_METAL
    if (use_gpu)
    {
        if (!gcmi_metal_probe())
            fprintf(stderr, "Note: no Metal device found — falling back to CPU.\n");
        else
        {
            gcmi_corrmatrix_metal(z.data(), natoms, N, mat);
            return;
        }
    }
#else
    (void)use_gpu;
#endif

    /* Diagonal */
    for (int i = 0; i < natoms; ++i)
        mat[i * natoms + i] = 2000.0;

#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
    fprintf(stderr, "  CPU threads: %d\n", omp_get_max_threads());
#else
    (void)nthreads;
    fprintf(stderr, "  CPU threads: 1\n");
#endif

    const int npairs  = natoms * (natoms - 1) / 2;
    int       completed = 0;
    int       last_pct  = -1;

#pragma omp parallel for schedule(dynamic, 1)
    for (int a1 = 1; a1 < natoms; ++a1)
    {
        for (int a2 = 0; a2 < a1; ++a2)
        {
            const double mi = gcmi_pair(z, N, a1, a2);
            mat[a1 * natoms + a2] = mi;
            mat[a2 * natoms + a1] = mi;

#pragma omp critical(gcmi_prog)
            {
                const int pct = ++completed * 100 / npairs;
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
