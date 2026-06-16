#ifndef GMX_CORRELATION_CORE_H
#define GMX_CORRELATION_CORE_H

/*! \file
 * \brief Small compatibility layer around the original g_correlation math code.
 *
 * The GROMACS 2025 trajectory-analysis front end stores selected coordinates in
 * C++ containers while reading frames. Before computing the matrix, those
 * coordinates are converted into the compact C-style layout used by the
 * original Kraskov estimator. Keeping this boundary narrow makes it easier to
 * compare the modern port with the legacy implementation.
 */

#include <cstddef>
#include <cstdlib>
#include <type_traits>

#ifndef DIM
#define DIM 3
#endif

typedef double gcorr_rvec[DIM];

/*! \brief Mean-centered trajectory for the selected atoms.
 *
 * `x[atom][frame][dimension]` stores the coordinate fluctuation after the
 * per-atom mean has been subtracted. `xav[atom][dimension]` stores that mean.
 * This mirrors the legacy `t_traj` structure closely enough that the original
 * estimator code can be reused with minimal numerical changes.
 */
struct t_traj
{
    int    nframes = 0;
    int    natoms  = 0;
    gcorr_rvec** x   = nullptr;
    gcorr_rvec*  xav = nullptr;
};

/*! \brief Normalized workspace consumed by the Kraskov estimator.
 *
 * `kraskov_prepare()` transforms the mean-centered trajectory into normalized
 * component arrays and precomputes digamma/scaling tables. `kraskov_wrap()`
 * then extracts atom-pair component blocks from this workspace.
 */
struct t_kraskov
{
    double** x    = nullptr;
    double*  psi  = nullptr;
    double*  scal = nullptr;
    int      N    = 0;
    int      dim  = 0;
};

/* Legacy allocation helpers retained for the copied estimator code. They are
 * intentionally local to this port and should not be used by new C++ code. */
#define snew(ptr, n) \
    ((ptr) = static_cast<std::remove_reference_t<decltype(ptr)>>(std::calloc((n), sizeof(*(ptr)))))
#define sfree(ptr)  \
    do              \
    {               \
        std::free(ptr); \
        (ptr) = nullptr; \
    } while (0)

void kraskov_prepare(t_traj* gmx_traj, t_kraskov* kr);
void kraskov_done(t_kraskov* kr);
double kraskov_wrap(t_kraskov* kr, int a1, int a2, int param_k);

//! Write the matrix in the original plain-text MATLAB-like format.
void write_matrix(const double* mat, int natoms, const char* outfile);

//! Write a simple grayscale XPM representation of the correlation matrix.
void write_xpm_matrix(const double* mat, int natoms, const char* outfile, const char* title, const char* legend);

//! Compute the Gaussian/linearized mutual-information approximation.
void gauss_corrmatrix(const t_traj* traj, double* mat);

//! Compute the Kraskov mutual-information matrix, optionally on GPU or via MPI.
void kraskov_corrmatrix(const t_traj* traj, double* mat, int k, bool use_gpu = false, int nthreads = 0);

//! Return true if a CUDA-capable GPU is available (always false without CUDA build).
bool gpu_available();

//! Convert mutual information to generalized correlation coefficients.
void pearsify(double* mat, int n, int dim);

//! Release allocations owned by a `t_traj`.
void done_traj(t_traj* traj);

#endif
