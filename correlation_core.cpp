#include "correlation_core.h"

/*! \file
 * \brief Matrix assembly, output, and MPI work sharing for gmx_correlation.
 *
 * This file contains the modern C++ replacement for the original
 * `gencorr_matrix.c`, `decompose.c`, and `mpi_stuff.c` responsibilities. The
 * Kraskov estimator itself still lives in `kraskov.cpp`; this file prepares the
 * pairwise matrix and handles output.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef GMX_CORRELATION_USE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GMX_CORRELATION_USE_CUDA
#include "kraskov_gpu.h"
#endif

#ifdef GMX_CORRELATION_USE_METAL
#include "kraskov_metal.h"
#endif

namespace
{

constexpr double c_pi = 3.14159265358979323846;

void checkFile(FILE* fp, const char* filename)
{
    if (fp == nullptr)
    {
        throw std::runtime_error(std::string("Could not open output file: ") + filename);
    }
}

double matrixDet(std::vector<double> mat, int n)
{
    /* The Gaussian approximation only needs determinants of 3x3 and 6x6
     * covariance blocks. A compact LU decomposition avoids keeping the old
     * Numerical Recipes helper code in the modern port. */
    double sign = 1.0;
    for (int col = 0; col < n; ++col)
    {
        int pivot = col;
        double best = std::fabs(mat[col * n + col]);
        for (int row = col + 1; row < n; ++row)
        {
            const double v = std::fabs(mat[row * n + col]);
            if (v > best)
            {
                best  = v;
                pivot = row;
            }
        }
        if (best == 0.0)
        {
            return 0.0;
        }
        if (pivot != col)
        {
            for (int j = 0; j < n; ++j)
            {
                std::swap(mat[col * n + j], mat[pivot * n + j]);
            }
            sign = -sign;
        }
        const double diag = mat[col * n + col];
        for (int row = col + 1; row < n; ++row)
        {
            const double factor = mat[row * n + col] / diag;
            mat[row * n + col] = factor;
            for (int j = col + 1; j < n; ++j)
            {
                mat[row * n + j] -= factor * mat[col * n + j];
            }
        }
    }

    double det = sign;
    for (int i = 0; i < n; ++i)
    {
        det *= mat[i * n + i];
    }
    return det;
}

double entropy(const std::vector<double>& mat, int n)
{
    /* Differential entropy of an n-dimensional Gaussian:
     * H = 1/2 * log((2*pi*e)^n * det(cov)).
     */
    const double det = matrixDet(mat, n);
    if (det <= 0.0)
    {
        return 0.0;
    }
    return 0.5 * (n * (1.0 + std::log(2.0 * c_pi)) + std::log(det));
}

std::vector<double> subMatrix(const std::vector<double>& mat, int n, int rowBegin, int rowEnd, int colBegin, int colEnd)
{
    const int rows = rowEnd - rowBegin;
    const int cols = colEnd - colBegin;
    std::vector<double> out(rows * cols);
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            out[row * cols + col] = mat[(rowBegin + row) * n + colBegin + col];
        }
    }
    return out;
}

void pushDimBlock(std::vector<double>* dest, const std::vector<double>& src, int rowBegin, int colBegin, bool transpose)
{
    for (int row = 0; row < DIM; ++row)
    {
        for (int col = 0; col < DIM; ++col)
        {
            (*dest)[(rowBegin + row) * 2 * DIM + colBegin + col] =
                    transpose ? src[col * DIM + row] : src[row * DIM + col];
        }
    }
}

void covarianceMatrix(const t_traj* traj, std::vector<double>* mat)
{
    /* Coordinates have already been mean-centered in gmx_correlation.cpp, so
     * the covariance is just the frame average of component products. */
    const int natoms = traj->natoms;
    const int ncov   = natoms * DIM;
    mat->assign(ncov * ncov, 0.0);

    for (int a1 = 0, i = 0; a1 < natoms; ++a1)
    {
        for (int d1 = 0; d1 < DIM; ++d1, ++i)
        {
            for (int a2 = 0, j = 0; a2 < natoms; ++a2)
            {
                for (int d2 = 0; d2 < DIM; ++d2, ++j)
                {
                    double sum = 0.0;
                    for (int frame = 0; frame < traj->nframes; ++frame)
                    {
                        sum += traj->x[a1][frame][d1] * traj->x[a2][frame][d2];
                    }
                    (*mat)[i * ncov + j] = sum / traj->nframes;
                }
            }
        }
    }
}

int mpiRank()
{
/* Keep MPI optional at compile time. In serial builds these helpers collapse to
 * rank 0 / size 1, so the same matrix code is used in both paths. */
#ifdef GMX_CORRELATION_USE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
#endif
    return 0;
}

int mpiSize()
{
#ifdef GMX_CORRELATION_USE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized)
    {
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
#endif
    return 1;
}

} // namespace

void write_matrix(const double* mat, int natoms, const char* outfile)
{
    /* Preserve the legacy row/column ordering so existing MATLAB/Octave helper
     * scripts can read files written by this port. */
    FILE* out = std::fopen(outfile, "w");
    checkFile(out, outfile);

    std::fprintf(out, "%d x %d [\n", natoms, natoms);
    int ncol = 0;
    for (int i = 0; i < natoms; ++i)
    {
        for (int j = 0; j < natoms; ++j)
        {
            std::fprintf(out, "%10.6g ", mat[j * natoms + i]);
            if (++ncol > 20)
            {
                ncol = 0;
                std::fprintf(out, "\n");
            }
        }
    }
    std::fprintf(out, "]\n");
    std::fclose(out);
}

void write_xpm_matrix(const double* mat, int natoms, const char* outfile, const char* title, const char* legend)
{
    /* The old tool used GROMACS' removed write_xpm() helper. This local writer
     * produces a minimal XPM that common viewers and plotting scripts can still
     * consume without depending on private GROMACS file I/O APIs. */
    FILE* out = std::fopen(outfile, "w");
    checkFile(out, outfile);

    double minValue = 1e100;
    double maxValue = -1e100;
    for (int i = 0; i < natoms; ++i)
    {
        for (int j = 0; j < natoms; ++j)
        {
            if (i != j)
            {
                const double value = mat[j * natoms + i];
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
        }
    }
    if (minValue >= maxValue)
    {
        minValue = 0.0;
        maxValue = 1.0;
    }

    constexpr int nlevels = 80;
    std::fprintf(out, "/* XPM */\nstatic char *gmx_correlation_xpm[] = {\n");
    std::fprintf(out, "\"%d %d %d 1\",\n", natoms, natoms, nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        const int shade = 255 - static_cast<int>(255.0 * level / (nlevels - 1));
        std::fprintf(out, "\"%c c #%02x%02x%02x\",\n", 33 + level, shade, shade, shade);
    }
    std::fprintf(out, "/* title: %s */\n/* legend: %s */\n", title, legend);
    for (int i = 0; i < natoms; ++i)
    {
        std::fprintf(out, "\"");
        for (int j = 0; j < natoms; ++j)
        {
            const double value = mat[j * natoms + i];
            int level = static_cast<int>((value - minValue) / (maxValue - minValue) * (nlevels - 1));
            level = std::clamp(level, 0, nlevels - 1);
            std::fprintf(out, "%c", 33 + level);
        }
        std::fprintf(out, "\"%s\n", i + 1 == natoms ? "" : ",");
    }
    std::fprintf(out, "};\n");
    std::fclose(out);
}

void gauss_corrmatrix(const t_traj* traj, double* mat)
{
    /* For each atom pair, build the 6x6 joint covariance from two 3x3 atom
     * covariance blocks and the cross-covariance block. Mutual information is
     * H(X) + H(Y) - H(X,Y). */
    std::vector<double> cov;
    covarianceMatrix(traj, &cov);

    const int natoms = traj->natoms;
    const int ncov   = natoms * DIM;
    for (int a1 = 0; a1 < natoms; ++a1)
    {
        mat[a1 * natoms + a1] = 2000.0;
        for (int a2 = a1 + 1; a2 < natoms; ++a2)
        {
            std::vector<double> combined(2 * DIM * 2 * DIM, 0.0);
            auto first = subMatrix(cov, ncov, a1 * DIM, a1 * DIM + DIM, a1 * DIM, a1 * DIM + DIM);
            pushDimBlock(&combined, first, 0, 0, false);
            const double e1 = entropy(first, DIM);

            auto second = subMatrix(cov, ncov, a2 * DIM, a2 * DIM + DIM, a2 * DIM, a2 * DIM + DIM);
            pushDimBlock(&combined, second, DIM, DIM, false);
            const double e2 = entropy(second, DIM);

            auto cross = subMatrix(cov, ncov, a1 * DIM, a1 * DIM + DIM, a2 * DIM, a2 * DIM + DIM);
            pushDimBlock(&combined, cross, 0, DIM, false);
            pushDimBlock(&combined, cross, DIM, 0, true);
            const double mi = e1 + e2 - entropy(combined, 2 * DIM);
            mat[a2 * natoms + a1] = mat[a1 * natoms + a2] = mi;
        }
    }
}

bool gpu_available()
{
#ifdef GMX_CORRELATION_USE_CUDA
    return kraskov_gpu_probe();
#elif defined(GMX_CORRELATION_USE_METAL)
    return kraskov_metal_probe();
#else
    return false;
#endif
}

void kraskov_corrmatrix(const t_traj* traj, double* mat, int k, bool use_gpu, int nthreads)
{
    const int natoms = traj->natoms;
    std::fill(mat, mat + natoms * natoms, 0.0);

    t_traj mutableTraj = *traj;
    t_kraskov kraskov;
    kraskov_prepare(&mutableTraj, &kraskov);

#if defined(GMX_CORRELATION_USE_CUDA)
    if (use_gpu) {
        if (k > 128) {
            fprintf(stderr, "Note: GPU path supports k ≤ 128; k=%d requested — falling back to CPU.\n", k);
        } else if (!kraskov_gpu_probe()) {
            fprintf(stderr, "Note: no CUDA device found — falling back to CPU.\n");
        } else {
            kraskov_corrmatrix_gpu(&kraskov, natoms, mat, k);
            kraskov_done(&kraskov);
            return;
        }
    }
#elif defined(GMX_CORRELATION_USE_METAL)
    if (use_gpu) {
        if (k > 128) {
            fprintf(stderr, "Note: Metal GPU path supports k ≤ 128; k=%d requested — falling back to CPU.\n", k);
        } else if (!kraskov_metal_probe()) {
            fprintf(stderr, "Note: no Metal device found — falling back to CPU.\n");
        } else {
            kraskov_corrmatrix_metal(&kraskov, natoms, mat, k);
            kraskov_done(&kraskov);
            return;
        }
    }
#else
    if (use_gpu) {
        fprintf(stderr, "Note: binary was built without GPU support (rebuild with -DGMX_CORRELATION_METAL=ON on macOS or -DGMX_CORRELATION_CUDA=ON on Linux).\n");
    }
#endif

    const int rank = mpiRank();
    const int size = mpiSize();

    /* ---- Thread count ---- */
#ifdef _OPENMP
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
    const int actual_threads = omp_get_max_threads();
#else
    const int actual_threads = 1;
    (void)nthreads;
#endif
    if (rank == 0)
        fprintf(stderr, "  CPU threads: %d\n", actual_threads);

    /* ---- Diagonal ---- */
    for (int i = 0; i < natoms; ++i)
        mat[i * natoms + i] = 2000.0;

    /* ---- Off-diagonal pairs: embarrassingly parallel ---- *
     * Each unique pair (a1 > a2) maps to a flat index:      *
     *   pair_id = a1*(a1-1)/2 + a2                          *
     * MPI rank r owns pairs where pair_id % size == rank.   *
     * OpenMP threads share the pairs owned by this rank.    */
    const int npairs  = natoms * (natoms - 1) / 2;
    int completed     = 0;   /* protected by omp critical(prog) */
    int last_pct      = -1;

#pragma omp parallel for schedule(dynamic, 1)
    for (int a1 = 1; a1 < natoms; ++a1)
    {
        for (int a2 = 0; a2 < a1; ++a2)
        {
            const int pair_id = a1 * (a1 - 1) / 2 + a2;
            if (pair_id % size != rank)
            {
                /* Count skipped pairs so progress reaches 100 % */
#pragma omp critical(prog)
                {
                    if (++completed * 100 / npairs > last_pct && rank == 0)
                    {
                        last_pct = completed * 100 / npairs;
                        fprintf(stderr, "\r  Progress: %3d%%", last_pct);
                        fflush(stderr);
                    }
                }
                continue;
            }

            const double value = kraskov_wrap(&kraskov, a1, a2, k);
            mat[a1 * natoms + a2] = value;
            mat[a2 * natoms + a1] = value;

#pragma omp critical(prog)
            {
                const int pct = ++completed * 100 / npairs;
                if (pct > last_pct && rank == 0)
                {
                    last_pct = pct;
                    fprintf(stderr, "\r  Progress: %3d%%", pct);
                    fflush(stderr);
                }
            }
        }
    }

    if (rank == 0)
    {
        fprintf(stderr, "\r  Progress: 100%%\n");
        fflush(stderr);
    }

    kraskov_done(&kraskov);

#ifdef GMX_CORRELATION_USE_MPI
    if (size > 1)
    {
        /* Every rank has zeros for work it did not own, so summing the matrices
         * reconstructs the complete symmetric matrix on all ranks. */
        std::vector<double> reduced(natoms * natoms, 0.0);
        MPI_Allreduce(mat, reduced.data(), natoms * natoms, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        std::copy(reduced.begin(), reduced.end(), mat);
    }
#endif
}

void pearsify(double* mat, int n, int dim)
{
    /* Lange and Grubmueller's generalized correlation coefficient maps mutual
     * information back to a [0, 1] correlation-like quantity. */
    for (int i = 0; i < n * n; ++i)
    {
        mat[i] = mat[i] > 0.0 ? std::sqrt(1.0 - std::exp(-2.0 / dim * mat[i])) : 0.0;
    }
}

void done_traj(t_traj* traj)
{
    if (traj->x != nullptr)
    {
        for (int i = 0; i < traj->natoms; ++i)
        {
            sfree(traj->x[i]);
        }
        sfree(traj->x);
    }
    sfree(traj->xav);
}
