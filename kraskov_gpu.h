#pragma once

/*! \file kraskov_gpu.h
 * \brief CUDA-accelerated KSG mutual-information matrix.
 *
 * When the build was configured with GMX_CORRELATION_CUDA=ON and a CUDA device
 * is present at runtime, kraskov_corrmatrix_gpu() replaces the serial CPU loop
 * in kraskov_corrmatrix(). The CPU path is never modified; the GPU path is an
 * independent implementation that produces numerically equivalent results.
 *
 * On macOS (no CUDA since CUDA 10.2) or any system without an NVIDIA GPU the
 * probe returns false and the caller falls back silently to the CPU path.
 */

#ifdef GMX_CORRELATION_USE_CUDA

#include "correlation_core.h"

/*! \brief Return true if at least one CUDA-capable device is present. */
bool kraskov_gpu_probe();

/*! \brief Compute the full KSG MI matrix on the GPU.
 *
 * \param kr      Normalised workspace from kraskov_prepare().
 * \param natoms  Number of atoms (matrix dimension).
 * \param mat     Output matrix [natoms*natoms].
 * \param k       Number of nearest neighbours.
 *
 * The kernel supports k ≤ 128.  For larger k the caller should use the CPU
 * path instead.
 */
void kraskov_corrmatrix_gpu(const t_kraskov* kr, int natoms, double* mat, int k);

#endif /* GMX_CORRELATION_USE_CUDA */
