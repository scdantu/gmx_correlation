#pragma once

/*! \file kraskov_metal.h
 * \brief Metal (macOS GPU) backend for the KSG mutual-information estimator.
 *
 * Apple Silicon GPUs operate in float32 only — there is no hardware float64
 * in the shader pipeline.  Coordinates and the digamma table are converted to
 * float before GPU transfer.  The resulting MI values agree with the CPU
 * double-precision path to roughly 1e-4, comparable to the cross-compiler
 * float/double spread already measured between the legacy and new CPU builds.
 *
 * The kernel supports k ≤ 128.  For k > 128 the caller falls back to CPU.
 *
 * This header must remain pure C++ so it can be included from .cpp files
 * compiled without Objective-C support.
 */

#ifdef GMX_CORRELATION_USE_METAL

#include "correlation_core.h"

/*! \brief Return true when a Metal-capable GPU is present (always true on
 *  modern macOS, but guarded so the caller need not special-case it). */
bool kraskov_metal_probe();

/*! \brief Compute the full KSG MI matrix on the macOS GPU via Metal.
 *
 * \param kr      Normalised workspace from kraskov_prepare().
 * \param natoms  Number of atoms (matrix dimension).
 * \param mat     Output matrix [natoms*natoms].
 * \param k       Number of nearest neighbours (≤ 128).
 */
void kraskov_corrmatrix_metal(const t_kraskov* kr, int natoms, double* mat, int k);

#endif /* GMX_CORRELATION_USE_METAL */
