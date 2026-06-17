// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "correlation_core.h"

// Gaussian Copula MI (Ince et al. 2017). O(N log N) per pair.
// mat[a1*n+a2] = mat[a2*n+a1] = MI (symmetric).
// Activate GPU with use_gpu=true (Metal on macOS). Falls back to CPU.
void gcmi_corrmatrix(const t_traj* traj, double* mat,
                     bool use_gpu = false, int nthreads = 0);
bool gcmi_gpu_available();
