#pragma once
#include "correlation_core.h"

// Transfer entropy matrix. mat[a1*n+a2] = TE(a2->a1, lag).
// Diagonal = 0. Full N x N asymmetric matrix.
// Uses Frenzel-Pompe CMI estimator (KSG-based).
void te_matrix(const t_traj* traj, double* mat, int k, int lag,
               bool use_gpu = false, int nthreads = 0);
bool te_gpu_available();
