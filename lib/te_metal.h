#pragma once
#ifdef GMX_CORRELATION_USE_METAL
#include "correlation_core.h"
bool te_metal_probe();
void te_corrmatrix_metal(const t_kraskov* kr, int natoms, double* mat, int k, int lag);
#endif
