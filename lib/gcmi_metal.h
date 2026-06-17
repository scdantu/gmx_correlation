#pragma once
#ifdef GMX_CORRELATION_USE_METAL
#include "correlation_core.h"
bool gcmi_metal_probe();
void gcmi_corrmatrix_metal(const double* z, int natoms, int N, double* mat);
#endif
