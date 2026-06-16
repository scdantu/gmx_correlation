# Changelog

## [1.1.0] — 2026-06-16

### Added
- **Metal GPU backend** (`kraskov_metal.mm`): Objective-C++ host code with an
  embedded Metal Shading Language kernel.  Auto-enabled on macOS at CMake
  configure time when CUDA is not requested.  Uses float32 (Apple Silicon
  hardware limitation); MI values agree with CPU double to < 9 × 10⁻⁴.
- **CUDA GPU backend** (`kraskov_gpu.cu`): brute-force 3-pass k-NN kernel for
  Linux / NVIDIA systems.  Enabled with `-DGMX_CORRELATION_CUDA=ON`.
- **`-gpu` flag**: opt-in GPU acceleration; falls back to CPU automatically if
  no GPU is present or if `k > 128`.
- **OpenMP multi-threading** for the CPU Kraskov pair loop
  (`schedule(dynamic, 1)`).  Linked automatically when OpenMP is found.
- **`-nt N` flag**: set the number of CPU threads; `0` (default) uses all
  available cores.
- **Status and timing output**: progress bar (`Progress: N%`) and elapsed
  seconds are printed to stderr for both Gaussian and Kraskov paths.
- **`gpu_available()`**: runtime probe so the front-end can print a diagnostic
  before falling back to CPU.

### Changed
- `kraskov_corrmatrix()` signature extended with `use_gpu` and `nthreads`
  parameters (both default to no-op values, so existing callers are unaffected).
- CMakeLists auto-enables `GMX_CORRELATION_METAL` on macOS when no explicit GPU
  backend is selected, and uses `enable_language(CUDA)` only inside the CUDA
  block to avoid spurious compiler searches.
- XPM writer no longer depends on removed GROMACS private I/O APIs; replaced
  with a self-contained minimal XPM writer.
- `opts.fastMathEnabled` replaced with `opts.mathMode = MTLMathModeFast` on
  macOS 15+ to silence the deprecation warning.

### Fixed
- `.gitignore` contained a spurious `.git/` entry; removed.
- `project()` declaration previously listed `CUDA` unconditionally, causing
  CMake to search for the CUDA compiler even when `GMX_CORRELATION_CUDA=OFF`.

---

## [1.0.0] — initial port

- Ported `g_correlation` (GROMACS 3/4) to the GROMACS 2025
  `TrajectoryAnalysisModule` API.
- Replaced removed APIs (`read_first_x`, `read_tps_conf`, `parse_common_args`,
  `write_xpm`, etc.) with modern equivalents.
- Preserved `kraskov.cpp` numerics verbatim for reproducibility; Gaussian MI
  output is identical to the legacy binary; Kraskov MI differs by < 6 × 10⁻⁴
  (float/double compiler differences).
- Added optional MPI work-sharing (`-DGMX_CORRELATION_MPI=ON`).
- Added `-dump` option to export the mean-centered trajectory for
  cross-validation against legacy builds.
