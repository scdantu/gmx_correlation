# Changelog

## [1.2.0] — 2026-06-17

### Fixed
- **GCMI tie handling**: `copula_transform()` now uses midrank averaging for
  tied values instead of leaving them in arbitrary `std::sort` order.
- **Default network threshold**: `--threshold` now defaults to
  `mean + 0.5·std` of positive off-diagonal values instead of `0.0`, which
  previously produced near-complete graphs for typical r(MI) matrices.
- **PyMOL CGO scalability**: `write_pml()` now emits a single compact
  `_EDGES` list instead of one Python block per edge; O(1) file size.
- **Bounds checks**: `kraskov_corrmatrix` and `gcmi_corrmatrix` now throw
  `std::runtime_error` for `natoms < 2`, `nframes ≤ k`, or `nframes < 6`.

### Added
- **Test suite** (`tests/`): 29 pytest tests — matrix I/O round-trip, GCMI
  vs. analytical Gaussian MI, KSG-1 reference vs. analytic, TE/CMI
  Frenzel-Pompe vs. coupled AR(1) analytical values.
- **GitHub Actions CI** (`.github/workflows/tests.yml`): Python 3.10 + 3.12.
- **Thread-safety audit**: verified `t_kraskov*` is read-only during the
  OpenMP parallel loop; documented in `correlation_core.cpp`.
- **`pearsify` warning for GCMI**: stderr note when the Lange-Grubmüller
  r(MI) transform is applied to GCMI output (not independently validated).
- `scipy>=1.10` and `pytest>=7.0` added to `requirements.txt`.

---

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
