# gmx_correlation

A GROMACS 2025 port of the `g_correlation` tool originally distributed with
GROMACS 3/4.  It computes the generalized correlation coefficient r(MI)
(Lange & Grubmüller, *Proteins*, 2006) between all selected atom pairs using
either the Kraskov–Stögbauer–Grassberger (KSG) nearest-neighbour mutual
information estimator or a faster Gaussian (linearized) approximation.

The output matrix format and numerical results are identical to the legacy
tool.  The GROMACS 3/4 C API (removed in GROMACS 5) has been replaced with
the modern C++ trajectory-analysis framework.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| GROMACS | 2025 | Must match the compiler used here |
| C++ compiler | C++17 | Same compiler GROMACS was built with |
| CMake | 3.28+ | |
| OpenMP | 4.5+ | Optional; enables multi-threaded CPU path |
| CUDA Toolkit | 11+ | Optional; NVIDIA GPU support (Linux) |
| Xcode / Metal | macOS 12+ | Auto-enabled on Apple Silicon |

---

## Build

### 1. Source the GROMACS environment

```sh
source /path/to/gromacs-2025/bin/GMXRC
```

### 2. Configure and build

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=/path/to/gromacs/cxx/compiler
cmake --build build -j
```

On **macOS with Homebrew GROMACS** (Apple Silicon):

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/gcc/bin/g++-15
cmake --build build -j
```

The Metal GPU backend is **auto-enabled on macOS** when neither CUDA nor an
explicit Metal flag is passed.  A message confirms this during configure:

```
-- GPU backend: Metal (kraskov_metal.mm) — float32 kernel on Apple GPU
-- CPU threading: OpenMP 4.5
```

### 3. Double-precision build

If your GROMACS library uses the `_d` suffix (double precision):

```sh
cmake -S . -B build -DGMX_DOUBLE=ON -DGMX_SUFFIX=_d \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/gcc/bin/g++-15
cmake --build build -j
```

### 4. CUDA build (Linux / NVIDIA)

```sh
cmake -S . -B build -DGMX_CORRELATION_CUDA=ON \
      -DCMAKE_CXX_COMPILER=$(which g++)
cmake --build build -j
```

### 5. MPI build (distributed atom pairs)

```sh
cmake -S . -B build -DGMX_CORRELATION_MPI=ON \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/gcc/bin/g++-15
cmake --build build -j
mpirun -np 8 ./build/gmx_correlation \
    -s topol.tpr -f fitted.xtc \
    -select "group Protein" -o correl.dat
```

---

## Usage

### Typical run

```sh
# 1. Fit the trajectory first (the tool does not fit internally)
gmx trjconv -s topol.tpr -f traj.xtc -o fitted.xtc -fit rot+trans

# 2. Run the correlation analysis
./build/gmx_correlation \
    -s topol.tpr \
    -f fitted.xtc \
    -select "group Protein" \
    -o correl.dat \
    -m correl.xpm
```

### With GPU (macOS Metal or Linux CUDA)

```sh
./build/gmx_correlation \
    -s topol.tpr -f fitted.xtc \
    -select "group Protein" \
    -o correl.dat -gpu
```

If no GPU is available the tool falls back to CPU automatically and prints a
note.  Metal (Apple Silicon) is about 3× faster than a single CPU core on
typical MD datasets; NVIDIA GPU speedup scales with device capability.

### Multi-threaded CPU

```sh
# Use all available cores (default when -nt is omitted or 0)
./build/gmx_correlation -s topol.tpr -f fitted.xtc \
    -select "group Protein" -o correl.dat -nt 0

# Cap at 8 threads
./build/gmx_correlation -s topol.tpr -f fitted.xtc \
    -select "group Protein" -o correl.dat -nt 8
```

### Gaussian (fast) approximation

```sh
./build/gmx_correlation -s topol.tpr -f fitted.xtc \
    -select "group Protein" -o correl.dat -linear
```

---

## Options reference

| Option | Default | Description |
|---|---|---|
| `-select` | required | Atom selection for the correlation matrix |
| `-o` | `correl.dat` | Output correlation matrix |
| `-m` | *(none)* | Optional XPM matrix image |
| `-skip N` | `1` | Use every Nth frame |
| `-k N` | `100` | Nearest-neighbour count for the KSG estimator |
| `-linear` | off | Gaussian (linearized) MI approximation |
| `-mi` | off | Write raw MI values instead of r(MI) coefficients |
| `-gpu` | off | Use GPU (Metal on macOS, CUDA on Linux) |
| `-nt N` | `0` | CPU thread count; 0 = all available (OpenMP) |

---

## Runtime output

The tool prints progress and timing to stderr so it can be redirected
independently of the matrix output:

```
Computing Kraskov correlation matrix (247 atoms, 2000 frames, k=100) on CPU...
  CPU threads: 10
  Progress:  47%
  Progress: 100%
Kraskov matrix done in 130.69 s
Writing output to correl.dat
```

With `-gpu` on Apple Silicon:

```
Computing Kraskov correlation matrix (247 atoms, 2000 frames, k=100) on GPU...
Metal GPU: Apple M4
  Dispatching Metal kernel: 30381 threadgroups × 128 threads
  Kernel execution: 38.00 s
Kraskov matrix done in 38.04 s
Writing output to correl.dat
```

---

## Output format

`correl.dat` uses the same format as the original `g_correlation` tool and can
be read by the same MATLAB / Octave post-processing scripts:

```
247 x 247 [
   1.000000   0.342817  ...
   ...
]
```

The diagonal is set to 1.0 (r(MI) of an atom with itself = 1.0 after
`pearsify`).

---

## Numerical accuracy

| Path | Precision | Max diff vs legacy |
|---|---|---|
| CPU Kraskov | float64 | < 6 × 10⁻⁴ (float/double compiler differences) |
| Metal GPU | float32 | < 9 × 10⁻⁴ (float32 limitation on Apple Silicon) |
| Gaussian | float64 | identical |

The Metal GPU path uses float32 because Apple Silicon shader hardware has no
float64 support.  The resulting MI values agree with the CPU double-precision
path to better than 10⁻³, within the same range as cross-compiler differences
observed between the legacy and modern CPU builds.

---

## Code layout

| File | Role |
|---|---|
| `gmx_correlation.cpp` | GROMACS 2025 trajectory-analysis module: command-line options, selection handling, frame collection, timing, and dispatch |
| `correlation_core.h` | Compatibility interface: `t_traj`, `t_kraskov`, allocator macros, and function declarations |
| `correlation_core.cpp` | Gaussian MI, matrix output (plain-text and XPM), OpenMP/MPI pair loop, `pearsify` |
| `kraskov.cpp` | KSG nearest-neighbour estimator ported from the original GROMACS 3/4 source — kept close to the original for numerical reproducibility |
| `kraskov_gpu.h` / `kraskov_gpu.cu` | CUDA backend (Linux / NVIDIA): brute-force 3-pass k-NN kernel, one block per atom pair |
| `kraskov_metal.h` / `kraskov_metal.mm` | Metal backend (macOS): Objective-C++ host + embedded MSL kernel, float32 |
| `cmake/FindGROMACS.cmake` | CMake helper that locates an installed GROMACS package config |
| `CMakeLists.txt` | Build system: auto-detects Metal on macOS, optional CUDA, optional MPI, OpenMP |

### Data flow

```
GROMACS trajectory
      │
      ▼  (gmx_correlation.cpp)
  t_traj   ← mean-centered, frame-major C arrays
      │
      ├─► gauss_corrmatrix()   CPU only, float64
      │
      └─► kraskov_corrmatrix()
              │
              ├─► kraskov_corrmatrix_metal()   macOS GPU, float32
              ├─► kraskov_corrmatrix_gpu()     Linux CUDA, float64
              └─► kraskov_wrap() × N²/2       CPU, OpenMP + MPI, float64
      │
      ▼  (pearsify)
  r(MI) matrix  →  correl.dat / correl.xpm
```

---

## Citation

If you use this tool, please cite the original method:

> Lange, O. F. & Grubmüller, H. (2006).  Generalized correlation for
> biomolecular dynamics.  *Proteins: Structure, Function, and Bioinformatics*,
> **62**(4), 1053–1061.  https://doi.org/10.1002/prot.20784

And the KSG estimator:

> Kraskov, A., Stögbauer, H. & Grassberger, P. (2004).  Estimating mutual
> information.  *Physical Review E*, **69**(6), 066138.
> https://doi.org/10.1103/PhysRevE.69.066138
