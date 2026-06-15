# gmx_correlation for GROMACS 2025

This is a GROMACS 2025 trajectory-analysis port of the original `g_correlation`
tool in `../g_correlation_src`, renamed as `gmx_correlation` for this repository.

It keeps the original output matrix format and the original mutual-information
math, but replaces the removed GROMACS 3/4 APIs with the current C++ trajectory
analysis framework.

## Build

Source the matching GROMACS installation first:

```sh
source /path/to/gromacs-2025/bin/GMXRC
cmake -S . -B build -DCMAKE_CXX_COMPILER=/path/to/the/gromacs/cxx/compiler
cmake --build build -j
```

For Homebrew GROMACS on macOS, this is typically:

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/gcc/bin/g++-15
cmake --build build -j
```

For MPI work sharing over atom pairs:

```sh
source /path/to/gromacs-2025/bin/GMXRC
cmake -S . -B build-mpi -DGMX_CORRELATION_MPI=ON -DCMAKE_CXX_COMPILER=/path/to/the/gromacs/cxx/compiler
cmake --build build-mpi -j
mpirun -np 8 ./build-mpi/gmx_correlation -s topol.tpr -f fitted.xtc -select "group Protein" -o correl.dat
```

Use `-DGMX_DOUBLE=ON` and `-DGMX_SUFFIX=_d` if your GROMACS library was built in
double precision with the usual suffix.

## Usage Notes

The old tool fitted frames internally with obsolete GROMACS routines. This port
expects a pre-fitted trajectory:

```sh
gmx trjconv -s topol.tpr -f traj.xtc -o fitted.xtc -fit rot+trans
./build/gmx_correlation -s topol.tpr -f fitted.xtc -select "group Protein" -o correl.dat -m correl.xpm
```

Important options:

- `-select`: static atom selection to analyze.
- `-skip`: use every nth frame.
- `-k`: nearest-neighbor count for the Kraskov estimator.
- `-linear`: use the faster Gaussian linearized mutual-information estimate.
- `-mi`: write mutual information instead of the generalized correlation coefficient.

## Code Layout

- `gmx_correlation.cpp`: GROMACS trajectory-analysis module. It owns command-line
  options, selection handling, frame collection, validation, and conversion into
  the legacy trajectory layout.
- `correlation_core.h`: Narrow compatibility interface between the modern
  GROMACS module and the original estimator data structures.
- `correlation_core.cpp`: Matrix output, Gaussian mutual-information path,
  generalized-correlation conversion, and optional MPI work distribution.
- `kraskov.cpp`: Ported Kraskov nearest-neighbor estimator from the legacy tool.
  The code is intentionally kept close to the original numerical implementation
  so results can be compared against older builds.
- `cmake/FindGROMACS.cmake`: GROMACS template helper for finding an installed
  GROMACS package config.

The main implementation boundary is `t_traj`: the modern GROMACS API is used
only while reading frames, then the selected atom coordinates are transformed
into a mean-centered `t_traj` before entering the original correlation math.
