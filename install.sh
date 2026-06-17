#!/usr/bin/env bash
# install.sh — Build and install gmx_correlation (C++) and Python analysis tools.
#
# Usage:
#   ./install.sh [OPTIONS]
#
# Options:
#   --prefix DIR      Install prefix (default: $HOME/.local)
#   --gromacs DIR     GROMACS installation root (default: auto-detect via gmxrc)
#   --cxx COMPILER    C++ compiler to use (default: auto-detect from GROMACS)
#   --jobs N          Parallel build jobs (default: all CPU cores)
#   --cuda            Enable CUDA GPU backend (Linux/NVIDIA)
#   --no-metal        Disable Metal GPU backend on macOS (auto-enabled otherwise)
#   --mpi             Enable MPI work-sharing
#   --double          Build against double-precision GROMACS
#   --python-only     Skip C++ build; only install Python dependencies
#   --cpp-only        Skip Python dependency install
#   --no-venv         Do not create/use a Python virtual environment
#   --venv DIR        Virtual-environment directory (default: .venv)
#   --debug           Build in Debug mode instead of Release
#   --clean           Remove previous build directory before configuring
#   -h / --help       Show this help and exit
#
# Examples:
#   # Typical macOS (Homebrew GROMACS + Apple Silicon GPU):
#   ./install.sh --prefix ~/tools \
#       --cxx /opt/homebrew/opt/gcc/bin/g++-15
#
#   # Linux with CUDA:
#   source /path/to/gromacs/bin/GMXRC
#   ./install.sh --prefix /usr/local --cuda --jobs 16
#
#   # Python tools only (C++ already built):
#   ./install.sh --python-only --venv .venv

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
step()  { echo -e "\n${BOLD}══ $* ══${RESET}"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
PREFIX="${HOME}/.local"
GROMACS_DIR=""
CXX_COMPILER=""
JOBS=$(python3 -c "import os; print(os.cpu_count())" 2>/dev/null || nproc 2>/dev/null || echo 4)
ENABLE_CUDA=OFF
ENABLE_METAL=""          # empty = auto (ON on macOS, OFF elsewhere)
ENABLE_MPI=OFF
GMX_DOUBLE=OFF
PYTHON_ONLY=false
CPP_ONLY=false
USE_VENV=true
VENV_DIR=".venv"
BUILD_TYPE="Release"
CLEAN_BUILD=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)      PREFIX="$2";      shift 2 ;;
        --gromacs)     GROMACS_DIR="$2"; shift 2 ;;
        --cxx)         CXX_COMPILER="$2"; shift 2 ;;
        --jobs)        JOBS="$2";        shift 2 ;;
        --cuda)        ENABLE_CUDA=ON;   shift   ;;
        --no-metal)    ENABLE_METAL=OFF; shift   ;;
        --mpi)         ENABLE_MPI=ON;    shift   ;;
        --double)      GMX_DOUBLE=ON;    shift   ;;
        --python-only) PYTHON_ONLY=true; shift   ;;
        --cpp-only)    CPP_ONLY=true;    shift   ;;
        --no-venv)     USE_VENV=false;   shift   ;;
        --venv)        VENV_DIR="$2";    shift 2 ;;
        --debug)       BUILD_TYPE="Debug"; shift ;;
        --clean)       CLEAN_BUILD=true; shift   ;;
        -h|--help)
            grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
            exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║          gmx_correlation  installer              ║"
echo "  ║  MI correlation + GCMI + Transfer Entropy        ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"

info "Source root : ${SCRIPT_DIR}"
info "Install to  : ${PREFIX}"
info "Build type  : ${BUILD_TYPE}"
info "Parallel    : ${JOBS} jobs"

# ── Check script is run from repo root ───────────────────────────────────────
[[ -f "${SCRIPT_DIR}/CMakeLists.txt" ]] || \
    error "Run install.sh from the project root (where CMakeLists.txt lives)."

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — C++ BUILD
# ═══════════════════════════════════════════════════════════════════════════════
if ! $PYTHON_ONLY; then

step "C++ prerequisites"

# cmake
if ! command -v cmake &>/dev/null; then
    error "cmake not found. Install cmake ≥ 3.28 and retry."
fi
CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
ok "cmake ${CMAKE_VERSION}"

# GROMACS
if [[ -n "${GROMACS_DIR}" ]]; then
    GMXRC="${GROMACS_DIR}/bin/GMXRC"
    [[ -f "${GMXRC}" ]] || error "GMXRC not found at ${GMXRC}"
    # shellcheck disable=SC1090
    source "${GMXRC}"
    ok "Sourced GROMACS from ${GROMACS_DIR}"
elif [[ -n "${GMXDATA:-}" ]]; then
    ok "GROMACS environment already active (GMXDATA=${GMXDATA})"
else
    # Try common install locations
    for candidate in \
        /opt/homebrew/share/gromacs \
        /usr/local/share/gromacs \
        /usr/share/gromacs; do
        rc="${candidate%share/gromacs}bin/GMXRC"
        if [[ -f "$rc" ]]; then
            # shellcheck disable=SC1090
            source "$rc"
            ok "Auto-detected GROMACS at ${rc%/bin/GMXRC}"
            break
        fi
    done
    if [[ -z "${GMXDATA:-}" ]]; then
        error "GROMACS environment not found.\n  Either run: source /path/to/gromacs/bin/GMXRC\n  Or pass:    --gromacs /path/to/gromacs"
    fi
fi

# C++ compiler — default to what GROMACS was compiled with
if [[ -z "${CXX_COMPILER}" ]]; then
    # GROMACS exports the compiler it was built with in some installations
    if command -v g++-15 &>/dev/null; then
        CXX_COMPILER=$(command -v g++-15)
    elif command -v g++ &>/dev/null; then
        CXX_COMPILER=$(command -v g++)
    elif command -v clang++ &>/dev/null; then
        CXX_COMPILER=$(command -v clang++)
    else
        error "No C++ compiler found. Install g++ or clang++ or pass --cxx /path/to/compiler"
    fi
fi
ok "C++ compiler : ${CXX_COMPILER}"

# CUDA check
if [[ "${ENABLE_CUDA}" == "ON" ]]; then
    if command -v nvcc &>/dev/null; then
        ok "nvcc found: $(nvcc --version | grep release | awk '{print $6}')"
    else
        warn "nvcc not found — CUDA may fail to configure. Make sure CUDA Toolkit is installed."
    fi
fi

# Metal auto-detect
if [[ -z "${ENABLE_METAL}" ]]; then
    if [[ "$(uname)" == "Darwin" ]] && [[ "${ENABLE_CUDA}" == "OFF" ]]; then
        ENABLE_METAL=ON
    else
        ENABLE_METAL=OFF
    fi
fi

step "Configuring C++ build"

if $CLEAN_BUILD && [[ -d "${BUILD_DIR}" ]]; then
    info "Removing existing build directory …"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# Build cmake command
CMAKE_ARGS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}"
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
    -DGMX_DOUBLE="${GMX_DOUBLE}"
    -DGMX_CORRELATION_MPI="${ENABLE_MPI}"
    -DGMX_CORRELATION_CUDA="${ENABLE_CUDA}"
    -DGMX_CORRELATION_METAL="${ENABLE_METAL}"
)

info "cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

step "Building C++ (${JOBS} jobs)"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

step "Installing C++ binary"
cmake --install "${BUILD_DIR}"

BINARY_PATH="${PREFIX}/bin/gmx_correlation"
if [[ -f "${BINARY_PATH}" ]]; then
    ok "Installed: ${BINARY_PATH}"
else
    # Fallback: copy from build dir
    cp "${BUILD_DIR}/gmx_correlation" "${PREFIX}/bin/"
    ok "Copied binary to ${PREFIX}/bin/gmx_correlation"
fi

fi  # end !PYTHON_ONLY

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — PYTHON TOOLS
# ═══════════════════════════════════════════════════════════════════════════════
if ! $CPP_ONLY; then

step "Python tools"

# Python interpreter
PYTHON=""
for py in python3 python; do
    if command -v "$py" &>/dev/null; then
        PY_VER=$($py --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 9 ]]; then
            PYTHON=$(command -v "$py")
            break
        fi
    fi
done
[[ -n "${PYTHON}" ]] || error "Python ≥ 3.9 not found."
ok "Python: ${PYTHON} ($($PYTHON --version))"

# Virtual environment
if $USE_VENV; then
    VENV_PATH="${SCRIPT_DIR}/${VENV_DIR}"
    if [[ ! -d "${VENV_PATH}" ]]; then
        info "Creating virtual environment at ${VENV_PATH} …"
        $PYTHON -m venv "${VENV_PATH}"
    else
        info "Using existing virtual environment at ${VENV_PATH}"
    fi
    # Activate
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
    PIP="${VENV_PATH}/bin/pip"
    PYTHON="${VENV_PATH}/bin/python"
    ok "Virtual environment activated"
else
    PIP=$(command -v pip3 || command -v pip)
    ok "Using system pip: ${PIP}"
fi

step "Installing Python dependencies"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
if [[ -f "${REQUIREMENTS}" ]]; then
    $PIP install --upgrade pip --quiet
    $PIP install -r "${REQUIREMENTS}"
    ok "Dependencies installed from requirements.txt"
else
    warn "requirements.txt not found — installing defaults"
    $PIP install --upgrade pip --quiet
    $PIP install "numpy>=1.24" "matplotlib>=3.7" "igraph>=0.11" "pandas>=2.0"
fi

step "Installing Python script"
PYTHON_INSTALL="${PREFIX}/bin"
mkdir -p "${PYTHON_INSTALL}"

SCRIPT_SRC="${SCRIPT_DIR}/bin/analyze_matrix.py"
SCRIPT_DST="${PYTHON_INSTALL}/gmx_analyze_matrix"

cp "${SCRIPT_SRC}" "${SCRIPT_DST}"
chmod +x "${SCRIPT_DST}"

# Patch shebang to point at the venv python if we used one
if $USE_VENV; then
    VENV_PYTHON="${SCRIPT_DIR}/${VENV_DIR}/bin/python3"
    # Replace first line shebang
    sed -i.bak "1s|.*|#!${VENV_PYTHON}|" "${SCRIPT_DST}" && rm "${SCRIPT_DST}.bak"
fi
ok "Installed: ${SCRIPT_DST}"

fi  # end !CPP_ONLY

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
step "Installation complete"

if ! $PYTHON_ONLY; then
    echo -e "  ${GREEN}C++ binary${RESET}     : ${PREFIX}/bin/gmx_correlation"
fi
if ! $CPP_ONLY; then
    echo -e "  ${GREEN}Python script${RESET}  : ${PREFIX}/bin/gmx_analyze_matrix"
    if $USE_VENV; then
        echo -e "  ${GREEN}Virtual env${RESET}    : ${SCRIPT_DIR}/${VENV_DIR}"
    fi
fi

# PATH reminder
if [[ ":${PATH}:" != *":${PREFIX}/bin:"* ]]; then
    echo ""
    warn "${PREFIX}/bin is not in your PATH. Add this to your shell profile:"
    echo -e "  export PATH=\"${PREFIX}/bin:\$PATH\""
fi

echo ""
echo -e "${BOLD}Quick start:${RESET}"
if ! $PYTHON_ONLY; then
echo "  # Run the correlation tool:"
echo "  gmx_correlation -s topol.tpr -f fitted.xtc -select 'group Protein' \\"
echo "      -o correl.dat -gpu"
echo ""
fi
if ! $CPP_ONLY; then
echo "  # Analyse the output matrix:"
echo "  gmx_analyze_matrix correl.dat --pdb protein.pdb --out results/"
echo ""
echo "  # Transfer entropy (directed network):"
echo "  gmx_analyze_matrix transfer_entropy.dat --asymmetric \\"
echo "      --pdb protein.pdb --out results_te/"
fi
echo ""
