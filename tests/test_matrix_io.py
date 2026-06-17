# SPDX-License-Identifier: GPL-3.0-or-later
"""
test_matrix_io.py — Round-trip tests for read_matrix().

Exercises the BLITZ++ header/footer format written by write_matrix() in
correlation_core.cpp and read back by analyze_matrix.py:read_matrix().
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Allow running from the repo root or from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
from analyze_matrix import read_matrix


def _write_blitz(mat: np.ndarray, path: Path) -> None:
    """Reproduce the exact format of write_matrix() in correlation_core.cpp.

    write_matrix stores column-major: inner loop is over j (column),
    outer loop is over i (row), writing mat[j * n + i] — i.e. the
    *transpose* of the logical matrix.  read_matrix() undoes this with .T.
    """
    n = mat.shape[0]
    with open(path, "w") as fh:
        fh.write(f"{n} x {n} [\n")
        ncol = 0
        for i in range(n):          # row of output file = column of matrix
            for j in range(n):      # col of output file = row of matrix
                fh.write(f"{mat[j, i]:10.6g} ")
                ncol += 1
                if ncol > 20:
                    fh.write("\n")
                    ncol = 0
        fh.write("]\n")


class TestRoundTrip:
    def test_identity_3x3(self):
        mat = np.eye(3)
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            p = Path(f.name)
        _write_blitz(mat, p)
        result = read_matrix(str(p))
        p.unlink()
        np.testing.assert_allclose(result, mat, atol=1e-9)

    def test_symmetric_5x5(self):
        rng = np.random.default_rng(0)
        raw = rng.random((5, 5))
        mat = (raw + raw.T) / 2
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            p = Path(f.name)
        _write_blitz(mat, p)
        result = read_matrix(str(p))
        p.unlink()
        np.testing.assert_allclose(result, mat, atol=1e-6)

    def test_asymmetric_4x4(self):
        """TE matrices are not symmetric."""
        rng = np.random.default_rng(1)
        mat = rng.random((4, 4))
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            p = Path(f.name)
        _write_blitz(mat, p)
        result = read_matrix(str(p))
        p.unlink()
        np.testing.assert_allclose(result, mat, atol=1e-6)

    def test_sentinel_diagonal(self):
        """Diagonal sentinel value (2000) must be preserved exactly."""
        mat = np.zeros((4, 4))
        np.fill_diagonal(mat, 2000.0)
        mat[0, 1] = mat[1, 0] = 0.75
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            p = Path(f.name)
        _write_blitz(mat, p)
        result = read_matrix(str(p))
        p.unlink()
        assert result[0, 0] == pytest.approx(2000.0)
        assert result[1, 1] == pytest.approx(2000.0)
        assert result[0, 1] == pytest.approx(0.75, abs=1e-6)

    def test_plain_no_header(self):
        """read_matrix must also handle headerless plain-text matrices."""
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            p = Path(f.name)
            for row in mat:
                f.write("  ".join(f"{v:.6g}" for v in row) + "\n")
        result = read_matrix(str(p))
        p.unlink()
        np.testing.assert_allclose(result, mat, atol=1e-6)

    def test_real_cpu_dat(self):
        """Smoke-test against the real CPU.dat test file if present."""
        cpu_dat = Path(__file__).parent.parent.parent / "test_files" / "CPU.dat"
        if not cpu_dat.exists():
            pytest.skip("test_files/CPU.dat not found")
        mat = read_matrix(str(cpu_dat))
        assert mat.shape == (159, 159), f"expected 159x159, got {mat.shape}"
        # Diagonal sentinel
        assert mat[0, 0] == pytest.approx(2000.0, abs=1.0)
        # Symmetric (KSG output is symmetric)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
