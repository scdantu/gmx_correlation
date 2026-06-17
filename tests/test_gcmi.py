# SPDX-License-Identifier: GPL-3.0-or-later
"""
test_gcmi.py — Pure-Python reference implementation of GCMI and comparison
against the analytical Gaussian MI.

The Gaussian Copula MI (Ince et al. 2017) should recover the true MI of
a bivariate Gaussian exactly in the large-N limit because the copula
transform of Gaussian data is the identity (up to monotone reparametrisation).

For correlated Gaussians with correlation r, the true MI is:
    MI_true = -0.5 * log(1 - r^2)

We implement the same steps as lib/gcmi.cpp and verify that:
    |GCMI_python(r, N) - MI_true(r)| < tolerance

This constitutes:
  (a) a regression test on the Python reference implementation itself, and
  (b) an indirect validation that the C++ implementation follows the same
      algorithm (since any deviation in logic would be visible in code review
      against this reference).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import special

# ── Python reference implementation of GCMI ──────────────────────────────────

def _erfinv(x: np.ndarray) -> np.ndarray:
    """Inverse error function via scipy.special."""
    return special.erfinv(x)


def copula_transform(X: np.ndarray) -> np.ndarray:
    """Van der Waerden transform: each column → probit scores.

    X shape: (N, d)  →  Z shape: (N, d)

    For tied values, ranks are averaged (midrank convention).
    """
    N, d = X.shape
    Z = np.empty_like(X, dtype=float)
    for col in range(d):
        x = X[:, col]
        # Rank with average tie handling (same as scipy rankdata default)
        from scipy.stats import rankdata
        ranks = rankdata(x, method="average")   # 1-based, ties averaged
        # Van der Waerden: p = (rank - 0.5) / N ← midpoint formula
        p = (ranks - 0.5) / N
        Z[:, col] = np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
    return Z


def gcmi_python(X: np.ndarray, Y: np.ndarray) -> float:
    """GCMI between X (N,3) and Y (N,3) — matches lib/gcmi.cpp:gcmi_pair()."""
    N = X.shape[0]
    XY = np.hstack([X, Y])                 # (N, 6)
    Z  = copula_transform(XY)              # (N, 6)

    Zx = Z[:, :3]
    Zy = Z[:, 3:]

    cov_x  = Zx.T @ Zx / N               # 3×3
    cov_y  = Zy.T @ Zy / N               # 3×3
    cov_xy = Z.T  @ Z  / N               # 6×6

    sign_x,  logdet_x  = np.linalg.slogdet(cov_x)
    sign_y,  logdet_y  = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)

    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        return 0.0

    return 0.5 * (logdet_x + logdet_y - logdet_xy)


def mi_gaussian_true(r: float) -> float:
    """Analytical MI for a bivariate Gaussian with correlation r (each component)."""
    # For X=(x1,x2,x3), Y=(y1,y2,y3) independently correlated by r:
    # MI = -0.5 * log(det(Sigma_joint) / (det(Sigma_x)*det(Sigma_y)))
    # With Sigma_x = Sigma_y = I_3 and Sigma_xy block off-diag = r*I_3:
    # det(Sigma_joint) = (1 - r^2)^3
    # MI = -0.5 * log((1-r^2)^3) = -1.5 * log(1-r^2)
    return -1.5 * np.log(1.0 - r ** 2)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCopulaTransform:
    def test_gaussian_identity(self):
        """Copula of Gaussian data should be approximately N(0,1)."""
        rng = np.random.default_rng(42)
        N = 2000
        X = rng.standard_normal((N, 3))
        Z = copula_transform(X)
        # Mean ≈ 0, std ≈ 1 (not exact due to finite-N midpoint formula)
        assert abs(Z.mean()) < 0.05
        assert abs(Z.std() - 1.0) < 0.05

    def test_monotone_order_preserved(self):
        """Copula transform must be strictly order-preserving."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(100)
        X = x.reshape(-1, 1)
        Z = copula_transform(X)
        assert np.all(np.argsort(x) == np.argsort(Z[:, 0]))

    def test_ties_average(self):
        """Tied values must receive the same (averaged) probit score."""
        x = np.array([1.0, 2.0, 2.0, 3.0]).reshape(-1, 1)
        Z = copula_transform(x)
        # Ranks: 1, 2.5, 2.5, 4 → p = 0.5/4, 1.5/4, 1.5/4, 3.5/4
        assert Z[1, 0] == pytest.approx(Z[2, 0], abs=1e-12)


@pytest.mark.parametrize("r", [0.2, 0.5, 0.8, 0.95])
class TestGCMIvsAnalytic:
    def test_gcmi_close_to_true(self, r):
        """GCMI should recover the true Gaussian MI within 5% at N=5000."""
        rng = np.random.default_rng(seed=int(r * 1000))
        N = 5000
        # Generate X,Y each (N,3) with correlation exactly r per component.
        # X = base, Y = r*base + sqrt(1-r^2)*eps  → corr(X_i, Y_i) = r.
        base = rng.standard_normal((N, 3))
        X = base
        Y = r * base + np.sqrt(1 - r**2) * rng.standard_normal((N, 3))

        mi_est  = gcmi_python(X, Y)
        mi_true = mi_gaussian_true(r)

        # Tolerance: 5% of true MI, with a floor of 0.02 nats.
        # At low r the MI is tiny so relative error can be noisy;
        # at high r the absolute error grows slightly (finite N).
        abs_err = abs(mi_est - mi_true)
        tol = max(0.02, 0.05 * mi_true)
        assert abs_err < tol, (
            f"r={r}: GCMI={mi_est:.4f} true={mi_true:.4f} |err|={abs_err:.4f} tol={tol:.4f}"
        )

    def test_gcmi_positive(self, r):
        """GCMI must be non-negative for correlated variables."""
        rng = np.random.default_rng(seed=int(r * 999))
        N = 1000
        base = rng.standard_normal((N, 3))
        X = base
        Y = r * base + np.sqrt(1 - r**2) * rng.standard_normal((N, 3))
        assert gcmi_python(X, Y) >= 0.0


class TestGCMIIndependent:
    def test_independent_near_zero(self):
        """GCMI of independent variables should be near zero."""
        rng = np.random.default_rng(99)
        N = 3000
        X = rng.standard_normal((N, 3))
        Y = rng.standard_normal((N, 3))
        mi = gcmi_python(X, Y)
        assert abs(mi) < 0.05, f"Expected ~0 for independent data, got {mi:.4f}"
