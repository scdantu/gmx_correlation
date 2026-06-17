# SPDX-License-Identifier: GPL-3.0-or-later
"""
test_kraskov.py — Python reference KSG-1 estimator and comparison against the
analytical MI of a bivariate Gaussian.

For a d-dimensional Gaussian with correlation matrix Sigma, the true MI is:
    MI = 0.5 * log( det(Sigma_x)*det(Sigma_y) / det(Sigma_xy) )

We implement KSG algorithm 1 (Kraskov 2004) in pure Python and verify it
recovers the known MI within a reasonable tolerance.  This serves as a ground-
truth reference for the C++ implementation in lib/kraskov.cpp.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import digamma
from scipy.spatial import cKDTree


# ── Python KSG-1 reference ────────────────────────────────────────────────────

def ksg1_mi(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """KSG algorithm 1 mutual information estimator.

    Parameters
    ----------
    X : (N, dx) array
    Y : (N, dy) array
    k : number of nearest neighbours

    Returns
    -------
    MI estimate in nats.
    """
    N = X.shape[0]
    XY = np.hstack([X, Y])

    # k-NN in the joint space (Chebyshev / L-inf norm)
    tree_xy = cKDTree(XY)
    # k+1 because the point itself is included
    dists, _ = tree_xy.query(XY, k=k + 1, p=np.inf, workers=-1)
    eps = dists[:, k]  # distance to k-th neighbour (exclusive)

    # Count neighbours in marginal spaces within eps (strict)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)
    nx = np.array([len(tree_x.query_ball_point(x, r - 1e-15, p=np.inf)) - 1
                   for x, r in zip(X, eps)])
    ny = np.array([len(tree_y.query_ball_point(y, r - 1e-15, p=np.inf)) - 1
                   for y, r in zip(Y, eps)])

    return digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))


def mi_gaussian_true_ksg(r: float, d: int = 3) -> float:
    """True MI for (d+d)-dim Gaussian where each pair of components is correlated by r."""
    # det(Sigma_x) = det(Sigma_y) = 1 (identity blocks)
    # det(Sigma_xy) = (1-r^2)^d
    return -0.5 * d * np.log(1.0 - r ** 2)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("r,tol", [
    (0.3, 0.08),
    (0.6, 0.08),
    (0.9, 0.30),   # KSG-1 has known downward bias near saturation (r→1)
])
def test_ksg1_vs_analytic(r, tol):
    """KSG-1 MI estimate must agree with true Gaussian MI within tolerance."""
    rng = np.random.default_rng(seed=int(r * 100))
    N = 4000
    # X = base, Y = r*base + sqrt(1-r^2)*eps  → corr(X_i, Y_i) = r exactly.
    base = rng.standard_normal((N, 3))
    X = base
    Y = r * base + np.sqrt(1 - r**2) * rng.standard_normal((N, 3))

    mi_est  = ksg1_mi(X, Y, k=5)
    mi_true = mi_gaussian_true_ksg(r)
    err = abs(mi_est - mi_true)

    assert err < tol, (
        f"r={r}: KSG1={mi_est:.4f}, true={mi_true:.4f}, |err|={err:.4f} > tol={tol}"
    )


def test_ksg1_independent():
    """KSG-1 MI of independent variables must be near zero."""
    rng = np.random.default_rng(0)
    N = 3000
    X = rng.standard_normal((N, 3))
    Y = rng.standard_normal((N, 3))
    mi = ksg1_mi(X, Y, k=5)
    assert abs(mi) < 0.05, f"Expected ~0 for independent data, got {mi:.4f}"


def test_ksg1_k_sensitivity():
    """MI estimate should be robust to k in [3, 10] for moderate N."""
    rng = np.random.default_rng(12)
    N = 2000
    r = 0.5
    base = rng.standard_normal((N, 3))
    X = base
    Y = r * base + np.sqrt(1 - r**2) * rng.standard_normal((N, 3))
    true = mi_gaussian_true_ksg(r)
    estimates = [ksg1_mi(X, Y, k=k) for k in [3, 5, 7, 10]]
    for k, est in zip([3, 5, 7, 10], estimates):
        assert abs(est - true) < 0.15, f"k={k}: est={est:.3f}, true={true:.3f}"
