# SPDX-License-Identifier: GPL-3.0-or-later
"""
test_te_validation.py — Transfer Entropy reference implementation and validation.

The TE estimator in lib/transfer_entropy.cpp uses the Frenzel-Pompe (2007)
conditional mutual information (CMI) estimator:

    TE(Y→X, lag τ) = CMI(X_{t+τ}; Y_t | X_t)
                   = MI(X_{t+τ}, Y_t) - MI(X_{t+τ}, X_t)  [only for Gaussian]

For a linear AR(1) process with known coupling this can be computed analytically.

We validate our Python reference CMI against:
  1. The known analytical TE for a coupled AR(1) process.
  2. A near-zero TE when Y does not drive X.

Additionally we reproduce the Frenzel-Pompe digamma formula used in
transfer_entropy.cpp and check it matches our KNN CMI implementation.

IMPORTANT NOTE ON C++ VALIDATION
---------------------------------
The C++ TE estimator (lib/transfer_entropy.cpp) has NOT been compared against
a binary output yet — the test_files/ directory does not include a pre-computed
TE matrix.  This test suite validates the *algorithm* implemented in Python and
provides a reference to compare against once a C++ binary is available.
To produce a TE matrix for comparison run:
    gmx_correlation -s topol.tpr -f traj.xtc -te -ote te.dat -lag 1
"""

import numpy as np
import pytest
from scipy.special import digamma
from scipy.spatial import cKDTree


# ── Frenzel-Pompe CMI (k-NN) ─────────────────────────────────────────────────

def cmi_knn(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, k: int = 5) -> float:
    """CMI I(X; Y | Z) using the Frenzel-Pompe (2007) k-NN estimator.

    Parameters
    ----------
    X, Y, Z : (N, d) arrays — the three variables
    k        : nearest-neighbour count

    Returns
    -------
    CMI estimate in nats.
    """
    N = X.shape[0]
    XYZ = np.hstack([X, Y, Z])

    tree_xyz = cKDTree(XYZ)
    dists, _ = tree_xyz.query(XYZ, k=k + 1, p=np.inf, workers=-1)
    eps = dists[:, k]

    XZ = np.hstack([X, Z])
    YZ = np.hstack([Y, Z])

    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)
    tree_z  = cKDTree(Z)

    n_xz = np.array([len(tree_xz.query_ball_point(xz, r - 1e-15, p=np.inf)) - 1
                     for xz, r in zip(XZ, eps)])
    n_yz = np.array([len(tree_yz.query_ball_point(yz, r - 1e-15, p=np.inf)) - 1
                     for yz, r in zip(YZ, eps)])
    n_z  = np.array([len(tree_z.query_ball_point(z,  r - 1e-15, p=np.inf)) - 1
                     for z,  r in zip(Z,  eps)])

    return (digamma(k)
            - np.mean(digamma(n_xz + 1) + digamma(n_yz + 1) - digamma(n_z + 1)))


def te_knn(X: np.ndarray, Y: np.ndarray, lag: int = 1, k: int = 5) -> float:
    """Transfer entropy TE(Y→X) at lag using CMI(X_{t+lag}; Y_t | X_t).

    X, Y : (N, 3) coordinate arrays as used by gmx_correlation.
    """
    T = X.shape[0]
    Xf = X[lag:]          # X future   shape (T-lag, 3)
    Xp = X[:-lag]         # X past     shape (T-lag, 3)
    Yp = Y[:-lag]         # Y past     shape (T-lag, 3)
    return cmi_knn(Xf, Yp, Xp, k=k)


# ── Analytical TE for coupled AR(1) ──────────────────────────────────────────

def analytical_te_ar1(a: float, b: float, sigma: float = 1.0) -> float:
    """Analytical TE(Y→X) for the bivariate AR(1) system (3-D per variable):

        X_{t+1} = a * X_t + b * Y_t + noise_x
        Y_{t+1} = a * Y_t           + noise_y

    In 1-D: TE(Y→X) = 0.5 * log(sigma_x_given_xpast^2 /
                                  sigma_x_given_xpast_ypast^2)
    The residual variance when we know X past only = sigma^2 (noise var).
    The residual variance when we know both X and Y past = sigma^2 (same),
    because b*Y_t adds information equal to b^2*var(Y_t).

    Exact 1-D formula (Schreiber 2000 / Barnett 2009):
        TE = -0.5 * log(1 - b^2 * var_y / var_x_forward)

    For the stationary process with equal sigma:
        var_y = sigma^2 / (1 - a^2)
        var_x_forward = (a^2 + b^2) * var_y + sigma^2

    We extend to 3-D independent components: TE_3D = 3 * TE_1D.
    """
    var_y    = sigma**2 / (1.0 - a**2)
    var_xf   = (a**2 + b**2) * var_y + sigma**2
    rho2     = b**2 * var_y / var_xf
    te_1d    = -0.5 * np.log(1.0 - rho2)
    return 3.0 * te_1d


def gen_ar1(N: int, a: float, b: float, sigma: float, rng) -> tuple:
    """Generate stationary bivariate AR(1) with coupling b (Y→X)."""
    X = np.zeros((N, 3))
    Y = np.zeros((N, 3))
    X[0] = rng.standard_normal(3)
    Y[0] = rng.standard_normal(3)
    for t in range(1, N):
        X[t] = a * X[t-1] + b * Y[t-1] + sigma * rng.standard_normal(3)
        Y[t] = a * Y[t-1]               + sigma * rng.standard_normal(3)
    return X, Y


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("b,tol", [
    (0.3, 0.10),
    (0.5, 0.12),
    (0.7, 0.15),
])
def test_te_vs_analytical(b, tol):
    """KNN TE estimate must match analytical AR(1) TE within tolerance."""
    rng = np.random.default_rng(seed=int(b * 100))
    N, a, sigma = 3000, 0.5, 1.0
    X, Y = gen_ar1(N, a, b, sigma, rng)

    te_est  = te_knn(X, Y, lag=1, k=5)
    te_true = analytical_te_ar1(a, b, sigma)

    err = abs(te_est - te_true)
    assert err < tol, (
        f"b={b}: TE_est={te_est:.4f}, TE_true={te_true:.4f}, |err|={err:.4f}"
    )


def test_te_direction():
    """TE(Y→X) must exceed TE(X→Y) when coupling is Y→X only."""
    rng = np.random.default_rng(55)
    N, a, b, sigma = 3000, 0.5, 0.6, 1.0
    X, Y = gen_ar1(N, a, b, sigma, rng)
    te_yx = te_knn(X, Y, lag=1, k=5)   # Y→X (true coupling)
    te_xy = te_knn(Y, X, lag=1, k=5)   # X→Y (should be ~0)
    assert te_yx > te_xy + 0.05, (
        f"TE(Y→X)={te_yx:.4f} should be > TE(X→Y)={te_xy:.4f}"
    )


def test_te_independent_near_zero():
    """TE between independent AR(1) processes should be near zero."""
    rng = np.random.default_rng(77)
    N = 2000
    a = 0.5
    X = np.zeros((N, 3)); Y = np.zeros((N, 3))
    for t in range(1, N):
        X[t] = a * X[t-1] + rng.standard_normal(3)
        Y[t] = a * Y[t-1] + rng.standard_normal(3)
    te = te_knn(X, Y, lag=1, k=5)
    assert abs(te) < 0.08, f"Expected ~0 for independent processes, got {te:.4f}"


def test_cmi_symmetry_sanity():
    """CMI(X;Y|Z) ≥ 0 for all inputs (information-theoretic lower bound)."""
    rng = np.random.default_rng(13)
    N = 1000
    X = rng.standard_normal((N, 3))
    Y = rng.standard_normal((N, 3))
    Z = rng.standard_normal((N, 3))
    cmi = cmi_knn(X, Y, Z, k=5)
    # KNN estimators can be slightly negative due to finite-sample bias,
    # but should be close to zero for independent triplets
    assert cmi > -0.1, f"CMI unexpectedly negative: {cmi:.4f}"
