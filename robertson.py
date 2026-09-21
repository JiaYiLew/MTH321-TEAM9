"""
robertson.py
============
Robertson stiff chemical kinetics problem (ODE IVP Project, Direction 3).

Contains the RHS, the analytic Jacobian, the reduced (2x2) Jacobian, the
one-step integrators, the high-accuracy reference solver, the common output
grid and the error norms used in the report.

Author : <team members>
Course : MT3H21-2627-S1, Numerical PDEs / Numerical Analysis, Project 1
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------
# Problem constants (the brief fixes the factor convention: no extra 2 in 3e7 y2^2)
# --------------------------------------------------------------------------
K1 = 0.04
K2 = 3.0e7
K3 = 1.0e4

Y0 = np.array([1.0, 0.0, 0.0])
T0, T1 = 0.0, 40.0

# Output grid: t = 0 plus 200 logarithmically spaced points from 1e-8 to 40
N_OUT = 200
TFIRST = 1e-8


# --------------------------------------------------------------------------
# Right-hand side and Jacobians
# --------------------------------------------------------------------------
def rhs(t: float, y: np.ndarray) -> np.ndarray:
    """
    Robertson RHS, y = (y1, y2, y3).

    Overflow/invalid warnings are suppressed: an unstable explicit run is
    *expected* to overflow, and that divergence is a result, not an error.
    """
    y1, y2, y3 = y
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array([
            -K1 * y1 + K3 * y2 * y3,
            K1 * y1 - K3 * y2 * y3 - K2 * y2 * y2,
            K2 * y2 * y2,
        ])


def jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """Full 3x3 analytic Jacobian J = df/dy (column sums are zero)."""
    y1, y2, y3 = y
    return np.array([
        [-K1,          K3 * y3,          K3 * y2],
        [K1,  -K3 * y3 - 2.0 * K2 * y2,  -K3 * y2],
        [0.0,          2.0 * K2 * y2,    0.0],
    ])


def reduced_jacobian(y: np.ndarray) -> np.ndarray:
    """
    2x2 Jacobian obtained by eliminating y3 = 1 - y1 - y2.
    Its two eigenvalues are exactly the two *nonzero* eigenvalues of the full
    3x3 Jacobian, so the stiffness ratio is well defined from it.
    """
    y1, y2 = y[0], y[1]
    y3 = 1.0 - y1 - y2
    a = -K1 - K3 * y2
    b = K3 * (y3 - y2)
    c = K1 + K3 * y2
    d = -K3 * (y3 - y2) - 2.0 * K2 * y2
    return np.array([[a, b], [c, d]])


def nonzero_eigenvalues(y: np.ndarray) -> np.ndarray:
    """The two nonzero Jacobian eigenvalues (from the reduced Jacobian)."""
    return np.linalg.eigvals(reduced_jacobian(y))


def stiffness_ratio(y: np.ndarray, zero_tol: float = 1e-12):
    """
    S(t) = max|Re lam_j| / min|Re lam_j| over the two nonzero eigenvalues.
    Returns (S, lam_fast, lam_slow). S is nan when the pair is degenerate
    (e.g. at t = 0, where both are zero).
    """
    lam = nonzero_eigenvalues(y)
    mag = np.abs(lam.real)
    mag = np.sort(mag)[::-1]
    if mag[-1] <= zero_tol:
        return np.nan, mag[0], mag[-1]
    return mag[0] / mag[-1], mag[0], mag[-1]
