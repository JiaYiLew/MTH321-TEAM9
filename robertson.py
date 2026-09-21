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