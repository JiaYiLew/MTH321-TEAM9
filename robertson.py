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


# --------------------------------------------------------------------------
# Fixed-step integrators (all land exactly on a supplied grid)
# --------------------------------------------------------------------------
def explicit_euler(f, y0, t_grid):
    y = np.array(y0, dtype=float)
    out = np.empty((len(t_grid), len(y)))
    out[0] = y
    for n in range(len(t_grid) - 1):
        h = t_grid[n + 1] - t_grid[n]
        y = y + h * f(t_grid[n], y)
        out[n + 1] = y
    return out


def heun(f, y0, t_grid):
    """Explicit RK2 (Heun) — order 2."""
    y = np.array(y0, dtype=float)
    out = np.empty((len(t_grid), len(y)))
    out[0] = y
    for n in range(len(t_grid) - 1):
        h = t_grid[n + 1] - t_grid[n]
        t_n = t_grid[n]
        k1 = f(t_n, y)
        k2 = f(t_n + h, y + h * k1)
        y = y + 0.5 * h * (k1 + k2)
        out[n + 1] = y
    return out


def _newton_solve(F, J, w0, tol=1e-12, maxit=50):
    """
    Damped Newton for F(w) = 0 with Jacobian J(w).
    Damping alpha is halved while the residual fails to decrease.
    Returns (w, iterations, converged).
    """
    w = np.array(w0, dtype=float)
    res = np.linalg.norm(F(w), np.inf)
    for it in range(maxit):
        if res < tol:
            return w, it, True
        delta = np.linalg.solve(J(w), -F(w))
        alpha = 1.0
        for _ in range(30):
            w_try = w + alpha * delta
            res_try = np.linalg.norm(F(w_try), np.inf)
            if res_try < res:
                break
            alpha *= 0.5
        else:
            return w, it, False
        w, res = w_try, res_try
    return w, maxit, res < tol


def backward_euler(f, jf, y0, t_grid, newton_tol=1e-12):
    """
    Implicit (backward) Euler with a damped-Newton solve and the analytic
    Jacobian:  F(w) = w - y_n - h f(t_{n+1}, w) = 0,  J_F = I - h J_f.
    """
    y = np.array(y0, dtype=float)
    out = np.empty((len(t_grid), len(y)))
    iters = np.zeros(len(t_grid) - 1, dtype=int)
    out[0] = y
    for n in range(len(t_grid) - 1):
        h = t_grid[n + 1] - t_grid[n]
        t_next = t_grid[n + 1]

        def F(w):
            return w - y - h * f(t_next, w)

        def J(w):
            return np.eye(len(y)) - h * jf(t_next, w)

        w0 = y + h * f(t_grid[n], y)          # explicit-Euler predictor
        y, k, _ = _newton_solve(F, J, w0, tol=newton_tol)
        iters[n] = k
        out[n + 1] = y
    return out, iters


def trapezoidal(f, jf, y0, t_grid, newton_tol=1e-12):
    """
    Implicit trapezoidal rule (A-stable, order 2 — but NOT L-stable):
      w = y_n + h/2 (f(t_n, y_n) + f(t_{n+1}, w)).
    """
    y = np.array(y0, dtype=float)
    out = np.empty((len(t_grid), len(y)))
    out[0] = y
    for n in range(len(t_grid) - 1):
        h = t_grid[n + 1] - t_grid[n]
        t_n, t_next = t_grid[n], t_grid[n + 1]
        f_n = f(t_n, y)

        def F(w):
            return w - y - 0.5 * h * (f_n + f(t_next, w))

        def J(w):
            return np.eye(len(y)) - 0.5 * h * jf(t_next, w)

        w0 = y + h * f_n
        y, _, _ = _newton_solve(F, J, w0, tol=newton_tol)
        out[n + 1] = y
    return out


# --------------------------------------------------------------------------
# Adaptive step-size control (step doubling)
#
# This is the deep-work solver from the Week-2 scaffold, generalised so it
# works for any one-step method and any system. The step is driven by the
# method's own local-error signal instead of staying fixed:
#
#     y_coarse = one  step of size h
#     y_fine   = two  steps of size h/2
#     e = ||y_fine - y_coarse||      (an O(h^{p+1}) local-error estimate)
#
# Controller: accept if e <= tol; halve h if e > tol; double h (up to h0) if
# e < tol/10. Accept y_fine, the more accurate of the two candidates.
# --------------------------------------------------------------------------
def adaptive_implicit_euler(f, jf, y0, t0, t1, h0, tol,
                            newton_tol=1e-12, max_steps=200000,
                            method="backward_euler"):
    """
    Step-doubling adaptive implicit solver.

    method='backward_euler' (order 1, L-stable) or 'trapezoidal' (order 2,
    A-stable but not L-stable). The order-2 method buys much more per step,
    so its controller is the more convincing demonstration of adaptivity.

    Returns (t, y, h_used, n_rejected) where t is the accepted grid, y the
    solution on it, and h_used the step actually taken at each accepted step
    (len(h_used) + 1 == len(t)).
    """
    step_fn = _be_step if method == "backward_euler" else _tr_step

    t = [t0]
    y = [np.array(y0, dtype=float)]
    h_used = []
    h = min(h0, t1 - t0)
    n_rejected = 0

    for _ in range(max_steps):
        if t[-1] >= t1:
            break
        h = min(h, t1 - t[-1])            # never overshoot the end

        y_coarse, _, ok1 = step_fn(f, jf, t[-1], y[-1], h, newton_tol)
        y_half, _, ok2 = step_fn(f, jf, t[-1], y[-1], 0.5 * h, newton_tol)
        y_fine, _, ok3 = step_fn(f, jf, t[-1] + 0.5 * h, y_half, 0.5 * h, newton_tol)

        if not (ok1 and ok2 and ok3):
            h *= 0.5
            n_rejected += 1
            continue

        e = float(np.linalg.norm(y_fine - y_coarse, np.inf))
        if e <= tol:
            t.append(t[-1] + h)
            y.append(y_fine)
            h_used.append(h)
            if e < tol / 10.0:            # comfortably inside budget: grow
                h = min(h0, 2.0 * h)
        else:
            h *= 0.5                      # over budget: shrink and retry
            n_rejected += 1

    return np.array(t), np.array(y), np.array(h_used), n_rejected


def _be_step(f, jf, t_n, y_n, h, newton_tol):
    """One backward-Euler step with the damped-Newton solve. Returns (y, k, ok)."""
    def F(w):
        return w - y_n - h * f(t_n + h, w)

    def J(w):
        return np.eye(len(y_n)) - h * jf(t_n + h, w)

    w0 = y_n + h * f(t_n, y_n)
    return _newton_solve(F, J, w0, tol=newton_tol)

