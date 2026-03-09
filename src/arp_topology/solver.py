"""
solver.py — Fixed-step RK4 and adaptive Dormand–Prince RK45 integrators.

Both solvers share the same interface:
    solver.integrate(state0, t_span) → (t_array, state_array)
"""

import numpy as np


# ---------------------------------------------------------------------------
# RK4 (classic 4th-order Runge–Kutta, fixed step)
# ---------------------------------------------------------------------------

class RK4Solver:
    """
    Classical 4th-order Runge–Kutta integrator with fixed time step.

    Parameters
    ----------
    rhs : callable
        Right-hand side function ``f(t, state) → dstate/dt``.
    dt  : float
        Fixed time step.
    """

    def __init__(self, rhs, dt: float = 0.01) -> None:
        self.rhs = rhs
        self.dt = float(dt)

    def integrate(self, state0: np.ndarray, t_span: tuple, **kwargs) -> tuple:
        """
        Integrate from t_span[0] to t_span[1].

        Parameters
        ----------
        state0 : ndarray, shape (n,)
        t_span : (t0, tf)

        Returns
        -------
        t      : ndarray, shape (m,)
        states : ndarray, shape (m, n)
        """
        t0, tf = float(t_span[0]), float(t_span[1])
        dt = self.dt

        t = t0
        state = np.array(state0, dtype=float)
        ts = [t]
        states = [state.copy()]

        while t < tf - 1e-12:
            h = min(dt, tf - t)
            k1 = np.asarray(self.rhs(t, state), dtype=float)
            k2 = np.asarray(self.rhs(t + h / 2, state + h / 2 * k1), dtype=float)
            k3 = np.asarray(self.rhs(t + h / 2, state + h / 2 * k2), dtype=float)
            k4 = np.asarray(self.rhs(t + h, state + h * k3), dtype=float)

            state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            ts.append(t)
            states.append(state.copy())

        return np.array(ts), np.array(states)


# ---------------------------------------------------------------------------
# RK45 — Dormand–Prince adaptive step integrator
# ---------------------------------------------------------------------------

# Dormand–Prince tableau coefficients
_C2, _C3, _C4, _C5 = 1/5, 3/10, 4/5, 8/9
_A21 = 1/5
_A31, _A32 = 3/40, 9/40
_A41, _A42, _A43 = 44/45, -56/15, 32/9
_A51, _A52, _A53, _A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
_A61, _A62, _A63, _A64, _A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

# 5th-order weights
_B1, _B3, _B4, _B5, _B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

# Error estimate (5th minus 4th order)
_E1 = 71/57600
_E3 = -71/16695
_E4 = 71/1920
_E5 = -17253/339200
_E6 = 22/525
_E7 = -1/40


class RK45Solver:
    """
    Adaptive-step Dormand–Prince RK45 integrator.

    Parameters
    ----------
    rhs  : callable
        ``f(t, state) → dstate/dt``
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    """

    def __init__(self, rhs, rtol: float = 1e-4, atol: float = 1e-6) -> None:
        self.rhs = rhs
        self.rtol = float(rtol)
        self.atol = float(atol)

    def _error_norm(self, err: np.ndarray, y: np.ndarray) -> float:
        sc = self.atol + self.rtol * np.abs(y)
        return float(np.sqrt(np.mean((err / sc) ** 2)))

    def integrate(self, state0: np.ndarray, t_span: tuple, **kwargs) -> tuple:
        """
        Integrate from t_span[0] to t_span[1] with adaptive step control.

        Returns
        -------
        t      : ndarray
        states : ndarray, shape (m, n)
        """
        t0, tf = float(t_span[0]), float(t_span[1])
        state = np.array(state0, dtype=float)

        # Initial step size heuristic
        h = (tf - t0) / 100.0
        h_min = 1e-10
        h_max = (tf - t0) / 5.0

        t = t0
        ts = [t]
        states = [state.copy()]

        f0 = np.asarray(self.rhs(t, state), dtype=float)

        while t < tf - 1e-12:
            h = min(h, tf - t)

            k1 = f0
            k2 = np.asarray(self.rhs(t + _C2 * h, state + h * _A21 * k1), dtype=float)
            k3 = np.asarray(self.rhs(t + _C3 * h, state + h * (_A31 * k1 + _A32 * k2)), dtype=float)
            k4 = np.asarray(self.rhs(t + _C4 * h, state + h * (_A41 * k1 + _A42 * k2 + _A43 * k3)), dtype=float)
            k5 = np.asarray(self.rhs(t + _C5 * h, state + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4)), dtype=float)
            k6 = np.asarray(self.rhs(t + h, state + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5)), dtype=float)

            # 5th-order solution
            y_new = state + h * (_B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)

            k7 = np.asarray(self.rhs(t + h, y_new), dtype=float)

            # Error estimate
            err = h * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
            err_norm = self._error_norm(err, y_new)

            if err_norm <= 1.0 or h <= h_min:
                # Accept step
                t += h
                state = y_new
                f0 = k7
                ts.append(t)
                states.append(state.copy())

            # Adjust step size (PI controller)
            if err_norm == 0.0:
                factor = 5.0
            else:
                factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            h = min(h_max, max(h_min, h * factor))

        return np.array(ts), np.array(states)
