"""
test_solver_consistency.py — Tests for RK4 and RK45 integrators.
"""

import numpy as np
import pytest
from arp_topology.solver import RK4Solver, RK45Solver


def _linear_rhs(t, state):
    """dx/dt = –x  →  exact solution x(t) = x(0) exp(–t)."""
    return -np.asarray(state)


def test_rk4_linear_ode():
    """RK4 should match exp(–1) within 1% for dx/dt = –x."""
    solver = RK4Solver(rhs=_linear_rhs, dt=0.01)
    t, states = solver.integrate(np.array([1.0]), (0.0, 1.0))
    x_numerical = states[-1, 0]
    x_exact = np.exp(-1.0)
    rel_err = abs(x_numerical - x_exact) / x_exact
    assert rel_err < 0.01, (
        f"RK4 relative error {rel_err:.4e} exceeds 1% (got {x_numerical:.6f}, "
        f"expected {x_exact:.6f})"
    )


def test_rk45_matches_rk4():
    """RK45 and RK4 should agree to within 1e-3 for the linear ODE."""
    rk4 = RK4Solver(rhs=_linear_rhs, dt=0.005)
    rk45 = RK45Solver(rhs=_linear_rhs, rtol=1e-6, atol=1e-8)

    t4, s4 = rk4.integrate(np.array([1.0]), (0.0, 1.0))
    t45, s45 = rk45.integrate(np.array([1.0]), (0.0, 1.0))

    # Evaluate both at t=1
    x4 = s4[-1, 0]
    x45 = s45[-1, 0]
    assert abs(x4 - x45) < 1e-3, (
        f"RK4 vs RK45 max deviation {abs(x4-x45):.2e} exceeds 1e-3"
    )


def test_rk4_conserves_vector_dimension():
    """RK4 should handle multi-component state."""
    def rhs2d(t, s):
        return np.array([-s[0], -2 * s[1]])

    solver = RK4Solver(rhs=rhs2d, dt=0.01)
    t, states = solver.integrate(np.array([1.0, 2.0]), (0.0, 0.5))
    assert states.shape[1] == 2
    # Check both components decay
    assert states[-1, 0] < states[0, 0]
    assert states[-1, 1] < states[0, 1]
