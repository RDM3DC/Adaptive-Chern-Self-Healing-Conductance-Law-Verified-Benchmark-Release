"""
test_reductions.py — Tests verifying Chern number reduces correctly across
the phase diagram.
"""

import numpy as np
import pytest
from arp_topology.lattice import HaldaneModel
from arp_topology.topology import chern_number


def test_M_sweep():
    """
    Chern number as a function of M at phi=pi/2.

    Expected pattern (t2=0.3, phase boundary M_c = 3√3·0.3 ≈ 1.56):
      M < -M_c  →  C = 0
      |M| < M_c →  C = ±1
      M > +M_c  →  C = 0
    """
    phi = np.pi / 2
    t2 = 0.3
    M_vals = np.linspace(-6.0, 6.0, 7)
    M_c = 3.0 * np.sqrt(3.0) * abs(t2 * np.sin(phi))

    for M in M_vals:
        model = HaldaneModel(t1=1.0, t2=t2, phi=phi, M=M)
        C = chern_number(model, n_k=30)
        if abs(M) > M_c + 0.3:       # clearly outside
            assert C == 0, f"Expected C=0 at M={M:.2f}, got C={C}"
        elif abs(M) < M_c - 0.3:    # clearly inside
            assert abs(C) == 1, f"Expected |C|=1 at M={M:.2f}, got C={C}"


def test_phi_sweep():
    """
    Chern number for varying phi at M=0 (always inside topological phase
    when t2 sin(phi) != 0).
    """
    # At phi = 0 or pi the phase boundary vanishes (sin(phi) = 0), so
    # M=0 sits at the gap-closing point; skip these special values.
    phi_vals = [np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]
    for phi in phi_vals:
        model = HaldaneModel(t1=1.0, t2=0.3, phi=phi, M=0.0)
        C = chern_number(model, n_k=30)
        assert abs(C) == 1, (
            f"Expected |C|=1 at phi={phi:.3f}, got C={C}"
        )


def test_chern_quantized():
    """
    For several random models inside the topological phase, C must be an integer.
    """
    rng = np.random.default_rng(42)
    for _ in range(8):
        phi = rng.uniform(0.1 * np.pi, 0.9 * np.pi)
        t2 = rng.uniform(0.1, 0.5)
        M_c = 3.0 * np.sqrt(3.0) * abs(t2 * np.sin(phi))
        # Pick M well inside the topological phase
        M = rng.uniform(-0.8 * M_c, 0.8 * M_c)
        model = HaldaneModel(t1=1.0, t2=t2, phi=phi, M=M)
        C = chern_number(model, n_k=30)
        assert C in (-1, 0, 1), f"C not quantized: C={C}"
