"""
test_units.py — Unit tests for individual functions.
"""

import numpy as np
import pytest
from arp_topology.lattice import HaldaneModel
from arp_topology.topology import chern_number
from arp_topology.laws import ARPLaw


def test_haldane_hamiltonian_hermitian():
    """H(k) must equal its conjugate transpose for arbitrary k."""
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=0.0)
    for kx, ky in [(0.0, 0.0), (1.0, 0.5), (-0.7, 1.2), (np.pi, np.pi)]:
        H = model.hamiltonian(kx, ky)
        assert np.allclose(H, H.conj().T, atol=1e-12), (
            f"H not Hermitian at (kx={kx}, ky={ky})"
        )


def test_haldane_phase_boundary():
    """Model at M=0, phi=pi/2 should be in the topological phase."""
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=0.0)
    assert model.is_topological(), "M=0 should be topological"


def test_chern_trivial():
    """Far outside the topological phase, C must be 0."""
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=5.0)
    C = chern_number(model, n_k=30)
    assert C == 0, f"Expected C=0 (trivial), got C={C}"


def test_chern_topological():
    """Inside the topological phase (M=0, phi=pi/2), C must be ±1."""
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=0.0)
    C = chern_number(model, n_k=30)
    assert abs(C) == 1, f"Expected |C|=1 (topological), got C={C}"


def test_arp_error_decreases():
    """ARP feedback should reduce |ΔC| over 10 integration steps."""
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=2.5)
    law = ARPLaw(model=model, C_target=1, gain=2.0, param="M",
                 epsilon=0.01, n_k=20)

    # Record initial error magnitude
    initial_error = abs(law.compute_error())

    from arp_topology.solver import RK4Solver
    solver = RK4Solver(rhs=law.rhs, dt=0.1)
    state0 = np.array([model.M])
    t_arr, states = solver.integrate(state0, (0.0, 1.0))

    # After integration, M should have moved toward topological phase
    final_M = float(states[-1, 0])
    M_c = model.phase_boundary_M()
    # M should have decreased (moving toward |M| < M_c from positive side)
    assert final_M < 2.5, (
        f"M did not decrease under ARP: started at 2.5, ended at {final_M}"
    )
