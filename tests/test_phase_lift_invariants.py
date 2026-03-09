"""
test_phase_lift_invariants.py — Tests for Wilson-loop phase lifting.
"""

import numpy as np
import pytest
from arp_topology.lattice import HaldaneModel
from arp_topology.topology import chern_number
from arp_topology.phase_lift import wilson_loop_spectrum, lift_wilson_phases


def test_lift_winding():
    """
    The winding number of the lifted Wilson-loop phases should equal the
    Chern number computed by the FHS method.
    """
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=0.0)
    n_k = 30

    C_fhs = chern_number(model, n_k=n_k)
    phases = wilson_loop_spectrum(model, n_k=n_k)
    lifted = lift_wilson_phases(phases)

    # Winding = total phase change / 2π
    winding = int(np.round((lifted[-1] - lifted[0]) / (2 * np.pi)))
    assert winding == C_fhs, (
        f"Wilson-loop winding {winding} != FHS Chern number {C_fhs}"
    )


def test_phases_continuous():
    """
    After unwrapping, consecutive lifted phase values must differ by less
    than π (no residual jumps).
    """
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=0.0)
    n_k = 30

    phases = wilson_loop_spectrum(model, n_k=n_k)
    lifted = lift_wilson_phases(phases)

    jumps = np.abs(np.diff(lifted))
    assert np.all(jumps < np.pi), (
        f"Lifted phases have jumps >= π: max jump = {jumps.max():.4f}"
    )


def test_lift_trivial_phase():
    """
    For a trivial (C=0) model the lifted phases should have zero net winding.
    """
    model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=5.0)
    n_k = 30

    phases = wilson_loop_spectrum(model, n_k=n_k)
    lifted = lift_wilson_phases(phases)

    winding = int(np.round((lifted[-1] - lifted[0]) / (2 * np.pi)))
    assert winding == 0, (
        f"Expected winding=0 for trivial phase, got {winding}"
    )
