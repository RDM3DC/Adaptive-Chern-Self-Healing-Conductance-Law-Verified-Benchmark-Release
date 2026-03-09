"""
arp_topology — Adaptive Chern Self-Healing Conductance Law
==========================================================
Public API surface.
"""

from .lattice import HaldaneModel
from .topology import chern_number, berry_curvature
from .phase_lift import lift_wilson_phases, wilson_loop_spectrum
from .laws import ARPLaw
from .solver import RK4Solver, RK45Solver
from .metrics import recovery_time, chern_fidelity, conductance_deviation
from .protocols import (
    ARPProtocol,
    PrincipalBranchProtocol,
    NoTopologyProtocol,
    FixedRulerProtocol,
)
from .plotting import (
    plot_phase_diagram,
    plot_recovery,
    plot_berry_curvature,
    plot_wilson_spectrum,
)

__version__ = "0.1.0"
__all__ = [
    "HaldaneModel",
    "chern_number", "berry_curvature",
    "lift_wilson_phases", "wilson_loop_spectrum",
    "ARPLaw",
    "RK4Solver", "RK45Solver",
    "recovery_time", "chern_fidelity", "conductance_deviation",
    "ARPProtocol", "PrincipalBranchProtocol",
    "NoTopologyProtocol", "FixedRulerProtocol",
    "plot_phase_diagram", "plot_recovery",
    "plot_berry_curvature", "plot_wilson_spectrum",
]
