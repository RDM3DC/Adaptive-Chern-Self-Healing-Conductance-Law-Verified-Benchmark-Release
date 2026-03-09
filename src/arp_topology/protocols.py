"""
protocols.py — High-level control protocols built on top of ARPLaw.

Each protocol exposes a ``run(t_span) → dict`` method that integrates the
relevant ODE and returns a trajectory dictionary with keys:
    t, M, phi, C, G, error

Protocols
---------
ARPProtocol            — full Adaptive Recovery Protocol feedback.
PrincipalBranchProtocol — uses only the principal-branch Berry phase for feedback.
NoTopologyProtocol     — amplitude-based feedback, no Chern sensing.
FixedRulerProtocol     — M is clamped to phase boundary; open-loop.
"""

import copy
import numpy as np

from .lattice import HaldaneModel
from .topology import chern_number
from .laws import ARPLaw, E2_OVER_H
from .solver import RK4Solver
from .phase_lift import wilson_loop_spectrum


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseProtocol:
    """
    Common interface for all control protocols.

    Parameters
    ----------
    model  : HaldaneModel
    config : dict — protocol configuration (gain, C_target, n_k, dt, …)
    """

    def __init__(self, model: HaldaneModel, config: dict) -> None:
        self.model = copy.deepcopy(model)
        self.config = config

    def run(self, t_span=(0.0, 20.0)) -> dict:
        """
        Integrate the protocol ODE and return trajectory data.

        Returns
        -------
        dict with keys: t, M, phi, C, G, error
            All values are numpy arrays of shape (n_steps,).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ARP — full feedback
# ---------------------------------------------------------------------------

class ARPProtocol(BaseProtocol):
    """Full Adaptive Recovery Protocol: senses Chern number, steers parameter."""

    def run(self, t_span=(0.0, 20.0)) -> dict:
        cfg = self.config
        law = ARPLaw(
            model=self.model,
            C_target=cfg.get("C_target", 1),
            gain=cfg.get("gain", 2.0),
            param=cfg.get("param", "M"),
            epsilon=cfg.get("epsilon", 0.01),
            n_k=cfg.get("n_k", 20),
        )
        dt = cfg.get("dt", 0.05)
        solver = RK4Solver(rhs=law.rhs, dt=dt)

        theta0 = np.array([getattr(self.model, cfg.get("param", "M"))])
        t_arr, states = solver.integrate(theta0, t_span)

        param = cfg.get("param", "M")
        C_target = cfg.get("C_target", 1)
        M_arr = []
        phi_arr = []
        C_arr = []
        G_arr = []

        for i, state in enumerate(states):
            law._set_param(float(state[0]))
            M_arr.append(self.model.M)
            phi_arr.append(self.model.phi)
            C = chern_number(self.model, n_k=cfg.get("n_k", 20))
            C_arr.append(C)
            G_arr.append(E2_OVER_H * C)

        C_arr = np.array(C_arr, dtype=float)
        return {
            "t": t_arr,
            "M": np.array(M_arr),
            "phi": np.array(phi_arr),
            "C": C_arr,
            "G": np.array(G_arr),
            "error": C_arr - C_target,
        }


# ---------------------------------------------------------------------------
# PrincipalBranch — feedback uses only principal phase branch
# ---------------------------------------------------------------------------

class PrincipalBranchProtocol(BaseProtocol):
    """
    Uses the mean Wilson-loop phase (principal branch, no unwrapping) as
    a proxy for Chern number. Feedback drives the phase toward π·C_target.
    """

    def run(self, t_span=(0.0, 20.0)) -> dict:
        cfg = self.config
        C_target = cfg.get("C_target", 1)
        gain = cfg.get("gain", 2.0)
        epsilon = cfg.get("epsilon", 0.01)
        n_k = cfg.get("n_k", 20)
        dt = cfg.get("dt", 0.05)
        param = cfg.get("param", "M")

        model = self.model
        target_phase = np.pi * C_target  # target total winding

        def rhs(t, state):
            setattr(model, param, float(state[0]))
            phases = wilson_loop_spectrum(model, n_k=n_k)
            # Use the mean phase as a scalar proxy
            mean_phase = float(np.mean(phases))
            error = mean_phase - target_phase
            dtheta = -gain * error / (abs(error) + epsilon)
            return np.array([dtheta])

        solver = RK4Solver(rhs=rhs, dt=dt)
        theta0 = np.array([getattr(model, param)])
        t_arr, states = solver.integrate(theta0, t_span)

        M_arr, phi_arr, C_arr, G_arr = [], [], [], []
        for state in states:
            setattr(model, param, float(state[0]))
            M_arr.append(model.M)
            phi_arr.append(model.phi)
            C = chern_number(model, n_k=n_k)
            C_arr.append(C)
            G_arr.append(E2_OVER_H * C)

        C_arr = np.array(C_arr, dtype=float)
        return {
            "t": t_arr,
            "M": np.array(M_arr),
            "phi": np.array(phi_arr),
            "C": C_arr,
            "G": np.array(G_arr),
            "error": C_arr - C_target,
        }


# ---------------------------------------------------------------------------
# NoTopology — amplitude feedback, no Chern sensing
# ---------------------------------------------------------------------------

class NoTopologyProtocol(BaseProtocol):
    """
    Amplitude-only feedback.  Drives the spectral gap (min eigenvalue gap)
    toward a positive target value; does not use Chern number.
    """

    def run(self, t_span=(0.0, 20.0)) -> dict:
        cfg = self.config
        C_target = cfg.get("C_target", 1)
        gain = cfg.get("gain", 2.0)
        epsilon = cfg.get("epsilon", 0.01)
        n_k = cfg.get("n_k", 20)
        dt = cfg.get("dt", 0.05)
        param = cfg.get("param", "M")

        model = self.model

        # Target gap amplitude: drive gap > 0
        target_gap = cfg.get("target_gap", 0.5)

        def _gap(mdl):
            """Mean spectral gap across a few BZ points."""
            b1, b2 = HaldaneModel.reciprocal_vectors()
            pts = [(i / n_k, j / n_k) for i in range(0, n_k, n_k // 4)
                   for j in range(0, n_k, n_k // 4)]
            gaps = []
            for fx, fy in pts:
                kpt = fx * b1 + fy * b2
                vals, _ = mdl.bloch_states(kpt[0], kpt[1])
                gaps.append(vals[1] - vals[0])
            return float(np.min(gaps))

        def rhs(t, state):
            setattr(model, param, float(state[0]))
            gap = _gap(model)
            error = gap - target_gap
            dtheta = -gain * error / (abs(error) + epsilon)
            return np.array([dtheta])

        solver = RK4Solver(rhs=rhs, dt=dt)
        theta0 = np.array([getattr(model, param)])
        t_arr, states = solver.integrate(theta0, t_span)

        M_arr, phi_arr, C_arr, G_arr = [], [], [], []
        for state in states:
            setattr(model, param, float(state[0]))
            M_arr.append(model.M)
            phi_arr.append(model.phi)
            C = chern_number(model, n_k=n_k)
            C_arr.append(C)
            G_arr.append(E2_OVER_H * C)

        C_arr = np.array(C_arr, dtype=float)
        return {
            "t": t_arr,
            "M": np.array(M_arr),
            "phi": np.array(phi_arr),
            "C": C_arr,
            "G": np.array(G_arr),
            "error": C_arr - C_target,
        }


# ---------------------------------------------------------------------------
# FixedRuler — M pinned to phase boundary (open-loop)
# ---------------------------------------------------------------------------

class FixedRulerProtocol(BaseProtocol):
    """
    Open-loop protocol.  M is fixed just inside the topological phase boundary
    (M = –M_c + ε_offset) for the duration of the simulation.
    No active feedback is applied.
    """

    def run(self, t_span=(0.0, 20.0)) -> dict:
        cfg = self.config
        C_target = cfg.get("C_target", 1)
        n_k = cfg.get("n_k", 20)
        dt = cfg.get("dt", 0.05)
        offset = cfg.get("boundary_offset", 0.1)

        model = self.model
        M_c = model.phase_boundary_M()
        # Pin M to –M_c + offset (inside topological phase, C=+1 side)
        model.M = -M_c + offset

        t0, tf = float(t_span[0]), float(t_span[1])
        n_steps = int((tf - t0) / dt) + 1
        t_arr = np.linspace(t0, tf, n_steps)

        C = chern_number(model, n_k=n_k)
        C_arr = np.full(n_steps, float(C))
        G_arr = np.full(n_steps, E2_OVER_H * C)
        M_arr = np.full(n_steps, model.M)
        phi_arr = np.full(n_steps, model.phi)

        return {
            "t": t_arr,
            "M": M_arr,
            "phi": phi_arr,
            "C": C_arr,
            "G": G_arr,
            "error": C_arr - C_target,
        }
