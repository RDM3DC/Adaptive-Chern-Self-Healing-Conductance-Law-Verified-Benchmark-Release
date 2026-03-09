"""
laws.py — Adaptive Recovery Protocol (ARP): the core self-healing conductance law.

The ARP feedback ODE reads:
    dθ/dt = –K · ΔC / (|ΔC| + ε) · grad_sign

where θ is the controlled parameter (M, phi, or t2), ΔC = C(t) – C_target, and
grad_sign = sign(∂C/∂θ) is estimated by finite difference.

Physical constants
------------------
e²/h ≈ 3.874 × 10⁻⁵ S  (conductance quantum, SI)
"""

import numpy as np
from .lattice import HaldaneModel
from .topology import chern_number

E2_OVER_H = 3.874e-5  # e²/h in Siemens (SI)

_DELTA_FD = 1e-3   # finite-difference step for gradient estimation


class ARPLaw:
    """
    Adaptive Recovery Protocol feedback law.

    The controlled ODE is:
        dθ/dt = –gain · ΔC / (|ΔC| + epsilon) · grad_sign(θ)

    Parameters
    ----------
    model     : HaldaneModel
        Will be *mutated* during integration (parameter θ is updated).
    C_target  : int
        Target Chern number (+1 or –1).
    gain      : float
        Feedback gain K.
    param     : str
        Which parameter to control: ``'M'``, ``'phi'``, or ``'t2'``.
    epsilon   : float
        Regularisation denominator to avoid division by zero.
    n_k       : int
        BZ grid size for Chern number evaluation (trade-off speed vs. accuracy).
    """

    def __init__(self, model: HaldaneModel, C_target: int = 1,
                 gain: float = 1.0, param: str = 'M',
                 epsilon: float = 1e-3, n_k: int = 20) -> None:
        self.model = model
        self.C_target = int(C_target)
        self.gain = float(gain)
        self.param = param
        self.epsilon = float(epsilon)
        self.n_k = int(n_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_param(self) -> float:
        return float(getattr(self.model, self.param))

    def _set_param(self, value: float) -> None:
        setattr(self.model, self.param, float(value))

    def _chern(self) -> int:
        return chern_number(self.model, n_k=self.n_k)

    def _grad_sign(self) -> int:
        """
        Estimate sign(∂C/∂θ) by a two-point finite difference.
        Returns +1, –1, or 0 (treated as +1 to avoid stalling).
        """
        theta0 = self._get_param()
        self._set_param(theta0 + _DELTA_FD)
        Cp = self._chern()
        self._set_param(theta0 - _DELTA_FD)
        Cm = self._chern()
        self._set_param(theta0)

        dC = Cp - Cm
        if dC > 0:
            return +1
        elif dC < 0:
            return -1
        # Gradient is zero (flat region): default to –1 for M parameter
        # (decreasing M moves toward topological phase from large positive M)
        return -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_error(self) -> float:
        """Return ΔC = C(current) – C_target."""
        return float(self._chern() - self.C_target)

    def compute_conductance(self) -> float:
        """Return ideal (target) Hall conductance G = (e²/h) · C_target."""
        return E2_OVER_H * self.C_target

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        ODE right-hand side.

        Parameters
        ----------
        t     : float  (unused, here for solver compatibility)
        state : ndarray, shape (1,), state[0] = θ

        Returns
        -------
        dstate/dt : ndarray, shape (1,)
        """
        theta = float(state[0])
        self._set_param(theta)

        dC = float(self._chern() - self.C_target)
        gs = self._grad_sign()

        dtheta_dt = -self.gain * dC / (abs(dC) + self.epsilon) * gs
        return np.array([dtheta_dt])

    def conductance_from_state(self, state: np.ndarray) -> float:
        """
        Return instantaneous Hall conductance G = (e²/h) · C for the given state.
        """
        theta = float(state[0])
        self._set_param(theta)
        C = self._chern()
        return E2_OVER_H * C
