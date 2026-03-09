"""
lattice.py — Haldane model on the honeycomb lattice.

The Haldane model is a 2-band tight-binding model that exhibits quantum Hall
physics (non-zero Chern number) without net magnetic flux.

Reference:
    Haldane, F. D. M. (1988). Phys. Rev. Lett. 61, 2015.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------
_A1 = np.array([1.0, 0.0])
_A2 = np.array([0.5, np.sqrt(3.0) / 2.0])

# NNN vectors (CCW around plaquettes, standard Haldane convention)
# These give C = +1 for φ > 0 inside the topological phase and
# phase boundary M_c = 3√3 t2 |sin φ|
_D1 = _A2 - _A1       # a2 − a1
_D2 = -_A2            # −a2
_D3 = _A1             # +a1


class HaldaneModel:
    """
    Haldane tight-binding model on the honeycomb lattice.

    Parameters
    ----------
    t1 : float
        Nearest-neighbour (NN) hopping amplitude.
    t2 : float
        Next-nearest-neighbour (NNN) hopping amplitude.
    phi : float
        NNN hopping phase (breaks time-reversal symmetry).
    M : float
        Staggered on-site potential (breaks inversion symmetry).
    """

    def __init__(self, t1: float = 1.0, t2: float = 0.3,
                 phi: float = np.pi / 2, M: float = 0.0) -> None:
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.phi = float(phi)
        self.M = float(M)

    # ------------------------------------------------------------------
    # Bloch Hamiltonian
    # ------------------------------------------------------------------

    def _d_vec(self, kx: float, ky: float):
        """Return (d0, dx, dy, dz) for the Bloch Hamiltonian."""
        k = np.array([kx, ky])

        # NNN structure factors
        nnn_cos = (np.cos(k @ _D1) + np.cos(k @ _D2) + np.cos(k @ _D3))
        nnn_sin = (np.sin(k @ _D1) + np.sin(k @ _D2) + np.sin(k @ _D3))

        d0 = 2.0 * self.t2 * np.cos(self.phi) * nnn_cos
        dx = self.t1 * (1.0 + np.cos(k @ _A1) + np.cos(k @ _A2))
        dy = self.t1 * (np.sin(k @ _A1) + np.sin(k @ _A2))
        dz = self.M - 2.0 * self.t2 * np.sin(self.phi) * nnn_sin
        return d0, dx, dy, dz

    def hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        Return the 2×2 Bloch Hamiltonian H(k).

        H(k) = d0·I + dx·σ₁ + dy·σ₂ + dz·σ₃
        """
        d0, dx, dy, dz = self._d_vec(kx, ky)
        H = np.array(
            [[d0 + dz,      dx - 1j * dy],
             [dx + 1j * dy, d0 - dz     ]],
            dtype=complex,
        )
        return H

    def bloch_states(self, kx: float, ky: float):
        """
        Diagonalise H(k).

        Returns
        -------
        eigenvalues : ndarray, shape (2,)
            Sorted in ascending order.
        eigenvectors : ndarray, shape (2, 2)
            Column *j* is the eigenvector for eigenvalue *j*.
        """
        H = self.hamiltonian(kx, ky)
        vals, vecs = np.linalg.eigh(H)
        return vals, vecs

    # ------------------------------------------------------------------
    # Phase boundary / topology diagnostics
    # ------------------------------------------------------------------

    def phase_boundary_M(self, phi: float = None, t2: float = None) -> float:
        """
        Return the critical |M| at which the gap closes.

        |M_c| = 3√3 · |t2 · sin(φ)|
        """
        phi = self.phi if phi is None else phi
        t2 = self.t2 if t2 is None else t2
        return 3.0 * np.sqrt(3.0) * abs(t2 * np.sin(phi))

    def is_topological(self, M: float = None, phi: float = None,
                       t2: float = None) -> bool:
        """
        Return True if the model is in a topological phase (|M| < M_c).
        """
        M = self.M if M is None else M
        return abs(M) < self.phase_boundary_M(phi=phi, t2=t2)

    def chern_phase(self) -> int:
        """
        Return the expected Chern number (+1 or –1) in the topological phase.

        Convention: C = +1 when M < 0 (inside topological phase), C = –1 when M > 0.
        For M = 0 return +1 by convention.
        """
        if self.M <= 0.0:
            return +1
        return -1

    # ------------------------------------------------------------------
    # Reciprocal lattice helpers
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_vectors():
        """
        Return reciprocal lattice vectors b1, b2 satisfying ai·bj = 2π δij.
        """
        a1, a2 = _A1, _A2
        # 2-D formula: b1 = 2π (a2⊥) / (a1 · a2⊥), etc.
        area = a1[0] * a2[1] - a1[1] * a2[0]
        b1 = 2.0 * np.pi * np.array([ a2[1], -a2[0]]) / area
        b2 = 2.0 * np.pi * np.array([-a1[1],  a1[0]]) / area
        return b1, b2

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HaldaneModel(t1={self.t1}, t2={self.t2}, "
            f"phi={self.phi:.4f}, M={self.M})"
        )
