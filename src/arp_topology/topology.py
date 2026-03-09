"""
topology.py — Chern number and Berry curvature via the
Fukui–Hatsugai–Suzuki (FHS) lattice gauge method.

Reference:
    Fukui, Hatsugai, Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005).
"""

import numpy as np
from .lattice import HaldaneModel


def _occupied_state(model: HaldaneModel, kx: float, ky: float) -> np.ndarray:
    """Return the lower-band (occupied) eigenvector at (kx, ky)."""
    _, vecs = model.bloch_states(kx, ky)
    return vecs[:, 0]   # lowest eigenvalue column


def _bz_grid(n_k: int):
    """
    Return a uniform (n_k × n_k) grid of (kx, ky) momenta spanning the BZ.

    We use crystal coordinates: k = (i/n_k)*b1 + (j/n_k)*b2
    with i,j in {0, …, n_k-1}.
    """
    b1, b2 = HaldaneModel.reciprocal_vectors()
    idx = np.arange(n_k)
    # shape: (n_k, n_k, 2)
    kx = np.outer(idx / n_k, b1[0]) + np.outer(np.ones(n_k), (idx / n_k) * b2[0])
    ky = np.outer(idx / n_k, b1[1]) + np.outer(np.ones(n_k), (idx / n_k) * b2[1])
    # reshape to (n_k, n_k) grids of kx, ky components
    KX = np.zeros((n_k, n_k))
    KY = np.zeros((n_k, n_k))
    for i in range(n_k):
        for j in range(n_k):
            kpt = (i / n_k) * b1 + (j / n_k) * b2
            KX[i, j] = kpt[0]
            KY[i, j] = kpt[1]
    return KX, KY, b1 / n_k, b2 / n_k


def berry_curvature(model: HaldaneModel, n_k: int = 50) -> np.ndarray:
    """
    Return Berry curvature F[i, j] on an n_k × n_k BZ grid for the lower band.

    Uses the Fukui–Hatsugai–Suzuki lattice gauge approach:
        F(k) = Im log[ U1(k) U2(k+dk1) U1*(k+dk2) U2*(k) ] / (2π)
    where U1(k) = <u(k)|u(k+dk1)> and U2(k) = <u(k)|u(k+dk2)>.

    Parameters
    ----------
    model : HaldaneModel
    n_k   : int, grid resolution per direction.

    Returns
    -------
    F : ndarray, shape (n_k, n_k)
        Berry curvature in units of 1/(BZ area).
    """
    b1, b2 = HaldaneModel.reciprocal_vectors()
    db1 = b1 / n_k
    db2 = b2 / n_k

    # Pre-compute all occupied states
    states = np.zeros((n_k, n_k, 2), dtype=complex)
    for i in range(n_k):
        for j in range(n_k):
            kpt = (i / n_k) * b1 + (j / n_k) * b2
            states[i, j] = _occupied_state(model, kpt[0], kpt[1])

    def u(i, j):
        return states[i % n_k, j % n_k]

    F = np.zeros((n_k, n_k))
    for i in range(n_k):
        for j in range(n_k):
            U1 = np.vdot(u(i, j), u(i + 1, j))          # <u(k)|u(k+dk1)>
            U2 = np.vdot(u(i, j), u(i, j + 1))          # <u(k)|u(k+dk2)>
            U1p = np.vdot(u(i, j + 1), u(i + 1, j + 1)) # <u(k+dk2)|u(k+dk1+dk2)>
            U2p = np.vdot(u(i + 1, j), u(i + 1, j + 1)) # <u(k+dk1)|u(k+dk1+dk2)>
            plaq = U1 * U2p * np.conj(U1p) * np.conj(U2)
            F[i, j] = np.angle(plaq) / (2.0 * np.pi)

    return F


def chern_number(model: HaldaneModel, n_k: int = 50) -> int:
    """
    Return the Chern number (integer) of the lower band using the FHS method.

    Parameters
    ----------
    model : HaldaneModel
    n_k   : int, BZ grid resolution (use ≥ 30 for accurate results).

    Returns
    -------
    C : int
        Chern number, rounded from the sum of Berry curvature plaquettes.
    """
    F = berry_curvature(model, n_k=n_k)
    return int(np.round(np.sum(F)))
