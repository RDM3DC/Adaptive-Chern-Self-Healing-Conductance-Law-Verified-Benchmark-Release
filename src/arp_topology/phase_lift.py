"""
phase_lift.py — Wilson-loop phase lifting (unwrapping) for Chern number detection.

The Wilson loop W(ky) is the ordered product of Berry-phase link matrices around
a closed loop in kx at fixed ky.  Its eigenvalue phases θ_n(ky) wind by 2π·C
as ky sweeps the BZ, providing an independent confirmation of the Chern number.

Implementation note
-------------------
numpy.linalg.eigh may return eigenvectors with opposite signs at k and k+G
(even though H(k) = H(k+G)) due to tiny floating-point perturbations in H.
This causes the naïve Wilson loop to pick up a spurious factor of –1 (phase π)
on the closing link.  The fix is to *pre-cache* all n_k eigenvectors per row and
close the loop with the stored entry at i=0 — ensuring a consistent gauge for
both the first and last link of the product.
"""

import numpy as np
from .lattice import HaldaneModel


def wilson_loop_spectrum(model: HaldaneModel, n_k: int = 50) -> np.ndarray:
    """
    Compute Wilson-loop eigenvalue phases θ(ky) for the occupied band.

    For each row ky_j = j/n_k · b2, the Wilson loop is
        W(ky) = Π_{i=0}^{n_k-1}  ⟨u(kx_i, ky)| u(kx_{i+1 mod n_k}, ky)⟩
    The closing link uses the pre-cached eigenvector at i=0, not a freshly
    evaluated one at kx = b1 + ky_j, which avoids sign-flip gauge artefacts.

    Parameters
    ----------
    model : HaldaneModel
    n_k   : int, grid resolution.

    Returns
    -------
    phases : ndarray, shape (n_k,)
        Wilson-loop phases in [−π, π) for each ky row.
    """
    b1, b2 = HaldaneModel.reciprocal_vectors()

    phases = np.zeros(n_k)
    for j in range(n_k):
        ky_frac = j / n_k

        # Pre-compute all n_k occupied eigenvectors along this kx row
        u_row = np.zeros((n_k, 2), dtype=complex)
        for i in range(n_k):
            kpt = (i / n_k) * b1 + ky_frac * b2
            _, vecs = model.bloch_states(kpt[0], kpt[1])
            u_row[i] = vecs[:, 0]

        # Accumulate Wilson loop with consistent closing link (i+1) % n_k
        # Using the reverse link ⟨u_{i+1}|u_i⟩ to match the FHS Chern sign convention
        W = complex(1.0, 0.0)
        for i in range(n_k):
            W *= np.vdot(u_row[(i + 1) % n_k], u_row[i])

        phases[j] = np.angle(W)

    return phases


def lift_wilson_phases(phases: np.ndarray) -> np.ndarray:
    """
    Unwrap Wilson-loop phases along the ky axis to yield a continuous spectrum.

    Parameters
    ----------
    phases : ndarray, shape (n_k,) or (n_k, n_bands)
        Raw Wilson-loop phases in [−π, π).

    Returns
    -------
    lifted : ndarray, same shape as *phases*
        Continuously lifted phases (unwrapped).
    """
    return np.unwrap(phases, axis=0)
