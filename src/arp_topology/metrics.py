"""
metrics.py — Scalar metrics for evaluating ARP recovery trajectories.
"""

import numpy as np


def recovery_time(t: np.ndarray, errors: np.ndarray,
                  threshold: float = 0.5) -> float:
    """
    Return the first time at which |ΔC| drops (and stays) below *threshold*.

    Parameters
    ----------
    t         : ndarray, shape (m,) — time axis.
    errors    : ndarray, shape (m,) — ΔC = C(t) – C_target time series.
    threshold : float — default 0.5 (half-way between integers).

    Returns
    -------
    float : recovery time, or np.inf if |ΔC| never drops below threshold.
    """
    mask = np.abs(errors) < threshold
    idx = np.argmax(mask)
    if not mask[idx]:
        return np.inf
    return float(t[idx])


def chern_fidelity(C_arr: np.ndarray, C_target: int) -> np.ndarray:
    """
    Element-wise fidelity: F = 1 – |C – C_target| / max(|C_target|, 1).

    A value of 1 means perfect Chern number match.

    Parameters
    ----------
    C_arr    : ndarray — Chern number time series (floats or ints).
    C_target : int     — target Chern number.

    Returns
    -------
    ndarray, same shape as C_arr.
    """
    denom = max(abs(int(C_target)), 1)
    return 1.0 - np.abs(np.asarray(C_arr, dtype=float) - C_target) / denom


def conductance_deviation(G_arr: np.ndarray, G_quantum: float) -> np.ndarray:
    """
    Relative deviation of Hall conductance from the quantised value.

    δG = |G – G_quantum| / G_quantum

    Parameters
    ----------
    G_arr     : ndarray — conductance time series.
    G_quantum : float   — target quantised conductance.

    Returns
    -------
    ndarray, same shape as G_arr.
    """
    if G_quantum == 0.0:
        return np.abs(np.asarray(G_arr, dtype=float))
    return np.abs(np.asarray(G_arr, dtype=float) - G_quantum) / abs(G_quantum)


def time_averaged_fidelity(fidelity: np.ndarray) -> float:
    """
    Return the time-averaged fidelity (mean over trajectory).

    Parameters
    ----------
    fidelity : ndarray — per-step fidelity values.

    Returns
    -------
    float
    """
    return float(np.mean(fidelity))
