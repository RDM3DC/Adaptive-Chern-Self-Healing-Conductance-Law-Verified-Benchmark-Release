"""
plotting.py — Matplotlib-based visualisation for arp_topology.

All functions use the non-interactive ``Agg`` backend and accept an optional
``save_path`` argument to write the figure to disk.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lattice import HaldaneModel
from .topology import chern_number, berry_curvature
from .phase_lift import wilson_loop_spectrum, lift_wilson_phases


def plot_phase_diagram(model_cls=HaldaneModel, n_M: int = 60, n_phi: int = 60,
                       n_k: int = 20, save_path: str = None) -> plt.Figure:
    """
    Plot the topological phase diagram in (M/t2, φ) space.

    Parameters
    ----------
    model_cls : class (default HaldaneModel)
    n_M       : int, number of M values.
    n_phi     : int, number of φ values.
    n_k       : int, BZ grid for Chern computation.
    save_path : str or None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    M_vals = np.linspace(-6.0, 6.0, n_M)
    phi_vals = np.linspace(0.0, np.pi, n_phi)
    C_grid = np.zeros((n_phi, n_M))

    for j, M in enumerate(M_vals):
        for i, phi in enumerate(phi_vals):
            model = model_cls(t1=1.0, t2=0.3, phi=phi, M=M)
            C_grid[i, j] = chern_number(model, n_k=n_k)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(M_vals, phi_vals / np.pi, C_grid,
                       cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    fig.colorbar(im, ax=ax, label="Chern number C")
    ax.set_xlabel("M / (t1=1)")
    ax.set_ylabel("φ / π")
    ax.set_title("Haldane Model Phase Diagram")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_recovery(result: dict, protocol_name: str = "",
                  save_path: str = None) -> plt.Figure:
    """
    Plot the ARP recovery trajectory: Chern error and conductance vs. time.

    Parameters
    ----------
    result        : dict from Protocol.run() with keys t, C, G, error, M.
    protocol_name : str, title suffix.
    save_path     : str or None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = result["t"]
    error = result["error"]
    G = result["G"]
    M = result["M"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(t, error, color="C1", lw=1.5)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("ΔC = C(t) – C*")
    axes[0].set_title(f"ARP Recovery{' — ' + protocol_name if protocol_name else ''}")

    axes[1].plot(t, G * 1e5, color="C0", lw=1.5)
    axes[1].set_ylabel("G / 10⁻⁵ S")

    axes[2].plot(t, M, color="C2", lw=1.5)
    axes[2].set_ylabel("M(t)")
    axes[2].set_xlabel("Time")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_berry_curvature(model: HaldaneModel, n_k: int = 50,
                         save_path: str = None) -> plt.Figure:
    """
    Plot the Berry curvature F(kx, ky) on the BZ grid.

    Parameters
    ----------
    model     : HaldaneModel
    n_k       : int
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    F = berry_curvature(model, n_k=n_k)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(F, origin="lower", extent=[0, 1, 0, 1],
                   cmap="seismic", vmin=-np.abs(F).max(), vmax=np.abs(F).max())
    fig.colorbar(im, ax=ax, label="Berry curvature F(k)")
    ax.set_xlabel("kx (BZ fraction)")
    ax.set_ylabel("ky (BZ fraction)")
    C = int(np.round(np.sum(F)))
    ax.set_title(f"Berry Curvature  (C = {C})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_wilson_spectrum(model: HaldaneModel, n_k: int = 50,
                         save_path: str = None) -> plt.Figure:
    """
    Plot the Wilson-loop eigenvalue spectrum θ(ky).

    Parameters
    ----------
    model     : HaldaneModel
    n_k       : int
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    phases = wilson_loop_spectrum(model, n_k=n_k)
    lifted = lift_wilson_phases(phases)
    ky_frac = np.linspace(0, 1, n_k, endpoint=False)

    winding = int(np.round((lifted[-1] - lifted[0]) / (2 * np.pi)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(ky_frac, phases / np.pi, "o-", ms=3)
    axes[0].set_xlabel("ky / (BZ)")
    axes[0].set_ylabel("θ / π (principal branch)")
    axes[0].set_title("Raw Wilson Phases")

    axes[1].plot(ky_frac, lifted / np.pi, "o-", ms=3, color="C1")
    axes[1].set_xlabel("ky / (BZ)")
    axes[1].set_ylabel("θ / π (lifted)")
    axes[1].set_title(f"Lifted Phases  (winding = {winding})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
