#!/usr/bin/env python3
"""
benchmarks/run_matched_present.py
-----------------------------------
Run all four protocols side-by-side and save a comparison figure.
"""

import sys, os, yaml
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arp_topology.lattice import HaldaneModel
from arp_topology.protocols import (ARPProtocol, PrincipalBranchProtocol,
                                     NoTopologyProtocol, FixedRulerProtocol)
from arp_topology.metrics import recovery_time, chern_fidelity

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'recovery_default.yaml')


def _make_model(m):
    return HaldaneModel(t1=m['t1'], t2=m['t2'], phi=m['phi'], M=m['M'])


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    m = cfg['model']
    p = cfg['protocol']
    p['dt'] = cfg['simulation']['dt']
    t_span = tuple(cfg['simulation']['t_span'])

    protocols = {
        'ARP (full)':           ARPProtocol(_make_model(m), dict(p)),
        'Principal Branch':     PrincipalBranchProtocol(_make_model(m), dict(p)),
        'No-Topology Feedback': NoTopologyProtocol(_make_model(m), dict(p)),
        'Fixed Ruler':          FixedRulerProtocol(_make_model(m), dict(p)),
    }

    results = {}
    for name, proto in protocols.items():
        print(f"Running {name}…", flush=True)
        results[name] = proto.run(t_span=t_span)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = ['C0', 'C1', 'C2', 'C3']

    for (name, res), col in zip(results.items(), colors):
        axes[0].plot(res['t'], res['error'], label=name, color=col, lw=1.5)
        fid = chern_fidelity(res['C'], p['C_target'])
        axes[1].plot(res['t'], fid, label=name, color=col, lw=1.5)

    axes[0].axhline(0, color='k', lw=0.8, ls='--')
    axes[0].set_ylabel('ΔC')
    axes[0].set_title('Protocol Comparison — Chern Error')
    axes[0].legend(fontsize=8)

    axes[1].axhline(1, color='k', lw=0.8, ls='--')
    axes[1].set_ylabel('Chern Fidelity')
    axes[1].set_xlabel('Time')
    axes[1].set_title('Chern Fidelity')

    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', cfg['output']['figures'])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'matched_present.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison figure saved: {save_path}")

    # Summary table
    print("\n{:<26} {:>14} {:>14}".format("Protocol", "Recovery Time", "Mean Fidelity"))
    print("-" * 56)
    for name, res in results.items():
        tr = recovery_time(res['t'], res['error'])
        fid = chern_fidelity(res['C'], p['C_target']).mean()
        tr_str = f"{tr:.2f}" if np.isfinite(tr) else "∞"
        print(f"{name:<26} {tr_str:>14} {fid:>14.3f}")


if __name__ == '__main__':
    main()
