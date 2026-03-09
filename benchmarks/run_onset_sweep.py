#!/usr/bin/env python3
"""
benchmarks/run_onset_sweep.py
------------------------------
Sweep initial M values; plot onset M vs ARP recovery time.
"""

import sys, os, yaml
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arp_topology.lattice import HaldaneModel
from arp_topology.protocols import ARPProtocol
from arp_topology.metrics import recovery_time

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'onset_sweep.yaml')


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    m_base = cfg['model']
    p = cfg['protocol']
    p['dt'] = cfg['simulation']['dt']
    t_span = tuple(cfg['simulation']['t_span'])
    M_values = cfg['sweep']['values']

    rec_times = []
    for M_onset in M_values:
        model = HaldaneModel(t1=m_base['t1'], t2=m_base['t2'],
                             phi=m_base['phi'], M=float(M_onset))
        proto = ARPProtocol(model=model, config=dict(p))
        result = proto.run(t_span=t_span)
        tr = recovery_time(result['t'], result['error'])
        rec_times.append(tr)
        print(f"  M_onset={M_onset:.2f}  recovery_time={tr:.2f}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', cfg['output']['figures'])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'onset_sweep.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    finite_mask = np.isfinite(rec_times)
    ax.plot(np.array(M_values)[finite_mask],
            np.array(rec_times)[finite_mask], 'o-', color='C0')
    ax.set_xlabel("Onset M")
    ax.set_ylabel("Recovery time")
    ax.set_title("ARP Recovery Time vs. Onset M")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")


if __name__ == '__main__':
    main()
