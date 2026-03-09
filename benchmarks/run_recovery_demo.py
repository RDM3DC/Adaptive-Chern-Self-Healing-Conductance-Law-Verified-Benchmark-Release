#!/usr/bin/env python3
"""
benchmarks/run_recovery_demo.py
--------------------------------
Runs the full ARP protocol using recovery_default.yaml, saves a recovery
curve figure to outputs/figures/.
"""

import sys
import os
import yaml
import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arp_topology.lattice import HaldaneModel
from arp_topology.protocols import ARPProtocol
from arp_topology.metrics import recovery_time, chern_fidelity
from arp_topology.plotting import plot_recovery

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'recovery_default.yaml')


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    m = cfg['model']
    model = HaldaneModel(t1=m['t1'], t2=m['t2'], phi=m['phi'], M=m['M'])

    p = cfg['protocol']
    p['dt'] = cfg['simulation']['dt']
    proto = ARPProtocol(model=model, config=p)

    t_span = tuple(cfg['simulation']['t_span'])
    print(f"Running ARPProtocol  t∈{t_span}  dt={p['dt']}  M₀={m['M']}")
    result = proto.run(t_span=t_span)

    tr = recovery_time(result['t'], result['error'])
    fid = chern_fidelity(result['C'], p['C_target'])
    print(f"  Recovery time : {tr:.2f}")
    print(f"  Mean fidelity : {fid.mean():.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', cfg['output']['figures'])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'recovery_demo.png')
    plot_recovery(result, protocol_name='ARP', save_path=save_path)
    print(f"  Figure saved  : {save_path}")

    log_dir = os.path.join(os.path.dirname(__file__), '..', cfg['output']['logs'])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'recovery_demo.txt')
    with open(log_path, 'w') as f:
        f.write(f"recovery_time={tr:.4f}\nmean_fidelity={fid.mean():.4f}\n")
    print(f"  Log saved     : {log_path}")


if __name__ == '__main__':
    main()
