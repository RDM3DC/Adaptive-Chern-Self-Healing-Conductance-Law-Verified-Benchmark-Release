#!/usr/bin/env python3
"""
benchmarks/run_no_topology_feedback.py
----------------------------------------
Ablation study: NoTopologyProtocol (amplitude-only feedback).
"""

import sys, os, yaml, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arp_topology.lattice import HaldaneModel
from arp_topology.protocols import NoTopologyProtocol
from arp_topology.metrics import recovery_time, chern_fidelity
from arp_topology.plotting import plot_recovery

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'ablation_no_topology.yaml')


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    m = cfg['model']
    model = HaldaneModel(t1=m['t1'], t2=m['t2'], phi=m['phi'], M=m['M'])

    p = cfg['protocol']
    p['dt'] = cfg['simulation']['dt']
    proto = NoTopologyProtocol(model=model, config=p)

    t_span = tuple(cfg['simulation']['t_span'])
    print(f"Running NoTopologyProtocol  t∈{t_span}  M₀={m['M']}")
    result = proto.run(t_span=t_span)

    tr = recovery_time(result['t'], result['error'])
    fid = chern_fidelity(result['C'], p['C_target'])
    print(f"  Recovery time : {tr:.2f}  (may be inf for ablation)")
    print(f"  Mean fidelity : {fid.mean():.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', cfg['output']['figures'])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'no_topology_feedback.png')
    plot_recovery(result, protocol_name='No-Topology Feedback', save_path=save_path)
    print(f"  Figure saved  : {save_path}")


if __name__ == '__main__':
    main()
