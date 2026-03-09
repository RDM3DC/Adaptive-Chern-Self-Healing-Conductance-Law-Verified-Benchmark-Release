#!/usr/bin/env python3
"""
benchmarks/run_fixed_ruler_control.py
---------------------------------------
Ablation study: FixedRulerProtocol (M clamped to phase boundary).
"""

import sys, os, yaml, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arp_topology.lattice import HaldaneModel
from arp_topology.protocols import FixedRulerProtocol
from arp_topology.metrics import recovery_time, chern_fidelity
from arp_topology.plotting import plot_recovery

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'ablation_fixed_ruler.yaml')


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    m = cfg['model']
    model = HaldaneModel(t1=m['t1'], t2=m['t2'], phi=m['phi'], M=m['M'])

    p = cfg['protocol']
    p['dt'] = cfg['simulation']['dt']
    proto = FixedRulerProtocol(model=model, config=p)

    t_span = tuple(cfg['simulation']['t_span'])
    print(f"Running FixedRulerProtocol  t∈{t_span}  M₀={m['M']}")
    result = proto.run(t_span=t_span)

    tr = recovery_time(result['t'], result['error'])
    fid = chern_fidelity(result['C'], p['C_target'])
    print(f"  Recovery time : {tr:.2f}")
    print(f"  Mean fidelity : {fid.mean():.3f}")
    print(f"  Fixed M       : {result['M'][0]:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', cfg['output']['figures'])
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'fixed_ruler.png')
    plot_recovery(result, protocol_name='Fixed Ruler', save_path=save_path)
    print(f"  Figure saved  : {save_path}")


if __name__ == '__main__':
    main()
