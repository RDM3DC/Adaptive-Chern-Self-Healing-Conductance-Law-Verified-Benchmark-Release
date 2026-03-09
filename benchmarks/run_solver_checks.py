#!/usr/bin/env python3
"""
benchmarks/run_solver_checks.py
---------------------------------
Compare RK4 vs RK45 on the ARP ODE; print max deviation and timing.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arp_topology.lattice import HaldaneModel
from arp_topology.laws import ARPLaw
from arp_topology.solver import RK4Solver, RK45Solver


def main():
    model_rk4 = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=2.5)
    model_rk45 = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi / 2, M=2.5)

    law_rk4 = ARPLaw(model=model_rk4, C_target=1, gain=2.0,
                     param='M', epsilon=0.01, n_k=20)
    law_rk45 = ARPLaw(model=model_rk45, C_target=1, gain=2.0,
                      param='M', epsilon=0.01, n_k=20)

    state0 = np.array([2.5])
    t_span = (0.0, 5.0)

    t0 = time.time()
    t4, s4 = RK4Solver(rhs=law_rk4.rhs, dt=0.05).integrate(state0, t_span)
    dt4 = time.time() - t0

    t0 = time.time()
    t45, s45 = RK45Solver(rhs=law_rk45.rhs, rtol=1e-4, atol=1e-6).integrate(state0, t_span)
    dt45 = time.time() - t0

    # Evaluate both at t_span[1]
    x4 = s4[-1, 0]
    x45 = s45[-1, 0]
    print(f"RK4  final M = {x4:.6f}  ({len(t4)} steps, {dt4:.2f}s)")
    print(f"RK45 final M = {x45:.6f}  ({len(t45)} steps, {dt45:.2f}s)")
    print(f"Max deviation: {abs(x4 - x45):.2e}")


if __name__ == '__main__':
    main()
