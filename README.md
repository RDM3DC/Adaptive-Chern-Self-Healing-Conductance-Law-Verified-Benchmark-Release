# arp_topology — Adaptive Chern Self-Healing Conductance Law

![Python ≥ 3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A complete, scientifically rigorous Python package implementing the
**Adaptive Recovery Protocol (ARP)** for topological self-healing of
Hall conductance in 2-D lattice systems.

This repository now includes a checked-in benchmark artifact bundle under
`outputs/` with a manifest, hashed artifacts, solver-verification results,
ablation summaries, onset sweeps, and rendered benchmark figures.

---

## Installation

```bash
git clone <repo-url>
cd Adaptive-Chern-Self-Healing-Conductance-Law-Verified-Benchmark-Release
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from arp_topology import HaldaneModel, chern_number, ARPProtocol

# Build a model outside the topological phase
model = HaldaneModel(t1=1.0, t2=0.3, phi=np.pi/2, M=2.5)
print("Initial Chern number:", chern_number(model, n_k=30))  # 0

# Run ARP to recover C=+1
proto = ARPProtocol(model, config={"C_target": 1, "gain": 2.0,
                                    "param": "M", "epsilon": 0.01,
                                    "n_k": 20, "dt": 0.1})
result = proto.run(t_span=(0.0, 20.0))
print("Final Chern number:", int(result["C"][-1]))  # 1
```

---

## Verified Output Bundle

The repository includes a recorded benchmark bundle at `outputs/` with the
following top-level artifacts:

```text
outputs/
  manifest.json
  matched_present/
  onset_map/
  recovery_demo/
  solver_verification/
```

The current manifest records:

- `overall = PASS`
- `artifact_count = 16`
- `git_hash = edc34448fea5c0f6dae28cd11cd5dde5d4941da2`
- `timestamp = 2026-03-09T02:09:12.695459+00:00`

### Solver Verification

The attached solver-verification bundle at `outputs/solver_verification/verdict.json`
passes all 5 checks:

- Determinism
- Current conservation
- Phase-lift clipping
- Solver cross-validation
- Lattice symmetry

Notable recorded values:

- `tests_passed = 5 / 5`
- `max_ratio_error = 0.0`
- `coherence_range = [0.9999368491814438, 1.0000000000000002]`
- `gmres_failures = 0`
- `flux_Yeff_recovery_ratio = 44.21463450561277`
- `row_symmetry_max_error = 0.004666194007826474`

### Parameter Sweep Coverage

The onset-map bundle at `outputs/onset_map/summary.json` records two grid sweeps:

- `alpha0_vs_lambda_s`: `recovery_fraction = 0.4375`, `bf_range = [0.4117083925029834, 0.9415103910783635]`
- `chi_vs_damage_scale`: `recovery_fraction = 0.59375`, `bf_range = [0.4232037621914788, 0.9909974792942029]`

### Included Benchmark Artifacts

The attached bundle includes:

- `outputs/recovery_demo/recovery_traces.csv`
- `outputs/recovery_demo/recovery_traces.png`
- `outputs/recovery_demo/summary.json`
- `outputs/recovery_demo/snapshot_healthy.png`
- `outputs/recovery_demo/snapshot_damaged.png`
- `outputs/recovery_demo/snapshot_recovered.png`
- `outputs/matched_present/matched_present_summary.csv`
- `outputs/matched_present/matched_present_summary.json`
- `outputs/matched_present/matched_present_traces.png`
- `outputs/onset_map/onset_alpha0_lambda_s.csv`
- `outputs/onset_map/onset_alpha0_lambda_s.png`
- `outputs/onset_map/onset_chi_damage_scale.csv`
- `outputs/onset_map/onset_chi_damage_scale.png`
- `outputs/onset_map/summary.json`
- `outputs/solver_verification/verdict.json`

---

## Physics Background

### Haldane Model
The Haldane model is a 2-band tight-binding model on the honeycomb lattice
with broken time-reversal symmetry (complex NNN hopping `t2 e^{iφ}`) and
broken inversion symmetry (staggered on-site energy `M`).  The Bloch
Hamiltonian reads `H(k) = d₀I + dₓσₓ + dyσy + dzσz`.

### Chern Number
The topological invariant is computed on a discrete k-space grid using the
**Fukui–Hatsugai–Suzuki (FHS)** lattice gauge method, which is robust
against gauge ambiguities.  The topological phase has `|C| = 1` when
`|M| < 3√3 |t₂ sin φ|`.

### ARP Feedback Law
The Adaptive Recovery Protocol steers a control parameter θ (e.g. M or φ)
back toward the topological phase via the ODE:

```
dθ/dt = –K · ΔC / (|ΔC| + ε) · sign(∂C/∂θ)
```

where `ΔC = C(t) – C*`.  As `C → C*`, the Hall conductance
`G = (e²/h)·C` heals toward its quantised value.

---

## Project Structure

```
src/arp_topology/
  lattice.py       — HaldaneModel: Bloch Hamiltonian, phase boundary
  topology.py      — chern_number, berry_curvature (FHS method)
  phase_lift.py    — wilson_loop_spectrum, lift_wilson_phases
  laws.py          — ARPLaw ODE with feedback gain and gradient sign
  solver.py        — RK4Solver (fixed step) and RK45Solver (Dormand–Prince)
  metrics.py       — recovery_time, chern_fidelity, conductance_deviation
  protocols.py     — ARPProtocol, PrincipalBranchProtocol,
                     NoTopologyProtocol, FixedRulerProtocol
  plotting.py      — Matplotlib figures for phase diagram, recovery, etc.
benchmarks/        — Standalone demo and ablation scripts
configs/           — YAML configuration files
tests/             — pytest test suite
outputs/           — Figures, tables, logs (auto-created)
```

For the attached benchmark release bundle, the concrete checked-in output
structure is:

```text
outputs/
  figures/
  logs/
  tables/
  videos/
  manifest.json
  matched_present/
  onset_map/
  recovery_demo/
  solver_verification/
```

---

## Benchmarks

| Script | Protocol | Description |
|--------|----------|-------------|
| `run_recovery_demo.py` | ARP (full) | Main recovery demonstration |
| `run_principal_branch_control.py` | Principal Branch | Ablation: phase-only sensing |
| `run_no_topology_feedback.py` | No-Topology | Ablation: amplitude-only feedback |
| `run_fixed_ruler_control.py` | Fixed Ruler | Ablation: open-loop boundary pinning |
| `run_solver_checks.py` | — | RK4 vs RK45 consistency check |
| `run_onset_sweep.py` | ARP | Recovery time vs. onset M sweep |
| `run_matched_present.py` | All four | Side-by-side comparison figure |

Run any benchmark:
```bash
python benchmarks/run_recovery_demo.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

To inspect the checked-in release artifacts directly, start with:

```bash
python -c "import json, pathlib; print(json.loads(pathlib.Path('outputs/manifest.json').read_text())['overall'])"
```

---

## Citation

If you use this package in academic work, please cite:

> McKenna, R. (2026). *Adaptive Chern Self-Healing Conductance Law —
> Verified Benchmark Release*. GitHub.
> `https://github.com/<owner>/<repo>`