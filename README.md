# R-VDCF

**Reliable Event-Driven Violation-Driven Constraint Field** is an experimental
architecture for iterative inference on sparse factor graphs. Instead of
updating every relation at every step, it measures constraint violations,
estimates factor reliability, and executes learned local corrections only for
the most useful factors.

[Русская версия](README_RU.md) · [Architecture](ARCHITECTURE.md) ·
[Results](RESULTS.md) · [Raw JSON](results/results.json)

## What this repository demonstrates

R-VDCF is a standalone inference architecture when the problem is already
represented as variables and sparse constraints. It is not merely an auxiliary
loss attached to a Transformer: in this experiment, the factor scheduler and
local correction rule are the complete communication mechanism.

It is also **not yet a general replacement for attention or a Transformer**.
Attention discovers content-dependent interactions between tokens; R-VDCF
currently assumes that the factor graph is supplied. Constructing useful
language factors without attention remains an open problem.

At each iteration R-VDCF:

1. measures the current violation of each factor;
2. estimates persistent trust from runtime behavior;
3. selects a small active set using violation, trust, aging, and exploration;
4. applies a capped learned correction to the incident variables;
5. refreshes affected local state and repeats.

## Current result

One model was trained only on graphs with 64 nodes and evaluated without
retraining on 64, 256, and 1,024 nodes. With 10% corrupted constraints:

| Nodes | R-VDCF | Naive sparse | Dense |
|---:|---:|---:|---:|
| 64 | **0.8964** | 0.8192 | 0.9330 |
| 256 | **0.8856** | 0.7966 | 0.9185 |
| 1,024 | **0.8843** | 0.7870 | 0.9161 |

R-VDCF used about **10.2% of the learned factor evaluations** required by the
dense mode and consistently beat an equally sparse scheduler without learned
reliability. The dense mode retained higher accuracy under noise.

This prototype does not yet produce a wall-clock speedup: the scalar top-k
scan is still `O(E)`, and gather/scatter plus Python/kernel overhead outweigh
the saved small MLP calls at this scale. The demonstrated result is selective
computation and length transfer, not optimized runtime.

## Run

Requirements: Python 3.10+ and PyTorch 2.1+.

```bash
python -m pip install -r requirements.txt
python rvdcf_experiment.py --smoke --device cuda
python rvdcf_experiment.py --device cuda --amp
```

The full run trains for 3,000 steps and writes `results.json` plus `rvdcf.pt`
to `rvdcf_runs/`. Use `--output-dir`, `--steps`, `--seed`, `--batch-size`, or
`--eval-nodes 64,256,1024` to override the defaults.

## Where it may be useful

- error-correcting codes and noisy factor graphs;
- SAT/CSP, graph coloring, scheduling, and structured repair;
- probabilistic graphical models and sensor fusion;
- program analysis, type constraints, and dataflow propagation.

It is a poor fit when the interactions are unknown, mostly dense, or require
content-addressed retrieval—the jobs attention currently handles well.

## Status and novelty boundary

This is research prototype code with one synthetic task and one seed. The
individual ingredients—factor graphs, residual scheduling, learned message
updates, gating, and exploration—have related prior work. The proposal being
tested is their architectural combination: make reliability-aware,
violation-driven sparse execution the central computation rather than an
optimization around a dense network. A broad priority or “first” claim is not
made here.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact mechanism and
[RESULTS.md](RESULTS.md) for all reported measurements and the next tests.
