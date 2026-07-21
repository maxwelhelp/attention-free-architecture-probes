# Attention-Free Architecture Probes

Two executable research probes for global information exchange without a
Transformer block, self-attention, recurrence, or a conventional stack of GNN
message-passing layers.

This repository tests a specific research question:

> Can a mechanism currently used as a correction, scheduler, or pooling
> operation become the organizing principle of an entire architecture?

The answer is not established by these small synthetic experiments. The
results do show that both proposed mechanisms are computationally real and
worth testing under stronger baselines.

## The two probes

### 1. Violation-Driven Constraint Field (VDCF)

Each variable stores a categorical logit state. Constraint factors measure
their current violation, but only the most violated fraction is executed at
each cycle. A shared learned local rule converts selected violations into state
corrections.

```text
state -> measure factor violations -> select top-k -> local corrections -> state
```

There is no all-pairs token interaction. Communication occurs only through an
explicit factor field, and computation is conditional on the current error.

### 2. Quadratic Shared-Basis Field (QSBF)

Each element is lifted with a symmetric outer product. The set builds a shared
second-moment matrix in the lifted space, extracts a spectral basis, and lets
each element update itself from its own projection and residual relative to
that basis.

```text
x_i -> phi(x_i)=vech(x_i x_i^T)
     -> global lifted moment -> eigenspace B
     -> local projection/residual -> local update
```

Elements never query or attend to other individual elements. Their common
communication channel is a global high-order geometric object.

## What v2 currently demonstrates

| Probe | Strongest observed result | Important failure |
|---|---:|---|
| VDCF | 96 variables: 1.000 final unknown accuracy using only 25% of factors per cycle | Corrupt high-violation factors can monopolize the compute budget |
| QSBF | 128 elements, zero noise: 0.999987 accuracy while an independent model scores 0.500586 and a raw linear basis scores 0.496315 | Repeated updates degrade sharply under unseen high noise |

For VDCF, reaching 99% accuracy required about 10.25 full-factor-sweep
equivalents with sparse execution versus 31 sweeps for dense execution in the
recorded run. This is roughly a 3x reduction in factor evaluations, not yet a
3x end-to-end speed claim.

For QSBF, the zero-noise matched-moment control is the central result: the raw
class covariance gap was approximately `1.36e-8`, so a linear shared basis and
an independent element classifier were at chance, while the quadratic shared
basis was almost perfect.

Full measurements and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

## Run

Requirements:

- Python 3.10+
- PyTorch 2.x

Install PyTorch using the build appropriate for your CUDA driver, then run:

```bash
python experiments/v2/two_architecture_experiments_v2.py \
  --experiment all --device cuda
```

A quick correctness run:

```bash
python experiments/v2/two_architecture_experiments_v2.py \
  --experiment all --device cpu --smoke
```

Outputs are written to `two_architecture_runs_v2/` by default. Use
`--output-dir` to change this. The first prototype is retained under
`experiments/v1/` for comparison; v2 is the current implementation.

## Research status and novelty

These are working architecture names and original experimental compositions,
not a claim that every primitive is new.

- VDCF overlaps with residual scheduling in belief propagation and with neural
  correction of factor-graph messages. The candidate contribution is the
  complete combination: violation is the compute scheduler, the learned local
  correction is the main state-transition rule, and sparse execution is the
  architecture rather than an optimization around a classical solver.
- QSBF overlaps with bilinear features, covariance pooling, spectral methods,
  and permutation-invariant set models. Its candidate contribution is using
  the eigenspace of a lifted global statistic as the exclusive communication
  field for per-element updates, including tasks deliberately invisible to
  first- and second-order baselines in the original coordinates.

Before using words such as *novel* or *first*, this needs a systematic
literature review and direct experimental comparisons. See
[docs/NOVELTY.md](docs/NOVELTY.md).

## Next milestones

1. Reproduce every result over at least 10 seeds with confidence intervals.
2. Benchmark equal compute budgets, wall time, memory, and scaling rather than
   accuracy alone.
3. Fix VDCF's corrupt-factor starvation with learned reliability and aging.
4. Turn QSBF into a stable one-pass or adaptively halted operator and replace
   the full quadratic lift with a compact approximation.
5. Compare against BP/RBP/GNN/Transformer for VDCF and Deep Sets/covariance
   pooling/Set Transformer for QSBF.
6. Only after synthetic falsification tests, move to real constraint and set
   benchmarks.

The detailed sequence and pass/fail criteria are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Repository layout

```text
experiments/v1/    Original feasibility prototype
experiments/v2/    Current architecture implementations and evaluations
docs/              Architecture, novelty, results, and roadmap notes
results/           Machine-readable recorded summary
```

## Reproducibility note

The checked-in result summary records one CUDA run with seed 0. It is evidence
of feasibility, not a benchmark result. Model checkpoints and generated output
directories are intentionally ignored.

## License

No license has been selected yet. Until one is added, normal copyright rules
apply; the source is visible for review but no broad reuse permission is
granted.
