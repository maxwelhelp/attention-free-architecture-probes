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

## v3: toward sequence models

Version 3 contains two new, separate falsification experiments:

- **Compact Causal QSBF (CQSBF):** a compact learned quadratic sketch and an
  online fixed-rank shared basis. It is causal, position-aware, uses a fixed
  generation state, and does not construct an `N x N` token matrix. It is
  trained beside ordinary causal attention and causal linear attention on
  associative recall, selective copy, and induction.
- **Reliable VDCF (R-VDCF):** an `O(N)` local/dilated/hash factor topology with
  persistent trust, a learned reliability gate, capped influence, aging,
  exploration, and incremental violation refresh. It compares reliable sparse,
  naive sparse, and dense execution and counts their actual factor work.

These experiments implement the proposed improvements; no v3 result is claimed
until the scripts are run. Design details and interpretation rules are in
[docs/V3_EXPERIMENTS.md](docs/V3_EXPERIMENTS.md).

## Run

Requirements:

- Python 3.10+
- PyTorch 2.x

Install PyTorch using the build appropriate for your CUDA driver. Run the new
CQSBF experiment:

```bash
python experiments/v3/compact_causal_qsbf_experiment.py \
  --models all --device cuda --amp
```

Run the reliable VDCF experiment:

```bash
python experiments/v3/reliable_vdcf_experiment.py \
  --device cuda --amp
```

Quick correctness runs:

```bash
python experiments/v3/compact_causal_qsbf_experiment.py --smoke --device cuda
python experiments/v3/reliable_vdcf_experiment.py --smoke --device cuda
```

The original feasibility studies remain under `experiments/v1/` and
`experiments/v2/`. Generated results and checkpoints are ignored by Git.

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

1. Run the two v3 smoke tests and then their full single-seed suites.
2. Use the v3 measurements to optimize the failed mechanism rather than repeat
   unchanged runs: CQSBF basis/sketch capacity or R-VDCF trust/scheduler policy.
3. Move CQSBF from synthetic language-like tasks to a tiny next-token corpus
   only if it competes with attention on recall, copy, and induction.
4. Replace R-VDCF's global scalar top-k scan with a true heap/bucket queue if
   sparse factor evaluations produce an accuracy/compute advantage.
5. Build the hybrid block only after both improved mechanisms pass separately.

The detailed sequence and pass/fail criteria are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Repository layout

```text
experiments/v1/    Original feasibility prototype
experiments/v2/    Matched-moment and sparse-factor feasibility tests
experiments/v3/    Compact causal QSBF and reliable event-driven VDCF
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
