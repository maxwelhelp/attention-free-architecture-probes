# Version 3 Experiments

Version 3 moves both probes toward sequence-model requirements. The programs
are intentionally separate so that a failure in one mechanism cannot be hidden
by the other.

## Experiment A — Compact Causal QSBF

File: `experiments/v3/compact_causal_qsbf_experiment.py`

### Architectural changes

The exact v2 lift has dimension `d(d+1)/2` and is not practical for an LLM
hidden width. CQSBF first projects a token to compact width `p`, then computes
`r` learned multiplicative features:

```text
phi_j(x) = tanh((a_j^T P x) (b_j^T P x) / sqrt(r))
```

It maintains a causal basis `B_t in R^(r x k)` and spectrum with an online
Oja-style update. Each token reads its coefficients, projected energy, residual
energy, and spectrum. A gated local update broadcasts this read into the token
state. Runtime state per layer is `r*k+k`, independent of prefix length.

Approximate mixer complexity:

```text
time:   O(N r k + N d p + N p r)
state:  O(r k)
token-pair scores: zero
```

The implementation is causal and uses sinusoidal positions. It uses one update
per layer; it does not repeat the unstable v2 refinement loop.

### Tasks and controls

All models receive exactly the same generated batches:

- associative key/value recall;
- selective copy of a marked value by index;
- induction from an earlier `[A,B]` occurrence.

Controls:

- ordinary causal multi-head self-attention;
- causal linear attention using cumulative `K^T V` state.

Every result includes parameters, per-task accuracy, length transfer, wall
time, peak CUDA memory, and theoretical attention score elements.

### Commands

```bash
python experiments/v3/compact_causal_qsbf_experiment.py \
  --smoke --device cuda

python experiments/v3/compact_causal_qsbf_experiment.py \
  --models all --device cuda --amp
```

Outputs: `cqsbf_v3_runs/results.json` and one checkpoint per model.

### Decision rule

- If CQSBF cannot learn the train-length tasks, first improve the causal basis
  update and sketch capacity.
- If it learns at length 64 but fails at 128–512, investigate state rank and
  positional extrapolation.
- If accuracy is competitive but Python recurrence is slow, preserve the
  operator and implement a fused scan kernel.
- A tiny language corpus is justified only after recall, copy, and induction
  work without an `N x N` interaction matrix.

## Experiment B — Reliable Event-Driven VDCF

File: `experiments/v3/reliable_vdcf_experiment.py`

### Architectural changes

R-VDCF uses a supplied `O(N)` topology containing local, dilated, and
deterministic hash factors. It never constructs every possible pair.

Each factor keeps runtime state:

- cached violation;
- persistent trust;
- progress exponential moving average;
- execution count and last-executed cycle.

A learned reliability network reads only these runtime quantities. Synthetic
corruption labels supervise its gate during training but are never given as an
input. Capped influence limits a bad factor, aging prevents starvation, and an
exploration quota tests stale priorities.

After a sparse correction, only factors incident to changed endpoints have
their violation refreshed. The prototype still performs a global scalar top-k
scan; it reports this separately from learned factor-network executions.

### Comparisons

The same trained correction network is evaluated as:

- `reliable_sparse`: trust, learned gate, aging, exploration;
- `naive_sparse`: violation-only top-k;
- `dense`: every factor every cycle.

Evaluation varies sequence size from 64 to 1024 and unseen factor corruption
from 0% to 20%. Reports include reconstruction accuracy, clean-constraint
accuracy, runtime, memory, learned corrections, violation refreshes, scalar
priority scans, and clean/corrupt trust separation.

### Commands

```bash
python experiments/v3/reliable_vdcf_experiment.py \
  --smoke --device cuda

python experiments/v3/reliable_vdcf_experiment.py \
  --device cuda --amp
```

Outputs: `reliable_vdcf_v3_runs/results.json` and
`reliable_vdcf_v3_runs/reliable_vdcf_v3.pt`.

### Decision rule

- Reliable sparse execution must beat naive sparse at 10–20% corruption.
- It must approach dense accuracy with substantially fewer learned factor
  executions.
- A wall-time claim is allowed only if the measured runtime improves; counting
  fewer learned factors alone is insufficient.
- If factor work improves but top-k dominates time, replace the scan with a
  bucketed priority queue rather than changing the learned architecture.

## What v3 still is not

These are not yet pretrained language models. CQSBF is now a causal token-mixer
candidate; R-VDCF is an event-driven sequence-repair candidate. A complete
attention-free LLM block would combine a successful global CQSBF read, sparse
VDCF correction, a local gated FFN, normalization, embeddings, and a language
output head. Building that hybrid before these two tests pass would obscure
which mechanism works.
