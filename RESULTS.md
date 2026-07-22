# R-VDCF experiment results

## Setup

- seed: `0`;
- device: CUDA with AMP;
- parameters: `31,256`;
- training: 3,000 steps on 64-node graphs;
- training corruption curriculum: 0% to 15%;
- evaluation: 64, 256, and 1,024 nodes without retraining;
- active factor fraction: 10%;
- modes: reliability-aware sparse, equally budgeted naive sparse, and dense;
- metric below: unknown-node categorical accuracy.

The complete machine-readable output is in
[`results/results.json`](results/results.json).

## Accuracy

| Nodes | Corruption | R-VDCF | Naive sparse | Dense |
|---:|---:|---:|---:|---:|
| 64 | 0% | 1.0000 | 1.0000 | 0.9995 |
| 64 | 5% | 0.9553 | 0.8995 | 0.9604 |
| 64 | 10% | 0.8964 | 0.8192 | 0.9330 |
| 64 | 20% | 0.7828 | 0.6241 | 0.8339 |
| 256 | 0% | 1.0000 | 1.0000 | 1.0000 |
| 256 | 5% | 0.9500 | 0.8867 | 0.9572 |
| 256 | 10% | 0.8856 | 0.7966 | 0.9185 |
| 256 | 20% | 0.7680 | 0.6301 | 0.8035 |
| 1,024 | 0% | 1.0000 | 1.0000 | 1.0000 |
| 1,024 | 5% | 0.9447 | 0.8646 | 0.9631 |
| 1,024 | 10% | 0.8843 | 0.7870 | 0.9161 |
| 1,024 | 20% | 0.7656 | 0.6183 | 0.8102 |

## What is supported

1. Reliability matters under a fixed sparse budget. Across every noisy case,
   R-VDCF outperformed the naive sparse policy.
2. The learned local rule transfers strongly in graph size. Training at 64 and
   testing at 1,024 nodes reduced R-VDCF accuracy by about 1.21 percentage
   points at 10% corruption and 1.72 points at 20% corruption.
3. Sparse execution used roughly 10.0–10.2% of dense learned factor calls,
   approximately a 9.8x reduction.
4. Dense execution still achieved better noisy accuracy, so the current result
   is a compute/quality tradeoff rather than dominance.

## What is not supported yet

- Statistical robustness: this is a single seed.
- A real-world advantage: the task is synthetic and its graph is supplied.
- Wall-clock acceleration: R-VDCF evaluations were usually about 1.5 seconds,
  while dense evaluations were commonly 1.25–1.35 seconds in this run.
- LLM suitability: no tokenizer, corpus, next-token objective, or learned
  language-factor constructor has been tested.

## Next decisive experiments

1. Repeat the full benchmark across at least 5–10 seeds and report mean,
   standard deviation, confidence intervals, and worst seed.
2. Run ablations removing trust, progress features, aging, exploration, and
   influence capping one at a time.
3. Replace global top-k and dense random exploration with a block/bucket event
   queue; benchmark 4K, 16K, and 64K nodes at 5%, 10%, and 20% budgets.
4. Test LDPC decoding against belief propagation, min-sum, neural BP, naive
   sparse execution, and dense R-VDCF under matched compute budgets.
5. Log reliability AUROC, the corrupted-factor selection rate, and trust
   separation to verify that the scheduler learns the intended mechanism.
