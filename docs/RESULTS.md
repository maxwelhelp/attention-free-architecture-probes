# Recorded Results

## Scope

These measurements come from one CUDA run with seed 0. They are feasibility
results, not statistically stable benchmarks. The exact machine-readable
summary is in `results/v2_seed0_summary.json`.

## VDCF

### Clean size transfer

| Evaluation | Variables | Cycles | Active factors | Final unknown accuracy | Clean-edge accuracy |
|---|---:|---:|---:|---:|---:|
| Train size | 24 | 18 | 25% | 0.936481 | 0.916976 |
| 4x size, sparse | 96 | 72 | 25% | 1.000000 | 1.000000 |
| 4x size, dense | 96 | 72 | 100% | 1.000000 | 1.000000 |

On the 96-variable case, sparse execution crossed 99% unknown accuracy around
cycle 41; dense execution crossed it around cycle 31. In factor-evaluation
units this is approximately:

```text
sparse: 41 * 0.25 = 10.25 full-sweep equivalents
dense:  31 * 1.00 = 31.00 full-sweep equivalents
```

This is about 3.0x fewer correction-factor evaluations to the threshold. The
implementation still computes all priority scores, so this must not be reported
as a measured 3.0x wall-time speedup.

### Unseen constraint corruption

Training used no corrupted constraints.

| Corrupt factors | Final unknown accuracy | Clean-edge accuracy |
|---:|---:|---:|
| 2% | 0.958635 | 0.945639 |
| 5% | 0.874342 | 0.850892 |
| 10% | 0.755140 | 0.724993 |

Diagnosis: a corrupt factor remains highly violated, so a pure top-violation
scheduler repeatedly allocates budget to an unsatisfiable constraint. The next
version needs factor reliability, priority aging, or a robust capped influence
function.

## QSBF

### Controls and scale

| Evaluation | Elements | Noise | QSBF | Independent | Raw linear basis | Raw covariance gap | Residual ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train size | 64 | 0.005 | 0.999974 | 0.501302 | 0.558073 | 5.74e-4 | 31.03 |
| 2x size | 128 | 0.005 | 0.999987 | 0.499818 | 0.564974 | 4.14e-4 | 47.84 |
| 2x size, exact | 128 | 0.000 | 0.999987 | 0.500586 | 0.496315 | 1.36e-8 | 48.09 |
| 2x size | 128 | 0.020 | 0.999818 | 0.503867 | 0.559935 | — | 42.29 |

The zero-noise row is the strongest mechanism check. Both controls are at
chance and raw class covariances are numerically matched, while the lifted
shared basis separates the classes almost perfectly. This shows that the
quadratic field accesses useful collective high-order structure; it does not
yet show superiority on a real task.

### Iterative stability under noise

| Noise | Accuracy after cycles 1 / 2 / 3 / 4 |
|---:|---|
| 0.050 | 0.997357 / 0.951146 / 0.972891 / 0.971224 |
| 0.100 | 0.967227 / 0.719206 / 0.709141 / 0.699492 |

The first QSBF read is robust, but the learned repeated update is not. On clean
data later cycles add only a small gain; on unseen noise they cause a large
loss. The current evidence therefore supports a one-pass quadratic shared-basis
operator more strongly than the iterative version.

## What the experiments do not establish

- No variance or confidence interval has been measured.
- No equal-FLOP comparison to strong neural baselines has been run.
- No wall-time or peak-memory advantage has been measured.
- The factor graph is given rather than learned.
- The QSBF lift is too expensive for large hidden dimensions.
- Neither probe has been tested on language modeling or another real sequence
  task.
