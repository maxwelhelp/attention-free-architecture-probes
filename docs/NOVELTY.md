# Novelty and Prior-Art Boundary

## Short answer

The implementations and the exact combinations tested here are original work
from this project. That does **not** yet establish that the architectures are
globally unprecedented.

The scientifically defensible wording is:

> We propose and test two attention-free architecture candidates assembled by
> elevating correction/scheduling and high-order pooling mechanisms into the
> primary communication rule.

Do not yet write “the first”, “entirely novel”, or “replaces Transformers”.

## VDCF boundary

Known nearby ideas include:

- Residual Belief Propagation schedules asynchronous messages using residuals
  to reach a fixed point faster and more reliably.
- Neural Enhanced Belief Propagation uses a factor-graph neural network to
  correct belief-propagation messages.

Therefore neither priority-by-residual nor learned correction on a factor graph
is new by itself.

The narrower candidate contribution is their architectural inversion:

- a categorical neural field is the primary state;
- current violation directly schedules a strict compute budget;
- the learned factor correction is the only state-transition operator;
- sparse execution, longer unrolling, and size transfer are designed together;
- the target is a general neural architecture, not only faster scheduling of a
  fixed classical inference algorithm.

This needs direct ablations against residual BP, synchronous BP, neural BP,
factor-graph GNNs, and an equal-budget Transformer before the boundary is clear.

Primary references:

- Gal Elidan, Ian McGraw, Daphne Koller, [Residual Belief Propagation: Informed
  Scheduling for Asynchronous Message Passing](https://arxiv.org/abs/1206.6837)
- Victor Garcia Satorras, Max Welling, [Neural Enhanced Belief Propagation on
  Factor Graphs](https://arxiv.org/abs/2003.01998)

## QSBF boundary

Known nearby ideas include:

- Deep Sets aggregate elements into a permutation-invariant global summary.
- Bilinear CNNs pool outer products to capture feature interactions.
- Global covariance pooling and matrix normalization use second-order matrices
  and sometimes eigendecomposition or matrix iterations.
- Set Transformer explicitly models element interactions with attention and
  provides an important accuracy/scaling baseline.

Thus outer-product lifting, global moments, eigenvectors, and set aggregation
are known primitives.

The narrower candidate contribution is using the spectral basis of a lifted
global moment as an exclusive, repeatedly readable communication field:

- fourth-order information in the original coordinates is exposed through a
  quadratic lift followed by a second moment;
- each element reads its own projection and residual rather than another
  element's representation;
- the global statistic feeds an equivariant per-element update instead of only
  a final pooled classifier;
- matched-moment tests deliberately remove signals available to independent
  and raw-covariance baselines.

Of the two probes, QSBF currently has the stronger case for a distinct
architecture-level contribution. It also has the larger unresolved scaling
problem because the exact lifted moment grows rapidly with feature dimension.

Primary references:

- Manzil Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114)
- Tsung-Yu Lin et al., [Bilinear CNNs for Fine-grained Visual
  Recognition](https://arxiv.org/abs/1504.07889)
- Yang Gao et al., [Compact Bilinear Pooling](https://arxiv.org/abs/1511.06062)
- Peihua Li et al., [Towards Faster Training of Global Covariance Pooling
  Networks by Iterative Matrix Square Root
  Normalization](https://arxiv.org/abs/1712.01034)
- Juho Lee et al., [Set Transformer](https://arxiv.org/abs/1810.00825)

## Evidence ladder

The project is currently at level 2 of this ladder:

1. **Definition:** the update rule is executable and does not hide attention.
2. **Synthetic mechanism test:** the intended signal is necessary and the
   architecture can use it.
3. **Controlled comparison:** multiple seeds, equal compute, strong baselines,
   and adversarial ablations.
4. **Scaling test:** advantage persists as sequence/set/graph size grows.
5. **Real benchmark:** value survives realistic noise and representation
   learning.
6. **General architecture claim:** several task families show a consistent
   quality/compute advantage.

The current experiments justify continued research, not level 6 language.
