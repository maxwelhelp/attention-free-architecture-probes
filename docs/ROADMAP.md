# Experimental Roadmap

The objective is not to collect attractive accuracies. It is to try to falsify
each claimed advantage as cheaply as possible.

## Current implementation status

Version 3 now implements the first optimization pass rather than repeating v2
unchanged:

- CQSBF replaces the full quadratic lift and eigendecomposition with compact
  learned multiplicative features and a causal online low-rank basis. It adds
  attention and linear-attention controls plus language-like memory tasks.
- R-VDCF adds persistent/learned reliability, capped influence, aging,
  exploration, an `O(N)` factor topology, and local cached-priority refresh.
  It measures learned factor executions separately from priority scans.

The immediate gate is running these two programs and improving whichever
specific mechanism fails. Multi-seed confirmation remains necessary before a
paper-level statistical claim, but it is not the next engineering step.

## Stage 0 — reproducibility harness

Deliverables:

- seeds 0–9 for every v2 condition;
- mean, standard deviation, bootstrap 95% intervals;
- wall time after warm-up, peak CUDA memory, and evaluated-factor count;
- frozen JSON schema and automatic plots;
- deterministic smoke tests on CPU.

Pass criterion: current qualitative conclusions survive across seeds. If they
do not, stop architecture expansion and fix the benchmark.

## Direction A — VDCF

### A1. Robust scheduling

Add a persistent factor state containing estimated reliability, last execution
time, and accumulated unresolved violation. Compare:

- pure top violation;
- violation multiplied by learned reliability;
- capped influence plus priority aging;
- small exploration quota for low-priority factors;
- dense execution.

Train with a corruption curriculum from 0% to 15%. A corrupt factor must not be
allowed to consume the active budget indefinitely.

Pass criterion: at 10% corruption, recover at least 0.90 unknown accuracy while
retaining most of the clean sparse-compute advantage.

### A2. Equal-budget baselines

Compare under the same number of local factor evaluations and similar parameter
counts:

- synchronous belief propagation;
- residual belief propagation;
- learned neural BP;
- factor-graph GNN;
- dense VDCF;
- sparse VDCF;
- small Transformer over serialized factors.

Report quality as a function of evaluated factors, FLOPs, and milliseconds.

Pass criterion: VDCF must improve the Pareto frontier, not merely use more
cycles.

### A3. Structural generalization

Train at 24 variables and evaluate at 48, 96, 192, and 384 on:

- chains and grids;
- random sparse graphs;
- graphs with hubs;
- hypergraphs with factor arity greater than two;
- inconsistent anchors and missing factors;
- unseen relation types.

Pass criterion: accuracy and compute grow gracefully without retraining for
each size or topology.

### A4. Real tasks

Start with domains where constraints are native and baselines are clear:

- LDPC decoding;
- graph coloring and small SAT/CSP suites;
- Sudoku-style constraint completion;
- program/data-flow consistency.

The required graph is an honest limitation. Graph induction should be a
separate later experiment, not hidden inside the first benchmark.

## Direction B — QSBF

### B1. Stabilize the operator

Test four variants:

- one-pass readout;
- residual update anchored to the initial input;
- contractive update with a bounded learned step;
- adaptive halting based on basis/subspace change.

Train over a noise curriculum and randomize the number of cycles. Treat
eigenspaces as subspaces rather than sign-sensitive individual eigenvectors.

Pass criterion: later cycles must never reduce mean accuracy by more than one
percentage point across the tested noise range. Otherwise freeze the design as
a one-pass layer.

### B2. Mechanism falsification suite

Generate classes with matched moments through increasing order and vary:

- set size, feature dimension, and latent rank;
- orthogonal and non-orthogonal subspaces;
- mixtures, curves, manifolds, and adversarial near-degenerate spectra;
- outliers and missing elements;
- train/test shifts in size and noise.

Controls:

- independent MLP;
- mean/max Deep Sets;
- raw covariance pooling;
- bilinear/compact bilinear pooling;
- Set Transformer;
- random shared basis;
- shuffled or frozen QSBF basis.

Pass criterion: the advantage must disappear when the shared lifted basis is
destroyed and persist when only nuisance variables change.

### B3. Make the lift scalable

Benchmark:

- low-rank learned quadratic features;
- Tensor Sketch / compact bilinear features;
- block-diagonal moments;
- streaming covariance updates;
- randomized SVD or Oja-style subspace tracking.

Measure complexity against set length `N` and feature dimension `d` separately.

Pass criterion: memory remains approximately linear in `N`, and quality stays
within two points of exact QSBF at a substantially lower total cost.

### B4. Real tasks

Choose tasks where collective geometry should matter:

- point-cloud and set classification;
- set-level and per-element anomaly detection;
- grouped tabular/measurement sets;
- long-sequence synthetic retrieval with explicit positional features.

Language modeling is not the first real benchmark. QSBF currently discards
order unless position is deliberately represented, and its feature-dimension
cost must be solved first.

## Stage 3 — cross-direction hybrid

Attempt a hybrid only if each direction beats a relevant baseline alone:

- QSBF broadcasts a global geometric context;
- context conditions factor creation or reliability;
- VDCF selectively executes the unresolved local corrections.

The hybrid passes only if it improves over both parents at matched compute.

## Decision gates

Continue VDCF if it shows robust conditional-compute savings on at least two
constraint families. Continue QSBF if a compact version retains its matched-
moment advantage and wins on at least one real set task. If neither condition
holds, the experiments remain useful negative results rather than being forced
into a replacement-for-attention narrative.
