# R-VDCF architecture

## Scope

R-VDCF performs iterative inference over a supplied sparse factor graph. Let
variable `i` have categorical logits `z_i^t` and probabilities

```text
p_i^t = softmax(z_i^t).
```

Each factor `a = (i, j, r_a)` states a modular relation between two variables.
The experiment uses local, dilated, and deterministic hash edges, so the number
of factors grows linearly with the number of variables.

## One inference iteration

For each factor, the system computes a cheap violation score `v_a^t` from the
current endpoint distributions and the required relation. It also maintains
persistent runtime state: cached violation, trust, previous progress, and age.

A learned reliability gate estimates whether executing a factor is likely to
produce useful progress. The scheduler combines approximately:

```text
priority_a = violation_a * trust_a + aging_bonus_a.
```

Most of the active budget goes to the highest-priority factors; a small quota
is sampled for exploration so an initially underestimated factor can recover.
For the active set `A_t`, a shared local network predicts factor messages, and
their capped contributions are aggregated at incident nodes:

```text
z_i^(t+1) = z_i^t - eta * sum_{a in A_t, i in a} g_theta(a, z_a^t).
```

Observed anchors are reimposed, affected violations are refreshed, and factor
trust is updated from whether the correction reduced the violation. This
state-update loop is the complete inference architecture in the experiment.

## Reliability additions

The reliable variant differs from the naive sparse scheduler through:

- persistent learned trust based on runtime factor features;
- capped factor influence, limiting damage from a misleading constraint;
- priority aging, preventing permanent starvation;
- explicit exploration;
- local cached-violation refresh after selected events.

The comparison uses the same trained correction rule for reliable sparse,
naive sparse, and dense execution, isolating the scheduler policy as far as the
prototype permits.

## Computational profile

With `E = O(N)` sparse factors, state storage and topology are linear. If a
fraction `rho` is active, expensive learned factor work is `O(rho E)` per
iteration instead of `O(E)` in dense execution. There is no `N x N` attention
matrix and no quadratic memory term.

The present implementation is not sublinear end to end. It performs an `O(E)`
scalar priority scan/global top-k, creates exploration state across factors,
and uses non-fused gather/MLP/scatter operations. Consequently, fewer learned
evaluations do not yet translate into lower wall time.

The next systems version should use bucketed or block-local queues, `O(k)`
rotating exploration, and fused factor execution. Only then can practical
scaling be judged.

## Relationship to attention

R-VDCF and attention solve different problems in their current forms:

| Property | R-VDCF prototype | Self-attention |
|---|---|---|
| Interaction graph | supplied sparse factors | learned content-dependent dense/sparse retrieval |
| Main operation | selected constraint correction | weighted value aggregation |
| Memory in this implementation | `O(N + E)` | commonly `O(N^2)` for full attention |
| Natural domain | explicit structured constraints | unstructured token sequences |

Thus R-VDCF can replace dense message passing or iterative inference on an
explicit factor graph. It cannot yet replace attention in an LLM because the
critical problem of discovering language interactions has not been solved.

## Novelty boundary

The architectural hypothesis is that a reliability-aware,
violation-driven scheduler can become the primary computation, with learned
local correction executed only on events. The primitives themselves overlap
with factor graphs, residual belief propagation, neural message passing,
conditional computation, and prioritized sweeping. This repository claims an
experimental composition and inversion of emphasis, not established priority.
