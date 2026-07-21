# Architecture Notes

## Research premise

Attention began as a mechanism attached to recurrent sequence models and later
became the central organizing operation of the Transformer. These probes ask
whether other supporting mechanisms can undergo the same inversion: remove the
host architecture and make the former helper mechanism carry communication,
state transition, and compute allocation.

This is a design heuristic, not proof that either probe can replace attention
on general language modeling.

## Violation-Driven Constraint Field (VDCF)

### State

For variable `i` at cycle `t`, the state is a categorical logit vector
`z_i^t`. Its normalized belief is:

```text
p_i^t = softmax(z_i^t)
```

A factor `a` connects a small subset of variables and defines a differentiable
violation score `v_a^t`. The score measures disagreement between the current
beliefs and that factor's constraint.

### Conditional transition

At each cycle, select an active set `A_t` containing the highest-scoring
fraction of factors. Only those factors execute the learned correction rule:

```text
z_i^(t+1) = z_i^t - eta * sum_{a in A_t, i in a} g_theta(a, z_a^t, v_a^t)
```

The model therefore treats error as both a learning signal and a routing
signal. A solved region becomes computationally quiet; unresolved regions
receive more work.

### Why this is an architecture rather than a bolt-on optimizer

- The state lives in the field; there is no Transformer or recurrent backbone.
- Factors are the only communication paths.
- Violation ranking determines which computation exists at a cycle.
- One shared local learned rule generates all state changes.
- The same rule can be unrolled for more cycles or applied to larger graphs.

### Complexity and limitations

With `E` factors and active fraction `rho`, correction evaluation is
`O(rho E)` per cycle, plus the cost of measuring or estimating violations.
Consequently, sparse correction alone does not guarantee proportional wall-time
savings: a naive implementation still scores every factor. A scalable version
must maintain approximate priorities incrementally.

The current factor graph is supplied by the task. This is not yet a replacement
for attention when relevant interactions are unknown. A future system needs an
attention-free way to construct or retrieve factors, such as locality,
hash-based routing, learned discrete structure, or a separate global field.

## Quadratic Shared-Basis Field (QSBF)

### Lifted state

Given element `x_i in R^d`, construct the symmetric quadratic lift:

```text
phi(x_i) = vech(x_i x_i^T) in R^m
m = d(d + 1) / 2
```

For a set of `N` elements, form the lifted second moment and its leading
eigenbasis:

```text
S_t = (1/N) * sum_i phi(x_i^t) phi(x_i^t)^T
B_t = leading_eigenvectors(S_t)
```

Although `S_t` is second order in the lifted coordinates, it contains
fourth-order information about the original `x_i`.

### Local read and transition

Each element reads only its relation to the shared basis:

```text
c_i^t = B_t^T phi(x_i^t)             # shared-subspace coordinates
r_i^t = phi(x_i^t) - B_t c_i^t       # residual
x_i^(t+1) = x_i^t + F_theta(x_i^t, c_i^t, r_i^t, lambda_t)
```

No element forms a query-key score with another element. The common basis and
spectrum are a broadcast global state computed from the set as a whole.

### Why this is an architecture rather than covariance pooling

Ordinary covariance pooling usually creates one global descriptor for a final
classifier. Here the global spectral object is read back by every element and
can drive an equivariant state transition. The set-level statistic is therefore
a communication medium, not only an output head.

### Complexity and limitations

The exact implementation uses lifted dimension `m=d(d+1)/2`, with a naive cost
of approximately `O(N m^2 + m^3)` for the moment and eigendecomposition. It is
linear in set size `N` for fixed `m`, but can be worse than attention when `d`
is large. Compact polynomial features, low-rank moments, streaming updates, and
randomized eigensolvers are required before making a scalability claim.

The shared moment is permutation invariant. Tasks that need order or identity
must introduce position or group structure explicitly. Eigenvectors also have
sign and near-degenerate-subspace ambiguities that a production implementation
must handle.

## What is intentionally absent

Neither v2 probe contains:

- Q/K/V projections or softmax attention;
- pairwise token similarity matrices;
- LSTM/GRU recurrence;
- a standard stack of graph-convolution/message-passing layers;
- a Transformer hidden inside the correction function.

The small MLPs are local learned transition functions. Removing all learned
functions was never the hypothesis; removing attention as the global
communication primitive was.

## Long-term combination

The two probes should remain separate until their causal advantages are
established. A later hybrid is natural:

1. QSBF produces a cheap global geometric state.
2. That state proposes or conditions constraint factors.
3. VDCF spends computation only where those factors are violated.

Such a hybrid would combine global broadcasting with selective local
correction, but testing it now would make failures impossible to attribute.
