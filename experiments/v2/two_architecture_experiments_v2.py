#!/usr/bin/env python3
"""
Attention-free architecture probes, version 2.

1) Violation-Driven Constraint Field (VDCF)
   Variables are class-logit fields.  Learned local factors propose corrections,
   but only constraints with the largest current violation are executed.  There
   is no layer stack, recurrent cell, self-attention, or all-to-all token mixer.

2) Quadratic Shared-Basis Field (QSBF)
   Every item is lifted with phi(x)=vech(xx^T).  Items communicate only through
   the eigenspace of the shared lifted second moment (a fourth-order statistic
   of x).  The benchmark makes raw means and covariances equal by construction,
   so ordinary PCA and element-wise networks cannot solve it.

Requirements: Python 3.10+ and PyTorch 2.x.

Examples:
    python two_architecture_experiments_v2.py --smoke --device cuda
    python two_architecture_experiments_v2.py --experiment all --device cuda
    python two_architecture_experiments_v2.py --experiment factor --device cuda
    python two_architecture_experiments_v2.py --experiment basis --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(1.0)


def gather_nodes(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def scatter_edges_to_nodes(
    messages: torch.Tensor,
    index: torch.Tensor,
    nodes: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, edges, width = messages.shape
    offsets = torch.arange(batch, device=messages.device).unsqueeze(1) * nodes
    flat_index = (index + offsets).reshape(-1)
    summed = torch.zeros(batch * nodes, width, device=messages.device, dtype=messages.dtype)
    summed.index_add_(0, flat_index, messages.reshape(-1, width))
    counts = torch.zeros(batch * nodes, 1, device=messages.device, dtype=messages.dtype)
    if weights is None:
        count_values = torch.ones(
            batch * edges, 1, device=messages.device, dtype=messages.dtype
        )
    else:
        count_values = weights.reshape(batch * edges, 1).to(messages.dtype)
    counts.index_add_(0, flat_index, count_values)
    return summed.view(batch, nodes, width), counts.view(batch, nodes, 1)


def shift_distribution(probability: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """If v=u+delta mod K, return the distribution over v implied by p(u)."""
    classes = probability.shape[-1]
    target = torch.arange(classes, device=probability.device).view(1, 1, classes)
    source = (target - delta.unsqueeze(-1)) % classes
    return probability.gather(-1, source.expand_as(probability))


def inverse_shift_distribution(probability: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """If v=u+delta mod K, return the distribution over u implied by p(v)."""
    classes = probability.shape[-1]
    target = torch.arange(classes, device=probability.device).view(1, 1, classes)
    source = (target + delta.unsqueeze(-1)) % classes
    return probability.gather(-1, source.expand_as(probability))


# -----------------------------------------------------------------------------
# Experiment 1: Violation-Driven Constraint Field
# -----------------------------------------------------------------------------


@dataclass
class FactorConfig:
    classes: int = 7
    train_nodes: int = 24
    test_nodes: int = 96
    extra_edges_per_node: float = 1.5
    hidden: int = 128
    active_fraction: float = 0.25
    train_iterations: int = 18
    test_iterations: int = 72
    anchor_strength: float = 9.0
    train_relation_noise: float = 0.0
    batch_size: int = 96
    train_steps: int = 3000
    learning_rate: float = 1.5e-3
    consistency_weight: float = 0.15
    log_every: int = 100


def make_random_constraint_graph(
    batch: int,
    nodes: int,
    classes: int,
    extra_edges_per_node: float,
    relation_noise: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Connected random graphs with one observed anchor and modular constraints."""
    labels = torch.randint(classes, (batch, nodes), device=device)

    chain_u = torch.arange(nodes - 1, device=device).view(1, -1).expand(batch, -1)
    chain_v = chain_u + 1
    extra_edges = max(1, int(round(extra_edges_per_node * nodes)))
    random_u = torch.randint(nodes, (batch, extra_edges), device=device)
    jump = torch.randint(1, nodes, (batch, extra_edges), device=device)
    random_v = (random_u + jump) % nodes
    edge_u = torch.cat((chain_u, random_u), dim=1)
    edge_v = torch.cat((chain_v, random_v), dim=1)

    value_u = labels.gather(1, edge_u)
    value_v = labels.gather(1, edge_v)
    clean_relation = (value_v - value_u) % classes
    corrupted = torch.rand_like(clean_relation.float()) < relation_noise
    nonzero_error = torch.randint(1, classes, clean_relation.shape, device=device)
    relation = torch.where(corrupted, (clean_relation + nonzero_error) % classes, clean_relation)

    observed = torch.zeros(batch, nodes, device=device, dtype=torch.bool)
    anchor = torch.randint(nodes, (batch, 1), device=device)
    observed.scatter_(1, anchor, True)
    return {
        "labels": labels,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "relation": relation,
        "clean_relation": clean_relation,
        "corrupted": corrupted,
        "observed": observed,
    }


class ViolationDrivenConstraintField(nn.Module):
    """A sparse event-driven field of learned constraint corrections."""

    def __init__(self, classes: int, hidden: int, anchor_strength: float) -> None:
        super().__init__()
        self.classes = classes
        self.anchor_strength = anchor_strength
        factor_input = 5 * classes + 1
        self.factor = nn.Sequential(
            nn.Linear(factor_input, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * classes),
        )
        self.step_logit = nn.Parameter(torch.tensor(-0.25))

    def anchor_logits(self, labels: torch.Tensor) -> torch.Tensor:
        result = torch.full(
            (*labels.shape, self.classes),
            -self.anchor_strength,
            device=labels.device,
            dtype=torch.float32,
        )
        result.scatter_(-1, labels.unsqueeze(-1), self.anchor_strength)
        return result

    def forward(
        self,
        labels: torch.Tensor,
        edge_u: torch.Tensor,
        edge_v: torch.Tensor,
        relation: torch.Tensor,
        observed: torch.Tensor,
        iterations: int,
        active_fraction: float,
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        batch, nodes = labels.shape
        edges = edge_u.shape[1]
        anchors = self.anchor_logits(labels)
        state = torch.zeros(batch, nodes, self.classes, device=labels.device)
        state = torch.where(observed.unsqueeze(-1), anchors, state)
        relation_one_hot = F.one_hot(relation, self.classes).float()
        eta = torch.sigmoid(self.step_logit)
        active_count = max(1, min(edges, int(math.ceil(active_fraction * edges))))

        states: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []
        for _ in range(iterations):
            probability = state.softmax(dim=-1)
            prob_u = gather_nodes(probability, edge_u)
            prob_v = gather_nodes(probability, edge_v)
            implied_v = shift_distribution(prob_u, relation)
            implied_u = inverse_shift_distribution(prob_v, relation)

            mismatch_v = (prob_v - implied_v).abs().mean(dim=-1)
            mismatch_u = (prob_u - implied_u).abs().mean(dim=-1)
            violation = 0.5 * (mismatch_u + mismatch_v)

            top_index = violation.topk(active_count, dim=1, sorted=False).indices
            active = torch.zeros_like(violation)
            active.scatter_(1, top_index, 1.0)

            factor_input = torch.cat(
                (
                    prob_u,
                    prob_v,
                    implied_u,
                    implied_v,
                    relation_one_hot,
                    violation.unsqueeze(-1),
                ),
                dim=-1,
            )
            correction_u, correction_v = self.factor(factor_input).chunk(2, dim=-1)
            gate = active.unsqueeze(-1) * violation.unsqueeze(-1)
            correction_u = torch.tanh(correction_u) * gate
            correction_v = torch.tanh(correction_v) * gate

            active_weight = active.unsqueeze(-1)
            sum_u, count_u = scatter_edges_to_nodes(
                correction_u, edge_u, nodes, active_weight
            )
            sum_v, count_v = scatter_edges_to_nodes(
                correction_v, edge_v, nodes, active_weight
            )
            correction = (sum_u + sum_v) / (count_u + count_v).clamp_min(1.0)
            state = state + eta * correction
            state = torch.where(observed.unsqueeze(-1), anchors, state)

            states.append(state)
            diagnostics.append(
                {
                    "mean_violation": violation.mean(dim=1),
                    "max_violation": violation.max(dim=1).values,
                    "active_fraction": active.mean(dim=1),
                }
            )
        return states, diagnostics


def factor_loss(
    states: list[torch.Tensor],
    graph: dict[str, torch.Tensor],
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = graph["labels"]
    unknown = ~graph["observed"]
    weighted_losses = []
    weights = torch.linspace(0.25, 1.0, len(states), device=labels.device)
    for weight, state in zip(weights, states):
        ce = F.cross_entropy(state.transpose(1, 2), labels, reduction="none")
        weighted_losses.append(weight * masked_mean(ce, unknown))
    supervised = torch.stack(weighted_losses).sum() / weights.sum()

    probability = states[-1].softmax(dim=-1)
    prob_u = gather_nodes(probability, graph["edge_u"])
    prob_v = gather_nodes(probability, graph["edge_v"])
    implied_v = shift_distribution(prob_u, graph["relation"])
    consistency = F.mse_loss(prob_v, implied_v)
    total = supervised + consistency_weight * consistency
    return total, {
        "supervised": supervised.detach().item(),
        "consistency": consistency.detach().item(),
    }


@torch.no_grad()
def evaluate_factor_case(
    model: ViolationDrivenConstraintField,
    cfg: FactorConfig,
    device: torch.device,
    nodes: int,
    iterations: int,
    relation_noise: float,
    active_fraction: float,
    batches: int = 8,
) -> dict[str, Any]:
    model.eval()
    correct_by_step = torch.zeros(iterations, device=device)
    total_unknown = 0
    clean_edge_correct = 0
    total_edges = 0
    final_violation = 0.0

    for _ in range(batches):
        eval_batch = min(cfg.batch_size, 32)
        graph = make_random_constraint_graph(
            eval_batch,
            nodes,
            cfg.classes,
            cfg.extra_edges_per_node,
            relation_noise,
            device,
        )
        states, diagnostics = model(
            graph["labels"],
            graph["edge_u"],
            graph["edge_v"],
            graph["relation"],
            graph["observed"],
            iterations,
            active_fraction,
        )
        unknown = ~graph["observed"]
        for index, state in enumerate(states):
            correct_by_step[index] += (state.argmax(-1).eq(graph["labels"]) & unknown).sum()
        total_unknown += unknown.sum().item()
        prediction = states[-1].argmax(-1)
        pred_u = prediction.gather(1, graph["edge_u"])
        pred_v = prediction.gather(1, graph["edge_v"])
        clean_ok = ((pred_v - pred_u) % cfg.classes).eq(graph["clean_relation"])
        clean_edge_correct += clean_ok.sum().item()
        total_edges += clean_ok.numel()
        final_violation += diagnostics[-1]["mean_violation"].sum().item()

    return {
        "nodes": nodes,
        "iterations": iterations,
        "relation_noise": relation_noise,
        "active_fraction": active_fraction,
        "unknown_accuracy_by_iteration": (correct_by_step / total_unknown).cpu().tolist(),
        "final_unknown_accuracy": (correct_by_step[-1] / total_unknown).item(),
        "clean_edge_accuracy": clean_edge_correct / total_edges,
        "mean_final_violation": final_violation / (batches * min(cfg.batch_size, 32)),
        "random_accuracy": 1.0 / cfg.classes,
    }


def run_factor(cfg: FactorConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n=== V2 / EXPERIMENT 1: VIOLATION-DRIVEN CONSTRAINT FIELD ===", flush=True)
    model = ViolationDrivenConstraintField(cfg.classes, cfg.hidden, cfg.anchor_strength).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    started = time.time()

    for step in range(1, cfg.train_steps + 1):
        model.train()
        graph = make_random_constraint_graph(
            cfg.batch_size,
            cfg.train_nodes,
            cfg.classes,
            cfg.extra_edges_per_node,
            cfg.train_relation_noise,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        states, diagnostics = model(
            graph["labels"],
            graph["edge_u"],
            graph["edge_v"],
            graph["relation"],
            graph["observed"],
            cfg.train_iterations,
            cfg.active_fraction,
        )
        loss, parts = factor_loss(states, graph, cfg.consistency_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            unknown = ~graph["observed"]
            acc = masked_mean(states[-1].argmax(-1).eq(graph["labels"]).float(), unknown).item()
            violation = diagnostics[-1]["mean_violation"].mean().item()
            print(
                f"[factor-v2] step {step:5d}/{cfg.train_steps} loss={loss.item():.4f} "
                f"acc={acc:.3f} violation={violation:.4f} "
                f"cons={parts['consistency']:.4f} time={time.time()-started:.1f}s",
                flush=True,
            )

    cases: dict[str, Any] = {}
    cases["train_size_sparse"] = evaluate_factor_case(
        model, cfg, device, cfg.train_nodes, cfg.train_iterations, 0.0, cfg.active_fraction
    )
    cases["large_sparse"] = evaluate_factor_case(
        model, cfg, device, cfg.test_nodes, cfg.test_iterations, 0.0, cfg.active_fraction
    )
    cases["large_dense_ablation"] = evaluate_factor_case(
        model, cfg, device, cfg.test_nodes, cfg.test_iterations, 0.0, 1.0
    )
    for noise in (0.02, 0.05, 0.10):
        cases[f"large_noise_{noise:.2f}"] = evaluate_factor_case(
            model, cfg, device, cfg.test_nodes, cfg.test_iterations, noise, cfg.active_fraction
        )
    print(json.dumps(cases, indent=2), flush=True)
    state = {"model": model.state_dict(), "config": asdict(cfg)}
    return cases, state


# -----------------------------------------------------------------------------
# Experiment 2: Quadratic Shared-Basis Field
# -----------------------------------------------------------------------------


@dataclass
class BasisConfig:
    dimension: int = 8
    train_repeats: int = 4
    test_repeats: int = 8
    basis_rank: int = 8
    hidden: int = 160
    iterations: int = 4
    train_noise: float = 0.005
    batch_size: int = 80
    train_steps: int = 2400
    learning_rate: float = 1e-3
    orthogonal_weight: float = 1e-3
    log_every: int = 100
    eigh_grad: bool = False


def random_orthogonal(batch_shape: tuple[int, ...], dimension: int, device: torch.device) -> torch.Tensor:
    raw = torch.randn(*batch_shape, dimension, dimension, device=device)
    flat = raw.reshape(-1, dimension, dimension)
    q = torch.linalg.qr(flat).Q
    return q.reshape(*batch_shape, dimension, dimension)


def make_matched_covariance_set(
    batch: int,
    dimension: int,
    repeats: int,
    noise: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Two per-point classes with the same marginal distribution and covariance.

    Inliers repeat axes from one random orthogonal frame.  Outliers use axes
    from independent random frames.  Each class contains complete orthogonal
    frames, so both class covariances are exactly I/d before optional noise.
    """
    shared_frame = random_orthogonal((batch,), dimension, device)
    inliers = shared_frame.transpose(-1, -2).unsqueeze(1)
    inliers = inliers.expand(-1, repeats, -1, -1).reshape(batch, repeats * dimension, dimension)

    independent_frames = random_orthogonal((batch, repeats), dimension, device)
    outliers = independent_frames.transpose(-1, -2).reshape(batch, repeats * dimension, dimension)

    sign_in = torch.randint(0, 2, inliers.shape[:-1], device=device).float().mul_(2).sub_(1)
    sign_out = torch.randint(0, 2, outliers.shape[:-1], device=device).float().mul_(2).sub_(1)
    inliers = inliers * sign_in.unsqueeze(-1)
    outliers = outliers * sign_out.unsqueeze(-1)
    if noise > 0:
        inliers = inliers + noise * torch.randn_like(inliers)
        outliers = outliers + noise * torch.randn_like(outliers)
    inliers = F.normalize(inliers, dim=-1)
    outliers = F.normalize(outliers, dim=-1)

    points = torch.cat((inliers, outliers), dim=1)
    labels = torch.cat(
        (
            torch.zeros(batch, repeats * dimension, device=device, dtype=torch.long),
            torch.ones(batch, repeats * dimension, device=device, dtype=torch.long),
        ),
        dim=1,
    )
    permutation = torch.rand(batch, points.shape[1], device=device).argsort(dim=1)
    points = points.gather(1, permutation.unsqueeze(-1).expand_as(points))
    labels = labels.gather(1, permutation)

    cov_in = torch.bmm(inliers.transpose(1, 2), inliers) / inliers.shape[1]
    cov_out = torch.bmm(outliers.transpose(1, 2), outliers) / outliers.shape[1]
    covariance_gap = (cov_in - cov_out).square().mean(dim=(-2, -1)).sqrt()
    return points, labels, {"covariance_gap": covariance_gap}


class QuadraticLift(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        row, col = torch.triu_indices(dimension, dimension)
        self.register_buffer("row", row)
        self.register_buffer("col", col)
        scale = torch.where(row.eq(col), torch.ones_like(row, dtype=torch.float32), math.sqrt(2.0) * torch.ones_like(row, dtype=torch.float32))
        self.register_buffer("scale", scale)
        self.output_dim = row.numel()
        self.projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        lifted = outer[..., self.row, self.col] * self.scale
        return self.projection(lifted)

    def orthogonality_loss(self) -> torch.Tensor:
        weight = self.projection.weight
        identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        return (weight @ weight.T - identity).square().mean()


class QuadraticSharedBasisField(nn.Module):
    def __init__(self, dimension: int, rank: int, hidden: int, eigh_grad: bool) -> None:
        super().__init__()
        self.lift = QuadraticLift(dimension)
        self.state_dim = self.lift.output_dim
        self.rank = rank
        self.eigh_grad = eigh_grad
        update_input = 4 * self.state_dim + rank
        self.update = nn.Sequential(
            nn.Linear(update_input, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.state_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(update_input, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.state_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 + rank, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )

    def geometry(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = state.shape[1]
        covariance = torch.bmm(state.transpose(1, 2), state) / count
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        basis = eigenvectors[:, :, -self.rank :].to(state.dtype)
        spectrum = eigenvalues[:, -self.rank :].to(state.dtype)
        if not self.eigh_grad:
            basis = basis.detach()
            spectrum = spectrum.detach()
        projected = torch.bmm(torch.bmm(state, basis), basis.transpose(1, 2))
        residual = state - projected
        spectrum = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return projected, residual, spectrum

    def forward(self, points: torch.Tensor, iterations: int) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        state = self.lift(points)
        logits_steps: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []
        for step in range(iterations):
            projected, residual, spectrum = self.geometry(state)
            projected_energy = projected.square().mean(dim=-1, keepdim=True)
            residual_energy = residual.square().mean(dim=-1, keepdim=True)
            spectrum_broadcast = spectrum.unsqueeze(1).expand(-1, state.shape[1], -1)
            readout = torch.cat((projected_energy, residual_energy, spectrum_broadcast), dim=-1)
            logits_steps.append(self.classifier(readout))
            diagnostics.append(
                {
                    "projected_energy": projected_energy,
                    "residual_energy": residual_energy,
                    "spectrum": spectrum,
                }
            )
            if step + 1 < iterations:
                update_input = torch.cat(
                    (state, projected, residual, residual.square(), spectrum_broadcast), dim=-1
                )
                state = state + self.gate(update_input) * torch.tanh(self.update(update_input))
                state = state / state.square().mean(dim=-1, keepdim=True).add(1e-5).sqrt()
        return logits_steps, diagnostics


class IndependentPointControl(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


class LinearBasisControl(nn.Module):
    """Set-aware control using only the raw second moment of x."""

    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.rank = max(1, dimension // 2)
        self.classifier = nn.Sequential(
            nn.Linear(2 + self.rank, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        covariance = torch.bmm(points.transpose(1, 2), points) / points.shape[1]
        values, vectors = torch.linalg.eigh(covariance.float())
        basis = vectors[:, :, -self.rank :].to(points.dtype).detach()
        spectrum = values[:, -self.rank :].to(points.dtype).detach()
        projected = torch.bmm(torch.bmm(points, basis), basis.transpose(1, 2))
        residual = points - projected
        spectrum = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        readout = torch.cat(
            (
                projected.square().mean(-1, keepdim=True),
                residual.square().mean(-1, keepdim=True),
                spectrum.unsqueeze(1).expand(-1, points.shape[1], -1),
            ),
            dim=-1,
        )
        return self.classifier(readout)


@torch.no_grad()
def evaluate_basis_case(
    model: QuadraticSharedBasisField,
    independent: IndependentPointControl,
    linear_basis: LinearBasisControl,
    cfg: BasisConfig,
    device: torch.device,
    repeats: int,
    noise: float,
    batches: int = 15,
) -> dict[str, Any]:
    model.eval()
    independent.eval()
    linear_basis.eval()
    correct_steps = torch.zeros(cfg.iterations, device=device)
    independent_correct = 0
    linear_correct = 0
    total = 0
    covariance_gap = 0.0
    residual_in = 0.0
    residual_out = 0.0
    count_in = 0
    count_out = 0

    for _ in range(batches):
        eval_batch = min(cfg.batch_size, 40)
        points, labels, data_diag = make_matched_covariance_set(
            eval_batch, cfg.dimension, repeats, noise, device
        )
        logits_steps, diagnostics = model(points, cfg.iterations)
        for index, logits in enumerate(logits_steps):
            correct_steps[index] += logits.argmax(-1).eq(labels).sum()
        independent_correct += independent(points).argmax(-1).eq(labels).sum().item()
        linear_correct += linear_basis(points).argmax(-1).eq(labels).sum().item()
        total += labels.numel()
        covariance_gap += data_diag["covariance_gap"].sum().item()

        residual = diagnostics[-1]["residual_energy"].squeeze(-1)
        inlier = labels.eq(0)
        outlier = labels.eq(1)
        residual_in += residual[inlier].sum().item()
        residual_out += residual[outlier].sum().item()
        count_in += inlier.sum().item()
        count_out += outlier.sum().item()

    return {
        "set_size": 2 * repeats * cfg.dimension,
        "noise": noise,
        "accuracy_by_iteration": (correct_steps / total).cpu().tolist(),
        "final_accuracy": (correct_steps[-1] / total).item(),
        "independent_control_accuracy": independent_correct / total,
        "linear_basis_control_accuracy": linear_correct / total,
        "mean_class_covariance_gap": covariance_gap / (
            batches * min(cfg.batch_size, 40)
        ),
        "residual_inlier": residual_in / count_in,
        "residual_outlier": residual_out / count_out,
        "residual_ratio": (residual_out / count_out) / max(residual_in / count_in, 1e-9),
        "chance_accuracy": 0.5,
    }


def run_basis(cfg: BasisConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n=== V2 / EXPERIMENT 2: QUADRATIC SHARED-BASIS FIELD ===", flush=True)
    model = QuadraticSharedBasisField(
        cfg.dimension, cfg.basis_rank, cfg.hidden, cfg.eigh_grad
    ).to(device)
    independent = IndependentPointControl(cfg.dimension, cfg.hidden).to(device)
    linear_basis = LinearBasisControl(cfg.dimension, cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    independent_optimizer = torch.optim.AdamW(
        independent.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    linear_optimizer = torch.optim.AdamW(
        linear_basis.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    started = time.time()

    for step in range(1, cfg.train_steps + 1):
        model.train()
        independent.train()
        linear_basis.train()
        points, labels, diagnostics_data = make_matched_covariance_set(
            cfg.batch_size,
            cfg.dimension,
            cfg.train_repeats,
            cfg.train_noise,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits_steps, diagnostics = model(points, cfg.iterations)
        weights = torch.linspace(0.35, 1.0, cfg.iterations, device=device)
        losses = [F.cross_entropy(x.reshape(-1, 2), labels.reshape(-1)) for x in logits_steps]
        classification = torch.stack([w * loss for w, loss in zip(weights, losses)]).sum() / weights.sum()
        orthogonal = model.lift.orthogonality_loss()
        loss = classification + cfg.orthogonal_weight * orthogonal
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        independent_optimizer.zero_grad(set_to_none=True)
        independent_logits = independent(points)
        independent_loss = F.cross_entropy(independent_logits.reshape(-1, 2), labels.reshape(-1))
        independent_loss.backward()
        independent_optimizer.step()

        linear_optimizer.zero_grad(set_to_none=True)
        linear_logits = linear_basis(points)
        linear_loss = F.cross_entropy(linear_logits.reshape(-1, 2), labels.reshape(-1))
        linear_loss.backward()
        linear_optimizer.step()

        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            acc = logits_steps[-1].argmax(-1).eq(labels).float().mean().item()
            ind_acc = independent_logits.argmax(-1).eq(labels).float().mean().item()
            lin_acc = linear_logits.argmax(-1).eq(labels).float().mean().item()
            residual = diagnostics[-1]["residual_energy"].squeeze(-1)
            ratio = residual[labels.eq(1)].mean() / residual[labels.eq(0)].mean().clamp_min(1e-9)
            cov_gap = diagnostics_data["covariance_gap"].mean().item()
            print(
                f"[basis-v2] step {step:5d}/{cfg.train_steps} loss={loss.item():.4f} "
                f"acc={acc:.3f} independent={ind_acc:.3f} linear={lin_acc:.3f} "
                f"res(out/in)={ratio.item():.2f} cov_gap={cov_gap:.6f} "
                f"time={time.time()-started:.1f}s",
                flush=True,
            )

    cases: dict[str, Any] = {}
    cases["train_size"] = evaluate_basis_case(
        model, independent, linear_basis, cfg, device, cfg.train_repeats, cfg.train_noise
    )
    cases["double_set_size"] = evaluate_basis_case(
        model, independent, linear_basis, cfg, device, cfg.test_repeats, cfg.train_noise
    )
    for noise in (0.0, 0.02, 0.05, 0.10):
        cases[f"double_size_noise_{noise:.2f}"] = evaluate_basis_case(
            model, independent, linear_basis, cfg, device, cfg.test_repeats, noise
        )
    print(json.dumps(cases, indent=2), flush=True)
    state = {
        "model": model.state_dict(),
        "independent_control": independent.state_dict(),
        "linear_basis_control": linear_basis.state_dict(),
        "config": asdict(cfg),
    }
    return cases, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("factor", "basis", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("two_architecture_runs_v2"))
    parser.add_argument("--factor-steps", type=int, default=None)
    parser.add_argument("--basis-steps", type=int, default=None)
    parser.add_argument("--eigh-grad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    factor_cfg = FactorConfig()
    basis_cfg = BasisConfig(eigh_grad=args.eigh_grad)
    if args.factor_steps is not None:
        factor_cfg.train_steps = args.factor_steps
    if args.basis_steps is not None:
        basis_cfg.train_steps = args.basis_steps
    if args.smoke:
        factor_cfg.train_steps = 8
        factor_cfg.batch_size = 6
        factor_cfg.hidden = 32
        factor_cfg.train_iterations = 4
        factor_cfg.test_iterations = 6
        factor_cfg.test_nodes = 32
        factor_cfg.log_every = 2
        basis_cfg.train_steps = 8
        basis_cfg.batch_size = 4
        basis_cfg.hidden = 32
        basis_cfg.iterations = 2
        basis_cfg.log_every = 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} seed={args.seed} smoke={args.smoke}", flush=True)
    results: dict[str, Any] = {"device": str(device), "seed": args.seed, "smoke": args.smoke}

    if args.experiment in ("factor", "all"):
        result, state = run_factor(factor_cfg, device)
        results["factor"] = result
        torch.save(state, args.output_dir / "factor_v2.pt")

    if args.experiment in ("basis", "all"):
        result, state = run_basis(basis_cfg, device)
        results["basis"] = result
        torch.save(state, args.output_dir / "basis_v2.pt")

    output = args.output_dir / "results_v2.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output}", flush=True)


if __name__ == "__main__":
    main()
