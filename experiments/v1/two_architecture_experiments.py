#!/usr/bin/env python3
"""
Two attention-free architecture probes.

Experiment 1: factor correction
--------------------------------
Learned local constraint factors iteratively repair a field of discrete
variables.  Information can travel only through factor -> variable correction
messages; there is no self-attention and no token mixer.

Experiment 2: basis communication
---------------------------------
Elements communicate only by contributing to a shared second-moment matrix.
The eigenspace of that matrix is broadcast back to every element and used for
iterative state updates.  There is no pairwise token interaction.

Both experiments include a control that processes each element independently.

Examples:
    python two_architecture_experiments.py --smoke
    python two_architecture_experiments.py --experiment factor
    python two_architecture_experiments.py --experiment basis
    python two_architecture_experiments.py --experiment all --device cuda
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    correct = logits.argmax(dim=-1).eq(labels)
    if mask is None:
        return correct.float().mean().item()
    return masked_mean(correct.float(), mask).item()


@dataclass
class FactorConfig:
    classes: int = 7
    train_nodes: int = 24
    test_nodes: int = 48
    observed_probability: float = 0.22
    hidden: int = 96
    relation_dim: int = 24
    train_iterations: int = 12
    test_iterations: int = 24
    batch_size: int = 128
    train_steps: int = 2500
    learning_rate: float = 2e-3
    consistency_weight: float = 0.25
    log_every: int = 100


def make_constraint_chain(
    batch_size: int,
    nodes: int,
    classes: int,
    observed_probability: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create chains satisfying value[i+1] = value[i] + delta[i] (mod K)."""
    root = torch.randint(classes, (batch_size, 1), device=device)
    delta = torch.randint(classes, (batch_size, nodes - 1), device=device)
    cumulative = torch.cumsum(delta, dim=1)
    labels = torch.cat((root, (root + cumulative) % classes), dim=1)

    observed = torch.rand(batch_size, nodes, device=device) < observed_probability
    # Always provide at least one anchor, at a random location.
    anchor = torch.randint(nodes, (batch_size, 1), device=device)
    observed.scatter_(1, anchor, True)
    return labels, delta, observed


class ConstraintFactorNet(nn.Module):
    """Shared learned factors repeatedly correct a chain of latent variables."""

    def __init__(self, classes: int, hidden: int, relation_dim: int) -> None:
        super().__init__()
        self.classes = classes
        self.hidden = hidden
        self.value_embedding = nn.Embedding(classes, hidden)
        self.unknown = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.relation_embedding = nn.Embedding(classes, relation_dim)

        factor_input = 2 * hidden + relation_dim
        self.factor = nn.Sequential(
            nn.Linear(factor_input, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden),
        )
        self.update_norm = nn.LayerNorm(hidden)
        self.step_logit = nn.Parameter(torch.tensor(-0.5))
        self.decoder = nn.Linear(hidden, classes)

    def forward(
        self,
        labels: torch.Tensor,
        delta: torch.Tensor,
        observed: torch.Tensor,
        iterations: int,
    ) -> list[torch.Tensor]:
        batch, nodes = labels.shape
        anchors = self.value_embedding(labels)
        unknown = self.unknown.view(1, 1, -1).expand(batch, nodes, -1)
        h = torch.where(observed.unsqueeze(-1), anchors, unknown)
        relation = self.relation_embedding(delta)
        degree = torch.ones(1, nodes, 1, device=h.device, dtype=h.dtype)
        if nodes > 2:
            degree[:, 1:-1] = 2.0
        eta = torch.sigmoid(self.step_logit)

        logits_by_step: list[torch.Tensor] = []
        for _ in range(iterations):
            pair = torch.cat((h[:, :-1], h[:, 1:], relation), dim=-1)
            correction = self.factor(pair)
            to_left, to_right = correction.chunk(2, dim=-1)

            aggregate = torch.zeros_like(h)
            aggregate[:, :-1] = aggregate[:, :-1] + to_left
            aggregate[:, 1:] = aggregate[:, 1:] + to_right
            aggregate = aggregate / degree

            h = self.update_norm(h + eta * torch.tanh(aggregate))
            # Observed variables are boundary conditions, not mutable guesses.
            h = torch.where(observed.unsqueeze(-1), anchors, h)
            logits_by_step.append(self.decoder(h))
        return logits_by_step


class IndependentNodeControl(nn.Module):
    """Control: identical inputs but no path between different nodes."""

    def __init__(self, classes: int, hidden: int) -> None:
        super().__init__()
        self.classes = classes
        self.net = nn.Sequential(
            nn.Linear(classes + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, labels: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, self.classes).float()
        visible = one_hot * observed.unsqueeze(-1)
        x = torch.cat((visible, observed.unsqueeze(-1).float()), dim=-1)
        return self.net(x)


def shifted_distribution(prob_left: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Distribution implied for the right variable by right=left+delta mod K."""
    classes = prob_left.shape[-1]
    right_class = torch.arange(classes, device=prob_left.device).view(1, 1, classes)
    source_class = (right_class - delta.unsqueeze(-1)) % classes
    return prob_left.gather(-1, source_class.expand_as(prob_left))


def factor_training_loss(
    logits_by_step: list[torch.Tensor],
    labels: torch.Tensor,
    delta: torch.Tensor,
    observed: torch.Tensor,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    unknown = ~observed
    losses = []
    for step_index, logits in enumerate(logits_by_step):
        token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        weight = (step_index + 1) / len(logits_by_step)
        losses.append(weight * masked_mean(token_loss, unknown))
    supervised = torch.stack(losses).sum() / sum(
        (i + 1) / len(logits_by_step) for i in range(len(logits_by_step))
    )

    final_prob = logits_by_step[-1].softmax(dim=-1)
    implied_right = shifted_distribution(final_prob[:, :-1], delta)
    consistency = F.mse_loss(final_prob[:, 1:], implied_right)
    total = supervised + consistency_weight * consistency
    return total, {
        "supervised": supervised.detach().item(),
        "soft_consistency": consistency.detach().item(),
    }


@torch.no_grad()
def evaluate_factor(
    model: ConstraintFactorNet,
    control: IndependentNodeControl,
    cfg: FactorConfig,
    device: torch.device,
    nodes: int,
    iterations: int,
    batches: int = 20,
) -> dict[str, Any]:
    model.eval()
    control.eval()
    step_correct = torch.zeros(iterations, device=device)
    step_total = torch.zeros(iterations, device=device)
    edge_correct = 0.0
    edge_total = 0
    control_correct = 0.0
    unknown_total = 0

    for _ in range(batches):
        labels, delta, observed = make_constraint_chain(
            cfg.batch_size, nodes, cfg.classes, cfg.observed_probability, device
        )
        unknown = ~observed
        logits_steps = model(labels, delta, observed, iterations)
        for index, logits in enumerate(logits_steps):
            correct = logits.argmax(-1).eq(labels) & unknown
            step_correct[index] += correct.sum()
            step_total[index] += unknown.sum()

        prediction = logits_steps[-1].argmax(-1)
        relation_ok = ((prediction[:, 1:] - prediction[:, :-1]) % cfg.classes).eq(delta)
        edge_correct += relation_ok.sum().item()
        edge_total += relation_ok.numel()

        control_prediction = control(labels, observed).argmax(-1)
        control_correct += (control_prediction.eq(labels) & unknown).sum().item()
        unknown_total += unknown.sum().item()

    return {
        "nodes": nodes,
        "iterations": iterations,
        "unknown_accuracy_by_iteration": (step_correct / step_total.clamp_min(1)).cpu().tolist(),
        "final_unknown_accuracy": (step_correct[-1] / step_total[-1]).item(),
        "edge_constraint_accuracy": edge_correct / edge_total,
        "independent_control_unknown_accuracy": control_correct / unknown_total,
        "random_unknown_accuracy": 1.0 / cfg.classes,
    }


def run_factor_experiment(cfg: FactorConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n=== EXPERIMENT 1: LEARNED FACTOR CORRECTION ===", flush=True)
    model = ConstraintFactorNet(cfg.classes, cfg.hidden, cfg.relation_dim).to(device)
    control = IndependentNodeControl(cfg.classes, cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    started = time.time()

    model.train()
    control.train()
    for step in range(1, cfg.train_steps + 1):
        labels, delta, observed = make_constraint_chain(
            cfg.batch_size,
            cfg.train_nodes,
            cfg.classes,
            cfg.observed_probability,
            device,
        )
        unknown = ~observed

        optimizer.zero_grad(set_to_none=True)
        logits_steps = model(labels, delta, observed, cfg.train_iterations)
        loss, parts = factor_training_loss(
            logits_steps, labels, delta, observed, cfg.consistency_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        control_optimizer.zero_grad(set_to_none=True)
        control_logits = control(labels, observed)
        control_loss_map = F.cross_entropy(
            control_logits.transpose(1, 2), labels, reduction="none"
        )
        control_loss = masked_mean(control_loss_map, unknown)
        control_loss.backward()
        control_optimizer.step()

        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            acc = accuracy(logits_steps[-1], labels, unknown)
            ctrl_acc = accuracy(control_logits, labels, unknown)
            elapsed = time.time() - started
            print(
                f"[factor] step {step:5d}/{cfg.train_steps} "
                f"loss={loss.item():.4f} acc={acc:.3f} "
                f"control={ctrl_acc:.3f} cons={parts['soft_consistency']:.4f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

    train_size_eval = evaluate_factor(
        model, control, cfg, device, cfg.train_nodes, cfg.train_iterations
    )
    longer_eval = evaluate_factor(
        model, control, cfg, device, cfg.test_nodes, cfg.test_iterations
    )
    result = {"train_length": train_size_eval, "longer_length": longer_eval}
    print(json.dumps(result, indent=2), flush=True)
    state = {"model": model.state_dict(), "control": control.state_dict(), "config": asdict(cfg)}
    return result, state


@dataclass
class BasisConfig:
    input_dim: int = 16
    subspace_rank: int = 4
    set_size: int = 64
    hidden: int = 128
    iterations: int = 4
    inlier_noise: float = 0.025
    outlier_probability: float = 0.50
    batch_size: int = 96
    train_steps: int = 1800
    learning_rate: float = 1e-3
    log_every: int = 100
    eigh_grad: bool = False


def make_subspace_set(
    batch_size: int,
    set_size: int,
    dimension: int,
    rank: int,
    inlier_noise: float,
    outlier_probability: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create sets whose inliers share a random subspace.

    Marginally, a single normalized point has random orientation, so an
    element-wise classifier cannot identify whether it belongs to the common
    subspace.  The distinction exists only at set level.
    """
    raw_basis = torch.randn(batch_size, dimension, rank, device=device)
    basis = torch.linalg.qr(raw_basis, mode="reduced").Q
    coordinates = torch.randn(batch_size, set_size, rank, device=device)
    inliers = torch.bmm(coordinates, basis.transpose(1, 2))
    inliers = inliers + inlier_noise * torch.randn_like(inliers)
    outliers = torch.randn(batch_size, set_size, dimension, device=device)

    inliers = F.normalize(inliers, dim=-1)
    outliers = F.normalize(outliers, dim=-1)
    labels = (torch.rand(batch_size, set_size, device=device) < outlier_probability).long()
    points = torch.where(labels.unsqueeze(-1).bool(), outliers, inliers)
    return points, labels


class SharedBasisNet(nn.Module):
    """Iterative communication through a shared covariance eigenspace only."""

    def __init__(self, dimension: int, rank: int, hidden: int, eigh_grad: bool) -> None:
        super().__init__()
        self.dimension = dimension
        self.rank = rank
        self.eigh_grad = eigh_grad
        self.phi = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.phi.weight.copy_(torch.eye(dimension))

        update_input = 4 * dimension + rank
        self.update = nn.Sequential(
            nn.Linear(update_input, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )
        self.gate = nn.Sequential(
            nn.Linear(update_input, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, dimension),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 + rank, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )

    def shared_geometry(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, dimension = h.shape
        covariance = torch.bmm(h.transpose(1, 2), h) / float(count)
        eye = torch.eye(dimension, device=h.device, dtype=h.dtype).unsqueeze(0)
        covariance = covariance + 1e-5 * eye
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        basis = eigenvectors[:, :, -self.rank :].to(h.dtype)
        top_values = eigenvalues[:, -self.rank :].to(h.dtype)
        if not self.eigh_grad:
            basis = basis.detach()
            top_values = top_values.detach()

        projected = torch.bmm(torch.bmm(h, basis), basis.transpose(1, 2))
        residual = h - projected
        spectrum = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return projected, residual, spectrum, covariance

    def forward(
        self, points: torch.Tensor, iterations: int
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        h = self.phi(points)
        logits_by_step: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []

        for step in range(iterations):
            projected, residual, spectrum, covariance = self.shared_geometry(h)
            projected_energy = projected.square().mean(dim=-1, keepdim=True)
            residual_energy = residual.square().mean(dim=-1, keepdim=True)
            spectrum_broadcast = spectrum.unsqueeze(1).expand(-1, h.shape[1], -1)

            readout = torch.cat(
                (projected_energy, residual_energy, spectrum_broadcast), dim=-1
            )
            logits_by_step.append(self.classifier(readout))
            diagnostics.append(
                {
                    "projected_energy": projected_energy,
                    "residual_energy": residual_energy,
                    "spectrum": spectrum,
                    "covariance_trace": covariance.diagonal(dim1=-2, dim2=-1).sum(-1),
                }
            )

            if step + 1 < iterations:
                update_input = torch.cat(
                    (h, projected, residual, residual.square(), spectrum_broadcast), dim=-1
                )
                delta = torch.tanh(self.update(update_input))
                gate = self.gate(update_input)
                h = h + gate * delta
                # Per-element scale control; no aggregation is performed here.
                h = h / h.square().mean(dim=-1, keepdim=True).add(1e-5).sqrt()

        return logits_by_step, diagnostics


class IndependentPointControl(nn.Module):
    """Control: sees each normalized point independently, without set statistics."""

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


@torch.no_grad()
def evaluate_basis(
    model: SharedBasisNet,
    control: IndependentPointControl,
    cfg: BasisConfig,
    device: torch.device,
    batches: int = 30,
) -> dict[str, Any]:
    model.eval()
    control.eval()
    correct_by_step = torch.zeros(cfg.iterations, device=device)
    total = 0
    control_correct = 0
    residual_inlier = 0.0
    residual_outlier = 0.0
    inlier_count = 0
    outlier_count = 0

    for _ in range(batches):
        points, labels = make_subspace_set(
            cfg.batch_size,
            cfg.set_size,
            cfg.input_dim,
            cfg.subspace_rank,
            cfg.inlier_noise,
            cfg.outlier_probability,
            device,
        )
        logits_steps, diagnostics = model(points, cfg.iterations)
        for index, logits in enumerate(logits_steps):
            correct_by_step[index] += logits.argmax(-1).eq(labels).sum()
        control_correct += control(points).argmax(-1).eq(labels).sum().item()
        total += labels.numel()

        residual = diagnostics[-1]["residual_energy"].squeeze(-1)
        inlier = labels.eq(0)
        outlier = labels.eq(1)
        residual_inlier += residual[inlier].sum().item()
        residual_outlier += residual[outlier].sum().item()
        inlier_count += inlier.sum().item()
        outlier_count += outlier.sum().item()

    return {
        "accuracy_by_iteration": (correct_by_step / total).cpu().tolist(),
        "final_accuracy": (correct_by_step[-1] / total).item(),
        "independent_control_accuracy": control_correct / total,
        "mean_residual_energy_inlier": residual_inlier / inlier_count,
        "mean_residual_energy_outlier": residual_outlier / outlier_count,
        "residual_separation_ratio":
            (residual_outlier / outlier_count) / max(residual_inlier / inlier_count, 1e-9),
        "chance_accuracy": max(cfg.outlier_probability, 1.0 - cfg.outlier_probability),
    }


def run_basis_experiment(cfg: BasisConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n=== EXPERIMENT 2: SHARED BASIS COMMUNICATION ===", flush=True)
    model = SharedBasisNet(
        cfg.input_dim, cfg.subspace_rank, cfg.hidden, cfg.eigh_grad
    ).to(device)
    control = IndependentPointControl(cfg.input_dim, cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    started = time.time()

    model.train()
    control.train()
    for step in range(1, cfg.train_steps + 1):
        points, labels = make_subspace_set(
            cfg.batch_size,
            cfg.set_size,
            cfg.input_dim,
            cfg.subspace_rank,
            cfg.inlier_noise,
            cfg.outlier_probability,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits_steps, diagnostics = model(points, cfg.iterations)
        losses = [F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1)) for logits in logits_steps]
        weights = torch.linspace(0.5, 1.0, cfg.iterations, device=device)
        loss = sum(weight * item for weight, item in zip(weights, losses)) / weights.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        control_optimizer.zero_grad(set_to_none=True)
        control_logits = control(points)
        control_loss = F.cross_entropy(control_logits.reshape(-1, 2), labels.reshape(-1))
        control_loss.backward()
        control_optimizer.step()

        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            acc = accuracy(logits_steps[-1], labels)
            ctrl_acc = accuracy(control_logits, labels)
            residual = diagnostics[-1]["residual_energy"].squeeze(-1)
            rin = residual[labels.eq(0)].mean().item()
            rout = residual[labels.eq(1)].mean().item()
            elapsed = time.time() - started
            print(
                f"[basis]  step {step:5d}/{cfg.train_steps} "
                f"loss={loss.item():.4f} acc={acc:.3f} control={ctrl_acc:.3f} "
                f"res(out/in)={rout / max(rin, 1e-9):.2f} time={elapsed:.1f}s",
                flush=True,
            )

    result = evaluate_basis(model, control, cfg, device)
    print(json.dumps(result, indent=2), flush=True)
    state = {"model": model.state_dict(), "control": control.state_dict(), "config": asdict(cfg)}
    return result, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("factor", "basis", "all"), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Very short correctness run")
    parser.add_argument("--output-dir", type=Path, default=Path("two_architecture_runs"))
    parser.add_argument("--factor-steps", type=int, default=None)
    parser.add_argument("--basis-steps", type=int, default=None)
    parser.add_argument("--eigh-grad", action="store_true", help="Backpropagate through eigenvectors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    factor_cfg = FactorConfig()
    basis_cfg = BasisConfig(eigh_grad=args.eigh_grad)
    if args.factor_steps is not None:
        factor_cfg.train_steps = args.factor_steps
    if args.basis_steps is not None:
        basis_cfg.train_steps = args.basis_steps
    if args.smoke:
        factor_cfg.train_steps = 12
        factor_cfg.batch_size = 12
        factor_cfg.hidden = 32
        factor_cfg.train_iterations = 4
        factor_cfg.test_iterations = 6
        factor_cfg.log_every = 4
        basis_cfg.train_steps = 12
        basis_cfg.batch_size = 8
        basis_cfg.hidden = 32
        basis_cfg.iterations = 2
        basis_cfg.log_every = 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} seed={args.seed} smoke={args.smoke}", flush=True)
    results: dict[str, Any] = {
        "device": str(device),
        "seed": args.seed,
        "smoke": args.smoke,
    }

    if args.experiment in ("factor", "all"):
        factor_result, factor_state = run_factor_experiment(factor_cfg, device)
        results["factor"] = factor_result
        torch.save(factor_state, args.output_dir / "factor_model.pt")

    if args.experiment in ("basis", "all"):
        basis_result, basis_state = run_basis_experiment(basis_cfg, device)
        results["basis"] = basis_result
        torch.save(basis_state, args.output_dir / "basis_model.pt")

    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
