#!/usr/bin/env python3
"""
Experiment v3-B: Reliable Event-Driven Violation Constraint Field (R-VDCF).

The model reconstructs categorical sequences from sparse anchors and modular
constraints.  Its factor graph is O(N): local, dilated, and deterministic hash
edges.  The reliable scheduler adds:

  * persistent factor trust;
  * a learned reliability gate trained only from runtime factor features;
  * capped correction influence;
  * priority aging and a small exploration quota;
  * cached violations refreshed only near executed factors.

The implementation distinguishes expensive learned factor evaluations from
violation refreshes and from the scalar top-k scan.  It compares reliable
sparse execution, naive sparse execution, and dense execution using the same
trained correction rule.

Examples:
  python reliable_vdcf_experiment.py --smoke --device cuda
  python reliable_vdcf_experiment.py --device cuda
  python reliable_vdcf_experiment.py --steps 4000 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


@dataclass
class Config:
    classes: int = 11
    train_nodes: int = 64
    eval_nodes: tuple[int, ...] = (64, 256, 1024)
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    hash_edges_per_node: int = 1
    hidden: int = 128
    active_fraction: float = 0.10
    exploration_fraction: float = 0.15
    observed_fraction: float = 0.06
    train_iterations: int = 24
    eval_iterations: int = 48
    anchor_strength: float = 9.0
    max_train_noise: float = 0.15
    influence_cap: float = 0.20
    aging_weight: float = 0.03
    reliability_loss_weight: float = 0.10
    batch_size: int = 64
    eval_batch_size: int = 8
    eval_batches: int = 4
    steps: int = 3000
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    log_every: int = 100


@dataclass
class Topology:
    edge_u: torch.Tensor
    edge_v: torch.Tensor
    neighbors: torch.Tensor
    nodes: int

    @property
    def edges(self) -> int:
        return self.edge_u.numel()

    @property
    def max_neighbors(self) -> int:
        return self.neighbors.shape[1]


def make_sparse_topology(cfg: Config, nodes: int, device: torch.device) -> Topology:
    edges: set[tuple[int, int]] = set()
    for dilation in cfg.dilations:
        for left in range(nodes - dilation):
            edges.add((left, left + dilation))
    # Deterministic content-independent long edges.  No pairwise token search.
    for route in range(cfg.hash_edges_per_node):
        multiplier = 17 + 2 * route
        offset = 13 + 11 * route
        for left in range(nodes):
            right = (multiplier * left + offset) % nodes
            if left != right:
                edges.add((min(left, right), max(left, right)))

    ordered = sorted(edges)
    edge_u_cpu = torch.tensor([edge[0] for edge in ordered], dtype=torch.long)
    edge_v_cpu = torch.tensor([edge[1] for edge in ordered], dtype=torch.long)
    incident: list[list[int]] = [[] for _ in range(nodes)]
    for edge_index, (left, right) in enumerate(ordered):
        incident[left].append(edge_index)
        incident[right].append(edge_index)

    neighbor_lists: list[list[int]] = []
    max_neighbors = 1
    for edge_index, (left, right) in enumerate(ordered):
        neighborhood = sorted(set(incident[left] + incident[right] + [edge_index]))
        neighbor_lists.append(neighborhood)
        max_neighbors = max(max_neighbors, len(neighborhood))
    padded = []
    for edge_index, neighborhood in enumerate(neighbor_lists):
        padded.append(neighborhood + [edge_index] * (max_neighbors - len(neighborhood)))
    neighbors_cpu = torch.tensor(padded, dtype=torch.long)
    return Topology(
        edge_u=edge_u_cpu.to(device),
        edge_v=edge_v_cpu.to(device),
        neighbors=neighbors_cpu.to(device),
        nodes=nodes,
    )


def make_constraint_batch(
    cfg: Config,
    topology: Topology,
    batch_size: int,
    relation_noise: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    labels = torch.randint(cfg.classes, (batch_size, topology.nodes), device=device)
    left_value = labels[:, topology.edge_u]
    right_value = labels[:, topology.edge_v]
    clean_relation = (right_value - left_value) % cfg.classes
    corrupted = torch.rand(batch_size, topology.edges, device=device) < relation_noise
    nonzero_error = torch.randint(
        1, cfg.classes, (batch_size, topology.edges), device=device
    )
    relation = torch.where(
        corrupted, (clean_relation + nonzero_error) % cfg.classes, clean_relation
    )
    observed = torch.rand(batch_size, topology.nodes, device=device) < cfg.observed_fraction
    observed[:, 0] = True
    # Guarantee at least two anchors when the sequence permits it.
    if topology.nodes > 2:
        second = torch.randint(1, topology.nodes, (batch_size, 1), device=device)
        observed.scatter_(1, second, True)
    return {
        "labels": labels,
        "relation": relation,
        "clean_relation": clean_relation,
        "corrupted": corrupted,
        "observed": observed,
    }


def shift_distribution(probability: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    classes = probability.shape[-1]
    target = torch.arange(classes, device=probability.device).view(1, 1, classes)
    source = (target - relation.unsqueeze(-1)) % classes
    return probability.gather(-1, source.expand_as(probability))


def inverse_shift_distribution(probability: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    classes = probability.shape[-1]
    target = torch.arange(classes, device=probability.device).view(1, 1, classes)
    source = (target + relation.unsqueeze(-1)) % classes
    return probability.gather(-1, source.expand_as(probability))


def gather_nodes(state: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    return state.gather(1, node_index.unsqueeze(-1).expand(-1, -1, state.shape[-1]))


def scatter_messages(
    messages: torch.Tensor,
    node_index: torch.Tensor,
    nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, factors, width = messages.shape
    offsets = torch.arange(batch, device=messages.device).unsqueeze(1) * nodes
    flat_index = (node_index + offsets).reshape(-1)
    summed = torch.zeros(batch * nodes, width, device=messages.device, dtype=messages.dtype)
    summed.index_add_(0, flat_index, messages.reshape(-1, width))
    counts = torch.zeros(batch * nodes, 1, device=messages.device, dtype=messages.dtype)
    counts.index_add_(
        0,
        flat_index,
        torch.ones(batch * factors, 1, device=messages.device, dtype=messages.dtype),
    )
    return summed.view(batch, nodes, width), counts.view(batch, nodes, 1)


class ReliableViolationField(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        factor_input = 5 * cfg.classes + 1
        self.factor = nn.Sequential(
            nn.Linear(factor_input, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 2 * cfg.classes),
        )
        self.reliability = nn.Sequential(
            nn.Linear(5, cfg.hidden // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden // 2, cfg.hidden // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden // 2, 1),
        )
        self.step_logit = nn.Parameter(torch.tensor(-0.25))

    def anchor_logits(self, labels: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*labels.shape, self.cfg.classes),
            -self.cfg.anchor_strength,
            device=labels.device,
            dtype=torch.float32,
        )
        logits.scatter_(-1, labels.unsqueeze(-1), self.cfg.anchor_strength)
        return logits

    def edge_data(
        self,
        probability: torch.Tensor,
        topology: Topology,
        relation: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        left_index = topology.edge_u[edge_index]
        right_index = topology.edge_v[edge_index]
        selected_relation = relation.gather(1, edge_index)
        left_probability = gather_nodes(probability, left_index)
        right_probability = gather_nodes(probability, right_index)
        implied_right = shift_distribution(left_probability, selected_relation)
        implied_left = inverse_shift_distribution(right_probability, selected_relation)
        violation = 0.5 * (
            (left_probability - implied_left).abs().mean(dim=-1)
            + (right_probability - implied_right).abs().mean(dim=-1)
        )
        return (
            left_index,
            right_index,
            selected_relation,
            left_probability,
            right_probability,
            implied_left,
            implied_right,
            violation,
        )

    def all_violations(
        self,
        probability: torch.Tensor,
        topology: Topology,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = torch.arange(topology.edges, device=probability.device)
        edge_index = edge_index.unsqueeze(0).expand(probability.shape[0], -1)
        return self.edge_data(probability, topology, relation, edge_index)[-1]

    def choose_active(
        self,
        cached_violation: torch.Tensor,
        trust: torch.Tensor,
        last_executed: torch.Tensor,
        iteration: int,
        mode: str,
        active_fraction: float,
        exploration_fraction: float,
    ) -> torch.Tensor:
        batch, edges = cached_violation.shape
        if mode == "dense":
            return torch.arange(edges, device=cached_violation.device).unsqueeze(0).expand(batch, -1)
        active_count = max(1, min(edges, math.ceil(active_fraction * edges)))
        if mode == "naive_sparse":
            return cached_violation.topk(active_count, dim=1, sorted=False).indices

        explore_count = min(active_count - 1, max(1, round(active_count * exploration_fraction)))
        main_count = active_count - explore_count
        age = (iteration - last_executed).clamp_min(0).to(cached_violation.dtype)
        normalized_age = age / max(iteration + 1, 1)
        score = cached_violation * trust + self.cfg.aging_weight * normalized_age
        main = score.topk(main_count, dim=1, sorted=False).indices
        random_score = torch.rand_like(score)
        random_score.scatter_(1, main, -1.0)
        explore = random_score.topk(explore_count, dim=1, sorted=False).indices
        return torch.cat((main, explore), dim=1)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        topology: Topology,
        iterations: int,
        mode: str = "reliable_sparse",
        active_fraction: float | None = None,
        exploration_fraction: float | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any], torch.Tensor]:
        cfg = self.cfg
        active_fraction = cfg.active_fraction if active_fraction is None else active_fraction
        exploration_fraction = (
            cfg.exploration_fraction
            if exploration_fraction is None
            else exploration_fraction
        )
        labels = graph["labels"]
        relation = graph["relation"]
        observed = graph["observed"]
        batch, nodes = labels.shape
        anchors = self.anchor_logits(labels)
        state = torch.zeros(batch, nodes, cfg.classes, device=labels.device)
        state = torch.where(observed.unsqueeze(-1), anchors, state)

        probability = state.softmax(dim=-1)
        cached_violation = self.all_violations(probability, topology, relation).detach()
        trust = torch.ones_like(cached_violation)
        progress_ema = torch.zeros_like(cached_violation)
        execution_count = torch.zeros_like(cached_violation)
        last_executed = torch.full_like(cached_violation, -1, dtype=torch.long)
        eta = torch.sigmoid(self.step_logit)
        checkpoints: list[torch.Tensor] = []
        reliability_losses: list[torch.Tensor] = []
        learned_evals = 0
        violation_evals = topology.edges
        scalar_priority_scans = 0
        active_fraction_observed = []

        for iteration in range(iterations):
            probability = state.softmax(dim=-1)
            active = self.choose_active(
                cached_violation,
                trust,
                last_executed,
                iteration,
                mode,
                active_fraction,
                exploration_fraction,
            )
            scalar_priority_scans += topology.edges
            active_fraction_observed.append(active.shape[1] / topology.edges)
            (
                left_index,
                right_index,
                selected_relation,
                left_probability,
                right_probability,
                implied_left,
                implied_right,
                violation,
            ) = self.edge_data(probability, topology, relation, active)

            relation_one_hot = F.one_hot(selected_relation, cfg.classes).to(probability.dtype)
            factor_input = torch.cat(
                (
                    left_probability,
                    right_probability,
                    implied_left,
                    implied_right,
                    relation_one_hot,
                    violation.unsqueeze(-1),
                ),
                dim=-1,
            )
            correction_left, correction_right = self.factor(factor_input).chunk(2, dim=-1)
            learned_evals += active.shape[1]

            selected_trust = trust.gather(1, active)
            selected_progress = progress_ema.gather(1, active)
            selected_executions = execution_count.gather(1, active)
            selected_last = last_executed.gather(1, active)
            age = (iteration - selected_last).clamp_min(0).to(probability.dtype)
            reliability_features = torch.stack(
                (
                    violation,
                    selected_trust,
                    age / max(iteration + 1, 1),
                    selected_progress,
                    selected_executions / max(iteration + 1, 1),
                ),
                dim=-1,
            )
            reliability_logit = self.reliability(reliability_features).squeeze(-1)
            learned_gate = torch.sigmoid(reliability_logit)
            if mode == "reliable_sparse":
                gate = learned_gate * selected_trust
                clean_target = (~graph["corrupted"].gather(1, active)).to(gate.dtype)
                reliability_losses.append(
                    F.binary_cross_entropy_with_logits(reliability_logit, clean_target)
                )
            else:
                gate = torch.ones_like(violation)

            capped = cfg.influence_cap * torch.tanh(violation / cfg.influence_cap)
            weight = gate.unsqueeze(-1) * capped.unsqueeze(-1)
            correction_left = torch.tanh(correction_left) * weight
            correction_right = torch.tanh(correction_right) * weight
            sum_left, count_left = scatter_messages(correction_left, left_index, nodes)
            sum_right, count_right = scatter_messages(correction_right, right_index, nodes)
            correction = (sum_left + sum_right) / (count_left + count_right).clamp_min(1.0)
            state = state + eta * correction
            state = torch.where(observed.unsqueeze(-1), anchors, state)

            # Measure actual progress of executed factors and update persistent trust.
            new_probability = state.softmax(dim=-1)
            new_active_violation = self.edge_data(
                new_probability, topology, relation, active
            )[-1].detach()
            violation_evals += active.shape[1]
            progress = (violation.detach() - new_active_violation).clamp(-1.0, 1.0)
            old_progress = progress_ema.gather(1, active)
            new_progress = 0.8 * old_progress + 0.2 * progress
            progress_ema = progress_ema.scatter(1, active, new_progress)
            if mode == "reliable_sparse":
                progress_target = torch.sigmoid(40.0 * (new_progress - 0.002))
                trust_target = 0.5 * learned_gate.detach() + 0.5 * progress_target
                new_trust = (0.85 * selected_trust + 0.15 * trust_target).clamp(0.05, 1.0)
                trust = trust.scatter(1, active, new_trust)
            execution_count = execution_count.scatter(
                1, active, selected_executions + 1.0
            )
            last_executed = last_executed.scatter(
                1, active, torch.full_like(active, iteration)
            )

            # Refresh only incident cached factors. Duplicates are counted because
            # the prototype really evaluates them; a production heap would dedupe.
            if mode == "dense":
                cached_violation = self.all_violations(
                    new_probability, topology, relation
                ).detach()
                violation_evals += topology.edges
            else:
                candidates = topology.neighbors[active].reshape(batch, -1)
                candidate_violation = self.edge_data(
                    new_probability, topology, relation, candidates
                )[-1].detach()
                cached_violation = cached_violation.scatter(
                    1, candidates, candidate_violation
                )
                violation_evals += candidates.shape[1]

            if iteration == 0 or iteration + 1 == max(1, iterations // 2) or iteration + 1 == iterations:
                checkpoints.append(state)

        reliability_loss = (
            torch.stack(reliability_losses).mean()
            if reliability_losses
            else state.new_zeros(())
        )
        if self.training:
            # Avoid a device synchronization on every training step.
            trust_clean = float("nan")
            trust_corrupt = float("nan")
        else:
            clean = ~graph["corrupted"]
            trust_clean = trust[clean].mean().item() if clean.any() else float("nan")
            corrupt = graph["corrupted"]
            trust_corrupt = trust[corrupt].mean().item() if corrupt.any() else float("nan")
        diagnostics = {
            "mode": mode,
            "edges": topology.edges,
            "max_neighbors": topology.max_neighbors,
            "mean_active_fraction": sum(active_fraction_observed) / len(active_fraction_observed),
            "learned_factor_evals_per_example": learned_evals,
            "violation_evals_per_example": violation_evals,
            "scalar_priority_scans_per_example": scalar_priority_scans,
            "dense_learned_eval_equivalent": topology.edges * iterations,
            "learned_eval_fraction_of_dense": learned_evals / (topology.edges * iterations),
            "mean_trust_clean": trust_clean,
            "mean_trust_corrupt": trust_corrupt,
        }
        return checkpoints, diagnostics, reliability_loss


def reconstruction_loss(
    checkpoints: list[torch.Tensor],
    graph: dict[str, torch.Tensor],
) -> torch.Tensor:
    unknown = ~graph["observed"]
    weights = torch.linspace(0.3, 1.0, len(checkpoints), device=unknown.device)
    losses = []
    for weight, state in zip(weights, checkpoints):
        cross_entropy = F.cross_entropy(
            state.transpose(1, 2), graph["labels"], reduction="none"
        )
        losses.append(weight * masked_mean(cross_entropy, unknown))
    return torch.stack(losses).sum() / weights.sum()


@torch.no_grad()
def evaluate_case(
    model: ReliableViolationField,
    cfg: Config,
    topology: Topology,
    device: torch.device,
    noise: float,
    mode: str,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total_unknown = 0
    clean_edge_correct = 0
    clean_edge_total = 0
    diagnostics_sum: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(cfg.eval_batches):
        graph = make_constraint_batch(
            cfg, topology, cfg.eval_batch_size, noise, device
        )
        with autocast_context(device, amp):
            checkpoints, diagnostics, _ = model(
                graph, topology, cfg.eval_iterations, mode=mode
            )
        prediction = checkpoints[-1].argmax(dim=-1)
        unknown = ~graph["observed"]
        correct += (prediction.eq(graph["labels"]) & unknown).sum().item()
        total_unknown += unknown.sum().item()
        predicted_left = prediction[:, topology.edge_u]
        predicted_right = prediction[:, topology.edge_v]
        edge_ok = ((predicted_right - predicted_left) % cfg.classes).eq(
            graph["clean_relation"]
        )
        clean_mask = ~graph["corrupted"]
        clean_edge_correct += (edge_ok & clean_mask).sum().item()
        clean_edge_total += clean_mask.sum().item()
        for key, value in diagnostics.items():
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value)
    synchronize(device)
    elapsed = time.perf_counter() - started
    averaged = {key: value / cfg.eval_batches for key, value in diagnostics_sum.items()}
    return {
        "nodes": topology.nodes,
        "edges": topology.edges,
        "noise": noise,
        "mode": mode,
        "unknown_accuracy": correct / total_unknown,
        "clean_edge_accuracy": clean_edge_correct / clean_edge_total,
        "seconds": elapsed,
        "examples_per_second": cfg.eval_batches * cfg.eval_batch_size / elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "diagnostics": averaged,
        "chance_accuracy": 1.0 / cfg.classes,
    }


def train_model(
    model: ReliableViolationField,
    cfg: Config,
    topology: Topology,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    history = []
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        curriculum = min(1.0, step / max(1, int(0.75 * cfg.steps)))
        noise = cfg.max_train_noise * curriculum
        graph = make_constraint_batch(cfg, topology, cfg.batch_size, noise, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            checkpoints, diagnostics, reliability_loss = model(
                graph,
                topology,
                cfg.train_iterations,
                mode="reliable_sparse",
            )
            task_loss = reconstruction_loss(checkpoints, graph)
            loss = task_loss + cfg.reliability_loss_weight * reliability_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            unknown = ~graph["observed"]
            accuracy = masked_mean(
                checkpoints[-1].argmax(dim=-1).eq(graph["labels"]).float(),
                unknown,
            ).item()
            record = {
                "step": step,
                "noise": noise,
                "loss": loss.detach().item(),
                "task_loss": task_loss.detach().item(),
                "reliability_loss": reliability_loss.detach().item(),
                "unknown_accuracy": accuracy,
                "learned_eval_fraction_of_dense": diagnostics[
                    "learned_eval_fraction_of_dense"
                ],
            }
            history.append(record)
            print(
                f"step {step:5d}/{cfg.steps} noise={noise:.3f} "
                f"loss={record['loss']:.4f} acc={accuracy:.3f} "
                f"rel_loss={record['reliability_loss']:.4f} "
                f"learned/dense={record['learned_eval_fraction_of_dense']:.3f} "
                f"time={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    return {"history": history, "seconds": time.perf_counter() - started}


def parse_nodes(value: str) -> tuple[int, ...]:
    nodes = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not nodes:
        raise argparse.ArgumentTypeError("at least one evaluation size is required")
    return nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-nodes", type=parse_nodes, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reliable_vdcf_v3_runs"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_device(args.device)
    cfg = Config()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.eval_nodes is not None:
        cfg.eval_nodes = args.eval_nodes
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.smoke:
        cfg.train_nodes = 24
        cfg.eval_nodes = (24, 48)
        cfg.dilations = (1, 2, 4)
        cfg.hidden = 32
        cfg.active_fraction = 0.20
        cfg.train_iterations = 5
        cfg.eval_iterations = 7
        cfg.batch_size = 4
        cfg.eval_batch_size = 4
        cfg.eval_batches = 1
        cfg.steps = 6
        cfg.log_every = 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_topology = make_sparse_topology(cfg, cfg.train_nodes, device)
    model = ReliableViolationField(cfg).to(device)
    print(
        f"device={device} seed={args.seed} amp={args.amp} "
        f"train_nodes={cfg.train_nodes} train_edges={train_topology.edges}",
        flush=True,
    )
    print(
        "factor topology is O(N); learned correction work is O(rho*E); "
        "the prototype still performs an O(E) scalar top-k scan",
        flush=True,
    )
    training = train_model(model, cfg, train_topology, device, args.amp)

    cases = []
    for nodes in cfg.eval_nodes:
        topology = make_sparse_topology(cfg, nodes, device)
        for noise in (0.0, 0.05, 0.10, 0.20):
            for mode in ("reliable_sparse", "naive_sparse", "dense"):
                case = evaluate_case(
                    model, cfg, topology, device, noise, mode, args.amp
                )
                cases.append(case)
                print(
                    f"eval nodes={nodes:4d} noise={noise:.2f} mode={mode:15s} "
                    f"acc={case['unknown_accuracy']:.4f} "
                    f"edge={case['clean_edge_accuracy']:.4f} "
                    f"time={case['seconds']:.2f}s",
                    flush=True,
                )

    result = {
        "config": asdict(cfg),
        "device": str(device),
        "seed": args.seed,
        "amp": args.amp,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": training,
        "cases": cases,
    }
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg)},
        args.output_dir / "reliable_vdcf_v3.pt",
    )
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
