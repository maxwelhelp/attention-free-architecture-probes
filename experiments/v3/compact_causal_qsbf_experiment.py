#!/usr/bin/env python3
"""
Experiment v3-A: Compact Causal Quadratic Shared-Basis Field (CQSBF).

This is a language-like test of an attention-free token mixer.  It replaces the
exact vech(xx^T) lift and repeated eigendecomposition from v2 with:

  * a learned compact quadratic sketch phi_j(x)=(a_j^T x)(b_j^T x);
  * a fixed-rank causal shared basis updated online with an Oja-style rule;
  * one gated read/update per block;
  * sinusoidal positions and a prefix-only state suitable for generation.

No N x N token matrix is constructed by CQSBF.  The script trains CQSBF,
ordinary causal self-attention, and causal linear attention on the same batches
of associative recall, selective copy, and induction tasks.  It reports
accuracy, wall time, peak CUDA memory, parameter counts, and length transfer.

Examples:
  python compact_causal_qsbf_experiment.py --smoke --device cuda
  python compact_causal_qsbf_experiment.py --models all --device cuda
  python compact_causal_qsbf_experiment.py --models cqsbf --steps 4000 --amp
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


TASK_ASSOC = 0
TASK_COPY = 1
TASK_INDUCTION = 2
TASK_NAMES = ("associative_recall", "selective_copy", "induction")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@dataclass
class Config:
    num_keys: int = 32
    num_values: int = 48
    max_copy_items: int = 8
    pair_count: int = 6
    train_length: int = 64
    eval_lengths: tuple[int, ...] = (64, 128, 256, 512)
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    ff_mult: int = 3
    compact_width: int = 32
    sketch_width: int = 96
    basis_rank: int = 12
    linear_feature_width: int = 48
    batch_size: int = 64
    eval_batch_size: int = 32
    steps: int = 3000
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    eval_batches: int = 12
    log_every: int = 100
    benchmark_repeats: int = 8

    @property
    def task_token_start(self) -> int:
        return 1

    @property
    def mark_token(self) -> int:
        return 4

    @property
    def query_token(self) -> int:
        return 5

    @property
    def key_start(self) -> int:
        return 8

    @property
    def value_start(self) -> int:
        return self.key_start + self.num_keys

    @property
    def index_start(self) -> int:
        return self.value_start + self.num_values

    @property
    def vocab_size(self) -> int:
        return self.index_start + self.max_copy_items


def validate_config(cfg: Config) -> None:
    required = max(2 + 2 * cfg.pair_count, 2 + 3 * cfg.pair_count, 8)
    if cfg.train_length < required:
        raise ValueError(f"train_length={cfg.train_length} must be >= {required}")
    if cfg.pair_count > cfg.max_copy_items:
        raise ValueError("pair_count cannot exceed max_copy_items")
    if cfg.pair_count > cfg.num_keys:
        raise ValueError("pair_count cannot exceed num_keys")
    if cfg.d_model % cfg.heads:
        raise ValueError("d_model must be divisible by heads")
    if cfg.basis_rank > cfg.sketch_width:
        raise ValueError("basis_rank cannot exceed sketch_width")


def _task_rows(task: torch.Tensor, value: int) -> torch.Tensor:
    return task.eq(value).nonzero(as_tuple=False).flatten()


def make_language_batch(
    cfg: Config,
    batch_size: int,
    length: int,
    device: torch.device,
    forced_task: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate fixed-length prefix classification examples entirely on device."""
    if length < max(2 + 3 * cfg.pair_count, 8):
        raise ValueError("sequence is too short for configured tasks")

    tokens = torch.randint(
        cfg.key_start,
        cfg.key_start + cfg.num_keys,
        (batch_size, length),
        device=device,
    )
    if forced_task is None:
        task = torch.randint(0, len(TASK_NAMES), (batch_size,), device=device)
    else:
        task = torch.full((batch_size,), forced_task, device=device, dtype=torch.long)
    target = torch.empty(batch_size, device=device, dtype=torch.long)
    tokens[:, 0] = cfg.task_token_start + task

    # Associative recall: [key,value] pairs followed by [QUERY,key].
    rows = _task_rows(task, TASK_ASSOC)
    if rows.numel():
        count = rows.numel()
        keys = torch.rand(count, cfg.num_keys, device=device).argsort(dim=1)
        keys = keys[:, : cfg.pair_count] + cfg.key_start
        values = torch.randint(
            cfg.value_start,
            cfg.value_start + cfg.num_values,
            (count, cfg.pair_count),
            device=device,
        )
        pair_positions = 2 + 2 * torch.arange(cfg.pair_count, device=device)
        tokens[rows[:, None], pair_positions[None, :]] = keys
        tokens[rows[:, None], (pair_positions + 1)[None, :]] = values
        selected = torch.randint(cfg.pair_count, (count,), device=device)
        batch_index = torch.arange(count, device=device)
        tokens[rows, -2] = cfg.query_token
        tokens[rows, -1] = keys[batch_index, selected]
        target[rows] = values[batch_index, selected]

    # Selective copy: marked values followed by [QUERY,index-token].
    rows = _task_rows(task, TASK_COPY)
    if rows.numel():
        count = rows.numel()
        values = torch.randint(
            cfg.value_start,
            cfg.value_start + cfg.num_values,
            (count, cfg.pair_count),
            device=device,
        )
        mark_positions = 2 + 3 * torch.arange(cfg.pair_count, device=device)
        tokens[rows[:, None], mark_positions[None, :]] = cfg.mark_token
        tokens[rows[:, None], (mark_positions + 1)[None, :]] = values
        selected = torch.randint(cfg.pair_count, (count,), device=device)
        batch_index = torch.arange(count, device=device)
        tokens[rows, -2] = cfg.query_token
        tokens[rows, -1] = cfg.index_start + selected
        target[rows] = values[batch_index, selected]

    # Induction: an earlier [A,B] occurrence and a final A; predict B.
    rows = _task_rows(task, TASK_INDUCTION)
    if rows.numel():
        count = rows.numel()
        first = torch.randint(cfg.num_values, (count,), device=device)
        offset = torch.randint(1, cfg.num_values, (count,), device=device)
        second = (first + offset) % cfg.num_values
        first = first + cfg.value_start
        second = second + cfg.value_start
        tokens[rows, 3] = first
        tokens[rows, 4] = second
        tokens[rows, -2] = cfg.query_token
        tokens[rows, -1] = first
        target[rows] = second

    return tokens, target, task


def sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = width // 2
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    denominator = max(half - 1, 1)
    frequency = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / denominator
    )
    angle = position * frequency.unsqueeze(0)
    result = torch.cat((angle.sin(), angle.cos()), dim=1)
    if result.shape[1] < width:
        result = F.pad(result, (0, width - result.shape[1]))
    return result.to(dtype=dtype)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale.to(x.dtype) * self.weight.to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, width: int, multiplier: int) -> None:
        super().__init__()
        hidden = width * multiplier
        self.up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(value * F.silu(gate))


class CompactQuadraticSketch(nn.Module):
    """Learned low-dimensional polynomial feature map without materializing xx^T."""

    def __init__(self, width: int, compact_width: int, sketch_width: int) -> None:
        super().__init__()
        self.pre = nn.Linear(width, compact_width)
        self.left = nn.Linear(compact_width, sketch_width, bias=False)
        self.right = nn.Linear(compact_width, sketch_width, bias=False)
        self.scale = sketch_width ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact = torch.tanh(self.pre(x))
        left = self.left(compact)
        right = self.right(compact)
        return torch.tanh(left * right * self.scale)


class CompactCausalBasisMixer(nn.Module):
    """Fixed-state causal communication through an online low-rank basis."""

    def __init__(
        self,
        width: int,
        compact_width: int,
        sketch_width: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.sketch = CompactQuadraticSketch(width, compact_width, sketch_width)
        self.rank = rank
        initial = torch.randn(sketch_width, rank)
        initial = torch.linalg.qr(initial, mode="reduced").Q
        self.initial_basis = nn.Parameter(initial)
        self.decay_logit = nn.Parameter(torch.tensor(2.7))
        self.step_logit = nn.Parameter(torch.tensor(-2.5))
        read_width = 2 * rank + 2
        self.readout = nn.Sequential(
            nn.Linear(read_width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.gate = nn.Linear(width + read_width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.sketch(x)
        batch, length, sketch_width = phi.shape
        basis = F.normalize(self.initial_basis, dim=0).to(phi.dtype)
        basis = basis.unsqueeze(0).expand(batch, -1, -1)
        spectrum = torch.zeros(batch, self.rank, device=x.device, dtype=phi.dtype)
        decay = torch.sigmoid(self.decay_logit).to(phi.dtype)
        step = F.softplus(self.step_logit).to(phi.dtype)
        outputs: list[torch.Tensor] = []

        for index in range(length):
            current = phi[:, index]
            coefficient = torch.einsum("br,brk->bk", current, basis)
            projected = torch.einsum("brk,bk->br", basis, coefficient)
            residual = current - projected

            # Online Oja-like subspace update.  State is r*k, independent of N.
            update = residual.unsqueeze(-1) * torch.tanh(coefficient).unsqueeze(-2)
            basis = decay * basis + step * update
            basis = basis / basis.float().square().sum(dim=1, keepdim=True).add(1e-6).sqrt().to(basis.dtype)

            coefficient = torch.einsum("br,brk->bk", current, basis)
            projected = torch.einsum("brk,bk->br", basis, coefficient)
            residual = current - projected
            spectrum = decay * spectrum + (1.0 - decay) * coefficient.square()
            normalized_spectrum = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            projected_energy = projected.float().square().mean(dim=-1, keepdim=True).to(phi.dtype)
            residual_energy = residual.float().square().mean(dim=-1, keepdim=True).to(phi.dtype)
            read = torch.cat(
                (coefficient, normalized_spectrum, projected_energy, residual_energy), dim=-1
            )
            delta = self.readout(read)
            gate = torch.sigmoid(self.gate(torch.cat((x[:, index], read), dim=-1)))
            outputs.append(gate * delta)

        return torch.stack(outputs, dim=1)

    @property
    def recurrent_state_elements(self) -> int:
        return self.initial_basis.numel() + self.rank


class CQSBFBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = CompactCausalBasisMixer(
            cfg.d_model,
            cfg.compact_width,
            cfg.sketch_width,
            cfg.basis_rank,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class CausalAttentionBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attention = nn.MultiheadAttention(
            cfg.d_model, cfg.heads, batch_first=True, dropout=0.0
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        length = x.shape[1]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        mixed, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + mixed
        return x + self.ffn(self.norm2(x))


class CausalLinearMixer(nn.Module):
    def __init__(self, width: int, feature_width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, feature_width, bias=False)
        self.key = nn.Linear(width, feature_width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = F.elu(self.query(x)) + 1.0
        key = F.elu(self.key(x)) + 1.0
        value = self.value(x)
        key_value = key.unsqueeze(-1) * value.unsqueeze(-2)
        prefix_key_value = key_value.cumsum(dim=1)
        prefix_key = key.cumsum(dim=1)
        numerator = torch.einsum("bnf,bnfd->bnd", query, prefix_key_value)
        denominator = (query * prefix_key).sum(dim=-1, keepdim=True).clamp_min(1e-5)
        return self.output(numerator / denominator)


class LinearAttentionBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = CausalLinearMixer(cfg.d_model, cfg.linear_feature_width)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class SequenceModel(nn.Module):
    def __init__(self, cfg: Config, kind: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.kind = kind
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if kind == "cqsbf":
            block_type = CQSBFBlock
        elif kind == "attention":
            block_type = CausalAttentionBlock
        elif kind == "linear":
            block_type = LinearAttentionBlock
        else:
            raise ValueError(f"unknown model kind: {kind}")
        self.blocks = nn.ModuleList(block_type(cfg) for _ in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        positions = sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)
        x = x + positions.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))

    @property
    def fixed_generation_state_elements(self) -> int | None:
        if self.kind != "cqsbf":
            return None
        return sum(block.mixer.recurrent_state_elements for block in self.blocks)


@torch.no_grad()
def evaluate_model(
    model: SequenceModel,
    cfg: Config,
    device: torch.device,
    length: int,
    batches: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    task_correct: dict[str, int] = {name: 0 for name in TASK_NAMES}
    task_total: dict[str, int] = {name: 0 for name in TASK_NAMES}
    loss_sum = 0.0
    examples = 0
    started = time.perf_counter()
    for task_id, task_name in enumerate(TASK_NAMES):
        for _ in range(batches):
            tokens, target, _ = make_language_batch(
                cfg, cfg.eval_batch_size, length, device, forced_task=task_id
            )
            with autocast_context(device, amp):
                logits = model(tokens)[:, -1]
                loss = F.cross_entropy(logits.float(), target)
            prediction = logits.argmax(dim=-1)
            task_correct[task_name] += prediction.eq(target).sum().item()
            task_total[task_name] += target.numel()
            loss_sum += loss.item() * target.numel()
            examples += target.numel()
    synchronize(device)
    accuracy = {name: task_correct[name] / task_total[name] for name in TASK_NAMES}
    return {
        "length": length,
        "accuracy_by_task": accuracy,
        "mean_accuracy": sum(accuracy.values()) / len(accuracy),
        "loss": loss_sum / examples,
        "examples": examples,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def benchmark_model(
    model: SequenceModel,
    cfg: Config,
    device: torch.device,
    length: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    batch = min(cfg.eval_batch_size, 16)
    tokens, _, _ = make_language_batch(cfg, batch, length, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        with autocast_context(device, amp):
            model(tokens)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(cfg.benchmark_repeats):
        with autocast_context(device, amp):
            model(tokens)
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "length": length,
        "batch_size": batch,
        "milliseconds_per_batch": 1000.0 * elapsed / cfg.benchmark_repeats,
        "tokens_per_second": batch * length * cfg.benchmark_repeats / elapsed,
        "peak_cuda_bytes": peak,
        "pair_score_elements_per_layer": batch * cfg.heads * length * length
        if model.kind == "attention"
        else 0,
        "fixed_generation_state_elements": model.fixed_generation_state_elements,
    }


def train_models(
    cfg: Config,
    kinds: list[str],
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, SequenceModel], dict[str, Any]]:
    models = {kind: SequenceModel(cfg, kind).to(device) for kind in kinds}
    optimizers = {
        kind: torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        for kind, model in models.items()
    }
    scalers = {
        kind: torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
        for kind in kinds
    }
    history: dict[str, list[dict[str, float]]] = {kind: [] for kind in kinds}
    started = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        tokens, target, task = make_language_batch(
            cfg, cfg.batch_size, cfg.train_length, device
        )
        log_values: dict[str, tuple[float, float]] = {}
        for kind in kinds:
            model = models[kind]
            optimizer = optimizers[kind]
            scaler = scalers[kind]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                logits = model(tokens)[:, -1]
                loss = F.cross_entropy(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            accuracy = logits.detach().argmax(dim=-1).eq(target).float().mean().item()
            log_values[kind] = (loss.detach().item(), accuracy)

        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            elapsed = time.perf_counter() - started
            task_counts = torch.bincount(task, minlength=len(TASK_NAMES)).tolist()
            line = [f"step {step:5d}/{cfg.steps}"]
            for kind in kinds:
                loss_value, accuracy = log_values[kind]
                history[kind].append(
                    {"step": step, "loss": loss_value, "accuracy": accuracy}
                )
                line.append(f"{kind}: loss={loss_value:.4f} acc={accuracy:.3f}")
            line.append(f"task_counts={task_counts} time={elapsed:.1f}s")
            print(" | ".join(line), flush=True)

    return models, {"history": history, "train_seconds": time.perf_counter() - started}


def parse_lengths(value: str) -> tuple[int, ...]:
    result = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one evaluation length is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("cqsbf", "attention", "linear", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-length", type=int, default=None)
    parser.add_argument("--eval-lengths", type=parse_lengths, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("cqsbf_v3_runs"))
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
    if args.train_length is not None:
        cfg.train_length = args.train_length
    if args.eval_lengths is not None:
        cfg.eval_lengths = args.eval_lengths
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.smoke:
        cfg.pair_count = 3
        cfg.train_length = 24
        cfg.eval_lengths = (24, 48)
        cfg.d_model = 32
        cfg.layers = 1
        cfg.heads = 4
        cfg.ff_mult = 2
        cfg.compact_width = 12
        cfg.sketch_width = 24
        cfg.basis_rank = 4
        cfg.linear_feature_width = 16
        cfg.batch_size = 4
        cfg.eval_batch_size = 4
        cfg.steps = 6
        cfg.eval_batches = 2
        cfg.log_every = 2
        cfg.benchmark_repeats = 2
    validate_config(cfg)

    kinds = ["cqsbf", "attention", "linear"] if args.models == "all" else [args.models]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} seed={args.seed} amp={args.amp} models={kinds} "
        f"train_length={cfg.train_length}",
        flush=True,
    )
    print(
        "complexity: CQSBF O(N*r*k), attention O(N^2*d), "
        "linear baseline O(N*f*d)",
        flush=True,
    )

    models, training = train_models(cfg, kinds, device, args.amp)
    result: dict[str, Any] = {
        "config": asdict(cfg),
        "device": str(device),
        "seed": args.seed,
        "amp": args.amp,
        "training": training,
        "models": {},
    }
    for kind, model in models.items():
        evaluations = []
        benchmarks = []
        for length in cfg.eval_lengths:
            evaluations.append(
                evaluate_model(model, cfg, device, length, cfg.eval_batches, args.amp)
            )
            benchmarks.append(benchmark_model(model, cfg, device, length, args.amp))
        result["models"][kind] = {
            "parameters": parameter_count(model),
            "evaluations": evaluations,
            "benchmarks": benchmarks,
        }
        torch.save(
            {"model": model.state_dict(), "config": asdict(cfg), "kind": kind},
            args.output_dir / f"{kind}.pt",
        )

    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["models"], indent=2), flush=True)
    print(f"Saved results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
