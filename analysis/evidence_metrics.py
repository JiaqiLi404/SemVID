#!/usr/bin/env python3
"""Compute and summarize Evidence Retention (ER) and Connectivity Strength (CS).

The tensor functions implement Eqs. 11-15 in the SemVID paper and are called by
the Qwen evaluator during evidence analysis. The command-line interface aggregates
the resulting ``attn_metrics.json`` files after evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _row_normalize(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return weights / denominator


def compute_landing_distribution(
    query_visual_seed: torch.Tensor,
    retained_indices: torch.Tensor,
    attention_indices: torch.Tensor,
    attention_weights: torch.Tensor,
    residual_weight: float = 0.1,
    fill_self_loop: bool = True,
) -> torch.Tensor:
    """Propagate query evidence backward through sparse visual attention (Eq. 14)."""
    if query_visual_seed.dim() != 2:
        raise ValueError("query_visual_seed must have shape [frames, patches].")

    frames, patches = query_visual_seed.shape
    total_tokens = int(frames * patches)
    distribution = query_visual_seed.reshape(-1).float()
    distribution = distribution / (distribution.sum() + 1e-12)

    retained_indices = retained_indices.to(device=distribution.device)
    layers = int(attention_indices.size(0))
    retained_count = int(retained_indices.numel())

    for layer in range(layers - 1, -1, -1):
        indices = attention_indices[layer]
        weights = _row_normalize(torch.clamp(attention_weights[layer].float(), min=0.0))
        next_distribution = distribution.clone()
        if retained_count == 0:
            distribution = next_distribution
            continue

        source = retained_indices.clamp(min=0, max=total_tokens - 1)
        next_distribution[source] -= distribution[source]
        next_distribution[source] += distribution[source] * float(residual_weight)
        routed_mass = distribution[source] * float(1.0 - residual_weight)

        valid = indices.ge(0)
        safe_indices = indices.clone()
        safe_indices[~valid] = 0
        contribution = routed_mass.unsqueeze(-1) * weights * valid.float()
        next_distribution.scatter_add_(0, safe_indices.reshape(-1), contribution.reshape(-1))

        if fill_self_loop:
            row_sum = (weights * valid.float()).sum(dim=-1)
            leftover = routed_mass * (1.0 - row_sum).clamp_min(0.0)
            next_distribution.scatter_add_(0, source, leftover)

        distribution = next_distribution / (next_distribution.sum() + 1e-12)

    return distribution.reshape(frames, patches)


def compute_connectivity_strength(
    retained_indices: torch.Tensor,
    attention_indices: torch.Tensor,
    attention_weights: torch.Tensor,
    frames: int,
    patches: int,
) -> tuple[float, list[float], list[list[float]]]:
    """Compute adjacent-frame retained attention mass per layer (Eqs. 12, 13, 15)."""
    total_tokens = int(frames * patches)
    retained = retained_indices.clamp(min=0, max=total_tokens - 1)
    retained_set = set(retained.detach().cpu().tolist())
    retained_frames = (retained // patches).detach().cpu().tolist()

    layer_scores: list[float] = []
    layer_transitions: list[list[float]] = []
    for layer in range(int(attention_indices.size(0))):
        indices = attention_indices[layer]
        weights = _row_normalize(torch.clamp(attention_weights[layer].float(), min=0.0))
        transition_scores = [0.0 for _ in range(max(frames - 1, 0))]

        indices_cpu = indices.detach().cpu().numpy()
        weights_cpu = weights.detach().cpu().numpy()
        for source_index, source_frame in enumerate(retained_frames):
            if source_frame < 0 or source_frame >= frames - 1:
                continue
            for neighbor_index in range(indices_cpu.shape[1]):
                destination = int(indices_cpu[source_index, neighbor_index])
                if destination < 0 or destination not in retained_set:
                    continue
                if int(destination // patches) == source_frame + 1:
                    transition_scores[source_frame] += float(weights_cpu[source_index, neighbor_index])

        layer_scores.append(float(sum(transition_scores)))
        layer_transitions.append(transition_scores)

    mean_score = float(sum(layer_scores) / max(len(layer_scores), 1))
    return mean_score, layer_scores, layer_transitions


def compute_evidence_cross_entropy(
    full_landing_distribution: torch.Tensor,
    pruned_landing_distribution: torch.Tensor,
    retained_indices: torch.Tensor,
) -> float:
    """Compute the ER cross-entropy term over retained and removed tokens (Eq. 11)."""
    frames, patches = full_landing_distribution.shape
    total_tokens = int(frames * patches)
    full_distribution = full_landing_distribution.reshape(-1).float()
    full_distribution = full_distribution / (full_distribution.sum() + 1e-12)
    pruned_distribution = pruned_landing_distribution.reshape(-1).float()

    retained_indices = retained_indices.clamp(min=0, max=total_tokens - 1)
    retained_mask = torch.zeros((total_tokens,), device=full_distribution.device, dtype=torch.bool)
    if retained_indices.numel() > 0:
        retained_mask[retained_indices] = True

    retained_mass = float(full_distribution[retained_mask].sum().item())
    retained_distribution = pruned_distribution[retained_mask]
    retained_distribution = retained_distribution / (retained_distribution.sum() + 1e-12)

    removed_count = int((~retained_mask).sum().item())
    if removed_count <= 0:
        completed_distribution = torch.zeros_like(full_distribution)
        completed_distribution[retained_mask] = retained_distribution
    else:
        completed_distribution = torch.full_like(
            full_distribution,
            fill_value=(1.0 - retained_mass) / float(removed_count),
        )
        completed_distribution[retained_mask] = retained_mass * retained_distribution

    return float(
        -(full_distribution * torch.log(completed_distribution.clamp_min(1e-12))).sum().item()
    )


def evidence_retention_score(cross_entropy: float, scale: float = 10_000.0) -> float:
    """Convert Eq. 11 cross-entropy to the positive ER score reported in the paper."""
    return math.exp(-float(cross_entropy)) * float(scale)


def summarize_metric_files(root: Path) -> dict[str, Any]:
    metric_files = sorted(root.rglob("attn_metrics.json"))
    er_scores: list[float] = []
    cs_scores: list[float] = []

    for path in metric_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ER") is not None:
            er_scores.append(evidence_retention_score(record["ER"]))
        if record.get("CS_mean") is not None:
            cs_scores.append(float(record["CS_mean"]))

    return {
        "metric_files": len(metric_files),
        "er_samples": len(er_scores),
        "cs_samples": len(cs_scores),
        "ER": sum(er_scores) / len(er_scores) if er_scores else None,
        "CS": sum(cs_scores) / len(cs_scores) if cs_scores else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_root", type=Path, help="directory containing attn_metrics.json files")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    if not args.metrics_root.is_dir():
        parser.error(f"metrics directory does not exist: {args.metrics_root}")
    summary = summarize_metric_files(args.metrics_root)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if summary["metric_files"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
