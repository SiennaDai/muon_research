#!/usr/bin/env python3
"""Shared helpers for the first low-precision batch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/mnt/d/muon_research")
SNAP_ROOT = Path("/mnt/d/experiments/structural_emergence_output/structural_emergence")

sys.path.insert(0, str(ROOT / "platform" / "src"))
from optim.muon_reference import zeropower_newton_schulz  # noqa: E402

SEEDS = ("seed0", "seed1")
UPDATES = (128, 1024, 4096, 8192)
LAYERS = tuple(range(6))
PARAMS = (
    ("fused_qkv", "transformer.h.{layer}.attn.c_attn.weight", "rows"),
    ("O", "transformer.h.{layer}.attn.c_proj.weight", "cols"),
)


def ns(x: torch.Tensor) -> torch.Tensor:
    return zeropower_newton_schulz(
        x, steps=5, coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7
    ).float()


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def load_snapshot(seed: str, update: int) -> tuple[dict, dict]:
    snap = SNAP_ROOT / seed / "snapshots" / f"update_{update:06d}"
    manifest = json.loads(snap.with_suffix(".manifest.json").read_text())
    payload = torch.load(snap.with_suffix(".pt"), map_location="cpu",
                         weights_only=False)
    return manifest, payload["tensors"]


def split_heads(x: torch.Tensor, axis: str, head_dim: int = 64):
    """Return [(head_index, block)] for one matrix."""
    if axis == "rows":
        return list(enumerate(x.split(head_dim, dim=0)))
    return list(enumerate(x.split(head_dim, dim=1)))


def head_shares(x: torch.Tensor, axis: str, head_dim: int = 64) -> torch.Tensor:
    energy = torch.tensor(
        [float((b.float() ** 2).sum()) for _, b in split_heads(x, axis, head_dim)]
    )
    return energy / energy.sum().clamp_min(1e-30)


def metrics(ref: torch.Tensor, q: torch.Tensor, axis: str) -> dict:
    ref = ref.float()
    q = q.float()
    ref_norm = float(ref.norm())
    q_norm = float(q.norm())
    rel = float((q - ref).norm()) / ref_norm if ref_norm > 0 else float("nan")
    cos = (float((ref * q).sum() / (ref_norm * q_norm))
           if ref_norm > 0 and q_norm > 0 else float("nan"))
    p_ref = head_shares(ref, axis)
    p_q = head_shares(q, axis)
    return {
        "rel_U": rel,
        "cos": cos,
        "R_norm": q_norm / ref_norm if ref_norm > 0 else float("nan"),
        "share_L1": float((p_q - p_ref).abs().sum()),
        "top1_same": int(int(p_q.argmax()) == int(p_ref.argmax())),
    }
