#!/usr/bin/env python3
"""Q2: how the natural BF16 error propagates through Newton-Schulz.

No synthetic perturbations.  For each attention matrix:
  D_bf16 = BF16(D)
  U      = NS(D)
  U_bf16 = NS(D_bf16)
  E_D    = D_bf16 - D
  E_U    = U_bf16 - U

Decompose E_D and E_U into:
  global scale, head-relative scale, direction (residual).
Then report rel_D, rel_U, A = rel_U / rel_D for each component,
per Q/K/V/O and for the whole matrix.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from lowp_common import (LAYERS, PARAMS, SEEDS, UPDATES, load_snapshot,
                         metrics, ns, round_bf16, split_heads, head_shares)

OUT_ROOT = Path("/mnt/d/muon_research/results/lowp_first_batch/q2_bf16_error")


def decompose(q: torch.Tensor, ref: torch.Tensor, axis: str):
    """Return components of q-ref in the head-scale basis.

    q and ref have the same shape.  ``axis`` is "rows" for fused QKV and
    "cols" for O, matching the actual head blocks.
    """
    q_blocks = split_heads(q, axis)
    r_blocks = split_heads(ref, axis)
    ref_norm2 = float((ref.float() ** 2).sum())
    c = float((q.float() * ref.float()).sum()) / ref_norm2 if ref_norm2 > 0 else 1.0
    e_global = (c - 1.0) * ref.float()

    head_terms = []
    c_hs = []
    for (_, qb), (_, rb) in zip(q_blocks, r_blocks):
        rb = rb.float()
        rb_norm2 = float((rb ** 2).sum())
        c_h = (float((qb.float() * rb).sum()) / rb_norm2) if rb_norm2 > 0 else c
        c_hs.append(c_h)
        head_terms.append((c_h - c) * rb)
    e_head = torch.cat(head_terms, dim=0 if axis == "rows" else 1)
    e_dir = (q.float() - ref.float()) - e_global - e_head
    return {"global": e_global, "head": e_head, "direction": e_dir}, c, c_hs


def group_slices(matrix: str, group: str):
    if matrix == "O":
        return slice(None)
    if group == "all":
        return slice(None)
    return {"Q": slice(0, 384), "K": slice(384, 768), "V": slice(768, 1152)}[group]


def slice_axis(x: torch.Tensor, axis: str, sl: slice) -> torch.Tensor:
    return x[sl] if axis == "rows" else x[:, sl]


def head_scale_rms(c_hs, c: float) -> float:
    if not c_hs:
        return 0.0
    vals = torch.tensor([ch - c for ch in c_hs])
    return float((vals ** 2).mean().sqrt())


def run_matrix(name: str, axis: str, d: torch.Tensor, seed: str, update: int,
               layer: int, writer: csv.DictWriter) -> None:
    d = d.float()
    d_bf16 = round_bf16(d)
    u = ns(d)
    u_bf16 = ns(d_bf16)

    e_d, c_d, c_d_h = decompose(d_bf16, d, axis)
    e_u, c_u, c_u_h = decompose(u_bf16, u, axis)

    groups = ("Q", "K", "V", "all") if name == "fused_qkv" else ("O",)
    for group in groups:
        sl = group_slices(name, group)
        ref_d = slice_axis(d, axis, sl)
        ref_u = slice_axis(u, axis, sl)
        q_d = slice_axis(d_bf16, axis, sl)
        q_u = slice_axis(u_bf16, axis, sl)

        d_norm = float(ref_d.norm())
        u_norm = float(ref_u.norm())
        e_d_total = float((q_d - ref_d).norm())
        e_u_total = float((q_u - ref_u).norm())
        rel_d_total = e_d_total / d_norm if d_norm > 0 else float("nan")
        rel_u_total = e_u_total / u_norm if u_norm > 0 else float("nan")

        if group in ("all", "O"):
            heads = c_d_h
        else:
            # c_h list follows the 18 heads; group Q/K/V are 6 consecutive heads.
            idx = {"Q": 0, "K": 1, "V": 2}[group]
            heads = c_d_h[idx * 6:(idx + 1) * 6]

        m = metrics(ref_u, q_u, axis)
        for comp in ("global", "head", "direction"):
            ed = slice_axis(e_d[comp], axis, sl)
            eu = slice_axis(e_u[comp], axis, sl)
            rel_d = float(ed.norm()) / d_norm if d_norm > 0 else float("nan")
            rel_u = float(eu.norm()) / u_norm if u_norm > 0 else float("nan")
            a = rel_u / rel_d if rel_d > 0 else float("nan")
            share_d = float(ed.norm()) / e_d_total if e_d_total > 0 else float("nan")
            share_u = float(eu.norm()) / e_u_total if e_u_total > 0 else float("nan")
            a_reliable = int(share_d >= 0.01 and rel_d > 1e-8)
            writer.writerow({
                "seed": seed, "update": update, "layer": layer,
                "matrix": name, "group": group, "component": comp,
                "rel_D": rel_d, "rel_U": rel_u, "A": a,
                "A_reliable": a_reliable,
                "share_D": share_d, "share_U": share_u,
                "share_L1": m["share_L1"], "top1_same": m["top1_same"],
                "rel_D_total": rel_d_total, "rel_U_total": rel_u_total,
                "scale_global": c_d, "head_scale_rms": head_scale_rms(heads, c_d),
                "d_norm": d_norm, "u_norm": u_norm,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = ("seed0",) if args.smoke else SEEDS
    updates = (128,) if args.smoke else UPDATES
    layers = (0,) if args.smoke else LAYERS

    csv_path = args.out / ("smoke.csv" if args.smoke else "q2_metrics.csv")
    fields = ["seed", "update", "layer", "matrix", "group", "component",
              "rel_D", "rel_U", "A", "A_reliable", "share_D", "share_U",
              "share_L1", "top1_same", "rel_D_total", "rel_U_total",
              "scale_global", "head_scale_rms", "d_norm", "u_norm"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            for update in updates:
                manifest, tensors = load_snapshot(seed, update)
                for layer in layers:
                    for name, pattern, axis in PARAMS:
                        param = pattern.format(layer=layer)
                        role = manifest["parameters"][param]["roles"]["D"]
                        d = tensors[role["tensor_id"]].float()
                        run_matrix(name, axis, d, seed, update, layer, writer)
                del manifest, tensors
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
