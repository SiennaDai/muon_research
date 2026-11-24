#!/usr/bin/env python3
"""Q2a/b/c: block-scaled numerical formats through Newton-Schulz.

Variants:
  bf16                 standard BF16 (Q1 baseline)
  bf16_head_fp32       BF16 element, per-head scale stored FP32
  bf16_head_bf16       BF16 element, per-head scale stored BF16
  bf16_head_fp8_e4m3   BF16 element, per-head scale stored FP8 e4m3
  int8_head_fp32       INT8 element, per-head scale stored FP32
  int8_head_bf16       INT8 element, per-head scale stored BF16

For each variant we decompose E_D = D_var - D and E_U = U_var - U into
global scale / head-relative scale / direction, per Q/K/V/O and whole matrix.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from lowp_common import (LAYERS, PARAMS, SEEDS, UPDATES, load_snapshot,
                         metrics, ns, round_bf16, split_heads)
from q2_bf16_error import decompose, group_slices, head_scale_rms, slice_axis

OUT_ROOT = Path("/mnt/d/muon_research/results/lowp_first_batch/q2_scale_variants")
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_SUB = 2.0 ** -9
FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN_SUB = 2.0 ** -16
INT8_MAX = 127.0

VARIANTS = (
    ("bf16", "bf16", None),
    ("bf16_head_fp32", "bf16", "fp32"),
    ("bf16_head_bf16", "bf16", "bf16"),
    ("bf16_head_fp8_e4m3", "bf16", "fp8_e4m3"),
    ("int8_head_fp32", "int8", "fp32"),
    ("int8_head_bf16", "int8", "bf16"),
)


def scale_value(s: float, dtype: str) -> tuple[float, bool]:
    if dtype == "fp32":
        return s, False
    if dtype == "bf16":
        return float(torch.tensor(s).to(torch.bfloat16).float()), False
    if dtype == "fp16":
        return float(torch.tensor(s).to(torch.float16).float()), False
    if dtype == "fp8_e4m3":
        underflow = s < FP8_E4M3_MIN_SUB
        stored = float(torch.tensor(max(s, FP8_E4M3_MIN_SUB))
                       .clamp(FP8_E4M3_MIN_SUB, FP8_E4M3_MAX)
                       .to(torch.float8_e4m3fn).float())
        return stored, underflow
    if dtype == "fp8_e5m2":
        underflow = s < FP8_E5M2_MIN_SUB
        stored = float(torch.tensor(max(s, FP8_E5M2_MIN_SUB))
                       .clamp(FP8_E5M2_MIN_SUB, FP8_E5M2_MAX)
                       .to(torch.float8_e5m2).float())
        return stored, underflow
    raise ValueError(dtype)


def quantize_heads(x: torch.Tensor, axis: str, element: str,
                   scale_dtype: str) -> tuple[torch.Tensor, dict]:
    blocks = split_heads(x, axis)
    out = []
    underflow = 0
    clamped = 0
    zeros = 0
    total = 0
    for _, b in blocks:
        b = b.float()
        max_abs = float(b.abs().max())
        if element == "bf16":
            s = max_abs if max_abs > 0 else 1.0
            s_stored, uf = scale_value(s, scale_dtype)
            q = round_bf16(b / s)
            dq = q * s_stored
        elif element == "int8":
            s = max_abs / INT8_MAX if max_abs > 0 else 1.0
            s_stored, uf = scale_value(s, scale_dtype)
            y = b / s
            clamped += int((y.abs() > INT8_MAX).sum())
            dq = y.round().clamp(-INT8_MAX, INT8_MAX) * s_stored
        else:
            raise ValueError(element)
        underflow += int(uf)
        zeros += int((dq == 0).sum())
        total += dq.numel()
        out.append(dq)
    dq = torch.cat(out, dim=0 if axis == "rows" else 1)
    info = {
        "underflow_heads": underflow,
        "clamp_frac": clamped / total if total else 0.0,
        "zero_frac": zeros / total if total else 0.0,
    }
    return dq, info


def run_matrix(name: str, axis: str, d: torch.Tensor, seed: str, update: int,
               layer: int, writer: csv.DictWriter) -> None:
    d = d.float()
    u = ns(d)

    for variant, element, scale_dtype in VARIANTS:
        if scale_dtype is None:
            d_var = round_bf16(d)
            info = {"underflow_heads": 0, "clamp_frac": 0.0, "zero_frac": 0.0}
        else:
            d_var, info = quantize_heads(d, axis, element, scale_dtype)
        u_var = ns(d_var)

        e_d, c_d, c_d_h = decompose(d_var, d, axis)
        e_u, c_u, c_u_h = decompose(u_var, u, axis)

        groups = ("Q", "K", "V", "all") if name == "fused_qkv" else ("O",)
        for group in groups:
            sl = group_slices(name, group)
            ref_d = slice_axis(d, axis, sl)
            ref_u = slice_axis(u, axis, sl)
            q_d = slice_axis(d_var, axis, sl)
            q_u = slice_axis(u_var, axis, sl)

            d_norm = float(ref_d.norm())
            u_norm = float(ref_u.norm())
            e_d_total = float((q_d - ref_d).norm())
            e_u_total = float((q_u - ref_u).norm())
            rel_d_total = e_d_total / d_norm if d_norm > 0 else float("nan")
            rel_u_total = e_u_total / u_norm if u_norm > 0 else float("nan")

            if group in ("all", "O"):
                heads = c_d_h
            else:
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
                    "matrix": name, "group": group, "variant": variant,
                    "element": element,
                    "scale_dtype": "" if scale_dtype is None else scale_dtype,
                    "component": comp,
                    "rel_D": rel_d, "rel_U": rel_u, "A": a,
                    "A_reliable": a_reliable,
                    "share_D": share_d, "share_U": share_u,
                    "share_L1": m["share_L1"], "top1_same": m["top1_same"],
                    "rel_D_total": rel_d_total, "rel_U_total": rel_u_total,
                    "scale_global": c_d,
                    "head_scale_rms": head_scale_rms(heads, c_d),
                    "underflow_heads": info["underflow_heads"],
                    "clamp_frac": info["clamp_frac"],
                    "zero_frac": info["zero_frac"],
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
    fields = ["seed", "update", "layer", "matrix", "group", "variant",
              "element", "scale_dtype", "component",
              "rel_D", "rel_U", "A", "A_reliable", "share_D", "share_U",
              "share_L1", "top1_same", "rel_D_total", "rel_U_total",
              "scale_global", "head_scale_rms", "underflow_heads",
              "clamp_frac", "zero_frac", "d_norm", "u_norm"]
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
