#!/usr/bin/env python3
"""Q1: global scale invariance and global representation quantization.

Default format is INT8 (symmetric, per-tensor absmax).  BF16 is kept as a
16-bit reference.  FP8 can be enabled with ``--format fp8`` or ``both``.

Reference is FP32.  All low-precision representations are simulated by
round-to-nearest; the Newton-Schulz map is always evaluated in FP32.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/mnt/d/muon_research")
SNAP_ROOT = Path("/mnt/d/experiments/structural_emergence_output/structural_emergence")
OUT_ROOT = ROOT / "results" / "lowp_first_batch" / "q1_global_scale"

sys.path.insert(0, str(ROOT / "platform" / "src"))
from optim.muon_reference import zeropower_newton_schulz  # noqa: E402

SEEDS = ("seed0", "seed1")
UPDATES = (128, 1024, 4096, 8192)
LAYERS = tuple(range(6))
PARAMS = (
    ("fused_qkv", "transformer.h.{layer}.attn.c_attn.weight", "rows"),
    ("O", "transformer.h.{layer}.attn.c_proj.weight", "cols"),
)
SCALE_C = (0.25, 0.5, 1.0, 2.0, 4.0)
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_SUB = 2.0 ** -9
FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN_SUB = 2.0 ** -16
INT8_MAX = 127.0


def ns(x: torch.Tensor) -> torch.Tensor:
    return zeropower_newton_schulz(
        x, steps=5, coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7
    ).float()


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def scale_value(s: float, dtype: str) -> tuple[float, bool]:
    """Return (stored_scale, underflowed)."""
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


def quant_int8(x: torch.Tensor, scale_dtype: str = "fp32"):
    max_abs = float(x.abs().max())
    s = max_abs / INT8_MAX if max_abs > 0 else 1.0
    s_stored, underflow = scale_value(s, scale_dtype)
    if s_stored <= 0:
        return torch.full_like(x, float("nan")), s_stored, True, 0.0, 0.0
    q = (x / s_stored).round().clamp(-INT8_MAX, INT8_MAX) * s_stored
    clamp_frac = float(((x.abs() / s_stored) > INT8_MAX).float().mean())
    zero_frac = float((q == 0).float().mean())
    return q, s_stored, underflow, clamp_frac, zero_frac


def int8_codes(x: torch.Tensor, s: float) -> torch.Tensor:
    return (x / s).round().clamp(-INT8_MAX, INT8_MAX)


def quant_fp8(x: torch.Tensor, scale_dtype: str = "fp32"):
    max_abs = float(x.abs().max())
    s = max_abs / FP8_E4M3_MAX if max_abs > 0 else 1.0
    s_stored, underflow = scale_value(s, scale_dtype)
    if s_stored <= 0:
        return torch.full_like(x, float("nan")), s_stored, True, 0.0, 0.0
    q = (x / s_stored).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = q.to(torch.float8_e4m3fn).float() * s_stored
    clamp_frac = float(((x.abs() / s_stored) > FP8_E4M3_MAX).float().mean())
    zero_frac = float((q == 0).float().mean())
    return q, s_stored, underflow, clamp_frac, zero_frac


def fp8_codes(x: torch.Tensor, s: float) -> torch.Tensor:
    return (x / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()


def head_shares(x: torch.Tensor, axis: str, head_dim: int = 64) -> torch.Tensor:
    blocks = list(x.split(head_dim, dim=0)) if axis == "rows" else list(x.split(head_dim, dim=1))
    energy = torch.tensor([float((b.float() ** 2).sum()) for b in blocks])
    return energy / energy.sum().clamp_min(1e-30)


def metrics(ref: torch.Tensor, q: torch.Tensor, axis: str) -> dict:
    ref = ref.float()
    q = q.float()
    diff = q - ref
    ref_norm = float(ref.norm())
    q_norm = float(q.norm())
    rel = float(diff.norm()) / ref_norm if ref_norm > 0 else float("nan")
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


def process_matrix(name: str, axis: str, d: torch.Tensor, seed: str, update: int,
                   layer: int, writer: csv.DictWriter, fmt: str) -> None:
    u_ref = ns(d)
    d_norm = float(d.norm())
    u_norm = float(u_ref.norm())

    def emit(variant: str, u_q: torch.Tensor, c: float | None = None,
             scale_dtype: str = "fp32", scale: float | None = None,
             underflow: bool = False, clamp_frac: float = 0.0,
             zero_frac: float = 0.0, compare: torch.Tensor | None = None,
             compare_variant: str = "") -> None:
        row = {
            "seed": seed, "update": update, "layer": layer, "matrix": name,
            "variant": variant, "c": "" if c is None else c,
            "scale_dtype": scale_dtype,
            "scale": "" if scale is None else scale,
            "underflow": int(underflow),
            "clamp_frac": clamp_frac, "zero_frac": zero_frac,
            "d_norm": d_norm, "u_norm": u_norm,
        }
        row.update(metrics(u_ref, u_q, axis))
        if compare is not None:
            cmp = metrics(compare, u_q, axis)
            row["compare_variant"] = compare_variant
            row["rel_U_vs_compare"] = cmp["rel_U"]
            row["cos_vs_compare"] = cmp["cos"]
            row["R_norm_vs_compare"] = cmp["R_norm"]
            row["share_L1_vs_compare"] = cmp["share_L1"]
            row["top1_same_vs_compare"] = cmp["top1_same"]
        else:
            row["compare_variant"] = ""
            row["rel_U_vs_compare"] = ""
            row["cos_vs_compare"] = ""
            row["R_norm_vs_compare"] = ""
            row["share_L1_vs_compare"] = ""
            row["top1_same_vs_compare"] = ""
        writer.writerow(row)

    # 16-bit reference.
    emit("value_bf16", ns(round_bf16(d)))

    if fmt in ("int8", "both"):
        d_int8, s_i8, uf, cf, zf = quant_int8(d, "fp32")
        u_value_int8 = ns(d_int8)
        emit("value_int8", u_value_int8, scale_dtype="fp32", scale=s_i8,
             underflow=uf, clamp_frac=cf, zero_frac=zf)

        # Q1a: same INT8 codes, scale stored in BF16 / FP16.
        max_abs = float(d.abs().max())
        s = max_abs / INT8_MAX if max_abs > 0 else 1.0
        codes = int8_codes(d, s)
        for sd in ("bf16", "fp16"):
            s_stored, uf = scale_value(s, sd)
            emit(f"int8_scale_{sd}", ns(codes * s_stored), scale_dtype=sd,
                 scale=s_stored, underflow=uf, compare=u_value_int8,
                 compare_variant="value_int8")

        # Q1c: global scale before INT8 quantization.
        for c in SCALE_C:
            d_c, s_c, uf, cf, zf = quant_int8(c * d, "fp32")
            emit(f"value_int8_c{c:g}", ns(d_c), c=c, scale_dtype="fp32",
                 scale=s_c, underflow=uf, clamp_frac=cf, zero_frac=zf,
                 compare=u_value_int8, compare_variant="value_int8")

    if fmt in ("fp8", "both"):
        d_fp8, s_fp8, uf, cf, zf = quant_fp8(d, "fp32")
        u_value_fp8 = ns(d_fp8)
        emit("value_fp8_e4m3", u_value_fp8, scale_dtype="fp32", scale=s_fp8,
             underflow=uf, clamp_frac=cf, zero_frac=zf)

        max_abs = float(d.abs().max())
        s = max_abs / FP8_E4M3_MAX if max_abs > 0 else 1.0
        codes = fp8_codes(d, s)
        for sd in ("bf16", "fp8_e4m3", "fp8_e5m2"):
            s_stored, uf = scale_value(s, sd)
            emit(f"fp8_scale_{sd}", ns(codes * s_stored), scale_dtype=sd,
                 scale=s_stored, underflow=uf, compare=u_value_fp8,
                 compare_variant="value_fp8_e4m3")

        for c in SCALE_C:
            d_c, s_c, uf, cf, zf = quant_fp8(c * d, "fp32")
            emit(f"value_fp8_c{c:g}", ns(d_c), c=c, scale_dtype="fp32",
                 scale=s_c, underflow=uf, clamp_frac=cf, zero_frac=zf,
                 compare=u_value_fp8, compare_variant="value_fp8_e4m3")


def load_snapshot(seed: str, update: int) -> tuple[dict, dict]:
    snap = SNAP_ROOT / seed / "snapshots" / f"update_{update:06d}"
    manifest = json.loads(snap.with_suffix(".manifest.json").read_text())
    payload = torch.load(snap.with_suffix(".pt"), map_location="cpu",
                         weights_only=False)
    return manifest, payload["tensors"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--format", choices=("int8", "fp8", "both"), default="int8")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = ("seed0",) if args.smoke else SEEDS
    updates = (128,) if args.smoke else UPDATES
    layers = (0,) if args.smoke else LAYERS

    suffix = "" if args.format == "int8" else f"_{args.format}"
    csv_path = args.out / (f"smoke{suffix}.csv" if args.smoke else f"q1_metrics{suffix}.csv")
    fields = ["seed", "update", "layer", "matrix", "variant", "c",
              "scale_dtype", "scale", "underflow", "clamp_frac", "zero_frac",
              "d_norm", "u_norm", "rel_U", "cos", "R_norm", "share_L1",
              "top1_same", "compare_variant", "rel_U_vs_compare",
              "cos_vs_compare", "R_norm_vs_compare", "share_L1_vs_compare",
              "top1_same_vs_compare"]
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
                        process_matrix(name, axis, d, seed, update, layer,
                                       writer, args.format)
                del manifest, tensors
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
