#!/usr/bin/env python3
"""Q4: type x spectral-band NS transfer.

For Q/K/V/O, compute:
  A_total = (||E_U,g||/||U_g||) / (||E_D,g||/||D_g||)
  A_rel,b = (||E_U,g,b||/||U_g,b||) / (||E_D,g,b||/||D_g,b||)
  share_U,g,b = ||E_U,g,b|| / ||E_U,g||

Q/K/V use the global fused-QKV SVD; O uses its own SVD.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from lowp_common import (LAYERS, PARAMS, SEEDS, UPDATES, load_snapshot,
                         metrics, ns, round_bf16)
from q2_bf16_error import group_slices, slice_axis
from q2_scale_variants import quantize_heads
from q3_spectral_bands import project_band, split_bands

OUT_ROOT = Path("/mnt/d/muon_research/results/lowp_first_batch/q4_type_band")
VARIANTS = (
    ("bf16", "bf16", None),
    ("bf16_head_bf16", "bf16", "bf16"),
    ("int8_head_fp32", "int8", "fp32"),
)
BANDS = ("total", "top", "middle", "tail")


def run_matrix(name: str, axis: str, d: torch.Tensor, seed: str, update: int,
               layer: int, writer: csv.DictWriter) -> None:
    d = d.float()
    u = ns(d)

    types = ("Q", "K", "V") if name == "fused_qkv" else ("O",)
    for variant, element, scale_dtype in VARIANTS:
        if scale_dtype is None:
            d_var = round_bf16(d)
        else:
            d_var, _ = quantize_heads(d, axis, element, scale_dtype)
        u_var = ns(d_var)
        e_d = d_var - d
        e_u = u_var - u

        for typ in types:
            sl = group_slices(name, typ)
            ref_d = slice_axis(d, axis, sl)
            ref_u = slice_axis(u, axis, sl)
            q_d = slice_axis(d_var, axis, sl)
            q_u = slice_axis(u_var, axis, sl)
            gd = slice_axis(e_d, axis, sl)
            gu = slice_axis(e_u, axis, sl)

            # Per-type SVD: bands are orthogonal inside this type block.
            type_svd, s, vt = torch.linalg.svd(ref_d, full_matrices=False)
            bands = split_bands(s)
            energy = s ** 2
            energy_share = {
                b: float(energy[idx].sum() / energy.sum()) if len(idx) else float("nan")
                for b, idx in bands.items()
            }

            d_g_norm = float(ref_d.norm())
            u_g_norm = float(ref_u.norm())
            e_d_g = float(gd.norm())
            e_u_g = float(gu.norm())
            rel_d_total = e_d_g / d_g_norm if d_g_norm > 0 else float("nan")
            rel_u_total = e_u_g / u_g_norm if u_g_norm > 0 else float("nan")
            a_total = rel_u_total / rel_d_total if rel_d_total > 0 else float("nan")
            m = metrics(ref_u, q_u, axis)

            for band in BANDS:
                if band == "total":
                    rank_count = len(s)
                    eshare = 1.0
                    rel_d_rel = rel_d_total
                    rel_u_rel = rel_u_total
                    a_rel = a_total
                    share_d = 1.0
                    share_u = 1.0
                else:
                    idx = bands[band]
                    ed = project_band(gd, type_svd, vt, idx)
                    eu = project_band(gu, type_svd, vt, idx)
                    db = project_band(ref_d, type_svd, vt, idx)
                    ub = project_band(ref_u, type_svd, vt, idx)
                    db_norm = float(db.norm())
                    ub_norm = float(ub.norm())
                    rel_d_rel = float(ed.norm()) / db_norm if db_norm > 0 else float("nan")
                    rel_u_rel = float(eu.norm()) / ub_norm if ub_norm > 0 else float("nan")
                    a_rel = rel_u_rel / rel_d_rel if rel_d_rel > 0 else float("nan")
                    share_d = float(ed.norm()) / e_d_g if e_d_g > 0 else float("nan")
                    share_u = float(eu.norm()) / e_u_g if e_u_g > 0 else float("nan")
                    rank_count = len(idx)
                    eshare = energy_share[band]

                a_reliable = int(rel_d_rel > 1e-8 and np.isfinite(a_rel))
                writer.writerow({
                    "seed": seed, "update": update, "layer": layer,
                    "matrix": name, "type": typ, "variant": variant,
                    "element": element,
                    "scale_dtype": "" if scale_dtype is None else scale_dtype,
                    "band": band, "rank_count": rank_count,
                    "energy_share": eshare,
                    "rel_D_total": rel_d_total, "rel_U_total": rel_u_total,
                    "A_total": a_total,
                    "rel_D_rel": rel_d_rel, "rel_U_rel": rel_u_rel,
                    "A_rel": a_rel, "A_rel_reliable": a_reliable,
                    "share_D": share_d, "share_U": share_u,
                    "share_L1": m["share_L1"], "top1_same": m["top1_same"],
                    "d_norm": d_g_norm, "u_norm": u_g_norm,
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

    csv_path = args.out / ("smoke.csv" if args.smoke else "q4_metrics.csv")
    fields = ["seed", "update", "layer", "matrix", "type", "variant",
              "element", "scale_dtype", "band", "rank_count", "energy_share",
              "rel_D_total", "rel_U_total", "A_total",
              "rel_D_rel", "rel_U_rel", "A_rel", "A_rel_reliable",
              "share_D", "share_U", "share_L1", "top1_same",
              "d_norm", "u_norm"]
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
