#!/usr/bin/env python3
"""Q3: spectral-band distribution of the natural low-precision error.

For each selected variant:
  D_var, U_var = NS(D_var)
  E_D = D_var - D,  E_U = U_var - U
Project E_D/E_U onto Top/Middle/Tail bands of the reference D
(by cumulative squared singular-value energy: 0-50%, 50-90%, 90-100%).
Report rel_D, rel_U, A = rel_U/rel_D per band.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from lowp_common import (LAYERS, PARAMS, SEEDS, UPDATES, load_snapshot,
                         metrics, ns, round_bf16)
from q2_scale_variants import quantize_heads

OUT_ROOT = Path("/mnt/d/muon_research/results/lowp_first_batch/q3_spectral_bands")
VARIANTS = (
    ("bf16", "bf16", None),
    ("bf16_head_fp32", "bf16", "fp32"),
    ("bf16_head_bf16", "bf16", "bf16"),
    ("int8_head_fp32", "int8", "fp32"),
)


def split_bands(s: torch.Tensor, cuts=(0.5, 0.9)):
    """Return disjoint (top, middle, tail) index arrays by cumulative energy."""
    e = (s.cpu().numpy().astype(np.float64) ** 2)
    c = np.cumsum(e) / e.sum()
    top = np.where(c <= cuts[0])[0]
    mid = np.where((c > cuts[0]) & (c <= cuts[1]))[0]
    tail = np.where(c > cuts[1])[0]
    if len(top) == 0:
        # First singular value alone already exceeds the top-energy cut.
        top = np.array([0])
        mid = mid[mid != 0]
    if len(mid) == 0:
        rest = np.setdiff1d(np.arange(len(s)), np.concatenate([top, tail]))
        if len(rest):
            mid = np.array([rest[0]])
    if len(tail) == 0:
        rest = np.setdiff1d(np.arange(len(s)), np.concatenate([top, mid]))
        if len(rest):
            tail = np.array([rest[-1]])
    return {"top": top, "middle": mid, "tail": tail}


def project_band(e: torch.Tensor, u_svd: torch.Tensor, vt: torch.Tensor,
                 idx: np.ndarray) -> torch.Tensor:
    if len(idx) == 0:
        return torch.zeros_like(e)
    ub = u_svd[:, idx]
    vb = vt[idx, :].T
    return ub @ (ub.T @ e @ vb) @ vb.T


def run_matrix(name: str, axis: str, d: torch.Tensor, seed: str, update: int,
               layer: int, writer: csv.DictWriter) -> None:
    d = d.float()
    u = ns(d)
    u_svd, s, vt = torch.linalg.svd(d, full_matrices=False)
    bands = split_bands(s)
    energy = (s ** 2)
    energy_share = {
        b: float(energy[idx].sum() / energy.sum()) if len(idx) else float("nan")
        for b, idx in bands.items()
    }
    d_norm = float(d.norm())
    u_norm = float(u.norm())

    for variant, element, scale_dtype in VARIANTS:
        if scale_dtype is None:
            d_var = round_bf16(d)
        else:
            d_var, _ = quantize_heads(d, axis, element, scale_dtype)
        u_var = ns(d_var)
        e_d = d_var - d
        e_u = u_var - u
        e_d_total = float(e_d.norm())
        e_u_total = float(e_u.norm())
        rel_d_total = e_d_total / d_norm if d_norm > 0 else float("nan")
        rel_u_total = e_u_total / u_norm if u_norm > 0 else float("nan")
        m = metrics(u, u_var, axis)

        for band, idx in bands.items():
            ed = project_band(e_d, u_svd, vt, idx)
            eu = project_band(e_u, u_svd, vt, idx)
            d_band = project_band(d, u_svd, vt, idx)
            u_band = project_band(u, u_svd, vt, idx)
            rel_d = float(ed.norm()) / d_norm if d_norm > 0 else float("nan")
            rel_u = float(eu.norm()) / u_norm if u_norm > 0 else float("nan")
            a_contrib = rel_u / rel_d if rel_d > 0 else float("nan")
            share_d = float(ed.norm()) / e_d_total if e_d_total > 0 else float("nan")
            share_u = float(eu.norm()) / e_u_total if e_u_total > 0 else float("nan")
            d_band_norm = float(d_band.norm())
            u_band_norm = float(u_band.norm())
            rel_d_rel = float(ed.norm()) / d_band_norm if d_band_norm > 0 else float("nan")
            rel_u_rel = float(eu.norm()) / u_band_norm if u_band_norm > 0 else float("nan")
            a_rel = rel_u_rel / rel_d_rel if rel_d_rel > 0 else float("nan")
            writer.writerow({
                "seed": seed, "update": update, "layer": layer,
                "matrix": name, "variant": variant,
                "element": element,
                "scale_dtype": "" if scale_dtype is None else scale_dtype,
                "band": band, "rank_count": len(idx),
                "energy_share": energy_share[band],
                "rel_D": rel_d, "rel_U": rel_u,
                "A_contrib": a_contrib,
                "A_contrib_reliable": int(share_d >= 0.01 and rel_d > 1e-8),
                "rel_D_rel": rel_d_rel, "rel_U_rel": rel_u_rel,
                "A_rel": a_rel,
                "A_rel_reliable": int(rel_d_rel > 1e-8 and d_band_norm > 0),
                "share_D": share_d, "share_U": share_u,
                "share_L1": m["share_L1"], "top1_same": m["top1_same"],
                "rel_D_total": rel_d_total, "rel_U_total": rel_u_total,
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

    csv_path = args.out / ("smoke.csv" if args.smoke else "q3_metrics.csv")
    fields = ["seed", "update", "layer", "matrix", "variant", "element",
              "scale_dtype", "band", "rank_count", "energy_share",
              "rel_D", "rel_U", "A_contrib", "A_contrib_reliable",
              "rel_D_rel", "rel_U_rel", "A_rel", "A_rel_reliable",
              "share_D", "share_U", "share_L1", "top1_same",
              "rel_D_total", "rel_U_total", "d_norm", "u_norm"]
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
