#!/usr/bin/env python3
"""Stage 5 A-5 — source decomposition of dS_h.

For each attention head:

    dP = dA V^T
    m  = sum_j A_ij dP_ij          (row-wise A-weighted mean)
    C  = dP - m
    dS = A * C

Exact identities:

    ||dS||^2 = ||dP||^2 * c_ratio * g
    c_ratio    = ||C||^2 / ||dP||^2
    g        = ||dS||^2 / ||C||^2

    ||dP||^2 = ||dA||^2 * ||V||^2 * eta_p
    eta_p    = ||dA V^T||^2 / (||dA||^2 ||V||^2)

One row is written per (seed, node, loss, layer, head, window).  Probe uses
8 windows; train uses the checkpoint sidecar's micro-batch groups (8 groups
of 4 windows for this recipe, 32 windows total).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform/src"))

from config.recipe import load_recipe  # noqa: E402
from data.frozen_tokens import DeterministicSampler, FrozenWindows, load_manifest  # noqa: E402
from models.llama import apply_rotary_emb  # noqa: E402
from train_platform import build_model  # noqa: E402

SEEDS = ("seed0", "seed1")
NODES = (0, 8, 16, 32, 48, 56, 64, 80, 96, 112, 128, 192, 256, 320, 384,
         448, 512, 576, 640, 704, 768, 784, 832, 896, 960, 1024,
         1152, 1280, 1312, 1408, 1536)
HEAD_DIM = 64
N_HEADS = 6
FIELDS = [
    "seed", "node", "loss", "layer", "head", "window",
    "da2", "v2", "dp2", "eta_p", "m2", "dot_dp_m", "c2", "ds2",
    "c_ratio", "g", "attn_entropy", "attn_max_excl0", "a2_fro",
    "frac_cancel_gt50", "ds_top10_share",
    "dp_row_cv", "c_row_cv", "ds_row_cv", "corr_ent_ds",
]


def set_math():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def fl(x):
    return float(x.detach())


class Capture:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.qkv = {}
        self.do = {}

    def register(self):
        for layer, block in enumerate(self.model.transformer.h):
            self.handles.append(block.attn.c_attn.register_forward_hook(
                self._qkv(layer)))
            self.handles.append(block.attn.c_proj.register_forward_hook(
                self._proj(layer)))

    def _qkv(self, layer):
        def hook(module, inputs, output):
            self.qkv[layer] = output.detach()
        return hook

    def _proj(self, layer):
        def hook(module, inputs, output):
            output.register_hook(lambda g: self.do.__setitem__(layer, g.detach()))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _row_cv(row2):
    if row2.numel() == 0:
        return 0.0
    mean = row2.mean()
    if mean <= 0:
        return 0.0
    return float(row2.std(unbiased=False) / mean)


def _pearson(a, b):
    if a.numel() < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt((a * a).sum() * (b * b).sum())
    if den <= 0:
        return 0.0
    return float((a * b).sum() / den)


def layer_rows(model, qkv, dout, layer, seed, node, loss_name, window_offset):
    B, T, _ = qkv.shape
    q, k, v = qkv.split(384, dim=2)
    q = q.view(B, T, N_HEADS, HEAD_DIM)
    k = k.view(B, T, N_HEADS, HEAD_DIM)
    v = v.view(B, T, N_HEADS, HEAD_DIM)
    freqs = model.freqs_cis.to(q.device)[:T]
    qr, kr = apply_rotary_emb(q, k, freqs)
    qh = qr.permute(0, 2, 1, 3)
    kh = kr.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    A = torch.softmax(scores.float(), dim=-1)

    W = model.transformer.h[layer].attn.c_proj.weight
    dA = torch.matmul(dout, W).view(B, T, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    rows = []
    with torch.no_grad():
        for h in range(N_HEADS):
            Ah = A[:, h]
            dah = dA[:, h]
            vv = vh[:, h]
            dP = torch.matmul(dah, vv.transpose(-1, -2))
            mvec = (dP * Ah).sum(dim=-1)              # B,T
            m = mvec.unsqueeze(-1)                    # B,T,1
            C = dP - m
            dS = Ah * C
            ent_row = -(Ah.clamp_min(1e-12) * Ah.clamp_min(1e-12).log()).sum(-1)
            for b in range(B):
                dp2 = float((dP[b].double() ** 2).sum())
                da2 = float((dah[b].double() ** 2).sum())
                v2 = float((vv[b].double() ** 2).sum())
                m2 = float(T * (mvec[b].double() ** 2).sum())
                dot = float((dP[b].double() * mvec[b, :, None].double()).sum())
                c2 = float((C[b].double() ** 2).sum())
                ds2 = float((dS[b].double() ** 2).sum())
                if dp2 <= 0 or da2 <= 0 or v2 <= 0:
                    continue
                eta_p = dp2 / (da2 * v2)
                c_ratio = c2 / dp2 if dp2 > 0 else 0.0
                g = ds2 / c2 if c2 > 0 else 0.0

                dp_row2 = (dP[b].double() ** 2).sum(-1)
                c_row2 = (C[b].double() ** 2).sum(-1)
                ds_row2 = (dS[b].double() ** 2).sum(-1)
                ent = ent_row[b].double()
                # row 0 has one visible key; exclude it from row statistics
                dp_r = dp_row2[1:]
                c_r = c_row2[1:]
                ds_r = ds_row2[1:]
                ent_r = ent[1:]
                frac_cancel = float((c_r < 0.5 * dp_r).double().mean())
                k_top = max(1, int(round(0.1 * ds_r.numel())))
                top = torch.sort(ds_r, descending=True).values[:k_top].sum()
                ds_top10 = float(top / ds_r.sum()) if ds_r.sum() > 0 else 0.0
                a_max = float(Ah[b, 1:, :].max()) if T > 1 else 0.0
                a2 = float((Ah[b].double() ** 2).sum())

                rows.append({
                    "seed": seed, "node": int(node), "loss": loss_name,
                    "layer": int(layer), "head": int(h),
                    "window": int(window_offset + b),
                    "da2": da2, "v2": v2, "dp2": dp2, "eta_p": eta_p,
                    "m2": m2, "dot_dp_m": dot, "c2": c2, "ds2": ds2,
                    "c_ratio": c_ratio, "g": g,
                    "attn_entropy": float(ent_r.mean()),
                    "attn_max_excl0": a_max, "a2_fro": a2,
                    "frac_cancel_gt50": frac_cancel,
                    "ds_top10_share": ds_top10,
                    "dp_row_cv": _row_cv(dp_r),
                    "c_row_cv": _row_cv(c_r),
                    "ds_row_cv": _row_cv(ds_r),
                    "corr_ent_ds": _pearson(ent_r, ds_r),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--nodes", default=",".join(map(str, NODES)))
    parser.add_argument("--losses", default="probe,train")
    parser.add_argument("--probe-windows", type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    nodes = tuple(int(x) for x in args.nodes.split(","))
    losses = tuple(s.strip() for s in args.losses.split(",") if s.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    rows = []
    summary = {"nodes": list(nodes), "seeds": list(seeds),
               "losses": list(losses), "runs": []}

    for seed in seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
        manifest = load_manifest(Path(args.data_root) / "manifest.json")
        train = FrozenWindows(manifest, "train", 256)
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        set_math()
        model.eval()
        for node in nodes:
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            node_dir = Path(args.ckpt_root) / seed / "node_checkpoints"
            ckpt = torch.load(node_dir / f"pre_u{node:06d}.pt",
                              map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            sidecar = json.loads((node_dir / f"pre_u{node:06d}.json").read_text())
            micro = sidecar.get("micro_batch_window_ids")
            if not micro:
                sampler = DeterministicSampler(train.num_windows, cfg["experiment"]["data_seed"],
                                               cfg["data"]["allow_repeated_epochs"])
                sampler.load_state_dict(ckpt["sampler"])
                ids = sampler.take(cfg["train"]["micro_batch_size"] * cfg["train"]["accumulation_steps"])
                micro = [ids[i * 4:(i + 1) * 4] for i in range(len(ids) // 4)]

            def batch(ids, split="train"):
                source = train if split == "train" else val
                pairs = [source.window(i) for i in ids]
                return (torch.stack([p[0] for p in pairs]).to(device),
                        torch.stack([p[1] for p in pairs]).to(device))

            groups = []
            if "probe" in losses:
                groups.append(("probe", [list(range(args.probe_windows))], "val"))
            if "train" in losses:
                groups.append(("train", micro, "train"))

            for loss_name, glist, split in groups:
                offset = 0
                for group in glist:
                    x, y = batch(group, split)
                    cap = Capture(model)
                    cap.register()
                    model.zero_grad(set_to_none=True)
                    loss = model(x, y)["loss"]
                    loss.backward()
                    cap.remove()
                    for layer in range(6):
                        rows.extend(layer_rows(model, cap.qkv[layer], cap.do[layer],
                                               layer, seed, node, loss_name, offset))
                    offset += len(group)
                    del x, y, loss, cap
                    torch.cuda.empty_cache()
            summary["runs"].append({"seed": seed, "node": node,
                                    "seconds": time.time() - started,
                                    "peak_MiB": torch.cuda.max_memory_allocated() / 1024 ** 2})
            print(f"[stage5a5] {seed} node {node}", flush=True)
            del ckpt
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    with (out / "ds_source_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "COMPLETE").write_text("ok\n")
    print(f"[stage5a5] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
