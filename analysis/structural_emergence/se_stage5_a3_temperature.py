#!/usr/bin/env python3
"""Stage 5 A-3 — local attention-temperature counterfactual.

At each node, capture Q/K/V and dA from the real probe forward/backward.
Then, with dA fixed, recompute attention A(T), dP, dS, dQ, dK, dV for
several temperatures T:
  A(T) = softmax(Q K^T / (sqrt(d) T))
  dQ   = (1/T) dS K / sqrt(d)
  dK   = (1/T) dS^T Q / sqrt(d)

This is a local backward-formula intervention; no training, no HVP.
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
from data.frozen_tokens import FrozenWindows, load_manifest  # noqa: E402
from models.llama import apply_rotary_emb  # noqa: E402
from train_platform import build_model  # noqa: E402

SEEDS = ("seed0", "seed1")
NODES = (0, 8, 16, 32, 48, 56, 64, 80, 96, 112, 128, 192, 256, 320, 384,
         448, 512, 576, 640, 704, 768, 784, 832, 896, 960, 1024,
         1152, 1280, 1312, 1408, 1536)
TYPES = ("Q", "K", "V")
HEAD_DIM = 64
FIELDS = [
    "seed", "node", "layer", "type", "head", "temperature",
    "entropy", "a_max", "dq_fro", "dk_fro", "dv_fro",
    "share_q", "share_k", "share_v",
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


def temperature_rows(model, qkv, dout, layer, temps):
    B, T, _ = qkv.shape
    q, k, v = qkv.split(384, dim=2)
    q = q.view(B, T, 6, HEAD_DIM)
    k = k.view(B, T, 6, HEAD_DIM)
    v = v.view(B, T, 6, HEAD_DIM)
    freqs = model.freqs_cis.to(q.device)[:T]
    qr, kr = apply_rotary_emb(q, k, freqs)
    qh = qr.permute(0, 2, 1, 3)
    kh = kr.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    W = model.transformer.h[layer].attn.c_proj.weight
    dA = dout @ W
    dA = dA.view(B, T, 6, HEAD_DIM).permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    rows = []
    for temp in temps:
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * (scale / temp)
        scores = scores.masked_fill(mask, float("-inf"))
        A = torch.softmax(scores.float(), dim=-1)
        metrics = []
        for h in range(6):
            Ah = A[:, h]
            dah = dA[:, h]
            vv = vh[:, h]
            qq = qh[:, h]
            kk = kh[:, h]
            dV = torch.matmul(Ah.transpose(-1, -2), dah)
            dP = torch.matmul(dah, vv.transpose(-1, -2))
            dS = Ah * (dP - (dP * Ah).sum(dim=-1, keepdim=True))
            dQ = torch.matmul(dS, kk) * (scale / temp)
            dK = torch.matmul(dS.transpose(-1, -2), qq) * (scale / temp)
            ent = -(Ah.clamp_min(1e-12) * Ah.clamp_min(1e-12).log()).sum(-1)
            metrics.append({
                "head": h, "entropy": fl(ent.mean()), "a_max": fl(Ah.max()),
                "dq": fl(dQ.norm()), "dk": fl(dK.norm()), "dv": fl(dV.norm()),
            })
        total_q = sum(x["dq"] ** 2 for x in metrics)
        total_k = sum(x["dk"] ** 2 for x in metrics)
        total_v = sum(x["dv"] ** 2 for x in metrics)
        for x in metrics:
            rows.append({
                "layer": layer, "type": "Q", "head": x["head"],
                "temperature": temp, "entropy": x["entropy"], "a_max": x["a_max"],
                "dq_fro": x["dq"], "dk_fro": x["dk"], "dv_fro": x["dv"],
                "share_q": x["dq"] ** 2 / total_q if total_q > 0 else 0.0,
                "share_k": x["dk"] ** 2 / total_k if total_k > 0 else 0.0,
                "share_v": x["dv"] ** 2 / total_v if total_v > 0 else 0.0,
            })
    # The three type rows are created below from the same head metrics.
    # Keep only per-head rows; use share_q/k/v columns for the three types.
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--nodes", default=",".join(map(str, NODES)))
    parser.add_argument("--temperatures", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--probe-windows", type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    nodes = tuple(int(x) for x in args.nodes.split(","))
    temps = tuple(float(x) for x in args.temperatures.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    rows = []
    summary = {"nodes": list(nodes), "seeds": list(seeds),
               "temperatures": list(temps), "runs": []}

    for seed in seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
        manifest = load_manifest(Path(args.data_root) / "manifest.json")
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        set_math(); model.eval()
        for node in nodes:
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            node_dir = Path(args.ckpt_root) / seed / "node_checkpoints"
            ckpt = torch.load(node_dir / f"pre_u{node:06d}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            pairs = [val.window(i) for i in range(args.probe_windows)]
            x = torch.stack([p[0] for p in pairs]).to(device)
            y = torch.stack([p[1] for p in pairs]).to(device)
            cap = Capture(model); cap.register()
            model.zero_grad(set_to_none=True)
            loss = model(x, y)["loss"]
            loss.backward()
            cap.remove()
            for layer in range(6):
                base = temperature_rows(model, cap.qkv[layer], cap.do[layer], layer, temps)
                # base has one row per head/temperature; convert each to Q/K/V rows
                for temp in temps:
                    block = [r for r in base if r["temperature"] == temp]
                    for typ, share_key in (("Q", "share_q"), ("K", "share_k"), ("V", "share_v")):
                        for r in block:
                            rows.append({"seed": seed, "node": node,
                                         "layer": layer, "type": typ,
                                         "head": r["head"], "temperature": temp,
                                         "entropy": r["entropy"], "a_max": r["a_max"],
                                         "dq_fro": r["dq_fro"], "dk_fro": r["dk_fro"],
                                         "dv_fro": r["dv_fro"],
                                         "share_q": r["share_q"], "share_k": r["share_k"],
                                         "share_v": r["share_v"]})
            summary["runs"].append({"seed": seed, "node": node,
                                    "seconds": time.time() - started,
                                    "peak_MiB": torch.cuda.max_memory_allocated() / 1024 ** 2})
            print(f"[stage5a3] {seed} node {node}", flush=True)
            del ckpt, cap, x, y, loss
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    with (out / "temperature_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "COMPLETE").write_text("ok\n")
    print(f"[stage5a3] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
