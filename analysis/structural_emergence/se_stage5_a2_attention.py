#!/usr/bin/env python3
"""Stage 5 A-2 — decompose the head-level backward signal.

For each node and attention head compute:
  A_h = softmax(Q_h K_h^T / sqrt(d) + causal)
  dA_h = dO W_proj[:, head columns]
  dV_h = A_h^T dA_h
  dP_h = dA_h V_h^T
  dS_h = A_h * (dP_h - sum(dP_h A_h))
  dQ_h = dS_h K_h / sqrt(d)
  dK_h = dS_h^T Q_h / sqrt(d)

The computed dQ/dK/dV are checked against the captured c_attn output gradients.
No HVP; forward/backward only.
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
TYPES = ("Q", "K", "V", "O")
HEAD_DIM = 64
FIELDS = [
    "seed", "node", "loss", "layer", "type", "head",
    "dy_fro", "d_fro", "rel", "cos",
    "da_fro", "dp_fro", "ds_fro", "a_entropy", "a_max",
    "q_fro", "k_fro", "v_fro", "dy_share", "da_share",
]


def set_math():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def fl(x):
    return float(x.detach())


def cos_abs(a, b):
    denom = a.norm() * b.norm()
    return float((a * b).sum().abs() / denom) if denom > 0 else float("nan")


class Capture:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.qkv = {}
        self.dy = {}
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
            output.register_hook(lambda g: self.dy.__setitem__(layer, g.detach()))
        return hook

    def _proj(self, layer):
        def hook(module, inputs, output):
            output.register_hook(lambda g: self.do.__setitem__(layer, g.detach()))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def compute_metrics(model, cap, layer, loss_name):
    qkv = cap.qkv[layer]  # [B,T,1152]
    dy = cap.dy[layer]    # [B,T,1152]
    dout = cap.do[layer]  # [B,T,384]
    B, T, _ = qkv.shape
    q, k, v = qkv.split(384, dim=2)
    q = q.view(B, T, 6, HEAD_DIM)
    k = k.view(B, T, 6, HEAD_DIM)
    v = v.view(B, T, 6, HEAD_DIM)
    freqs = model.freqs_cis.to(q.device)[:T]
    qr, kr = apply_rotary_emb(q, k, freqs)
    qh = qr.permute(0, 2, 1, 3)  # B,H,T,D
    kh = kr.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    A = torch.softmax(scores.float(), dim=-1)
    W = model.transformer.h[layer].attn.c_proj.weight
    dA = dout @ W  # B,T,384; c_proj: y=A W^T
    dA = dA.view(B, T, 6, HEAD_DIM).permute(0, 2, 1, 3)  # B,H,T,D
    dy_q = dy[:, :, :384].reshape(B, T, 6, HEAD_DIM).permute(0, 2, 1, 3)
    dy_k = dy[:, :, 384:768].reshape(B, T, 6, HEAD_DIM).permute(0, 2, 1, 3)
    dy_v = dy[:, :, 768:].reshape(B, T, 6, HEAD_DIM).permute(0, 2, 1, 3)

    per_head = []
    for h in range(6):
        Ah = A[:, h]
        dah = dA[:, h]
        vv = vh[:, h]
        qq = qh[:, h]
        kk = kh[:, h]
        dV = torch.matmul(Ah.transpose(-1, -2), dah)
        dP = torch.matmul(dah, vv.transpose(-1, -2))
        dS = Ah * (dP - (dP * Ah).sum(dim=-1, keepdim=True))
        dQ = torch.matmul(dS, kk) * scale
        dK = torch.matmul(dS.transpose(-1, -2), qq) * scale
        ent = -(Ah.clamp_min(1e-12) * Ah.clamp_min(1e-12).log()).sum(-1)
        per_head.append({
            "dy_q": dy_q[:, h], "dy_k": dy_k[:, h], "dy_v": dy_v[:, h],
            "dQ": dQ, "dK": dK, "dV": dV,
            "da": fl(dah.norm()), "dp": fl(dP.norm()), "ds": fl(dS.norm()),
            "ent": fl(ent.mean()), "amax": fl(Ah.max()),
            "q": fl(qq.norm()), "k": fl(kk.norm()), "v": fl(vv.norm()),
            "dyq_fro": fl(dy_q[:, h].norm()), "dyk_fro": fl(dy_k[:, h].norm()),
            "dyv_fro": fl(dy_v[:, h].norm()),
        })

    rows = []
    for kind, dykey, dkey, relkey in (
        ("Q", "dy_q", "dQ", "rel_q"),
        ("K", "dy_k", "dK", "rel_k"),
        ("V", "dy_v", "dV", "rel_v"),
    ):
        total_dy = sum(x[f"{dykey}_fro"] ** 2 for x in per_head)
        total_da = sum(x["da"] ** 2 for x in per_head)
        for h, x in enumerate(per_head):
            dy_block = x[dykey]
            d_block = x[dkey]
            rel = fl((dy_block - d_block).norm() / max(dy_block.norm(), 1e-30))
            rows.append({
                "type": kind, "head": h,
                "dy_fro": fl(dy_block.norm()), "d_fro": fl(d_block.norm()),
                "rel": rel,
                "cos": cos_abs(dy_block, d_block),
                "da_fro": x["da"], "dp_fro": x["dp"], "ds_fro": x["ds"],
                "a_entropy": x["ent"], "a_max": x["amax"],
                "q_fro": x["q"], "k_fro": x["k"], "v_fro": x["v"],
                "dy_share": (x[f"{dykey}_fro"] ** 2 / total_dy) if total_dy > 0 else 0.0,
                "da_share": (x["da"] ** 2 / total_da) if total_da > 0 else 0.0,
            })
    return rows

def aggregate(rows_list):
    keys = [k for k in FIELDS if k not in ("seed", "node", "loss", "layer", "type", "head")]
    out = []
    for kind in ("Q", "K", "V"):
        for h in range(6):
            vals = [r for rows in rows_list for r in rows if r["type"] == kind and r["head"] == h]
            d = {k: float(np.nanmean([r[k] for r in vals])) for k in keys}
            d["type"] = kind; d["head"] = h
            out.append(d)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--nodes", default=",".join(map(str, NODES)))
    parser.add_argument("--probe-windows", type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    nodes = tuple(int(x) for x in args.nodes.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    rows = []
    summary = {"nodes": list(nodes), "seeds": list(seeds), "runs": []}

    for seed in seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
        manifest = load_manifest(Path(args.data_root) / "manifest.json")
        train = FrozenWindows(manifest, "train", 256)
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        set_math(); model.eval()
        for node in nodes:
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            node_dir = Path(args.ckpt_root) / seed / "node_checkpoints"
            ckpt = torch.load(node_dir / f"pre_u{node:06d}.pt", map_location="cpu", weights_only=False)
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

            for loss_name, groups in (("train", micro),
                                      ("probe", [list(range(args.probe_windows))])):
                per_layer = {l: [] for l in range(6)}
                for group in groups:
                    cap = Capture(model)
                    cap.register()
                    x, y = batch(group, "train" if loss_name == "train" else "val")
                    model.zero_grad(set_to_none=True)
                    loss = model(x, y)["loss"]
                    (loss / len(groups)).backward() if loss_name == "train" else loss.backward()
                    cap.remove()
                    for layer in range(6):
                        per_layer[layer].append(compute_metrics(model, cap, layer, loss_name))
                    del x, y, loss, cap
                    torch.cuda.empty_cache()
                for layer in range(6):
                    vals = aggregate(per_layer[layer])
                    for d in vals:
                        rows.append({"seed": seed, "node": node, "loss": loss_name,
                                     "layer": layer, **d})
            summary["runs"].append({"seed": seed, "node": node,
                                    "seconds": time.time() - started,
                                    "peak_MiB": torch.cuda.max_memory_allocated() / 1024 ** 2})
            print(f"[stage5a2] {seed} node {node}", flush=True)
            del ckpt, sidecar
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    with (out / "attention_dynamics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "COMPLETE").write_text("ok\n")
    print(f"[stage5a2] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
