#!/usr/bin/env python3
"""Stage 5 Module A — gradient/representation dynamics collection.

For every Stage 4 pre-update node, replay the exact update batch and a fixed
probe batch.  Capture the attention input X, c_proj input A, the backward
signals dY (Q/K/V) and dO (O), and derive per-head G = dY_h^T X (or dO_h^T A).
Only spectra, top directions, norms and energy shares are saved; raw
8192x384/1152 activations are not written.

Run with the repo on Kaggle:
    python -u analysis/structural_emergence/se_stage5_a_collect.py \
        --ckpt-root .../structural_concentration_ckpt \
        --data-root .../slimpajama_8x \
        --out /kaggle/working/stage5_a
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
from train_platform import build_model  # noqa: E402

SEEDS = ("seed0", "seed1")
NODES = (0, 8, 16, 32, 48, 56, 64, 80, 96, 112, 128, 192, 256, 320, 384,
         448, 512, 576, 640, 704, 768, 784, 832, 896, 960, 1024,
         1152, 1280, 1312, 1408, 1536)
TYPES = ("Q", "K", "V", "O")
HEAD_DIM = 64
PROBE_WINDOWS = 8
HEAD_FIELDS = [
    "seed", "node", "loss", "layer", "type", "head",
    "g_fro", "g_share", "g_top1_share", "g_eff_rank", "g_stable_rank",
    "dy_fro", "dy_top1", "dy_eff_rank",
    "input_fro", "input_top1", "input_eff_rank",
    "cos_g_dy", "cos_g_x", "angle_g_dy4", "angle_g_x4",
    "clip_coef", "total_grad_norm",
]
MB_FIELDS = ["seed", "node", "layer", "type", "head", "mb",
             "g_fro", "cos_to_total", "g_share_mb"]
REPLAY_FIELDS = ["seed", "node", "layer", "parameter", "g_rel_error"]


def effective_rank(s: torch.Tensor) -> float:
    p = s.float() ** 2
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    return float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))


def stable_rank(s: torch.Tensor) -> float:
    total = (s.float() ** 2).sum()
    if total <= 0:
        return 0.0
    return float(total / (s[0].float() ** 2 + 1e-30))


def top1_share(s: torch.Tensor) -> float:
    total = (s.float() ** 2).sum()
    return float(s[0].float() ** 2 / total) if total > 0 else 0.0


def subspace_angle(u: torch.Tensor, v: torch.Tensor, k: int = 4) -> float:
    if u.shape[1] < 2 or v.shape[1] < 2:
        return float("nan")
    k = min(k, u.shape[1], v.shape[1])
    qu = torch.linalg.qr(u[:, :k]).Q
    qv = torch.linalg.qr(v[:, :k]).Q
    s = torch.linalg.svdvals(qu.T @ qv).clamp(-1.0, 1.0)
    return float(torch.acos(s).max())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    return float((a * b).abs().sum() / denom) if denom > 0 else float("nan")


def split_qkv(dy: torch.Tensor):
    """Return Q/K/V head gradients from c_attn output gradient [B,T,1152]."""
    n = dy.shape[0] * dy.shape[1]
    dy = dy.reshape(n, 1152)
    out = {}
    for kind, base in (("Q", 0), ("K", 384), ("V", 768)):
        for head in range(6):
            out[(kind, head)] = dy[:, base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM]
    return out


def split_o(do: torch.Tensor):
    n = do.shape[0] * do.shape[1]
    do = do.reshape(n, 384)
    return {(("O"), head): do[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
            for head in range(6)}


class Capture:
    """Forward inputs and output gradients for every attention block."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.x = {}
        self.a = {}
        self.dy = {}
        self.do = {}

    def register(self):
        for layer, block in enumerate(self.model.transformer.h):
            self.handles.append(block.attn.c_attn.register_forward_hook(
                self._forward(layer, "c_attn")))
            self.handles.append(block.attn.c_proj.register_forward_hook(
                self._forward(layer, "c_proj")))

    def _forward(self, layer, kind):
        def hook(module, inputs, output):
            if kind == "c_attn":
                self.x[layer] = inputs[0].detach()
                output.register_hook(lambda grad: self._store(layer, "dy", grad))
            else:
                self.a[layer] = inputs[0].detach()
                output.register_hook(lambda grad: self._store(layer, "do", grad))
        return hook

    def _store(self, layer, name, grad):
        if grad is not None:
            getattr(self, name)[layer] = grad.detach()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compute_g(datas):
    """Compute per-head G from one or more captured passes."""
    if len(datas) == 1:
        x = datas[0].x
        a = datas[0].a
        dy = datas[0].dy
        do = datas[0].do
    else:
        x = {l: torch.cat([d.x[l] for d in datas], dim=0) for l in datas[0].x}
        a = {l: torch.cat([d.a[l] for d in datas], dim=0) for l in datas[0].a}
        dy = {l: torch.cat([d.dy[l] for d in datas], dim=0) for l in datas[0].dy}
        do = {l: torch.cat([d.do[l] for d in datas], dim=0) for l in datas[0].do}
    g = {}
    for layer in sorted(x):
        xm = x[layer].reshape(-1, 384).float()
        am = a[layer].reshape(-1, 384).float()
        for key, dh in split_qkv(dy[layer]).items():
            g[(layer, key[0], key[1])] = dh.float().T @ xm
        for key, dh in split_o(do[layer]).items():
            g[(layer, key[0], key[1])] = dh.float().T @ am
    return g, x, a, dy, do


def svd_metrics(mat: torch.Tensor, k: int):
    u, s, vh = torch.linalg.svd(mat.float(), full_matrices=False)
    return u[:, :k], s[:k], vh[:k], s


def collect_pass(model, loss_fn, params, cfg):
    model.zero_grad(set_to_none=True)
    cap = Capture(model)
    cap.register()
    loss = loss_fn()
    loss.backward()
    cap.remove()
    data = cap
    total_norm = torch.sqrt(sum((p.grad.detach().float() ** 2).sum()
                                for p in params if p.grad is not None))
    g, x, a, dy, do = compute_g([data])
    return {
        "g": g, "x": x, "a": a, "dy": dy, "do": do,
        "total_norm": float(total_norm), "coef": 1.0,
    }


def collect_train_micro(model, groups, batch_fn, params, cfg):
    model.zero_grad(set_to_none=True)
    datas = []
    g_mb = []
    for group in groups:
        cap = Capture(model)
        cap.register()
        x, y = batch_fn(group)
        loss = model(x, y)["loss"]
        (loss / len(groups)).backward()
        cap.remove()
        datas.append(cap)
        g, _, _, _, _ = compute_g([cap])
        g_mb.append(g)
        del x, y, loss
    total_norm = torch.sqrt(sum((p.grad.detach().float() ** 2).sum()
                                for p in params if p.grad is not None))
    coef = min(1.0, float(cfg["train"]["grad_clip_norm"]) / float(total_norm))
    g, x, a, dy, do = compute_g(datas)
    return {
        "g": g, "g_mb": g_mb, "x": x, "a": a, "dy": dy, "do": do,
        "total_norm": float(total_norm), "coef": coef,
    }


def head_metrics(pass_data, loss_name, seed, node, clip_coef, total_norm):
    g = pass_data["g"]
    x = pass_data["x"]
    a = pass_data["a"]
    dy = pass_data["dy"]
    do = pass_data["do"]
    rows = []
    spectra = {}
    for layer in sorted(x):
        xm = x[layer].reshape(-1, 384).float()
        am = a[layer].reshape(-1, 384).float()
        xu, xs, xv, xfull = svd_metrics(xm, 16)
        au, ass, av, afull = svd_metrics(am, 16)
        spectra[f"{loss_name}|L{layer}|x|s"] = xs.cpu().numpy()
        spectra[f"{loss_name}|L{layer}|x|v"] = xv.cpu().numpy()
        spectra[f"{loss_name}|L{layer}|a|s"] = ass.cpu().numpy()
        spectra[f"{loss_name}|L{layer}|a|v"] = av.cpu().numpy()
        input_x = {
            "fro": float(xm.norm()), "top1": float(xfull[0]), "eff": effective_rank(xfull),
            "u": xv.T,
        }
        input_a = {
            "fro": float(am.norm()), "top1": float(afull[0]), "eff": effective_rank(afull),
            "u": av.T,
        }
        shares = {}
        for type_name in TYPES:
            vals = [float(g[(layer, type_name, h)].norm() ** 2) for h in range(6)]
            total = sum(vals)
            for h, value in enumerate(vals):
                shares[(type_name, h)] = value / total if total > 0 else 0.0
        for type_name in TYPES:
            for head in range(6):
                G = g[(layer, type_name, head)]
                gu, gs, gv, gfull = svd_metrics(G, 8)
                if type_name == "O":
                    dh = do[layer].reshape(-1, 384).float()[:, head * 64:(head + 1) * 64]
                    inp = input_a
                else:
                    base = {"Q": 0, "K": 384, "V": 768}[type_name]
                    dh = dy[layer].reshape(-1, 1152).float()[
                        :, base + head * 64:base + (head + 1) * 64]
                    inp = input_x
                # dY_h is [N,64]; transpose so its left singular vectors
                # live in the 64-dimensional feature space and align with G.
                du, ds, dv, dfull = svd_metrics(dh.T, 8)
                g_fro = float(G.norm())
                spectra[f"{loss_name}|L{layer}|{type_name}|H{head}|dy|s"] = ds.cpu().numpy()
                spectra[f"{loss_name}|L{layer}|{type_name}|H{head}|dy|u"] = du[:, :8].T.cpu().numpy()
                spectra[f"{loss_name}|L{layer}|{type_name}|H{head}|g|s"] = gs.cpu().numpy()
                spectra[f"{loss_name}|L{layer}|{type_name}|H{head}|g|u"] = gu[:, :8].T.cpu().numpy()
                spectra[f"{loss_name}|L{layer}|{type_name}|H{head}|g|v"] = gv.cpu().numpy()
                rows.append({
                    "seed": seed, "node": node, "loss": loss_name,
                    "layer": layer, "type": type_name, "head": head,
                    "g_fro": g_fro, "g_share": shares[(type_name, head)],
                    "g_top1_share": top1_share(gfull), "g_eff_rank": effective_rank(gfull),
                    "g_stable_rank": stable_rank(gfull),
                    "dy_fro": float(dh.norm()), "dy_top1": float(dfull[0]),
                    "dy_eff_rank": effective_rank(dfull),
                    "input_fro": inp["fro"], "input_top1": inp["top1"],
                    "input_eff_rank": inp["eff"],
                    "cos_g_dy": cosine(gu[:, 0], du[:, 0]),
                    "cos_g_x": cosine(gv[0], inp["u"][:, 0]),
                    "angle_g_dy4": subspace_angle(gu, du, 4),
                    "angle_g_x4": subspace_angle(gv.T, inp["u"], 4),
                    "clip_coef": clip_coef, "total_grad_norm": total_norm,
                })
    return rows, spectra


def _mb_rows(pass_data, g_total, seed, node):
    rows = []
    g_mb = pass_data["g_mb"]
    for layer in sorted(pass_data["x"]):
        for type_name in TYPES:
            shares = []
            for mb in range(len(g_mb)):
                vals = [float(g_mb[mb][(layer, type_name, h)].norm() ** 2) for h in range(6)]
                total = sum(vals)
                shares.append([v / total if total > 0 else 0.0 for v in vals])
            for head in range(6):
                total = g_total[(layer, type_name, head)]
                for mb in range(len(g_mb)):
                    gm = g_mb[mb][(layer, type_name, head)]
                    denom = float(gm.norm() * total.norm())
                    cos = float((gm * total).sum() / denom) if denom > 0 else float("nan")
                    rows.append({
                        "seed": seed, "node": node, "layer": layer,
                        "type": type_name, "head": head, "mb": mb,
                        "g_fro": float(gm.norm()), "cos_to_total": cos,
                        "g_share_mb": shares[mb][head],
                    })
    return rows


def _replay_rows(pass_data, coef, seed, node, role_path):
    if not role_path.is_file():
        return []
    roles = torch.load(role_path, map_location="cpu", weights_only=False)
    rows = []
    for layer in sorted(pass_data["x"]):
        attn = torch.zeros(1152, 384)
        proj = torch.zeros(384, 384)
        for head in range(6):
            base = {"Q": 0, "K": 384, "V": 768}
            for kind in ("Q", "K", "V"):
                attn[base[kind] + head * 64:base[kind] + (head + 1) * 64, :] = \
                    pass_data["g"][(layer, kind, head)].cpu()
            proj[:, head * 64:(head + 1) * 64] = pass_data["g"][(layer, "O", head)].cpu()
        for name, matrix in (
            (f"transformer.h.{layer}.attn.c_attn.weight", attn),
            (f"transformer.h.{layer}.attn.c_proj.weight", proj),
        ):
            key = name + "|G"
            if key not in roles["tensors"]:
                continue
            stored = roles["tensors"][key].float()
            rel = float((coef * matrix - stored).norm() / stored.norm())
            rows.append({"seed": seed, "node": node, "layer": layer,
                         "parameter": name, "g_rel_error": rel})
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--nodes", default=",".join(map(str, NODES)))
    parser.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    args = parser.parse_args()
    args.seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    args.nodes = tuple(int(x) for x in args.nodes.split(","))
    if any(s not in SEEDS for s in args.seeds):
        parser.error("--seeds must be subset of seed0,seed1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(exist_ok=True)
    device = torch.device("cuda:0")
    head_rows, mb_rows, replay_rows = [], [], []
    summary = {"nodes": list(args.nodes), "seeds": list(args.seeds),
               "probe_windows": args.probe_windows, "runs": []}

    def flush():
        write_csv(out / "head_dynamics.csv", HEAD_FIELDS, head_rows)
        write_csv(out / "microbatch.csv", MB_FIELDS, mb_rows)
        write_csv(out / "replay_validation.csv", REPLAY_FIELDS, replay_rows)
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))

    for seed in args.seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
        manifest = load_manifest(Path(args.data_root) / "manifest.json")
        train = FrozenWindows(manifest, "train", 256)
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        params = [p for p in model.parameters() if p.requires_grad]
        summary.setdefault("data_fingerprint", manifest["fingerprint"])
        summary.setdefault("recipe_fingerprints", {})[seed] = cfg["fingerprint"]
        for node in args.nodes:
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            node_dir = Path(args.ckpt_root) / seed / "node_checkpoints"
            checkpoint = torch.load(node_dir / f"pre_u{node:06d}.pt",
                                    map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.train()
            sidecar = json.loads((node_dir / f"pre_u{node:06d}.json").read_text())
            micro = sidecar.get("micro_batch_window_ids")
            if not micro:
                ids = sidecar["window_ids"]
                micro = [ids[i * 4:(i + 1) * 4] for i in range(len(ids) // 4)]
            def batch(ids, split="train"):
                source = train if split == "train" else val
                pairs = [source.window(i) for i in ids]
                return (torch.stack([p[0] for p in pairs]).to(device),
                        torch.stack([p[1] for p in pairs]).to(device))
            x_probe, y_probe = batch(list(range(args.probe_windows)), "val")
            train_data = collect_train_micro(
                model, micro, lambda ids: batch(ids, "train"), params, cfg)
            probe_data = collect_pass(
                model, lambda: model(x_probe, y_probe)["loss"], params, cfg)
            train_rows, train_spectra = head_metrics(
                train_data, "train", seed, node, train_data["coef"], train_data["total_norm"])
            probe_rows, probe_spectra = head_metrics(
                probe_data, "probe", seed, node, probe_data["coef"], probe_data["total_norm"])
            head_rows.extend(train_rows); head_rows.extend(probe_rows)
            mb_rows.extend(_mb_rows(train_data, train_data["g"], seed, node))
            role_path = node_dir / f"pre_u{node:06d}_roles.pt"
            replay_rows.extend(_replay_rows(train_data, train_data["coef"],
                                            seed, node, role_path))
            spectra = {**train_spectra, **probe_spectra}
            np.savez_compressed(out / "spectra" / f"{seed}_u{node:06d}.npz", **{
                k: v for k, v in spectra.items()
            })
            summary["runs"].append({
                "seed": seed, "node": node,
                "seconds": time.time() - started,
                "peak_MiB": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "train_grad_norm": train_data["total_norm"],
                "train_clip_coef": train_data["coef"],
            })
            flush()
            print(f"[stage5a] {seed} node {node}: {len(train_rows)}+{len(probe_rows)} rows", flush=True)
            del checkpoint, sidecar, x_probe, y_probe, train_data, probe_data
            torch.cuda.empty_cache()
        del model, params
        torch.cuda.empty_cache()
    flush()
    (out / "COMPLETE").write_text("ok\n")
    print(f"[stage5a] wrote {out}")


if __name__ == "__main__":
    main()
