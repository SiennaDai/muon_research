#!/usr/bin/env python3
"""Stage 5 Module C — Muon role dynamics and Head-subspace loss contribution.

Part 1 (CPU): read the 20 role files per seed and quantify G -> M -> D -> U -> ΔW
              per head.
Part 2 (forward-only): at the event nodes 784/1312, apply the actual ΔW of a
              Head subspace, measure ΔL, and compare it with the Taylor
              prediction built from B's q_h/q_hk and the role G.

No Hessian-vector product and no double backward are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform/src"))

from config.recipe import load_recipe  # noqa: E402
from data.frozen_tokens import FrozenWindows, load_manifest  # noqa: E402
from train_platform import build_model  # noqa: E402

SEEDS = ("seed0", "seed1")
TYPES = ("Q", "K", "V", "O")
LAYERS = {"seed0": (0, 2, 3), "seed1": (0, 3, 4)}
HEAD_DIM = 64
DEFAULT_B = ROOT / "analysis/structural_emergence/stage5_b_metrics"
ROLE_FIELDS = [
    "seed", "node", "layer", "type", "head",
    "g_fro", "m_fro", "d_fro", "u_fro", "dw_fro",
    "g_share", "u_share", "dw_share",
    "cos_g_u", "cos_u_dw", "cos_g_dw", "ratio_u_g", "ratio_dw_u",
]
LOSS_FIELDS = [
    "seed", "node", "layer", "subspace", "n_blocks",
    "blocks", "L_base", "delta_L_actual", "first_order", "second_order",
    "delta_L_taylor", "residual",
]


def set_math_sdpa():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def block(tensor, kind, head):
    if kind == "O":
        return tensor[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
    base = {"Q": 0, "K": 384, "V": 768}[kind]
    return tensor[base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM, :]


def add_block(full, value, kind, head):
    if kind == "O":
        full[:, head * HEAD_DIM:(head + 1) * HEAD_DIM] += value
    else:
        base = {"Q": 0, "K": 384, "V": 768}[kind]
        full[base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM, :] += value


def name_for(layer, kind):
    return (f"transformer.h.{layer}.attn.c_proj.weight" if kind == "O"
            else f"transformer.h.{layer}.attn.c_attn.weight")


def cosine(a, b):
    denom = a.norm() * b.norm()
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def role_dynamics(ckpt_root, seeds):
    rows = []
    for seed in seeds:
        directory = Path(ckpt_root) / seed / "node_checkpoints"
        for path in sorted(directory.glob("pre_u*_roles.pt")):
            node = int(path.stem.split("_u")[1].split("_")[0])
            roles = torch.load(path, map_location="cpu", weights_only=False)["tensors"]
            for layer in range(6):
                for kind in TYPES:
                    name = name_for(layer, kind)
                    local = []
                    for head in range(6):
                        g = block(roles[f"{name}|G"], kind, head).float()
                        m = block(roles[f"{name}|M_t_candidate"], kind, head).float()
                        d = block(roles[f"{name}|D"], kind, head).float()
                        u = block(roles[f"{name}|U"], kind, head).float()
                        wb = block(roles[f"{name}|W_before"], kind, head).float()
                        wa = block(roles[f"{name}|W_after"], kind, head).float()
                        dw = wa - wb
                        local.append({
                            "head": head,
                            "g": float(g.norm()), "m": float(m.norm()),
                            "d": float(d.norm()), "u": float(u.norm()),
                            "dw": float(dw.norm()),
                            "cos_g_u": cosine(g, u), "cos_u_dw": cosine(u, dw),
                            "cos_g_dw": cosine(g, dw),
                        })
                    gsum = sum(x["g"] ** 2 for x in local)
                    usum = sum(x["u"] ** 2 for x in local)
                    dwsum = sum(x["dw"] ** 2 for x in local)
                    for x in local:
                        rows.append({
                            "seed": seed, "node": node, "layer": layer,
                            "type": kind, "head": x["head"],
                            "g_fro": x["g"], "m_fro": x["m"], "d_fro": x["d"],
                            "u_fro": x["u"], "dw_fro": x["dw"],
                            "g_share": x["g"] ** 2 / gsum if gsum > 0 else 0.0,
                            "u_share": x["u"] ** 2 / usum if usum > 0 else 0.0,
                            "dw_share": x["dw"] ** 2 / dwsum if dwsum > 0 else 0.0,
                            "cos_g_u": x["cos_g_u"], "cos_u_dw": x["cos_u_dw"],
                            "cos_g_dw": x["cos_g_dw"],
                            "ratio_u_g": x["u"] / max(x["g"], 1e-30),
                            "ratio_dw_u": x["dw"] / max(x["u"], 1e-30),
                        })
    return rows


def read_b(b_dir):
    direction = list(csv.DictReader((Path(b_dir) / "direction_curvature.csv").open()))
    cross = list(csv.DictReader((Path(b_dir) / "cross_terms.csv").open()))
    top = {}
    for r in direction:
        if r["loss"] != "probe":
            continue
        key = (r["seed"], int(r["node"]), int(r["layer"]), r["type"])
        value = float(r["kappa"])
        if key not in top or value > top[key][1]:
            top[key] = (int(r["head"]), value)
    q = {}
    for r in cross:
        if r["loss"] != "probe":
            continue
        key = (r["seed"], int(r["node"]), int(r["layer"]),
               r["h_type"], int(r["h_head"]), r["k_type"], int(r["k_head"]))
        q[key] = float(r["q_hk"])
    return top, q


def head_loss(ckpt_root, data_root, b_dir, seeds, events, probe_windows=8):
    top, qmap = read_b(b_dir)
    device = torch.device("cuda:0")
    rows = []
    for seed in seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
        manifest = load_manifest(Path(data_root) / "manifest.json")
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        params = [p for p in model.parameters() if p.requires_grad]
        set_math_sdpa()
        model.eval()
        for node in events:
            node_dir = Path(ckpt_root) / seed / "node_checkpoints"
            ckpt = torch.load(node_dir / f"pre_u{node:06d}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            roles = torch.load(node_dir / f"pre_u{node:06d}_roles.pt",
                               map_location="cpu", weights_only=False)["tensors"]
            pairs = [val.window(i) for i in range(probe_windows)]
            x = torch.stack([p[0] for p in pairs]).to(device)
            y = torch.stack([p[1] for p in pairs]).to(device)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                grad_loss = model(x, y)["loss"]
                grad_loss.backward()
            probe_g = {}
            for layer in LAYERS[seed]:
                g_attn = model.get_parameter(name_for(layer, "Q")).grad.detach()
                g_proj = model.get_parameter(name_for(layer, "O")).grad.detach()
                for kind in TYPES:
                    source = g_proj if kind == "O" else g_attn
                    for head in range(6):
                        probe_g[(layer, kind, head)] = block(source, kind, head).detach()
            model.zero_grad(set_to_none=True)
            del grad_loss
            with torch.no_grad():
                base = float(model(x, y)["loss"].detach())
            for layer in LAYERS[seed]:
                spec = {}
                for kind in TYPES:
                    h, _ = top.get((seed, node, layer, kind), (0, 0.0))
                    spec[f"all_{kind}"] = [(kind, hh) for hh in range(6)]
                    spec[f"top_{kind}"] = [(kind, h)]
                spec["all_QKV"] = [(k, hh) for k in ("Q", "K", "V") for hh in range(6)]
                spec["all_attn"] = [(k, hh) for k in TYPES for hh in range(6)]
                for label, blocks in spec.items():
                    deltas = {}
                    first = 0.0
                    for kind, head in blocks:
                        param_name = name_for(layer, kind)
                        delta_block = (block(roles[f"{param_name}|W_after"], kind, head).float()
                                       - block(roles[f"{param_name}|W_before"], kind, head).float())
                        first += float((probe_g[(layer, kind, head)] *
                                        delta_block.to(device)).sum())
                        if param_name not in deltas:
                            deltas[param_name] = torch.zeros_like(model.get_parameter(param_name).detach())
                        add_block(deltas[param_name], delta_block.to(device), kind, head)
                    second = 0.0
                    for hkind, hhead in blocks:
                        for kkind, khead in blocks:
                            key = (seed, node, layer, hkind, hhead, kkind, khead)
                            if key in qmap:
                                second += qmap[key]
                    second *= 0.5
                    originals = {}
                    with torch.no_grad():
                        for pname, delta in deltas.items():
                            param = model.get_parameter(pname)
                            originals[pname] = param.detach().clone()
                            param.add_(delta)
                        actual = float(model(x, y)["loss"].detach()) - base
                        for pname, original in originals.items():
                            model.get_parameter(pname).copy_(original)
                    rows.append({
                        "seed": seed, "node": node, "layer": layer,
                        "subspace": label, "n_blocks": len(blocks),
                        "blocks": ";".join(f"{k}H{h}" for k, h in blocks),
                        "L_base": base, "delta_L_actual": actual,
                        "first_order": first, "second_order": second,
                        "delta_L_taylor": first + second,
                        "residual": actual - (first + second),
                    })
        del model, params
        torch.cuda.empty_cache()
    return rows


def write_csv(path, fields, rows):
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
    parser.add_argument("--b-dir", default=str(DEFAULT_B))
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--events", default="784,1312")
    parser.add_argument("--probe-windows", type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    events = tuple(int(x) for x in args.events.split(","))
    if any(s not in SEEDS for s in seeds):
        parser.error("--seeds must be subset of seed0,seed1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    role_rows = role_dynamics(args.ckpt_root, seeds)
    write_csv(out / "role_dynamics.csv", ROLE_FIELDS, role_rows)
    print(f"[stage5c] role rows {len(role_rows)}", flush=True)
    loss_rows = head_loss(args.ckpt_root, args.data_root, args.b_dir, seeds, events,
                          args.probe_windows)
    write_csv(out / "head_loss.csv", LOSS_FIELDS, loss_rows)
    (out / "run_summary.json").write_text(json.dumps({
        "seeds": list(seeds), "events": list(events),
        "role_rows": len(role_rows), "loss_rows": len(loss_rows),
    }, indent=2))
    (out / "COMPLETE").write_text("ok\n")
    print(f"[stage5c] loss rows {len(loss_rows)}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
