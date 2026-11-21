#!/usr/bin/env python3
"""Stage 5 C-2 — QK vs VO cross-Hessian and actual loss interaction.

At event nodes 784/1312 and B-covered layers L0/L3, evaluate:
  QK = all Q/K heads; VO = all V/O heads; joint = QK+VO.

  I_H      = sum_{h in QK, k in VO} q_hk = DeltaW_QK^T H DeltaW_VO
  I_actual = DeltaL_joint - DeltaL_QK - DeltaL_VO
  I_taylor = Taylor_joint - Taylor_QK - Taylor_VO

No HVP; only forward-only actual loss plus B's existing q_hk.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform/src"))

from config.recipe import load_recipe  # noqa: E402
from data.frozen_tokens import FrozenWindows, load_manifest  # noqa: E402
from train_platform import build_model  # noqa: E402

TYPES = ("Q", "K", "V", "O")
HEAD_DIM = 64
DEFAULT_B = ROOT / "analysis/structural_emergence/stage5_b_metrics"
FIELDS = [
    "seed", "node", "layer", "L_base",
    "q_QK", "q_VO", "I_H", "rho_H", "rho_H_reliable",
    "dw_QK_fro", "dw_VO_fro", "rho_scale",
    "first_QK", "first_VO", "first_joint",
    "second_QK", "second_VO", "second_joint",
    "taylor_QK", "taylor_VO", "taylor_joint", "I_taylor",
    "actual_QK", "actual_VO", "actual_joint", "I_actual", "I_residual",
    "I_actual_norm",
]


def set_math():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def block(t, kind, head):
    if kind == "O":
        return t[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
    base = {"Q": 0, "K": 384, "V": 768}[kind]
    return t[base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM, :]


def add_block(full, val, kind, head):
    if kind == "O":
        full[:, head * HEAD_DIM:(head + 1) * HEAD_DIM] += val
    else:
        base = {"Q": 0, "K": 384, "V": 768}[kind]
        full[base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM, :] += val


def name_for(layer, kind):
    return (f"transformer.h.{layer}.attn.c_proj.weight" if kind == "O"
            else f"transformer.h.{layer}.attn.c_attn.weight")


def load_q(b_dir):
    q = {}
    for r in csv.DictReader((Path(b_dir) / "cross_terms.csv").open()):
        if r["loss"] != "probe":
            continue
        q[(r["seed"], int(r["node"]), int(r["layer"]),
           r["h_type"], int(r["h_head"]), r["k_type"], int(r["k_head"]))] = float(r["q_hk"])
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--b-dir", default=str(DEFAULT_B))
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--events", default="784,1312")
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--probe-windows", type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    events = tuple(int(x) for x in args.events.split(","))
    layers = tuple(int(x) for x in args.layers.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qmap = load_q(args.b_dir)
    device = torch.device("cuda:0")
    rows = []

    for seed in seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
        manifest = load_manifest(Path(args.data_root) / "manifest.json")
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        set_math(); model.eval()
        for node in events:
            node_dir = Path(args.ckpt_root) / seed / "node_checkpoints"
            ckpt = torch.load(node_dir / f"pre_u{node:06d}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            roles = torch.load(node_dir / f"pre_u{node:06d}_roles.pt",
                               map_location="cpu", weights_only=False)["tensors"]
            pairs = [val.window(i) for i in range(args.probe_windows)]
            x = torch.stack([p[0] for p in pairs]).to(device)
            y = torch.stack([p[1] for p in pairs]).to(device)

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = model(x, y)["loss"]
                loss.backward()
            probe_g = {}
            for layer in layers:
                ga = model.get_parameter(name_for(layer, "Q")).grad.detach()
                gp = model.get_parameter(name_for(layer, "O")).grad.detach()
                for kind in TYPES:
                    source = gp if kind == "O" else ga
                    for head in range(6):
                        probe_g[(layer, kind, head)] = block(source, kind, head).detach()
            model.zero_grad(set_to_none=True)
            del loss
            with torch.no_grad():
                base = float(model(x, y)["loss"].detach())

            for layer in layers:
                qk = [(k, h) for k in ("Q", "K") for h in range(6)]
                vo = [(k, h) for k in ("V", "O") for h in range(6)]
                joint = qk + vo
                groups = {"QK": qk, "VO": vo, "joint": joint}
                first = {}
                taylor = {}
                actual = {}
                qsum = {}
                dwnorm = {}
                for label, blocks in groups.items():
                    first[label] = 0.0
                    qsum[label] = 0.0
                    dwnorm[label] = 0.0
                    for kind, head in blocks:
                        pname = name_for(layer, kind)
                        dw = (block(roles[f"{pname}|W_after"], kind, head).float()
                              - block(roles[f"{pname}|W_before"], kind, head).float())
                        first[label] += float((probe_g[(layer, kind, head)] *
                                               dw.to(device)).sum())
                        dwnorm[label] += float(dw.float().norm() ** 2)
                    for hk, hh in blocks:
                        for kk, kh in blocks:
                            qsum[label] += qmap.get((seed, node, layer, hk, hh, kk, kh), 0.0)
                    dwnorm[label] = dwnorm[label] ** 0.5
                    taylor[label] = first[label] + 0.5 * qsum[label]
                # Actual loss for each group.
                for label, blocks in groups.items():
                    deltas = {}
                    for kind, head in blocks:
                        pname = name_for(layer, kind)
                        dw = (block(roles[f"{pname}|W_after"], kind, head).float()
                              - block(roles[f"{pname}|W_before"], kind, head).float())
                        if pname not in deltas:
                            deltas[pname] = torch.zeros_like(model.get_parameter(pname).detach())
                        add_block(deltas[pname], dw.to(device), kind, head)
                    originals = {}
                    with torch.no_grad():
                        for pname, delta in deltas.items():
                            param = model.get_parameter(pname)
                            originals[pname] = param.detach().clone()
                            param.add_(delta)
                        actual[label] = float(model(x, y)["loss"].detach()) - base
                        for pname, original in originals.items():
                            model.get_parameter(pname).copy_(original)
                q_qk = qsum["QK"]; q_vo = qsum["VO"]
                I_H = 0.0
                for kh, hh in qk:
                    for kv, hv in vo:
                        I_H += qmap.get((seed, node, layer, kh, hh, kv, hv), 0.0)
                denom = (abs(q_qk * q_vo)) ** 0.5
                I_taylor = taylor["joint"] - taylor["QK"] - taylor["VO"]
                I_actual = actual["joint"] - actual["QK"] - actual["VO"]
                scale = abs(actual["QK"]) + abs(actual["VO"]) + 1e-30
                rows.append({
                    "seed": seed, "node": node, "layer": layer, "L_base": base,
                    "q_QK": q_qk, "q_VO": q_vo, "I_H": I_H,
                    "rho_H": I_H / denom if denom > 0 else float("nan"),
                    "rho_H_reliable": bool(abs(q_qk) > 1e-12 and abs(q_vo) > 1e-12),
                    "dw_QK_fro": dwnorm["QK"], "dw_VO_fro": dwnorm["VO"],
                    "rho_scale": I_H / (dwnorm["QK"] * dwnorm["VO"]) if dwnorm["QK"] * dwnorm["VO"] > 0 else float("nan"),
                    "first_QK": first["QK"], "first_VO": first["VO"],
                    "first_joint": first["joint"],
                    "second_QK": 0.5 * q_qk, "second_VO": 0.5 * q_vo,
                    "second_joint": 0.5 * qsum["joint"],
                    "taylor_QK": taylor["QK"], "taylor_VO": taylor["VO"],
                    "taylor_joint": taylor["joint"], "I_taylor": I_taylor,
                    "actual_QK": actual["QK"], "actual_VO": actual["VO"],
                    "actual_joint": actual["joint"], "I_actual": I_actual,
                    "I_residual": I_actual - I_taylor,
                    "I_actual_norm": I_actual / scale,
                })
            print(f"[stage5c2] {seed} node {node}: {len(layers)} layers", flush=True)
        del model
        torch.cuda.empty_cache()

    with (out / "cross_interaction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (out / "run_summary.json").write_text(json.dumps({
        "seeds": list(seeds), "events": list(events), "layers": list(layers),
        "rows": len(rows),
    }, indent=2))
    (out / "COMPLETE").write_text("ok\n")
    print(f"[stage5c2] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
