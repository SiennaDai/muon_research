#!/usr/bin/env python3
"""Stage 2 — single-step loss sensitivity of low-precision Muon states.

Modules:
  L1  state sensitivity: G / M_prev / D / U / W
  L2  spectral band -> loss: Top / Middle / Tail on D or U
  L3  type -> loss: QK / VO / Q / K / V / O on D or U
  L4  granularity / bit-width on selected states

All experiments are single-step, fixed-G perturbations at pre-update node
checkpoints.  Only forward passes are used for loss; no training, no HVP.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform" / "src"))

from config.recipe import load_recipe  # noqa: E402
from data.frozen_tokens import FrozenWindows, load_manifest  # noqa: E402
from optim.muon_reference import zeropower_newton_schulz  # noqa: E402
from train_platform import build_model, learning_rate  # noqa: E402

HEAD_DIM = 64
TYPES = ("Q", "K", "V", "O")
PARAMS = {
    "QKV": "transformer.h.{layer}.attn.c_attn.weight",
    "O": "transformer.h.{layer}.attn.c_proj.weight",
}
INT8_MAX = {8: 127.0, 4: 7.0}
BANDS = ("top", "middle", "tail")


def set_math() -> None:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def ns(x: torch.Tensor) -> torch.Tensor:
    return zeropower_newton_schulz(
        x, steps=5, coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7
    ).float()


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def block(t: torch.Tensor, kind: str, head: int) -> torch.Tensor:
    if kind == "O":
        return t[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
    base = {"Q": 0, "K": 384, "V": 768}[kind]
    return t[base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM, :]


def add_block(full: torch.Tensor, val: torch.Tensor, kind: str, head: int) -> None:
    if kind == "O":
        full[:, head * HEAD_DIM:(head + 1) * HEAD_DIM] += val
    else:
        base = {"Q": 0, "K": 384, "V": 768}[kind]
        full[base + head * HEAD_DIM:base + (head + 1) * HEAD_DIM, :] += val


def head_shares(t: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "O":
        blocks = [t[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] for h in range(6)]
    else:
        blocks = [t[h * HEAD_DIM:(h + 1) * HEAD_DIM, :] for h in range(18)]
    energy = torch.tensor([float((b.float() ** 2).sum()) for b in blocks],
                          device=t.device)
    return energy / energy.sum().clamp_min(1e-30)


def tensor_metrics(ref: torch.Tensor, q: torch.Tensor, kind: str) -> dict:
    ref = ref.float()
    q = q.float()
    rn = float(ref.norm())
    qn = float(q.norm())
    rel = float((q - ref).norm()) / rn if rn > 0 else float("nan")
    cos = (float((ref * q).sum() / (rn * qn))
           if rn > 0 and qn > 0 else float("nan"))
    pr = head_shares(ref, kind)
    pq = head_shares(q, kind)
    return {
        "rel": rel,
        "cos": cos,
        "share_L1": float((pq - pr).abs().sum()),
        "top1": int(int(pq.argmax()) == int(pr.argmax())),
    }


def aggregate_metrics(refs: dict, qs: dict) -> dict:
    """Aggregate QKV/O metrics: combined relative error, mean share_L1/top1."""
    num = 0.0
    den = 0.0
    cos_num = 0.0
    share = []
    top1 = []
    for kind in ("QKV", "O"):
        ref = refs[kind].float()
        q = qs[kind].float()
        num += float((q - ref).pow(2).sum())
        den += float(ref.pow(2).sum())
        rn = float(ref.norm())
        qn = float(q.norm())
        if rn > 0 and qn > 0:
            cos_num += float((ref * q).sum() / (rn * qn))
        m = tensor_metrics(ref, q, kind)
        share.append(m["share_L1"])
        top1.append(m["top1"])
    return {
        "rel": math.sqrt(num / den) if den > 0 else float("nan"),
        "cos": cos_num / 2.0,
        "share_L1": sum(share) / 2.0,
        "top1": sum(top1) / 2.0,
    }


def quant_int8_heads(t: torch.Tensor, kind: str, bits: int) -> torch.Tensor:
    qmax = INT8_MAX[bits]
    out = t.clone()
    n = 18 if kind == "QKV" else 6
    for h in range(n):
        if kind == "QKV":
            b = out[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]
        else:
            b = out[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        s = float(b.abs().max()) / qmax
        if s <= 0:
            continue
        q = (b / s).round().clamp(-qmax, qmax) * s
        if kind == "QKV":
            out[h * HEAD_DIM:(h + 1) * HEAD_DIM, :] = q
        else:
            out[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] = q
    return out


def quant_int8_blocks(t: torch.Tensor, kind: str, bits: int, block_size: int,
                      align: bool = True) -> torch.Tensor:
    qmax = INT8_MAX[bits]
    out = t.clone()
    if kind == "QKV":
        total, axis = t.shape[0], 0
    else:
        total, axis = t.shape[1], 1
    starts = list(range(0, total, block_size))
    if align:
        starts = [s for s in starts if s % HEAD_DIM == 0]
    for start in starts:
        end = min(start + block_size, total)
        if axis == 0:
            b = out[start:end, :]
        else:
            b = out[:, start:end]
        s = float(b.abs().max()) / qmax
        if s <= 0:
            continue
        q = (b / s).round().clamp(-qmax, qmax) * s
        if axis == 0:
            out[start:end, :] = q
        else:
            out[:, start:end] = q
    return out


def quant_int8_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = INT8_MAX[bits]
    s = float(t.abs().max()) / qmax
    if s <= 0:
        return t.clone()
    return (t / s).round().clamp(-qmax, qmax) * s


def quantize(t: torch.Tensor, kind: str, fmt: str, bits: int = 8,
             block_size: int = 64, align: bool = True) -> torch.Tensor:
    if fmt == "bf16":
        return round_bf16(t)
    if fmt == "int8_head":
        return quant_int8_heads(t, kind, bits)
    if fmt == "int8_block":
        return quant_int8_blocks(t, kind, bits, block_size, align)
    if fmt == "int8_tensor":
        return quant_int8_tensor(t, bits)
    raise ValueError(fmt)


def split_bands(s: torch.Tensor, cuts=(0.5, 0.9)) -> dict:
    import numpy as np
    e = s.detach().cpu().numpy().astype("float64") ** 2
    c = np.cumsum(e) / e.sum()
    top = np.where(c <= cuts[0])[0]
    mid = np.where((c > cuts[0]) & (c <= cuts[1]))[0]
    tail = np.where(c > cuts[1])[0]
    if len(top) == 0:
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


def project_band(e: torch.Tensor, u: torch.Tensor, vt: torch.Tensor,
                 idx) -> torch.Tensor:
    if len(idx) == 0:
        return torch.zeros_like(e)
    ub = u[:, idx]
    vb = vt[idx, :].T
    return ub @ (ub.T @ e @ vb) @ vb.T


def band_perturb(x: torch.Tensor, ref_d: torch.Tensor, idx, alpha: float,
                 gen: torch.Generator) -> torch.Tensor:
    u, s, vt = torch.linalg.svd(ref_d, full_matrices=False)
    xb = project_band(x, u, vt, idx)
    nb = float(xb.norm())
    if nb <= 0:
        return x.clone()
    e = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    eb = project_band(e, u, vt, idx)
    en = float(eb.norm())
    if en <= 0:
        return x.clone()
    return x + eb * (alpha * nb / en)


def band_perturb_abs(x: torch.Tensor, ref_d: torch.Tensor, idx,
                     alpha_abs: float, target_norm: float,
                     gen: torch.Generator) -> torch.Tensor:
    """Perturb one band with ||E|| = alpha_abs * target_norm."""
    u, s, vt = torch.linalg.svd(ref_d, full_matrices=False)
    target = alpha_abs * target_norm
    if target <= 0:
        return x.clone()
    e = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    eb = project_band(e, u, vt, idx)
    en = float(eb.norm())
    if en <= 0:
        return x.clone()
    return x + eb * (target / en)


def group_mask(kind: str, group: str) -> list[tuple[str, slice]]:
    """Return (type, slice) blocks for a group on QKV or O."""
    if kind == "QKV":
        if group == "QK":
            return [("Q", slice(0, 384)), ("K", slice(384, 768))]
        if group == "VO":
            return [("V", slice(768, 1152))]
        if group == "Q":
            return [("Q", slice(0, 384))]
        if group == "K":
            return [("K", slice(384, 768))]
        if group == "V":
            return [("V", slice(768, 1152))]
        return []
    if group == "VO":
        return [("O", slice(None))]
    if group == "O":
        return [("O", slice(None))]
    return []


def perturb_group(x: torch.Tensor, kind: str, group: str, alpha: float,
                  gen: torch.Generator) -> torch.Tensor:
    out = x.clone()
    for typ, sl in group_mask(kind, group):
        if kind == "QKV":
            b = out[sl, :]
            e = torch.randn(b.shape, generator=gen, device=x.device, dtype=x.dtype)
            nb = float(b.norm())
            if nb > 0:
                out[sl, :] = b + e * (alpha * nb / float(e.norm()))
        else:
            b = out[:, sl]
            e = torch.randn(b.shape, generator=gen, device=x.device, dtype=x.dtype)
            nb = float(b.norm())
            if nb > 0:
                out[:, sl] = b + e * (alpha * nb / float(e.norm()))
    return out


def perturb_group_abs(x: torch.Tensor, kind: str, group: str,
                      alpha_abs: float, target_norm: float,
                      gen: torch.Generator) -> torch.Tensor:
    """Perturb one group with total ||E|| = alpha_abs * target_norm."""
    blocks = group_mask(kind, group)
    if not blocks:
        return x.clone()
    out = x.clone()
    per_block = alpha_abs * target_norm / math.sqrt(len(blocks))
    if per_block <= 0:
        return out
    for typ, sl in blocks:
        if kind == "QKV":
            b = out[sl, :]
            e = torch.randn(b.shape, generator=gen, device=x.device,
                            dtype=x.dtype)
            en = float(e.norm())
            if en > 0:
                out[sl, :] = b + e * (per_block / en)
        else:
            b = out[:, sl]
            e = torch.randn(b.shape, generator=gen, device=x.device,
                            dtype=x.dtype)
            en = float(e.norm())
            if en > 0:
                out[:, sl] = b + e * (per_block / en)
    return out


def load_node(ckpt_root: Path, seed: str, node: int):
    d = ckpt_root / seed / "node_checkpoints"
    ckpt = torch.load(d / f"pre_u{node:06d}.pt", map_location="cpu",
                      weights_only=False)
    roles_path = d / f"pre_u{node:06d}_roles.pt"
    if not roles_path.is_file():
        raise FileNotFoundError(
            f"{roles_path} missing; Stage 2 needs role tensors at this node")
    payload = torch.load(roles_path, map_location="cpu", weights_only=False)
    tensors = payload["tensors"]

    def get(param: str, role: str) -> torch.Tensor:
        key = f"{param}|{role}"
        if key in tensors:
            return tensors[key].float()
        for alias in payload.get("aliases", []):
            if (alias["parameter_name"] == param
                    and alias["alias_role"] == role):
                return tensors[f"{param}|{alias['canonical_role']}"].float()
        raise KeyError(key)

    return ckpt, get


def roles_for_layer(get, layer: int) -> dict:
    out = {}
    for kind, pattern in PARAMS.items():
        param = pattern.format(layer=layer)
        out[kind] = {role: get(param, role)
                     for role in ("W_before", "W_after", "G", "M_prev",
                                  "M_t_candidate", "D", "U")}
    return out


def delta_from_u(roles: dict, u: dict, lr: float, wd: float) -> dict:
    return {
        kind: -lr * u[kind] - lr * wd * roles[kind]["W_before"]
        for kind in ("QKV", "O")
    }


def update_from_state(roles: dict, state: str, mu: float, fmt: str,
                      bits: int = 8, block_size: int = 64,
                      align: bool = True) -> dict:
    u = {}
    for kind in ("QKV", "O"):
        r = roles[kind]
        if state == "G":
            gq = quantize(r["G"], kind, fmt, bits, block_size, align)
            mt = r["M_prev"] * mu + gq
            dq = gq + mu * mt
            u[kind] = ns(dq)
        elif state == "M_prev":
            mq = quantize(r["M_prev"], kind, fmt, bits, block_size, align)
            mt = mq * mu + r["G"]
            dq = r["G"] + mu * mt
            u[kind] = ns(dq)
        elif state == "D":
            dq = quantize(r["D"], kind, fmt, bits, block_size, align)
            u[kind] = ns(dq)
        elif state == "U":
            u[kind] = quantize(r["U"], kind, fmt, bits, block_size, align)
        else:
            raise ValueError(state)
    return u


def eval_delta(model, x, y, layer: int, deltas: dict) -> float:
    originals = {}
    for kind, pattern in PARAMS.items():
        p = model.get_parameter(pattern.format(layer=layer))
        originals[p] = p.detach().clone()
        with torch.no_grad():
            p.add_(deltas[kind].to(p.device))
    with torch.no_grad():
        loss = float(model(x, y)["loss"])
    for p, orig in originals.items():
        with torch.no_grad():
            p.copy_(orig)
    return loss


def eval_w_quant(model, x, y, layer: int, roles: dict, fmt: str,
                 bits: int, block_size: int, align: bool,
                 delta_ref: dict) -> tuple[float, float]:
    originals = {}
    for kind, pattern in PARAMS.items():
        p = model.get_parameter(pattern.format(layer=layer))
        originals[p] = p.detach().clone()
        q = quantize(p.detach(), kind, fmt, bits, block_size, align)
        with torch.no_grad():
            p.copy_(q)
    with torch.no_grad():
        loss_w_only = float(model(x, y)["loss"])
        for kind, pattern in PARAMS.items():
            p = model.get_parameter(pattern.format(layer=layer))
            p.add_(delta_ref[kind].to(p.device))
        loss = float(model(x, y)["loss"])
    for p, orig in originals.items():
        with torch.no_grad():
            p.copy_(orig)
    return loss, loss_w_only


def write_row(writer, row):
    writer.writerow(row)


def base_metrics(l_base, l_ref, l_q, d_l_ref, d_l_q):
    gap = l_q - l_ref
    rel_gap = abs(gap) / abs(d_l_ref) if abs(d_l_ref) > 1e-30 else float("nan")
    return {
        "L_base": l_base, "L_ref": l_ref, "L_q": l_q,
        "dL_ref": d_l_ref, "dL_q": d_l_q,
        "loss_gap": gap, "rel_gap": rel_gap,
        "sign_ok": int(d_l_q < 0),
        "sign_match": int((d_l_q >= 0) == (d_l_ref >= 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="seed0,seed1")
    parser.add_argument("--nodes", default="784,1312")
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--probe-windows", type=int, default=8)
    parser.add_argument("--module",
                        choices=("L1", "L2", "L3", "L4", "L2abs", "L3abs", "all"),
                        default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--alpha", default="0.05,0.10")
    parser.add_argument("--alpha-abs", default="0.03,0.07")
    parser.add_argument("--noise-seeds", type=int, default=3)
    parser.add_argument("--l4-states", default="D,U")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    nodes = tuple(int(x) for x in args.nodes.split(","))
    layers = tuple(int(x) for x in args.layers.split(","))
    alphas = tuple(float(x) for x in args.alpha.split(","))
    alphas_abs = tuple(float(x) for x in args.alpha_abs.split(","))
    if args.smoke:
        seeds, nodes, layers = ("seed0",), (784,), (0,)
        alphas = (0.10,)
        alphas_abs = (0.07,)

    device = torch.device("cuda:0")
    set_math()
    fields_common = ["seed", "node", "layer", "module",
                     "L_base", "L_ref", "L_q", "L_w_only",
                     "dL_ref", "dL_q",
                     "loss_gap", "rel_gap", "sign_ok", "sign_match",
                     "rel_state", "rel_U", "cos_U", "share_L1_U", "top1_U",
                     "rel_W", "share_L1_W", "top1_W"]
    modules = (("L1", "L2", "L3", "L4", "L2abs", "L3abs")
               if args.module == "all" else (args.module,))
    writers = {}
    handles = {}
    for module in modules:
        name = f"{module.lower()}_loss.csv" if not args.smoke else f"smoke_{module.lower()}.csv"
        path = out / name
        h = path.open("w", newline="")
        handles[module] = h
        w = csv.DictWriter(h, fieldnames=fields_common + ["state", "format",
                                                          "space", "band",
                                                          "group", "alpha",
                                                          "noise", "granularity",
                                                          "bits", "block_size"])
        w.writeheader()
        writers[module] = w

    summary = {"modules": list(modules), "rows": {}, "status": "running"}
    try:
        for seed in seeds:
            cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{seed[-1]}.json")
            manifest = load_manifest(Path(args.data_root) / "manifest.json")
            val = FrozenWindows(manifest, "validation", 256)
            model = build_model(cfg, device)
            lr_peak = float(cfg["optimizer"]["lr"])
            wd = float(cfg["optimizer"]["weight_decay"])
            mu = float(cfg["optimizer"]["muon_momentum"])
            total = int(cfg["derived"]["schedule_total_updates"])
            warmup = int(cfg["schedule"]["warmup_updates"])
            ratio = float(cfg["schedule"]["final_lr_ratio"])
            schedule = cfg["schedule"]["name"]

            for node in nodes:
                ckpt, get = load_node(Path(args.ckpt_root), seed, node)
                model.load_state_dict(ckpt["model"], strict=True)
                model.eval()
                pairs = [val.window(i) for i in range(args.probe_windows)]
                x = torch.stack([p[0] for p in pairs]).to(device)
                y = torch.stack([p[1] for p in pairs]).to(device)
                with torch.no_grad():
                    l_base = float(model(x, y)["loss"])
                lr = learning_rate(node, total, lr_peak, warmup, ratio, schedule)
                rows_before = {m: summary["rows"].get(m, 0) for m in modules}

                for layer in layers:
                    roles = roles_for_layer(get, layer)
                    u_ref = {k: roles[k]["U"] for k in ("QKV", "O")}
                    delta_ref = {k: roles[k]["W_after"] - roles[k]["W_before"]
                                 for k in ("QKV", "O")}
                    l_ref = eval_delta(model, x, y, layer, delta_ref)
                    d_l_ref = l_ref - l_base

                    def emit(module, **kw):
                        row = {
                            "seed": seed, "node": node, "layer": layer,
                            "module": module,
                            "L_w_only": "",
                            "rel_state": "", "rel_U": "", "cos_U": "",
                            "share_L1_U": "", "top1_U": "",
                            "rel_W": "", "share_L1_W": "", "top1_W": "",
                            "state": "", "format": "", "space": "", "band": "",
                            "group": "", "alpha": "", "noise": "",
                            "granularity": "", "bits": "", "block_size": "",
                        }
                        row.update(kw)
                        write_row(writers[module], row)
                        summary["rows"][module] = summary["rows"].get(module, 0) + 1

                    # L1
                    if "L1" in modules:
                        for state in ("G", "M_prev", "D", "U", "W"):
                            for fmt in ("bf16", "int8_head"):
                                if args.smoke and not (state == "D" and fmt == "bf16"):
                                    continue
                                if state == "W":
                                    l_q, l_w = eval_w_quant(
                                        model, x, y, layer, roles, fmt, 8, 64,
                                        True, delta_ref)
                                    m_w = aggregate_metrics(
                                        {k: roles[k]["W_before"] for k in ("QKV", "O")},
                                        {k: quantize(roles[k]["W_before"], k, fmt)
                                         for k in ("QKV", "O")})
                                    base = base_metrics(l_base, l_ref, l_q,
                                                        d_l_ref, l_q - l_base)
                                    emit("L1", state=state, format=fmt,
                                         L_w_only=l_w,
                                         rel_state=m_w["rel"],
                                         rel_W=m_w["rel"],
                                         share_L1_W=m_w["share_L1"],
                                         top1_W=m_w["top1"], **base)
                                else:
                                    u_q = update_from_state(roles, state, mu, fmt)
                                    d_q = delta_from_u(roles, u_q, lr, wd)
                                    l_q = eval_delta(model, x, y, layer, d_q)
                                    q_state = {k: quantize(roles[k][state], k, fmt)
                                               for k in ("QKV", "O")}
                                    m_state = aggregate_metrics(
                                        {k: roles[k][state] for k in ("QKV", "O")},
                                        q_state)
                                    m_u = aggregate_metrics(
                                        {k: roles[k]["U"] for k in ("QKV", "O")},
                                        u_q)
                                    base = base_metrics(l_base, l_ref, l_q,
                                                        d_l_ref, l_q - l_base)
                                    emit("L1", state=state, format=fmt,
                                         rel_state=m_state["rel"],
                                         rel_U=m_u["rel"],
                                         cos_U=m_u["cos"],
                                         share_L1_U=m_u["share_L1"],
                                         top1_U=m_u["top1"], **base)

                    # L2
                    if "L2" in modules:
                        for space in ("D", "U"):
                            for band in BANDS:
                                for alpha in alphas:
                                    if args.smoke and not (space == "D" and band == "tail" and alpha == alphas[0]):
                                        continue
                                    noise_seeds = 1 if args.smoke else args.noise_seeds
                                    for noise in range(noise_seeds):
                                        gen = torch.Generator(device="cpu")
                                        gen.manual_seed(1000 * (noise + 1) + layer)
                                        xq = {}
                                        for kind in ("QKV", "O"):
                                            src = roles[kind][space]
                                            idx = split_bands(
                                                torch.linalg.svd(
                                                    roles[kind]["D"],
                                                    full_matrices=False)[1])[band]
                                            xq[kind] = band_perturb(
                                                src, roles[kind]["D"], idx,
                                                alpha, gen)
                                        u_q = ({k: ns(xq[k]) for k in ("QKV", "O")}
                                               if space == "D" else xq)
                                        d_q = delta_from_u(roles, u_q, lr, wd)
                                        l_q = eval_delta(model, x, y, layer, d_q)
                                        m = aggregate_metrics(
                                            {k: roles[k]["U"] for k in ("QKV", "O")},
                                            u_q)
                                        base = base_metrics(l_base, l_ref, l_q,
                                                            d_l_ref, l_q - l_base)
                                        emit("L2", space=space, band=band,
                                             alpha=alpha, noise=noise,
                                             rel_state=m["rel"], rel_U=m["rel"],
                                             cos_U=m["cos"],
                                             share_L1_U=m["share_L1"],
                                             top1_U=m["top1"], **base)

                    # L3
                    if "L3" in modules:
                        for space in ("D", "U"):
                            for group in ("QK", "VO", "Q", "K", "V", "O"):
                                for alpha in alphas:
                                    if args.smoke and not (space == "D" and group == "QK" and alpha == alphas[0]):
                                        continue
                                    noise_seeds = 1 if args.smoke else args.noise_seeds
                                    for noise in range(noise_seeds):
                                        gen = torch.Generator(device="cpu")
                                        gen.manual_seed(2000 * (noise + 1) + layer)
                                        xq = {}
                                        for kind in ("QKV", "O"):
                                            src = roles[kind][space]
                                            xq[kind] = perturb_group(
                                                src, kind, group, alpha, gen)
                                        u_q = ({k: ns(xq[k]) for k in ("QKV", "O")}
                                               if space == "D" else xq)
                                        d_q = delta_from_u(roles, u_q, lr, wd)
                                        l_q = eval_delta(model, x, y, layer, d_q)
                                        m = aggregate_metrics(
                                            {k: roles[k]["U"] for k in ("QKV", "O")},
                                            u_q)
                                        base = base_metrics(l_base, l_ref, l_q,
                                                            d_l_ref, l_q - l_base)
                                        emit("L3", space=space, group=group,
                                             alpha=alpha, noise=noise,
                                             rel_state=m["rel"], rel_U=m["rel"],
                                             cos_U=m["cos"],
                                             share_L1_U=m["share_L1"],
                                             top1_U=m["top1"], **base)

                    # L2abs
                    if "L2abs" in modules:
                        for space in ("D", "U"):
                            for band in BANDS:
                                for alpha_abs in alphas_abs:
                                    if args.smoke and not (
                                            space == "D" and band == "tail"
                                            and alpha_abs == alphas_abs[0]):
                                        continue
                                    noise_seeds = 1 if args.smoke else args.noise_seeds
                                    for noise in range(noise_seeds):
                                        gen = torch.Generator(device="cpu")
                                        gen.manual_seed(3000 * (noise + 1) + layer)
                                        xq = {}
                                        for kind in ("QKV", "O"):
                                            idx = split_bands(
                                                torch.linalg.svd(
                                                    roles[kind]["D"],
                                                    full_matrices=False)[1])[band]
                                            xq[kind] = band_perturb_abs(
                                                roles[kind][space],
                                                roles[kind]["D"], idx,
                                                alpha_abs,
                                                float(roles[kind][space].norm()),
                                                gen)
                                        u_q = ({k: ns(xq[k]) for k in ("QKV", "O")}
                                               if space == "D" else xq)
                                        d_q = delta_from_u(roles, u_q, lr, wd)
                                        l_q = eval_delta(model, x, y, layer, d_q)
                                        m = aggregate_metrics(
                                            {k: roles[k]["U"] for k in ("QKV", "O")},
                                            u_q)
                                        base = base_metrics(l_base, l_ref, l_q,
                                                            d_l_ref, l_q - l_base)
                                        emit("L2abs", space=space, band=band,
                                             alpha=alpha_abs, noise=noise,
                                             rel_state=m["rel"], rel_U=m["rel"],
                                             cos_U=m["cos"],
                                             share_L1_U=m["share_L1"],
                                             top1_U=m["top1"], **base)

                    # L3abs
                    if "L3abs" in modules:
                        for space in ("D", "U"):
                            for group in ("QK", "VO", "Q", "K", "V", "O"):
                                for alpha_abs in alphas_abs:
                                    if args.smoke and not (
                                            space == "D" and group == "QK"
                                            and alpha_abs == alphas_abs[0]):
                                        continue
                                    noise_seeds = 1 if args.smoke else args.noise_seeds
                                    for noise in range(noise_seeds):
                                        gen = torch.Generator(device="cpu")
                                        gen.manual_seed(4000 * (noise + 1) + layer)
                                        qkv_norm = float(roles["QKV"][space].norm())
                                        o_norm = float(roles["O"][space].norm())
                                        combined = math.sqrt(qkv_norm ** 2 + o_norm ** 2)
                                        if group == "VO":
                                            targets = {"QKV": combined / math.sqrt(2),
                                                       "O": combined / math.sqrt(2)}
                                        elif group == "O":
                                            targets = {"QKV": 0.0, "O": combined}
                                        else:
                                            targets = {"QKV": combined, "O": 0.0}
                                        xq = {}
                                        for kind in ("QKV", "O"):
                                            xq[kind] = perturb_group_abs(
                                                roles[kind][space], kind, group,
                                                alpha_abs, targets[kind], gen)
                                        u_q = ({k: ns(xq[k]) for k in ("QKV", "O")}
                                               if space == "D" else xq)
                                        d_q = delta_from_u(roles, u_q, lr, wd)
                                        l_q = eval_delta(model, x, y, layer, d_q)
                                        m = aggregate_metrics(
                                            {k: roles[k]["U"] for k in ("QKV", "O")},
                                            u_q)
                                        base = base_metrics(l_base, l_ref, l_q,
                                                            d_l_ref, l_q - l_base)
                                        emit("L3abs", space=space, group=group,
                                             alpha=alpha_abs, noise=noise,
                                             rel_state=m["rel"], rel_U=m["rel"],
                                             cos_U=m["cos"],
                                             share_L1_U=m["share_L1"],
                                             top1_U=m["top1"], **base)

                    # L4
                    if "L4" in modules:
                        states = tuple(s.strip() for s in args.l4_states.split(","))
                        for state in states:
                            for fmt, gran, bits, bs in (
                                ("int8_tensor", "tensor", 8, ""),
                                ("int8_head", "head", 8, ""),
                                ("int8_block", "block", 8, 64),
                                ("int8_block", "block", 8, 128),
                                ("int8_block", "block", 4, 128),
                            ):
                                if args.smoke and not (state == "D" and gran == "head" and bits == 8):
                                    continue
                                u_q = update_from_state(
                                    roles, state, mu, fmt, bits,
                                    int(bs) if bs != "" else 64, True)
                                d_q = delta_from_u(roles, u_q, lr, wd)
                                l_q = eval_delta(model, x, y, layer, d_q)
                                m = aggregate_metrics(
                                    {k: roles[k]["U"] for k in ("QKV", "O")},
                                    u_q)
                                base = base_metrics(l_base, l_ref, l_q,
                                                    d_l_ref, l_q - l_base)
                                emit("L4", state=state, format=fmt,
                                     granularity=gran, bits=bits,
                                     block_size=bs, rel_state=m["rel"],
                                     rel_U=m["rel"], cos_U=m["cos"],
                                     share_L1_U=m["share_L1"],
                                     top1_U=m["top1"], **base)

                print(f"[{seed} u{node}] done", flush=True)
            del model
            torch.cuda.empty_cache()

        summary["status"] = "completed"
    finally:
        for h in handles.values():
            h.close()

    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "COMPLETE").write_text("ok\n")
    print(f"COMPLETE {out}")


if __name__ == "__main__":
    main()
