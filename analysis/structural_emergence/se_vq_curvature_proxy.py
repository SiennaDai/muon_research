"""Curvature-weighted single-step proxy for compressed Muon states.

At one FP32 pre-update node, for each candidate state representation
(base uniform rank-8 / C / C3) it computes the loss change of the corresponding
optimizer step predicted by the local quadratic model of the probe loss:

    dL ~= g . dW + 0.5 dW^T H dW ,   dW = -lr * (U_hat - U_ref)

where H is the Hessian of L_probe (validation windows 0..7), and compares that
ranking with the 4096-step NLL ranking already measured
(base 4.7584 < C 4.7756 < C3 4.8061, seed 0).

The point of the experiment is to test whether a curvature-weighted one-step
number predicts long-horizon quality better than plain reconstruction error.

Usage (Kaggle GPU):
    python -u analysis/structural_emergence/se_vq_curvature_proxy.py \
        --ckpt-root /kaggle/input/datasets/siennadai/fp32-muon-ckpt-1536/structural_concentration_ckpt \
        --data-root /kaggle/input/datasets/siennadai/lowp-benchmark-slimpajama-8x-v1/slimpajama_8x \
        --codebook /kaggle/working/benchmark-experiments/reports/muon_vector_int3_robustness/calibration_codebooks.pt \
        --seed seed0 --nodes 1536 --layers 0 1 2 3 4 5 \
        --out /kaggle/working/vq_curvature_proxy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform" / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "structural_emergence"))

from config.recipe import load_recipe  # noqa: E402
from data.frozen_tokens import FrozenWindows, load_manifest  # noqa: E402
from optim.muon_reference import zeropower_newton_schulz  # noqa: E402
from train_platform import build_model, learning_rate  # noqa: E402
from se_stage5_hessian import (HessianEngine, load_node,  # noqa: E402
                               reconstruct_update, set_attention, PROBE_WINDOWS)

BLOCK = 2048
CANDIDATES = ("base", "C", "C3")
LONG_HORIZON = {"base": 4.7584, "C": 4.7756, "C3": 4.8061}


def ns(x, cfg):
    return zeropower_newton_schulz(
        x, cfg["optimizer"].get("muon_ns_steps", 5),
        tuple(cfg["optimizer"].get("muon_ns_coefficients", [3.4445, -4.7750, 2.0315])),
        cfg["optimizer"].get("muon_eps", 1e-7))


def vq_roundtrip(mat: torch.Tensor, rank: int, book: torch.Tensor) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros_like(mat)
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    u, s, vh = u[:, :rank], s[:rank], vh[:rank]
    low = (u * s) @ vh
    res = mat - low
    flat = res.reshape(-1)
    blocks = flat.reshape(-1, BLOCK).abs()
    scales = torch.quantile(blocks, .98, dim=1)
    nonzero = blocks.amax(dim=1) != 0
    scales = torch.where(nonzero, scales.clamp_min(torch.finfo(torch.float32).tiny),
                         torch.zeros_like(scales))
    pairs = flat.reshape(-1, 2)
    pair_scales = torch.repeat_interleave(scales, BLOCK // 2)[:pairs.shape[0]]
    safe = pair_scales.clamp_min(torch.finfo(torch.float32).tiny)
    normalized = (pairs / safe[:, None]).clamp(-1, 1)
    idx = torch.empty(pairs.shape[0], dtype=torch.long, device=mat.device)
    for start in range(0, pairs.shape[0], 65536):
        end = min(start + 65536, pairs.shape[0])
        idx[start:end] = torch.cdist(normalized[start:end], book).argmin(dim=1)
    idx = torch.where(pair_scales > 0, idx, torch.zeros_like(idx))
    rec = (book[idx] * pair_scales[:, None]).reshape(res.shape)
    return low + rec


def candidate_state(name: str, mat: torch.Tensor, cand: str,
                    books: dict[str, torch.Tensor]) -> torch.Tensor:
    if cand == "base":
        return vq_roundtrip(mat, 8, books["w64"])
    book = books["w64"] if cand == "C" else books["w32"]
    if name.endswith("attn.c_attn.weight"):
        return torch.cat((vq_roundtrip(mat[:384], 2, book),
                          vq_roundtrip(mat[384:768], 2, book),
                          vq_roundtrip(mat[768:1152], 8, book)), dim=0)
    if name.endswith("attn.c_proj.weight"):
        return vq_roundtrip(mat, 8, book)
    return vq_roundtrip(mat, 2, book)


def grad_and_hvp(model, params, probe_loss, deltas):
    """Return per delta: sum g.d and 0.5 d^T H d for the probe loss."""
    model.zero_grad(set_to_none=True)
    loss = probe_loss()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    del loss
    used = [(p, g) for p, g in zip(params, grads) if g is not None]
    first = [sum(float((g * d[id(p)]).sum()) for p, g in used) for d in deltas]
    second = []
    for i, d in enumerate(deltas):
        hv = torch.autograd.grad([g for _, g in used], [p for p, _ in used],
                                 grad_outputs=[d[id(p)] for p, _ in used],
                                 retain_graph=i < len(deltas) - 1,
                                 allow_unused=True)
        second.append(0.5 * sum(float((h * d[id(p)]).sum())
                                for (p, _), h in zip(used, hv) if h is not None))
        del hv
    del grads, used
    return first, second


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--seed", default="seed0")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1536])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_grad_enabled(True)

    blob = torch.load(args.codebook, map_location="cpu", weights_only=False)
    books = {"w64": blob["codebooks"]["s1_k8_w64_t8_v1200"].float().to(device),
             "w32": blob["codebooks"]["s1_k8_w32_t8_v1200"].float().to(device)}

    cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_s{args.seed[-1]}.json")
    manifest = load_manifest(Path(args.data_root) / "manifest.json")
    train = FrozenWindows(manifest, "train", 256)
    val = FrozenWindows(manifest, "validation", 256)
    model = build_model(cfg, device)
    engine = HessianEngine(model, device)
    engine.muon_names = tuple(name for name, p in model.named_parameters()
                              if p.ndim == 2 and name.startswith("transformer.h."))
    assert len(engine.muon_names) == 30, len(engine.muon_names)

    pairs = [val.window(i) for i in range(PROBE_WINDOWS)]
    x_probe = torch.stack([p[0] for p in pairs]).to(device)
    y_probe = torch.stack([p[1] for p in pairs]).to(device)
    probe_loss = lambda: model(x_probe, y_probe)["loss"]

    mu = float(cfg["optimizer"].get("muon_momentum", 0.95))
    zeros = {id(p): torch.zeros_like(p) for p in engine.params}
    rows = []
    for node in args.nodes:
        node_dir = Path(args.ckpt_root) / args.seed / "node_checkpoints"
        checkpoint = load_node(model, engine, node_dir / f"pre_u{node:06d}.pt", device)
        sidecar = json.loads((node_dir / f"pre_u{node:06d}.json").read_text())

        def batch(ids, split):
            source = train if split == "train" else val
            items = [source.window(i) for i in ids]
            return (torch.stack([p[0] for p in items]).to(device),
                    torch.stack([p[1] for p in items]).to(device))

        _, g_opt, _, _ = reconstruct_update(engine, checkpoint, cfg,
                                            sidecar["micro_batch_window_ids"],
                                            batch, node)
        lr = learning_rate(node, cfg["derived"]["schedule_total_updates"],
                           cfg["optimizer"]["lr"], cfg["schedule"]["warmup_updates"],
                           cfg["schedule"]["final_lr_ratio"], cfg["schedule"]["name"])
        momentum = checkpoint.get("muon_momentum", {})
        started = time.time()
        for layer in args.layers:
            names = [name for name in engine.muon_names
                     if name.startswith(f"transformer.h.{layer}.")]
            model.zero_grad(set_to_none=True)
            m_t, u_ref = {}, {}
            with torch.no_grad():
                for name in names:
                    prev = momentum.get(name)
                    prev = prev.to(device) if prev is not None else torch.zeros_like(
                        model.get_parameter(name))
                    m_t[name] = mu * prev + g_opt[name]
                    u_ref[name] = ns(g_opt[name] + mu * m_t[name], cfg)

            deltas, rel_terms = [], []
            for cand in CANDIDATES:
                delta = dict(zeros)
                num = den = 0.0
                for name in names:
                    m_hat = candidate_state(name, m_t[name], cand, books)
                    u_hat = ns(g_opt[name] + mu * m_hat, cfg)
                    diff = (u_hat - u_ref[name]).float()
                    delta[id(model.get_parameter(name))] = (-lr * diff).to(device)
                    num += float((diff ** 2).sum())
                    den += float((u_ref[name].float() ** 2).sum())
                rel_terms.append(num ** 0.5 / den ** 0.5 if den else 0.0)
                deltas.append(delta)

            set_attention("math")
            first, second = grad_and_hvp(model, engine.params, probe_loss, deltas)
            for cand, rel, f, s in zip(CANDIDATES, rel_terms, first, second):
                rows.append({"seed": args.seed, "node": node, "layer": layer,
                             "candidate": cand, "rel_u": rel,
                             "first_order": f, "second_order": s,
                             "predicted_gap": f + s})
            print(f"[proxy] node {node} layer {layer} done in {time.time()-started:.1f}s",
                  flush=True)

    path = out / "curvature_proxy.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for cand in CANDIDATES:
        sel = [r for r in rows if r["candidate"] == cand]
        summary[cand] = {
            "predicted_gap_mean": sum(r["predicted_gap"] for r in sel) / len(sel),
            "first_order_mean": sum(r["first_order"] for r in sel) / len(sel),
            "second_order_mean": sum(r["second_order"] for r in sel) / len(sel),
            "rel_u_mean": sum(r["rel_u"] for r in sel) / len(sel),
            "nll_4096_s0": LONG_HORIZON[cand],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    base_pred = summary["base"]["predicted_gap_mean"]
    print("\ncandidate  pred_gap   rel_u    NLL4096   NLL-vs-base")
    for cand in CANDIDATES:
        s = summary[cand]
        print(f"{cand:9s} {s['predicted_gap_mean']:+.6f}  {s['rel_u_mean']:.4f}  "
              f"{s['nll_4096_s0']:.4f}   {s['nll_4096_s0'] - LONG_HORIZON['base']:+.4f}")
    print(f"(predicted gaps are relative to {base_pred:+.6f})")


if __name__ == "__main__":
    main()
