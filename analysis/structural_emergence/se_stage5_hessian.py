"""Stage 5 Module B — Head-wise Hessian curvature and cross-head coupling.

Runs on the Stage 4 node checkpoints:

* reconstructs G, M_t, D, U and ΔW at each node from the pre-update model,
  Muon momentum and the exact update windows;
* computes L_probe direction curvature and full cross-block coupling;
* computes L_train direction curvature with 8-window chunks (a full 32-window
  HVP does not fit on a T4 with math attention);
* runs Lanczos on representative heads only (30 steps by default).

Usage (Kaggle GPU):
    python -u analysis/structural_emergence/se_stage5_hessian.py \
        --ckpt-root /kaggle/input/datasets/siennadai/fp32-muon-ckpt-1536/structural_concentration_ckpt \
        --data-root /kaggle/input/datasets/siennadai/lowp-benchmark-slimpajama-8x-v1/slimpajama_8x \
        --out /kaggle/working/stage5_hessian
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform/src"))

from config.recipe import load_recipe  # noqa: E402
from data.frozen_tokens import FrozenWindows, load_manifest  # noqa: E402
from optim.muon_reference import zeropower_newton_schulz  # noqa: E402
from train_platform import build_model, learning_rate  # noqa: E402

SEEDS = ("seed0", "seed1")
NODES = (32, 128, 512, 768, 784, 1024, 1312, 1536)
LAYERS = {"seed0": (0, 2, 3), "seed1": (0, 3, 4)}
VO_LAYERS = (0, 3)
HEAD_DIM = 64
PROBE_WINDOWS = 8
TRAIN_CHUNK = 8
TRAIN_WINDOWS = 32
LANCZOS_STEPS = 30
DIRECTION_FIELDS = ["seed", "node", "layer", "type", "head", "loss", "q",
                    "kappa", "dw_norm", "coef", "total_norm"]
LANCZOS_FIELDS = ["seed", "node", "layer", "type", "head", "rank",
                  "eigenvalue", "lanczos_steps", "last_beta"]


def head_specs(layer: int, include_vo: bool) -> list[dict]:
    specs = []
    for kind, base in (("Q", 0), ("K", 384), ("V", 768)):
        if kind == "V" and not include_vo:
            continue
        for head in range(6):
            start = base + head * HEAD_DIM
            specs.append({"layer": layer, "type": kind, "head": head,
                          "name": f"transformer.h.{layer}.attn.c_attn.weight",
                          "axis": "row", "start": start, "end": start + HEAD_DIM})
    if include_vo:
        for head in range(6):
            start = head * HEAD_DIM
            specs.append({"layer": layer, "type": "O", "head": head,
                          "name": f"transformer.h.{layer}.attn.c_proj.weight",
                          "axis": "col", "start": start, "end": start + HEAD_DIM})
    return specs


def set_attention(mode: str) -> None:
    """Replay G with the training backend, Hessian HVPs with math SDPA."""
    if mode == "auto":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    elif mode == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        raise ValueError(mode)


def block(tensor: torch.Tensor, spec: dict) -> torch.Tensor:
    if spec["axis"] == "row":
        return tensor[spec["start"]:spec["end"], :]
    return tensor[:, spec["start"]:spec["end"]]


def set_block(tensor: torch.Tensor, spec: dict, value: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(tensor)
    if spec["axis"] == "row":
        out[spec["start"]:spec["end"], :] = value
    else:
        out[:, spec["start"]:spec["end"]] = value
    return out


class HessianEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.param_by_id = {id(p): p for p in self.params}

    def vector(self, spec: dict, value: torch.Tensor) -> dict:
        out = {id(p): torch.zeros_like(p) for p in self.params}
        parameter = self.model.get_parameter(spec["name"])
        out[id(parameter)] = set_block(parameter, spec, value)
        return out

    def hvp(self, loss_fn, v_by_id: dict) -> dict:
        # Never return the loss: holding it keeps the previous double-backward
        # graph alive and doubles the peak (the pilot's 14 GiB artifact).
        self.model.zero_grad(set_to_none=True)
        loss = loss_fn()
        grads = torch.autograd.grad(loss, self.params, create_graph=True,
                                    allow_unused=True)
        used = [(p, g) for p, g in zip(self.params, grads) if g is not None]
        hv = torch.autograd.grad([g for _, g in used], [p for p, _ in used],
                                 grad_outputs=[v_by_id[id(p)] for p, _ in used],
                                 allow_unused=True)
        out = {id(p): h for (p, _), h in zip(used, hv) if h is not None}
        del loss, grads, used, hv
        return out

    def hvp_block(self, loss_fn, spec: dict, value: torch.Tensor) -> torch.Tensor:
        hv = self.hvp(loss_fn, self.vector(spec, value))
        parameter = self.model.get_parameter(spec["name"])
        result = block(hv[id(parameter)], spec).detach()
        del hv
        return result


def load_node(model, engine, path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.train()
    return checkpoint


def reconstruct_update(engine, checkpoint, cfg, micro_groups, batch, node: int):
    model = engine.model
    # Match training exactly: 8 micro-batches of 4 windows, each loss divided
    # by accumulation_steps, then one global clip over all parameters.
    model.zero_grad(set_to_none=True)
    for group in micro_groups:
        x, y = batch(group, "train")
        loss = model(x, y)["loss"]
        (loss / len(micro_groups)).backward()
        del x, y, loss
    grads = [p.grad for p in engine.params]
    total_norm = torch.sqrt(sum((g.detach().float() ** 2).sum()
                                for g in grads if g is not None))
    coef = min(1.0, float(cfg["train"]["grad_clip_norm"]) / float(total_norm))
    grad_by_id = {id(p): g * coef for p, g in zip(engine.params, grads)
                  if g is not None}
    g_opt_by_name = {name: grad_by_id[id(model.get_parameter(name))]
                     for name in engine.muon_names}
    model.zero_grad(set_to_none=True)
    del grads
    mu = float(cfg["optimizer"].get("muon_momentum", 0.95))
    nesterov = bool(cfg["optimizer"].get("muon_nesterov", True))
    lr = learning_rate(node, cfg["derived"]["schedule_total_updates"],
                       cfg["optimizer"]["lr"], cfg["schedule"]["warmup_updates"],
                       cfg["schedule"]["final_lr_ratio"], cfg["schedule"]["name"])
    wd = float(cfg["optimizer"]["weight_decay"])
    dw = {}
    momentum = checkpoint.get("muon_momentum", {})
    with torch.no_grad():
        for name in engine.muon_names:
            parameter = model.get_parameter(name)
            g = grad_by_id[id(parameter)]
            prev = momentum.get(name)
            prev = prev.to(engine.device) if prev is not None else torch.zeros_like(parameter)
            m_t = mu * prev + g
            direction = g + mu * m_t if nesterov else m_t
            u = zeropower_newton_schulz(
                direction, cfg["optimizer"].get("muon_ns_steps", 5),
                tuple(cfg["optimizer"].get("muon_ns_coefficients",
                                           [3.4445, -4.7750, 2.0315])),
                cfg["optimizer"].get("muon_eps", 1e-7))
            dw[name] = -lr * u - lr * wd * parameter
    return dw, g_opt_by_name, coef, float(total_norm)


def run_nodes(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    direction_rows, cross_rows, replay_rows, lanczos_rows = [], [], [], []
    summary = {"nodes": list(args.nodes), "seeds": list(args.seeds),
               "layers": {s: list(v) for s, v in args.layers.items()},
               "vo_layers": list(VO_LAYERS),
               "lanczos_steps": args.lanczos_steps,
               "probe_windows": PROBE_WINDOWS, "train_chunk": TRAIN_CHUNK,
               "runs": []}

    def flush() -> None:
        def write(name, fields, rows):
            if not rows:
                return
            with (out / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
        write("direction_curvature.csv", DIRECTION_FIELDS, direction_rows)
        write("cross_terms.csv", ["seed", "node", "layer", "loss", "h_type",
                                  "h_head", "k_type", "k_head", "q_hk", "rho"], cross_rows)
        write("replay_validation.csv", ["seed", "node", "parameter",
                                        "delta_w_rel_error", "g_rel_error"], replay_rows)
        write("lanczos.csv", LANCZOS_FIELDS, lanczos_rows)
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))

    device = torch.device("cuda:0")
    for seed in args.seeds:
        cfg = load_recipe(ROOT / f"platform/recipes/structural_emergence_30m_fp32_muon_{seed[-1]}.json")
        manifest = load_manifest(Path(args.data_root) / "manifest.json")
        train = FrozenWindows(manifest, "train", 256)
        val = FrozenWindows(manifest, "validation", 256)
        model = build_model(cfg, device)
        engine = HessianEngine(model, device)
        engine.muon_names = tuple(name for name, p in model.named_parameters()
                                  if p.ndim == 2 and name.startswith("transformer.h."))
        assert len(engine.muon_names) == 30, len(engine.muon_names)
        summary.setdefault("data_fingerprint", manifest["fingerprint"])
        summary.setdefault("recipe_fingerprints", {})[seed] = cfg["fingerprint"]
        for node in args.nodes:
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            set_attention("auto")  # replay the exact training backend first
            node_dir = Path(args.ckpt_root) / seed / "node_checkpoints"
            checkpoint = load_node(model, engine, node_dir / f"pre_u{node:06d}.pt", device)
            sidecar = json.loads((node_dir / f"pre_u{node:06d}.json").read_text())
            ids_train = sidecar["window_ids"]
            def batch(window_ids, split):
                source = train if split == "train" else val
                pairs = [source.window(i) for i in window_ids]
                return (torch.stack([p[0] for p in pairs]).to(device),
                        torch.stack([p[1] for p in pairs]).to(device))
            x_probe, y_probe = batch(list(range(PROBE_WINDOWS)), "val")
            probe_loss = lambda: model(x_probe, y_probe)["loss"]
            dw, g_opt, coef, total_norm = reconstruct_update(
                engine, checkpoint, cfg, sidecar["micro_batch_window_ids"],
                batch, node)
            role_path = node_dir / f"pre_u{node:06d}_roles.pt"
            if role_path.is_file():
                roles = torch.load(role_path, map_location="cpu", weights_only=False)
                for name in sorted({s["name"] for s in head_specs(0, True)}):
                    if name + "|W_after" not in roles["tensors"]:
                        continue
                    stored = (roles["tensors"][name + "|W_after"].to(device)
                              - roles["tensors"][name + "|W_before"].to(device))
                    stored_g = roles["tensors"][name + "|G"].to(device)
                    replay_rows.append({
                        "seed": seed, "node": node, "parameter": name,
                        "delta_w_rel_error": float((dw[name] - stored).norm() / stored.norm()),
                        "g_rel_error": float((g_opt[name] - stored_g).norm() / stored_g.norm())})
                del roles
            set_attention("math")  # double backward requires math SDPA
            for layer in args.layers[seed]:
                torch.manual_seed(2026 + node * 100 + layer)
                include_vo = layer in VO_LAYERS
                specs = head_specs(layer, include_vo)
                # L_probe cross terms: one HVP per block gives the full row/column.
                probe_hv = {}
                for spec in specs:
                    probe_hv[id(spec)] = engine.hvp_block(
                        probe_loss, spec, block(dw[spec["name"]], spec))
                q_diag = {id(s): float((block(dw[s["name"]], s)
                                        * probe_hv[id(s)]).sum()) for s in specs}
                for h_spec in specs:
                    q_h = q_diag[id(h_spec)]
                    dw_norm = float(block(dw[h_spec["name"]], h_spec).norm())
                    direction_rows.append({
                        "seed": seed, "node": node, "layer": layer,
                        "type": h_spec["type"], "head": h_spec["head"],
                        "loss": "probe", "q": q_h,
                        "kappa": q_h / (dw_norm ** 2) if dw_norm > 0 else float("nan"),
                        "dw_norm": dw_norm, "coef": coef, "total_norm": total_norm})
                    for k_spec in specs:
                        q_k = q_diag[id(k_spec)]
                        q_hk = float((block(dw[h_spec["name"]], h_spec)
                                      * probe_hv[id(k_spec)]).sum())
                        denom = math.sqrt(abs(q_h * q_k)) if q_h != 0 and q_k != 0 else 0.0
                        cross_rows.append({
                            "seed": seed, "node": node, "layer": layer,
                            "loss": "probe",
                            "h_type": h_spec["type"], "h_head": h_spec["head"],
                            "k_type": k_spec["type"], "k_head": k_spec["head"],
                            "q_hk": q_hk,
                            "rho": q_hk / denom if denom > 0 else float("nan")})
                # L_train direction curvature, 8-window chunks (mean loss Hessian
                # is the equal-chunk average; a 32-window HVP does not fit).
                chunks = [ids_train[i * TRAIN_CHUNK:(i + 1) * TRAIN_CHUNK]
                          for i in range(TRAIN_WINDOWS // TRAIN_CHUNK)]
                for spec in specs:
                    value = block(dw[spec["name"]], spec)
                    acc = torch.zeros_like(value)
                    for chunk in chunks:
                        xc, yc = batch(chunk, "train")
                        fn = lambda xc=xc, yc=yc: model(xc, yc)["loss"]
                        acc += engine.hvp_block(fn, spec, value)
                        del xc, yc
                        torch.cuda.empty_cache()
                    acc /= len(chunks)
                    q_h = float((value * acc).sum())
                    dw_norm = float(value.norm())
                    direction_rows.append({
                        "seed": seed, "node": node, "layer": layer,
                        "type": spec["type"], "head": spec["head"],
                        "loss": "train", "q": q_h,
                        "kappa": q_h / (dw_norm ** 2) if dw_norm > 0 else float("nan"),
                        "dw_norm": dw_norm, "coef": coef, "total_norm": total_norm})
                # Lanczos on one representative head per type (largest |ΔW|).
                chosen = []
                for kind in ("Q", "K", "V", "O"):
                    candidates = [s for s in specs if s["type"] == kind]
                    if candidates:
                        chosen.append(max(candidates,
                                          key=lambda s: float(block(dw[s["name"]], s).norm())))
                for spec in chosen:
                    alphas, betas = lanczos(engine, probe_loss, spec,
                                            args.lanczos_steps)
                    T = torch.zeros(len(alphas), len(alphas))
                    for i, alpha in enumerate(alphas):
                        T[i, i] = alpha
                    for i, beta in enumerate(betas):
                        T[i, i + 1] = T[i + 1, i] = beta
                    top = sorted(torch.linalg.eigvalsh(T).tolist(), reverse=True)[:4]
                    for rank, eigenvalue in enumerate(top, start=1):
                        lanczos_rows.append({
                            "seed": seed, "node": node, "layer": layer,
                            "type": spec["type"], "head": spec["head"],
                            "rank": rank, "eigenvalue": float(eigenvalue),
                            "lanczos_steps": args.lanczos_steps,
                            "last_beta": betas[-1] if betas else float("nan")})
                flush()
                print(f"[stage5] {seed} node {node} layer {layer}: "
                      f"probe/train/lanczos done", flush=True)
            summary["runs"].append({"seed": seed, "node": node,
                                    "seconds": time.time() - started,
                                    "peak_MiB": torch.cuda.max_memory_allocated()/1024**2})
            flush()
            del checkpoint, sidecar, x_probe, y_probe, dw
            torch.cuda.empty_cache()
        del model, engine
        torch.cuda.empty_cache()
    flush()
    print(f"[stage5] wrote {out}")


def lanczos(engine, loss_fn, spec, steps):
    q0 = torch.randn_like(engine.model.get_parameter(spec["name"]))
    q0 = block(q0, spec)
    q0 = q0 / max(float(q0.norm()), 1e-30)
    Q = [q0]
    alphas, betas = [], []
    for j in range(steps):
        w = engine.hvp_block(loss_fn, spec, Q[-1])
        alpha = float((Q[-1] * w).sum())
        w = w - alpha * Q[-1]
        if j > 0:
            w = w - betas[-1] * Q[-2]
        for qi in Q:
            w = w - (qi * w).sum() * qi
        beta = float(w.norm())
        alphas.append(alpha)
        if j < steps - 1 and beta > 1e-14:
            Q.append(w / beta)
            betas.append(beta)
        else:
            break
    return alphas, betas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nodes", default=",".join(map(str, NODES)))
    parser.add_argument("--seeds", default=",".join(SEEDS))
    parser.add_argument("--layers", default="",
                        help="override layers for both seeds, e.g. 2")
    parser.add_argument("--lanczos-steps", type=int, default=LANCZOS_STEPS)
    args = parser.parse_args()
    args.nodes = tuple(int(x) for x in args.nodes.split(","))
    args.seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    if any(s not in SEEDS for s in args.seeds):
        parser.error("--seeds must be a subset of seed0,seed1")
    if args.layers:
        shared = tuple(int(x) for x in args.layers.split(","))
        args.layers = {seed: shared for seed in args.seeds}
    else:
        args.layers = {seed: LAYERS[seed] for seed in args.seeds}
    run_nodes(args)


if __name__ == "__main__":
    main()
