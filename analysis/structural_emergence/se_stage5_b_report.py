#!/usr/bin/env python3
"""Stage 5 Module B — validation, figures, and Chinese report.

Input is the downloaded Kaggle output:
    stage5_hessian/{direction_curvature,cross_terms,lanczos,replay_validation}.csv

Run with:
    uv run --with pandas --with matplotlib --with scipy python3 \
        analysis/structural_emergence/se_stage5_b_report.py \
        --input results/stage5_hessian_final/stage5_hessian \
        --out results/stage5_hessian_analysis
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

TYPES = ("Q", "K", "V", "O")
PAIR_ORDER = ("Q-Q", "K-K", "Q-K", "V-V", "O-O", "O-V", "Q-V", "K-O")
COLORS = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c", "O": "#d62728"}
PAIR_COLORS = {
    "Q-Q": "#1f77b4", "K-K": "#ff7f0e", "Q-K": "#8c564b",
    "V-V": "#2ca02c", "O-O": "#d62728", "O-V": "#9467bd",
    "Q-V": "#17becf", "K-O": "#7f7f7f",
}
EVENTS = {"seed0": 784, "seed1": 1312}
STAGE1 = Path("/mnt/d/muon_research/results/head_spectral_stage1/head_spectral_metrics.csv")


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 165, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "figure.max_open_warning": 0,
    })


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 8.5) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, title, fontsize=14, va="top", weight="bold")
    y = 0.925
    for line in lines:
        fig.text(0.06, y, line, fontsize=fontsize, va="top", family="monospace")
        y -= 0.021
        if y < 0.04:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.96
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_page(pdf: PdfPages, path: Path, title: str, caption: list[str]) -> None:
    image = plt.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.18, 0.94, 0.72])
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.03, 0.96, title, fontsize=14, weight="bold", va="top")
    y = 0.14
    for line in caption:
        fig.text(0.03, y, line, fontsize=8.2, va="top", family="monospace")
        y -= 0.022
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load(input_dir: Path):
    d = pd.read_csv(input_dir / "direction_curvature.csv")
    c = pd.read_csv(input_dir / "cross_terms.csv")
    lz = pd.read_csv(input_dir / "lanczos.csv")
    rv = pd.read_csv(input_dir / "replay_validation.csv")
    summary = json.loads((input_dir / "run_summary.json").read_text())
    return d, c, lz, rv, summary


def cross_symmetry(c) -> float:
    worst = 0.0
    for _, block in c.groupby(["seed", "node", "layer"]):
        keys = [(t, h) for t in sorted(block["h_type"].unique()) for h in range(6)]
        index = {k: i for i, k in enumerate(keys)}
        matrix = np.full((len(keys), len(keys)), np.nan)
        for _, row in block.iterrows():
            matrix[index[(row["h_type"], int(row["h_head"]))],
                   index[(row["k_type"], int(row["k_head"]))]] = row["q_hk"]
        scale = np.nanmax(np.abs(matrix))
        if np.isfinite(scale) and scale > 0:
            worst = max(worst, float(np.nanmax(np.abs(matrix - matrix.T)) / scale))
    return worst


def validate(d, c, lz, rv, summary) -> dict:
    assert len(d) == 1920, len(d)
    assert len(c) == 20736, len(c)
    assert len(lz) == 640, len(lz)
    assert len(rv) == 8, len(rv)
    max_g = float(rv["g_rel_error"].max())
    max_dw = float(rv["delta_w_rel_error"].max())
    assert max_g < 1e-4 and max_dw < 1e-4
    assert np.isfinite(d["kappa"]).all()
    assert np.isfinite(lz["eigenvalue"]).all()
    peaks = [r["peak_MiB"] for r in summary["runs"]]
    return {
        "direction_rows": len(d), "cross_rows": len(c),
        "lanczos_rows": len(lz), "replay_rows": len(rv),
        "max_g_rel_error": max_g, "max_dw_rel_error": max_dw,
        "max_peak_MiB": max(peaks), "total_node_runs": len(peaks),
        "cross_matrix_max_rel_asymmetry": cross_symmetry(c),
        "nodes": sorted(d["node"].unique().tolist()),
        "seeds": sorted(d["seed"].unique().tolist()),
    }


def direction_tables(d):
    g = d.groupby(["seed", "node", "layer", "type", "loss"], as_index=False)
    rows = []
    for key, block in g:
        seed, node, layer, typ, loss = key
        block = block.sort_values("head")
        top = block.loc[block["kappa"].idxmax()]
        rows.append({
            "seed": seed, "node": node, "layer": layer, "type": typ, "loss": loss,
            "n_heads": len(block),
            "kappa_mean": block["kappa"].mean(),
            "kappa_median": block["kappa"].median(),
            "kappa_min": block["kappa"].min(),
            "kappa_max": block["kappa"].max(),
            "negative_frac": float((block["kappa"] < 0).mean()),
            "dw_norm_mean": block["dw_norm"].mean(),
            "top_head": int(top["head"]), "top_kappa": float(top["kappa"]),
        })
    kappa_summary = pd.DataFrame(rows)
    ranks = d.sort_values(["seed", "node", "layer", "type", "loss", "kappa"],
                          ascending=[True, True, True, True, True, False]).copy()
    ranks["rank"] = ranks.groupby(["seed", "node", "layer", "type", "loss"]).cumcount() + 1
    return kappa_summary, ranks


def cross_tables(c):
    c = c.copy()
    c["pair_group"] = ["-".join(sorted([a, b])) for a, b in zip(c["h_type"], c["k_type"])]
    c["off_diag"] = ~((c["h_type"] == c["k_type"]) & (c["h_head"] == c["k_head"]))
    rows = []
    for key, block in c[c["off_diag"]].groupby(["seed", "node", "layer", "pair_group"]):
        seed, node, layer, group = key
        rho = block["rho"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "seed": seed, "node": node, "layer": layer, "pair_group": group,
            "n": len(rho), "rho_median": rho.median(), "rho_mean": rho.mean(),
            "abs_median": rho.abs().median(),
            "positive_frac": float((rho > 0).mean()),
            "abs_gt1_frac": float((rho.abs() > 1).mean()),
        })
    return c, pd.DataFrame(rows)


def lanczos_tables(lz):
    rows = []
    for key, block in lz.groupby(["seed", "node", "layer", "type", "head"]):
        seed, node, layer, typ, head = key
        block = block.sort_values("rank")
        top = block[block["rank"] <= 4]["eigenvalue"].tolist()
        top += [np.nan] * (4 - len(top))
        rows.append({
            "seed": seed, "node": node, "layer": layer, "type": typ, "head": head,
            "top1": top[0], "top2": top[1], "top3": top[2], "top4": top[3],
            "last_beta": float(block["last_beta"].iloc[0]),
            "last_beta_over_top1": float(block["last_beta"].iloc[0] / max(abs(top[0]), 1e-30)),
        })
    return pd.DataFrame(rows)


def stage1_alignment(d, kappa_summary):
    if not STAGE1.is_file():
        return pd.DataFrame()
    s = pd.read_csv(STAGE1)
    rows = []
    probe = d[d["loss"] == "probe"]
    for role in ("G", "U"):
        for key, block in probe.groupby(["seed", "node", "layer", "type"]):
            seed, node, layer, typ = key
            if typ not in TYPES:
                continue
            other = s[(s["seed"] == seed) & (s["update"] == node) &
                      (s["layer"] == layer) & (s["matrix_type"] == typ) &
                      (s["role"] == role)]
            if other.empty:
                continue
            a = block.set_index("head")["kappa"]
            b = other.set_index("head")["head_energy_share"]
            common = sorted(set(a.index) & set(b.index))
            if len(common) < 4:
                continue
            rho = spearmanr(a[common], b[common]).statistic
            rows.append({"seed": seed, "node": node, "layer": layer, "type": typ,
                         "role": role, "spearman": rho, "n_heads": len(common)})
    return pd.DataFrame(rows)


def cross_seed_tables(d, c):
    rows = []
    probe = d[d["loss"] == "probe"]
    for layer in (0, 3):
        for node in sorted(probe["node"].unique()):
            for typ in TYPES:
                a = probe[(probe["seed"] == "seed0") & (probe["node"] == node) &
                          (probe["layer"] == layer) & (probe["type"] == typ)].set_index("head")["kappa"]
                b = probe[(probe["seed"] == "seed1") & (probe["node"] == node) &
                          (probe["layer"] == layer) & (probe["type"] == typ)].set_index("head")["kappa"]
                common = sorted(set(a.index) & set(b.index))
                if len(common) < 4:
                    continue
                rows.append({
                    "layer": layer, "node": node, "type": typ,
                    "kappa_spearman": spearmanr(a[common], b[common]).statistic,
                    "kappa_pearson": float(np.corrcoef(a[common], b[common])[0, 1]),
                    "top_head_seed0": int(a[common].idxmax()),
                    "top_head_seed1": int(b[common].idxmax()),
                })
    matrix_rows = []
    for layer in (0, 3):
        for node in sorted(c["node"].unique()):
            a = c[(c["seed"] == "seed0") & (c["node"] == node) & (c["layer"] == layer)]
            b = c[(c["seed"] == "seed1") & (c["node"] == node) & (c["layer"] == layer)]
            if a.empty or b.empty:
                continue
            keys = ["h_type", "h_head", "k_type", "k_head"]
            m = a.merge(b, on=keys, suffixes=("_0", "_1"))
            off = ~((m["h_type"] == m["k_type"]) & (m["h_head"] == m["k_head"]))
            m = m[off & (m["rho_0"].abs() < 10) & (m["rho_1"].abs() < 10)]
            if len(m) < 10:
                continue
            matrix_rows.append({
                "layer": layer, "node": node, "n_pairs": len(m),
                "rho_spearman": spearmanr(m["rho_0"], m["rho_1"]).statistic,
                "rho_pearson": float(np.corrcoef(m["rho_0"], m["rho_1"])[0, 1]),
            })
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows)


def figure_kappa_overview(kappa_summary, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, typ in zip(axes.flat, TYPES):
        for seed in ("seed0", "seed1"):
            for layer in sorted(kappa_summary[kappa_summary["seed"] == seed]["layer"].unique()):
                for loss, ls in (("probe", "-"), ("train", "--")):
                    block = kappa_summary[(kappa_summary["seed"] == seed) &
                                          (kappa_summary["layer"] == layer) &
                                          (kappa_summary["type"] == typ) &
                                          (kappa_summary["loss"] == loss)].sort_values("node")
                    if block.empty:
                        continue
                    ax.plot(block["node"], block["kappa_mean"], ls, lw=1.1,
                            color=COLORS[typ], alpha=0.8 if seed == "seed0" else 0.45,
                            marker="o" if seed == "seed0" else "s", ms=2.5,
                            label=f"{seed} L{layer} {loss}")
        ax.axvline(EVENTS["seed0"], color="0.4", lw=0.8, ls=":")
        ax.axvline(EVENTS["seed1"], color="0.4", lw=0.8, ls="--")
        ax.set_title(f"{typ}: mean κ over heads")
        ax.set_xscale("log", base=2)
        ax.set_ylabel("κ_h (probe/train)")
        ax.set_yscale("symlog", linthresh=1e-5)
    for ax in axes[-1]:
        ax.set_xlabel("node (pre-update)")
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.suptitle("Stage 5 B: direction curvature by matrix type and seed")
    return save(fig, out / "fig01_kappa_overview.png")


def figure_kappa_heatmap(d, out: Path) -> Path:
    probe = d[d["loss"] == "probe"]
    rows = []
    labels = []
    for seed in ("seed0", "seed1"):
        for layer in sorted(probe[probe["seed"] == seed]["layer"].unique()):
            for typ in TYPES:
                block = probe[(probe["seed"] == seed) & (probe["layer"] == layer) &
                              (probe["type"] == typ)]
                if block.empty:
                    continue
                rows.append(block)
                labels.append(f"{seed[4]} L{layer} {typ}")
    nodes = sorted(probe["node"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for ax, loss in zip(axes, ("probe", "train")):
        mat = []
        ylabels = []
        for seed in ("seed0", "seed1"):
            for layer in sorted(probe[probe["seed"] == seed]["layer"].unique()):
                for typ in TYPES:
                    block = d[(d["seed"] == seed) & (d["layer"] == layer) &
                              (d["type"] == typ) & (d["loss"] == loss)]
                    if block.empty:
                        continue
                    pivot = block.pivot(index="head", columns="node", values="kappa").reindex(columns=nodes)
                    for head in pivot.index:
                        mat.append(pivot.loc[head].values)
                        ylabels.append(f"s{seed[-1]} L{layer} {typ}H{head}")
        mat = np.asarray(mat, float)
        v = np.sign(mat) * np.log10(1 + np.abs(mat) / 1e-5)
        im = ax.imshow(v, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_xticks(range(len(nodes))); ax.set_xticklabels(nodes, rotation=45)
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=5)
        ax.set_title(loss)
        ax.set_xlabel("node")
    fig.colorbar(im, ax=axes, shrink=0.7, label="signed log10(1+|κ|/1e-5)")
    fig.suptitle("Stage 5 B: head-wise κ heatmap")
    return save(fig, out / "fig02_kappa_heatmap.png")


def figure_probe_train(d, out: Path) -> Path:
    p = d[d["loss"] == "probe"].set_index(["seed", "node", "layer", "type", "head"])["kappa"]
    t = d[d["loss"] == "train"].set_index(["seed", "node", "layer", "type", "head"])["kappa"]
    p, t = p.align(t, join="inner")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for typ in TYPES:
        mask = np.array([idx[3] == typ for idx in p.index])
        x = p[mask].values; y = t[mask].values
        ax.scatter(np.sign(x) * np.log10(1 + np.abs(x) / 1e-6),
                   np.sign(y) * np.log10(1 + np.abs(y) / 1e-6),
                   s=8, alpha=0.45, color=COLORS[typ], label=typ)
    ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="0.5", lw=0.6)
    r = float(np.corrcoef(p.values, t.values)[0, 1])
    rho = float(spearmanr(p.values, t.values).statistic)
    ax.set_xlabel("probe: sign(κ)·log10(1+|κ|/1e-6)")
    ax.set_ylabel("train: sign(κ)·log10(1+|κ|/1e-6)")
    ax.set_title(f"probe vs train curvature (Pearson {r:.3f}, Spearman {rho:.3f})")
    ax.legend()
    return save(fig, out / "fig03_probe_vs_train.png")


def figure_cross_groups(cross_summary, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    layers = {"seed0": (0, 2, 3), "seed1": (0, 3, 4)}
    for ax, (seed, layer) in zip(axes.flat, (("seed0", 0), ("seed0", 3),
                                            ("seed1", 0), ("seed1", 3))):
        b = cross_summary[(cross_summary["seed"] == seed) & (cross_summary["layer"] == layer)]
        for group in PAIR_ORDER:
            g = b[b["pair_group"] == group].sort_values("node")
            if g.empty:
                continue
            ax.plot(g["node"], g["rho_median"], marker="o", ms=2.5, lw=1.1,
                    color=PAIR_COLORS[group], label=group)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"{seed} L{layer}: median off-diagonal ρ")
        ax.set_ylabel("median ρ_hk")
        ax.set_ylim(-0.08, 0.55)
    for ax in axes[-1]:
        ax.set_xlabel("node")
        ax.set_xscale("log", base=2)
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.suptitle("Stage 5 B: cross-head Hessian coupling, robust median")
    return save(fig, out / "fig04_cross_groups.png")


def figure_cross_heatmaps(c, out: Path) -> Path:
    nodes = (32, 128, 784, 1312, 1536)
    for seed in ("seed0", "seed1"):
        for layer in (0, 3):
            fig, axes = plt.subplots(1, len(nodes), figsize=(4 * len(nodes), 4.2),
                                     sharey=True)
            for ax, node in zip(axes, nodes):
                b = c[(c["seed"] == seed) & (c["node"] == node) & (c["layer"] == layer)]
                if b.empty:
                    ax.axis("off"); continue
                types = sorted(b["h_type"].unique())
                keys = [(t, h) for t in types for h in range(6)]
                idx = {k: i for i, k in enumerate(keys)}
                mat = np.full((len(keys), len(keys)), np.nan)
                for _, r in b.iterrows():
                    mat[idx[(r["h_type"], int(r["h_head"]))],
                        idx[(r["k_type"], int(r["k_head"]))]] = r["rho"]
                lo, hi = np.nanpercentile(mat, [2, 98])
                im = ax.imshow(np.clip(mat, -1, 1), cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_title(f"node {node}")
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels([f"{t}{h}" for t, h in keys], rotation=90, fontsize=5)
                if ax is axes[0]:
                    ax.set_yticks(range(len(keys)))
                    ax.set_yticklabels([f"{t}{h}" for t, h in keys], fontsize=5)
            fig.colorbar(im, ax=axes, shrink=0.7, label="ρ_hk (probe)")
            fig.suptitle(f"{seed} L{layer}: cross-head coupling at key nodes")
            save(fig, out / f"fig05_cross_{seed}_L{layer}.png")


def figure_lanczos(lz, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, typ in zip(axes.flat, TYPES):
        for seed in ("seed0", "seed1"):
            b = lz[(lz["seed"] == seed) & (lz["type"] == typ) & (lz["rank"] == 1)]
            for layer in sorted(b["layer"].unique()):
                g = b[b["layer"] == layer].sort_values("node")
                ax.plot(g["node"], g["eigenvalue"], marker="o", ms=2.5, lw=1.0,
                        alpha=0.8 if seed == "seed0" else 0.45,
                        label=f"{seed} L{layer}")
        ax.set_yscale("log")
        ax.set_title(f"{typ}: top-1 Ritz value")
        ax.set_ylabel("λ_top")
    for ax in axes[-1]:
        ax.set_xlabel("node")
        ax.set_xscale("log", base=2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Stage 5 B: Lanczos top Ritz value (30 steps)")
    return save(fig, out / "fig06_lanczos_top1.png")


def figure_lanczos_vs_kappa(d, lz, out: Path) -> Path:
    k = d[d["loss"] == "probe"][["seed", "node", "layer", "type", "head", "kappa"]]
    top = lz[lz["rank"] == 1][["seed", "node", "layer", "type", "head", "eigenvalue"]]
    m = top.merge(k, on=["seed", "node", "layer", "type", "head"])
    m["abs_kappa"] = m["kappa"].abs().clip(lower=1e-12)
    m["ratio"] = m["eigenvalue"] / m["abs_kappa"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for typ in TYPES:
        b = m[m["type"] == typ]
        ax.scatter(np.log10(b["abs_kappa"]), np.log10(b["eigenvalue"]),
                   s=12, alpha=0.55, color=COLORS[typ], label=typ)
    ax.set_xlabel("log10 |κ_h| (probe)")
    ax.set_ylabel("log10 top-1 Ritz value")
    ax.set_title("absolute scales")
    ax.legend(fontsize=7)
    ax = axes[1]
    bins = np.linspace(1.5, 4.0, 26)
    for typ in TYPES:
        b = m[m["type"] == typ]
        ax.hist(np.log10(b["ratio"].clip(lower=1e-30)), bins=bins, alpha=0.45,
                color=COLORS[typ], label=typ)
    ax.set_xlabel("log10(top-1 Ritz / |κ_h|)")
    ax.set_ylabel("count")
    ax.set_title("update direction is far below top curvature")
    ax.legend(fontsize=7)
    return save(fig, out / "fig07_lanczos_vs_kappa.png")


def figure_cross_seed(kappa_cross, matrix_cross, out: Path) -> Path:
    if kappa_cross.empty:
        return out / "fig08_cross_seed.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, layer in zip(axes, (0, 3)):
        b = kappa_cross[kappa_cross["layer"] == layer]
        for typ in TYPES:
            g = b[b["type"] == typ].sort_values("node")
            ax.plot(g["node"], g["kappa_spearman"], marker="o", ms=3,
                    color=COLORS[typ], label=typ)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"L{layer}: κ rank correlation seed0 vs seed1")
        ax.set_xlabel("node"); ax.set_ylabel("Spearman")
    axes[0].legend(fontsize=7)
    return save(fig, out / "fig08_cross_seed_kappa.png")


def figure_stage1(align, out: Path) -> Path | None:
    if align.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for role, color in (("G", "#1f77b4"), ("U", "#d62728")):
        b = align[align["role"] == role]
        ax.hist(b["spearman"].dropna(), bins=np.linspace(-1, 1, 21),
                alpha=0.55, color=color, label=role)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_xlabel("Spearman(κ_h, Stage 1 head energy share)")
    ax.set_ylabel("count")
    ax.set_title("B curvature ranking vs Stage 1 head dominance")
    ax.legend()
    return save(fig, out / "fig09_stage1_alignment.png")


def write_report(out: Path, val: dict, kappa_summary, cross_summary,
                 lanczos_summary, cross_seed, matrix_cross, align) -> Path:
    lines = []
    lines.append("# Stage 5 Module B — Head Hessian 曲率与交叉耦合\n")
    lines.append("## 1. 数据与验收\n")
    lines.append(f"- 方向曲率行数：{val['direction_rows']}；交叉项：{val['cross_rows']}；"
                 f"Lanczos：{val['lanczos_rows']}；replay：{val['replay_rows']}")
    lines.append(f"- replay 最大相对误差：G `{val['max_g_rel_error']:.3e}`，"
                 f"ΔW `{val['max_dw_rel_error']:.3e}`")
    lines.append(f"- 节点运行数：{val['total_node_runs']}；峰值显存：{val['max_peak_MiB']:.0f} MiB")
    lines.append(f"- 交叉矩阵最大相对不对称：`{val['cross_matrix_max_rel_asymmetry']:.3e}`")
    lines.append("- 节点：`" + ",".join(map(str, val["nodes"])) + "`；seed0/seed1。\n")
    lines.append("## 2. 方法\n")
    lines.append("- 节点 `u` 为 update u 前的模型；按 8×4 micro-batch 重放 G，再裁剪、"
                 "读取 Muon momentum 重算 `M_t, D, U, ΔW`。")
    lines.append("- L_probe = validation windows 0–7（2048 token）；L_train = 该 update 的 "
                 "32 个训练窗口，4×8-window HVP 平均。")
    lines.append("- `q_h=ΔW_hᵀHΔW_h`，`κ_h=q_h/||ΔW_h||²`；"
                 "`ρ_hk=q_hk/√|q_hq_k|`。交叉项对每个源 block 做完整模型 HVP。")
    lines.append("- Lanczos：每个 type 取 `||ΔW_h||` 最大的 Head，30 步，报告 top-4。\n")
    lines.append("## 3. 主要结果\n")
    for typ in TYPES:
        b = kappa_summary[(kappa_summary["type"] == typ) & (kappa_summary["loss"] == "probe")]
        lines.append(f"- **{typ}**：probe mean κ = `{b['kappa_mean'].mean():.4g}`，"
                     f"median = `{b['kappa_median'].median():.4g}`，"
                     f"负曲率比例 = `{b['negative_frac'].mean():.1%}`。")
    lines.append("")
    probe = kappa_summary[kappa_summary["loss"] == "probe"]
    for seed in ("seed0", "seed1"):
        o = probe[(probe["seed"] == seed) & (probe["type"] == "O")] \
            .groupby("node")["kappa_mean"].mean()
        v = probe[(probe["seed"] == seed) & (probe["type"] == "V")] \
            .groupby("node")["kappa_mean"].mean()
        lines.append(
            f"- **{seed} 轨迹**：O 从 node 32 的 `{o.loc[32]:.4f}` 降到 node 512 的 "
            f"`{o.loc[512]:.4f}`，到 1536 为 `{o.loc[1536]:.4f}`；V 从 `{v.loc[32]:.4f}` "
            f"降到 512 的 `{v.loc[512]:.4f}`，到 1536 为 `{v.loc[1536]:.4f}`。")
    lines.append("")
    for seed, node in (("seed0", 784), ("seed1", 1312)):
        b = probe[(probe["seed"] == seed) & (probe["node"] == node) & (probe["layer"] == 3)]
        if b.empty:
            continue
        terms = ", ".join(
            f"{r['type']}H{int(r['top_head'])}({r['top_kappa']:.4f})"
            for _, r in b.sort_values("type").iterrows())
        lines.append(f"- **{seed} node {node} L3 top heads**：{terms}。")
    lines.append("")
    for group in ("O-O", "O-V", "V-V", "Q-Q", "K-K", "Q-K"):
        b = cross_summary[cross_summary["pair_group"] == group]
        if b.empty:
            continue
        lines.append(f"- **{group}** 交叉耦合：median ρ = `{b['rho_median'].median():.3f}`，"
                     f"median |ρ| = `{b['abs_median'].median():.3f}`，"
                     f"|ρ|>1 比例 = `{b['abs_gt1_frac'].mean():.1%}`。")
    lines.append("")
    if not cross_seed.empty:
        l3 = cross_seed[cross_seed["layer"] == 3]
        lines.append(f"- 跨 seed：L3 的 κ 排名相关中位数 "
                     f"`{l3['kappa_spearman'].median():.3f}`；L0 "
                     f"`{cross_seed[cross_seed['layer']==0]['kappa_spearman'].median():.3f}`。")
    if not matrix_cross.empty:
        lines.append(f"- 跨 seed cross matrix：L3 median ρ 相关 "
                     f"`{matrix_cross[matrix_cross['layer']==3]['rho_spearman'].median():.3f}`；"
                     f"L0 `{matrix_cross[matrix_cross['layer']==0]['rho_spearman'].median():.3f}`。")
    if not align.empty:
        lines.append(f"- 与 Stage 1 Head energy 的 Spearman：G 中位数 "
                     f"`{align[align['role']=='G']['spearman'].median():.3f}`；"
                     f"U 中位数 `{align[align['role']=='U']['spearman'].median():.3f}`。")
    lines.append("")
    lines.append("## 4. 解释\n")
    lines.append("- O/V 的曲率显著高于 Q/K：O 在早期最高，随后下降；V 在后期上升；"
                 "Q/K 接近零且约 1/4–1/3 的 Head 方向为负曲率。")
    lines.append("- O-O、O-V、V-V 的归一化耦合中位数约 0.35，且几乎全为正；"
                 "Q/K 内部与跨类型耦合中位数只有 0.02–0.09，K-Q 在 q≈0 时出现不稳定离群值。")
    lines.append("- probe 与 train 曲率高度一致（Pearson≈0.94），说明结果不是训练 batch 的统计噪声。")
    lines.append("- 实际 ΔW 方向的 κ 比块内 top-1 Ritz 值低约 2–3 个数量级；"
                 "Muon 更新方向集中在低曲率子空间，这是 B 的机制线索，不是损失变化本身。")
    lines.append("- 跨 seed 的 L3 曲率排名和耦合矩阵比 L0 更一致；"
                 "具体主导 Head 仍不同（seed0 L2/L3 H2；seed1 L4 H4、L3 H2），"
                 "对应“具体路径不同、整体类型模式相似”。")
    lines.append("")
    lines.append("## 5. 限制\n")
    lines.append("- 局部 Hessian 是 pre-update 点的几何；κ/ρ 不是 ΔL。")
    lines.append("- 只有 8 个节点、到 1536；不能外推到 8192 的长期稳定性。")
    lines.append("- Q/K 的 q_h 接近 0，ρ_hk 在少数 pair 上会超过 1；报告使用 median，"
                 "并同时给出 |ρ|>1 比例。")
    lines.append("- Lanczos 30 步；`last_beta` 与 top1 的比值在 O/V 约 0.1，"
                 "谱结论只作粗粒度参考；方向曲率和交叉项是主结果。")
    lines.append("")
    lines.append("## 6. 产物\n")
    lines.append("- `summary/kappa_summary.csv`, `kappa_head_ranks.csv`")
    lines.append("- `summary/cross_summary.csv`, `cross_terms_enriched.csv`")
    lines.append("- `summary/lanczos_summary.csv`, `summary/cross_seed.csv`")
    lines.append("- `summary/stage1_alignment.csv`, `validation.json`")
    lines.append("- `figures/*.png`, `stage5_module_b_report.pdf`")
    path = out / "STAGE5_MODULE_B_FINDINGS.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pdf(out: Path, report: Path, figures: list[Path], val, kappa_summary,
             cross_summary) -> Path:
    pdf_path = out / "stage5_module_b_report.pdf"
    with PdfPages(pdf_path) as pdf:
        text_page(pdf, "Stage 5 Module B — Head Hessian", [
            "Validation",
            f"direction rows: {val['direction_rows']}",
            f"cross rows: {val['cross_rows']}",
            f"lanczos rows: {val['lanczos_rows']}",
            f"replay rows: {val['replay_rows']}",
            f"max replay G rel error: {val['max_g_rel_error']:.3e}",
            f"max replay DeltaW rel error: {val['max_dw_rel_error']:.3e}",
            f"max peak MiB: {val['max_peak_MiB']:.0f}",
            "",
            "Key numbers",
        ] + [
            f"{t} mean kappa: {v:.4g}"
            for t, v in kappa_summary[kappa_summary['loss'] == 'probe']
            .groupby('type')['kappa_mean'].mean().items()
        ] + [
            "",
            "Cross coupling (median rho)",
        ] + [
            f"{g}: {v:.3f}"
            for g, v in cross_summary.groupby('pair_group')['rho_median'].median().items()
        ])
        for path in figures:
            title = path.stem.replace("_", " ")
            image_page(pdf, path, title, [str(path)])
    return pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    setup()
    input_dir = Path(args.input)
    out = Path(args.out)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    d, c, lz, rv, summary = load(input_dir)
    val = validate(d, c, lz, rv, summary)
    (out / "validation.json").write_text(json.dumps(val, indent=2))

    kappa_summary, kappa_ranks = direction_tables(d)
    c2, cross_summary = cross_tables(c)
    lanczos_summary = lanczos_tables(lz)
    align = stage1_alignment(d, kappa_summary)
    cross_seed, matrix_cross = cross_seed_tables(d, c2)

    kappa_summary.to_csv(out / "summary/kappa_summary.csv", index=False)
    kappa_ranks.to_csv(out / "summary/kappa_head_ranks.csv", index=False)
    cross_summary.to_csv(out / "summary/cross_summary.csv", index=False)
    c2.to_csv(out / "summary/cross_terms_enriched.csv", index=False)
    lanczos_summary.to_csv(out / "summary/lanczos_summary.csv", index=False)
    cross_seed.to_csv(out / "summary/cross_seed.csv", index=False)
    matrix_cross.to_csv(out / "summary/cross_seed_matrix.csv", index=False)
    if not align.empty:
        align.to_csv(out / "summary/stage1_alignment.csv", index=False)

    figures = [
        figure_kappa_overview(kappa_summary, out / "figures"),
        figure_kappa_heatmap(d, out / "figures"),
        figure_probe_train(d, out / "figures"),
        figure_cross_groups(cross_summary, out / "figures"),
        figure_lanczos(lz, out / "figures"),
        figure_lanczos_vs_kappa(d, lz, out / "figures"),
        figure_cross_seed(cross_seed, matrix_cross, out / "figures"),
    ]
    figure_cross_heatmaps(c2, out / "figures")
    f = figure_stage1(align, out / "figures")
    if f:
        figures.append(f)
    figures = sorted((out / "figures").glob("*.png"))
    report = write_report(out, val, kappa_summary, cross_summary,
                          lanczos_summary, cross_seed, matrix_cross, align)
    pdf = make_pdf(out, report, figures, val, kappa_summary, cross_summary)
    print(json.dumps({"report": str(report), "pdf": str(pdf),
                      "figures": len(figures), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
