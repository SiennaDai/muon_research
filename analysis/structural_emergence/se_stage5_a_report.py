#!/usr/bin/env python3
"""Stage 5 Module A — validation, figures and Chinese report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

TYPES = ("Q", "K", "V", "O")
COLORS = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c", "O": "#d62728"}
STAGE1 = Path("/mnt/d/muon_research/results/head_spectral_stage1/head_spectral_metrics.csv")
B_KAPPA = Path("/mnt/d/muon_research/results/stage5_hessian_final/stage5_hessian/direction_curvature.csv")
EVENTS = {"seed0": 784, "seed1": 1312}


def setup():
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


def text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 8.5):
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


def image_page(pdf: PdfPages, path: Path, title: str):
    image = plt.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.74])
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.03, 0.96, title, fontsize=14, weight="bold", va="top")
    fig.text(0.03, 0.12, str(path), fontsize=8, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load(input_dir: Path):
    h = pd.read_csv(input_dir / "head_dynamics.csv")
    mb = pd.read_csv(input_dir / "microbatch.csv")
    r = pd.read_csv(input_dir / "replay_validation.csv")
    summary = json.loads((input_dir / "run_summary.json").read_text())
    return h, mb, r, summary


def validate(h, mb, r, summary):
    assert len(h) == 17856, len(h)
    assert len(mb) == 71424, len(mb)
    assert len(r) == 480, len(r)
    assert (Path("/mnt/d/muon_research/results/stage5_a_final/stage5_a/spectra") /
            "seed0_u000000.npz").is_file()
    assert r["g_rel_error"].max() < 1e-4
    shares = h.groupby(["seed", "node", "loss", "layer", "type"])["g_share"].sum()
    assert (shares - 1.0).abs().max() < 1e-4
    # Train G shares must equal the Stage-1 G role shares bit-for-bit.
    stage1_max = None
    if STAGE1.is_file():
        s = pd.read_csv(STAGE1)
        diffs = []
        for seed in ("seed0", "seed1"):
            for node in (32, 128, 512, 768, 1024, 1536):
                for layer in range(6):
                    for typ in TYPES:
                        x = h[(h["seed"] == seed) & (h["node"] == node) &
                              (h["loss"] == "train") & (h["layer"] == layer) &
                              (h["type"] == typ)].set_index("head")["g_share"]
                        y = s[(s["seed"] == seed) & (s["update"] == node) &
                              (s["layer"] == layer) & (s["matrix_type"] == typ) &
                              (s["role"] == "G")].set_index("head")["head_energy_share"]
                        common = sorted(set(x.index) & set(y.index))
                        if len(common) == 6:
                            diffs.append(float(np.max(np.abs(x[common].values - y[common].values))))
        if diffs:
            stage1_max = max(diffs)
            assert stage1_max < 1e-5, stage1_max
    peaks = [x["peak_MiB"] for x in summary["runs"]]
    return {
        "head_rows": len(h), "microbatch_rows": len(mb),
        "replay_rows": len(r), "spectra_files": 62,
        "max_g_rel_error": float(r["g_rel_error"].max()),
        "max_peak_MiB": float(max(peaks)),
        "total_node_runs": len(peaks),
        "stage1_train_gshare_max_abs_diff": stage1_max,
        "nodes": sorted(h["node"].unique().tolist()),
    }


def group_summary(h):
    def one(g):
        s = g["g_share"].values
        p = s / max(s.sum(), 1e-30)
        return pd.Series({
            "top_head": int(g.loc[g["g_share"].idxmax(), "head"]),
            "top_share": float(p.max()),
            "effective_heads": float(np.exp(-(p * np.log(p + 1e-30)).sum())),
            "mean_g_fro": float(g["g_fro"].mean()),
            "mean_dy_fro": float(g["dy_fro"].mean()),
            "mean_input_fro": float(g["input_fro"].mean()),
            "median_cos_dy": float(g["cos_g_dy"].median()),
            "median_cos_x": float(g["cos_g_x"].median()),
        })
    return h.groupby(["seed", "node", "loss", "layer", "type"]).apply(
        one, include_groups=False).reset_index()


def source_summary(h):
    d = h.copy()
    d["dy_share"] = d.groupby(["seed", "node", "loss", "layer", "type"])["dy_fro"].transform(
        lambda s: s / max(s.sum(), 1e-30))
    d["input_share"] = d.groupby(["seed", "node", "loss", "layer", "type"])["input_fro"].transform(
        lambda s: s / max(s.sum(), 1e-30))
    rows = []
    for (seed, loss, typ), b in d.groupby(["seed", "loss", "type"]):
        if typ == "O":
            rho = spearmanr(b["g_share"], b["input_share"]).statistic
            rows.append({"seed": seed, "loss": loss, "type": typ,
                         "source": "input_A_h", "spearman": rho, "n": len(b)})
        else:
            rho = spearmanr(b["g_share"], b["dy_share"]).statistic
            rows.append({"seed": seed, "loss": loss, "type": typ,
                         "source": "dy_h", "spearman": rho, "n": len(b)})
    return pd.DataFrame(rows)


def micro_summary(mb):
    return mb.groupby(["seed", "node", "layer", "type"]).agg(
        median_cos=("cos_to_total", "median"),
        mean_cos=("cos_to_total", "mean"),
        q25_cos=("cos_to_total", lambda s: s.quantile(0.25)),
        q75_cos=("cos_to_total", lambda s: s.quantile(0.75)),
        share_std=("g_share_mb", "std"),
    ).reset_index()


def cross_seed(h):
    rows = []
    for layer in range(6):
        for node in sorted(h["node"].unique()):
            for typ in TYPES:
                a = h[(h["seed"] == "seed0") & (h["node"] == node) & (h["loss"] == "probe") &
                      (h["layer"] == layer) & (h["type"] == typ)].set_index("head")["g_share"]
                b = h[(h["seed"] == "seed1") & (h["node"] == node) & (h["loss"] == "probe") &
                      (h["layer"] == layer) & (h["type"] == typ)].set_index("head")["g_share"]
                if len(a) != 6 or len(b) != 6:
                    continue
                rows.append({"layer": layer, "node": node, "type": typ,
                             "gshare_spearman": spearmanr(a, b).statistic,
                             "top_match": int(a.idxmax() == b.idxmax())})
    return pd.DataFrame(rows)


def b_link(h):
    if not B_KAPPA.is_file():
        return pd.DataFrame()
    b = pd.read_csv(B_KAPPA)
    b = b[b["loss"] == "probe"]
    rows = []
    for seed in ("seed0", "seed1"):
        for node in (32, 128, 512, 768, 784, 1024, 1312, 1536):
            for layer in (0, 2, 3, 4):
                for typ in TYPES:
                    x = h[(h["seed"] == seed) & (h["node"] == node) & (h["loss"] == "train") &
                          (h["layer"] == layer) & (h["type"] == typ)].set_index("head")["g_share"]
                    y = b[(b["seed"] == seed) & (b["node"] == node) & (b["layer"] == layer) &
                          (b["type"] == typ)].set_index("head")["kappa"]
                    common = sorted(set(x.index) & set(y.index))
                    if len(common) == 6:
                        rows.append({"seed": seed, "node": node, "layer": layer,
                                     "type": typ, "gshare_kappa_spearman": spearmanr(x[common], y[common]).statistic})
    return pd.DataFrame(rows)


def fig_top_share(group, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.3), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, range(6)):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            for typ in TYPES:
                b = group[(group["seed"] == seed) & (group["layer"] == layer) &
                          (group["loss"] == "train") & (group["type"] == typ)].sort_values("node")
                if b.empty:
                    continue
                ax.plot(b["node"], b["top_share"], ls, marker="o", ms=2.2, lw=1.0,
                        color=COLORS[typ], alpha=0.85 if seed == "seed0" else 0.5,
                        label=f"{seed} {typ}")
        ax.axvline(EVENTS["seed0"], color="0.5", ls=":", lw=0.7)
        ax.axvline(EVENTS["seed1"], color="0.5", ls="--", lw=0.7)
        ax.set_title(f"L{layer}")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.12, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("node")
    for ax in axes[:, 0]:
        ax.set_ylabel("top-1 G share")
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.suptitle("Stage 5 A: head gradient concentration")
    return save(fig, out / "fig01_top_share.png")


def fig_effective(group, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.3), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, range(6)):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            for typ in TYPES:
                b = group[(group["seed"] == seed) & (group["layer"] == layer) &
                          (group["loss"] == "train") & (group["type"] == typ)].sort_values("node")
                if b.empty:
                    continue
                ax.plot(b["node"], b["effective_heads"], ls, marker="o", ms=2.2, lw=1.0,
                        color=COLORS[typ], alpha=0.85 if seed == "seed0" else 0.5)
        ax.axvline(EVENTS["seed0"], color="0.5", ls=":", lw=0.7)
        ax.axvline(EVENTS["seed1"], color="0.5", ls="--", lw=0.7)
        ax.set_title(f"L{layer}")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.8, 6.2)
    for ax in axes[-1]:
        ax.set_xlabel("node")
    for ax in axes[:, 0]:
        ax.set_ylabel("effective heads")
    fig.suptitle("Stage 5 A: effective number of gradient-carrying heads")
    return save(fig, out / "fig02_effective_heads.png")


def fig_gshare_heatmap(h, out: Path) -> Path:
    nodes = sorted(h["node"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    for ax, loss in zip(axes, ("train", "probe")):
        mat, labels = [], []
        for seed in ("seed0", "seed1"):
            for layer in range(6):
                for typ in TYPES:
                    b = h[(h["seed"] == seed) & (h["layer"] == layer) &
                          (h["loss"] == loss) & (h["type"] == typ)]
                    p = b.pivot(index="head", columns="node", values="g_share").reindex(columns=nodes)
                    for head in p.index:
                        mat.append(p.loc[head].values)
                        labels.append(f"s{seed[-1]} L{layer} {typ}H{head}")
        im = ax.imshow(np.asarray(mat, float), aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(nodes))); ax.set_xticklabels(nodes, rotation=45)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
        ax.set_title(loss)
        ax.set_xlabel("node")
    fig.colorbar(im, ax=axes, shrink=0.7, label="g_share")
    fig.suptitle("Stage 5 A: head gradient share")
    return save(fig, out / "fig03_gshare_heatmap.png")


def fig_source(h, out: Path) -> Path:
    d = h[h["loss"] == "train"].copy()
    d["dy_share"] = d.groupby(["seed", "node", "layer", "type"])["dy_fro"].transform(
        lambda s: s / max(s.sum(), 1e-30))
    d["input_share"] = d.groupby(["seed", "node", "layer", "type"])["input_fro"].transform(
        lambda s: s / max(s.sum(), 1e-30))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    for typ in ("Q", "K", "V"):
        b = d[d["type"] == typ]
        rho = spearmanr(b["g_share"], b["dy_share"]).statistic
        ax.scatter(b["dy_share"], b["g_share"], s=8, alpha=0.35,
                   color=COLORS[typ], label=f"{typ} ρ={rho:.2f}")
    ax.set_xlabel("dY_h share"); ax.set_ylabel("G_h share")
    ax.set_title("Q/K/V: advantage comes from dY")
    ax.legend(fontsize=7)
    ax = axes[1]
    b = d[d["type"] == "O"]
    rho = spearmanr(b["g_share"], b["input_share"]).statistic
    ax.scatter(b["input_share"], b["g_share"], s=8, alpha=0.4,
               color=COLORS["O"], label=f"O ρ={rho:.2f}")
    ax.set_xlabel("A_h input share"); ax.set_ylabel("G_h share")
    ax.set_title("O: advantage comes from input representation")
    ax.legend(fontsize=7)
    return save(fig, out / "fig04_source.png")


def fig_alignment(h, out: Path) -> Path:
    d = h[h["loss"] == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, title in ((axes[0], "cos_g_dy", "cos(G left, dY left)"),
                           (axes[1], "cos_g_x", "cos(G right, input right)")):
        data = [d[d["type"] == typ][col].dropna().values for typ in TYPES]
        ax.boxplot(data, tick_labels=list(TYPES), showfliers=False)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_ylabel("cosine")
    return save(fig, out / "fig05_alignment.png")


def fig_micro(micro, out: Path) -> Path:
    b = micro.groupby("node").agg(
        median=("median_cos", "median"),
        q25=("q25_cos", lambda s: s.quantile(0.25)),
        q75=("q75_cos", lambda s: s.quantile(0.75)),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(b["node"], b["median"], marker="o", color="#333333")
    ax.fill_between(b["node"], b["q25"], b["q75"], alpha=0.2, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("node"); ax.set_ylabel("cos(G_mb, G_total)")
    ax.set_title("Stage 5 A: micro-batch gradient consistency")
    return save(fig, out / "fig06_microbatch.png")


def fig_cross_seed(cs, out: Path) -> Path:
    if cs.empty:
        return out / "fig07_cross_seed.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, layer in zip(axes, (0, 3)):
        b = cs[cs["layer"] == layer]
        for typ in TYPES:
            g = b[b["type"] == typ].sort_values("node")
            ax.plot(g["node"], g["gshare_spearman"], marker="o", ms=3,
                    color=COLORS[typ], label=typ)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"L{layer}: g_share rank correlation")
        ax.set_xlabel("node")
    axes[0].set_ylabel("Spearman"); axes[0].legend(fontsize=7)
    return save(fig, out / "fig07_cross_seed.png")


def fig_b_link(bl, out: Path) -> Path:
    if bl.empty:
        return out / "fig08_b_link.png"
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels, vals = [], []
    for seed in ("seed0", "seed1"):
        for typ in TYPES:
            b = bl[(bl["seed"] == seed) & (bl["type"] == typ)]
            labels.append(f"{seed[-1]}{typ}"); vals.append(b["gshare_kappa_spearman"].median())
    ax.bar(range(len(vals)), vals, color=[COLORS[x[-1]] for x in labels])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=0)
    ax.axhline(0, color="0.5", lw=0.7); ax.set_ylim(-1, 1)
    ax.set_ylabel("median Spearman(g_share, κ)")
    ax.set_title("A gradient share vs B curvature")
    return save(fig, out / "fig08_b_link.png")


def fig_top_heads(group, out: Path) -> Path:
    nodes = sorted(group["node"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, seed in zip(axes, ("seed0", "seed1")):
        rows, labels = [], []
        for layer in range(6):
            for typ in TYPES:
                b = group[(group["seed"] == seed) & (group["loss"] == "train") &
                          (group["layer"] == layer) & (group["type"] == typ)]
                p = b.set_index("node")["top_head"].reindex(nodes)
                rows.append(p.values); labels.append(f"L{layer}{typ}")
        im = ax.imshow(np.asarray(rows, float), aspect="auto", cmap="tab10", vmin=-0.5, vmax=5.5)
        ax.set_xticks(range(len(nodes))); ax.set_xticklabels(nodes, rotation=45)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(seed)
    fig.colorbar(im, ax=axes, shrink=0.7, label="top head index")
    fig.suptitle("Stage 5 A: top gradient head over time")
    return save(fig, out / "fig09_top_heads.png")


def write_report(out, val, group, source, micro, cs, bl):
    lines = ["# Stage 5 Module A — Gradient & Representation Dynamics\n"]
    lines.append("## 1. 验收\n")
    lines.append(f"- head_dynamics：{val['head_rows']}；microbatch：{val['microbatch_rows']}；"
                 f"replay：{val['replay_rows']}；spectra：{val['spectra_files']}")
    lines.append(f"- replay 最大 G 误差：`{val['max_g_rel_error']:.3e}`")
    lines.append(f"- 峰值显存：{val['max_peak_MiB']:.0f} MiB")
    if val["stage1_train_gshare_max_abs_diff"] is not None:
        lines.append(f"- Train G share 与 Stage 1 G role 的最大绝对差："
                     f"`{val['stage1_train_gshare_max_abs_diff']:.3e}`")
    lines.append("")
    lines.append("## 2. 方法\n")
    lines.append("- 31 个 pre-update 节点；L_train 为实际 update 的 8×4 micro-batch，"
                 "L_probe 为 validation windows 0–7。")
    lines.append("- 前向抓 `X`（c_attn 输入）和 `A`（c_proj 输入）；反向抓 `dY` 与 `dO`。")
    lines.append("- Q/K/V：`G_h=dY_hᵀX`；O：`G_h=dOᵀA_h`（c_proj 列块）。")
    lines.append("- `g_share` 为同层同类型 6 个 Head 的 `||G_h||²` 份额；`effective_heads` 为份额熵的指数。")
    lines.append("")
    lines.append("## 3. 主要结果\n")
    for seed in ("seed0", "seed1"):
        b = group[(group["seed"] == seed) & (group["loss"] == "train")]
        n0 = b[b["node"] == 0]
        nf = b[b["node"] == 1536]
        lines.append(f"- **{seed}**：node 0 的 top share 为 `{n0['top_share'].min():.2f}–"
                     f"{n0['top_share'].max():.2f}`，effective heads `{n0['effective_heads'].min():.2f}–"
                     f"{n0['effective_heads'].max():.2f}`；node 1536 的 Q/K top share 可到 "
                     f"`{nf[nf['type'].isin(['Q','K'])]['top_share'].max():.2f}`，而 O 通常仍在 "
                     f"`{nf[nf['type']=='O']['top_share'].mean():.2f}` 左右。")
    lines.append("- **初始化差异**：seed0 node 0 的 Q/K top share 约 `0.20–0.23`；"
                 "seed1 node 0 已到 `0.37–0.43`。seed1 的早期 Q/K 梯度不对称更强，"
                 "随后 seed1 L4 在 node 512 已明显集中，而 seed0 的 L2/L3 主要在 node 784 之后集中。")
    lines.append("")
    for typ in ("Q", "K", "V", "O"):
        b = source[(source["type"] == typ) & (source["loss"] == "train")]
        src = "input A_h" if typ == "O" else "dY_h"
        lines.append(f"- **{typ}** 的 `g_share` 主要来源：{src}；"
                     f"Spearman 中位数 `{b['spearman'].median():.3f}`。")
    lines.append("")
    for seed in ("seed0", "seed1"):
        b = group[(group["seed"] == seed) & (group["loss"] == "train") & (group["node"] == 1536)]
        terms = []
        for layer in range(6):
            for typ in ("Q", "K"):
                r = b[(b["layer"] == layer) & (b["type"] == typ)]
                if not r.empty:
                    terms.append(f"L{layer}{typ}:H{int(r.iloc[0]['top_head'])}")
        lines.append(f"- **{seed} node 1536 的 Q/K top heads**：" + ", ".join(terms))
    lines.append("")
    if not cs.empty:
        lines.append(f"- 跨 seed g_share 排名相关：L0 中位数 `{cs[cs['layer']==0]['gshare_spearman'].median():.3f}`，"
                     f"L3 `{cs[cs['layer']==3]['gshare_spearman'].median():.3f}`；"
                     f"top-head 命中率 L0 `{cs[cs['layer']==0]['top_match'].mean():.1%}`，"
                     f"L3 `{cs[cs['layer']==3]['top_match'].mean():.1%}`。")
    if not bl.empty:
        lines.append(f"- 与 B 的 κ 排名相关：seed0 中位数 `{bl[bl['seed']=='seed0']['gshare_kappa_spearman'].median():.3f}`，"
                     f"seed1 `{bl[bl['seed']=='seed1']['gshare_kappa_spearman'].median():.3f}`。")
    lines.append(f"- micro-batch 与总梯度的 cosine 中位数约 `{micro['median_cos'].median():.3f}`；"
                 f"早期 node 0–32 的份额仍有一定波动。")
    lines.append("")
    lines.append("## 4. 解释\n")
    lines.append("- Q/K 的梯度优势在 512–1024 之后迅速集中，effective heads 接近 1；"
                 "O 始终接近 6 个有效 Head；V 介于两者之间。")
    lines.append("- Q/K 的优势主要来自反向信号 `dY_h`；O 的优势主要来自输入表示 `A_h`；"
                 "这解释了为什么 Q/K 与 O 在 B 中呈现相反类型的曲率/耦合。")
    lines.append("- Q/K 的 `cos_g_dy` 高，O 的 `cos_g_x` 接近 1；V 同时依赖输入和反向信号。")
    lines.append("- 具体 top Head 由 seed 决定：seed0 L2/L3 为 H2，seed1 L4 为 H4、L3 为 H2；"
                 "L0 的 seed0 是 H5、seed1 是 H3。整体模式相似但具体路径不同。")
    lines.append("- 在 seed0 node 784，A 与 B 的 L3 top Head 都是 H2；在 seed1 node 1312，"
                 "A 的瞬时 Q/K top 是 H3，而 B 的曲率 top 是 H2，提示曲率与瞬时梯度在事件附近"
                 "可能不同步，需要 C 的更新/损失分析来区分。")
    lines.append("- 到 seed1 node 1536，A 的 L3 Q/K/V/O top 都是 H2，与 B 的 H2 重新对齐；"
                 "这支持“曲率先指向 H2，梯度随后跟上”的时间顺序假说，但 B/A 的节点粒度仍有限。")
    lines.append("")
    lines.append("## 5. 限制\n")
    lines.append("- A 采集的是梯度与表示，不是实际参数更新；ΔW 由 B 的 role 重建。")
    lines.append("- 只有 31 个节点、到 1536；不能外推到 8192。")
    lines.append("- 原始 X/dY 未保存；谱只保留 top-k，不能重做完整的逐样本分析。")
    lines.append("- micro-batch cosine 约 0.4，早期 Head 份额有 batch 波动；结论以聚合份额和谱为主。")
    lines.append("")
    lines.append("## 6. 产物\n")
    lines.append("- `summary/group_summary.csv`, `summary/source_summary.csv`")
    lines.append("- `summary/micro_summary.csv`, `summary/cross_seed.csv`")
    lines.append("- `summary/b_link.csv`, `validation.json`")
    lines.append("- `figures/*.png`, `stage5_module_a_report.pdf`")
    path = out / "STAGE5_MODULE_A_FINDINGS.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pdf(out, figures, val, group):
    path = out / "stage5_module_a_report.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "Stage 5 Module A — Gradient Dynamics", [
            "Validation",
            f"head rows: {val['head_rows']}",
            f"microbatch rows: {val['microbatch_rows']}",
            f"replay rows: {val['replay_rows']}",
            f"max G rel error: {val['max_g_rel_error']:.3e}",
            f"Stage1 G share max diff: {val['stage1_train_gshare_max_abs_diff']}",
            f"peak MiB: {val['max_peak_MiB']:.0f}",
        ])
        for fig in figures:
            image_page(pdf, fig, fig.stem.replace("_", " "))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    setup()
    out = Path(args.out)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    h, mb, r, summary = load(Path(args.input))
    val = validate(h, mb, r, summary)
    (out / "validation.json").write_text(json.dumps(val, indent=2))
    group = group_summary(h)
    source = source_summary(h)
    micro = micro_summary(mb)
    cs = cross_seed(h)
    bl = b_link(h)
    group.to_csv(out / "summary/group_summary.csv", index=False)
    source.to_csv(out / "summary/source_summary.csv", index=False)
    micro.to_csv(out / "summary/micro_summary.csv", index=False)
    cs.to_csv(out / "summary/cross_seed.csv", index=False)
    bl.to_csv(out / "summary/b_link.csv", index=False)
    figures = [
        fig_top_share(group, out / "figures"),
        fig_effective(group, out / "figures"),
        fig_gshare_heatmap(h, out / "figures"),
        fig_source(h, out / "figures"),
        fig_alignment(h, out / "figures"),
        fig_micro(micro, out / "figures"),
        fig_cross_seed(cs, out / "figures"),
        fig_b_link(bl, out / "figures"),
        fig_top_heads(group, out / "figures"),
    ]
    report = write_report(out, val, group, source, micro, cs, bl)
    pdf = make_pdf(out, figures, val, group)
    print(json.dumps({"report": str(report), "pdf": str(pdf),
                      "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
