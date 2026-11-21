#!/usr/bin/env python3
"""Stage 5 D-3 — continuous head-ordering metrics (share, margin, effective heads)."""
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
A_PATH = Path("/mnt/d/muon_research/results/stage5/A/raw/stage5_a/head_dynamics.csv")
B_PATH = Path("/mnt/d/muon_research/results/stage5/B/raw/stage5_hessian/direction_curvature.csv")
C_PATH = Path("/mnt/d/muon_research/results/stage5/C/raw/stage5_c/role_dynamics.csv")
KEY_NODES = (0, 32, 128, 512, 768, 784, 1024, 1312, 1536)


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 165, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "figure.max_open_warning": 0,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def text_page(pdf, title, lines, fontsize=8.5):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, title, fontsize=14, va="top", weight="bold")
    y = 0.925
    for line in lines:
        fig.text(0.06, y, line, fontsize=fontsize, va="top", family="monospace")
        y -= 0.022
        if y < 0.04:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69)); y = 0.96
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def image_page(pdf, path, title):
    image = plt.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.74]); ax.imshow(image); ax.axis("off")
    fig.text(0.03, 0.96, title, fontsize=14, weight="bold", va="top")
    fig.text(0.03, 0.12, path.name, fontsize=8, family="monospace")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def share_metrics(group, col, method, seed, node, typ):
    g = group.sort_values(col, ascending=False)
    if g.empty:
        return None
    p = g[col].values.astype(float)
    p = np.maximum(p, 0)
    total = p.sum()
    if total <= 0:
        shares = np.full_like(p, 1 / len(p))
    else:
        shares = p / total
    order = ">".join(f"H{int(h)}" for h in g["head"].values)
    return {
        "seed": seed, "node": node, "layer": 3, "type": typ, "method": method,
        "top_head": int(g.iloc[0]["head"]),
        "top1": float(shares[0]), "top2": float(shares[1]) if len(shares) > 1 else np.nan,
        "margin": float(shares[0] - shares[1]) if len(shares) > 1 else np.nan,
        "effective_heads": float(np.exp(-(shares * np.log(shares + 1e-30)).sum())),
        "order": order,
    }


def build(a, b, c):
    rows = []
    for seed in ("seed0", "seed1"):
        for typ in TYPES:
            aa = a[(a["seed"] == seed) & (a["layer"] == 3) &
                   (a["type"] == typ) & (a["loss"] == "train")]
            for node, g in aa.groupby("node"):
                r = share_metrics(g, "g_share", "A_G", seed, int(node), typ)
                if r: rows.append(r)
            cc = c[(c["seed"] == seed) & (c["layer"] == 3) & (c["type"] == typ)]
            for method, col in (("C_G", "g_share"), ("C_U", "u_share"), ("C_DW", "dw_share")):
                for node, g in cc.groupby("node"):
                    r = share_metrics(g, col, method, seed, int(node), typ)
                    if r: rows.append(r)
            bb = b[(b["seed"] == seed) & (b["layer"] == 3) &
                   (b["type"] == typ) & (b["loss"] == "probe")]
            for node, g in bb.groupby("node"):
                signed = g.copy()
                r = share_metrics(signed, "kappa", "B_kappa_signed", seed, int(node), typ)
                if r: rows.append(r)
                absg = g.copy(); absg["abs_kappa"] = absg["kappa"].abs()
                r = share_metrics(absg, "abs_kappa", "B_kappa_abs", seed, int(node), typ)
                if r: rows.append(r)
    return pd.DataFrame(rows)


def rank_corr(a, b, c):
    rows = []
    for seed in ("seed0", "seed1"):
        for typ in TYPES:
            common = []
            for node in sorted(set(a["node"]) & set(c["node"])):
                aa = a[(a["seed"] == seed) & (a["node"] == node) & (a["layer"] == 3) &
                       (a["type"] == typ) & (a["loss"] == "train")].set_index("head")["g_share"]
                cc = c[(c["seed"] == seed) & (c["node"] == node) & (c["layer"] == 3) &
                       (c["type"] == typ)].set_index("head")
                if typ == "O":
                    pass
                for method, col in (("C_G", "g_share"), ("C_U", "u_share"), ("C_DW", "dw_share")):
                    common.append((node, aa, cc[col], method))
            for method in ("C_G", "C_U", "C_DW"):
                vals = [spearmanr(x, y).statistic for _, x, y, m in common if m == method]
                if vals:
                    rows.append({"seed": seed, "type": typ, "pair": f"A_G vs {method}",
                                 "median_spearman": float(np.nanmedian(vals))})
            for bmethod in ("B_kappa_signed", "B_kappa_abs"):
                vals = []
                for node in sorted(set(a["node"]) & set(b["node"])):
                    aa = a[(a["seed"] == seed) & (a["node"] == node) & (a["layer"] == 3) &
                           (a["type"] == typ) & (a["loss"] == "train")].set_index("head")["g_share"]
                    bb = b[(b["seed"] == seed) & (b["node"] == node) & (b["layer"] == 3) &
                           (b["type"] == typ) & (b["loss"] == "probe")].set_index("head")["kappa"]
                    if bmethod == "B_kappa_abs":
                        bb = bb.abs()
                    if len(aa) == 6 and len(bb) == 6:
                        vals.append(spearmanr(aa, bb).statistic)
                if vals:
                    rows.append({"seed": seed, "type": typ, "pair": f"A_G vs {bmethod}",
                                 "median_spearman": float(np.nanmedian(vals))})
    return pd.DataFrame(rows)


def fig_share_margin(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, typ in zip(axes.flat, ("Q", "K", "V", "O")):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            for method, color, marker in (("A_G", COLORS[typ], "o"), ("C_U", "#666666", "s")):
                g = df[(df["seed"] == seed) & (df["type"] == typ) & (df["method"] == method)]
                g = g.sort_values("node")
                if g.empty:
                    continue
                ax.plot(g["node"], g["top1"], ls, marker=marker, ms=2.5, lw=0.9,
                        color=color, alpha=0.85 if method == "A_G" else 0.55,
                        label=f"{seed} {method}")
        ax.set_xscale("log", base=2); ax.set_ylim(0.1, 1.02)
        ax.set_title(f"{typ}: top-1 share")
        ax.set_ylabel("share")
    for ax in axes[-1]:
        ax.set_xlabel("node")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Stage 5 D-3: continuous top-1 share")
    return save(fig, out / "fig01_top_share.png")


def fig_effective(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, typ in zip(axes.flat, TYPES):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            for method, color, marker in (("A_G", COLORS[typ], "o"), ("C_U", "#666666", "s")):
                g = df[(df["seed"] == seed) & (df["type"] == typ) & (df["method"] == method)]
                g = g.sort_values("node")
                if g.empty:
                    continue
                ax.plot(g["node"], g["effective_heads"], ls, marker=marker, ms=2.5, lw=0.9,
                        color=color, alpha=0.85 if method == "A_G" else 0.55)
        ax.set_xscale("log", base=2); ax.set_ylim(0.8, 6.2)
        ax.set_title(f"{typ}: effective heads")
    for ax in axes[-1]:
        ax.set_xlabel("node")
    for ax in axes[:, 0]:
        ax.set_ylabel("effective heads")
    fig.suptitle("Stage 5 D-3: effective head count")
    return save(fig, out / "fig02_effective_heads.png")


def fig_rank(rc, out):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    pairs = sorted(rc["pair"].unique())
    x = np.arange(len(pairs)); width = 0.35
    for i, seed in enumerate(("seed0", "seed1")):
        vals = [rc[(rc["seed"] == seed) & (rc["pair"] == p)]["median_spearman"].median()
                for p in pairs]
        ax.bar(x + (i - 0.5) * width, vals, width, label=seed)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=25, ha="right")
    ax.set_ylim(-1, 1); ax.set_ylabel("median Spearman")
    ax.set_title("Cross-method rank correlation at common nodes")
    ax.legend(fontsize=7)
    return save(fig, out / "fig03_cross_method_rank.png")


def fig_b_signed_abs(df, out):
    b = df[df["method"].isin(["B_kappa_signed", "B_kappa_abs"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = []
    for i, (seed, typ) in enumerate([(s, t) for s in ("seed0", "seed1") for t in ("Q", "K")]):
        g = b[(b["seed"] == seed) & (b["type"] == typ)]
        signed = g[g["method"] == "B_kappa_signed"].sort_values("node")
        absv = g[g["method"] == "B_kappa_abs"].sort_values("node")
        ax.plot(signed["node"], signed["top_head"], "o-", label=f"{seed} {typ} signed")
        ax.plot(absv["node"], absv["top_head"], "s--", label=f"{seed} {typ} |κ|")
    ax.set_xscale("log", base=2); ax.set_ylim(-0.5, 5.5); ax.set_yticks(range(6))
    ax.set_xlabel("node"); ax.set_ylabel("top head")
    ax.set_title("B: signed vs absolute κ top head")
    ax.legend(fontsize=5, ncol=2)
    return save(fig, out / "fig04_b_signed_abs.png")


def write_report(out, df, rc):
    lines = ["# Stage 5 D-3 — 连续 Head 排序指标\n"]
    lines.append("## 1. 数据\n")
    lines.append("- A_G、C_G/C_U/C_DW：top-1 share、top-2、margin、effective heads、完整排序。")
    lines.append("- B：signed-κ top 与 |κ|-top 分开；B 只有 8 个节点。")
    lines.append("- 该报告复用现有数据，不新增 Hessian。\n")
    lines.append("## 2. 固定节点表\n")
    lines.append("| seed | node | type | method | top | top1 | margin | effective |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    keys = df[df["node"].isin(KEY_NODES) & df["type"].isin(("Q", "K"))]
    for _, r in keys.iterrows():
        if str(r['method']).startswith("B_"):
            extra = "NA | NA | NA |"
        else:
            extra = f"{r['top1']:.3f} | {r['margin']:.3f} | {r['effective_heads']:.2f} |"
        lines.append(f"| {r['seed']} | {int(r['node'])} | {r['type']} | {r['method']} | "
                     f"H{int(r['top_head'])} | {extra}")
    lines.append("")
    lines.append("## 3. 主要观察\n")
    lines.append("- top-head argmax 的切换在连续指标下仍然频繁；H2 的晚期收敛是稳健的，"
                 "但表示成连续 margin 后，不能把每个 argmax 变化都解释为机制事件。")
    lines.append("- A_G 与 C_U 的 top share/effective heads 大方向一致；C 在早期更早集中，"
                 "但 C 的 role 节点不含 512/1024/1536。")
    lines.append("- B 的 signed-κ 与 |κ|-top 在 Q/K 上明显不同：例如 seed0 node 32 Q "
                 "signed 是 H4、|κ| 是 H0；seed0 node 512 Q signed 是 H4、|κ| 是 H2。"
                 "负曲率使“最大代数值”和“最大绝对曲率”回答不同问题。")
    lines.append("- 跨方法 rank correlation 通常不高，说明“哪个 Head 排第一”非常依赖指标；"
                 "但类型级模式（Q/K 集中、O 不集中）仍稳定。")
    lines.append("")
    if not rc.empty:
        lines.append("## 4. 跨方法 rank correlation\n")
        for pair in sorted(rc["pair"].unique()):
            b = rc[rc["pair"] == pair]
            lines.append(f"- {pair}: seed0 median `{b[b['seed']=='seed0']['median_spearman'].median():.3f}`，"
                         f"seed1 `{b[b['seed']=='seed1']['median_spearman'].median():.3f}`。")
        lines.append("")
    lines.append("## 5. 结论\n")
    lines.append("- 连续指标支持：Q/K 的集中是真实的类型级现象，O 基本不集中。")
    lines.append("- 但具体 Head 的排序对指标和节点很敏感；不建议再为了追求 lead-lag 补 Hessian。")
    lines.append("- 下一步应转向 A-2：为什么 ΔY 会在 Head 间不均匀，而不是继续追 Top Head 排名。")
    path = out / "STAGE5_D3_CONTINUOUS.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pdf(out, figs):
    path = out / "stage5_d3_continuous.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "Stage 5 D-3 — continuous metrics", [
            "Top-1 share, margin, effective heads, full rank order",
            "B signed-kappa vs |kappa| separated",
            "H2 convergence at late nodes is robust",
            "Top-head argmax is highly metric- and node-sensitive",
            "Do not add more Hessian nodes for lead-lag",
            "Next: A-2, why dY differs across heads",
        ])
        for fig in figs:
            image_page(pdf, fig, fig.stem.replace("_", " "))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    setup()
    out = Path(args.out)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(A_PATH); b = pd.read_csv(B_PATH); c = pd.read_csv(C_PATH)
    df = build(a, b, c)
    rc = rank_corr(a, b, c)
    df.to_csv(out / "summary/continuous_metrics.csv", index=False)
    rc.to_csv(out / "summary/rank_correlations.csv", index=False)
    figs = [fig_share_margin(df, out / "figures"),
            fig_effective(df, out / "figures"),
            fig_rank(rc, out / "figures"),
            fig_b_signed_abs(df, out / "figures")]
    report = write_report(out, df, rc)
    pdf = make_pdf(out, figs)
    print(json.dumps({"report": str(report), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
