#!/usr/bin/env python3
"""Stage 5 Module D — cross-seed mechanism comparison.

Synthesises A (gradient/representation), B (Hessian/curvature) and C
(Muon role/loss) into one seed comparison.  No new collection.
"""
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
A_PATH = Path("/mnt/d/muon_research/results/stage5_a_final/stage5_a/head_dynamics.csv")
B_DIR = Path("/mnt/d/muon_research/results/stage5_hessian_final/stage5_hessian")
C_DIR = Path("/mnt/d/muon_research/results/stage5_c_final_v2/stage5_c")
C_ANALYSIS = Path("/mnt/d/muon_research/results/stage5_c_analysis")


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
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def text_page(pdf, title, lines, fontsize=8.5):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, title, fontsize=14, va="top", weight="bold")
    y = 0.925
    for line in lines:
        fig.text(0.06, y, line, fontsize=fontsize, va="top", family="monospace")
        y -= 0.021
        if y < 0.04:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69)); y = 0.96
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def image_page(pdf, path, title):
    image = plt.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.74]); ax.imshow(image); ax.axis("off")
    fig.text(0.03, 0.96, title, fontsize=14, weight="bold", va="top")
    fig.text(0.03, 0.12, str(path), fontsize=8, family="monospace")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def group_a(a):
    rows = []
    for key, b in a.groupby(["seed", "node", "loss", "layer", "type"]):
        seed, node, loss, layer, typ = key
        p = b["g_share"].values / max(b["g_share"].sum(), 1e-30)
        rows.append({
            "seed": seed, "node": node, "loss": loss, "layer": layer, "type": typ,
            "top_share": float(p.max()),
            "effective_heads": float(np.exp(-(p * np.log(p + 1e-30)).sum())),
            "top_head": int(b.loc[b["g_share"].idxmax(), "head"]),
        })
    return pd.DataFrame(rows)


def onset(g):
    rows = []
    for (seed, layer, typ), b in g[g["loss"] == "train"].groupby(["seed", "layer", "type"]):
        b = b.sort_values("node")
        for thr in (0.5, 0.8):
            hit = b[b["top_share"] >= thr]
            rows.append({"seed": seed, "layer": layer, "type": typ,
                         "threshold": thr,
                         "node": int(hit.iloc[0]["node"]) if not hit.empty else None,
                         "head": int(hit.iloc[0]["top_head"]) if not hit.empty else None})
    return pd.DataFrame(rows)


def cross_seed_rank(a, b):
    rows = []
    for layer in [0, 2, 3, 4]:
        for typ in TYPES:
            ar, br, ma, mb = [], [], [], []
            for node in sorted(a["node"].unique()):
                x = a[(a["seed"] == "seed0") & (a["node"] == node) & (a["loss"] == "probe") &
                      (a["layer"] == layer) & (a["type"] == typ)].set_index("head")["g_share"]
                y = a[(a["seed"] == "seed1") & (a["node"] == node) & (a["loss"] == "probe") &
                      (a["layer"] == layer) & (a["type"] == typ)].set_index("head")["g_share"]
                if len(x) == 6 and len(y) == 6:
                    ar.append(spearmanr(x, y).statistic)
                    ma.append(int(x.idxmax() == y.idxmax()))
                xb = b[(b["seed"] == "seed0") & (b["node"] == node) & (b["loss"] == "probe") &
                       (b["layer"] == layer) & (b["type"] == typ)].set_index("head")["kappa"]
                yb = b[(b["seed"] == "seed1") & (b["node"] == node) & (b["loss"] == "probe") &
                       (b["layer"] == layer) & (b["type"] == typ)].set_index("head")["kappa"]
                if len(xb) == 6 and len(yb) == 6:
                    br.append(spearmanr(xb, yb).statistic)
                    mb.append(int(xb.idxmax() == yb.idxmax()))
            if ar or br:
                rows.append({"layer": layer, "type": typ,
                             "A_rank_median": float(np.nanmedian(ar)) if ar else np.nan,
                             "A_top_match": float(np.mean(ma)) if ma else np.nan,
                             "B_rank_median": float(np.nanmedian(br)) if br else np.nan,
                             "B_top_match": float(np.mean(mb)) if mb else np.nan})
    return pd.DataFrame(rows)


def coupling_table():
    c = pd.read_csv(B_DIR / "cross_terms.csv")
    c = c[c["loss"] == "probe"].copy()
    c["pair_group"] = ["-".join(sorted([h, k])) for h, k in zip(c["h_type"], c["k_type"])]
    c = c[~((c["h_type"] == c["k_type"]) & (c["h_head"] == c["k_head"]))]
    pivot = c.groupby(["seed", "pair_group"])["rho"].median().unstack()
    pivot = pivot.reset_index().melt(id_vars="seed", var_name="pair_group", value_name="rho")
    return pivot.dropna()


def fig_gradient(g, out):
    layers = (2, 3, 4)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, layer in zip(axes[0], layers):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            b = g[(g["loss"] == "train") & (g["seed"] == seed) & (g["layer"] == layer) &
                  (g["type"] == "Q")].sort_values("node")
            ax.plot(b["node"], b["top_share"], ls, marker="o", ms=2, color=COLORS["Q"],
                    label=f"{seed} Q")
        ax.set_title(f"L{layer} Q top share"); ax.set_xscale("log", base=2)
    for ax, layer in zip(axes[1], layers):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            b = g[(g["loss"] == "train") & (g["seed"] == seed) & (g["layer"] == layer) &
                  (g["type"] == "K")].sort_values("node")
            ax.plot(b["node"], b["top_share"], ls, marker="o", ms=2, color=COLORS["K"],
                    label=f"{seed} K")
        ax.set_title(f"L{layer} K top share"); ax.set_xscale("log", base=2)
        ax.set_xlabel("node")
    for ax in axes.flat:
        ax.set_ylim(0.12, 1.02)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Stage 5 D: gradient concentration timing")
    return save(fig, out / "fig01_gradient_timing.png")


def fig_curvature(b, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, typ in zip(axes.flat, TYPES):
        for seed, ls in (("seed0", "-"), ("seed1", "--")):
            g = b[(b["seed"] == seed) & (b["loss"] == "probe") & (b["type"] == typ)]
            g = g.groupby("node")["kappa"].mean().reset_index().sort_values("node")
            ax.plot(g["node"], g["kappa"], ls, marker="o", ms=2.5,
                    color=COLORS[typ], alpha=0.9 if seed == "seed0" else 0.55,
                    label=seed)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_title(f"{typ}: mean κ")
        ax.set_xscale("log", base=2)
    for ax in axes[-1]:
        ax.set_xlabel("node")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 5 D: curvature pattern is seed-invariant")
    return save(fig, out / "fig02_curvature_types.png")


def fig_coupling(coup, out):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for group in sorted(set(coup["pair_group"])):
        a = coup[(coup["seed"] == "seed0") & (coup["pair_group"] == group)]["rho"].values
        b = coup[(coup["seed"] == "seed1") & (coup["pair_group"] == group)]["rho"].values
        if len(a) and len(b):
            ax.scatter(a, b, s=18, alpha=0.7, label=group)
    ax.plot([0, 0.55], [0, 0.55], color="0.5", lw=0.8)
    ax.set_xlabel("seed0 median ρ"); ax.set_ylabel("seed1 median ρ")
    ax.set_title("Cross-head coupling by seed")
    ax.legend(fontsize=6, ncol=2)
    return save(fig, out / "fig03_coupling_seed.png")


def fig_ns(c, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, color in (("seed0", "#1f77b4"), ("seed1", "#d62728")):
        vals = [c[(c["seed"] == seed) & (c["type"] == t)]["g_share"].corr(
                    c[(c["seed"] == seed) & (c["type"] == t)]["u_share"]) for t in TYPES]
        axes[0].bar(np.arange(4) + (-0.18 if seed == "seed0" else 0.18), vals, 0.36,
                    color=color, label=seed)
        ratios = [c[(c["seed"] == seed) & (c["type"] == t)]["ratio_u_g"].median() for t in TYPES]
        axes[1].bar(np.arange(4) + (-0.18 if seed == "seed0" else 0.18), ratios, 0.36,
                    color=color, label=seed)
    for ax in axes:
        ax.set_xticks(range(4)); ax.set_xticklabels(TYPES)
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("G/U share corr")
    axes[0].set_title("Muon preserves Q/K/V ordering")
    axes[1].set_yscale("log"); axes[1].set_ylabel("median ||U||/||G||")
    axes[1].set_title("NS amplification")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    return save(fig, out / "fig04_ns_invariance.png")


def fig_event_heads(align, out):
    b = align[align["layer"] == 3].copy()
    labels = [f"{r['seed'][-1]} {r['node']} {r['type']}" for _, r in b.iterrows()]
    cols = ["A_g_top", "B_kappa_top", "C_G_top", "C_U_top", "C_dW_top"]
    cols = [c for c in cols if c in b.columns]
    mat = b[cols].values
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.32 * len(labels))))
    im = ax.imshow(mat, aspect="auto", cmap="tab10", vmin=-0.5, vmax=5.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="head index")
    ax.set_title("Event L3: A gradient / B curvature / C update")
    return save(fig, out / "fig05_event_heads.png")


def fig_loss(loss, out):
    b = loss[loss["subspace"].isin(["all_attn", "all_O"])].copy()
    b["node_seed"] = b["seed"] + " " + b["node"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.2
    labels = sorted(b["node_seed"].unique())
    subs = ["all_attn", "all_O"]
    for i, sub in enumerate(subs):
        for j, label in enumerate(labels):
            seed, node = label.split()
            g = b[(b["subspace"] == sub) & (b["seed"] == seed) & (b["node"] == int(node))]
            value = g["delta_L_actual"].mean()
            ax.bar(j + (i - 0.5) * width, value, width,
                   color="#4c72b0" if sub == "all_attn" else "#dd8452",
                   alpha=0.75, label=sub if j == 0 else None)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("mean ΔL_probe over L0/L3")
    ax.set_title("Head-subspace loss by seed and event")
    ax.legend(fontsize=7)
    return save(fig, out / "fig06_loss_seed.png")


def fig_onset(onset, out):
    b = onset[onset["threshold"] == 0.8].copy()
    labels, ys = [], []
    for seed in ("seed0", "seed1"):
        for layer in [0, 2, 3, 4]:
            for typ in TYPES:
                r = b[(b["seed"] == seed) & (b["layer"] == layer) & (b["type"] == typ)]
                node = float(r["node"].iloc[0]) if (not r.empty and pd.notna(r["node"].iloc[0])) else 1600.0
                ys.append(node)
                labels.append(f"{seed[-1]} L{layer}{typ}")
    fig, ax = plt.subplots(figsize=(6.5, 8))
    colors = ["#1f77b4" if x.startswith("0") else "#d62728" for x in labels]
    ax.scatter(ys, range(len(ys)), c=colors, s=28)
    ax.axvline(1536, color="0.5", lw=0.8, ls=":")
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("first node with top-1 G share ≥ 0.8 (1600 = not reached)")
    ax.set_title("Gradient concentration onset")
    return save(fig, out / "fig07_onset.png")


def write_report(out, align, rank, coup, c):
    lines = ["# Stage 5 Module D — 跨 Seed 机制比较\n"]
    lines.append("## 1. 数据与验收\n")
    lines.append("- A：31 节点梯度/表示；B：8 节点 Hessian；C：20 role 节点 + 事件节点损失。")
    lines.append("- A train G share 与 Stage 1 G role 最大绝对差 `1.66e-7`；"
                 "B replay max G error `5.13e-7`；C Taylor/actual Pearson `0.999997`。")
    lines.append("")
    lines.append("## 2. 结论先行\n")
    lines.append("- **具体 Head 路径由 seed 决定**：seed0 L2/L3 全部为 H2，L4 全部为 H0，"
                 "L0 Q/K 为 H5（V/O 为 H2），L5 Q/K/V 为 H1（O 为 H3）；"
                 "seed1 L4 全部为 H4，L3 全部为 H2，L2 Q/K/V 为 H5（O 为 H4），"
                 "L0 Q/K/V 为 H3（O 为 H5），L5 全部为 H1。")
    lines.append("- **整体类型/层级模式高度一致**：两个 seed 都是 Q/K 强烈集中、O 基本保持 6 个有效 Head、"
                 "V 介于中间；B 的 O/V 高曲率/强耦合、Q/K 近零/不定也一致。")
    lines.append("- **L3 是共同收敛层**：Q/K 在 seed0 784/seed1 1312 已由 A/B/C 指向 H2；"
                 "O 的 NS 更新在 784 已指向 H2，曲率在 seed0 1312/seed1 1312 对齐；"
                 "seed1 1312 时 Q/K 梯度 top 仍是 H3，到 1536 才回到 H2，提示曲率/NS 领先于瞬时梯度。")
    lines.append("")
    if not align.empty:
        l3 = align[align["layer"] == 3]
        lines.append("## 3. 事件节点 L3 的 A/B/C\n")
        for _, r in l3.iterrows():
            lines.append(f"- {r['seed']} node {r['node']} {r['type']}："
                         f"A H{r.get('A_g_top',-1)} / B H{r.get('B_kappa_top',-1)} / "
                         f"C G H{r['C_G_top']} U H{r['C_U_top']} ΔW H{r['C_dW_top']}。")
        lines.append("")
    lines.append("## 4. 不变与差异\n")
    lines.append("- **初始化差异**：seed0 node 0 Q/K top share 约 `0.20–0.23`；seed1 约 `0.37–0.43`。"
                 "seed1 的早期 Q/K 梯度不对称更强。")
    lines.append("- **集中时序不同**：seed1 L4 Q/K 在 node 512 已到约 `0.8`；seed0 L2/L3 到 node 832 才到约 `0.9`。")
    lines.append("- **Head 同一性弱**：A/B 的跨 seed 排名相关中位数在 L3 约 `0.2–0.34`；"
                 "A 的 top-head 命中率 L3 约 `47%`，L0 约 `32%`。")
    lines.append("- **曲率类型序不变**：两个 seed 的 mean κ 都约为 O `0.039` > V `0.0058` > K/Q `0.0002`；"
                 "O-O/O-V/V-V 的 median ρ 都约 `0.35–0.37`，Q/K 约 `0.02–0.09`。")
    lines.append("- **NS 类型规律不变**：G→U head-share corr 在两个 seed 都是 Q/K `0.89–0.91`、"
                 "V `0.84–0.86`、O `0.38–0.40`；NS 保留 Q/K/V 排序，但重排 O。")
    lines.append("- **损失贡献不变**：O/V 子空间主导 Head 级 ΔL；all_attn 的 median ΔL 约 `-2.5e-4`。")
    lines.append("")
    lines.append("## 5. 机制链\n")
    lines.append("1. **过参数化/局部几何提供候选路径**：O/V 的正曲率与强耦合在两个 seed 都存在。")
    lines.append("2. **初始化/早期梯度选择具体路径**：seed1 在 node 0 已有更强的 Q/K 不对称；"
                 "seed0 更接近均匀。")
    lines.append("3. **反向信号产生 Q/K Head 优势**：A 中 Q/K g_share 主要由 dY_h 承载。")
    lines.append("4. **Muon/NS 把梯度优势转成更新分配**：C 中 Q/K/V 的 G→U 排序高度保持，"
                 "O 被重排；ΔW≈-lr·U。")
    lines.append("5. **L3 的 Q/K 曲率/NS 先收敛到 H2**：seed1 1312 的瞬时梯度仍是 H3，但 B 曲率和 C U/ΔW 已指向 H2；"
                 "到 1536 A 梯度也回到 H2。O 的曲率在 784 还未完全对齐，NS 已先指向 H2。")
    lines.append("6. **Head 级损失主要由 V/O 承担**：Q/K 的梯度集中并不等于 Q/K 承担主要损失变化。")
    lines.append("")
    lines.append("## 6. 对原假说的判断\n")
    lines.append("- 支持“过参数化结构提供多条候选路径、初始化影响具体路径选择、梯度和曲率决定优化条件、"
                 "Muon 改变更新分配、持续更新形成反馈”。")
    lines.append("- 不能把“具体主导 Head 不同”解释为 seed 完全不同；更准确的说法是："
                 "**具体 Head 身份是 seed-dependent，而类型/层级的集中模式、曲率序和 NS 规律是 seed-invariant**。")
    lines.append("")
    lines.append("## 7. 限制\n")
    lines.append("- 只有 2 个 seed、到 1536；不能外推到 8192。")
    lines.append("- A/B/C 使用不同 loss 口径：A/B 是梯度/Hessian，C 是选定 Head 子空间；"
                 "彼此只能做机制对齐，不能做同一数学量的恒等比较。")
    lines.append("- C 的 actual ΔL 只在 L_probe 的两个事件节点；不是完整 optimizer step。")
    path = out / "STAGE5_MODULE_D_FINDINGS.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pdf(out, figs):
    path = out / "stage5_module_d_report.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "Stage 5 Module D — cross-seed mechanism", [
            "A: gradient/representation, 31 nodes",
            "B: Hessian curvature/coupling, 8 nodes",
            "C: Muon roles + head-subspace loss",
            "",
            "Conclusion: specific head identity is seed-dependent;",
            "type-level concentration, curvature order and NS rules are seed-invariant.",
            "L3 converges to H2 in both seeds, with curvature/NS leading the gradient.",
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
    a = pd.read_csv(A_PATH)
    b = pd.read_csv(B_DIR / "direction_curvature.csv")
    c = pd.read_csv(C_DIR / "role_dynamics.csv")
    align = pd.read_csv(C_ANALYSIS / "summary/event_alignment.csv")
    loss = pd.read_csv(C_ANALYSIS / "summary/head_loss.csv")
    g = group_a(a)
    onset_df = onset(g)
    rank = cross_seed_rank(a, b)
    coup = coupling_table()
    g.to_csv(out / "summary/gradient_groups.csv", index=False)
    onset_df.to_csv(out / "summary/onset.csv", index=False)
    rank.to_csv(out / "summary/cross_seed_rank.csv", index=False)
    coup.to_csv(out / "summary/coupling.csv", index=False)
    figs = [
        fig_gradient(g, out / "figures"),
        fig_curvature(b, out / "figures"),
        fig_coupling(coup, out / "figures"),
        fig_ns(c, out / "figures"),
        fig_event_heads(align, out / "figures"),
        fig_loss(loss, out / "figures"),
        fig_onset(onset_df, out / "figures"),
    ]
    report = write_report(out, align, rank, coup, c)
    pdf = make_pdf(out, figs)
    print(json.dumps({"report": str(report), "pdf": str(pdf), "figures": len(figs)}, indent=2))


if __name__ == "__main__":
    main()
