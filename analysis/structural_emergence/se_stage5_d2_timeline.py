#!/usr/bin/env python3
"""Stage 5 D-2 — expand L3 head-identity timelines across all available nodes."""
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

TYPES = ("Q", "K", "V", "O")
COLORS = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c", "O": "#d62728"}
METHOD_COLORS = {
    "A_G": "#1f77b4", "B_kappa": "#d62728",
    "C_G": "#ff7f0e", "C_U": "#2ca02c", "C_DW": "#9467bd",
}
A_PATH = Path("/mnt/d/muon_research/results/stage5/A/raw/stage5_a/head_dynamics.csv")
B_PATH = Path("/mnt/d/muon_research/results/stage5/B/raw/stage5_hessian/direction_curvature.csv")
C_PATH = Path("/mnt/d/muon_research/results/stage5/C/raw/stage5_c/role_dynamics.csv")


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


def top_row(group, col):
    group = group.sort_values(col, ascending=False)
    top = group.iloc[0]
    second = group.iloc[1] if len(group) > 1 else None
    return {
        "top_head": int(top["head"]),
        "top_value": float(top[col]),
        "margin": float(top[col] - second[col]) if second is not None else float("nan"),
    }


def build_timeline(a, b, c):
    rows = []
    for seed in ("seed0", "seed1"):
        for typ in TYPES:
            aa = a[(a["seed"] == seed) & (a["layer"] == 3) &
                   (a["type"] == typ) & (a["loss"] == "train")]
            for node, g in aa.groupby("node"):
                r = top_row(g, "g_share")
                rows.append({"seed": seed, "node": int(node), "layer": 3,
                             "type": typ, "method": "A_G", "coverage": "31 nodes", **r})
            bb = b[(b["seed"] == seed) & (b["layer"] == 3) &
                   (b["type"] == typ) & (b["loss"] == "probe")]
            for node, g in bb.groupby("node"):
                r = top_row(g, "kappa")
                rows.append({"seed": seed, "node": int(node), "layer": 3,
                             "type": typ, "method": "B_kappa", "coverage": "8 nodes", **r})
            cc = c[(c["seed"] == seed) & (c["layer"] == 3) & (c["type"] == typ)]
            for method, col in (("C_G", "g_share"), ("C_U", "u_share"), ("C_DW", "dw_share")):
                for node, g in cc.groupby("node"):
                    r = top_row(g, col)
                    rows.append({"seed": seed, "node": int(node), "layer": 3,
                                 "type": typ, "method": method, "coverage": "20 nodes", **r})
    return pd.DataFrame(rows)


def switch_summary(tl):
    rows = []
    for key, g in tl.groupby(["seed", "type", "method"]):
        seed, typ, method = key
        g = g.sort_values("node")
        switches = []
        prev = None
        for _, r in g.iterrows():
            if prev is not None and int(r["top_head"]) != prev:
                switches.append(int(r["node"]))
            prev = int(r["top_head"])
        rows.append({"seed": seed, "type": typ, "method": method,
                     "n_switches": len(switches),
                     "switch_nodes": ",".join(map(str, switches)),
                     "first_head": int(g.iloc[0]["top_head"]),
                     "final_head": int(g.iloc[-1]["top_head"]),
                     "final_margin": float(g.iloc[-1]["margin"])})
    return pd.DataFrame(rows)


def overlap(tl):
    bnodes = sorted(tl[tl["method"] == "B_kappa"]["node"].unique())
    rows = []
    for seed in ("seed0", "seed1"):
        for typ in TYPES:
            for node in bnodes:
                vals = {}
                for method in ("A_G", "B_kappa", "C_G", "C_U", "C_DW"):
                    r = tl[(tl["seed"] == seed) & (tl["type"] == typ) &
                           (tl["method"] == method) & (tl["node"] == node)]
                    vals[method] = int(r["top_head"].iloc[0]) if not r.empty else -1
                rows.append({"seed": seed, "node": node, "type": typ, **vals})
    return pd.DataFrame(rows)


def fig_timeline(tl, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, sharey=True)
    for i, seed in enumerate(("seed0", "seed1")):
        for j, typ in enumerate(TYPES):
            ax = axes[i, j]
            for method in ("A_G", "C_G", "C_U", "C_DW", "B_kappa"):
                g = tl[(tl["seed"] == seed) & (tl["type"] == typ) &
                       (tl["method"] == method)].sort_values("node")
                if g.empty:
                    continue
                ax.plot(g["node"], g["top_head"], marker="o" if method == "B_kappa" else ".",
                        ms=5 if method == "B_kappa" else 3, lw=0.9 if method != "B_kappa" else 0,
                        alpha=0.9, color=METHOD_COLORS[method], label=method)
            ax.set_title(f"{seed} L3 {typ}")
            ax.set_ylim(-0.5, 5.5)
            ax.set_yticks(range(6))
            ax.set_xscale("log", base=2)
            ax.axvline(784, color="0.5", ls=":", lw=0.7)
            ax.axvline(1312, color="0.5", ls="--", lw=0.7)
    for ax in axes[-1]:
        ax.set_xlabel("node")
    for ax in axes[:, 0]:
        ax.set_ylabel("top head index")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Stage 5 D-2: L3 top-head timelines across available nodes")
    return save(fig, out / "fig01_l3_top_heads.png")


def fig_switches(sw, out):
    methods = ("A_G", "B_kappa", "C_G", "C_U", "C_DW")
    seeds = ("seed0", "seed1")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, seed in zip(axes, seeds):
        b = sw[sw["seed"] == seed]
        mat = np.zeros((len(methods), len(TYPES)))
        for i, m in enumerate(methods):
            for j, typ in enumerate(TYPES):
                r = b[(b["method"] == m) & (b["type"] == typ)]
                mat[i, j] = r["n_switches"].iloc[0] if not r.empty else np.nan
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=7)
        ax.set_title(seed)
    fig.colorbar(im, ax=axes, shrink=0.8, label="number of top-head switches")
    fig.suptitle("Stage 5 D-2: top-head switching is frequent")
    return save(fig, out / "fig02_switch_counts.png")


def fig_overlap(ov, out):
    methods = ("A_G", "B_kappa", "C_G", "C_U", "C_DW")
    labels, mat = [], []
    for seed in ("seed0", "seed1"):
        for typ in TYPES:
            b = ov[(ov["seed"] == seed) & (ov["type"] == typ)].sort_values("node")
            for _, r in b.iterrows():
                labels.append(f"{seed[-1]} {int(r['node'])} {typ}")
                mat.append([int(r[m]) for m in methods])
    fig, ax = plt.subplots(figsize=(6.5, max(5, 0.22 * len(labels))))
    im = ax.imshow(mat, aspect="auto", cmap="tab10", vmin=-0.5, vmax=5.5)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="top head")
    ax.set_title("At the 8 B nodes: A/B/C top heads")
    return save(fig, out / "fig03_overlap_heads.png")


def write_report(out, tl, sw, ov):
    lines = ["# Stage 5 D-2 — L3 Head 身份时间线\n"]
    lines.append("## 1. 数据覆盖\n")
    lines.append("- A_G：31 个节点；C_G/C_U/C_DW：20 个 role 节点；B_kappa：8 个 Hessian 节点。")
    lines.append("- L3、seed0/1、Q/K/V/O。")
    lines.append("- 只有 node `784` 与 `1312` 同时有 A/B/C 的 top head；其他节点不能做三方时序比较。")
    lines.append("- 该报告不新增 Hessian；B 的缺口无法用 A/C 填补。\n")
    lines.append("## 2. 共同覆盖节点（784 / 1312）\n")
    sel = ov[ov["node"].isin([784, 1312])]
    for _, r in sel.iterrows():
        lines.append(f"- {r['seed']} node {int(r['node'])} {r['type']}："
                     f"A_G H{r['A_G']} / B_kappa H{r['B_kappa']} / "
                     f"C_G H{r['C_G']} / C_U H{r['C_U']} / C_DW H{r['C_DW']}。")
    lines.append("")
    lines.append("## 3. 主要观察\n")
    lines.append("- L3 的 H2 收敛是**晚期稳健现象**：seed0 从 node 768 起 Q/K/V/O 多数为 H2；"
                 "seed1 在 1312/1536 也回到 H2（A/C/B 之间有局部切换）。")
    lines.append("- 但 **top-head 切换很频繁**；仅凭 argmax 不能把 H2 当作连续、单调的吸引子。"
                 "A_G 的 Q/K 在 seed0/seed1 分别有 13/14 次 top-head 切换，C_G 有 7–9 次。")
    lines.append("- 时间顺序并不一致：")
    lines.append("  - seed0：A 的 Q/K 在 384–512 已到 H2，C_U 在 576 到 H2，B κ 在 768 才到 H2；"
                 "这看起来像 A → C → B，但 C 缺少 512、B 缺少 576，不能把稀疏观测当成连续 lead-lag。")
    lines.append("  - seed1 node 1312：B κ 和 C_U 已到 H2，但 A 的 Q/K 仍是 H3；O 的 A/C_G 仍是 H4，"
                 "而 B/C_U 已到 H2。")
    lines.append("  - seed0 node 784：A/B/C 在 Q/K/V 上共同 H2；O 的 A/B/C_G 是 H3，而 C_U 已是 H2。")
    lines.append("- 更稳妥的表述：**H2 在晚期成为共同 top head；切换顺序是 event- and seed-specific，"
                 "top-head argmax 有噪声；没有普遍的“曲率/NS 一定领先梯度”。**")
    lines.append("")
    lines.append("## 4. Switch 统计（Q/K）\n")
    for seed in ("seed0", "seed1"):
        b = sw[(sw["seed"] == seed) & (sw["type"] == "Q")]
        for method in ("A_G", "B_kappa", "C_G", "C_U", "C_DW"):
            r = b[b["method"] == method]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"- {seed} Q {method}: {int(r['n_switches'])} switches，"
                         f"final H{int(r['final_head'])}，final margin `{float(r['final_margin']):.3f}`。")
    lines.append("")
    lines.append("## 5. 结论与下一步\n")
    lines.append("- 不建议仅凭当前 8 个 B 节点声称 lead-lag；A/C 的完整 L3 轨迹已显示 top-head 噪声。")
    lines.append("- 若仍要验证曲率领先，需要补 L3 Q/K 的稀疏 Hessian 节点（832/896/960/1152/1280/1408），"
                 "并用 top-share margin 而不是 argmax 做判据。")
    lines.append("- 这仍然只是 2 个 seed；跨 seed 稳健性需要 seed2/3。")
    path = out / "STAGE5_D2_L3_TIMELINE.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path

def make_pdf(out, figs):
    path = out / "stage5_d2_l3_timeline.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "Stage 5 D-2 — L3 head timeline", [
            "Coverage: A 31 nodes, C 20 role nodes, B 8 Hessian nodes",
            "H2 convergence is robust at late nodes.",
            "Top-head switching is frequent and noisy.",
            "No consistent lead-lag across seeds/events.",
            "seed0: A reaches H2 before B, but C/B coverage is sparse.",
            "seed1 1312: B/C H2 while A Q/K still H3.",
            "seed0 784: A/B/C agree on Q/K/V H2; O differs.",
            "Top-head argmax is not a reliable lead-lag statistic.",
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
    b = pd.read_csv(B_PATH)
    c = pd.read_csv(C_PATH)
    tl = build_timeline(a, b, c)
    sw = switch_summary(tl)
    ov = overlap(tl)
    tl.to_csv(out / "summary/l3_timeline.csv", index=False)
    sw.to_csv(out / "summary/switch_summary.csv", index=False)
    ov.to_csv(out / "summary/overlap_8nodes.csv", index=False)
    figs = [fig_timeline(tl, out / "figures"),
            fig_switches(sw, out / "figures"),
            fig_overlap(ov, out / "figures")]
    report = write_report(out, tl, sw, ov)
    pdf = make_pdf(out, figs)
    print(json.dumps({"report": str(report), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
