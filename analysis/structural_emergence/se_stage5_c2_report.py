#!/usr/bin/env python3
"""Stage 5 C-2 report: QK-VO cross-Hessian and actual loss interaction."""
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

SEED_COLORS = {"seed0": "#1f77b4", "seed1": "#d62728"}


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


def metrics(df):
    return {
        "rows": len(df),
        "max_abs_I_H": float(df["I_H"].abs().max()),
        "max_abs_I_actual": float(df["I_actual"].abs().max()),
        "max_abs_I_residual": float(df["I_residual"].abs().max()),
        "median_abs_I_actual": float(df["I_actual"].abs().median()),
        "median_abs_I_H": float(df["I_H"].abs().median()),
        "max_abs_I_actual_norm": float(df["I_actual_norm"].abs().max()),
        "I_actual_positive": int((df["I_actual"] > 0).sum()),
        "I_H_positive": int((df["I_H"] > 0).sum()),
        "max_taylor_minus_H": float((df["I_taylor"] - df["I_H"]).abs().max()),
        "I_H_I_actual_pearson": float(df["I_H"].corr(df["I_actual"]))
        if len(df) > 2 else float("nan"),
    }


def fig_scatter(df, out):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for seed in ("seed0", "seed1"):
        b = df[df["seed"] == seed]
        for _, r in b.iterrows():
            ax.scatter(r["I_H"], r["I_actual"], s=55,
                       color=SEED_COLORS[seed],
                       marker="o" if int(r["layer"]) == 0 else "s")
            ax.annotate(f"{int(r['node'])} L{int(r['layer'])}",
                        (r["I_H"], r["I_actual"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
    ax.set_xlabel("I_H"); ax.set_ylabel("I_actual")
    ax.set_title("QK–VO interaction: Hessian vs actual")
    ax.scatter([], [], color=SEED_COLORS["seed0"], label="seed0")
    ax.scatter([], [], color=SEED_COLORS["seed1"], label="seed1")
    ax.legend(fontsize=7)
    return save(fig, out / "fig01_interaction_scatter.png")


def fig_bars(df, out):
    df = df.copy()
    df["label"] = df["seed"].str[-1] + " " + df["node"].astype(str) + " L" + df["layer"].astype(str)
    x = np.arange(len(df)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 1.5*w, df["actual_QK"], w, label="actual QK", color="#1f77b4")
    ax.bar(x - 0.5*w, df["actual_VO"], w, label="actual VO", color="#ff7f0e")
    ax.bar(x + 0.5*w, df["I_actual"], w, label="I_actual", color="#d62728")
    ax.bar(x + 1.5*w, df["I_H"], w, label="I_H", color="#2ca02c")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=25)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_ylabel("loss / interaction")
    ax.set_title("QK, VO and their interaction")
    ax.legend(fontsize=7, ncol=4)
    return save(fig, out / "fig02_components.png")


def fig_residual(df, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(df["I_taylor"], df["I_actual"], s=55, color="#4c72b0")
    lo = min(df["I_taylor"].min(), df["I_actual"].min())
    hi = max(df["I_taylor"].max(), df["I_actual"].max())
    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=0.8)
    ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlabel("I_taylor = I_H"); ax.set_ylabel("I_actual")
    ax.set_title("Taylor interaction vs actual interaction")
    return save(fig, out / "fig03_residual.png")


def fig_norm(df, out):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    df = df.copy()
    df["label"] = df["seed"].str[-1] + " " + df["node"].astype(str) + " L" + df["layer"].astype(str)
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["rho_H"], 0.36, label="rho_H", color="#4c72b0")
    ax.bar(x + 0.18, df["rho_scale"], 0.36, label="rho_scale", color="#dd8452")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=25)
    ax.set_ylabel("normalized interaction")
    ax.set_title("Normalized QK–VO interaction")
    ax.legend(fontsize=7)
    return save(fig, out / "fig04_normalized.png")


def write_report(out, df, m):
    lines = ["# Stage 5 C-2 — QK–VO 交叉 Hessian 与实际交互\n"]
    lines.append("## 1. 验收\n")
    lines.append(f"- 行数：{m['rows']}；事件 784/1312；L0/L3；seed0/1。")
    lines.append(f"- `I_taylor - I_H` 最大绝对差：`{m['max_taylor_minus_H']:.3e}`（应接近 0）。")
    lines.append(f"- `I_H` 最大绝对值：`{m['max_abs_I_H']:.3e}`；"
                 f"`I_actual` 最大绝对值：`{m['max_abs_I_actual']:.3e}`。")
    lines.append(f"- `I_actual` 正号：{m['I_actual_positive']}/{m['rows']}；"
                 f"`I_H` 正号：{m['I_H_positive']}/{m['rows']}。")
    lines.append("")
    lines.append("## 2. 结果表\n")
    lines.append("| seed | node | layer | q_QK | q_VO | I_H | I_actual | I_taylor | I_residual | I_actual_norm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(f"| {r['seed']} | {int(r['node'])} | {int(r['layer'])} | "
                     f"{r['q_QK']:.2e} | {r['q_VO']:.2e} | {r['I_H']:.2e} | "
                     f"{r['I_actual']:.2e} | {r['I_taylor']:.2e} | "
                     f"{r['I_residual']:.2e} | {r['I_actual_norm']:.3e} |")
    lines.append("")
    lines.append("## 3. 解释\n")
    lines.append("- `q_QK` 比 `q_VO` 小约 2–3 个数量级；Q/K 的局部方向曲率很弱。")
    lines.append("- `I_H` 和 `I_actual` 都在 `1e-7` 量级；`I_actual_norm` 约 `0–0.008`，"
                 "即相对 QK+VO 的独立损失变化，交互不到 1%。")
    lines.append("- `I_actual` 多为正，表示两组更新在当前 probe/步长下更多是轻微干扰，"
                 "而不是明显协同。")
    lines.append("- `I_taylor=I_H` 与 `I_actual` 的符号并不总一致，`I_residual` 常常大于 `I_taylor`；"
                 "在当前尺度下高阶项/局部近似误差不可忽略。")
    lines.append("- **结论：没有证据支持 Q/K 与 V/O 在这些事件上存在强二阶协同；"
                 "Q/K 的梯度集中不能用 QK–VO 的局部损失交互来解释。**")
    lines.append("")
    lines.append("## 4. 限制\n")
    lines.append("- 只测 L_probe、两个事件节点、L0/L3；不是完整 optimizer step。")
    lines.append("- `rho_Euc` 需要 `HΔW_VO` 向量；B 未保存，本轮用 `rho_scale` 替代。")
    lines.append("- 交互很小，不能据此否定 Q/K 在 L_train、路由或其他层上的作用。")
    path = out / "STAGE5_C2_FINDINGS.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pdf(out, figs, m):
    path = out / "stage5_c2_report.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "Stage 5 C-2 — QK–VO interaction", [
            "Rows: 8 (784/1312, L0/L3, seed0/1)",
            f"max |I_H|: {m['max_abs_I_H']:.3e}",
            f"max |I_actual|: {m['max_abs_I_actual']:.3e}",
            f"max |I_actual_norm|: {m['max_abs_I_actual_norm']:.3e}",
            f"I_actual positive: {m['I_actual_positive']}/{m['rows']}",
            "",
            "No strong QK-VO synergy at these events.",
            "Higher-order residual dominates the tiny local interaction.",
        ])
        for fig in figs:
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
    df = pd.read_csv(Path(args.input) / "cross_interaction.csv")
    m = metrics(df)
    (out / "validation.json").write_text(json.dumps(m, indent=2))
    df.to_csv(out / "summary/cross_interaction.csv", index=False)
    figs = [fig_scatter(df, out / "figures"),
            fig_bars(df, out / "figures"),
            fig_residual(df, out / "figures"),
            fig_norm(df, out / "figures")]
    report = write_report(out, df, m)
    pdf = make_pdf(out, figs, m)
    print(json.dumps({"report": str(report), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
