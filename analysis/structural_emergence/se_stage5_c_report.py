#!/usr/bin/env python3
"""Stage 5 Module C — role dynamics, Taylor/actual loss, and A/B alignment."""
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
LAYERS = {"seed0": (0, 2, 3), "seed1": (0, 3, 4)}
A_PATH = Path("/mnt/d/muon_research/results/stage5_a_final/stage5_a/head_dynamics.csv")
B_DIR = Path("/mnt/d/muon_research/analysis/structural_emergence/stage5_b_metrics")


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


def text_page(pdf, title, lines, fontsize=8.4):
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


def image_page(pdf, path, title):
    image = plt.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.74])
    ax.imshow(image); ax.axis("off")
    fig.text(0.03, 0.96, title, fontsize=14, weight="bold", va="top")
    fig.text(0.03, 0.12, str(path), fontsize=8, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load(input_dir: Path):
    role = pd.read_csv(input_dir / "role_dynamics.csv")
    loss = pd.read_csv(input_dir / "head_loss.csv")
    # B has V/O Hessian only at L0/L3.  Drop the historical L2/L4 V/O rows
    # from the first C run; keep Q/K and add all_QK when present.
    keep = {"all_Q", "all_K", "all_QK", "top_Q", "top_K"}
    mask = loss.apply(
        lambda r: (int(r["layer"]) in (0, 3)) or (r["subspace"] in keep), axis=1)
    loss = loss[mask].reset_index(drop=True)
    summary = json.loads((input_dir / "run_summary.json").read_text())
    return role, loss, summary


def validate(role, loss, summary):
    assert len(role) == 5760, len(role)
    assert len(loss) in (96, 108), len(loss)
    shares = role.groupby(["seed", "node", "layer", "type"])["g_share"].sum()
    assert (shares - 1).abs().max() < 1e-4
    assert np.isfinite(loss[["delta_L_actual", "delta_L_taylor"]].values).all()
    x = loss["delta_L_taylor"].values
    y = loss["delta_L_actual"].values
    pearson = float(np.corrcoef(x, y)[0, 1])
    rel = float(np.median(np.abs(x - y) / np.maximum(np.abs(y), 1e-30)))
    return {
        "role_rows": len(role), "loss_rows": len(loss),
        "taylor_actual_pearson": pearson,
        "median_relative_residual": rel,
        "max_abs_residual": float(np.max(np.abs(x - y))),
        "subspaces": sorted(loss["subspace"].unique().tolist()),
    }


def role_summary(role):
    rows = []
    for key, b in role.groupby(["seed", "node", "layer", "type"]):
        seed, node, layer, typ = key
        tg = b.loc[b["g_share"].idxmax()]
        tu = b.loc[b["u_share"].idxmax()]
        td = b.loc[b["dw_share"].idxmax()]
        rows.append({
            "seed": seed, "node": node, "layer": layer, "type": typ,
            "g_top_head": int(tg["head"]), "g_top_share": float(tg["g_share"]),
            "u_top_head": int(tu["head"]), "u_top_share": float(tu["u_share"]),
            "dw_top_head": int(td["head"]), "dw_top_share": float(td["dw_share"]),
            "cos_g_u_median": float(b["cos_g_u"].median()),
            "ratio_u_g_median": float(b["ratio_u_g"].median()),
            "ratio_dw_u_median": float(b["ratio_dw_u"].median()),
        })
    return pd.DataFrame(rows)


def event_alignment(role, events=(784, 1312)):
    a = pd.read_csv(A_PATH) if A_PATH.is_file() else None
    b = pd.read_csv(B_DIR / "direction_curvature.csv")
    rows = []
    for seed in ("seed0", "seed1"):
        for node in events:
            for layer in LAYERS[seed]:
                for typ in (TYPES if layer in (0, 3) else ("Q", "K")):
                    r = role[(role["seed"] == seed) & (role["node"] == node) &
                             (role["layer"] == layer) & (role["type"] == typ)]
                    if r.empty:
                        continue
                    cg = int(r.loc[r["g_share"].idxmax(), "head"])
                    cu = int(r.loc[r["u_share"].idxmax(), "head"])
                    cd = int(r.loc[r["dw_share"].idxmax(), "head"])
                    row = {"seed": seed, "node": node, "layer": layer, "type": typ,
                           "C_G_top": cg, "C_U_top": cu, "C_dW_top": cd}
                    if a is not None:
                        aa = a[(a["seed"] == seed) & (a["node"] == node) &
                               (a["loss"] == "train") & (a["layer"] == layer) &
                               (a["type"] == typ)]
                        row["A_g_top"] = int(aa.loc[aa["g_share"].idxmax(), "head"]) if not aa.empty else -1
                    bb = b[(b["seed"] == seed) & (b["node"] == node) &
                           (b["loss"] == "probe") & (b["layer"] == layer) &
                           (b["type"] == typ)]
                    row["B_kappa_top"] = int(bb.loc[bb["kappa"].idxmax(), "head"]) if not bb.empty else -1
                    rows.append(row)
    return pd.DataFrame(rows)


def fig_taylor(loss, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = loss["delta_L_taylor"].values
    y = loss["delta_L_actual"].values
    ax = axes[0]
    ax.scatter(x, y, s=12, alpha=0.7, color="#333333")
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=0.8)
    ax.set_xlabel("Taylor prediction"); ax.set_ylabel("actual ΔL_probe")
    ax.set_title("Taylor vs actual")
    ax = axes[1]
    ax.hist(y - x, bins=24, color="#1f77b4", alpha=0.75)
    ax.set_xlabel("actual - Taylor"); ax.set_ylabel("count")
    ax.set_title("residual")
    return save(fig, out / "fig01_taylor_actual.png")


def fig_delta_by_subspace(loss, out):
    med = loss.groupby("subspace")["delta_L_actual"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(range(len(med)), med.values, color="#4c72b0")
    ax.set_yticks(range(len(med))); ax.set_yticklabels(med.index)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_xlabel("median ΔL_probe (actual)")
    ax.set_title("Head-subspace loss change")
    return save(fig, out / "fig02_delta_by_subspace.png")


def fig_delta_heatmap(loss, out):
    b = loss[loss["subspace"] == "all_attn"].copy()
    nodes = sorted(b["node"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, seed in zip(axes, ("seed0", "seed1")):
        s = b[b["seed"] == seed]
        layers = [l for l in LAYERS[seed] if l in (0, 3)]
        mat = np.full((len(layers), len(nodes)), np.nan)
        for i, layer in enumerate(layers):
            for j, node in enumerate(nodes):
                r = s[(s["node"] == node) & (s["layer"] == layer)]
                if not r.empty:
                    mat[i, j] = r["delta_L_actual"].iloc[0]
        im = ax.imshow(mat, cmap="coolwarm", aspect="auto",
                       vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
        ax.set_xticks(range(len(nodes))); ax.set_xticklabels(nodes)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{x}" for x in layers])
        ax.set_title(seed); ax.set_xlabel("node")
    fig.colorbar(im, ax=axes, shrink=0.8, label="ΔL_probe (all attention)")
    return save(fig, out / "fig03_delta_heatmap.png")


def fig_role_amplification(role, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    b = role.groupby(["type", "layer"]).agg(
        ratio=("ratio_u_g", "median"), cos=("cos_g_u", "median")).reset_index()
    for ax, col, title in ((axes[0], "ratio", "median ||U|| / ||G||"),
                           (axes[1], "cos", "median cos(G,U)")):
        for typ in TYPES:
            g = b[b["type"] == typ]
            ax.plot(g["layer"], g[col], marker="o", color=COLORS[typ], label=typ)
        ax.set_xlabel("layer"); ax.set_title(title)
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    return save(fig, out / "fig04_role_amplification.png")


def fig_share_corr(role, out):
    rows = []
    for typ, b in role.groupby("type"):
        rows.append((typ, b["g_share"].corr(b["u_share"]),
                     b["u_share"].corr(b["dw_share"])))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w/2, [r[1] for r in rows], w, label="G vs U")
    ax.bar(x + w/2, [r[2] for r in rows], w, label="U vs ΔW")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 1.02); ax.set_ylabel("Pearson share correlation")
    ax.set_title("Does Muon preserve the head ordering?")
    ax.legend(fontsize=7)
    return save(fig, out / "fig05_share_correlation.png")


def fig_top_transfer(align, out):
    if align.empty:
        return out / "fig06_top_transfer.png"
    cols = [c for c in ("A_g_top", "B_kappa_top", "C_G_top", "C_U_top", "C_dW_top")
            if c in align.columns]
    labels = [f"{r['seed'][-1]} {r['node']} L{r['layer']} {r['type']}"
              for _, r in align.iterrows()]
    mat = align[cols].values
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.24 * len(labels))))
    im = ax.imshow(mat, aspect="auto", cmap="tab10", vmin=-0.5, vmax=5.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="head index")
    ax.set_title("Top head: A gradient / B curvature / C roles")
    return save(fig, out / "fig06_top_transfer.png")


def write_report(out, val, role, role_sum, loss, align):
    lines = ["# Stage 5 Module C — Muon 动力学与 Head 级损失贡献\n"]
    lines.append("## 1. 验收\n")
    lines.append(f"- role_dynamics：{val['role_rows']}；head_loss：{val['loss_rows']}")
    lines.append(f"- Taylor vs actual Pearson：`{val['taylor_actual_pearson']:.6f}`")
    lines.append(f"- 中位相对残差：`{val['median_relative_residual']:.3%}`；"
                 f"最大绝对残差：`{val['max_abs_residual']:.3e}`")
    lines.append("")
    lines.append("## 2. 方法\n")
    lines.append("- Role 动力学：20 个 role 节点/seed，读取 `G/M/D/U/W_before/W_after`。")
    lines.append("- 损失：事件节点 784/1312；`g_S` 用 L_probe 的一次普通 backward；"
                 "`q_hk` 用 B probe；actual 是只施加选定 Head 子空间的实际 ΔW。")
    lines.append("- 子空间：all Q/K/V/O、all QKV、all attn、top Q/K/V/O。")
    lines.append("")
    lines.append("## 3. Role 动力学\n")
    for typ in TYPES:
        b = role_sum[role_sum["type"] == typ]
        lines.append(f"- **{typ}**：median ||U||/||G|| = `{b['ratio_u_g_median'].median():.2f}`，"
                     f"median cos(G,U) = `{b['cos_g_u_median'].median():.3f}`。")
    corr = []
    for typ, b in role.groupby("type"):
        corr.append(f"{typ}: G/U share corr `{b['g_share'].corr(b['u_share']):.2f}`")
    lines.append("- Head 排序保持：" + "；".join(corr) + "。")
    lines.append("- `ΔW ≈ -lr·U`：`cos(U,ΔW)` 接近 `-1`，`||ΔW||/||U||` 接近学习率。")
    lines.append("")
    lines.append("## 4. Head 级损失\n")
    for sub in ("all_Q", "all_K", "all_QK", "all_V", "all_O", "all_QKV",
                "all_attn", "top_Q", "top_K", "top_V", "top_O"):
        b = loss[loss["subspace"] == sub]
        if b.empty:
            continue
        lines.append(f"- **{sub}**：median ΔL_actual = `{b['delta_L_actual'].median():.3e}`，"
                     f"median Taylor = `{b['delta_L_taylor'].median():.3e}`。")
    lines.append("")
    att = loss[loss["subspace"] == "all_attn"]
    for seed in ("seed0", "seed1"):
        b = att[att["seed"] == seed]
        lines.append(f"- **{seed} all_attn**：ΔL 范围 `{b['delta_L_actual'].min():.3e}` 到 "
                     f"`{b['delta_L_actual'].max():.3e}`；最大下降在 node/layer "
                     f"`{b.loc[b['delta_L_actual'].idxmin(), 'node']}`/"
                     f"`{b.loc[b['delta_L_actual'].idxmin(), 'layer']}`。")
    lines.append("")
    if not align.empty:
        lines.append("## 5. 事件节点的 A/B/C 对齐\n")
        for _, r in align.iterrows():
            lines.append(f"- {r['seed']} node {r['node']} L{r['layer']} {r['type']}："
                         f"A g_top H{r.get('A_g_top', -1)}；B κ_top H{r['B_kappa_top']}；"
                         f"C G_top H{r['C_G_top']}，U_top H{r['C_U_top']}，ΔW_top H{r['C_dW_top']}。")
        lines.append("")
    lines.append("## 6. 解释\n")
    lines.append("- Muon/NS 把 G 放大到 U（约 50–1200×），但 `cos(G,U)` 只有约 0.15–0.24；"
                 "NS 不是简单保留梯度方向。")
    lines.append("- Q/K/V 的 G→U Head 份额高度保持（约 0.85–0.90）；O 只有约 0.39，"
                 "NS 会重新分配 O 的 Head。")
    lines.append("- actual 与 Taylor 几乎完全一致（Pearson≈0.9997，中位相对残差约 1%），"
                 "说明选定 Head 子空间的二阶展开在事件节点已经足够。")
    lines.append("- all_O 的损失贡献通常最大，all_Q/K 很小；Head 级损失主要由 V/O 子空间承担。")
    lines.append("- seed1 node 1312 L3：A 的梯度 top 是 H3，B 的曲率 top 是 H2，"
                 "C 的 U/ΔW top 是 H2；即 NS 更新选择了 H2，解释了 A/B 的差异。"
                 "seed0 node 784 L2/L3 的 G/U/B 都指向 H2。")
    lines.append("")
    lines.append("## 7. 限制\n")
    lines.append("- C 只施加选定 Head 子空间，不是完整 optimizer step；辅助 AdamW 和其他 Muon block 未动。")
    lines.append("- 实际损失只在 L_probe 上；训练 batch 的损失变化未计算。")
    lines.append("- 只有两个事件节点；不能外推到所有节点或 8192 步。")
    lines.append("- Role 动力学只在 20 个 role 节点；不是全部 31 节点。")
    path = out / "STAGE5_MODULE_C_FINDINGS.zh-CN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pdf(out, figures, val):
    path = out / "stage5_module_c_report.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "Stage 5 Module C — Muon dynamics", [
            "Validation",
            f"role rows: {val['role_rows']}",
            f"head-loss rows: {val['loss_rows']}",
            f"Taylor vs actual Pearson: {val['taylor_actual_pearson']:.6f}",
            f"median relative residual: {val['median_relative_residual']:.3%}",
            f"max abs residual: {val['max_abs_residual']:.3e}",
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
    role, loss, summary = load(Path(args.input))
    val = validate(role, loss, summary)
    (out / "validation.json").write_text(json.dumps(val, indent=2))
    role_sum = role_summary(role)
    align = event_alignment(role)
    role_sum.to_csv(out / "summary/role_summary.csv", index=False)
    loss.to_csv(out / "summary/head_loss.csv", index=False)
    align.to_csv(out / "summary/event_alignment.csv", index=False)
    figs = [
        fig_taylor(loss, out / "figures"),
        fig_delta_by_subspace(loss, out / "figures"),
        fig_delta_heatmap(loss, out / "figures"),
        fig_role_amplification(role, out / "figures"),
        fig_share_corr(role, out / "figures"),
        fig_top_transfer(align, out / "figures"),
    ]
    report = write_report(out, val, role, role_sum, loss, align)
    pdf = make_pdf(out, figs, val)
    print(json.dumps({"report": str(report), "pdf": str(pdf), "figures": len(figs)}, indent=2))


if __name__ == "__main__":
    main()
