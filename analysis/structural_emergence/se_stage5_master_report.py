#!/usr/bin/env python3
"""Stage 5 master report: integrate A/B/C/D into one Chinese report and PDF."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("/mnt/d/muon_research/results/stage5")
A = ROOT / "A/analysis"
B = ROOT / "B/analysis"
C_ROOT = ROOT / "C"
C = C_ROOT / "analysis"
C2 = C_ROOT / "cross_interaction/analysis"
D = ROOT / "D/analysis"
D2 = ROOT / "D/timeline"
OUT = ROOT / "master"


def j(path):
    return json.loads(path.read_text())


def copy_figures():
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    selected = {
        "A_gradient_concentration.png": A / "figures/fig01_top_share.png",
        "A_gradient_source.png": A / "figures/fig04_source.png",
        "B_curvature.png": B / "figures/fig01_kappa_overview.png",
        "B_cross_coupling.png": B / "figures/fig04_cross_groups.png",
        "B_cross_heatmap.png": B / "figures/fig05_cross_seed0_L3.png",
        "C_taylor_actual.png": C / "figures/fig01_taylor_actual.png",
        "C_loss_subspace.png": C / "figures/fig02_delta_by_subspace.png",
        "C_ns_preservation.png": C / "figures/fig05_share_correlation.png",
        "D_gradient_timing.png": D / "figures/fig01_gradient_timing.png",
        "D_curvature_seed.png": D / "figures/fig02_curvature_types.png",
        "D_coupling_seed.png": D / "figures/fig03_coupling_seed.png",
        "D_event_heads.png": D / "figures/fig05_event_heads.png",
        "D2_l3_timeline.png": D2 / "figures/fig01_l3_top_heads.png",
        "C2_interaction.png": C2 / "figures/fig01_interaction_scatter.png",
    }
    paths = {}
    for name, source in selected.items():
        target = figs / name
        shutil.copyfile(source, target)
        paths[name] = target
    return paths


def text_page(pdf, title, lines, fontsize=8.5):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, title, fontsize=14, va="top", weight="bold")
    y = 0.925
    for line in lines:
        fig.text(0.06, y, line, fontsize=fontsize, va="top", family="monospace")
        y -= 0.022
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
    fig.text(0.03, 0.96, title, fontsize=13, weight="bold", va="top")
    fig.text(0.03, 0.12, path.name, fontsize=8, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = copy_figures()
    va = j(A / "validation.json")
    vb = j(B / "validation.json")
    vc = j(C / "validation.json")

    lines = []
    lines.append("# Stage 5 — Head 结构形成机制联合分析（Master Report）\n")
    lines.append("> A：梯度/表示；B：Hessian 曲率/耦合；C：Muon 动力学与 Head 级损失；D：跨 seed 机制比较。\n")
    lines.append("> 本报告只整合 A/B/C/D 的既有结果，不新增训练。\n")
    lines.append("## 0. 摘要\n")
    lines.append("Stage 5 的核心结论不是“某个 Head 一定占优”，而是：")
    lines.append("")
    lines.append("- **具体由哪个 Head 主导是 seed-dependent**：seed0 的 L2/L3 最终指向 H2，seed1 的 L3 也指向 H2，"
                 "但 L0/L2/L4 的具体 Head 不同；")
    lines.append("- **类型/层级的集中模式、曲率序和 Muon 的 NS 规律是 seed-invariant**："
                 "Q/K 强烈集中，O 基本不集中；O/V 正曲率且强耦合，Q/K 近零/不定；NS 保留 Q/K/V 排序但重排 O；")
    lines.append("- **L3 是两个 seed 的晚期共同收敛层**：H2 在晚期成为共同 top head；"
                 "但 D-2 全节点时间线显示 top-head 切换频繁，切换顺序 event/seed-specific，"
                 "不能普遍化为“曲率/NS 领先梯度”。")
    lines.append("- **Head 级损失主要由 V/O 子空间承担**，Q/K 的梯度集中不等于 Q/K 承担主要损失变化。")
    lines.append("- **C-2 没有发现 QK–VO 的强局部协同**：`I_H`、`I_actual` 都在 `1e-7` 量级，"
                 "相对独立损失变化的交互不到 1%；Q/K 的集中不能用 QK–VO 的二阶损失交互解释。")
    lines.append("- **D-3 连续指标**显示 top-head argmax 有噪声，但 A_G 与 B_abs 的排序相关约 "
                 "`0.77/0.81`，与 B_signed 仅 `0.59/0.49`；负曲率使 signed-κ 排序更不稳定。")
    lines.append("")
    lines.append("## 1. 研究问题与设计\n")
    lines.append("研究链路：")
    lines.append("")
    lines.append("```text")
    lines.append("初始表示/梯度差异 → 局部损失景观 → Muon 更新动力学 → Head 结构形成")
    lines.append("```")
    lines.append("")
    lines.append("- 数据：6 层 LLaMA，hidden 384，6 Heads，FFN 1024；SlimPajama 8x；")
    lines.append("- 轨迹：seed0/seed1，LR horizon 8192，Stage 4 采集到 1536；")
    lines.append("- A：31 个 pre-update 节点；B：8 个 Hessian 节点；C：20 个 role 节点 + 事件节点；")
    lines.append("- 不新增训练；所有分析基于 Stage 4 checkpoint 和 role。")
    lines.append("")
    lines.append("## 2. 数据与验收\n")
    lines.append("| 模块 | 数据量 | 关键验收 |")
    lines.append("|---|---:|---|")
    lines.append(f"| A | {va['head_rows']} head rows；{va['replay_rows']} replay | "
                 f"replay G `{va['max_g_rel_error']:.2e}`；Stage 1 G share diff `{va['stage1_train_gshare_max_abs_diff']:.2e}` |")
    lines.append(f"| B | {vb['direction_rows']} direction；{vb['cross_rows']} cross；{vb['lanczos_rows']} Lanczos | "
                 f"replay G `{vb['max_g_rel_error']:.2e}`，ΔW `{vb['max_dw_rel_error']:.2e}`；cross asym `{vb['cross_matrix_max_rel_asymmetry']:.2e}` |")
    lines.append(f"| C | {vc['role_rows']} role；{vc['loss_rows']} valid loss | "
                 f"Taylor/actual Pearson `{vc['taylor_actual_pearson']:.6f}`；rel residual `{vc['median_relative_residual']:.2%}` |")
    lines.append("")
    lines.append("验收说明：A 的 train G share 与 Stage 1 G role 最大绝对差 `1.66e-7`；"
                 "B 的交叉矩阵相对不对称 `6.87e-7`；C 的 Taylor 与 actual 几乎重合。")
    lines.append("")
    lines.append("## 3. Module A — Gradient & Representation Dynamics\n")
    lines.append("### 3.1 梯度优势在 Q/K，O 基本不集中\n")
    lines.append(f"A 的 Q/K top-1 G share 可到 `0.9–0.99`，effective heads 接近 1；"
                 f"O 始终接近 6 个有效 Head（top share 约 `0.2`）；V 介于中间。")
    lines.append("")
    lines.append("### 3.2 优势来源\n")
    lines.append("- Q/K/V：`g_share` 与 `dY_h` 份额的 Spearman 约 `0.83/0.83/0.64`；")
    lines.append("- O：`g_share` 与 `A_h` 输入份额的 Spearman 约 `0.92`；")
    lines.append("- Q/K 的 `cos_g_dy` 高，O 的 `cos_g_x` 接近 1；V 两者兼有。")
    lines.append("")
    lines.append("### 3.3 Seed 早期差异\n")
    lines.append("seed0 node 0 的 Q/K top share 约 `0.20–0.23`；seed1 已到 `0.37–0.43`。"
                 "seed1 的早期 Q/K 梯度不对称更强。")
    lines.append("")
    lines.append(f"![A gradient concentration](figures/A_gradient_concentration.png)")
    lines.append(f"![A gradient source](figures/A_gradient_source.png)")
    lines.append("")
    lines.append("## 4. Module B — Loss Landscape & Hessian\n")
    lines.append("### 4.1 曲率的类型序\n")
    lines.append("probe mean κ：O `0.0389` > V `0.0058` > K `0.00027` ≈ Q `0.00020`。"
                 "O/V 的负曲率比例为 0%，Q/K 分别为 28%/36%。")
    lines.append("")
    lines.append("### 4.2 交叉耦合\n")
    lines.append("median off-diagonal ρ：O-O `0.352`、O-V `0.371`、V-V `0.364`；"
                 "Q-Q `0.047`、K-K `0.058`、Q-K `0.037`。"
                 "O/V 子空间内部强正耦合，Q/K 弱耦合。")
    lines.append("")
    lines.append("### 4.3 probe 与 train\n")
    lines.append("probe 与 train 曲率的 Pearson `0.940`、Spearman `0.956`；"
                 "曲率模式不是训练 batch 的统计噪声。")
    lines.append("")
    lines.append("### 4.4 实际更新方向\n")
    lines.append("实际 ΔW 方向的 κ 比块内 top-1 Ritz 值低约 2–3 个数量级；"
                 "Muon 更新方向集中在低曲率子空间。Lanczos 30 步只作粗粒度谱参考。")
    lines.append("")
    lines.append(f"![B curvature](figures/B_curvature.png)")
    lines.append(f"![B cross coupling](figures/B_cross_coupling.png)")
    lines.append(f"![B cross heatmap](figures/B_cross_heatmap.png)")
    lines.append("")
    lines.append("## 5. Module C — Muon Dynamics & Head-level Loss\n")
    lines.append("### 5.1 NS 放大但不保留方向\n")
    lines.append("median `||U||/||G||`：O `53×`，V `145×`，Q `1084×`，K `1242×`；"
                 "`cos(G,U)` 仅约 `0.15–0.24`。")
    lines.append("")
    lines.append("### 5.2 Head 排序\n")
    lines.append("G→U 的 Head 份额相关：Q/K `0.90`，V `0.85`，O `0.39`。"
                 "NS 基本保留 Q/K/V 的 Head 排序，但会重排 O。")
    lines.append("")
    lines.append("### 5.3 Head 级损失\n")
    lines.append("median actual ΔL：all_O `-1.50e-4`，all_V `-5.6e-5`，all_QKV `-7.2e-5`，"
                 "all_Q `-6.4e-6`，all_K `-9.8e-6`。O/V 主导 Head 级损失。")
    lines.append("")
    lines.append("### 5.4 Taylor 与 actual\n")
    lines.append("Pearson `0.999997`，中位相对残差 `0.94%`，最大绝对残差 `6.3e-7`。"
                 "选定 Head 子空间的二阶展开在事件节点足够。")
    lines.append("")
    lines.append(f"![C Taylor actual](figures/C_taylor_actual.png)")
    lines.append(f"![C loss subspace](figures/C_loss_subspace.png)")
    lines.append(f"![C NS preservation](figures/C_ns_preservation.png)")
    lines.append("")
    lines.append("### 5.5 C-2：QK–VO 交互\n")
    lines.append("在 784/1312 的 L0/L3 上：`I_H` 最大约 `2.5e-7`，`I_actual` 最大约 `9.5e-7`，"
                 "`I_actual_norm` 最大约 `0.0074`；`I_actual` 只有 4/8 为正。")
    lines.append("`I_taylor` 与 `I_H` 精确一致，但与 `I_actual` 的符号并不总一致，"
                 "高阶残差常常大于二阶项。结论：在当前 L_probe 和事件节点上，"
                 "没有证据支持 QK–VO 强协同；Q/K 的梯度集中不是由 QK–VO 局部损失交互直接解释的。")
    lines.append("")
    lines.append(f"![C2 interaction](figures/C2_interaction.png)")
    lines.append("")
    lines.append("## 6. Module D — Cross-Seed Mechanism\n")
    lines.append("### 6.1 具体 Head 身份 seed-dependent\n")
    lines.append("seed0：L2/L3 全部 H2，L4 全部 H0，L0 Q/K H5（V/O H2）；"
                 "seed1：L4 全部 H4，L3 全部 H2，L2 Q/K/V H5（O H4），L0 Q/K/V H3（O H5）。")
    lines.append("")
    lines.append("### 6.2 类型/几何模式 seed-invariant\n")
    lines.append("- 两个 seed 都是 Q/K 强烈集中、O 约 6 个有效 Head；")
    lines.append("- B 的 O/V 高曲率/强耦合、Q/K 近零/不定在两边一致；")
    lines.append("- C 的 G→U 份额相关 Q/K `0.89–0.91`、V `0.84–0.86`、O `0.38–0.40` 在两边一致。")
    lines.append("")
    lines.append("### 6.3 L3 收敛与时间顺序\n")
    lines.append("seed1 node 1312 L3：A 的瞬时梯度 top 是 H3，B 的曲率 top 是 H2，"
                 "C 的 U/ΔW top 是 H2；到 node 1536，A 的梯度也回到 H2。"
                 "seed0 node 784 L3 的 A/B/C 直接一致指向 H2。")
    lines.append("")
    lines.append("### 6.4 D-2 全节点时间线\n")
    lines.append("D-2 展开 L3：A 31 节点、C 20 role 节点、B 8 Hessian 节点。")
    lines.append("- seed0：A Q/K 在 384–512 已到 H2，C_U 576、B 768；看起来 A→C→B，"
                 "但 C/B 节点覆盖稀疏。")
    lines.append("- seed1 1312：B/C_U 已到 H2，而 A Q/K 仍是 H3；方向相反。")
    lines.append("- A_G 的 Q/K top-head 在 seed0/seed1 分别切换 13/14 次；top-head argmax 噪声很大。")
    lines.append("- 因此只保留“H2 晚期收敛稳健”，不保留“曲率/NS 普遍领先梯度”。")
    lines.append("")
    lines.append("### 6.5 D-3 连续指标\n")
    lines.append("D-3 补了 top-1 share、top-2、margin、effective heads 和完整排序。"
                 "A_G 与 C_G 的排序完全一致；A_G 与 B_abs 约 `0.77/0.81`，"
                 "与 B_signed 约 `0.59/0.49`，与 C_U 约 `0.63/0.74`。"
                 "具体 Head 排序仍依赖指标，尤其 B 的 signed/abs 口径；"
                 "但连续指标不是纯噪声，类型级模式仍稳定。")
    lines.append("")
    lines.append(f"![D gradient timing](figures/D_gradient_timing.png)")
    lines.append(f"![D curvature seed](figures/D_curvature_seed.png)")
    lines.append(f"![D coupling seed](figures/D_coupling_seed.png)")
    lines.append(f"![D event heads](figures/D_event_heads.png)")
    lines.append("")
    lines.append("## 7. 综合机制链\n")
    lines.append("1. 过参数化结构提供多条候选路径：O/V 的正曲率与强耦合在两个 seed 都存在；")
    lines.append("2. 初始化和早期梯度选择具体路径：seed1 在 node 0 已有更强的 Q/K 不对称；")
    lines.append("3. 反向信号产生 Q/K Head 优势：Q/K 的 g_share 主要由 dY_h 承载；")
    lines.append("4. Muon/NS 把梯度优势转成更新分配：Q/K/V 排序保留，O 被重排；ΔW≈-lr·U；")
    lines.append("5. L3 晚期收敛到 H2：单事件上 B/C 可领先 A（seed1 1312），"
                 "但 seed0 的顺序相反；D-2 表明 lead-lag 不是普遍规律。")
    lines.append("6. Head 级损失主要由 V/O 承担：Q/K 的梯度集中不等于 Q/K 承担主要损失变化。")
    lines.append("7. C-2 未发现 QK–VO 强协同；Q/K 的角色更可能与本层路由、L_train 或其他层交互有关，"
                 "不能由事件点的局部 probe 交叉项解释。")
    lines.append("")
    lines.append("## 8. 对原假说的判断\n")
    lines.append("支持“过参数化提供候选路径、初始化影响具体路径、梯度和曲率决定优化条件、"
                 "Muon 改变更新分配、持续更新形成反馈”。")
    lines.append("")
    lines.append("更精确的结论是：**具体 Head 身份是 seed-dependent；类型/层级的集中模式、"
                 "曲率序和 NS 规律是 seed-invariant。**")
    lines.append("")
    lines.append("## 9. 限制\n")
    lines.append("- 只有 2 个 seed、到 1536；不能外推到 8192 的长期稳定性。")
    lines.append("- A/B/C 使用不同 loss 口径：A/B 是梯度/Hessian，C 是选定 Head 子空间；"
                 "只能做机制对齐，不能做同一数学量的恒等比较。")
    lines.append("- C 的 actual ΔL 只在 L_probe 的两个事件节点，不是完整 optimizer step。")
    lines.append("- B 的 Lanczos 30 步只作粗粒度谱参考；Q/K 的 q≈0 使少数 ρ 不稳定。")
    lines.append("")
    lines.append("## 10. 产物索引\n")
    lines.append("- A：`/mnt/d/muon_research/results/stage5_a_analysis/`")
    lines.append("- B：`/mnt/d/muon_research/results/stage5_hessian_analysis/`")
    lines.append("- C：`/mnt/d/muon_research/results/stage5_c_analysis/`")
    lines.append("- D：`/mnt/d/muon_research/results/stage5_d_analysis/`")
    lines.append("- 本报告：`/mnt/d/muon_research/results/stage5_master/`")
    md = OUT / "STAGE5_MASTER_REPORT.zh-CN.md"
    md.write_text("\n".join(lines) + "\n")

    pdf_path = OUT / "stage5_master_report.pdf"
    with PdfPages(pdf_path) as pdf:
        text_page(pdf, "Stage 5 — Head Structure Formation", [
            "A: gradient/representation dynamics",
            "B: loss landscape / Hessian curvature and cross-head coupling",
            "C: Muon dynamics and head-subspace loss contribution",
            "D: cross-seed mechanism comparison",
            "",
            "Central result",
            "Specific dominant-head identity is seed-dependent.",
            "Type/layer concentration, curvature order and NS rules are seed-invariant.",
            "L3 converges to H2 at late nodes.",
            "Lead-lag is event/seed-specific; D-2 shows noisy top-head switches.",
        ])
        text_page(pdf, "Data and validation", [
            f"A: {va['head_rows']} head rows, {va['replay_rows']} replay rows",
            f"A replay G max error: {va['max_g_rel_error']:.3e}",
            f"A Stage1 G-share max diff: {va['stage1_train_gshare_max_abs_diff']:.3e}",
            f"B: {vb['direction_rows']} direction rows, {vb['cross_rows']} cross rows",
            f"B replay G/DeltaW: {vb['max_g_rel_error']:.3e} / {vb['max_dw_rel_error']:.3e}",
            f"B cross asymmetry: {vb['cross_matrix_max_rel_asymmetry']:.3e}",
            f"C: {vc['role_rows']} role rows, {vc['loss_rows']} valid loss rows",
            f"C Taylor/actual Pearson: {vc['taylor_actual_pearson']:.6f}",
            f"C relative residual: {vc['median_relative_residual']:.3%}",
        ])
        for name in (
            "A_gradient_concentration.png", "A_gradient_source.png",
            "B_curvature.png", "B_cross_coupling.png", "B_cross_heatmap.png",
            "C_taylor_actual.png", "C_loss_subspace.png", "C_ns_preservation.png",
            "D_gradient_timing.png", "D_curvature_seed.png", "D_coupling_seed.png",
            "D_event_heads.png", "D2_l3_timeline.png", "C2_interaction.png",
        ):
            image_page(pdf, figures[name], name.replace("_", " "))
        text_page(pdf, "Integrated mechanism", [
            "1. Over-parameterized paths: O/V positive curvature and strong coupling.",
            "2. Init/early gradient selects a seed-specific path.",
            "3. Backward signal creates Q/K head advantage.",
            "4. Muon/NS preserves Q/K/V ordering but reallocates O.",
            "5. L3 converges to H2 late; no universal curvature/NS lead.",
            "6. Head-level loss is dominated by V/O, not by Q/K gradient concentration.",
            "C-2: QK-VO local interaction is tiny; no strong synergy at 784/1312.",
            "D-3: continuous ranks are not pure noise (A_G vs B_abs ~0.77/0.81),",
            "but signed-kappa ranks are less stable.",
            "",
            "Specific head identity: seed-dependent.",
            "Type-level pattern: seed-invariant.",
        ])
        text_page(pdf, "Limitations", [
            "Only 2 seeds and updates up to 1536; no 8192 extrapolation.",
            "A/B/C use different loss definitions; mechanism alignment, not identity.",
            "C actual loss is a selected head subspace on L_probe at two events.",
            "B Lanczos uses 30 steps; Q/K q near zero makes some rho unstable.",
        ])
    summary = {
        "a": va, "b": vb, "c": vc,
        "report": str(md), "pdf": str(pdf_path),
        "figures": len(figures),
    }
    (OUT / "master_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"report": str(md), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
