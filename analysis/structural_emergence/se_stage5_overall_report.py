#!/usr/bin/env python3
"""Overall report: 8192-update, two-seed FP32 Muon study through Stage 5."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

R = Path("/mnt/d/muon_research/results")
S1 = R / "structural_emergence_analysis"
S2 = R / "muon_stage2_head_selective"
S3 = R / "stage3_gradient_coupling"
S5 = R / "stage5"
A = S5 / "A"
A2 = A / "attention_source/analysis"
B = S5 / "B/analysis"
C = S5 / "C/analysis"
C2 = S5 / "C/cross_interaction/analysis"
D = S5 / "D/analysis"
D2 = S5 / "D/timeline"
D3 = S5 / "D/continuous"
OUT = S5 / "overall"


def copy(src, name):
    dst = OUT / "figures" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def text_page(pdf, title, lines, fontsize=8.4):
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
    fig.text(0.03, 0.96, title, fontsize=13, weight="bold", va="top")
    fig.text(0.03, 0.12, path.name, fontsize=8, family="monospace")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def a2_seed_figure():
    h = pd.read_csv(S5 / "A/attention_source/stage5_a2/attention_dynamics.csv")
    b = h[(h["loss"] == "train") & (h["layer"] == 3)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    ax = axes[0]
    for typ, color in (("Q", "#1f77b4"), ("K", "#ff7f0e"), ("V", "#2ca02c")):
        x = b[(b["seed"] == "seed0") & (b["type"] == typ)].set_index(
            ["node", "head"])["dy_share"]
        y = b[(b["seed"] == "seed1") & (b["type"] == typ)].set_index(
            ["node", "head"])["dy_share"]
        common = sorted(set(x.index) & set(y.index))
        rho = spearmanr(x[common], y[common]).statistic
        ax.scatter(x[common], y[common], s=10, alpha=0.35, color=color,
                   label=f"{typ} ρ={rho:.2f}")
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8)
    ax.set_xlabel("seed0 dY share"); ax.set_ylabel("seed1 dY share")
    ax.set_title("L3: dY share across seeds")
    ax.legend(fontsize=7)
    ax = axes[1]
    for seed, color in (("seed0", "#1f77b4"), ("seed1", "#d62728")):
        for typ in ("Q", "K", "V"):
            g = b[(b["seed"] == seed) & (b["type"] == typ)]
            top = g.loc[g.groupby("node")["dy_share"].idxmax()]
            ax.scatter(top["a_entropy"], top["dy_share"], s=16, color=color,
                       marker={"Q": "o", "K": "s", "V": "^"}[typ],
                       label=f"{seed} {typ}")
    ax.set_xlabel("attention entropy of top head")
    ax.set_ylabel("top-1 dY share")
    ax.set_title("Top head: entropy vs dY")
    ax.legend(fontsize=5, ncol=2)
    path = OUT / "figures/A2_seed_difference.png"
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    figs = {}
    figs["S1_concentration"] = copy(S1 / "03_cross_seed/head_max_share_overlay.png",
                                     "S1_head_concentration.png")
    figs["S1_similarity"] = copy(S1 / "03_cross_seed/head_pattern_similarity_final.png",
                                 "S1_cross_seed_similarity.png")
    figs["S2_fused"] = copy(S2 / "fused_head_share_seed0_L2.png",
                            "S2_fused_head_share.png")
    figs["S2_gamma"] = copy(S2 / "gamma_bar_seed0_L2.png", "S2_gamma.png")
    figs["S3_gradient"] = copy(S3 / "mainlineA/A_pdY_vs_pG_seed0.png",
                               "S3_pdY_vs_pG.png")
    figs["S3_gram"] = copy(S3 / "mainlineB/B_gram_cosine_seed0.png",
                           "S3_gram_cosine.png")
    figs["A_top"] = copy(A / "analysis/figures/fig01_top_share.png",
                         "A_gradient_concentration.png")
    figs["A_source"] = copy(A / "analysis/figures/fig04_source.png",
                            "A_gradient_source.png")
    figs["A2_corr"] = copy(A2 / "figures/fig03_source_correlations.png",
                           "A2_source_correlations.png")
    figs["A2_event"] = copy(A2 / "figures/fig04_event_source.png",
                            "A2_event_source.png")
    figs["A2_seed"] = a2_seed_figure()
    figs["B_curv"] = copy(B / "figures/fig01_kappa_overview.png",
                          "B_curvature.png")
    figs["B_couple"] = copy(B / "figures/fig04_cross_groups.png",
                            "B_coupling.png")
    figs["C_taylor"] = copy(C / "figures/fig01_taylor_actual.png",
                            "C_taylor_actual.png")
    figs["C_loss"] = copy(C / "figures/fig02_delta_by_subspace.png",
                          "C_loss_subspace.png")
    figs["C_ns"] = copy(C / "figures/fig05_share_correlation.png",
                        "C_ns_preservation.png")
    figs["D_timing"] = copy(D / "figures/fig01_gradient_timing.png",
                            "D_gradient_timing.png")
    figs["D_couple"] = copy(D / "figures/fig03_coupling_seed.png",
                            "D_coupling_seed.png")
    figs["C2"] = copy(C2 / "figures/fig01_interaction_scatter.png",
                      "C2_interaction.png")
    figs["D3"] = copy(D3 / "figures/fig01_top_share.png", "D3_top_share.png")

    lines = []
    lines.append("# 从 8192-update 双 Seed 到 Stage 5：FP32 Reference Muon 的结构形成研究\n")
    lines.append("> 研究对象：30M LLaMA、SlimPajama 8x、seed0/1、FP32 Reference Muon、8192 updates。\n")
    lines.append("> 本报告汇总 Stage 1–5；数据、图和结论均来自对应阶段，不重新训练。\n")
    lines.append("## 0. 一页摘要\n")
    lines.append("我们最初的问题是：**Transformer 的多 Head/多层是否提供了大量候选计算路径，训练是否会把贡献集中到少数路径上？**")
    lines.append("")
    lines.append("结论是：")
    lines.append("")
    lines.append("1. **结构确实形成**：Q/K 的 Head 能量在约 500–1000 update 内集中到少数 Head，并在 8192 步内基本保持；V 较弱，O 最弱。")
    lines.append("2. **选择首先出现在梯度/优化器中间量**：G、M、D、U、ΔW 有强 Head 集中；W 本身仍接近均匀。")
    lines.append("3. **Muon/NS 主要做组间再分配**：fused QKV 中 V 组从约 98% 降到 75–79%，Q/K 从约 1% 升到 10–13%；类型内排序主要由 G/D 决定。")
    lines.append("4. **Q/K 的梯度集中来自反向信号**：Stage 3/5 的 p(G)≈p(ΔY)，A-2 进一步把它拆成 dA、attention entropy、score 和 Q/K/V 范数。")
    lines.append("5. **曲率能描述类型级优化几何，但不能稳定决定具体 Head**：O/V 正曲率、强耦合；Q/K 近零/不定、弱耦合；没有普遍 lead-lag。")
    lines.append("6. **具体 Head 身份 seed-dependent，类型/层级模式 seed-invariant**；L3 在晚期共同收敛到 H2。")
    lines.append("7. **QK–VO 没有强局部协同**：C-2 的 `I_H`、`I_actual` 都在 `1e-7` 量级；Q/K 的集中不能用事件点的 QK–VO 二阶交互解释。")
    lines.append("")
    lines.append("## 1. 实验系统与逻辑\n")
    lines.append("### 1.1 模型与训练")
    lines.append("- 6 层 LLaMA，hidden 384，6 Heads，FFN 1024，seq 256，vocab 50304；")
    lines.append("- 29,938,560 参数；Muon 组 10,616,832；辅助 AdamW 19,321,728；")
    lines.append("- SlimPajama 8x，fingerprint `ba3ac442...`；seed0/1，data_seed 1337，algorithm_seed 2026；")
    lines.append("- FP32 Reference Muon，8192 updates；NS 5 步，系数 `(3.4445,−4.775,2.0315)`，eps `1e−7`。")
    lines.append("")
    lines.append("### 1.2 五个阶段的分工")
    lines.append("| 阶段 | 作用 | 主要证据 |")
    lines.append("|---|---|---|")
    lines.append("| Stage 1 | 结构是否存在、何时形成、是否跨 seed | 8192 步，18 Snapshot/seed（36 total），Head/行列/谱 |")
    lines.append("| Stage 2 | Muon 的哪一步产生 Head 选择 | G/M/D/U/ΔW Role，NS 精确 replay |")
    lines.append("| Stage 3 | 梯度集中从哪里来、跨投影如何耦合 | G=ΔYᵀX，V 主导 fused NS |")
    lines.append("| Stage 4 | 为机制分析补节点资产 | 31 个 pre-update checkpoint + 20 个 QKV/O Role |")
    lines.append("| Stage 5 | 把梯度、曲率、更新、损失和 seed 串起来 | A/B/C/D + A-2/C-2/D-3 |")
    lines.append("")
    lines.append("## 2. Stage 1：结构存在，而且很早形成\n")
    lines.append("- Q/K 出现强而持久的 Head 能量集中：多数主导层在 update 500–1000 内达到高份额，之后到 8192 基本不迁移。")
    lines.append("- V 的集中度较低（最高约 0.5–0.7），O 最弱（约 0.25–0.55），形成更慢。")
    lines.append("- Q/K 的全矩阵有效秩大幅下降（例如 seed0 L2Q 192→53，L3K 197→53）；V/O/MLP 没有同样塌缩。")
    lines.append("- 跨 seed：同位置 Head 份额相关仅中位 `0.26`，允许同层内最佳匹配后 `0.77`，排序后模式 `0.965`。")
    lines.append("")
    lines.append("结论：**“少数 Head 主导且长期保持”的结构形式高度可重复；具体 Head 位置几乎不可重复。**")
    lines.append("")
    lines.append(f"![Stage 1 concentration](figures/{figs['S1_concentration'].name})")
    lines.append(f"![Stage 1 similarity](figures/{figs['S1_similarity'].name})")
    lines.append("")
    lines.append("## 3. Stage 2：Muon 的选择机制\n")
    lines.append("- Head 集中最早出现在 **G**，随后进入 M_t、D、U、ΔW；W_before/W_after 的 Head 份额仍接近均匀。")
    lines.append("- fused QKV 中 V 组占约 98% 的 D 能量；NS 后 V 降到约 75–79%，Q/K 从约 1% 升到 10–13%。")
    lines.append("- 类型内主导 Head 的排序主要由 G/D 决定；NS 主要做组间再分配和份额锐化。")
    lines.append("- 精确 NS replay：U 最大相对误差 `1.28e-5`；gamma 预测误差 `1e-7`；p(U) 与 p(ΔW) 相关 `0.99`。")
    lines.append("")
    lines.append("结论：**Head 选择/排序主要由梯度阶段给出；NS 改变幅度和组间分配。**")
    lines.append("")
    lines.append(f"![Stage 2 fused share](figures/{figs['S2_fused'].name})")
    lines.append(f"![Stage 2 gamma](figures/{figs['S2_gamma'].name})")
    lines.append("")
    lines.append("## 4. Stage 3：梯度来源与跨投影\n")
    lines.append("- `G=ΔYᵀX`；`p(G)` 与 `p(ΔY)` 的逐层 Spearman 为 `0.89–0.99`。Head 梯度集中绝大部分已存在于反向信号。")
    lines.append("- 输入 X 的有效秩随层升高（L2 约 14–17，L5 约 147）；X 主要做方向性加权。")
    lines.append("- fused QKV 中 V 组平均份额 `0.847`，Q/K `0.079/0.074`；Q/K 的输出几乎由 V 的 Gram 决定（Q+V 余弦 `0.988`）。")
    lines.append("- 早期 G 的 Head 排序可预测后期 U 的排序（Q/K Spearman `0.77–0.87`）。")
    lines.append("")
    lines.append("结论：**Q/K 的 Head 选择在梯度阶段已经形成；V 主导 fused NS 的全局尺度，Q/K 被 V 抑制。**")
    lines.append("")
    lines.append(f"![Stage 3 pdY vs pG](figures/{figs['S3_gradient'].name})")
    lines.append(f"![Stage 3 gram](figures/{figs['S3_gram'].name})")
    lines.append("")
    lines.append("## 5. Stage 4：为机制分析补节点\n")
    lines.append("- 沿用原轨迹，重新跑到 1536，保存 31 个 pre-update 节点：完整模型、Muon momentum、pre-take sampler、RNG、window ids。")
    lines.append("- 20 个新节点额外保存 QKV_fused/O 的 7 个 Role；10 个旧节点复用 Snapshot Role。")
    lines.append("- 不保存辅助 AdamW moments；A/B/C 通过 replay 与 Role 对齐。")
    lines.append("- 验收：A 的 train G share 与 Stage 1 G role 最大绝对差 `1.66e-7`；B replay G `5.13e-7`。")
    lines.append("")
    lines.append("## 6. Stage 5 A：梯度优势从哪里来\n")
    lines.append("- Q/K 的 top-1 G share 到 `0.9–0.99`，effective heads 接近 1；O 始终约 6 个有效 Head（top share 约 0.2）；V 介于中间。")
    lines.append("- Q/K/V 的 `g_share` 与 `dY_h` 份额的 Spearman 约 `0.83/0.83/0.64`；O 与 `A_h` 输入份额约 `0.92`。")
    lines.append("- seed0 node 0 的 Q/K top share 约 `0.20–0.23`；seed1 约 `0.37–0.43`。")
    lines.append("")
    lines.append(f"![A gradient concentration](figures/{figs['A_top'].name})")
    lines.append(f"![A gradient source](figures/{figs['A_source'].name})")
    lines.append("")
    lines.append("### 6.1 A-2：ΔY 的来源分解\n")
    lines.append("- 13,392 行；`dQ/dK/dV` 与 c_attn 梯度对照 `rel≈1e-6`，`cos≈1`。")
    lines.append("- `dY_share` 与 `dA_share`：Q `0.75`、K `0.77`、V `0.80`；")
    lines.append("- `dY_share` 与 attention entropy：Q `−0.26`、K `−0.21`、V `−0.17`；")
    lines.append("- seed0 784 L3：H2 的 dA share 仅约 `0.25`，但 Q/K dY share 达 `0.75/0.82`；attention 更聚焦。")
    lines.append("- A 的 `g_share` 与 A-2 `dy_share`：Q `0.83`、K `0.77`、V `0.60`。")
    lines.append("")
    lines.append("结论：**Q/K 的 Head 集中是上游 dA 经过 attention 聚焦/score 几何放大，而不是简单按 dA 比例分配。**")
    lines.append("")
    lines.append(f"![A2 source correlations](figures/{figs['A2_corr'].name})")
    lines.append(f"![A2 event](figures/{figs['A2_event'].name})")
    lines.append(f"![A2 seed difference](figures/{figs['A2_seed'].name})")
    lines.append("")
    lines.append("## 7. Stage 5 B：局部损失景观\n")
    lines.append("- probe mean κ：O `0.039` > V `0.0058` > K `0.00027` ≈ Q `0.00020`；O/V 负曲率 0%，Q/K 28%/36%。")
    lines.append("- median cross ρ：O-O `0.352`、O-V `0.371`、V-V `0.364`；Q/K 约 `0.02–0.09`。")
    lines.append("- probe vs train Pearson `0.940`；曲率不是 batch 噪声。")
    lines.append("- 实际 ΔW 方向的 κ 比块内 top Ritz 低约 2–3 个数量级。")
    lines.append("")
    lines.append("结论：**O/V 有正曲率和强耦合；Q/K 近零/不定、弱耦合。曲率描述类型级几何，但不直接决定 Head。**")
    lines.append("")
    lines.append(f"![B curvature](figures/{figs['B_curv'].name})")
    lines.append(f"![B coupling](figures/{figs['B_couple'].name})")
    lines.append("")
    lines.append("## 8. Stage 5 C：Muon 动力学与损失\n")
    lines.append("- `||U||/||G||`：O `53×`、V `145×`、Q `1084×`、K `1242×`；`cos(G,U)` 仅 `0.15–0.24`。")
    lines.append("- G→U Head 份额相关：Q/K `0.90`、V `0.85`、O `0.39`：NS 保留 Q/K/V，重排 O。")
    lines.append("- median ΔL：O `-1.50e-4`、V `-5.6e-5`、QKV `-7.2e-5`、Q `-6.4e-6`、K `-9.8e-6`；O/V 主导。")
    lines.append("- Taylor vs actual：Pearson `0.999997`，中位相对残差 `0.94%`。")
    lines.append("")
    lines.append(f"![C Taylor](figures/{figs['C_taylor'].name})")
    lines.append(f"![C loss](figures/{figs['C_loss'].name})")
    lines.append(f"![C NS](figures/{figs['C_ns'].name})")
    lines.append("")
    lines.append("### 8.1 C-2：QK–VO 交互是负结果\n")
    lines.append("- 8 行（784/1312 × L0/L3 × seed0/1）；`I_taylor=I_H`（差 `6e-14`）。")
    lines.append("- max |I_H| `2.5e-7`；max |I_actual| `9.5e-7`；`I_actual_norm` 最大 `0.0074`。")
    lines.append("- 没有稳定的负协同；`I_actual` 4/8 为正，高阶残差常常大于二阶项。")
    lines.append("")
    lines.append("结论：**Q/K 的梯度集中不能用事件点的 QK–VO 局部损失协同解释。**")
    lines.append("")
    lines.append(f"![C2 interaction](figures/{figs['C2'].name})")
    lines.append("")
    lines.append("## 9. Stage 5 D：跨 Seed 与时间顺序\n")
    lines.append("- seed0 L2/L3 最终 H2，L4 H0，L0 Q/K H5；seed1 L4 H4，L3 H2，L2 H5，L0 Q/K/V H3。")
    lines.append("- 但类型级模式一致：Q/K 集中、O 不集中；O/V 曲率序、耦合和 NS 规则在两边一致。")
    lines.append("- D-2 全节点 L3：H2 晚期收敛稳健，但 top-head 切换频繁；seed0 出现 A 早于 B 到 H2，seed1 1312 则是 B/C 先到 H2、A 仍 H3。")
    lines.append("- D-3 连续指标：A_G vs B_abs `0.77/0.81`，vs B_signed `0.59/0.49`，vs C_U `0.63/0.74`。")
    lines.append("")
    lines.append("结论：**具体 Head 身份 seed-dependent；类型/层级模式 seed-invariant；没有普遍“曲率/NS 领先梯度”。**")
    lines.append("")
    lines.append(f"![D timing](figures/{figs['D_timing'].name})")
    lines.append(f"![D coupling](figures/{figs['D_couple'].name})")
    lines.append(f"![D3 top share](figures/{figs['D3'].name})")
    lines.append("")
    lines.append("## 10. 综合机制链\n")
    lines.append("```text")
    lines.append("初始表示 / 反向信号差异")
    lines.append("  → dA + attention 聚焦 / score 几何")
    lines.append("  → Q/K 的 ΔY 与 G Head 集中")
    lines.append("  → Muon/NS 保留 Q/K/V 排序，重排 O")
    lines.append("  → ΔW 与 Head 级损失（主要由 V/O 承担）")
    lines.append("  → 曲率/耦合描述局部几何")
    lines.append("  → A/B/C 在 L3 晚期共同收敛到 H2")
    lines.append("```")
    lines.append("")
    lines.append("更准确的分工是：")
    lines.append("")
    lines.append("- **选择/排序**：主要由早期 G/D 和反向信号结构决定；")
    lines.append("- **幅度/组间分配**：主要由 Muon/NS 决定；")
    lines.append("- **局部几何**：曲率/耦合描述不同矩阵类型的优化性质，但没有稳定的具体 Head 预测能力；")
    lines.append("- **损失贡献**：由 V/O 子空间主导，Q/K 的梯度集中不等于 Q/K 承担主要损失变化。")
    lines.append("")
    lines.append("## 11. 证据强度\n")
    lines.append("| 问题 | 当前认识 | 证据 |")
    lines.append("|---|---|---|")
    lines.append("| 结构是否形成 | 是，早期形成并长期保持 | Stage 1 |")
    lines.append("| 选择出现在哪一步 | G/D 先于 U/ΔW | Stage 2/3 |")
    lines.append("| Q/K 梯度来源 | 反向信号，受 attention 聚焦放大 | Stage 3/5 A-2 |")
    lines.append("| O 的梯度来源 | 输入 A_h 表示 | Stage 5 A |")
    lines.append("| Muon 如何处理 | 保留 Q/K/V 排序，重排 O | Stage 2/5 C |")
    lines.append("| 类型级曲率 | O/V 正曲率强耦合；Q/K 近零/弱 | Stage 5 B |")
    lines.append("| 具体 Head | seed-dependent | Stage 1/5 D |")
    lines.append("| QK–VO 局部协同 | 未发现 | C-2 |")
    lines.append("| 曲率 lead-lag | 无普遍规律 | D-2/D-3 |")
    lines.append("")
    lines.append("## 12. 限制\n")
    lines.append("- 只有 2 个 seed；不能做总体统计推断。")
    lines.append("- Stage 1–5 的 loss 口径不同：Stage 1–3 是 8192 步全轨迹；Stage 5 到 1536，且 C 只做选定 Head 子空间/L_probe。")
    lines.append("- A-2 是一阶前向/反向，不做因果干预；attention 分解已验证但仍是相关性。")
    lines.append("- C-2 只在 784/1312、L0/L3、L_probe；不能排除 L_train/其他层/多步反馈。")
    lines.append("- D-2 的 B 曲率只有 8 个节点；top-head 切换频繁。")
    lines.append("")
    lines.append("## 13. 下一步任务\n")
    lines.append("### 已完成（本轮）")
    lines.append("- Stage 5 A/B/C/D；A-2；C-2；D-2；D-3。")
    lines.append("")
    lines.append("### 优先 1：A-3，解释 A-2 的 seed 差异")
    lines.append("A-2 已显示：L3 的 `dy_share` 跨 seed 排名相关仅 Q `0.14`、K `0.09`、V `-0.03`；"
                 "node 784 两边 Q/K/V top 都是 H2，但 entropy seed0 `4.31` vs seed1 `4.39`；"
                 "node 1312 seed0 H2 强、entropy `2.92`，seed1 H3、entropy `3.34`。")
    lines.append("下一步把这一差异拆成：`dA_share`、attention entropy、score 范数、Q/K/V 范数四部分；"
                 "目标是回答 seed1 为什么在 1312 仍停在 H3。")
    lines.append("")
    lines.append("### 优先 2：Seed 2/3 独立验证")
    lines.append("在明确 A-3 预期后，新增 2 个 seed 到 1536；预注册 type-level 指标：")
    lines.append("- Q/K top share/effective heads；")
    lines.append("- O/V κ 类型序；")
    lines.append("- O-O/O-V/V-V 耦合；")
    lines.append("- G→U share correlation；")
    lines.append("- L3 是否再次 H2 收敛。")
    lines.append("")
    lines.append("### 优先 3：低精度研究")
    lines.append("从 FP32 结论出发，检查：")
    lines.append("1. 量化是否改变反向信号已经形成的 Head 梯度分布；")
    lines.append("2. 量化误差经动量累积后是否改变 Head 相对更新优势；")
    lines.append("3. NS 是否放大/削弱不同矩阵类型的量化误差；")
    lines.append("4. 低精度是否改变 seed-dependent 的 Head 集中路径。")
    lines.append("")
    lines.append("### 不建议现在做")
    lines.append("- 不再为了 lead-lag 补 B-L3 Hessian；")
    lines.append("- 不再沿 QK–VO 局部协同追加事件；")
    lines.append("- 不把 A-2 的 attention 相关性当成因果。")
    lines.append("")
    lines.append("## 14. 产物索引\n")
    lines.append("- Stage 1：`results/structural_emergence_analysis/`")
    lines.append("- Stage 2：`results/muon_stage2_head_selective/`")
    lines.append("- Stage 3：`results/stage3_gradient_coupling/`")
    lines.append("- Stage 4/5：`results/stage5/`")
    lines.append("- Master / A-2 / C-2 / D-2 / D-3：`results/stage5/` 下各子目录")
    lines.append("")
    md = OUT / "STAGE1_5_OVERALL_REPORT.zh-CN.md"
    md.write_text("\n".join(lines) + "\n")

    pdf_path = OUT / "stage1_5_overall_report.pdf"
    with PdfPages(pdf_path) as pdf:
        text_page(pdf, "From 8192-update two-seed training to Stage 5", [
            "30M LLaMA / SlimPajama 8x / FP32 Reference Muon / 8192 updates / seed0,1",
            "",
            "Question: do over-parameterized heads/layers provide candidate paths,",
            "and does training concentrate contributions into a few paths?",
            "",
            "Answer: yes, but the mechanism is not curvature-first.",
            "Gradient/backward structure selects heads; Muon/NS distributes updates;",
            "curvature describes local geometry, not a stable lead-lag selector.",
        ])
        text_page(pdf, "Evidence map", [
            "Stage 1: structure exists and persists; Q/K strong, V/O weak.",
            "Stage 2: selection appears first in G/D; NS mainly redistributes V vs Q/K.",
            "Stage 3: p(G) ~ p(dY); V dominates fused QKV Gram; early G predicts late U.",
            "Stage 4: 31 pre-update checkpoints, 20 QKV/O role nodes.",
            "Stage 5 A: Q/K gradient concentration; source is dY.",
            "Stage 5 A-2: dA + attention entropy/score geometry amplifies Q/K dY.",
            "Stage 5 B: O/V positive curvature/coupling, Q/K weak/indefinite.",
            "Stage 5 C: NS preserves Q/K/V ordering, reallocates O; V/O dominate loss.",
            "Stage 5 D: specific heads seed-dependent, type-level pattern invariant.",
            "C-2: no strong QK-VO local synergy. D-2/D-3: no universal lead-lag.",
        ])
        for key in ("S1_concentration", "S1_similarity", "S2_fused", "S2_gamma",
                    "S3_gradient", "S3_gram", "A_top", "A_source", "A2_corr",
                    "A2_event", "A2_seed", "B_curv", "B_couple", "C_taylor",
                    "C_loss", "C_ns", "D_timing", "D_couple", "C2", "D3"):
            p = figs[key]
            image_page(pdf, p, p.stem.replace("_", " "))
        text_page(pdf, "Integrated mechanism", [
            "Initialization / early representation",
            "  -> dA and attention concentration/score geometry",
            "  -> Q/K dY and G head advantage",
            "  -> Muon/NS preserves Q/K/V, reallocates O",
            "  -> head-level loss dominated by V/O",
            "  -> curvature describes local geometry",
            "  -> A/B/C converge to H2 at L3 late",
            "",
            "Specific head identity is seed-dependent.",
            "Type-level concentration, curvature order and NS rules are seed-invariant.",
            "No strong QK-VO synergy; no universal curvature lead-lag.",
        ])
        text_page(pdf, "Limitations and next", [
            "2 seeds; Stage 5 only to 1536; Stage 1-3 to 8192.",
            "A/B/C use different loss definitions; mechanism alignment, not identity.",
            "A-2 is first-order, not causal.",
            "C-2 only 784/1312, L0/L3, L_probe.",
            "",
            "Next: A-3 seed-difference decomposition (dA/entropy/score/QKV norms).",
            "Then seed 2/3 independent validation.",
            "Then low-precision state representation.",
            "Do not add B-L3 Hessian for lead-lag.",
        ])
    summary = {"report": str(md), "pdf": str(pdf_path), "figures": len(figs)}
    (OUT / "overall_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
