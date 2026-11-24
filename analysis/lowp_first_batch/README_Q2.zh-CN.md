# Q2 实验运行说明（BF16 自然误差经过 NS）

## 目的

不引入人工扰动，只分析 BF16 的**自然误差**如何经过实际 5 步 NS：

- 参考：`D`（FP32），`U = NS(D)`；
- 低精度：`D_bf16 = BF16(D)`，`U_bf16 = NS(D_bf16)`；
- 误差：`E_D = D_bf16 − D`，`E_U = U_bf16 − U`。

把 `E_D` / `E_U` 分解为：

1. **全局 scale**：`c = <D_bf16, D>/||D||²`，`E_global = (c−1)D`；
2. **Head 相对 scale**：`c_h = <D_bf16,h, D_h>/||D_h||²`，`E_head = Σ_h (c_h−c)D_h`；
3. **方向误差**：`E_dir = E_D − E_global − E_head`。

## 数据

- seed0、seed1；
- snapshot：128、1024、4096、8192；
- 层：L0–L5；
- 矩阵：fused QKV（1152×384）和 O（384×384）；
- `D` 取 snapshot 的 `D` role；
- 共 96 个矩阵。

## 指标

- `rel_D_k = ||E_D,k|| / ||D||`；
- `rel_U_k = ||E_U,k|| / ||U||`；
- `A_k = rel_U_k / rel_D_k`；
- `share_D_k = ||E_D,k|| / ||E_D||`，`share_U_k = ||E_U,k|| / ||E_U||`；
- `share_L1`、`top1_same`；
- 分 Q/K/V/O，以及 fused QKV 的 whole-matrix `all`。

`A_reliable = 1` 仅在 `share_D ≥ 0.01` 且 `rel_D > 1e-8` 时成立；对可忽略的分量，`A` 不稳定，不要用它下结论。

## 命令

```bash
cd /mnt/d/muon_research/analysis/lowp_first_batch

# 冒烟：seed0/u128/L0
/home/sienna/.venvs/muon/bin/python q2_bf16_error.py --smoke

# 全量
/home/sienna/.venvs/muon/bin/python q2_bf16_error.py

# 汇总
/home/sienna/.venvs/muon/bin/python summarize_q2.py \
    /mnt/d/muon_research/results/lowp_first_batch/q2_bf16_error/q2_metrics.csv
```

## 输出

- `results/lowp_first_batch/q2_bf16_error/q2_metrics.csv`
- `results/lowp_first_batch/q2_bf16_error/summary.csv`
- 冒烟：`smoke.csv`、`smoke_summary.csv`

## 冒烟结果（seed0 / u128 / L0）

以 fused QKV 的 `all` 为例：

| 分量 | `rel_D` | `rel_U` | `A` | `share_D` | `share_U` |
|---|---|---|---|---|---|
| global | ~2e-6 | ~1e-4 | 不稳定 | 0.002 | 0.006 |
| head | ~2.5e-5 | ~1e-4 | ~10 | 0.004 | 0.004 |
| direction | **1.7e-3** | **1.9e-2** | **11.2** | **0.999** | **0.999** |

- BF16 的自然误差几乎完全是**方向误差**：global/head 分量 <2%，direction >99%。
- NS 对方向误差的相对放大 `A ≈ 7–12`（Q 6.97、K 7.47、V 11.98、O 12.25）。
- global/head 的 `A` 看起来很大，但 `rel_D` 只有 1e-6–1e-5，`A_reliable=0`，不要据此说“scale 被放大”。

## 注意

- 本实验不做合成扰动、不做 per-head/per-block INT8、不测 loss。
- BF16 的自然误差是方向主导的；它不能用来回答“Head 相对 scale 会不会被放大”，因为 BF16 几乎不产生 head-scale 误差。那个问题要等 per-head INT8 或方法阶段的扰动实验。
- NS 计算保持 FP32。
