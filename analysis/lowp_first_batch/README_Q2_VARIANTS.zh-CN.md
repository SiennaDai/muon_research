# Q2a/b/c 实验运行说明（block-scaled BF16 / INT8）

## 目的

在 Q1 的 BF16 基线上，加入 per-head scale，分离三件事：

- **Q2a**：BF16 元素 + per-head FP32 scale——block 归一化本身的影响；
- **Q2b**：BF16 元素 + per-head BF16/FP8 scale——scale 因子精度的影响；
- **Q2c**：INT8 元素 + per-head FP32/BF16 scale——粗 mantissa + block scale 的联合影响。

所有表示都是数值模拟；NS 计算保持 FP32。

## 变体

| variant | element | scale | 说明 |
|---|---|---|---|
| `bf16` | BF16 | 无 | Q1 基线 |
| `bf16_head_fp32` | BF16 | per-head FP32 | Q2a |
| `bf16_head_bf16` | BF16 | per-head BF16 | Q2b |
| `bf16_head_fp8_e4m3` | BF16 | per-head FP8 e4m3 | Q2b |
| `int8_head_fp32` | INT8 | per-head FP32 | Q2c |
| `int8_head_bf16` | INT8 | per-head BF16 | Q2c |

per-head 块：fused QKV 的 18 个 64 行块；O 的 6 个 64 列块。

## 指标

- `E_D = D_var − D`，`E_U = U_var − U`；
- 分解为 global / head / direction；
- `rel_D`、`rel_U`、`A = rel_U/rel_D`、`share_D`、`share_U`；
- `share_L1`、`top1_same`；
- 分 Q/K/V/O 与 whole matrix；
- `A_reliable=1` 仅当 `share_D ≥ 0.01` 且 `rel_D > 1e-8`。

## 命令

```bash
cd /mnt/d/muon_research/analysis/lowp_first_batch

# 冒烟
/home/sienna/.venvs/muon/bin/python q2_scale_variants.py --smoke

# 全量
/home/sienna/.venvs/muon/bin/python q2_scale_variants.py

# 汇总
/home/sienna/.venvs/muon/bin/python summarize_q2_variants.py \
    /mnt/d/muon_research/results/lowp_first_batch/q2_scale_variants/q2_metrics.csv
```

## 冒烟结果（seed0 / u128 / L0，fused QKV `all`）

| variant | `rel_D_total` | `rel_U_total` | `share_L1` | `top1_same` | underflow heads |
|---|---|---|---|---|---|
| bf16 | 0.00166 | 0.0186 | 0.0001 | 1 | 0 |
| bf16_head_fp32 | 0.00165 | 0.0186 | 0.0001 | 1 | 0 |
| bf16_head_bf16 | 0.00206 | 0.0186 | 0.0007 | 1 | 0 |
| bf16_head_fp8_e4m3 | 0.2735 | **0.5151** | **0.5896** | **0** | **12** |
| int8_head_fp32 | 0.0122 | **0.1377** | 0.0032 | 1 | 0 |
| int8_head_bf16 | 0.0123 | **0.1377** | 0.0033 | 1 | 0 |

## 初步观察

1. **BF16 + FP32 scale ≈ BF16 基线**：block 归一化本身几乎不改变 U；BF16 的 exponent 已经处理了动态范围。
2. **BF16 + BF16 scale**：D 的误差里 global/head 分量变大（`share_D` 约 0.26 / 0.54），但 U 的误差仍由 direction 主导（`share_U` 约 0.999）；scale 分量的 `A < 1`，说明 NS 会**衰减** block-scale 误差，而不是放大它。
   - 所以“block scale 误差不会被 NS 抹掉”是对的，但它也不一定被放大；在本例里它被相对衰减了。
3. **BF16 + FP8 e4m3 scale 不可用**：fused QKV 有 12 个 head 的 scale underflow；`rel_U` 0.515，`share_L1` 0.59，top-1 改变。FP8 e4m3 不能直接存小 head 的 scale。
4. **INT8 per-head**：`rel_U` 从 Q1 per-tensor 的 0.542 降到 **0.1377**；top-1 保持，`share_L1` 0.0032。方向误差仍是主导，`A_dir ≈ 11.2`。
5. **INT8 + BF16 scale** 与 **INT8 + FP32 scale** 的 `rel_U` 相同（0.1377）：INT8 的 error 由元素 precision 主导，scale 存 BF16 足够。

## O 的冒烟结果

| variant | `rel_U_total` | `share_L1` | `top1_same` |
|---|---|---|---|
| bf16 | 0.0202 | 0.0001 | 1 |
| bf16_head_fp32 | 0.0203 | 0.0001 | 1 |
| bf16_head_bf16 | 0.0203 | 0.0003 | 1 |
| bf16_head_fp8_e4m3 | 0.0236 | 0.0043 | 1 |
| int8_head_fp32 | 0.1890 | 0.0028 | 0 |
| int8_head_bf16 | 0.1890 | 0.0028 | 0 |

O 的 FP8 scale 没有 underflow；INT8 per-head 的 `rel_U` 比 fused QKV 更大，top-1 改变。

## 全量结果（96 个矩阵）

### whole matrix

| variant | fused `rel_U` | fused `share_L1` | fused top1 | O `rel_U` | O `share_L1` | O top1 | underflow heads |
|---|---|---|---|---|---|---|---|
| bf16 | 0.0196 | 0.0002 | 0.958 | 0.0221 | 0.0002 | 1.000 | 0 |
| bf16_head_fp32 | 0.0197 | 0.0002 | 0.979 | 0.0220 | 0.0002 | 1.000 | 0 |
| bf16_head_bf16 | 0.0197 | 0.0009 | 0.958 | 0.0220 | 0.0007 | 0.938 | 0 |
| bf16_head_fp8_e4m3 | **0.5788** | **0.5854** | 0.583 | 0.0374 | 0.0162 | 0.750 | **12** |
| int8_head_fp32 | **0.2379** | 0.0114 | 0.812 | 0.2074 | 0.0061 | 0.792 | 0 |
| int8_head_bf16 | 0.2378 | 0.0116 | 0.812 | 0.2074 | 0.0061 | 0.771 | 0 |

### fused Q/K/V `rel_U`

| variant | Q | K | V |
|---|---|---|---|
| bf16 | 0.0116 | 0.0120 | 0.0203 |
| bf16_head_fp32 | 0.0116 | 0.0119 | 0.0202 |
| bf16_head_bf16 | 0.0117 | 0.0120 | 0.0203 |
| bf16_head_fp8_e4m3 | **1.0675** | **1.1793** | 0.2698 |
| int8_head_fp32 | 0.1434 | 0.1537 | **0.2346** |
| int8_head_bf16 | 0.1435 | 0.1537 | **0.2346** |

### 分量分解（fused `all`）

| variant | 分量 | `share_D` | `share_U` | `A` |
|---|---|---|---|---|
| bf16_head_bf16 | global | 0.262 | 0.007 | **0.233** |
| bf16_head_bf16 | head | 0.578 | 0.030 | **0.440** |
| bf16_head_bf16 | direction | 0.730 | **0.9995** | **11.93** |
| int8_head_fp32 | global | 0.002 | 0.064 | 不稳定 |
| int8_head_fp32 | head | 0.006 | 0.015 | ~12.96（仅 1 个可靠点） |
| int8_head_fp32 | direction | **1.000** | **0.998** | **12.09** |
| bf16_head_fp8_e4m3 | global | 0.126 | 0.084 | 1.78 |
| bf16_head_fp8_e4m3 | head | **0.992** | **0.609** | 1.33 |
| bf16_head_fp8_e4m3 | direction | 0.007 | 0.788 | 129（不可靠） |

## 全量结论

1. **BF16 + FP32 scale ≈ BF16 基线**：per-head block 归一化本身不改变 U。
2. **BF16 + BF16 scale**：D 里出现了明显的 global/head scale 误差（`share_D` 0.26/0.58），但 U 里仍由 direction 主导（`share_U` 0.9995）；scale 分量的 `A < 1`，即 NS **衰减**了 block-scale 误差。
   - “block scale 误差不会被 NS 抹掉”成立，但“会被放大”不成立；在本例里它是被相对衰减的。
3. **BF16 + FP8 e4m3 scale 不可用**：fused QKV 有 12 个 head underflow；`rel_U` 0.579，`share_L1` 0.585，top1 0.583。O 反而还行（0.037，top1 0.75）。
4. **INT8 per-head 修复了 Q/K 的动态范围**：Q/K 从 per-tensor 的 >1 降到 **0.143/0.154**；V 从 0.363 降到 **0.235**；fused `all` 从 0.542 降到 **0.238**。
   - 但距离 BF16 的 0.0196 仍差一个数量级；此时 V 成为最大残差组（0.235），不是 Q/K。
5. **INT8 的误差仍是 direction 主导**（`share_U` 0.998），`A_dir ≈ 12`；scale 分量很小。**要进一步提升，应该针对 direction 误差，而不是继续细化 scale。**
6. **INT8 的 scale 存 BF16 与存 FP32 没有差别**：INT8 的瓶颈是元素 precision，不是 scale 精度。
7. **类型差异**：direction 放大比 Q 6.43、K 6.90、V 12.29、O 12.25；V/O 的 direction 误差被放大得更多，是 INT8 per-head 的瓶颈。

## 注意

- 冒烟只覆盖 seed0/u128/L0；全量结论以全量 CSV 为准。
- 不做合成扰动、不测 loss、不做累积。
- BF16/INT8/FP8 都是数值模拟，不是低精度 kernel。
