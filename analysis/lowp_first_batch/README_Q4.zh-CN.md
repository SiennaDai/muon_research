# Q4 实验运行说明（type × spectral-band 的 NS 传递）

## 目的

Q2 给了 `A(type)`，Q3 给了 `A(band)`；Q4 补 `A(type, band)`，回答：

- Tail 放大是不是所有类型都一样？
- V/O 的瓶颈是 Tail 误差更大，还是 Tail 放大更强？
- 类型级精度分配有没有依据？

## 关键设计：per-type SVD

Q4 的 band 用**每个类型自己的 SVD**（Q/K/V/O 各 384×384），不是全局 fused SVD。

原因：如果用全局 fused SVD 分带、再限制到类型行块，band 在类型块内不再正交，`share_U` 会 >1（冒烟里 Q/K 出现过 1.18–1.26）。per-type SVD 保证 band 在类型内正交，share 才有意义。

NS 传递仍由 `A_rel = (||E_U,g,b||/||U_g,b||) / (||E_D,g,b||/||D_g,b||)` 度量；band 基是类型内的。

## 变体

| variant | 说明 |
|---|---|
| `bf16` | Q1 基线 |
| `bf16_head_bf16` | BF16 + per-head BF16 scale |
| `int8_head_fp32` | INT8 + per-head FP32 scale |

## 指标

- `A_total = (||E_U,g||/||U_g||) / (||E_D,g||/||D_g||)`；
- `A_rel,g,b = (||E_U,g,b||/||U_g,b||) / (||E_D,g,b||/||D_g,b||)`；
- `share_D`、`share_U`：band 在该类型 D/U 误差里的份额；
- `share_L1`、`top1_same`：类型内 6 个 head 的结构 guardrail。

## 命令

```bash
cd /mnt/d/muon_research/analysis/lowp_first_batch

# 冒烟
/home/sienna/.venvs/muon/bin/python q4_type_band.py --smoke

# 全量
/home/sienna/.venvs/muon/bin/python q4_type_band.py

# 汇总
/home/sienna/.venvs/muon/bin/python summarize_q4.py \
    /mnt/d/muon_research/results/lowp_first_batch/q4_type_band/q4_metrics.csv
```

## 冒烟结果（seed0 / u128 / L0）

| variant | type | `A_total` | `A_rel_tail` | `share_U_tail` | tail energy |
|---|---|---|---|---|---|
| bf16 | Q | 6.97 | 2.56 | 0.972 | 0.122 |
| bf16 | K | 7.47 | 2.57 | 0.985 | 0.114 |
| bf16 | V | 11.98 | 4.04 | 1.000 | 0.103 |
| bf16 | O | 12.25 | 4.16 | 1.000 | 0.104 |
| bf16_head_bf16 | Q | 4.86 | 2.49 | 0.971 | 0.122 |
| bf16_head_bf16 | K | 4.91 | 2.42 | 0.985 | 0.114 |
| bf16_head_bf16 | V | 9.70 | 3.95 | 1.000 | 0.103 |
| bf16_head_bf16 | O | 8.65 | 3.97 | 1.000 | 0.104 |
| int8_head_fp32 | Q | 6.32 | 2.30 | 0.979 | 0.122 |
| int8_head_fp32 | K | 6.89 | 2.36 | 0.989 | 0.114 |
| int8_head_fp32 | V | 11.98 | 4.03 | 1.000 | 0.103 |
| int8_head_fp32 | O | 12.14 | 4.10 | 1.000 | 0.104 |

## 初步观察

1. **每个类型都是 Tail 主导**：`share_U_tail` 0.97–1.00；Top/Middle 几乎不贡献。
2. **Tail 放大有类型差异**：`A_rel_tail` Q/K ≈ 2.5，V/O ≈ 4.0；V/O 的 Tail 相对放大更强。
3. **A_total**：Q/K ≈ 7–7.5，V/O ≈ 12；差距比 Tail 的 A_rel 更大，因为还叠加了总误差结构。
4. **INT8 与 BF16 的 A_rel 接近**：INT8 的残差主要是 `rel_D_tail` 更大（约 10×），不是 NS 传递规律变了。
5. **BF16 + BF16 scale**：`A_total` 反而更低，因为 scale 误差让 `rel_D_total` 变大，`rel_U_total` 变化小；Tail share 仍 ~1，scale 误差被衰减。

## 全量结果（96 个矩阵）

### `A_total` / `A_rel_tail` / `share_U_tail`

| variant | type | `A_total` | `A_rel_tail` | `share_U_tail` | `rel_D_total` | `rel_U_total` | top1 |
|---|---|---|---|---|---|---|---|
| bf16 | Q | 6.97 | 2.57 | 0.935 | 1.66e-3 | 1.16e-2 | 1.000 |
| bf16 | K | 7.26 | 2.61 | 0.948 | 1.66e-3 | 1.20e-2 | 1.000 |
| bf16 | V | 12.25 | 4.22 | 1.000 | 1.66e-3 | 2.03e-2 | 0.958 |
| bf16 | O | 13.29 | 4.66 | 1.000 | 1.66e-3 | 2.21e-2 | 1.000 |
| bf16_head_bf16 | Q | 4.97 | 2.44 | 0.934 | 2.20e-3 | 1.17e-2 | 1.000 |
| bf16_head_bf16 | K | 5.64 | 2.52 | 0.949 | 2.26e-3 | 1.20e-2 | 0.979 |
| bf16_head_bf16 | V | 9.43 | 4.07 | 1.000 | 2.26e-3 | 2.03e-2 | 0.958 |
| bf16_head_bf16 | O | 9.86 | 4.52 | 1.000 | 2.29e-3 | 2.20e-2 | 0.938 |
| int8_head_fp32 | Q | 6.45 | 2.45 | 0.949 | 2.25e-2 | 1.43e-1 | 0.958 |
| int8_head_fp32 | K | 6.92 | 2.47 | 0.959 | 2.26e-2 | 1.54e-1 | 0.938 |
| int8_head_fp32 | V | 12.34 | 4.14 | 1.000 | 1.88e-2 | 2.35e-1 | 0.812 |
| int8_head_fp32 | O | 13.34 | 4.62 | 1.000 | 1.57e-2 | 2.07e-1 | 0.792 |

### 全量结论

1. **Tail 主导是普遍的**：所有类型、所有变体的 `share_U_tail` 都在 0.93–1.00；Top/Middle 几乎不贡献。
2. **Tail 放大有稳定的类型差异**：`A_rel_tail` Q/K ≈ 2.4–2.6，V/O ≈ 4.1–4.7。V/O 的 Tail 相对误差被 NS 放大得更强。
3. **`A_total` 也有类型差异**：Q/K ≈ 5–7.3，V/O ≈ 9.4–13.3；差距比 Tail 的 A_rel 更大，说明总误差结构里还有类型因素。
4. **BF16 + BF16 scale**：`A_total` / `A_rel_tail` 反而略低，因为 scale 误差让 `rel_D_total` 变大、`rel_U_total` 变化小；Tail share 仍 ~1。scale 误差是二阶项，被衰减。
5. **INT8 与 BF16 的 A_rel 接近**：Q/K 2.45–2.47、V/O 4.14–4.62；INT8 的残差主要是 `rel_D_total` / `rel_U_total` 大约 13×，不是 NS 传递规律变了。
6. **结构 guardrail**：INT8 的 `share_L1` 0.006–0.007，top1 Q/K 0.94–0.96、V 0.812、O 0.792；V/O 的结构受量化影响更大。

## 注意

- 冒烟只覆盖 seed0/u128/L0；全量以 CSV 为准。
- Q4 用 per-type SVD；Q3 用全局 SVD，两者的 band 定义不同，数字不能直接逐项对比。
- 不做合成 matched-error 扰动、不测 loss、不做累积。
- BF16/INT8 都是数值模拟，不是低精度 kernel。
