# Q3 实验运行说明（自然误差的谱带分布）

## 目的

在 Q2 的变体基础上，按参考 `D` 的奇异值把误差分成 Top / Middle / Tail 三个谱带，回答：

- 低精度误差主要落在哪个谱带？
- NS 对哪个谱带的误差放大/衰减最多？
- Tail（小奇异方向）是不是 Muon 的脆弱点？

## 谱带定义

- 对参考 `D` 做 SVD：`D = U S Vᵀ`；
- 按累计平方奇异值能量分带：
  - **Top**：累计能量 0–50%；
  - **Middle**：50–90%；
  - **Tail**：90–100%；
- 记录每带的 `rank_count` 和 `energy_share`。

误差投影：

```
E_b = U_b U_bᵀ E V_b V_bᵀ
```

`E_D = D_var − D`，`E_U = U_var − U`；两边的投影都用参考 `D` 的奇异向量。

## 变体

| variant | 说明 |
|---|---|
| `bf16` | Q1 基线 |
| `bf16_head_fp32` | BF16 + per-head FP32 scale |
| `bf16_head_bf16` | BF16 + per-head BF16 scale |
| `int8_head_fp32` | INT8 + per-head FP32 scale |

## 指标

- `rel_D_b = ||E_D,b||/||D||`
- `rel_U_b = ||E_U,b||/||U||`
- `A_b = rel_U_b/rel_D_b`
- `share_D_b = ||E_D,b||/||E_D||`，`share_U_b = ||E_U,b||/||E_U||`
- `share_L1`、`top1_same`
- `A_reliable=1` 仅当 `share_D_b ≥ 0.01` 且 `rel_D_b > 1e-8`

## 命令

```bash
cd /mnt/d/muon_research/analysis/lowp_first_batch

# 冒烟
/home/sienna/.venvs/muon/bin/python q3_spectral_bands.py --smoke

# 全量
/home/sienna/.venvs/muon/bin/python q3_spectral_bands.py

# 汇总
/home/sienna/.venvs/muon/bin/python summarize_q3.py \
    /mnt/d/muon_research/results/lowp_first_batch/q3_spectral_bands/q3_metrics.csv
```

## 冒烟结果（seed0 / u128 / L0，fused QKV）

| variant | band | rank | energy | `rel_D` | `rel_U` | `A` | `share_D` | `share_U` |
|---|---|---|---|---|---|---|---|---|
| bf16 | top | 1 | 0.529 | 1e-5 | 0 | 0.054 | 0.005 | 0.000 |
| bf16 | middle | 10 | 0.366 | 5e-5 | 1e-5 | 0.289 | 0.028 | 0.001 |
| bf16 | **tail** | **373** | **0.105** | **1.38e-3** | **1.24e-2** | **9.03** | **0.831** | **0.670** |
| bf16_head_bf16 | top | 1 | 0.529 | 3.8e-4 | 0 | 0.002 | 0.183 | 0.000 |
| bf16_head_bf16 | middle | 10 | 0.366 | 3.4e-4 | 3e-5 | 0.079 | 0.165 | 0.001 |
| bf16_head_bf16 | **tail** | **373** | **0.105** | **1.42e-3** | **1.25e-2** | **8.84** | **0.687** | **0.673** |
| int8_head_fp32 | top | 1 | 0.529 | 3e-5 | 1e-5 | 0.294 | 0.002 | 0.000 |
| int8_head_fp32 | middle | 10 | 0.366 | 3.1e-4 | 1e-4 | 0.314 | 0.025 | 0.001 |
| int8_head_fp32 | **tail** | **373** | **0.105** | **1.02e-2** | **9.18e-2** | **9.04** | **0.829** | **0.666** |

## 初步观察

1. **误差几乎全部落在 Tail**：Tail 有 373 个方向、只占 10.5% 能量，却贡献 **67% 的 U 误差**；Top（1 个方向、53% 能量）对 U 误差几乎无贡献。
2. **NS 放大 Tail、衰减 Top/Middle**：Tail 的 `A ≈ 9`，Top/Middle 的 `A < 0.3`。
3. **BF16 + BF16 scale** 的 scale 误差主要落在 Top/Middle（`share_D` 0.18/0.17），但 U 误差仍由 Tail 主导（`share_U` 0.673）；Tail 的 `A` 8.84。
4. **INT8 per-head** 的误差同样 Tail 主导（`share_U` 0.666，`A≈9`）；它的 `rel_D_tail` 比 BF16 大得多（1e-2 vs 1.4e-3），这是 INT8 残差的主要来源。

## 全量结果（96 个矩阵）

### fused QKV

| variant | band | rank | energy | `rel_D` | `rel_U` | `A` | `share_D` | `share_U` |
|---|---|---|---|---|---|---|---|---|
| bf16 | top | 1 | 0.571 | 1.4e-5 | 7.8e-7 | 0.11 | 0.008 | 0.000 |
| bf16 | middle | 12 | 0.325 | 7.8e-5 | 3.6e-5 | 0.51 | 0.047 | 0.002 |
| bf16 | **tail** | **370** | **0.103** | **1.39e-3** | **1.49e-2** | **10.62** | **0.838** | **0.741** |
| bf16_head_bf16 | top | 1 | 0.571 | 4.3e-4 | 3.5e-6 | 0.00 | 0.195 | 0.000 |
| bf16_head_bf16 | middle | 12 | 0.325 | 4.3e-4 | 7.2e-5 | 0.20 | 0.195 | 0.003 |
| bf16_head_bf16 | **tail** | **370** | **0.103** | **1.46e-3** | **1.49e-2** | **10.13** | **0.629** | **0.741** |
| int8_head_fp32 | top | 1 | 0.571 | 4.2e-5 | 2.1e-5 | 0.27 | 0.002 | 0.000 |
| int8_head_fp32 | middle | 12 | 0.325 | 7.0e-4 | 3.1e-4 | 0.51 | 0.039 | 0.001 |
| int8_head_fp32 | **tail** | **370** | **0.103** | **1.56e-2** | **1.71e-1** | **10.82** | **0.837** | **0.726** |

### O

| variant | band | rank | energy | `rel_D` | `rel_U` | `A` | `share_D` | `share_U` |
|---|---|---|---|---|---|---|---|---|
| bf16 | tail | 369 | 0.103 | 1.57e-3 | 2.21e-2 | **14.19** | 0.950 | **1.000** |
| bf16_head_bf16 | tail | 369 | 0.103 | 1.65e-3 | 2.20e-2 | 13.76 | 0.700 | **1.000** |
| int8_head_fp32 | tail | 369 | 0.103 | 1.49e-2 | 2.07e-1 | **14.04** | 0.957 | **1.000** |

O 的 top/middle `share_U` 都接近 0；U 误差全部来自 tail。

## 全量结论

1. **Tail 几乎承载全部 U 误差**：fused QKV 的 tail 有约 370 个方向、只占 10.3% 能量，却贡献 **74% 的 U 误差**；O 的 tail 贡献 **100%**。
2. **NS 放大 Tail、衰减 Top/Middle**：fused `A_tail ≈ 10.6`，O `A_tail ≈ 14.2`；Top/Middle 的 `A` 都 <0.6。
3. **Top 方向几乎不受影响**：Top 只有 1 个方向、占 57% 能量，但 `share_U` 接近 0。
4. **BF16 + BF16 scale 的 scale 误差落在 Top/Middle**（`share_D` 各约 0.195），但 U 里仍由 Tail 主导（`share_U` 0.741）；scale 误差被衰减，是二阶项。
5. **INT8 per-head 的残差也在 Tail**：tail 的 `rel_D` 比 BF16 大一个数量级（1.56e-2 vs 1.39e-3），`share_U` 0.726；这是 INT8 的主要误差来源。
6. **设计启示**：低精度设计的关键不是保护大方向，而是保护 **Tail / 小奇异方向**；低秩压缩（丢掉 Tail）对 Muon 尤其危险，因为 Tail 正是 U 误差的主要载体。

## 带内分母的 `A_rel`

上面的 `A` 是**贡献比**（分母是总 `||D||`、`||U||`）。为了看“带内相对误差被 NS 放大多少”，又算了带内分母：

```
A_rel,b = (||E_U,b||/||U_b||) / (||E_D,b||/||D_b||)
```

全量（fused QKV）：

| variant | band | `A_contrib` | `A_rel` | `rel_D_rel` | `rel_U_rel` | `share_U` |
|---|---|---|---|---|---|---|
| bf16 | top | 0.11 | 0.99 | 1.8e-5 | 1.3e-5 | 0.000 |
| bf16 | middle | 0.51 | 1.30 | 1.4e-4 | 1.9e-4 | 0.002 |
| bf16 | **tail** | **10.62** | **3.46** | **4.3e-3** | **1.5e-2** | **0.741** |
| bf16_head_bf16 | top | 0.00 | 0.14 | 5.4e-4 | 6.5e-5 | 0.000 |
| bf16_head_bf16 | middle | 0.20 | 0.62 | 7.6e-4 | 4.1e-4 | 0.003 |
| bf16_head_bf16 | **tail** | **10.13** | **3.30** | **4.6e-3** | **1.5e-2** | **0.741** |
| int8_head_fp32 | top | 0.27 | 4.09 | 5.7e-5 | 3.0e-4 | 0.000 |
| int8_head_fp32 | middle | 0.51 | 1.31 | 1.2e-3 | 1.9e-3 | 0.001 |
| int8_head_fp32 | **tail** | **10.82** | **3.54** | **4.8e-2** | **1.7e-1** | **0.726** |

关键修正：

- `A_contrib,tail ≈ 10` 是**贡献比**，会被 Tail 的小信号放大；
- **带内相对误差放大** `A_rel,tail ≈ 3.3–3.5`，不是 10×；
- Top/Middle 的 `A_rel ≈ 1`（BF16）或被衰减（scale 误差）；
- Q3 的主结论“Tail 占 U 误差的 74%/100%”依据的是 `share_U`，不受 A 的定义影响。

## 注意

- 冒烟只覆盖 seed0/u128/L0；全量以 CSV 为准。
- Top 带只有 1 个方向，是能量分带的自然结果；不要把它理解成“Top 一定只有一个方向”。
- 不做合成扰动、不测 loss、不做累积。
- BF16/INT8 都是数值模拟，不是低精度 kernel。
