# Q1 实验运行说明（INT8 默认）

## 目的

用 FP32 作为全精度参考，数值模拟 INT8 / BF16（可选 FP8）表示，回答：

- Q1a：per-tensor scale 因子用 BF16 / FP16 存，是否影响 `U = Φ_K(D)`？
- Q1b：`D` 本身量化到 INT8 / BF16，`U` 的基线误差多大？
- Q1c：先乘全局 scale `c` 再量化，结果是否与 Q1b 一致？

只量化状态表示；NS 计算始终是 FP32，和 `ReferenceMuon` 一致。

## 数据

- seed0、seed1；
- snapshot：128、1024、4096、8192；
- 层：L0–L5；
- 矩阵：fused QKV（1152×384）和 O（384×384）；
- `D` 直接取 snapshot 里的 `D` role；
- 共 `2 × 4 × 6 × 2 = 96` 个矩阵。

## 默认格式：INT8 per-tensor

- symmetric INT8，zero-point = 0；
- per-tensor absmax scale：`s = max(|D|)/127`；
- round-to-nearest；
- scale 因子用 FP32 / BF16 / FP16 存；
- 本实验不做 per-block/per-head；那是 Q2/Q3 的粒度问题。

## 变体

| variant | 含义 |
|---|---|
| `value_bf16` | D → BF16（16-bit 参考），NS |
| `value_int8` | D → per-tensor INT8（scale FP32），NS |
| `int8_scale_bf16` | 固定 INT8 codes，scale 用 BF16 存 |
| `int8_scale_fp16` | 固定 INT8 codes，scale 用 FP16 存 |
| `value_int8_c{0.25,0.5,1,2,4}` | 先乘 c，再 per-tensor INT8 量化，NS |
| `value_fp8_e4m3` | 可选：per-tensor FP8 e4m3 |
| `fp8_scale_*` | 可选：FP8 codes + BF16/FP8 scale |
| `value_fp8_c*` | 可选：先乘 c 再 FP8 量化 |

`*_scale_*` 和 `value_*_c*` 的 `rel_U_vs_compare` 是相对同一批 codes 的 FP32-scale 版本，用来隔离“scale 因子精度”的影响。

## 指标

- `rel_U = ‖U_q − U‖_F / ‖U‖_F`
- `cos`
- `R_norm`
- `share_L1`
- `top1_same`
- `rel_U_vs_compare` / `cos_vs_compare`（仅 scale/c 变体）
- `underflow`：scale 因子低于格式最小 subnormal
- `clamp_frac` / `zero_frac`

## 命令

```bash
cd /mnt/d/muon_research/analysis/lowp_first_batch

# 默认：INT8；冒烟测试 seed0/u128/L0
/home/sienna/.venvs/muon/bin/python q1_global_scale.py --smoke

# 全量 INT8
/home/sienna/.venvs/muon/bin/python q1_global_scale.py

# 可选：FP8 对照
/home/sienna/.venvs/muon/bin/python q1_global_scale.py --format fp8

# 两种都跑
/home/sienna/.venvs/muon/bin/python q1_global_scale.py --format both

# 汇总
/home/sienna/.venvs/muon/bin/python summarize_q1.py \
    /mnt/d/muon_research/results/lowp_first_batch/q1_global_scale/q1_metrics.csv
```

## 输出

- 默认：`results/lowp_first_batch/q1_global_scale/q1_metrics.csv`
- FP8：`q1_metrics_fp8.csv`
- both：`q1_metrics_both.csv`
- 汇总：`summary.csv`
- 冒烟：`smoke.csv`

## 冒烟测试结果（seed0 u128 L0，fused QKV）

| 变体 | `rel_U` | `cos` | `share_L1` | `top1_same` |
|---|---|---|---|---|
| BF16 value | ~0.0186 | 0.9998 | 0.00015 | 1 |
| INT8 per-tensor value | ~0.324 | 0.949 | 0.108 | 1 |
| FP8 e4m3 per-tensor value | ~0.285 | 0.960 | 0.0146 | 0 |

- **per-tensor INT8 不比 per-tensor FP8 好**：INT8 的 `rel_U` 更大、`share_L1` 更大。原因是 D 是重尾的，per-tensor absmax 被最大元素主导，小值拿不到足够的量化级；FP8 的 exponent 对跨量级更友好。
- 但 INT8 的 `top1_same=1`，FP8 是 0：全局 `rel_U` 和 top-1 不一致，说明必须同时看 `share_L1` 和 `top1_same`。
- scale 因子精度（BF16/FP16）相对同一批 INT8 codes 只增加约 `4e-6 – 1e-5` 的 `rel_U`，全局 scale 基本被 NS 消掉。
- `value_int8_c*` 的 `rel_U` 在不同 `c` 下基本一致。

## 全量结果（96 个矩阵）

| 变体 | matrix | `rel_U` median | `cos` median | `share_L1` median | `top1_same` mean |
|---|---|---|---|---|---|
| BF16 | fused QKV | 0.0196 | 0.9998 | 0.0002 | 0.958 |
| BF16 | O | 0.0221 | 0.9998 | 0.0002 | 1.000 |
| INT8 per-tensor | fused QKV | 0.5423 | 0.8627 | 0.1970 | 0.958 |
| INT8 per-tensor | O | 0.2613 | 0.9665 | 0.0060 | 0.875 |
| INT8 scale BF16/FP16 | — | 与 `value_int8` 相同 | — | — | — |

scale 因子精度（相对同一批 INT8 codes）：`rel_U_vs_compare` 中位数 **6.5e-6**，最大 **4.6e-5**，无 underflow。Q1c 的 `rel_U` 在不同 `c` 下完全一致。

**per-group 拆分（fused QKV，per-tensor INT8）：**

| 组 | `rel_U` median | mean | p10 | p90 |
|---|---|---|---|---|
| Q | **1.016** | 1.206 | 0.634 | 2.135 |
| K | **1.083** | 1.227 | 0.668 | 2.165 |
| V | **0.363** | 0.368 | 0.220 | 0.505 |

per-tensor scale 被 V 组主导，Q/K 拿不到足够的量化级，更新几乎被破坏。这是 Q2 上 per-head/per-group scale 的直接动机；不代表 INT8 本身不行。

## 注意

- BF16 / INT8 / FP8 都是 `torch` dtype / round-to-nearest 的数值模拟，不是低精度 kernel。
- FP8 固定用 e4m3；e5m2 只用于 scale 因子的可表示性对照。
- 本实验不涉及模型前向、loss、G 重算；是纯矩阵级实验。
- 不做 FP64；FP32 就是参考。
