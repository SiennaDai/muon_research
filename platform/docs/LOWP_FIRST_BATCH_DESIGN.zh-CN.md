# 低精度 Muon 第一批实验设计思路（单步，attention）

## 1. 范围

- 对象：attention 的 Q/K/V/O；MLP 暂不纳入。
- 阶段：单步量化。累积训练、M 的长期记忆、path selection 留到下一阶段。
- 数据：已有检查点与已存 role（`W_before, M_prev, G, D, U`）；本地只做矩阵级计算，需要模型前向时上 Kaggle。
- 指标：以 `platform/docs/评价指标.txt` 的层级为准。主指标是方向保真度与单步 loss；结构保真度是 guardrail；状态保真度、曲率等是诊断量。

## 2. 第一批要解决的核心不确定性

第一批不是找最好的低精度方案，而是**校准 NS 这个非线性算子**：

> NS 对不同类型的量化误差，是消除、保留还是放大？

单步链条是：

```
量化状态 → D_hat → U_hat = Φ_K(D_hat) → ΔW_hat → loss
```

NS 是链条上唯一的非线性、尺度不变、谱压平算子。它会把不同方向的误差区别对待：

- 全局 scale 误差可能被直接消掉；
- Head/block 间的相对 scale 误差可能被保留或放大；
- 小奇异值方向的方向误差会被 γ 放大。

因此“状态误差 → 更新误差”不是单调、均匀的映射。不先摸清这个映射，后面测“哪个状态敏感、QK/VO 谁更容忍、什么粒度更好”都会归因不清。

## 3. 四个问题

### Q1：全局 scale 误差会被 NS 消掉吗（sanity check）

- 预期：会。优化器在 NS 前做 Frobenius 归一化，`Φ_K(cD) = Φ_K(D)`（c>0）。
- 定位：实现核验 + 数值参照，不是研究重点。
- 精度方向：以 FP32 为全精度参考，**往下**用数值模拟的 BF16 / INT8 / FP8 表示；不用 FP64。
- 三个子问题：
  - Q1a：scale 因子本身用 BF16/FP16/FP8 存，值仍按该 scale 量化，NS 用 FP32——全局 scale 因子能不能粗存？
  - Q1b：值本身量化到 BF16/INT8/FP8，scale 用 FP32，NS 用 FP32——全局表示量化的基线误差。
  - Q1c：先乘全局 scale `c` 再量化，比较是否与 Q1b 一致——量化发生在归一化之前还是之后是否影响结论。
- 注意：本实验只量化**状态表示**，NS 计算保持 FP32；BF16/INT8/FP8 计算精度另开实验。
- Q1 实测：BF16 `rel_U ≈ 0.02`；per-tensor INT8 在 fused QKV 上 `rel_U = 0.542`，Q/K `>1`，V 0.363，O 0.261。per-tensor INT8 太粗，不适合作 NS 误差传递研究的基座；Q2 起用 BF16。

### Q2：BF16 自然误差经过 NS 的分解（研究问题）

Q1 的 side observation：per-tensor INT8 在 fused QKV 上 `rel_U = 0.542`，Q/K 的 `rel_U > 1`，V 约 0.363，O 约 0.261。原因是 per-tensor absmax scale 被 V 主导，Q/K 拿不到量化级。**per-tensor INT8 太粗，不适合作 NS 误差传递研究的基座**；本阶段改用 BF16 作为低精度基座，INT8 的粒度问题留到方法阶段。

Q2 不引入人工扰动，只分析 BF16 的**自然误差**如何经过 NS：

- 参考：`D`（FP32），`U = NS(D)`；
- 低精度：`D_bf16 = BF16(D)`，`U_bf16 = NS(D_bf16)`；
- 误差：`E_D = D_bf16 − D`，`E_U = U_bf16 − U`。

把 `E_D` 和 `E_U` 分解为三个分量：

1. **全局 scale**：`c = <D_bf16, D>_F / ||D||_F²`，`E_global = (c−1)D`；
2. **Head 相对 scale**：每个 Head `c_h = <D_bf16,h, D_h>_F / ||D_h||_F²`，`E_head = Σ_h (c_h−c)D_h`；
3. **方向误差**：`E_dir = E_D − E_global − E_head`。

对 `E_U` 做同样分解（用 `U` 的 Head 块和全局内积）。指标：

- `rel_D_k = ||E_D,k||_F / ||D||_F`；
- `rel_U_k = ||E_U,k||_F / ||U||_F`；
- `A_k = rel_U_k / rel_D_k`；
- `share_L1`、`top1_same`，分 Q/K/V/O。

判据：

- `E_head` 的 `A` 最大 → Head 相对 scale 是主要放大项；方法阶段优先 per-head scale；
- `E_dir` 的 `A` 最大 → 方向/格式精度是主因，粒度不是第一杠杆；
- `E_global` 的 `A` 接近 0 → 与 Q1 一致，全局 scale 被 NS 消掉。

不回答：合成 scale/direction 扰动、per-head/per-block INT8、单步 loss、M 累积。

### Q3：BF16 自然误差的谱带分布（研究问题）

- 用 `D` 的奇异向量把 `E_D` 和 `E_U` 投影到 Top/Middle/Tail；
- 每个带算 `rel_D_b`、`rel_U_b`、`A_b = rel_U_b / rel_D_b`；
- 判据：Tail 的 `A` 是否显著大于 Top；如果是，方向保真度必须分带，全局 cosine 不够；
- 不做合成谱带扰动；只分析 BF16 自然误差。

### Q4：BF16 自然误差的类型差异（研究问题）

- 对 Q/K/V/O 分别算 `rel_U`、`A`、`share_L1`、`top1_same`；
- 判据：QK 与 VO 的 `A` 是否不同；不同才支持类型级精度分配；
- 不做合成类型扰动；只分析 BF16 自然误差。

## 4. 方法论原则

1. **本阶段只分析自然低精度误差**：BF16 vs FP32，经过实际 5 步 NS。合成 scale/direction 扰动、per-head/per-block 粒度对比留到方法阶段。
2. **误差分量分解**：全局 scale / Head 相对 scale / 方向误差；分解是描述性的，不改变输入。
3. **分析维度**：Head 基（Q/K/V 行块、O 列块）、谱带基（Top/Middle/Tail）、类型（Q/K/V/O）。
4. **fused QKV 与 O 分开**：Q/K/V 的 NS 输入是 fused 1152×384；O 是独立 384×384。不能把 Q/K/V 单独做 NS 当成实际优化器行为。
5. **NS 实现与训练一致**：5 步、系数 `(3.4445, -4.775, 2.0315)`、eps=1e-7、转置与归一化一致；不要用精确 SVD 代替 `Φ_K`。
6. **分类型、分 seed、分节点报告**：Head 身份是 seed-dependent，聚合会掩盖差异。

## 5. 第一批的输出

BF16 自然误差的传递图：

| 误差分量 | Q/K | V | O |
|---|---|---|---|
| 全局 scale | A、share_L1 | ... | ... |
| Head 相对 scale | A、share_L1 | ... | ... |
| 方向误差 | A、share_L1 | ... | ... |
| 谱带 Top/Middle/Tail | 分带 A | ... | ... |

决策树：

- Head 相对 scale 的 A 最大 → 方法阶段优先 per-head scale；
- 方向误差的 A 最大 → 格式/mantissa 是主因，粒度不是第一杠杆；
- Tail 的 A 明显大于 Top → 方向保真度必须分带；
- QK/VO 的 A 不同 → 类型级精度分配有依据。

## 6. 边界与可证伪点

- 第一批不测 bit-width、不测 block size、不测状态敏感度；这些是 downstream。INT8 的 per-head/per-block 粒度留到方法阶段。
- 第一批用**数值模拟的低精度表示**：BF16 作为低精度基座；FP8 可选对照；INT8 per-tensor 只作 side observation，不进入 Q2–Q4。NS 计算保持 FP32。实际 kernel 的误差（累加、缩放、溢出）另开实验。
- 可证伪点：
  - 全局 scale 没被消掉 → 实现或归一化假设有问题；
  - Head 相对 scale 的 A 与方向误差的 A 接近 → “per-head scale 是特殊风险”不成立；
  - QK/VO 的 A 没有差异 → 类型级设计不成立；
  - Tail 的 A 不比 Top 大 → 分带方向保真度不是必须的。

## 7. 后续批次（不在本批）

- 第二批：状态敏感度（`M_prev / D / U / W`）、QK vs VO 的 matched-error 容忍度、单步 `ΔL`。
- 第三批：粒度与位宽（per-tensor / per-head / per-block；8/4-bit）。
- 累积阶段：loss 曲线、稳定性、长期 Head 身份、path selection、完整训练时间。

## 8. 与评价指标的关系

主指标：分谱带方向保真度 + 单步 `ΔL_probe`。
辅助 guardrail：`share_L1`、`top1_same`、`γ_h`/`effective_heads` 漂移。
诊断：状态保真度、`R_norm`、误差分解、曲率、Principal Angles、存储/计算。

详见 `platform/docs/评价指标.txt`。
