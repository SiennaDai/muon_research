# FP32 Muon Snapshot Collector 设计文档

> 任务：Task 1 — FP32 Muon Snapshot Collector 接口与数据协议设计  
> 设计对象：`platform/`（当前工作树）  
> 设计日期：2026-11-17  
> 状态：**设计完成，未实现**  
> 约束：本任务只创建设计文档；未修改任何 Python 源码、现有配置、测试、README 或其他文档；未安装依赖、未下载数据、未启动训练。

---

## 0. 文档使用方式与源码基线

本文是后续实现的唯一设计依据。所有接口、字段名、目录名、写入顺序和错误策略均按本文执行；实现时若发现与当前源码不一致，以当前源码为准，并在实现 PR/说明中记录差异。

源码基线（本轮已复核）：

- 训练入口：`platform/src/main.py:8-53`
- 训练循环：`platform/src/train_platform.py:169-413`
- 一次 update 的循环体：`platform/src/train_platform.py:317-385`
- `ReferenceMuon`：`platform/src/optim/muon_reference.py:26-97`
- Muon step 内部：`platform/src/optim/muon_reference.py:62-97`
- 参数分组：`platform/src/train_platform.py:106-128`
- Checkpoint：`platform/src/train_platform.py:162-166`、`:380-381`、`:401-403`
- 文件与 RNG：`platform/src/experiment_io.py:18-93`
- Recipe 校验：`platform/src/config/recipe.py:50-201`
- 数据采样：`platform/src/data/frozen_tokens.py:99-127`
- 现有 Mechanism/Fidelity Observer：`platform/src/optim/muon_mechanism.py:27-208`、`platform/src/optim/muon_update_fidelity.py:162-212`

> 与上一轮审计报告的差异见 §1.5。此前审计报告中尚待决策的“完整模型 W/G”“M_t 单一角色”等，本文已按 Task 1 的确定项收敛为 Muon-only 与 M_t 双角色。

---

## 1. 设计目标与边界

### 1.1 设计目标

1. 在指定 landmark update 采集 Muon 参数组的七类逻辑 Tensor：`W_before`、`W_after`、`G`、`M_prev`、`M_t`（拆为 `M_t_candidate` / `M_t_persisted`）、`D`、`U`。
2. 每个 Snapshot 对应一个完整 update，使用绝对 update 编号，保存 FP32 CPU Tensor 副本。
3. Snapshot 可独立理解、独立校验、离线复算；Manifest 是提交标记。
4. Collector 完全不改变训练计算、优化器状态、RNG、Checkpoint 和 resume 语义。
5. 采集失败、磁盘不足、文件损坏都不能中断训练；失败可诊断、可清理、可跳过。
6. 配置使用独立 sidecar，不进入 recipe 的 `scientific_fingerprint`。
7. v1 同步采集、同步写盘，不引入后台线程、异步拷贝、跨 update 内容寻址缓存。

### 1.2 已确定的研究范围（不再讨论）

- **采集范围 A**：七类逻辑 Tensor 全部限定在 Muon 参数组，不采集完整模型。
- 30M 配置的 Muon 组：6 层 × 每层 5 个矩阵 = **30 个矩阵**。
- 参数名只允许：
  - `transformer.h.{i}.attn.c_attn.weight`
  - `transformer.h.{i}.attn.c_proj.weight`
  - `transformer.h.{i}.mlp.w1.weight`
  - `transformer.h.{i}.mlp.w2.weight`
  - `transformer.h.{i}.mlp.c_proj.weight`
- 不采集 tied embedding、`lm_head`、RMSNorm、辅助 AdamW 参数及其一阶/二阶矩。
- 参数集合必须来自当前训练平台实际生效的 `parameter_groups`（`train_platform.py:106-128`），不得在 Collector 中重新定义筛选规则。
- `G` 是全局梯度裁剪后的梯度；不采集裁剪前的完整梯度。
- `D` 是传入 `zeropower_newton_schulz` 的原始 `direction`，不是 NS 内部归一化/转置后的临时矩阵。
- `U` 是 NS 返回的原始输出，不乘学习率、不提前乘负号。
- `W_after` 必须直接来自优化器计算的 `updated`，不能在 Collector 中重新计算。
- 第一次 update 的 `M_prev` 是零张量，v1 保存完整 FP32 Tensor，不引入零值编码。
- Snapshot 不参与训练恢复，不写入 `optimizer.state_dict()` 或 Checkpoint。
- 配置不进入 recipe 指纹。
- v1 不做跨 update 去重；只处理同一个 Snapshot 内明确的 Tensor alias。

### 1.3 非目标

以下内容不在本设计范围内，后续阶段单独设计：

- Snapshot 离线分析、Landscape、Hessian、Mode Connectivity。
- 低精度状态编码、VQ、压缩算法。
- 离线实验平台、可视化 Dashboard、Snapshot replay 工具。
- 完整模型参数/梯度快照。
- 辅助 AdamW 状态快照。
- 异步采集、后台写盘、跨 update 内容寻址去重。
- 严格模式、自动回滚、复杂的优雅停止。

### 1.4 术语

| 术语 | 含义 |
|---|---|
| update | 一次完整的 `optimizer.step()`，从 1 开始，绝对编号 |
| landmark | 配置中要求采集的 update 编号 |
| logical Tensor | 七类语义张量：W_before/W_after/G/M_prev/M_t/D/U |
| role | 物理 Schema 中的一个角色槽；`M_t` 拆成 `M_t_candidate` 和 `M_t_persisted`，所以有 8 个 role |
| payload | `update_XXXXXX.pt` 内部的纯 Tensor 容器 |
| manifest | `update_XXXXXX.manifest.json`，元数据 + 引用 + 校验，提交标记 |
| alias | 同一个参数内两个 role 指向同一个 Python Tensor 对象，只存一份 |
| committed | Manifest 已原子写入并通过校验的 Snapshot |
| orphan | 有 `.pt` 但没有有效 Manifest 的文件，不视为 Snapshot |

### 1.5 与上一轮审计报告的关系与差异

| 审计报告内容 | 本文处理 | 说明 |
|---|---|---|
| 七类张量 scope 可能是完整模型或 Muon-only | **已收敛为 Muon-only（方案 A）** | Task 1 确定项 |
| `M_t` 只有一个角色 | **拆为 `M_t_candidate` / `M_t_persisted`** | Task 1 确定项 |
| 建议在 `ReferenceMuon` 中加 hook | 采用；最终位置为现有 observer 之后、`p.copy_` 之前 | 见 §4.4 |
| `source_snapshot()` 可能因无 origin 失败 | 列为前置工程问题 A，要求 origin 缺失时记 null、不中断训练 | 见 §10.1 |
| `KeyboardInterrupt` 可能保存 mid-update checkpoint | 列为前置工程问题 B，要求只保留最后一个完整边界 Checkpoint | 见 §10.2 |
| 辅助 AdamW weight decay 与 `eval.attention_backend` 问题 | 列入 FP32 Reference 启动前配置核查清单，本任务不修改 | 见 §10.3 |
| 建议 sidecar 配置 | 采用，文件为 `run_dir/snapshot_config.json` | 见 §6 |
| 建议内容寻址去重 | **v1 不做跨 update 去重** | Task 1 确定项 |

复核结论：当前源码与上一轮审计报告的关键行号一致；`ReferenceMuon.step`、`train_platform` 循环、`experiment_io`、`main.py` 均未变化。

---

## 2. 最终 Tensor Schema

### 2.1 逻辑角色与物理角色

七类逻辑 Tensor 对应 **8 个 role 槽**：

| role | 逻辑类 | 定义 | 来源（`ReferenceMuon.step`） |
|---|---|---|---|
| `W_before` | W_t | 本次 update 前的 Muon 参数 | `p`，`:67`，`p.copy_` 之前 |
| `W_after` | W_{t+1} | 本次 update 完成后的 Muon 参数 | `updated`，`:76` |
| `G` | G_t | 全局裁剪后、实际进入 Muon 的梯度 | `grad = p.grad.float()`，`:69` |
| `M_prev` | M_{t-1} | 本次更新开始前实际使用的持久化动量 | `momentum_prev`，`:72` 的局部变量 |
| `M_t_candidate` | M_t（候选） | `μ·M_prev + G`，参与方向构造 | `momentum`，`:73` |
| `M_t_persisted` | M_t（持久化） | `persist_state(...)` 后写入 state 的值 | `persisted`，`:79-81` |
| `D` | D_t | 传入 NS 的原始方向矩阵 | `direction`，`:74` |
| `U` | U_t | NS 返回的原始输出 | `ns_output`，`:75`（实现时由 `update` 改名） |

**不得混用：**

- `M_t_candidate` 是未舍入、本步实际参与 `D` 构造的动量。
- `M_t_persisted` 是写入 `optimizer.state[p]["muon_momentum"]` 的值。
- `state_simulation == "none"` 时两者是同一对象（`state_simulation.py:161-162`），可通过 alias 只存一份。
- 其他模拟（bf16/int8/int4）时两者通常不同，必须分别保存。
- 即使两者数值相等但对象不同，也分别保存；alias 只依据对象身份，不依据数值相等。

### 2.2 Snapshot 文件的物理分层

每个 update 由两个文件组成：

```text
run_dir/snapshots/
  update_000001.pt
  update_000001.manifest.json
```

分工：

| 文件 | 内容 | 是否提交标记 |
|---|---|---|
| `update_000001.pt` | 纯 Tensor 容器：`format/version/run_id/update/segment_id/tensors` | 否 |
| `update_000001.manifest.json` | 全部元数据、Tensor 引用、shape/dtype/bytes/checksum、alias、完整性、文件级 SHA-256 | **是** |

设计原则：

- `.pt` 只放 Tensor 和最小身份字段；不放训练上下文、不放 Python 对象、不放路径。
- Manifest 是唯一权威元数据；`Tensor` 的 shape/dtype/字节数/checksum 都在 Manifest 中，`.pt` 内不重复这些元数据（`.pt` 的 Tensor 自身仍带 shape/dtype）。
- 加载时必须同时校验 `.pt` 与 Manifest；只有 Manifest 存在且通过校验，Snapshot 才算 committed。
- `torch.load` 使用 `weights_only=True`（payload 限制为 `dict/list/str/int/float/bool/None/Tensor`），避免加载任意 Python 对象。

### 2.3 `.pt` payload Schema

```json
{
  "format": "muon_snapshot_tensors",
  "version": 1,
  "snapshot_id": "<run_id>:<update>",
  "run_id": "<run_id>",
  "update": 1,
  "segment_id": 0,
  "tensors": {
    "t0000": "<torch.Tensor: CPU float32, shape (1152, 384)>",
    "t0001": "<torch.Tensor>"
  }
}
```

约束：

- `tensors` 的 key 是 `t{ordinal:04d}`，从 `t0000` 开始。
- Tensor 已 detach、CPU、FP32、contiguous。
- 同一参数内的 alias 不产生新的 `tensor_id`；alias role 在 Manifest 中引用已有的 `tensor_id`。
- 不允许出现未被任何 role 引用的 `tensor_id`（orphan tensor）。
- 不允许出现 `.pt` 内 Tensor 的 shape/dtype 与 Manifest 不一致。
- `snapshot_id` 固定为 `f"{run_id}:{update:06d}"`；`update` 为绝对编号。
- `format`/`version` 必须与 Manifest 的 `tensors_file.format`/`version` 一致。

### 2.4 Manifest Schema

Manifest 是严格 JSON（`allow_nan=False`），顶层字段如下：

```json
{
  "format": "muon_snapshot_manifest",
  "version": 1,
  "snapshot_id": "<run_id>:<update>",
  "status": "committed",
  "run_id": "<run_id>",
  "update": 1,
  "segment_id": 0,
  "created_at": "2026-11-17T12:34:56Z",
  "identities": {
    "recipe_fingerprint": "<sha256>",
    "data_fingerprint": "<sha256>",
    "snapshot_config_hash": "<sha256>",
    "collector_version": 1
  },
  "source": {
    "remote": null,
    "commit": "<git sha or null>",
    "dirty": true,
    "diff_sha256": "<sha256 or null>",
    "provenance_status": "ok"
  },
  "environment": {
    "python": "3.11.x",
    "torch": "2.8.0",
    "numpy": "2.3.2",
    "platform": "...",
    "device": "cuda:0",
    "cuda_available": true,
    "cuda_devices": ["..."]
  },
  "model": {
    "config": {
      "family": "llama",
      "n_layer": 6,
      "n_embd": 384,
      "n_head": 6,
      "ffn_dim": 1024,
      "multiple_of": 256,
      "vocab_size": 50304,
      "sequence_length": 256,
      "dropout": 0,
      "bias": false,
      "tie_embeddings": true,
      "init_std": 0.02,
      "rmsnorm_eps": 1e-05,
      "rope_theta": 10000
    },
    "parameter_records": [
      {
        "name": "transformer.h.0.attn.c_attn.weight",
        "group": "muon",
        "shape": [1152, 384],
        "numel": 442368,
        "dtype": "torch.float32",
        "parameter_index": 0,
        "optimizer_param_index": 0
      }
    ],
    "muon_parameter_order": [
      "transformer.h.0.attn.c_attn.weight",
      "transformer.h.0.attn.c_proj.weight",
      "transformer.h.0.mlp.w1.weight",
      "transformer.h.0.mlp.w2.weight",
      "transformer.h.0.mlp.c_proj.weight",
      "transformer.h.1.attn.c_attn.weight"
    ],
    "muon_parameter_count": 30,
    "full_parameter_records_sha256": "<sha256 or null>"
  },
  "optimizer": {
    "name": "reference_muon",
    "weight_decay": 0.1,
    "muon_momentum": 0.95,
    "muon_nesterov": true,
    "state_simulation": "none",
    "state_quantization_granularity": "per_state_tensor",
    "state_quantization_block_size": 2048,
    "muon_ns_steps": 5,
    "muon_ns_coefficients": [3.4445, -4.775, 2.0315],
    "muon_eps": 1e-07
  },
  "training": {
    "tokens_this_update": 8192,
    "processed_target_tokens": 1048576,
    "train_nll": 5.123456,
    "learning_rate": 0.0001,
    "pre_clip_grad_norm": 0.876543,
    "clipped": false,
    "grad_clip_norm": 1.0
  },
  "data": {
    "sequence_length": 256,
    "micro_batch_size": 4,
    "accumulation_steps": 8,
    "window_ids": [17, 3, 900, 42],
    "micro_batch_window_ids": [[17, 3, 900, 42]],
    "sampler_epoch_before": 0,
    "sampler_offset_before": 0,
    "sampler_epoch_after": 0,
    "sampler_offset_after": 32
  },
  "tensors_file": {
    "path": "update_000001.pt",
    "sha256": "<sha256-of-entire-pt-file>",
    "nbytes": 297271296,
    "payload_nbytes": 297271296,
    "format": "muon_snapshot_tensors",
    "version": 1
  },
  "parameters": {
    "transformer.h.0.attn.c_attn.weight": {
      "parameter_index": 0,
      "optimizer_param_index": 0,
      "shape": [1152, 384],
      "dtype": "float32",
      "numel": 442368,
      "first_step": false,
      "roles": {
        "W_before": {
          "tensor_id": "t0000",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        },
        "W_after": {
          "tensor_id": "t0001",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        },
        "G": {
          "tensor_id": "t0002",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        },
        "M_prev": {
          "tensor_id": "t0003",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        },
        "M_t_candidate": {
          "tensor_id": "t0004",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        },
        "M_t_persisted": {
          "alias_of": "M_t_candidate"
        },
        "D": {
          "tensor_id": "t0005",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        },
        "U": {
          "tensor_id": "t0006",
          "shape": [1152, 384],
          "dtype": "float32",
          "numel": 442368,
          "nbytes": 1769472,
          "sha256": "<sha256>"
        }
      }
    },
    "transformer.h.0.mlp.w1.weight": {
      "parameter_index": 2,
      "optimizer_param_index": 2,
      "shape": [1024, 384],
      "dtype": "float32",
      "numel": 393216,
      "first_step": false,
      "roles": {
        "W_before": {
          "tensor_id": "t0014",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        },
        "W_after": {
          "tensor_id": "t0015",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        },
        "G": {
          "tensor_id": "t0016",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        },
        "M_prev": {
          "tensor_id": "t0017",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        },
        "M_t_candidate": {
          "tensor_id": "t0018",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        },
        "M_t_persisted": {
          "alias_of": "M_t_candidate"
        },
        "D": {
          "tensor_id": "t0019",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        },
        "U": {
          "tensor_id": "t0020",
          "shape": [1024, 384],
          "dtype": "float32",
          "numel": 393216,
          "nbytes": 1572864,
          "sha256": "<sha256>"
        }
      }
    }
  },
  "aliases": [
    {
      "parameter_name": "transformer.h.0.attn.c_attn.weight",
      "alias_role": "M_t_persisted",
      "canonical_role": "M_t_candidate",
      "tensor_id": "t0004",
      "reason": "same_tensor_object"
    },
    {
      "parameter_name": "transformer.h.0.mlp.w1.weight",
      "alias_role": "M_t_persisted",
      "canonical_role": "M_t_candidate",
      "tensor_id": "t0018",
      "reason": "same_tensor_object"
    }
  ],
  "integrity": {
    "expected_parameter_count": 30,
    "observed_parameter_count": 30,
    "expected_role_slots_per_parameter": 8,
    "expected_role_slots": 240,
    "observed_role_slots": 240,
    "unique_tensor_count": 210,
    "alias_count": 30,
    "all_parameters_present": true,
    "all_roles_present": true,
    "all_shapes_match": true,
    "all_dtypes_float32": true,
    "per_tensor_checksums_ok": true,
    "tensors_file_sha256_ok": true,
    "no_orphan_tensors": true,
    "byte_order": "little"
  }
}
```

说明：

- 示例只展示两个参数：一个 attention 矩阵和一个 MLP 矩阵；其余 28 个参数结构相同。
- `parameter_index` 是 Muon 组内顺序（0..29）；`optimizer_param_index` 是优化器全局参数索引，用于与 Checkpoint 交叉校验。当前分组中 Muon 组在前，因此 Muon 参数的全局索引等于 `parameter_index`；实现时仍应从实际 `groups` 枚举，不硬编码。
- `full_parameter_records_sha256` 可选：如果 `parameter_records` 只放 Muon 记录，用它对完整 `parameters.json` 做引用；如果 Manifest 已放完整 `parameter_records`，该字段可为 null。本文建议 Manifest 放 **完整** `parameter_records`（44 条），同时 `muon_parameter_order` 明确 30 个；字段 `full_parameter_records_sha256` 可省略。示例为简洁只展示一条 Muon 记录。
- `tensors_file.nbytes` 是 `.pt` 文件实际字节数；`payload_nbytes` 是唯一 Tensor 的理论字节和；二者之差是 `torch.save` 的容器开销。
- `integrity.unique_tensor_count` 在 `state_simulation=="none"` 时为 210（30×7），因为 30 个 `M_t_persisted` alias 到 `M_t_candidate`；如果模拟使两者不同，则为 240。若 `muon_nesterov=false`，`D` 也可能 alias 到 `M_t_candidate`，unique 数会再减少 30（变为 180）。Collector 必须按实际对象身份计算，不得硬编码。
- `status` 只允许 `"committed"`；失败/跳过不写 Manifest，因此不存在 `"failed"` Manifest。

### 2.5 Tensor descriptor 字段

每个非 alias role 的 descriptor 必须包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `tensor_id` | string | `t0000` 形式；在同一个 `.pt` 内唯一 |
| `shape` | list[int] | 与 `parameter_records` 一致 |
| `dtype` | string | 固定 `"float32"` |
| `numel` | int | `prod(shape)` |
| `nbytes` | int | `numel * 4` |
| `sha256` | string | CPU FP32 contiguous 字节的 SHA-256 |

Alias role 的 descriptor 只包含：

```json
{"alias_of": "<canonical_role_name>"}
```

并同时在顶层 `aliases` 中记录一条记录，包含 `parameter_name`、`alias_role`、`canonical_role`、`tensor_id`、`reason`。

### 2.6 Alias 规则

**允许的 alias：**

1. `M_t_persisted` is `M_t_candidate`：`state_simulation == "none"` 时 `persist_state` 原样返回（`state_simulation.py:161-162`）。
2. `D` is `M_t_candidate`：`muon_nesterov == false` 时 `direction = momentum`（`muon_reference.py:74`）。
3. 上述两者可同时发生。

**不在 v1 处理：**

- 不同参数之间共享 storage：Muon 参数应为独立 Tensor；若检测到跨参数共享，记录 `unexpected_shared_storage` 错误，不静默去重。
- 数值相等但对象不同：不 alias；分别保存。
- View/非 contiguous：先 `.contiguous()` 再存；不按 storage 合并。

**检测方式：**

- 在 `observe` 内按固定 role 顺序遍历 8 个 role。
- 用 Python `is` 比较当前 Tensor 与同一参数已处理的 role Tensor。
- 若命中，记录 alias，不 clone、不新增 `tensor_id`。
- 若未命中，分配新 `tensor_id`，执行一次 CPU FP32 clone，计算 checksum。
- 这样 `M_t_persisted` 和 `D` 的 alias 都能正确处理，且不会因数值相等误合并。

### 2.7 如何保证一个参数的所有 Tensor 属于同一次 update

四层保证：

1. **Hook 位置**：`ReferenceMuon` 的 Snapshot hook 在一次 `p.copy_` 之前、同一个 `step()` 调用内触发；8 个 role 来自同一组局部变量（`W_before=p`、`W_after=updated`、`G=grad`、`M_prev=momentum_prev`、`M_t_candidate=momentum`、`M_t_persisted=persisted`、`D=direction`、`U=ns_output`）。
2. **update 传递**：`train_platform` 在 step 前调用 `optimizer.set_training_update(update)`；hook 把该 `update` 传给 `observe`。Collector 在 `begin_update` 记录 `expected_update`，每个 `observe` 校验 `update == expected_update`，不匹配则失败并禁用。
3. **参数名唯一**：hook 通过 `id(p) -> parameter_name` 映射传入名称；Collector 校验名称属于 `muon_parameter_order`，且同一 update 内不重复。名称是唯一键，不使用 Python `id` 作为持久标识。
4. **Manifest 双层标识**：`.pt` 和 Manifest 都写 `snapshot_id/run_id/update/segment_id`；`finish_update` 校验二者一致，再校验 30 个参数、8 个 role、shape/dtype 全部完成。

### 2.8 如何校验 30 个 Muon 矩阵是否全部采集完成

`finish_update` 执行以下完整性检查；任何一项失败都不写 Manifest，记录错误并禁用 Collector：

1. `expected_parameter_names = tuple(muon_parameter_order)`，来自实际 `parameter_groups` 的 Muon 组，不是硬编码。
2. `observed_names = set(parameters.keys())`；必须与 expected 集合完全相等。
3. `len(expected) == len(set(expected)) == muon_parameter_count`；当前 30M 为 30。
4. 每个参数必须恰好出现一次；重复 `observe` 记录 `duplicate_parameter`。
5. 每个参数必须有且仅有 8 个 role；alias 也计入 role 槽。
6. 每个非 alias role 的 `shape/dtype/numel/nbytes` 与 `parameter_records` 一致；dtype 必须是 `float32`。
7. 每个 alias 的 canonical role 存在，且 shape/dtype 一致；alias 链最多一层（不允许 alias 指向 alias）。
8. 所有 `tensor_id` 在 `.pt` 中存在；所有 `.pt` 中的 `tensor_id` 至少被一个 role 引用（无 orphan）。
9. `first_step` 与 `M_prev` 的语义一致：若 `first_step=true`，记录（v1 不额外扫描零值；离线可校验 checksum/零值）。
10. `integrity` 计数与上述检查一致；`observed_parameter_count == expected_parameter_count`。

失败时错误记录包含：

```json
{
  "code": "incomplete_snapshot",
  "missing_parameters": ["..."],
  "unexpected_parameters": ["..."],
  "missing_roles": {"parameter": ["..."]},
  "duplicate_parameters": ["..."],
  "shape_mismatches": [{"parameter": "...", "role": "...", "expected": [1152, 384], "observed": [384, 384]}]
}
```

### 2.9 示例：一个 attention 矩阵 + 一个 MLP 矩阵

上面的 Manifest 示例已展示：

- `transformer.h.0.attn.c_attn.weight`：shape `[1152, 384]`，8 个 role；`M_t_persisted` alias 到 `M_t_candidate`；`D`、`U` 独立；`M_prev` 独立。
- `transformer.h.0.mlp.w1.weight`：shape `[1024, 384]`，结构相同。

`.pt` 对应内容（示意）：

```python
{
  "format": "muon_snapshot_tensors",
  "version": 1,
  "snapshot_id": "run-abc:000001",
  "run_id": "run-abc",
  "update": 1,
  "segment_id": 0,
  "tensors": {
    "t0000": torch.Tensor,  # W_before, (1152, 384), cpu, float32
    "t0001": torch.Tensor,  # W_after
    "t0002": torch.Tensor,  # G
    "t0003": torch.Tensor,  # M_prev
    "t0004": torch.Tensor,  # M_t_candidate (M_t_persisted alias)
    "t0005": torch.Tensor,  # D
    "t0006": torch.Tensor,  # U
    "t0014": torch.Tensor,  # mlp.w1 W_before
    # ...
  },
}
```

注意：`t0007..t0013` 在示例中不存在，因为第一个参数的 `M_t_persisted` 是 alias，不产生新 ID；实际编号按“参数顺序 × role 顺序，仅对非 alias 分配”连续生成。实现时应保证编号生成规则确定、可复现。

### 2.10 Tensor 校验和与字节序

- 单个 Tensor checksum：

```python
t = t.detach().to("cpu", torch.float32).contiguous()
payload = t.numpy().astype("<f4", copy=False).tobytes()
sha256 = hashlib.sha256(payload).hexdigest()
```

- 使用 little-endian `<f4`；Manifest `integrity.byte_order = "little"`。
- 加载后重新计算 checksum，必须与 Manifest 完全一致。
- `.pt` 文件 checksum：对完整文件字节流做 SHA-256，使用 1 MiB 分块读取。
- Manifest 自身不做文件级 hash；它是提交标记。若要额外审计，可在 run-level status 中记录 manifest 文件 hash，但不是 v1 必需。

---

## 3. 完整 Metadata Schema

### 3.1 训练信息

| 字段 | 类型 | 来源 | 采集位置 | 是否需要新增采集 |
|---|---|---|---|---|
| `update` | int | 循环变量 `update` | `train_platform.py:319` | 否 |
| `run_id` | string | `run_id`（新 run 为 `run_dir.name`；resume 从 checkpoint 恢复） | `train_platform.py:194`, `:200` | 否 |
| `segment_id` | int | `segment`（resume 时为 `next_segment_id`） | `train_platform.py:193`, `:199` | 否 |
| `tokens_this_update` | int | `n_tokens`，累加 `y.numel()` | `train_platform.py:324` | 否（begin 时已可得） |
| `processed_target_tokens` | int | `processed + n_tokens` | `train_platform.py:324`, `:372` | 否（begin 时计算） |
| `train_nll` | float | `nll_sum / n_tokens`，经 `require_finite_metric` | `train_platform.py:341` | 传给 `finish_update` |
| `learning_rate` | float | `lr`，由 `learning_rate(...)` 计算 | `train_platform.py:330-331` | 否 |
| `pre_clip_grad_norm` | float | `grad_norm_value`，`clip_grad_norm_` 返回值 | `train_platform.py:329` | 否 |
| `clipped` | bool | `grad_norm_value > grad_clip_norm` | `train_platform.py:329` | 否 |
| `grad_clip_norm` | float | `cfg["train"]["grad_clip_norm"]` | recipe | 否 |

说明：

- `train_nll` 在 step 之后计算（`:341`）；因此 `finish_update` 必须在 `:341` 之后调用。
- 如果 `train_nll` 非有限，`require_finite_metric` 会抛 `NonFiniteMetricError`，run 进入 `diverged_nonfinite`，`finish_update` 不会执行；不会产生该 update 的 Snapshot。
- `processed_target_tokens` 应使用 `processed + n_tokens` 的捕获值，不要在 `:372` 更新 `processed` 之后读取。
- `tokens_this_update` 当前等于 `micro_batch_size * sequence_length * accumulation_steps`；从 `n_tokens` 读取，避免重复计算。

### 3.2 数据采样信息

| 字段 | 类型 | 来源 | 采集位置 | 说明 |
|---|---|---|---|---|
| `sequence_length` | int | `cfg["model"]["sequence_length"]` | recipe | 用于解释 window |
| `micro_batch_size` | int | `cfg["train"]["micro_batch_size"]` | recipe | 分组依据 |
| `accumulation_steps` | int | `cfg["train"]["accumulation_steps"]` | recipe | 分组数量 |
| `window_ids` | list[int] | `ids = sampler.take(...)` | `train_platform.py:319` | 本轮实际使用的 window id，扁平顺序 |
| `micro_batch_window_ids` | list[list[int]] | 按 `micro_batch_size` 切分 `ids` | 由 `train_platform` 构造，等价于 `:321` 的切片 | 每个 micro-batch 的实际 window id |
| `sampler_epoch_before` | int | `sampler.epoch` | `train_platform.py:319` 之前 | take 前 |
| `sampler_offset_before` | int | `sampler.offset` | 同上 | take 前 |
| `sampler_epoch_after` | int | `sampler.epoch` | `:319` 之后 | take 后 |
| `sampler_offset_after` | int | `sampler.offset` | 同上 | take 后 |

不保存：

- 完整 `sampler.permutation`。
- 完整 `sampler.state_dict()`。
- DataLoader 状态（当前没有 DataLoader）。
- 完整 token 内容或 batch tensor。

最小新增采集位置：

- 当前 `:319` 把 `zero_grad`、`take`、初始化写在一行。实现时拆成多条语句：

```python
optimizer.zero_grad(set_to_none=True)
sampler_epoch_before, sampler_offset_before = sampler.epoch, sampler.offset
ids = sampler.take(cfg["train"]["micro_batch_size"] * cfg["train"]["accumulation_steps"])
sampler_epoch_after, sampler_offset_after = sampler.epoch, sampler.offset
micro_batch_window_ids = [
    ids[a * cfg["train"]["micro_batch_size"]:(a + 1) * cfg["train"]["micro_batch_size"]]
    for a in range(cfg["train"]["accumulation_steps"])
]
```

这不会改变采样顺序或训练行为，只是把已有值暴露给 Collector。

### 3.3 优化器与 Newton–Schulz 信息

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `optimizer.name` | string | `cfg["optimizer"]["name"]` | 固定 `"reference_muon"`（v1 只支持该路径） |
| `weight_decay` | float | 实际 Muon group 的 `weight_decay` | `ReferenceMuon` 使用 group 值（`muon_reference.py:76`） |
| `muon_momentum` | float | Muon group `muon_momentum` | 默认 0.95 |
| `muon_nesterov` | bool | Muon group `muon_nesterov` | 默认 true |
| `state_simulation` | string | Muon group `state_simulation` | `"none"` / `"bf16_roundtrip"` / int8/int4 模拟 |
| `state_quantization_granularity` | string | Muon group 值 | 默认 `"per_state_tensor"` |
| `state_quantization_block_size` | int | Muon group 值 | 默认 2048 |
| `muon_ns_steps` | int | Muon group `muon_ns_steps` | 默认 5 |
| `muon_ns_coefficients` | list[float] | Muon group 值 | 默认 `[3.4445, -4.775, 2.0315]` |
| `muon_eps` | float | Muon group `muon_eps` | 默认 1e-7 |

来源原则：

- 不要只从 `cfg["optimizer"]` 读取并假设默认值；应读取**实际传给 `ReferenceMuon` 的 Muon group**，因为 `make_optimizer`（`lowp_adapter.py:19-24`）和 `ReferenceMuon.__init__`（`muon_reference.py:34-47`）会应用默认值。
- 在 `train_platform` 中，`optimizer` 构造完成后：

```python
muon_group = next(g for g in optimizer.param_groups if g.get("optimizer_group") == "muon")
optimizer_metadata = {k: v for k, v in muon_group.items() if k != "params"}
```

- `lr` 是每个 update 动态写入 group 的，不作为“优化器配置”静态值；Snapshot 的 `training.learning_rate` 记录本次实际 lr。

### 3.4 实验环境与来源信息

| 字段 | 类型 | 来源 | 采集位置 | 说明 |
|---|---|---|---|---|
| `model.config` | dict | `cfg["model"]` | recipe | 模型结构 |
| `model.parameter_records` | list[dict] | `parameter_groups(model, ...)` 返回的 `records` | `train_platform.py:186` | 包含 name/group/shape/numel/dtype |
| `model.muon_parameter_order` | list[string] | 从实际 Muon group 枚举 | `groups` | 30 个名称，顺序即优化器顺序 |
| `model.muon_parameter_count` | int | 上述列表长度 | 实际 groups | 当前 30M 为 30 |
| `seed` | int | `cfg["experiment"]["seed"]` | recipe | 模型初始化/全局种子 |
| `data_seed` | int | `cfg["experiment"]["data_seed"]` | recipe | sampler 种子 |
| `algorithm_seed` | int | `cfg["experiment"]["algorithm_seed"]` | recipe | 当前训练未使用；作为 provenance 记录 |
| `recipe_fingerprint` | string | `cfg["fingerprint"]` | recipe | 科学指纹 |
| `data_fingerprint` | string | `manifest["fingerprint"]` | `train_platform.py:178` | 数据指纹 |
| `source.commit` | string/null | `source_snapshot(root)["upstream_commit"]` | run 开始 | git commit |
| `source.dirty` | bool/null | `source_snapshot(root)["dirty"]` | run 开始 | 工作区是否 dirty |
| `source.diff_sha256` | string/null | 对 `source_snapshot(root)["diff"]` 做 SHA-256 | run 开始 | 不保存完整 diff |
| `source.provenance_status` | string | 新增：`ok` / `remote_missing` / `git_unavailable` / `ownership_error` | run 开始 | 见 §10.1 |
| `environment` | dict | `environment_snapshot()` | `train_platform.py:222` 新建 run；resume 需补采或读 `environment.json` | python/torch/numpy/platform/cuda |
| `device` | string | `str(device)` | `train_platform.py:171` | `cpu` / `cuda:0` |
| `precision` | dict | `cfg["precision"]` | recipe | compute/parameter/gradient dtype/deterministic/tf32/attention_backend |
| `snapshot_config_hash` | string | sidecar 配置 canonical hash | run 开始 | 见 §6 |

最小新增采集位置：

- `source_snapshot(root)` 当前在 `train_platform.py:222`（新建 run）和 `:357`/`:413` 调用；Collector 需要一份**每次 run 只采集一次**的 source 信息。建议在 `run` 开始处调用一次 `source_snapshot(root)`，保存到局部变量，传给 Collector；resume 时也调用，或读取已有 `source.json` 并记录当前 provenance。不要每个 snapshot 调 git。
- `environment_snapshot()` 当前只在新建 run 的 `:222` 调用；resume 时 `environment.json` 已存在。Collector 可在 run 开始读取 `environment.json` 或调用一次 `environment_snapshot()`；建议调用一次并传入，记录 `created_at` 为 run 开始时间。
- `parameter_records` 和 `groups` 已在 `:186` 获得，直接传入，不重新计算。
- `optimizer_metadata` 在 `optimizer` 构造后从 Muon group 读取，不重新应用默认值。

### 3.5 Metadata 采集原则

1. **已有值直接传**：update、lr、grad norm、clipped、n_tokens、window ids、sampler epoch/offset、groups/records、optimizer group、source、environment、device、precision。
2. **Collector 不重复计算**：不重新算 `learning_rate`、不重新算 `clip_grad_norm`、不重新推导 parameter grouping、不重新应用优化器默认值。
3. **只新增最小暴露点**：
   - 拆分 `:319` 的 `take` 语句，捕获 sampler epoch/offset 前后值；
   - 构造 `micro_batch_window_ids`；
   - 在 `:341` 之后把 `train_nll` 传给 `finish_update`；
   - run 开始处采集一次 source/environment；
   - 从实际 Muon group 读取优化器元数据。
4. **不在 Collector 中读取训练 Tensor**：所有训练 Tensor 由 hook 直接传入；Collector 不在 `finish_update` 里回读 `p`/`p.grad`/`optimizer.state`。

---

## 4. Collector API

### 4.1 模块与导出

新文件：

```text
platform/src/optim/muon_snapshot_collector.py
```

导出：

```python
SnapshotConfig
SnapshotRunContext
SnapshotBeginContext
MuonSnapshotCollector
load_snapshot_config
validate_snapshot_config
scan_committed_snapshots
```

不修改：

- `platform/src/optim/muon_mechanism.py`
- `platform/src/optim/muon_update_fidelity.py`
- `platform/src/optim/muon.py`（旧实现，不在当前训练调用链）
- `platform/src/models/`
- `platform/src/data/`
- 现有 recipes/benchmarks

### 4.2 配置与上下文 dataclass

```python
@dataclass(frozen=True)
class SnapshotConfig:
    schema_version: int = 1
    enabled: bool = False
    updates: tuple[int, ...] = ()
    on_error: str = "disable"          # v1 仅允许 "disable"
    max_total_bytes: int | None = None
    min_free_bytes: int = 5 * 1024**3  # 5 GiB

    @property
    def config_hash(self) -> str: ...
```

```python
@dataclass(frozen=True)
class SnapshotRunContext:
    run_id: str
    segment_id: int
    recipe_fingerprint: str
    data_fingerprint: str
    snapshot_config_hash: str
    model_config: dict
    parameter_records: list[dict]        # 完整 records（Muon + aux）
    muon_parameter_order: tuple[str, ...]
    optimizer_param_index: dict[str, int]
    optimizer_metadata: dict
    train_config: dict
    data_config: dict
    seeds: dict
    source: dict
    environment: dict
    device: str
    precision: dict
```

```python
@dataclass(frozen=True)
class SnapshotBeginContext:
    update: int
    segment_id: int
    tokens_this_update: int
    processed_target_tokens: int
    learning_rate: float
    pre_clip_grad_norm: float
    clipped: bool
    grad_clip_norm: float
    window_ids: tuple[int, ...]
    micro_batch_window_ids: tuple[tuple[int, ...], ...]
    sampler_epoch_before: int
    sampler_offset_before: int
    sampler_epoch_after: int
    sampler_offset_after: int
```

### 4.3 `MuonSnapshotCollector` 初始化

```python
class MuonSnapshotCollector:
    FORMAT_VERSION = 1
    COLLECTOR_VERSION = 1

    def __init__(
        self,
        *,
        run_dir: str | Path,
        config: SnapshotConfig,
        context: SnapshotRunContext,
    ) -> None:
        ...
```

初始化职责：

1. `run_dir = Path(run_dir).resolve()`；`snapshots_dir = run_dir / "snapshots"`；`errors_path = run_dir / "snapshot_errors.jsonl"`。
2. 校验 `config.config_hash == context.snapshot_config_hash`；不一致 → 构造失败（启动期错误，训练不启动）。
3. 校验 `muon_parameter_order` 非空、无重复；校验 `parameter_records` 中每个 Muon record 的 name/shape/dtype。
4. 创建 `snapshots_dir`（`parents=True, exist_ok=True`）。
5. 清理临时文件 `*.tmp`；扫描孤儿 `.pt`（无 Manifest）并移入 `snapshots/orphaned/`。
6. 扫描已提交 Manifest，构建 `committed_updates`、`conflicted_updates`、`missed_updates`；校验 run_id/fingerprint/config hash/checksum。
7. 初始化内存状态：`self._buffer = {}`、`self._begin_context = None`、`self._disabled = False`、`self._disabled_reason = None`、`self._collecting = False`。
8. 不读取/修改 optimizer、model、sampler、RNG。

### 4.4 API 签名

```python
def should_collect(self, update: int) -> bool:
    """True iff enabled, update in config.updates, not committed/conflicted,
    not disabled, and no incomplete previous update."""

def begin_update(self, context: SnapshotBeginContext) -> None:
    """Begin a snapshot transaction for context.update.
    Clears the buffer; records expected update/segment and training context.
    No disk I/O."""

@torch.no_grad()
def observe(
    self,
    *,
    update: int | None,
    parameter_name: str | None,
    W_before: torch.Tensor,
    G: torch.Tensor,
    M_prev: torch.Tensor,
    M_t_candidate: torch.Tensor,
    M_t_persisted: torch.Tensor,
    D: torch.Tensor,
    U: torch.Tensor,
    W_after: torch.Tensor,
    first_step: bool,
) -> None:
    """Receive one Muon parameter's eight role tensors from ReferenceMuon.
    Only clones/metadata computation; no disk I/O; must not raise."""

def finish_update(self, *, update: int, train_nll: float) -> None:
    """Validate completeness, precheck disk, write .pt + manifest atomically.
    No-op if this update was not begun. Must not raise."""

def close(self) -> None:
    """Idempotent; record incomplete update if any; flush error log.
    Never writes a snapshot; never raises."""
```

Optional read-only helpers (recommended, not required):

```python
def committed_updates(self) -> tuple[int, ...]: ...
def missed_updates(self) -> tuple[int, ...]: ...
def conflicted_updates(self) -> tuple[int, ...]: ...
def status(self) -> dict: ...
```

### 4.5 `observe` 内部流程

1. 若 `self._disabled` 或 `not self._collecting`：直接 return。
2. 校验 `update is not None` 且 `update == self._expected_update`；否则 `_fail("wrong_update")`。
3. 校验 `parameter_name` 非空、在 `expected_names` 中、未在本 update 出现过；否则 `_fail(...)`。
4. 取 `expected_shape = self._expected_shapes[parameter_name]`；逐个校验 8 个 role：
   - `isinstance(t, torch.Tensor)`
   - `t.ndim == 2`
   - `t.dtype == torch.float32`
   - `tuple(t.shape) == expected_shape`
   - 否则 `_fail("tensor_contract_violation")`。
5. 按固定 role 顺序检测 alias：
   - `role_order = ("W_before","W_after","G","M_prev","M_t_candidate","M_t_persisted","D","U")`
   - 对每个 role，若当前 Tensor 与同一参数已处理的某个 role Tensor `is` 相同，记录 `alias_of=canonical_role`；否则分配新 `tensor_id`。
6. 对每个新 `tensor_id`：
   - `cpu = t.detach().to("cpu", torch.float32).contiguous().clone()`
   - `sha256 = compute_tensor_sha256(cpu)`
   - `nbytes = cpu.numel() * 4`
   - 存入 `self._tensor_blobs[tensor_id] = cpu`
   - 存 descriptor。
7. 存 `self._parameters[parameter_name] = {...}`，包含 `parameter_index`、`optimizer_param_index`、`shape`、`dtype`、`numel`、`first_step`、`roles`、`aliases`。
8. 任何异常：`_fail("observe_exception", exc)`；不 re-raise。
9. 不写磁盘、不调用 RNG、不修改传入 Tensor。

### 4.6 `finish_update` 内部流程

1. 若 `not self._collecting`：return（not a landmark）。
2. 校验 `update == self._expected_update`；否则 `_fail("finish_update_mismatch")`。
3. 校验 `train_nll` 是 finite float；否则 `_fail("nonfinite_train_nll")`。
4. 完整性校验（§2.8）。
5. 构造 Manifest（除 `tensors_file` 的 hash/size 外）。
6. 磁盘预检（§7.4）。
7. 事务式写入（§7.3）。
8. 成功：加入 `committed_updates`；清空 buffer；`self._collecting=False`。
9. 失败：`_fail(...)`；删除临时文件；不写 Manifest；清空 buffer；`self._collecting=False`。
10. 不修改训练状态；返回值可忽略（或返回内部 `SnapshotResult`，但训练循环不依赖）。

### 4.7 `close` 行为

- 若 `self._collecting`：记录 `incomplete_update_at_close`（含 update、已收集参数数、缺失参数），不写 Snapshot。
- 清空 buffer；关闭错误日志文件句柄（如果保持打开）。
- 写入可选的 `run_dir/snapshot_status.json`（best-effort，非提交标记）。
- 幂等；重复调用无副作用；不抛异常。

---

## 5. 模块职责、调用关系与时序

### 5.1 职责边界

| 模块 | 职责 | 不负责 |
|---|---|---|
| `ReferenceMuon` | 在 step 内提供真实计算产生的 8 个 role Tensor；提供 `id(p)->name` 映射；在 `p.copy_` 前调用 hook | 不做 clone、不做磁盘 I/O、不做完整性校验 |
| `MuonSnapshotCollector` | 判断是否采集；接收 Tensor；创建 CPU FP32 副本；暂存；完整性校验；生成 Manifest；事务写盘；扫描/校验已有 Snapshot；记录错误 | 不改训练状态；不写 optimizer.state_dict；不参与 resume 模型恢复 |
| `train_platform` | 加载 sidecar 配置；创建 Collector；设置 hook；在正确时点调用 `begin_update`/`finish_update`/`close`；提供训练上下文；校验 resume 配置一致性 | 不重新定义 Muon 参数筛选；不重新计算优化器/NS 张量 |
| `main.py` | 增加 `--snapshot-config` CLI；把路径传给 `run` | 不解析 Collector 内部 schema |
| `experiment_io.py` | 提供原子写 + digest + fsync + json 辅助；git provenance 硬化 | 不实现 Collector 业务 |

### 5.2 `ReferenceMuon` hook 设计

#### 新增状态

```python
self._snapshot_observer = None
self._snapshot_parameter_names: dict[int, str] = {}
self._snapshot_update: int | None = None
```

#### 新增/扩展方法

```python
def set_snapshot_observer(self, observer, parameter_names: dict[int, str]) -> None:
    self._snapshot_observer = observer
    self._snapshot_parameter_names = dict(parameter_names)

def set_training_update(self, update: int) -> None:
    # 保留现有 _mechanism_update 语义
    self._mechanism_update = int(update)
    self._snapshot_update = int(update)
```

说明：

- `train_platform.py:335-336` 已经在 step 前调用 `optimizer.set_training_update(update)`；扩展该方法即可，无需新增调用点。
- 如果实现时不想改 `set_training_update` 的语义，可新增 `set_snapshot_update(update)`，并在 `train_platform` 同一位置调用；二选一，本文推荐扩展 `set_training_update`，减少调用点。

#### step 内局部变量与 hook

在 `muon_reference.py:72-90` 做**仅重命名、不改数值**的调整：

```python
momentum_prev = state.get("muon_momentum", torch.zeros_like(p, dtype=torch.float32))
momentum = momentum_prev.mul(group["muon_momentum"]).add(grad)
direction = grad.add(momentum, alpha=group["muon_momentum"]) if group["muon_nesterov"] else momentum
ns_output = zeropower_newton_schulz(direction, group["muon_ns_steps"], group["muon_ns_coefficients"], group["muon_eps"])
updated = p.float().mul(1 - group["lr"] * group["weight_decay"]).add(ns_output, alpha=-group["lr"])
persisted = persist_state(momentum, group["state_simulation"], "muon_momentum", ...)
first_step = "muon_momentum" not in state

if self._mechanism_observer is not None:
    ...  # 保持现有调用不变
if self._diagnostic_observer is not None:
    ...  # 保持现有调用不变
if self._snapshot_observer is not None:
    self._snapshot_observer(
        update=self._snapshot_update,
        parameter_name=self._snapshot_parameter_names.get(id(p)),
        W_before=p,
        G=grad,
        M_prev=momentum_prev,
        M_t_candidate=momentum,
        M_t_persisted=persisted,
        D=direction,
        U=ns_output,
        W_after=updated,
        first_step=first_step,
    )

p.copy_(updated.to(p.dtype))
state["muon_momentum"] = persisted
```

关键点：

- **Hook 位于 `p.copy_` 之前**（`:89`），保证 `W_before` 还是 \(W_t\)。
- `W_after` 直接来自 `updated`，不是 `p`。
- `U` 直接来自 `ns_output`（原 `update` 局部变量），不是重算。
- `M_prev` 使用 `:72` 的局部变量，不在 observer 中重新 `state.get(...)`。
- `first_step` 在 `:90` 赋值前计算，反映本次是否首次使用动量。
- 现有 mechanism/diagnostic observer 的调用顺序和参数**保持不变**；Snapshot hook 放在它们之后，避免改变既有 artifact 的时序。
- 如果 `self._snapshot_update is None` 或 `parameter_name is None`，Collector 会记录错误并禁用；训练继续。

### 5.3 `train_platform` 调用点

#### 5.3.1 `run` 签名

```python
def run(
    cfg,
    run_dir,
    resume=None,
    max_wall_seconds=None,
    to_device="auto",
    stop_at_update=None,
    snapshot_config_path=None,
):
```

`main.py` 调用：

```python
summary = run(
    cfg, run_dir, resume, a.max_wall_seconds, a.to_device,
    stop_at_update=a.stop_at_update,
    snapshot_config_path=a.snapshot_config,
)
```

#### 5.3.2 启动阶段

在 `run` 最开始、`run_dir.mkdir` 之前：

```python
snapshot_config = (
    load_snapshot_config(snapshot_config_path, total_updates=cfg["derived"]["total_updates"])
    if snapshot_config_path is not None else SnapshotConfig.disabled()
)
```

- 配置非法 → 在创建 run_dir/训练前抛错（启动期错误）。
- 新建 run：写 `run_dir/snapshot_config.json`（规范化 config + hash），见 §6.4。
- resume：读取 `run_dir/snapshot_config.json`，与传入配置比对；规则见 §6.5。

#### 5.3.3 参数分组与名称映射

在 `groups, records = parameter_groups(...)`（`:186`）之后：

```python
muon_records = [r for r in records if r["group"] == "muon"]
muon_parameter_order = tuple(r["name"] for r in muon_records)

flattened = [p for g in groups for p in g["params"]]
name_by_id = {}
optimizer_param_index = {}
for index, p in enumerate(flattened):
    # 只映射 Muon 组参数；Muon 组在前，但仍按实际 groups 枚举
    for g in groups:
        if g.get("optimizer_group") == "muon" and p in g["params"]:
            name = next(r["name"] for r in muon_records if ...)
            ...
```

更清晰的实现：

```python
name_by_id = {}
optimizer_param_index = {}
running_index = 0
for group in groups:
    is_muon = group.get("optimizer_group") == "muon"
    for p in group["params"]:
        if is_muon:
            name = record_name_by_id[id(p)]   # 由 records 构建
            name_by_id[id(p)] = name
            optimizer_param_index[name] = running_index
        running_index += 1
```

`record_name_by_id` 可由 `records` 和 `model.named_parameters()` 构建，或直接遍历 `model.named_parameters()` 过滤 group；关键是使用实际 `groups` 的顺序，不重新定义筛选规则。

#### 5.3.4 优化器元数据

```python
muon_group = next(g for g in optimizer.param_groups if g.get("optimizer_group") == "muon")
optimizer_metadata = {k: v for k, v in muon_group.items() if k != "params"}
```

包含 `weight_decay`、`muon_momentum`、`muon_nesterov`、`state_simulation`、`state_quantization_granularity`、`state_quantization_block_size`、`muon_ns_steps`、`muon_ns_coefficients`、`muon_eps`、`betas` 等。JSON 序列化前过滤 tuple/list 中的非 JSON 类型（通常都是 float/int/bool/str）。

#### 5.3.5 创建 Collector

在 resume 块（`:195-200`）之后，`run_id`/`segment`/`completed` 已确定：

```python
collector = None
if snapshot_config.enabled or snapshots_exist(run_dir):
    context = SnapshotRunContext(
        run_id=run_id,
        segment_id=segment,
        recipe_fingerprint=cfg["fingerprint"],
        data_fingerprint=manifest["fingerprint"],
        snapshot_config_hash=snapshot_config.config_hash,
        model_config=cfg["model"],
        parameter_records=records,
        muon_parameter_order=muon_parameter_order,
        optimizer_param_index=optimizer_param_index,
        optimizer_metadata=optimizer_metadata,
        train_config=cfg["train"],
        data_config=cfg["data"],
        seeds={k: cfg["experiment"][k] for k in ("seed", "data_seed", "algorithm_seed")},
        source=source_info,          # run 开始采集一次
        environment=environment_info,
        device=str(device),
        precision=cfg["precision"],
    )
    collector = MuonSnapshotCollector(run_dir=run_dir, config=snapshot_config, context=context)
    collector.scan_existing(completed_updates=completed)
    optimizer.set_snapshot_observer(collector.observe, name_by_id)
```

说明：

- 如果 `snapshot_config.enabled == false` 且 run_dir 中已有 snapshots（例如 resume 一个曾经启用 Collector 的 run），仍创建 Collector 用于扫描/校验已有 Snapshot，但 `should_collect` 为 false；或者仅在 enabled 时创建。设计建议：**始终创建**一个轻量 Collector 来扫描/校验已有文件（即使本次禁用），以便 resume 时发现冲突/缺失；但它不采集新数据。实现时可拆分 `scan_committed_snapshots` 独立函数，避免创建完整 Collector。
- 如果本次禁用且没有已有 snapshots，可以不创建 Collector、不设置 hook。
- 设置 hook 用 `optimizer.set_snapshot_observer(collector.observe, name_by_id)`；`ReferenceMuon` 保存映射。

#### 5.3.6 循环内调用点

当前 `:319` 行拆分为：

```python
update = completed + 1
active_update = update
attempted_processed = processed
optimizer.zero_grad(set_to_none=True)

sampler_epoch_before, sampler_offset_before = sampler.epoch, sampler.offset
ids = sampler.take(cfg["train"]["micro_batch_size"] * cfg["train"]["accumulation_steps"])
sampler_epoch_after, sampler_offset_after = sampler.epoch, sampler.offset
micro_batch_window_ids = [
    ids[a * cfg["train"]["micro_batch_size"]:(a + 1) * cfg["train"]["micro_batch_size"]]
    for a in range(cfg["train"]["accumulation_steps"])
]

nll_sum = 0.0
n_tokens = 0
step_started = time.monotonic()
```

在 `:331` 之后、`:335` 之前：

```python
if collector is not None and collector.should_collect(update):
    collector.begin_update(SnapshotBeginContext(
        update=update,
        segment_id=segment,
        tokens_this_update=n_tokens,
        processed_target_tokens=processed + n_tokens,
        learning_rate=lr,
        pre_clip_grad_norm=grad_norm_value,
        clipped=clipped,
        grad_clip_norm=cfg["train"]["grad_clip_norm"],
        window_ids=tuple(ids),
        micro_batch_window_ids=tuple(tuple(x) for x in micro_batch_window_ids),
        sampler_epoch_before=sampler_epoch_before,
        sampler_offset_before=sampler_offset_before,
        sampler_epoch_after=sampler_epoch_after,
        sampler_offset_after=sampler_offset_after,
    ))
```

在 `:341` 之后：

```python
if collector is not None:
    collector.finish_update(update=update, train_nll=train_nll)
```

在 `run` 的 try/except 之外（或 `finally`）：

```python
finally:
    if collector is not None:
        collector.close()
```

注意：

- `begin_update` 在 `optimizer.step()` 之前；`finish_update` 在 `train_nll` 计算之后。
- `finish_update` 的磁盘 I/O 发生在 eval/checkpoint 之前；只影响时间，不影响数值。
- 如果 `train_nll` 非有限，`:341` 会抛 `NonFiniteMetricError`，`finish_update` 不执行；不会产生 Snapshot。
- 现有 `fidelity_observer.begin_update`/`mechanism_observer.begin_update` 和 `finish_update` 调用保持不变；Collector 不依赖它们。

### 5.4 时序图

```text
main.py
  │  --snapshot-config PATH
  ▼
train_platform.run
  │  1. load_snapshot_config / SnapshotConfig.disabled
  │  2. write/verify run_dir/snapshot_config.json
  │  3. build model, parameter_groups, optimizer
  │  4. source_snapshot / environment_snapshot (once)
  │  5. collector = MuonSnapshotCollector(...)
  │  6. collector.scan_existing(completed)
  │  7. optimizer.set_snapshot_observer(collector.observe, names)
  │
  ├─ loop update N:
  │    ├─ zero_grad
  │    ├─ sampler epoch/offset before
  │    ├─ ids = sampler.take(...)
  │    ├─ sampler epoch/offset after
  │    ├─ build micro_batch_window_ids
  │    ├─ forward/backward (accumulation)
  │    ├─ global clip_grad_norm_
  │    ├─ compute lr; set group lr
  │    ├─ collector.begin_update(context)          [before step]
  │    ├─ optimizer.set_training_update(N)          [before step]
  │    ├─ optimizer.step():
  │    │     ReferenceMuon, per Muon parameter:
  │    │       momentum_prev = state.get(...)
  │    │       momentum = μ*momentum_prev + grad
  │    │       direction = ...
  │    │       ns_output = NS(direction)
  │    │       updated = ...
  │    │       persisted = persist_state(momentum)
  │    │       existing observers...
  │    │       snapshot_hook(update=N, parameter_name, W_before=p,
  │    │                     G=grad, M_prev=momentum_prev,
  │    │                     M_t_candidate=momentum,
  │    │                     M_t_persisted=persisted,
  │    │                     D=direction, U=ns_output,
  │    │                     W_after=updated, first_step=...)
  │    │       p.copy_(updated)
  │    │       state["muon_momentum"] = persisted
  │    ├─ zero_grad
  │    ├─ train_nll = ...
  │    ├─ collector.finish_update(N, train_nll)      [validate + write]
  │    ├─ existing observer finish calls
  │    ├─ metrics / eval / checkpoint
  │
  └─ finally: collector.close()

Collector.finish_update:
  validate 30 params × 8 roles
  → disk precheck
  → write temp .pt → fsync → hash → replace
  → build manifest → write temp JSON → fsync → replace
  → mark committed
```

---

## 6. 配置文件与 CLI

### 6.1 CLI

`platform/src/main.py` 增加：

```python
p.add_argument("--snapshot-config", type=Path, default=None,
               help="optional sidecar JSON for Muon snapshot collection")
```

- 未提供：新 run 的 Collector 完全关闭。
- 提供：在训练开始前加载/校验；非法配置直接退出，不创建训练产物。
- 不把 Collector 配置写入 recipe，不改变 `scientific_fingerprint`。

### 6.2 Sidecar JSON 示例

```json
{
  "schema_version": 1,
  "enabled": true,
  "updates": [1, 8, 32, 128, 512, 1024, 2048, 4096],
  "on_error": "disable",
  "max_total_bytes": 34359738368,
  "min_free_bytes": 5368709120
}
```

### 6.3 字段类型、默认值、非法输入

| 字段 | 类型 | 默认值 | 规则 |
|---|---|---|---|
| `schema_version` | int | 1 | 必须为 1；缺失按 1 处理；其他值报错 |
| `enabled` | bool | false（无文件时） | 文件中必须显式给出；`enabled=false` 时 `updates` 必须为空 |
| `updates` | list[int] | `[]` | 每个为 1-based 绝对 update；正整数；不重复；`enabled=true` 时非空；必须 `<= total_updates` |
| `on_error` | string | `"disable"` | v1 仅允许 `"disable"`；其他值报错 |
| `max_total_bytes` | int 或 null | null | null 表示无上限；否则正整数；统计 `run_dir/snapshots/` 下已提交 `.pt` + `.manifest.json` 的大小 |
| `min_free_bytes` | int | `5368709120`（5 GiB） | 非负整数；写入前要求 `free >= estimate + min_free_bytes` |

非法输入处理：

- 文件不存在 / 不可读 / JSON 解析失败 → 启动期错误。
- 未知字段 → 启动期错误（严格 schema）。
- 类型错误（bool 当 int、字符串当 int 等）→ 启动期错误，错误信息包含字段路径。
- `updates` 重复、<=0、> total_updates → 启动期错误。
- `enabled=true` 且 `updates=[]` → 启动期错误（没有 landmark 没有意义）。
- `enabled=false` 且 `updates` 非空 → 启动期错误（避免误以为会采集）。
- `on_error` 非 `"disable"` → 启动期错误。
- `max_total_bytes <= 0`（非 null）→ 启动期错误。
- `min_free_bytes < 0` → 启动期错误。
- `updates` 顺序：接受任意顺序；规范化为升序 tuple；重复报错。
- 所有错误在 `run_dir` 创建/训练开始前抛出；不产生半成品 run。

### 6.4 配置规范化与 hash

规范化：

```python
normalized = {
    "schema_version": 1,
    "enabled": bool,
    "updates": sorted(unique_updates),  # tuple/list of int
    "on_error": "disable",
    "max_total_bytes": int | None,
    "min_free_bytes": int,
}
config_hash = sha256(json.dumps({
    "snapshot_config": normalized,
    "collector_format": "muon_snapshot",
    "collector_version": 1,
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
```

保存：

`run_dir/snapshot_config.json`：

```json
{
  "format": "muon_snapshot_config",
  "version": 1,
  "config_hash": "<sha256>",
  "config": { ... normalized ... }
}
```

- 新 run 总是写该文件（包括 disabled），使用 `atomic_write_json`。
- 该文件是 run-level metadata，不是 recipe，不参与 scientific fingerprint。
- 每个 Snapshot Manifest 的 `identities.snapshot_config_hash` 必须等于该 hash。

### 6.5 Resume 时的一致性校验

规则：

1. **新 run**：
   - 提供 `--snapshot-config` → 加载/校验/规范化；写 `snapshot_config.json`。
   - 未提供 → `SnapshotConfig.disabled()`；写 `snapshot_config.json`（`enabled=false`）。
2. **Resume**：
   - 读取 `run_dir/snapshot_config.json`。
   - 若提供 `--snapshot-config`：加载/校验/规范化；`config_hash` 必须与存储值一致；否则启动期错误。
   - 若未提供：使用存储的 config（包括 `enabled`/`updates`），保持该 run 的采集语义；如果存储文件缺失，Collector 禁用，并在 `snapshot_errors.jsonl` 记录 `stored_config_missing`（不中断训练）。
   - 已有 Manifest 的 `snapshot_config_hash` 必须等于当前 config hash；不一致 → 该 Manifest 标记为 conflict，不覆盖；记录错误。
3. **训练恢复逻辑不变**：
   - Collector 的启用/禁用、Manifest 存在/缺失、Snapshot 损坏，都**不得**改变 `completed`、`processed`、sampler、RNG、model/optimizer 的恢复。
   - Snapshot 不参与 `model.load_state_dict` / `optimizer.load_state_dict`。
   - Collector 初始化失败（除配置非法外）不应阻止训练；配置非法是启动期错误，训练不启动。

> 设计选择：resume 未显式提供 `--snapshot-config` 时，使用 run 内已记录的配置，而不是强制关闭。这样同一 run 的采集语义在分段训练中保持一致；显式提供时则必须 hash 匹配。若研究者希望“resume 未提供配置就关闭”，只需把规则 2 的“使用存储 config”改为“disabled”，不影响其他设计。

---

## 7. 文件存储与事务式写入

### 7.1 目录结构

```text
run_dir/
  snapshot_config.json              # run-level normalized config + hash
  snapshot_errors.jsonl             # append-only error log
  snapshot_status.json              # optional close-time summary
  snapshots/
    update_000001.pt
    update_000001.manifest.json
    update_000128.pt
    update_000128.manifest.json
    orphaned/                       # .pt without manifest, moved here on startup
    *.tmp                           # transient; cleaned on startup
  checkpoints/
    latest.pt
    update_000128.pt
  metrics.jsonl
  ...
```

- Snapshot 目录与 `checkpoints/` 完全分离。
- 不使用现有 `muon_momentum_snapshots/` 或 `mechanism/` 目录，避免 schema 混淆。
- `.pt` 文件名固定 `update_{update:06d}.pt`；Manifest 同名 `.manifest.json`。
- `update` 必须与文件内 `update` 一致；实现时只从文件名和 manifest 读取，不从 glob 猜。

### 7.2 `.pt` 与 Manifest 分工

| 内容 | `.pt` | Manifest |
|---|---|---|
| Tensor 数据 | 是 | 否 |
| Tensor 的 shape/dtype/numel/nbytes/checksum | Tensor 自带 shape/dtype；其余以 Manifest 为准 | 是 |
| run_id/update/segment_id/snapshot_id | 是（最小身份） | 是 |
| 训练上下文 | 否 | 是 |
| source/environment/model/parameter_records | 否 | 是 |
| alias 映射 | 否 | 是 |
| 文件级 SHA-256/大小 | 否 | 是 |
| 提交状态 | 否 | `status: "committed"` |

### 7.3 写入顺序（事务协议）

严格按以下顺序；任何一步失败都不写 Manifest：

1. **采集完成**：本 update 的 30 个参数 × 8 个 role 全部在内存 buffer 中；所有 Tensor 已 CPU FP32 clone。
2. **完整性校验**：§2.8 全部通过；否则失败。
3. **构造 payload**：`format/version/snapshot_id/run_id/update/segment_id/tensors`。
4. **磁盘预检**：
   - `estimate_pt = payload_nbytes + 1 MiB`（保守估计）；
   - `estimate_manifest = 512 KiB`（保守估计）；
   - `required = estimate_pt + estimate_manifest`；
   - `free = shutil.disk_usage(snapshots_dir).free`；
   - 若 `free < required + min_free_bytes` → 跳过，记录 `skipped_no_space`；
   - 若 `max_total_bytes` 非 null，且 `committed_bytes + required > max_total_bytes` → 跳过，记录 `skipped_max_total_bytes`。
5. **写临时 `.pt`**：`tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=snapshots_dir)`；`torch.save(payload, temp_path)`。
6. **flush/fsync**：打开临时文件，`flush()`，`os.fsync(fd)`；关闭。
7. **计算文件 SHA-256 和实际大小**：流式读取临时文件；`sha256_file(temp_path)`、`os.stat(temp_path).st_size`。
8. **构造最终 Manifest**：加入 `tensors_file.sha256/nbytes/payload_nbytes`；严格 JSON。
9. **`os.replace(temp_pt, final_pt)`**：同目录原子替换。
10. **fsync 目录**：best-effort `fsync_directory(snapshots_dir)`（Windows/WSL 可能不支持，吞掉 `OSError`）。
11. **写 manifest 临时文件**：`atomic_write_json` 内部同样临时文件 + flush/fsync + `os.replace`；Manifest 是提交点。
12. **fsync 目录**：best-effort。
13. **提交**：将 update 加入 `committed_updates`；清空 buffer。
14. **失败处理**：删除临时 `.pt`/临时 manifest；不写最终 Manifest；记录错误；禁用 Collector（除非是磁盘跳过）。

为什么 Manifest 最后写：

- `.pt` 先 rename，Manifest 后 rename。若崩溃在两者之间，只有 orphan `.pt`，没有 Manifest，不会被误认为有效 Snapshot。
- 若 Manifest 先写而 `.pt` 未 rename，会出现“有效 Manifest 指向不存在的 `.pt`”，更糟。
- 若 Manifest 写失败，`.pt` 成为 orphan；下次启动清理/隔离。

### 7.4 磁盘空间策略

- `min_free_bytes`：写入后必须保留的空闲空间，避免 Snapshot 挤占 Checkpoint。默认 5 GiB。
- `max_total_bytes`：所有已提交 Snapshot（`.pt` + manifest）总大小上限；null 表示无上限。
- 预检使用保守估计；如果实际文件比估计大很多（不太可能），记录 `estimated_size_exceeded` 警告，但保留已写文件（数据完整优先）。
- 磁盘不足时：**跳过该 Snapshot**，记录错误，训练继续；Collector 保持启用（后续 landmark 仍可尝试）。
- 如果 `shutil.disk_usage` 本身抛异常：记录 `disk_usage_error`，跳过，训练继续。
- Checkpoint 写盘失败是训练层问题，不由 Collector 处理；Collector 只保证不抢在 Checkpoint 前占满空间（通过 `min_free_bytes`）。

### 7.5 现有 `atomic_torch_save` 评估与最小补充

现有实现（`experiment_io.py:85-93`）：

```python
def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
```

评估：

| 需求 | 现有实现 | 结论 |
|---|---|---|
| 同目录临时文件 | 是 | 可复用 |
| `os.replace` 原子替换 | 是 | 可复用 |
| 失败删除临时文件 | 是 | 可复用 |
| flush/fsync 文件 | 否 | 需补充 |
| 计算文件 SHA-256/大小 | 否 | 需补充 |
| 原子写 JSON（Manifest） | 否 | 需新增 |
| fsync 目录 | 否 | 需新增（best-effort） |
| 孤儿清理 | 否 | Collector 启动时实现 |
| per-tensor checksum | 否 | Collector 实现 |

最小补充（建议放在 `experiment_io.py`）：

```python
def atomic_torch_save_with_digest(value, path: Path, *, fsync: bool = True) -> dict:
    """Write torch.save payload atomically; return {sha256, nbytes, path}."""

def atomic_write_json(path: Path, value, *, fsync: bool = True) -> None:
    """Strict JSON (allow_nan=False); temp + flush/fsync + os.replace."""

def sha256_file(path: Path) -> str: ...
def sha256_tensor(t: torch.Tensor) -> str: ...
def fsync_directory(path: Path) -> None:  # best-effort
    ...
def fsync_file(path: Path) -> None: ...
```

- 保留 `atomic_torch_save` 不变，供 Checkpoint 使用；Collector 使用新 helper。
- 不要把 Collector 的业务逻辑塞进 `experiment_io.py`；只放通用 I/O。
- `fsync_directory` 在 Windows/WSL 可能不支持 `O_DIRECTORY`；用 `try/except OSError` 吞掉，记录 `fsync_directory_unsupported`（可选 debug）。

### 7.6 重复运行、覆盖、损坏、孤儿规则

| 场景 | 行为 |
|---|---|
| 目标 Manifest 存在且校验通过 | **跳过**；不重新写；记录 `skipped_existing`（info） |
| 目标 Manifest 存在但校验失败 | **冲突**；不覆盖；记录 `existing_snapshot_corrupt`；该 update 不重新采集 |
| 目标 `.pt` 存在、Manifest 不存在 | 视为 orphan；移到 `snapshots/orphaned/`，记录；然后按正常流程写入（同 run/update） |
| 目标 `.pt` 和 Manifest 都不存在 | 正常写入 |
| 目标 `.pt` 存在且 Manifest 存在，但 `.pt` hash 不匹配 | 冲突；不覆盖；记录 |
| 临时 `.tmp` 存在 | 启动时删除；不视为 Snapshot |
| 重复 `begin_update` 同一 update | 记录 `duplicate_begin_update`；禁用 Collector（v1 不尝试恢复） |
| 重复 `finish_update` 同一 update | 第二次 no-op（已 committed）或记录 `duplicate_finish_update`；不覆盖 |
| 已有有效 Snapshot 的 update 再次被 `should_collect` | 返回 false；不采集 |
| 跨 run（run_id 不同）写入同一 run_dir | 不应发生；若发生，记录 `run_id_mismatch` 冲突，不覆盖 |

**不静默覆盖**：任何已存在的 Manifest 都不能被 `os.replace` 覆盖；实现时在写之前检查 `final_manifest.exists()` 并校验；只有确认是 orphan（`.pt` 无 manifest）才允许清理后重写。

### 7.7 Manifest 校验范围

校验通过的 Snapshot 必须满足：

1. `format == "muon_snapshot_manifest"`，`version == 1`，`status == "committed"`。
2. `snapshot_id == f"{run_id}:{update:06d}"`。
3. 文件名 `update_{update:06d}.manifest.json` 与内容 `update` 一致。
4. `run_id` 与当前 run 一致；`recipe_fingerprint`/`data_fingerprint` 与当前 run 一致。
5. `identities.snapshot_config_hash` 与当前 config hash 一致。
6. `tensors_file.path` 是 basename，等于 `update_{update:06d}.pt`；不得包含 `/`、`\`、`..`。
7. `.pt` 存在，`st_size == tensors_file.nbytes`，`sha256_file(.pt) == tensors_file.sha256`。
8. `torch.load(.pt, weights_only=True)` 成功；`format/version/run_id/update/segment_id/snapshot_id` 与 Manifest 一致。
9. `tensors` 的 key 集合等于 Manifest 中所有非 alias role 的 `tensor_id` 集合；无 orphan。
10. 每个 `.pt` Tensor 是 CPU FP32 2D；shape/dtype 与 Manifest 一致；`sha256_tensor` 与 Manifest 一致。
11. 30 个参数、8 个 role、alias 解析、shape/dtype/numel 全部通过。
12. `integrity` 计数与实际一致。

任何一项失败 → 该 Manifest 标记为 conflict；不视为 committed；不覆盖。

---

## 8. 异常处理与断点恢复

### 8.1 采集异常

硬性规则：

- `observe` 内部任何异常**不得传播到 `optimizer.step()`**。用 `try/except Exception` 捕获；记录错误；`self._disabled = True`；清空当前 buffer；return。
- `finish_update` 内部任何异常**不得传播到训练循环**。捕获；记录；禁用；清理临时文件；不写 Manifest；return。
- `close()` 不抛异常。
- 一旦 Collection 失败（除磁盘跳过/已有冲突外），本次 run 后续 Collector 禁用；不再采集。
- 不允许生成“成功” Manifest 的不完整 Snapshot。
- 训练继续运行；模型/优化器状态不受影响。

错误日志：

`run_dir/snapshot_errors.jsonl`，每行严格 JSON：

```json
{
  "timestamp": "2026-11-17T12:34:56Z",
  "run_id": "run-abc",
  "update": 128,
  "phase": "observe",
  "code": "observe_exception",
  "message": "RuntimeError: ...",
  "parameter_name": "transformer.h.0.attn.c_attn.weight",
  "required_bytes": null,
  "free_bytes": null,
  "details": {}
}
```

常用 `code`：

- `observe_exception`
- `wrong_update`
- `unknown_parameter`
- `duplicate_parameter`
- `tensor_contract_violation`
- `incomplete_snapshot`
- `nonfinite_train_nll`
- `skipped_no_space`
- `skipped_max_total_bytes`
- `disk_usage_error`
- `temp_write_error`
- `replace_error`
- `manifest_write_error`
- `existing_snapshot_corrupt`
- `existing_snapshot_conflict`
- `duplicate_begin_update`
- `duplicate_finish_update`
- `finish_without_begin`
- `incomplete_update_at_close`
- `stored_config_missing`
- `config_hash_mismatch`
- `run_id_mismatch`

### 8.2 状态机

Collector 顶层状态：

```text
DISABLED
  │ enabled & config valid
  ▼
IDLE ──should_collect(update)──► COLLECTING
  ▲                                  │
  │                                  ├─ observe × 30
  │                                  │
  │                                  ├─ finish_update success ──► COMMITTED ──► IDLE
  │                                  │
  │                                  ├─ validation/write error ──► FAILED (disabled)
  │                                  │
  │                                  └─ disk skip ──► IDLE (still enabled)
  │
  └──────────────── close() ◄────────┘
```

状态定义：

- `DISABLED`：配置关闭、或运行中因错误禁用、或已 close。
- `IDLE`：等待下一个 landmark。
- `COLLECTING`：`begin_update` 已调用，buffer 正在收集。
- `COMMITTED`：本 update 已成功写 Manifest；立即回到 `IDLE`。
- `FAILED`：本 update 失败且 Collector 被禁用；后续 update 不再采集。
- `SKIPPED`：本 update 因磁盘/已有冲突跳过；Collector 仍可用。

### 8.3 状态转换与异常场景表

| 场景 | 触发点 | Collector 状态 | 写 Manifest？ | 训练继续？ | 记录 |
|---|---|---|---|---|---|
| update 不是 landmark | `should_collect` false | IDLE | 否 | 是 | 无 |
| 正常成功 | `finish_update` 完成 | IDLE（committed） | 是 | 是 | Manifest |
| `observe` 异常 | `observe` | FAILED（disabled） | 否 | 是 | `snapshot_errors.jsonl` |
| 参数缺失 | `finish_update` 完整性 | FAILED | 否 | 是 | `incomplete_snapshot` + missing list |
| 参数重复 | `observe` | FAILED | 否 | 是 | `duplicate_parameter` |
| role 缺失 | `finish_update` | FAILED | 否 | 是 | `incomplete_snapshot` |
| shape/dtype 不符 | `observe`/`finish` | FAILED | 否 | 是 | `tensor_contract_violation` |
| update 不匹配 | `observe`/`finish` | FAILED | 否 | 是 | `wrong_update` |
| 磁盘低于 `min_free_bytes` | `finish` 预检 | IDLE（SKIPPED） | 否 | 是 | `skipped_no_space` |
| 超过 `max_total_bytes` | `finish` 预检 | IDLE（SKIPPED） | 否 | 是 | `skipped_max_total_bytes` |
| 临时 `.pt` 写失败 | `finish` | FAILED | 否 | 是 | `temp_write_error` |
| `os.replace` 失败 | `finish` | FAILED | 否 | 是 | `replace_error` |
| Manifest 写失败 | `finish` | FAILED | 否；`.pt` 成为 orphan | 是 | `manifest_write_error` |
| 已有有效 Snapshot | `finish` 预检 | IDLE（SKIPPED） | 否（保留已有） | 是 | `skipped_existing`（info） |
| 已有损坏 Snapshot | `finish`/扫描 | IDLE（CONFLICT） | 否 | 是 | `existing_snapshot_corrupt` |
| orphan `.pt` 无 Manifest | 启动扫描 | IDLE | 否（隔离） | 是 | `orphaned_tensor_file` |
| resume 时错过 landmark | 启动扫描 | IDLE | 否 | 是 | `missed_before_resume` |
| `begin` 后未 `finish` | 下一次 `begin`/`close` | IDLE/FAILED | 否 | 是 | `incomplete_previous_update` |
| `finish` 无 `begin` | `finish` | IDLE | 否 | 是 | `finish_without_begin` |
| 配置非法 | 启动加载 | 未创建 | 否 | **否（启动失败）** | 配置错误，抛出 |
| resume config hash 不匹配 | 启动加载 | 未创建 | 否 | **否（启动失败）** | `config_hash_mismatch`，抛出 |
| `close` 时仍在采集 | `close` | DISABLED | 否 | 运行结束 | `incomplete_update_at_close` |

### 8.4 Resume 扫描与校验

`scan_existing(completed_updates)`：

1. 清理 `snapshots_dir` 下 `*.tmp`（只删除 `.tmp`）。
2. 遍历 `update_*.pt`：若对应 `.manifest.json` 不存在，移到 `snapshots/orphaned/`，记录 `orphaned_tensor_file`。
3. 遍历 `update_*.manifest.json`：
   - 解析 JSON；校验 §7.7 全部项。
   - 成功 → `committed_updates.add(update)`。
   - 失败 → `conflicted_updates.add(update)`；记录 `existing_snapshot_corrupt`/`existing_snapshot_conflict`。
4. 对配置中的每个 landmark：
   - 若 `update <= completed_updates` 且不在 `committed_updates` 且不在 `conflicted_updates` → `missed_updates.add(update)`；记录 `missed_before_resume`。
   - 若 `update > completed_updates` → 正常等待采集。
5. `should_collect(update)`：
   - `enabled` 且 `update in config.updates` 且 `update not in committed_updates` 且 `update not in conflicted_updates` 且未 disabled。
6. 不修改 `completed`/`processed`/sampler/RNG/model/optimizer。

### 8.5 错过的 landmark

- 如果 `landmark < completed_updates` 且没有有效 Snapshot：**不能通过当前状态直接补采**。该 update 的 \(W_t/G/M/D/U\) 已不可得。
- 记录 `missed_before_resume`；在 `snapshot_status.json`/run summary 中标记 `snapshot_complete=false`。
- 训练继续；不重跑（重跑会改变后续轨迹）。
- 研究者可选择从更早的 Checkpoint 重跑以补齐，但这是显式的实验决策，不是 Collector 自动行为。
- 如果 `landmark == completed_updates` 且是 `stop_at_update` 触发的边界：resume 后不会重跑该 update；如果该 landmark 没有 Snapshot，也属于 missed；不能在 resume 后补采。

### 8.6 与训练恢复的隔离

- Snapshot 不参与 `model.load_state_dict` / `optimizer.load_state_dict` / `sampler.load_state_dict` / `restore_rng`。
- Snapshot 的存在、缺失、损坏、config hash 不匹配（Manifest 层面）不影响 resume 的模型/优化器/数据恢复；只有 **startup config 非法** 才阻止训练启动。
- `run_dir/snapshot_config.json` 是 run metadata，不是 Checkpoint 字段；不修改 `_checkpoint` schema。
- `collector.close()` 在异常路径也必须执行；但它不改变训练 status。

---

## 9. 正确性测试方案

> 本阶段只设计测试，不编写测试代码。

### 9.1 Optimizer 单元测试（逐 bit 不变性）

目标：证明 Collector 开启不改变优化器计算。

测试设置：

- 2–3 个小型 Muon 参数（例如 `(4,2)`、`(2,4)`、`(3,3)`），固定初始值。
- 固定梯度序列。
- `ReferenceMuon`，参数组 `optimizer_group="muon"`。
- 分别运行：
  - Collector disabled（不设置 hook）；
  - Collector enabled（设置 hook + collector，landmark 包含每一步）。
- 覆盖配置：
  - `state_simulation="none"`（`M_t_persisted is M_t_candidate`）；
  - `state_simulation="bf16_roundtrip"`（两者不同）；
  - `muon_nesterov=True` / `False`（后者 `D is M_t_candidate`）；
  - `weight_decay=0` / `0.1`。
- 比较：
  - 每一步每个参数的 `p`：`torch.equal`；
  - 每一步 `optimizer.state_dict()`：逐 tensor `torch.equal`，包括 `muon_momentum`；
  - `torch.get_rng_state()`：`torch.equal`；
  - 后续 `torch.rand(8)` 序列：`torch.equal`；
  - `optimizer.state_dict()` 的 key 集合、`param_groups`：完全一致；
  - collector 不出现在 `optimizer.state_dict()` 或 `optimizer.state[p]`。
- 额外断言：`collector` 的 `observe` 内没有 in-place 修改（可通过 clone 前后 checksum 或 monkeypatch 检查）。

### 9.2 Snapshot 数据正确性

测试设置：小型 `ReferenceMuon` + 2–3 个参数；landmark `[1,2]`；真实 step。

逐项断言：

1. **七类逻辑数据完整**：每个参数 8 个 role 都有值；alias 解析后 7 个逻辑类齐全。
2. **M_t 双角色**：
   - `state_simulation="none"`：`M_t_persisted` alias 到 `M_t_candidate`；只有一份 Tensor。
   - `state_simulation="bf16_roundtrip"`：两者独立；`M_t_persisted == M_t_candidate.bfloat16().float()`；分别有 checksum。
3. **W_before/W_after**：
   - 在 step 前 clone `p` 作为 expected W_before；step 后 clone `p` 作为 expected W_after；
   - Snapshot 中 `W_before`/`W_after` 与 expected `torch.equal`；
   - `W_after` 与 `updated` 一致（可通过 monkeypatch 捕获）。
4. **G**：
   - 手动对梯度做 `clip_grad_norm_` 后，Snapshot `G` 与 `p.grad` `torch.equal`；
   - 确认 `G` 是裁剪后的值（构造一个会被裁剪的梯度，比较缩放）。
5. **D**：
   - monkeypatch `zeropower_newton_schulz` 包装 production 函数，捕获第一个参数（输入 `D`）；
   - Snapshot `D` 与捕获输入 `torch.equal`；
   - 对 `muon_nesterov=False`，`D` 与 `M_t_candidate` 是同一对象/alias。
6. **U**：
   - 同一个 monkeypatch 捕获 production 返回值 `U`；
   - Snapshot `U` 与捕获返回值 `torch.equal`；
   - 确认 `U` 未乘 lr、未乘负号。
7. **shape/dtype/name**：
   - 每个 role 的 `shape` 等于参数 shape；
   - dtype 为 `float32`；
   - `parameter_name` 与 `muon_parameter_order` 中的名称一致；
   - `parameter_index`/`optimizer_param_index` 与实际 groups 一致。
8. **CPU clone 独立性**：
   - `observe` 之后，修改原始 `p`、`p.grad`、`optimizer.state[p]["muon_momentum"]`；
   - 已写入 buffer/文件的 Tensor 不变（checksum 不变）；
   - 证明 clone 不是引用。
9. **First step**：
   - update 1：`M_prev` 是全零 FP32 Tensor，`first_step=true`；
   - update 2：`M_prev` 等于 update 1 的 `M_t_persisted`。
10. **Alias 记录**：
    - `aliases` 列表与 role descriptor 一致；
    - alias 的 canonical role 存在；
    - `unique_tensor_count = role_slots - alias_count`。
11. **Checksum**：
    - 从 `.pt` 重新加载后逐 Tensor `sha256_tensor` 与 Manifest 一致；
    - 修改一个 Tensor 字节后重新计算 checksum 能检测到。
12. **30 矩阵完整性**（可用合成参数）：
    - 构造 30 个小型 Muon 参数（不需要真实 30M 模型）；
    - 完整采集 → success；
    - 少一个参数 → `incomplete_snapshot`，无 Manifest；
    - 多一个参数 → `unexpected_parameter`，无 Manifest；
    - 重复一个参数 → `duplicate_parameter`，无 Manifest；
    - 少一个 role → `incomplete_snapshot`。

### 9.3 训练循环集成测试

目标：证明 Collector 开启不改变训练轨迹和科学指标。

设置：

- CPU、FP32、`deterministic=true`。
- 短 horizon（例如 4–8 updates）。
- 相同 recipe、seed、data。
- 两个 run：
  - Collector disabled；
  - Collector enabled，landmarks 包含 1、中间 update、最后 update。
- 数据：使用现有小型 recipe（`recipes/diagnostic_cpu.json` 或 monkeypatch 的轻量 `run`，参考 `tests/test_nonfinite_divergence.py:44-63` 的 monkeypatch 方式），避免依赖外部数据。
- 比较：
  - `metrics.jsonl` 的 `train` 事件：`train_nll`、`pre_clip_grad_norm`、`lr`、`clipped`、`tokens_this_update`、`processed_target_tokens`；
  - 排除时间/吞吐/CUDA 峰值等非科学字段；
  - 最终 `checkpoints/final.pt` 的 `model` 和 `optimizer` state：逐 tensor `torch.equal`；
  - `summary.json` 的科学字段（completed_updates、processed_target_tokens、final NLL 等）。
- 期望：逐 bit 相等。
- 如果使用真实 recipe 且模型较小，Collector 的 `muon_parameter_count` 应按实际 groups（例如 2 层 × 5 = 10），不是硬编码 30；测试应覆盖“非 30”情况，证明完整性校验使用实际 groups。

### 9.4 Snapshot 与 Checkpoint 交叉验证

设置：

- 让 `checkpoint.every_updates` 与 landmark 对齐，或使用 `stop_at_update` 在 landmark 边界保存 Checkpoint。
- 同一 update 同时存在 Snapshot 和 Checkpoint。

校验：

1. `Snapshot.parameters[name].roles["W_after"]` 与 `checkpoint["model"][name]` `torch.equal`。
2. `Snapshot.parameters[name].roles["M_t_persisted"]` 与 `checkpoint["optimizer"]["state"][optimizer_param_index]["muon_momentum"]` `torch.equal`。
3. `Snapshot.update == checkpoint["completed_updates"]`。
4. `Snapshot.run_id == checkpoint["run_id"]`。
5. `Snapshot.identities.recipe_fingerprint == checkpoint["scientific_fingerprint"]`。
6. `Snapshot.identities.data_fingerprint == checkpoint["data_fingerprint"]`。
7. 如果 Checkpoint 在 Snapshot 之后保存，eval 不改变参数；W_after 仍应一致。
8. 如果 Checkpoint 保存失败而 Snapshot 成功，交叉验证跳过该 update，记录 `checkpoint_missing`；不影响 Snapshot 有效性。

### 9.5 异常测试

| 测试 | 注入方式 | 期望 |
|---|---|---|
| 磁盘空间不足 | monkeypatch `shutil.disk_usage` 返回低 free | 无 Manifest；`skipped_no_space`；训练参数正常更新；Collector 仍 enabled |
| 超过 `max_total_bytes` | 配置小上限；预置已提交文件 | 无新 Manifest；`skipped_max_total_bytes`；训练继续 |
| Tensor 缺失 | 阻止一个参数 `observe`（monkeypatch hook 或合成 groups） | `incomplete_snapshot`；无 Manifest；Collector disabled；训练继续 |
| 重复 update | 连续 `begin_update` 同一 update，或重复 `observe` | `duplicate_begin_update`/`duplicate_parameter`；无 Manifest；disabled |
| Snapshot 文件损坏 | truncate `.pt` 或翻转字节 | 扫描标记 conflict；不覆盖；`existing_snapshot_corrupt` |
| Manifest 缺失 | 只保留 `.pt` | 启动扫描移到 `orphaned/`；不视为 Snapshot |
| 写入过程异常 | monkeypatch `torch.save`/`os.replace`/`open` 抛异常 | 无 Manifest；临时文件清理；`temp_write_error`/`replace_error`；Collector disabled；训练完成 |
| Manifest 写失败 | monkeypatch `atomic_write_json` 抛异常 | `.pt` 成为 orphan；无 Manifest；`manifest_write_error`；下次启动隔离 |
| resume 后重复采集 | 先跑到 landmark 并提交，再 resume | 已 committed 的 landmark 跳过；Manifest 内容/哈希不变 |
| resume 后错过 landmark | 从更晚 Checkpoint resume，landmark 在 `completed` 之前且缺失 | 记录 `missed_before_resume`；不补采；训练继续 |
| config hash 不匹配 | 修改 sidecar 后 resume | 启动期错误；训练不启动 |
| 配置非法 | 未知字段/类型错误/updates 重复 | 启动期错误；不创建 run 产物 |
| `close` 时未完成 | `begin_update` 后直接 `close` | 记录 `incomplete_update_at_close`；无 Manifest；不抛异常 |

### 9.6 测试文件与覆盖矩阵

| 文件 | 内容 |
|---|---|
| `platform/tests/test_snapshot_invariance.py` | 9.1 optimizer 逐 bit；9.3 训练循环集成；Collector 不在 state_dict |
| `platform/tests/test_snapshot_collector.py` | 9.2 数据正确性；alias；checksum；30 矩阵完整性；schema 校验 |
| `platform/tests/test_snapshot_resume.py` | 9.4 checkpoint 交叉验证；9.5 resume/损坏/orphan/config hash |
| 现有 `platform/tests/test_muon_reference.py` | 回归：Muon 数值不变（hook 关闭时） |
| 现有 `platform/tests/test_muon_update_fidelity.py` | 回归：现有 fidelity observer 不受影响 |
| 现有 `platform/tests/test_muon_mechanism.py` | 回归：现有 mechanism observer 不受影响 |
| 现有 `platform/tests/test_resume.py` | 回归：Checkpoint resume 不变 |

---

## 10. 前置工程问题与实现依赖

### 10.1 A. Git provenance

现状（`platform/src/experiment_io.py:39-42`）：

```python
def source_snapshot(root: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True).stdout.strip()
    return {"remote": git("remote", "get-url", "origin"),
            "upstream_commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "diff": git("diff", "--", "src", "scripts", "recipes", "tests", "docs")}
```

问题：

- `check=True`；没有 origin 时 `git remote get-url origin` 抛 `CalledProcessError`。
- 当前 WSL 环境还报告 dubious ownership；git 可能拒绝访问。
- `run()` 在 `train_platform.py:222`（新 run 的 `source.json`）、`:357`（fidelity snapshot metadata）、`:413`（summary）调用；任何一次抛错都可能中断训练。
- 审计确认：`platform/` 没有独立 `.git`；git 向上找到 `/mnt/d/muon_research/.git`；当前仓库没有 `origin` remote。

最小修复要求（单独任务，建议在 Collector 实现前完成）：

1. `source_snapshot` 不因缺少 origin 抛错：
   - `remote`：失败 → `None`；
   - `upstream_commit`：失败 → `None`；
   - `dirty`：失败 → `None`；
   - `diff`：失败 → `None`；
   - 新增 `provenance_status`：`"ok"` / `"remote_missing"` / `"git_unavailable"` / `"ownership_error"` / `"unknown"`。
2. `git` 命令逐项捕获 `CalledProcessError`/`FileNotFoundError`/`OSError`；返回结构化结果，不 `raise`。
3. 保留现有字段名；新增字段向后兼容；`source.json` 仍写完整 diff（如果可用），Collector 只存 `diff_sha256`。
4. WSL dubious ownership：
   - 文档说明在宿主机/容器中配置 `git config --global --add safe.directory <path>`；
   - 或在 `source_snapshot` 中使用 `git -c safe.directory=<root> ...`（只读、局部配置）；
   - 不在代码中静默信任任意路径；记录 `provenance_status`。
5. Git provenance 失败**不得中断训练**；Snapshot Manifest 仍可写，`source` 字段为 null + status。

与 Collector 的依赖：

- Manifest 的 `source.commit`/`dirty`/`diff_sha256` 来自 `source_snapshot`。
- 如果不在前置任务修复，Collector 可以在自己的调用中 catch 异常并写 null，但 `train_platform` 在 `:222`/`:413` 的其他调用仍会失败；因此必须在启动参考训练前修复。

### 10.2 B. KeyboardInterrupt 与 Checkpoint

现状（`train_platform.py:386-387`, `:401-403`）：

- `KeyboardInterrupt` 只设置 `status="interrupted"`。
- 退出循环后，`if status != "diverged_nonfinite" and (completed > 0 or status == "paused_budget")` 会保存当前内存中的 checkpoint。
- 此时 `completed` 还是上一个 update 的编号；`sampler.take` 已前进 offset；`optimizer.step()` 可能只完成了一部分参数/状态更新。
- 保存的 checkpoint 标记为 completed=N，但模型/优化器是“第 N+1 次 update 做了一半”，sampler 已跳到下一批。resume 会不一致。
- `benchmark_suite.artifact_status` 把 `interrupted` 当作 paused（`platform/src/benchmark_suite.py:183` 附近），可能自动 resume 这个不完整 checkpoint。

最小修复方案（单独任务，建议在 Collector 实现前完成）：

1. `KeyboardInterrupt` 时**不保存**当前内存中的中间状态。
2. 只保留上一个完整 update 边界的有效 Checkpoint（`latest.pt` 由 `:381` 在边界写入）。
3. resume 从最后一个完整 Checkpoint 继续；`completed`/`processed`/sampler/RNG 均对应同一完整 update。
4. 正常 `max_wall_seconds` pause 和 `stop_at_update` 仍在完整 update 边界保存（`:318`, `:380-385`）。
5. 结束块的条件改为排除 `interrupted`（例如 `if status == "paused_budget" or (completed > 0 and status not in {"interrupted","diverged_nonfinite"})`，具体实现由该任务决定）。
6. `artifact_status` 对 `interrupted` 的 resume 目标应是最后一个完整 `latest.pt`；可增加 `checkpoint["completed_updates"]` 校验。
7. 新增测试：模拟 `KeyboardInterrupt` 在 forward/backward/step 中抛出；断言 `latest.pt` 仍是上一个边界，`completed_updates` 与 sampler 状态一致；resume 与不中断轨迹一致。

与 Collector 的依赖：

- Collector 的 Snapshot 在完整 update 的 `finish_update` 写入；如果 mid-update checkpoint 被保存并在 resume 时使用，训练可能重跑/跳过 update，Snapshot 与轨迹不对应。
- 在启用 Collector 的分段训练前，必须先修复该问题。
- 该修复不改变正常边界 checkpoint 的 schema；不需要 Collector 参与。

### 10.3 C. FP32 Reference 启动前配置核查清单

以下两项由上一轮审计发现，本任务**不修改**，但在启动 FP32 Reference 训练前必须由研究者确认：

1. **辅助 AdamW weight decay**：
   - `parameter_groups` 在 `train_platform.py:125-127` 给 Muon 和 `auxiliary_adamw` 两个组都设置 `weight_decay`。
   - 因此 1D RMSNorm gain 和 tied embedding 也承受 `weight_decay=0.1`。
   - 对比：AdamW 路径按 `ndim>=2` 分为 `decay`/`no_decay`，1D 参数 `weight_decay=0.0`（`train_platform.py:110`, `:128`）。
   - 需确认这是否是 Reference Muon 的有意设计；如果不是，应在训练启动前修正分组，并重新确定参考轨迹/fingerprint。
2. **`eval.attention_backend`**：
   - `model_namespace` 只使用 `cfg["precision"]["attention_backend"]`（`train_platform.py:90-93`）。
   - `eval.attention_backend` 只在 recipe 校验中被检查（`config/recipe.py:160`），训练/评估都使用同一个 model 的 backend。
   - 如果 recipe 写 `"eval": {"attention_backend": "math"}`，它不会切换 eval 到 math。
   - 需确认是否要修正 `evaluate` 使用 eval backend，或删除/澄清该配置字段；这会改变 eval 数值，不改变训练。

这两项不是 Collector 的实现依赖，但会影响 FP32 Reference 轨迹和评估结果；应列入启动前 checklist，并由研究者在后续配置任务中决定。

---

## 11. 后续实现文件清单

| 文件 | 类型 | 修改目的 | 具体接口/内容 | 依赖 |
|---|---|---|---|---|
| `platform/src/optim/muon_reference.py` | 修改 | 提供真实计算 Tensor 的 hook | `set_snapshot_observer(observer, parameter_names)`；扩展 `set_training_update`；保留 `momentum_prev` 局部；`ns_output` 命名；在 `p.copy_` 前调用 hook | Collector API |
| `platform/src/optim/muon_snapshot_collector.py` | 新增 | Collector 核心 | `SnapshotConfig`、`SnapshotRunContext`、`SnapshotBeginContext`、`MuonSnapshotCollector`、`load_snapshot_config`、schema/校验/扫描辅助 | `experiment_io` 新 helper |
| `platform/src/train_platform.py` | 修改 | 集成 Collector | `run(..., snapshot_config_path=None)`；加载/校验配置；写/校验 `snapshot_config.json`；创建 Collector；设置 hook；捕获 sampler epoch/offset；构造 window 分组；`begin_update`/`finish_update`/`close`；传上下文 | Collector API、配置 |
| `platform/src/main.py` | 修改 | CLI | `--snapshot-config PATH`；传给 `run` | `run` 新参数 |
| `platform/src/experiment_io.py` | 修改 | 通用原子 I/O + provenance | `atomic_torch_save_with_digest`、`atomic_write_json`、`sha256_file`、`sha256_tensor`、`fsync_file`、`fsync_directory`；`source_snapshot` 容错（前置任务 A） | 无 |
| `platform/tests/test_snapshot_collector.py` | 新增 | 数据/Schema/异常测试 | §9.2、§9.5 的数据正确性、alias、checksum、完整性、损坏、磁盘、写入异常 | Collector |
| `platform/tests/test_snapshot_invariance.py` | 新增 | 轨迹一致性 | §9.1 optimizer 逐 bit；§9.3 训练循环集成 | Collector + ReferenceMuon hook |
| `platform/tests/test_snapshot_resume.py` | 新增 | Resume/交叉验证 | §9.4 checkpoint 交叉验证；§9.5 resume/orphan/config hash | Collector + `run` 集成 |
| `platform/docs/SNAPSHOT_COLLECTOR_DESIGN.md` | 新增 | 本文档 | 设计依据 | 无 |

不修改：

- `platform/src/optim/muon_mechanism.py`：保持现有 schema/行为。
- `platform/src/optim/muon_update_fidelity.py`：保持现有 schema/行为。
- `platform/src/optim/muon.py`：旧实现，不在当前训练调用链。
- `platform/src/models/`、`platform/src/data/`：不需要改。
- 现有 recipes/benchmarks：Collector 配置走 sidecar，不改 recipe。
- 现有 README/其他 docs：本任务只新增本文档；操作手册可后续单独更新。

---

## 12. 实现顺序与依赖关系

### Phase 0：前置工程修复（必须先于长训练）

1. **Git provenance 硬化**（§10.1）
   - 修改 `experiment_io.source_snapshot`，origin 缺失/ownership 错误不抛。
   - 增加测试：无 origin、git 不可用、dirty/commit 可用。
   - 验收：新 run 在无 origin 时能写 `source.json`/`summary.json`，`provenance_status` 正确。
2. **KeyboardInterrupt checkpoint 修复**（§10.2）
   - `interrupted` 不保存 mid-update 状态；只保留最后完整边界。
   - 增加测试：中断前后 `latest.pt` 的 `completed_updates`/sampler 一致。
   - 验收：中断后 resume 与不中断轨迹一致（小模型）。
3. **配置核查**（§10.3）
   - 研究者确认 aux weight decay 与 eval attention backend；如需要，单独配置任务修改。
   - 验收：FP32 Reference recipe 的预期轨迹/评估语义明确。

### Phase 1：通用 I/O helper

- `experiment_io` 新增 `atomic_torch_save_with_digest`、`atomic_write_json`、`sha256_file`、`sha256_tensor`、`fsync_*`。
- 单元测试：临时文件清理、hash 正确、JSON 严格、fsync 容错。
- 不改变现有 `atomic_torch_save` 行为。

### Phase 2：配置 loader/validator

- `muon_snapshot_collector.py` 中的 `SnapshotConfig`/`load_snapshot_config`。
- 测试：默认值、合法/非法 JSON、未知字段、类型错误、updates 边界、config hash 稳定性。
- 验收：非法配置在训练前失败；disabled 配置正确。

### Phase 3：ReferenceMuon hook

- 添加 `set_snapshot_observer`/扩展 `set_training_update`；重命名局部变量；插入 hook。
- 测试：hook 关闭时 `test_muon_reference.py` 全部通过；hook 开启时参数/state/RNG 与关闭逐 bit 一致（可以先用一个记录型 dummy observer）。
- 验收：`W_before/G/M_prev/M_t_candidate/M_t_persisted/D/U/W_after/first_step` 全部与生产计算一致；`U` 是 NS 返回，`W_after` 是 `updated`。

### Phase 4：Collector 核心（内存）

- 实现 `observe`：clone、alias、descriptor、checksum、完整性。
- 实现 `begin_update`/`should_collect`/`close`（先不写盘，或写内存结果）。
- 测试：§9.2 的数据正确性、alias、first step、CPU clone 独立性、30 矩阵完整性。
- 验收：所有 role/alias/shape/dtype/checksum 正确；无 I/O。

### Phase 5：事务写盘、Manifest、错误处理

- 实现 `finish_update`：完整性 → 磁盘预检 → 临时 `.pt` → fsync → hash → replace → Manifest → fsync → commit。
- 实现错误日志、禁用、磁盘跳过、冲突/孤儿规则。
- 测试：§9.5 磁盘、写入异常、损坏、orphan、Manifest 缺失、重复 update。
- 验收：无半成品被当作有效；失败不中断训练；Manifest 是唯一提交标记。

### Phase 6：train_platform 集成

- `run` 新参数；配置加载/写入/校验；创建 Collector；设置 hook；采样上下文；`begin/finish/close`。
- 测试：§9.3 训练循环 off/on 逐 bit；窗口 ID/epoch/offset 正确；`train_nll` 传递正确。
- 验收：Collector 开启/关闭的最终模型、优化器、科学指标一致；Collector 不改变 Checkpoint。

### Phase 7：Resume 扫描与交叉验证

- `scan_existing`：清理 `.tmp`、隔离 orphan、校验 Manifest、committed/conflicted/missed。
- resume config hash 一致性。
- 测试：§9.4 checkpoint 交叉验证；§9.5 resume 重复/错过/损坏/config mismatch。
- 验收：已提交 landmark 不重复写；损坏不覆盖；错过 clear 记录；训练 resume 语义不变。

### Phase 8：文档与端到端验收

- 更新操作手册/README（后续单独任务；本任务不修改）。
- 小模型 CPU 端到端：4–8 updates，landmarks 1/中间/最后，Collector off/on 对比；Snapshot/Checkpoint 交叉验证；错误注入。
- 真实 30M FP32 Muon 参考训练前：完成 Phase 0 配置核查、磁盘预算、landmark 列表。

### 依赖关系图

```text
Phase 0A (git) ──┐
Phase 0B (interrupt) ──┤
Phase 0C (config audit) ──┤
                          ▼
Phase 1 (I/O helper) ──► Phase 5 (transaction)
Phase 2 (config) ──────► Phase 6 (train_platform)
Phase 3 (hook) ────────► Phase 4 (collector core) ──► Phase 5 ──► Phase 6 ──► Phase 7 ──► Phase 8
```

关键路径：Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7。Phase 0 与 Phase 1/2 可并行，但 Phase 0 必须在长训练前完成。

### 验收标准（整体）

1. `test_snapshot_invariance.py` 全部通过：Collector off/on 的参数、optimizer state、RNG、随机序列、最终 checkpoint 逐 bit 一致。
2. `test_snapshot_collector.py` 全部通过：七类/八 role、M_t 双角色、alias、first step、CPU clone 独立性、30 矩阵完整性、checksum、损坏检测。
3. `test_snapshot_resume.py` 全部通过：checkpoint 交叉验证、已提交跳过、损坏冲突、orphan、missed、config hash。
4. 失败注入测试：磁盘不足、写入异常、Tensor 缺失、重复 update、Manifest 缺失、文件损坏；训练均能继续（配置非法除外）。
5. 现有 `test_muon_reference.py`、`test_muon_update_fidelity.py`、`test_muon_mechanism.py`、`test_resume.py` 全部通过；现有 observer schema 未改变。
6. 新 run 在没有 git origin 时仍能写 snapshot/summary（Phase 0A）。
7. `interrupted` 不再保存 mid-update checkpoint；resume 从最后完整边界继续（Phase 0B）。
8. Snapshot 目录中不存在有效 Manifest 指向缺失/损坏 `.pt` 的情况；orphan 被隔离。

---

## 13. 需要研究者最终确认的事项

以下仅列真正无法从现有源码/任务信息确定的事项；其余设计选择已在文中明确。

1. **landmark update 列表**：`updates` 的具体数值。本文示例 `[1, 8, 32, 128, 512, 1024, 2048, 4096]` 只是占位，不是研究决策。需要研究者根据 30M FP32 Muon 参考训练计划确定。
2. **磁盘预算**：`max_total_bytes` 和 `min_free_bytes` 的实际值。本文建议 `max_total_bytes=null` 或 32 GiB、`min_free_bytes=5 GiB`；需要按实际磁盘/Checkpoint 大小确认。
3. **Resume 时未显式提供 `--snapshot-config` 的行为**：本文设计为“使用 run 内已记录的 config”；如果研究者希望“未提供就关闭”，需要改这一条。两种都不影响训练恢复逻辑。
4. **`algorithm_seed` 是否纳入 Manifest**：本文建议纳入 provenance；当前训练未使用它，若研究者不希望记录未使用字段，可删除。
5. **`max_total_bytes` 是否统计 Manifest 文件**：本文设计为统计 `.pt` + `.manifest.json`；若只统计 `.pt`，需要调整预检。

---

## 附录 A：关键源码位置索引

| 主题 | 位置 |
|---|---|
| `ReferenceMuon.__init__` | `platform/src/optim/muon_reference.py:34-50` |
| `set_diagnostic_observer` / `set_mechanism_observer` / `set_training_update` | `platform/src/optim/muon_reference.py:52-60` |
| `ReferenceMuon.step` | `platform/src/optim/muon_reference.py:62-97` |
| `momentum_prev` / `momentum` / `direction` / NS / `updated` / `persisted` | `platform/src/optim/muon_reference.py:72-81` |
| 现有 mechanism/diagnostic observer 调用 | `platform/src/optim/muon_reference.py:82-88` |
| `p.copy_` / state 写回 | `platform/src/optim/muon_reference.py:89-90` |
| `parameter_groups` | `platform/src/train_platform.py:106-128` |
| Muon eligible 规则 | `platform/src/train_platform.py:113` |
| `_checkpoint` | `platform/src/train_platform.py:162-166` |
| `run` 签名/初始化 | `platform/src/train_platform.py:169-200` |
| 新 run metadata | `platform/src/train_platform.py:201-222` |
| 训练循环 | `platform/src/train_platform.py:317-385` |
| zero_grad / sampler.take | `platform/src/train_platform.py:319` |
| forward/backward | `platform/src/train_platform.py:320-326` |
| clip / lr | `platform/src/train_platform.py:329-331` |
| observer begin / set_training_update | `platform/src/train_platform.py:332-336` |
| step / zero_grad | `platform/src/train_platform.py:337` |
| `train_nll` | `platform/src/train_platform.py:341` |
| checkpoint 写 | `platform/src/train_platform.py:380-381` |
| 异常处理 | `platform/src/train_platform.py:386-397` |
| 结束 checkpoint | `platform/src/train_platform.py:401-403` |
| `source_snapshot` | `platform/src/experiment_io.py:39-42` |
| `atomic_torch_save` | `platform/src/experiment_io.py:85-93` |
| `EventWriter` | `platform/src/experiment_io.py:45-53` |
| `rng_state` / `restore_rng` | `platform/src/experiment_io.py:56-82` |
| Recipe 字段/指纹 | `platform/src/config/recipe.py:16-31`, `:199-201` |
| sampler | `platform/src/data/frozen_tokens.py:99-127` |
| Llama MLP 参数名 | `platform/src/models/llama.py:79-95` |
| Llama attention 参数名 | `platform/src/models/base.py:37-39` |
| 现有 mechanism observer | `platform/src/optim/muon_mechanism.py:27-208` |
| 现有 fidelity observer | `platform/src/optim/muon_update_fidelity.py:162-212` |

## 附录 B：实现时最容易出错的点

1. **不要用 `state.get("muon_momentum")` 在 observer 里重新取 \(M_{t-1}\)**：必须在 `:72` 保留局部变量并传入；否则在某些实现下可能拿到更新后的值或新建零张量。
2. **不要用 `p` 作为 `W_after`**：`p` 在 `:89` 被原地覆盖；`W_after` 必须用 `updated`。
3. **不要重算 \(U\)**：必须用 NS 的返回值 `ns_output`；重算可能因实现/设备产生 bit 级差异。
4. **不要把 `M_t_candidate` 和 `M_t_persisted` 混为一个字段**：`state_simulation=="none"` 时它们恰好同一对象，但语义不同；Schema 必须保留两个 role。
5. **不要按数值相等做 alias**：只按对象身份 `is`；数值相等但对象不同必须分别保存。
6. **不要在 `observe` 里写盘**：所有 I/O 放到 `finish_update`；避免 step 内异常/长停顿。
7. **不要让 `observe`/`finish_update` 抛异常**：用 `except Exception` 捕获并禁用；不要捕获 `KeyboardInterrupt`/`SystemExit`。
8. **不要先写 Manifest 再写 `.pt`**：Manifest 是提交标记，必须最后写。
9. **不要静默覆盖已有有效 Manifest**：先校验；有效则跳过；损坏则冲突。
10. **不要把 Collector 配置写进 recipe**：会改变 scientific fingerprint；使用 sidecar。
11. **不要在 resume 时用 Snapshot 恢复模型/优化器**：Snapshot 不参与恢复。
12. **不要硬编码 30**：使用实际 `groups`/`parameter_records`；测试也要覆盖非 30 情况。
