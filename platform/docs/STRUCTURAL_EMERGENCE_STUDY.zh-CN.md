# Muon Structural Emergence Study（30M FP32 Reference Muon，seed 0/1，8192 updates）

> 本文档是 Structural Emergence Study 的实验准备交付物：统一 Recipe、数据集、
> 学习率调度、三档 Collector、Schema/Mapping、暂停/归档/恢复流程和可直接复制的
> Kaggle Cells。**准备完成后停止，由研究者执行正式双卡训练。**

本阶段只研究 FP32 Reference Muon 参考轨迹中的结构涌现；不包含 Hessian、Loss
Landscape、量化、扰动、低精度实验，也不包含事件触发 Snapshot 或在线 SVD。

---

## 0. 冻结事实

| 项目 | 固定值 |
|---|---|
| 代码仓库 | `https://github.com/SiennaDai/muon_research` |
| 代码目录 | `/kaggle/working/muon_research` |
| 平台目录 | `/kaggle/working/muon_research/platform` |
| 数据挂载 | `/kaggle/working/muon_data` |
| Kaggle Dataset | `siennadai/lowp-benchmark-slimpajama-8x-v1`（private） |
| Dataset 内布局 | `slimpajama_8x/{manifest.json,train.bin,validation.bin,test.bin}` |
| Dataset fingerprint | `ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489` |
| 训练集 tokens | 70,000,418（validation 68,296，test 68,602） |
| Recipe（seed 0） | `recipes/structural_emergence_30m_fp32_muon_s0.json` |
| Recipe fingerprint（s0） | `d28b230454b53fdffa0a4dfa45672e823f326d0cb40f8260708b857c51e5a285` |
| Recipe（seed 1） | `recipes/structural_emergence_30m_fp32_muon_s1.json` |
| Recipe fingerprint（s1） | `398597e149a2d69527f17617b55de8524480788f1cc67e9b33ab91a8f3a76fc2` |
| Snapshot sidecar | `configs/muon_snapshot_structural_18nodes.json` |
| Snapshot config hash | `06bfd036619198750d29c952e261d479ccbf9bca495155d999d5983231819e9c` |
| Structural sidecar | `configs/structural_emergence_8192.json` |
| Structural config hash | `fa5f9f9e8e9051cf1208185adf7ec5c2ac82d4f4d1b38cd2ea0430c4347bc29e` |
| 模型 | 29,938,560 参数；6 layers / 384 hidden / 6 heads / head_dim 64 |
| 优化器 / 精度 | `reference_muon` / FP32，`state_simulation=none` |
| seeds | model 0、1；data 1337；algorithm 2026 |
| tokens_per_update | 8192（micro 4 × accum 8 × seq 256） |
| total_updates | 8192 |
| 训练 tokens | 67,108,864 |
| Checkpoint | 单文件滚动 `checkpoints/latest.pt`，每 128 updates 原子覆盖 |
| Snapshot 节点 | `1, 8, 32, 48, 56, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192` |

正式训练前把上面的 `<FIXED_COMMIT_SHA>` 替换为本次交付的实际 commit。

---

## 1. 数据集

### 1.1 构建结果

新数据集由 `platform/scripts/prepare_slimpajama_stream.py` 构建。它沿用
`prepare_data.py --kind slimpajama` 的源、revision、tokenizer、去重、split
算法和 manifest schema，但以流式方式写 token，内存占用与数据规模无关。

构建参数（固定）：

```bash
python scripts/prepare_slimpajama_stream.py \
  --output data/slimpajama_8x \
  --source-revision b5f90f419b7489cdba26fdbc8c022fcb5562f968 \
  --source-max-bytes 6442450944 \
  --train-tokens 70000000 --validation-tokens 65537 --test-tokens 65537
```

结果：

- train：70,000,418 tokens，sha256 `8d6ad484891920204f5997c00c19dd18117de424b6eed2a17a2632e9602bc6cb`
- validation：68,296 tokens，sha256 `fefd7f1b52d8c90f04bbdc3e1c9206497db827c2a0cbbfbd8d7e940f71be96d9`
- test：68,602 tokens，sha256 `0196dad2e03f0bb519cf7b18b751aa93ce77a35732851b38782e70c95c690f9f`
- manifest：9.5 MB，66742 documents，fingerprint `ba3ac442...`

**与 Exploratory Study 的关系**：validation/test 与 `slimpajama_4x` 的对应
split **逐字节相同**；train 是 4x train 流的**逐字节前缀扩展**（前
33,594,156 tokens 完全一致）。因此 4x 上的既有分析与新数据的早期节点可直接对齐，
但两者的 dataset fingerprint 不同，实验记录必须使用新 fingerprint。

### 1.2 Kaggle 发布与挂载

已发布为 private dataset：
`https://www.kaggle.com/datasets/siennadai/lowp-benchmark-slimpajama-8x-v1`

上传目录结构：

```text
lowp-benchmark-slimpajama-8x-v1/
├── dataset-metadata.json
└── data/slimpajama_8x/{manifest.json,train.bin,validation.bin,test.bin}
```

Kaggle 解压后实际挂载为
`/kaggle/input/lowp-benchmark-slimpajama-8x-v1/slimpajama_8x`。Notebook 用
Cell 1 的 symlink 映射到 recipe 期望的 `data/slimpajama_8x`。

> 更新数据集时使用**新 slug**，不要 `kaggle datasets version` 覆盖冻结的 4x；
> 也不要覆盖本 slug 已发布的版本。任何数据变化都必须产生新 fingerprint。

---

## 2. 学习率调度

固定为 Linear Warmup + Cosine Decay：

- peak LR = `1e-3`（与已稳定训练的 Exploratory Recipe 相同）
- warmup = `128` updates（1.56%）
- final LR ratio = `0.1`（末值 `1e-4`）
- schedule horizon = `8192`（resume 不改变 horizon）
- 不做超参数搜索，LR 不是本阶段研究变量

曲线（`train_platform.learning_rate`）：

| update | LR |
|---:|---:|
| 1 | 7.8125e-06 |
| 8 | 6.2500e-05 |
| 32 | 2.5000e-04 |
| 64 | 5.0000e-04 |
| 128 | 1.0000e-03 |
| 256 | 9.9944e-04 |
| 512 | 9.9497e-04 |
| 1024 | 9.7286e-04 |
| 2048 | 8.7987e-04 |
| 4096 | 5.6122e-04 |
| 6144 | 2.3579e-04 |
| 8192 | 1.0000e-04 |

公式：`u=(k-warmup)/(8192-warmup)`，
`lr=peak*(0.1+0.9*(1+cos(pi*u))/2)`；`k<=warmup` 时线性爬升。
两个 Seed 使用完全相同的调度。

---

## 3. 三档采集

| 档位 | 内容 | 节点数 | 输出目录 |
|---|---|---:|---|
| 1 | 每步矩阵标量、Head 能量、Role Relations | 8192 | `scalar_stats/`, `head_stats/`, `role_relations/` |
| 2 | 每步（前 128）后每 4 步的行列范数 | 2144 | `row_col_norms/` |
| 3 | 完整 Snapshot（复用原 Collector） | 18 | `snapshots/` |

### 3.1 矩阵与 Role

- 30 个实际 Muon 矩阵：每层 `QKV_fused, O, gate, up, down`
- 18 个 Q/K/V 派生视图：`Q=[0:384], K=[384:768], V=[768:1152]`（行切片）
- 48 个分析视图 = 30 actual + 18 derived；42 个功能矩阵 = 24 actual
  （O/gate/up/down）+ 18 derived
- Role 顺序：`W_before, W_after, G, M_prev, M_t_candidate, D, U, Delta_W`
  （`M_t_persisted` 在 `state_simulation=none` 时与 `M_t_candidate` 同一
  tensor，是 alias，不重复计算；`Delta_W = W_after - W_before`）
- Q/K/V/O 的 Head 映射：Q/K/V 用 64 行一个 head block；O 用 64 列一个
  head block。由 `build_matrix_index` 从 `n_embd/n_head` 生成并写入
  `metadata/structural_collector.json`，不硬编码。

### 3.2 第一档 Schema

每个 chunk（默认 128 updates）一个 NPZ：

```text
scalar_stats/scalar-NNNNN.npz
  updates   (n,)              float32
  stats     (n,48,8,10)       float32
  aux       (n,12)            float32
aux = [lr, weight_decay, lr*wd, muon_momentum, nesterov, muon_update_scale,
       pre_clip_grad_norm, grad_clip_norm, clipped, tokens_this_update,
       processed_target_tokens, train_nll]
stats 指标顺序 = [mean, std, rms, fro, max_abs, positive_ratio, negative_ratio,
                 zero_ratio, row_norm_cv, column_norm_cv]

head_stats/head-NNNNN.npz
  energy    (n,24,8,6)   float32   E = ||A_head||_F^2
  relative  (n,24,8,6)   float32   E / sum_h E；未定义处为 NaN
  defined   (n,24,8)     bool

role_relations/relation-NNNNN.npz
  relations (n,48,10,5)  float32
  valid     (n,48,10)    bool
metrics = [frobenius_inner, cosine, norm_ratio_A_over_B,
           relative_frobenius_difference, optimal_scalar_projection_residual]
```

Role Relations 的 10 对：`G/M_prev`, `G/M_t_candidate`, `G/D`,
`M_prev/M_t_candidate`, `M_t_candidate/D`, `M_t_candidate/U`, `D/U`,
`Delta_W/U`, `Delta_W/D`, `W_before/W_after`。A 是第一个 role、B 是第二个；
`relative_frobenius_difference = ||A-B||/||B||`，
`residual = ||B-alpha A||/||B||`, `alpha=<A,B>/||A||^2`（直接差值计算，
避免 A≈B 时的相消误差）。

### 3.3 第二档 Schema

采样：`1..128` 每步，`129..8192` 中所有 4 的倍数 → **2144** 节点
（`row_col_schedule` 校验）。

```text
row_col_norms/rowcol-NNNNN.npz
  updates        (n,)
  row_norms      (n,5,23808)      float32
  col_norms      (n,5,15360)      float32
  qkv_col_norms  (n,5,18,384)     float32
```

Role 顺序 `[W_before, G, M_t_candidate, D, U]`；30 个实际矩阵按 layer-major
`QKV_fused, O, gate, up, down` 拼接，偏移量见 metadata 的 `row_offsets` /
`col_offsets`。fused QKV 的行范数已覆盖 Q/K/V，不重复保存；Q/K/V 的列范数
无法从 fused 整体列范数恢复，因此额外保存。

所有数组保持原始 FP32，不做归一化或截断。

### 3.4 Schema 与读取接口

- 完整 Schema、索引、矩阵映射、Head 映射：`metadata/structural_collector.json`
- 每个 group 的 chunk 列表、SHA-256、字节数：`<group>/index.json`
- 只读读取接口：`src/structural_io.py`
  （`load_metadata` / `load_group` / `view_table` / `head_table` / `coverage`）
- 离线校验：`scripts/smoke_structural_collectors.py`

---

## 4. 存储与时间预算

### 4.1 单 Seed

| 数据类型 | 计算 | 大小 |
|---|---|---:|
| 18 个 Snapshot | 18 × 297,271,611 B | 5.35 GB |
| 第一档 | 34,896 B/update × 8192 | 0.286 GB |
| 第二档 | 921,600 B/node × 2144 | 1.976 GB |
| `latest.pt` | 单文件 | 0.318 GB |
| 日志/metadata | metrics、index、manifest | ≈0.03 GB |
| **合计** | | **≈7.96 GB (7.41 GiB)** |

双 Seed ≈ **15.92 GB**。按 Kaggle `/kaggle/working` 20 GB 配额，**可以直接
训练到 8192 并保留全部数据**，理论余量约 4.1 GB（若配额按 20 GiB 计约
5.6 GB）。原子写入的临时文件峰值约为 Snapshot +297 MB、Checkpoint +318 MB，
不会同时发生，仍在余量内。

因此 Cell 4–7（暂停/归档/校验/恢复）是**安全网而非必需步骤**，在以下情况使用：

- `df -h /kaggle/working` 剩余 **< 2.5 GB**；
- 需要在 Kaggle 会话之间保存数据；
- 后续扩大到 3 个 Seed 或更多 Snapshot 节点。

注意：**不要同时保留归档 tar 和原数据**。tar 会把同一批数据再复制一份；
如需归档，必须按 Cell 5 → 6 先上传校验、再 prune、最后删除 tar。

### 4.2 运行时间

Exploratory 4096-update 运行实测约 1.2 s/update（含低频 eval 和 9 个
Snapshot）。本轮 8192 updates、每步 Collector，T4 实测 **1.204 s/update**
（Cell 2b，2026-11-19），单 Seed **约 2.8 小时**；双卡并行总墙钟约
**3 小时**，低于 Kaggle 12 小时上限。GPU 开销由 Cell 2b 在正式训练前实测
（2026-11-19 结果：1.204 s/update，1.93×，显存 +0.88 MiB）。

### 4.3 本地 CPU Smoke Test（2026-11-19）

真实 30M Recipe，CPU（无 CUDA），2 updates：

| 配置 | mean update | 说明 |
|---|---:|---|
| 无 Collector | 16.77 s | 基准 |
| 三档 Collector | 27.92 s | +66%（CPU 上的算子 dispatch 主导） |

这是 **CPU-only 上界**，不代表 T4：Collector 的每步约 7000 个小 kernel 在
GPU 上主要受 launch 开销约束。Kaggle 首个 Cell 会在前 32 updates 记录
`update_elapsed_seconds`，若实测 overhead > 20%，再决定是否合并同形状矩阵的
reduction。

数值校验（`scripts/smoke_structural_collectors.py`，update 2 对 Snapshot 重算）：

| 检查 | 最大绝对误差 | 最大相对误差 |
|---|---:|---:|
| scalar_stats | 1.75e-06 | 5.05e-05 |
| head_energy | 2.41e-06 | 1.80e-07 |
| role_relations | 2.01e-05 | 1.39e-03（近零 cosine） |
| row_norms | 9.23e-08 | 1.42e-07 |
| col_norms | 9.28e-08 | 1.56e-07 |
| qkv_col_norms | 8.73e-08 | 1.84e-07 |

判定标准：`max_abs < 1e-4` 或 `max_rel < 2e-4`，全部通过。训练轨迹逐位一致
（同一 Seed 下 Collector 开/关的最终 checkpoint 张量 `torch.equal`）。

---

## 5. Kaggle Notebook Cells

在 Kaggle Notebook 设置中开启 **GPU（T4 × 2）** 和 **Internet**。

### Cell 1 — 环境、代码、数据挂载与指纹验证

```bash
%%bash
set -euo pipefail

# ===== 研究者填写 =====
TARGET_COMMIT="<FIXED_COMMIT_SHA>"
REPO="https://github.com/SiennaDai/muon_research"
# =====================

BASE=/kaggle/working/structural_emergence
REPO_ROOT=/kaggle/working/muon_research
mkdir -p "$BASE"/logs "$BASE"/archive /kaggle/working/muon_data/data

if [ ! -d "$REPO_ROOT/.git" ]; then git clone "$REPO" "$REPO_ROOT"; fi
git -C "$REPO_ROOT" fetch --all --tags
git -C "$REPO_ROOT" checkout --detach "$TARGET_COMMIT"
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$TARGET_COMMIT"

DATASET_ROOT="$(dirname "$(find /kaggle/input -path '*slimpajama_8x/manifest.json' | head -1)")"
test -n "$DATASET_ROOT"
ln -sfn "$DATASET_ROOT" /kaggle/working/muon_data/data/slimpajama_8x

cat > "$BASE/env.sh" <<EOF
export REPO_ROOT="$REPO_ROOT"
export PLATFORM="$REPO_ROOT/platform"
export BASE="$BASE"
export DATA_ROOT="/kaggle/working/muon_data"
export SNAPSHOT_CONFIG="\$PLATFORM/configs/muon_snapshot_structural_18nodes.json"
export STRUCTURAL_CONFIG="\$PLATFORM/configs/structural_emergence_8192.json"
export ARCHIVE_DIR="\$BASE/archive"
EOF

BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
cd "$PLATFORM"
python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.environ["PLATFORM"], "src"))
from data.frozen_tokens import load_manifest
from optim.muon_snapshot_collector import load_snapshot_config
from optim.muon_structural_collector import load_structural_config

manifest = load_manifest(Path(os.environ["DATA_ROOT"]) / "data/slimpajama_8x/manifest.json")
assert manifest["fingerprint"] == "ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489"
assert manifest["splits"]["train"]["tokens"] == 70000418
snap = load_snapshot_config(os.environ["SNAPSHOT_CONFIG"], total_updates=8192)
assert snap.config_hash == "06bfd036619198750d29c952e261d479ccbf9bca495155d999d5983231819e9c"
struct = load_structural_config(os.environ["STRUCTURAL_CONFIG"])
assert struct.config_hash == "fa5f9f9e8e9051cf1208185adf7ec5c2ac82d4f4d1b38cd2ea0430c4347bc29e"
print("data fingerprint:", manifest["fingerprint"])
print("snapshot hash:", snap.config_hash, "nodes:", len(snap.updates))
print("structural hash:", struct.config_hash)
print("environment: PASS")
PY

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), 'GPUs')"
df -h /kaggle/working
```

### Cell 2 — 双卡并行训练

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
cd "$PLATFORM"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

python src/main.py --recipe "${PLATFORM}/recipes/structural_emergence_30m_fp32_muon_s0.json" \
  --data-root "$DATA_ROOT" --to-device cpu --dry-run >/dev/null

for SEED in 0 1; do
  RUN_DIR="$BASE/seed$SEED"
  LOG="$BASE/logs/seed$SEED.log"
  mkdir -p "$RUN_DIR"
  CUDA_VISIBLE_DEVICES=$SEED nohup python src/main.py \
    --recipe "$PLATFORM/recipes/structural_emergence_30m_fp32_muon_s$SEED.json" \
    --data-root "$DATA_ROOT" --run-dir "$RUN_DIR" --to-device cuda:0 \
    --snapshot-config "$SNAPSHOT_CONFIG" \
    --structural-config "$STRUCTURAL_CONFIG" \
    > "$LOG" 2>&1 &
  echo $! > "$BASE/logs/seed$SEED.pid"
  echo "seed$SEED pid=$(cat "$BASE/logs/seed$SEED.pid")"
done
sleep 30
tail -n 5 "$BASE/logs/seed0.log" "$BASE/logs/seed1.log"
```

两个进程各自持有独立 Run Directory、Collector 输出、日志和 `latest.pt`；
共享只读数据集；不使用 DDP，不改变有效 batch。

### Cell 2b — Collector 性能基准（GPU，正式训练前）

在正式 8192 步之前，用 8 updates 对比“仅结构 Collector”和“无 Collector”的
真实 T4 开销，并记录显存。结果写入 `perf.json`，完成后可删除临时目录。

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
cd "$PLATFORM"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PERF="$BASE/perf"; rm -rf "$PERF"; mkdir -p "$PERF"

cat > "$PERF/structural_only.json" <<EOF
{"schema_version":1,"enabled":true,
 "scalar_stats":{"enabled":true,"chunk_updates":128},
 "head_stats":{"enabled":true,"chunk_updates":128},
 "role_relations":{"enabled":true,"chunk_updates":128},
 "row_col_norms":{"enabled":true,"warmup_updates":128,"every_updates":4,"chunk_nodes":32},
 "min_free_bytes":0,"max_total_bytes":null}
EOF
cat > "$PERF/snapshots_off.json" <<EOF
{"schema_version":1,"enabled":false,"updates":[],"on_error":"disable","max_total_bytes":null,"min_free_bytes":0}
EOF

CUDA_VISIBLE_DEVICES=0 python src/main.py \
  --recipe "$PLATFORM/recipes/structural_emergence_30m_fp32_muon_s0.json" \
  --data-root "$DATA_ROOT" --run-dir "$PERF/with" --to-device cuda:0 \
  --stop-at-update 8 --snapshot-config "$PERF/snapshots_off.json" \
  --structural-config "$PERF/structural_only.json" > "$PERF/with.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python src/main.py \
  --recipe "$PLATFORM/recipes/structural_emergence_30m_fp32_muon_s0.json" \
  --data-root "$DATA_ROOT" --run-dir "$PERF/without" --to-device cuda:0 \
  --stop-at-update 8 > "$PERF/without.log" 2>&1

python - "$PERF" <<'PY'
import json, sys
from pathlib import Path
perf = Path(sys.argv[1])
with_run = json.loads((perf / "with/summary.json").read_text())
without_run = json.loads((perf / "without/summary.json").read_text())
ratio = with_run["mean_update_seconds"] / without_run["mean_update_seconds"]
report = {
    "with_structural_mean_update_seconds": with_run["mean_update_seconds"],
    "without_collector_mean_update_seconds": without_run["mean_update_seconds"],
    "overhead_ratio": ratio,
    "with_cuda_peak_allocated_bytes": with_run["cuda_peak_allocated_bytes"],
    "without_cuda_peak_allocated_bytes": without_run["cuda_peak_allocated_bytes"],
    "completed_updates_each": 8,
}
(perf / "perf.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
cp "$PERF/perf.json" "$BASE/perf.json"
rm -rf "$PERF"
echo "saved $BASE/perf.json"
```

2026-11-19 首次实测（T4，8 updates）：`1.204 s/update` vs `0.622 s/update`，
`overhead_ratio = 1.93`，显存增加 921,600 B（0.88 MiB，恰好是第二档单节点
缓冲）。按绝对时间推算单 Seed ≈2.8 h，双卡并行仍在 12 h 内，因此**继续训练**。

判定改为绝对时间：推算单 Seed **> 6 h** 或 `overhead_ratio > 3.0` 时才停下做
同形状矩阵批量归约优化（当前实现每个 update 约 7000 个小 kernel，Python
dispatch 是主要成本）；否则按当前实现采集，保证统计口径与已验收的
Smoke Test 一致。

### Cell 3 — 状态监控（可反复执行）

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
for SEED in 0 1; do
  RUN_DIR="$BASE/seed$SEED"
  echo "===== seed$SEED ====="
  pgrep -af "structural_emergence.*_s$SEED" || true
  grep -E '^\[(run|eval|ckpt|pause|done)\]|ERROR|Traceback' "$BASE/logs/seed$SEED.log" | tail -n 6 || true
  python - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
print("metrics rows:", sum(1 for _ in open(run/"metrics.jsonl")) if (run/"metrics.jsonl").exists() else 0)
print("checkpoint:", (run/"checkpoints/latest.pt").exists())
print("structural:", json.loads((run/"structural_status.json").read_text())["groups"]
      if (run/"structural_status.json").exists() else "not closed yet")
print("snapshots:", len(list((run/"snapshots").glob("update_*.pt"))) if (run/"snapshots").exists() else 0)
PY
done
df -h /kaggle/working | tail -1
```

### Cell 4 — 安全暂停

在完整 update 边界暂停；每个进程会写一次 `latest.pt`，并刷新 Collector 缓冲。

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
for SEED in 0 1; do touch "$BASE/seed$SEED/PAUSE"; done
echo "waiting for both processes to pause..."
for i in $(seq 1 240); do
  done=0
  for SEED in 0 1; do
    R="$BASE/seed$SEED/summary.json"
    if [ -f "$R" ] && grep -q '"status": "paused_manual"' "$R"; then done=$((done+1)); fi
  done
  [ "$done" -eq 2 ] && break
  sleep 10
done
for SEED in 0 1; do
  cat "$BASE/seed$SEED/summary.json" | python -c "import json,sys; d=json.load(sys.stdin); print('seed$SEED', d['status'], d['completed_updates'])"
done
```

> 不推荐 `kill -INT`：它会停在中途，恢复点只能回退到最近一个 128 边界。
> 暂停文件方式在下一个 update 边界停止并立即保存 Checkpoint。

### Cell 5 — 流式归档与 GitHub Release 上传

先添加 Kaggle Secret `GITHUB_TOKEN`（repo scope），再执行。归档范围按需选择
（示例：到 2048）。

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
python - <<'PY'
import os
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
open("/kaggle/working/.gh_token", "w").write(os.environ["GITHUB_TOKEN"])
PY
export GITHUB_TOKEN="$(cat /kaggle/working/.gh_token)"
cd "$PLATFORM"

for SEED in 0 1; do
  python scripts/archive_structural_run.py archive \
    --run-dir "$BASE/seed$SEED" --out-dir "$ARCHIVE_DIR" \
    --through-update 2048 --shard-bytes 1500000000
done
ls -lh "$ARCHIVE_DIR"

python scripts/archive_structural_run.py upload \
  --manifest "$ARCHIVE_DIR/seed0-archive-manifest.json" \
  --repo <DATA_REPO> --tag se-seed0-part1 --verify-download
python scripts/archive_structural_run.py upload \
  --manifest "$ARCHIVE_DIR/seed1-archive-manifest.json" \
  --repo <DATA_REPO> --tag se-seed1-part1 --verify-download
```

### Cell 6 — 校验与清理

只有 `verified=true` 的 manifest 才允许 prune。

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
cd "$PLATFORM"
for SEED in 0 1; do
  python scripts/archive_structural_run.py prune \
    --run-dir "$BASE/seed$SEED" \
    --manifest "$ARCHIVE_DIR/seed$SEED-archive-manifest.json"
done
rm -f "$ARCHIVE_DIR"/*.tar /kaggle/working/.gh_token
df -h /kaggle/working | tail -1

python - "$BASE" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for seed in ("seed0", "seed1"):
    run = base / seed
    for group in ("scalar_stats", "head_stats", "role_relations", "row_col_norms"):
        index = json.loads((run / group / "index.json").read_text())
        pruned = sum(bool(e.get("pruned")) for e in index["chunks"])
        live = len(index["chunks"]) - pruned
        print(seed, group, "live", live, "pruned", pruned, "last", index["last_update"])
PY
```

### Cell 7 — 双卡恢复

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
cd "$PLATFORM"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  RUN_DIR="$BASE/seed$SEED"
  rm -f "$RUN_DIR/PAUSE"
  CUDA_VISIBLE_DEVICES=$SEED nohup python src/main.py \
    --recipe "$PLATFORM/recipes/structural_emergence_30m_fp32_muon_s$SEED.json" \
    --data-root "$DATA_ROOT" --run-dir "$RUN_DIR" --to-device cuda:0 \
    --resume "$RUN_DIR/checkpoints/latest.pt" \
    --snapshot-config "$SNAPSHOT_CONFIG" \
    --structural-config "$STRUCTURAL_CONFIG" \
    >> "$BASE/logs/seed$SEED.log" 2>&1 &
  echo $! > "$BASE/logs/seed$SEED.pid"
done
sleep 30; tail -n 4 "$BASE/logs/seed0.log"
```

恢复要求：同一个 Run Directory、同一个 Recipe、两个 sidecar 的 hash 必须与
存储值一致；调度 horizon 仍为 8192；Collector 跳过已提交节点与已归档节点；
已 prune 的 chunk 不会被重建，新数据以更大的 chunk 序号追加。

### Cell 8 — 最终验收

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_emergence
source "$BASE/env.sh"
cd "$PLATFORM"
for SEED in 0 1; do
  python - "$BASE/seed$SEED" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text())
status = json.loads((run / "structural_status.json").read_text())
assert summary["status"] == "completed", summary["status"]
assert summary["completed_updates"] == 8192
assert summary["data_fingerprint"] == "ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489"
assert status["failed"] is False and not status["errors"]
print(run.name, "summary OK", summary["final_validation_nll"])
for group, expected in (("scalar_stats", [1, 8192]), ("head_stats", [1, 8192]),
                        ("role_relations", [1, 8192]), ("row_col_norms", [1, 8192])):
    entries = json.loads((run/group/"index.json").read_text())["chunks"]
    first = min(int(e["first_update"]) for e in entries)
    last = max(int(e["last_update"]) for e in entries)
    assert [first, last] == expected, (group, first, last)
    print(group, "coverage", first, last)
PY
done
# 对最新一个仍在本地的 Snapshot 做离线重算校验
LATEST=$(ls "$BASE"/seed0/snapshots/update_*.pt | sed 's/.*update_//; s/\.pt//' | sort -n | tail -1)
test -n "$LATEST"
python scripts/smoke_structural_collectors.py \
  --run-dir "$BASE/seed0" --snapshot-update "$((10#$LATEST))"
```

验收标准：

1. 两个 Seed 的 `summary.status == completed` 且 `completed_updates == 8192`。
2. `data_fingerprint` 均为 `ba3ac442...`；两个 Recipe 仅 `seed` 不同。
3. `structural_status.failed == false`，`errors` 为空。
4. 四个 group 的 chunk 覆盖区间均为 `[1, 8192]`（live + pruned 合并计）。
5. 18 个 Snapshot 全部已提交（本地或归档，`snapshots/archived.json` 记录）。
6. 离线重算误差满足 4.3 的判定标准。
7. `metrics.jsonl` 恰好 8192 条 `train` 事件，Update 索引连续。

---

## 6. 实现说明与边界

- 优化器数值路径未改动；Collector 只在 `ReferenceMuon.step` 的既有只读
  observation hook 上增加计算，不改变 `p.copy_`、动量、梯度或调度。
- 每步统计在 tensor 所在 device 上完成，只把小的统计向量传回 CPU；每步一次
  同步。
- `M_t_persisted` 与 `M_t_candidate` 必须为同一 tensor（`state_simulation=none`），
  否则 Collector 拒绝启动。
- Snapshot Collector 的 18 节点、alias、checksum、manifest 机制完全复用；
  既有 Batch1–5 分析代码可直接读取新 Snapshot。
- 本轮不提供新的完整分析与可视化体系；只提供 Schema、读取接口和离线校验脚本。
- 暂停文件 `PAUSE`、`paused_manual` 状态、归档 `pruned` 标记和
  `snapshots/archived.json` 都是本轮新增的恢复机制。

## 7. 新增/修改文件

新增：

- `platform/src/optim/muon_structural_collector.py`
- `platform/src/structural_io.py`
- `platform/scripts/prepare_slimpajama_stream.py`
- `platform/scripts/smoke_structural_collectors.py`
- `platform/scripts/archive_structural_run.py`
- `platform/configs/structural_emergence_8192.json`
- `platform/configs/muon_snapshot_structural_18nodes.json`
- `platform/recipes/structural_emergence_30m_fp32_muon_s0.json`
- `platform/recipes/structural_emergence_30m_fp32_muon_s1.json`
- `platform/tests/test_structural_collector.py`
- `platform/tests/test_structural_archive.py`
- `platform/docs/STRUCTURAL_EMERGENCE_STUDY.zh-CN.md`
- `platform/docs/STRUCTURAL_SMOKE_TEST.json`

修改：

- `platform/src/train_platform.py`（sidecar 接入、observer 组合、暂停/恢复）
- `platform/src/main.py`（`--structural-config`、`paused_manual`）
- `platform/src/optim/muon_snapshot_collector.py`（归档节点不计为 missed）
