# Structural Concentration Checkpoints — 1536-step FP32 Hessian-ready Runbook

> 目标:沿用 seed0/seed1 的原始 FP32 Reference Muon 轨迹,重新运行到 **update 1536**,
> 在固定节点保存 **pre-update 分析 Checkpoint**(完整模型 + Muon 动量 + sampler + 该 update 的
> window ids),并在新增节点保存 **QKV_fused / O** 的 Role 张量,供离线 Hessian / 损失景观
> 分析与 Stage 1–3 的结构涌现结果结合。
>
> 训练期间不计算 Hessian,不做低精度训练;不改 recipe、数据、种子、LR horizon。
> 一次运行完成,不需要中途暂停、分片或延长。

## 0. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 原运行 commit | `72beef8c63eb131f38a5ae7a5ca89bfb51eef593` |
| 本轮 commit | 执行前填写 `<FIXED_COMMIT_SHA>` |
| Recipes | `platform/recipes/structural_emergence_30m_fp32_muon_s0/s1.json` |
| Dataset | `siennadai/lowp-benchmark-slimpajama-8x-v1` |
| Dataset fingerprint | `ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489` |
| Seeds | model 0/1;data 1337;algorithm 2026 |
| LR horizon | 8192(不得改;只用 `--stop-at-update 1536`) |
| 节点配置 | `platform/configs/node_checkpoints_s0s1.json` |
| 配置 hash | `d0a936c0a0e7737138a0ae139f24e5703ad0b5bb78596c5018ada78cf6881774` |
| Kaggle | T4×2,每个 Seed 一个进程;`/kaggle/working` ≈20 GB |

## 1. 节点与内容

节点 `u` 表示 **update u 开始前**的状态(`completed_updates = u-1`;节点 0 为初始化,
`completed_updates=0`)。节点清单(31 个/Seed):

```text
0,8,16,32,48,56,64,80,96,112,128,
192,256,320,384,448,512,
576,640,704,768,784,832,896,960,1024,
1152,1280,1312,1408,1536
```

每个节点:

```text
node_checkpoints/pre_uNNNNNN.pt
  model.state_dict()               完整 29,938,560 参数
  muon_momentum                    30 个 Muon 矩阵(节点 0/1 尚无动量,记为空)
  sampler.state_dict()             take 前状态
  rng / completed_updates / processed_target_tokens / fingerprints
node_checkpoints/pre_uNNNNNN.json
  window_ids / micro_batch_window_ids / sampler epoch+offset / sha256
```

新增节点(20 个)额外保存 QKV_fused + O 的 7 个独立 Role:

```text
16,64,80,96,112,192,320,384,448,576,640,704,784,832,896,960,1152,1280,1312,1408
```

- 与旧 Snapshot 重合的 10 个节点(8,32,48,56,128,256,512,768,1024,1536)直接复用旧 Role;
- 不保存辅助 AdamW moments、MLP Role、probe;
- `M_t_persisted` 与 `M_t_candidate` 为 alias,只存一次,alias 记录在 sidecar。

## 2. 存储预算(实测)

| 项目 | 大小 |
|---|---:|
| 节点 0/1 基础 checkpoint | 118 MiB ×2 |
| 节点 ≥2 基础 checkpoint | 159 MiB ×29 |
| 基础 checkpoint/Seed | 4.73 GiB |
| QKV/O Role(20 节点) | 95 MiB ×20 = 1.86 GiB |
| **单 Seed 合计** | **≈6.59 GiB** |
| **两 Seed 合计** | **≈13.18 GiB** |
| 两个 rolling latest.pt | ≈0.6 GiB |
| **总占用** | **≈13.8 GiB** |

不需要分片;commit 结束后由 Kaggle Output 保存。

## 3. Notebook Cells

### Cell 1 — 环境、代码、数据

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_concentration_ckpt
REPO=/kaggle/working/muon_research
mkdir -p "$BASE"/logs /kaggle/working/muon_data/data
[ -d "$REPO/.git" ] || git clone https://github.com/SiennaDai/muon_research "$REPO"
TARGET_COMMIT="<FIXED_COMMIT_SHA>"
git -C "$REPO" fetch --all --tags
git -C "$REPO" checkout --detach "$TARGET_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$TARGET_COMMIT"
DATASET_ROOT="$(dirname "$(find /kaggle/input -path '*slimpajama_8x/manifest.json' | head -1)")"
test -n "$DATASET_ROOT"
ln -sfn "$DATASET_ROOT" /kaggle/working/muon_data/data/slimpajama_8x
cd "$REPO/platform"
python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd() + "/src")
from data.frozen_tokens import load_manifest
from optim.muon_node_checkpoint import load_node_config
m = load_manifest("/kaggle/working/muon_data/data/slimpajama_8x/manifest.json")
assert m["fingerprint"] == "ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489"
c = load_node_config("configs/node_checkpoints_s0s1.json")
assert c.config_hash == "d0a936c0a0e7737138a0ae139f24e5703ad0b5bb78596c5018ada78cf6881774"
print("data fingerprint:", m["fingerprint"])
print("node config hash:", c.config_hash, "nodes:", len(c.updates))
PY
```

### Cell 2 — 配置与 dry run

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
cd "$REPO/platform"
cp configs/node_checkpoints_s0s1.json /kaggle/working/node_checkpoints_s0s1.json
python src/main.py --recipe recipes/structural_emergence_30m_fp32_muon_s0.json \
  --data-root /kaggle/working/muon_data --to-device cpu --dry-run \
  --node-checkpoint-config configs/node_checkpoints_s0s1.json
```

### Cell 3 — 双卡训练到 1536

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_concentration_ckpt
REPO=/kaggle/working/muon_research
cd "$REPO/platform"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  RUN_DIR="$BASE/seed$SEED"
  mkdir -p "$RUN_DIR"
  CUDA_VISIBLE_DEVICES=$SEED nohup python src/main.py \
    --recipe "recipes/structural_emergence_30m_fp32_muon_s$SEED.json" \
    --data-root /kaggle/working/muon_data --run-dir "$RUN_DIR" \
    --to-device cuda:0 --stop-at-update 1536 \
    --node-checkpoint-config configs/node_checkpoints_s0s1.json \
    > "$BASE/logs/seed$SEED.log" 2>&1 &
  echo $! > "$BASE/logs/seed$SEED.pid"
  echo "seed$SEED pid=$(cat "$BASE/logs/seed$SEED.pid")"
done
sleep 30
tail -n 3 "$BASE/logs/seed0.log" "$BASE/logs/seed1.log"
```

### Cell 4 — 阻塞监控(每 5 分钟打印一次)

放在 Cell 3 之后。它保持 commit 会话不关闭,直到两个训练进程退出;每 5 分钟打印两个
Seed 的最新 update、loss、节点计数和磁盘。

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_concentration_ckpt
echo "monitor started; printing every 5 minutes"
while true; do
  alive=0
  for SEED in 0 1; do
    PID=$(cat "$BASE/logs/seed$SEED.pid" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then alive=$((alive+1)); fi
    LAST=$(grep -E '^\[(train|eval|ckpt|pause|done)\]' "$BASE/logs/seed$SEED.log" 2>/dev/null | tail -1 || echo "starting...")
    NODES=$(find "$BASE/seed$SEED/node_checkpoints" -maxdepth 1 -name 'pre_u*.pt' 2>/dev/null | wc -l)
    ROLES=$(find "$BASE/seed$SEED/node_checkpoints" -maxdepth 1 -name 'pre_u*_roles.pt' 2>/dev/null | wc -l)
    echo "[$(date -u +%H:%M:%S)] seed$SEED | nodes=$NODES roles=$ROLES | $LAST"
  done
  echo "[$(date -u +%H:%M:%S)] disk: $(df -h /kaggle/working | tail -1 | awk '{print $4" free"}') | alive=$alive"
  [ "$alive" -eq 0 ] && break
  sleep 300
done
for SEED in 0 1; do
  python - "$BASE/seed$SEED/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(sys.argv[1], d["status"], d["completed_updates"], d["schedule_total_updates"])
assert d["status"] in {"completed", "paused_staged"}, d["status"]
PY
done
```

### Cell 5 — 最终验收与存储核对

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working/structural_concentration_ckpt
for SEED in 0 1; do
  echo "=== seed$SEED ==="
  find "$BASE/seed$SEED/node_checkpoints" -maxdepth 1 -name 'pre_u*.pt' | wc -l
  find "$BASE/seed$SEED/node_checkpoints" -maxdepth 1 -name 'pre_u*_roles.pt' | wc -l
  du -sh "$BASE/seed$SEED"
done
python - "$BASE" <<'PY'
import json, sys, torch
from pathlib import Path
base = Path(sys.argv[1])
for seed in ("seed0", "seed1"):
    directory = base / seed / "node_checkpoints"
    index = json.loads((directory / "index.json").read_text())
    nodes = [int(n["update"]) for n in index["nodes"]]
    assert nodes == [0,8,16,32,48,56,64,80,96,112,128,192,256,320,384,448,512,576,640,704,768,784,832,896,960,1024,1152,1280,1312,1408,1536], nodes
    payload = torch.load(directory / "pre_u001536.pt", map_location="cpu", weights_only=False)
    assert payload["completed_updates"] == 1535
    assert len(payload["model"]) == 45
    assert payload["muon_momentum"]
    roles = torch.load(directory / "pre_u001408_roles.pt", map_location="cpu", weights_only=False)
    assert len(roles["tensors"]) == 84
    print(seed, "nodes", len(nodes), "last completed", payload["completed_updates"])
PY
df -h /kaggle/working | tail -1
```

## 4. commit and run all 说明

- 用 "Save & Run All" 运行整个 notebook;Cell 4 会阻塞会话直到两个 Seed 完成,
  因此不会出现训练进程被提前杀掉。
- 训练约 **25–35 分钟**(1536 updates,双卡并行);
  节点保存约 1–3 分钟;commit 结束保存约 14 GiB Output 需额外时间。
- 完成后在本机用 `kaggle kernels output <owner>/<notebook> -p <dest>` 下载整个 Output。
- 不需要中途比较旧 Snapshot、pause、export 或 prune。

## 5. 分析与复用

- 节点 0–1536 的 `pre_u*.pt` 是 Hessian/损失景观的 pre-update 状态;
  数据由 `window_ids` + 冻结数据集唯一确定。
- 新增节点的 `pre_u*_roles.pt` 提供 QKV/O 的 7 个 Role;与旧 Snapshot 的 10 个节点合并后,
  可直接复用 Stage 1–3 的 Head 谱分析代码(读取接口:`platform/src/node_checkpoint_io.py`)。
- 不保存 MLP Role;如后续需要,可由模型 + Muon 动量 + 数据离线重算。
