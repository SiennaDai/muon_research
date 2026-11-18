# Kaggle 30M FP32 Reference Muon — Full 4096-Update Runbook

> 目标：在 Kaggle T4 上从头训练 30M LLaMA + Reference Muon + FP32 至 **4096 updates**，只在 update **256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096** 采集 9 份研究 Snapshot。Checkpoint 采用单文件滚动保存，只保留 `checkpoints/latest.pt`。
>
> 本文件只提供训练前准备、正式训练和验收所需的可复制 Cells；**不实际启动训练、不下载数据、不上传结果、不修改本地已验收的 Stage 1 归档**。
>
> 最终研究数据 = 本地已有 Stage 1 的 15 个 Snapshot（update 1–128）+ 本轮 9 个 Snapshot（update 256–4096）= **24 个节点**。

---

## 0. 冻结事实与路径

| 项目 | 固定值 |
|---|---|
| 代码仓库 | `https://github.com/SiennaDai/muon_research` |
| 代码根目录 | `/kaggle/working/muon_research` |
| 平台目录 | `/kaggle/working/muon_research/platform` |
| 数据目录 | `/kaggle/working/muon_data` |
| 数据集实际路径 | `/kaggle/input/datasets/siennadai/lowp-benchmark-slimpajama-4x-v1` |
| 新 run_dir | `/kaggle/working/runs/muon_30m_fp32_muon_s0_full` |
| Snapshot config | `/kaggle/working/muon_snapshot_config.json` |
| Recipe | `recipes/mini_fp32_reference_muon_4x_s0.json` |
| 参数规模 | 29,938,560 |
| Muon 组 | 30 个矩阵 / 10,616,832 参数 |
| 优化器 / 精度 | `reference_muon` / FP32 |
| seed / data_seed / algorithm_seed | `0 / 1337 / 2026` |
| tokens_per_update | 8192 |
| total_updates / schedule_total_updates | `4096 / 4096` |
| Checkpoint 间隔 | 128 updates |
| Recipe fingerprint | `fda8bd1101831b1d9b54d7bbd389d762e8dae353a4fbd7dfb3dce79518081870` |
| Dataset fingerprint | `30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18` |
| 本轮 Snapshot config hash | `d9e7c3df0cec1f048b27489f332c3f5dbe054fd0ddad5cb118138e9e83dcd73e` |

本轮 Snapshot 节点：

```text
[256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
```

Checkpoint 行为（本轮代码）：

- update 0 写入 `checkpoints/latest.pt`（若 recipe 的 `save_initial=true`）。
- 每 128 updates 原子覆盖同一个 `latest.pt`。
- update 4096 正常完成时，`latest.pt` 就是最终 Checkpoint。
- 不生成 `initial.pt`、`update_XXXXXX.pt`、`final.pt`。
- 不保存不完整 update 的状态；KeyboardInterrupt 后 `latest.pt` 仍是最后一个完整 update 边界。

> **重要**：Stage 1 的 15 个 Snapshot 是旧 run 用 24 节点配置（hash `9a073ba2...`）产生的；本轮 9 个 Snapshot 使用 9 节点配置（hash `d9e7c3df...`）。合并后的 24 节点数据会同时包含两个 `snapshot_config_hash`，这是预期结果，不是损坏。每个 Manifest 的 hash 必须与它所属轮次的 sidecar 配置一致。

---

## 1. Environment Check Cell

先在 Kaggle Notebook 设置中开启 **GPU** 和 **Internet**。

```bash
%%bash
set -euo pipefail

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== python / torch ==="
python - <<'PY'
import shutil
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory bytes:", props.total_memory)
    print("capability:", torch.cuda.get_device_capability(0))
free = shutil.disk_usage("/kaggle/working").free
print("working free bytes:", free)
print("working free GiB:", free / 1024**3)
PY

echo "=== disk ==="
df -h /kaggle/working /kaggle/input || true

echo "=== input mounts ==="
ls -la /kaggle/input
ls -la /kaggle/input/datasets/siennadai/lowp-benchmark-slimpajama-4x-v1 2>/dev/null || true
```

要求：

- `torch.cuda.is_available() == True`
- GPU 名称包含 `T4`
- `/kaggle/working` 可用空间建议 **≥ 8 GiB**（9 个 Snapshot + 1 个 latest.pt + 临时 release 打包）
- 能看到数据集挂载路径

---

## 2. Code Acquisition Cell（固定 Commit）

研究者先把本轮代码提交并推送到 `https://github.com/SiennaDai/muon_research`，然后填写 `TARGET_COMMIT`。

```bash
%%bash
set -euo pipefail

# ===== 研究者必须填写 =====
NEW_REPO="https://github.com/SiennaDai/muon_research"
TARGET_COMMIT="<FIXED_COMMIT_SHA>"
# =========================

REPO_ROOT="/kaggle/working/muon_research"
PLATFORM="$REPO_ROOT/platform"

if [ ! -d "$REPO_ROOT/.git" ]; then
  git clone "$NEW_REPO" "$REPO_ROOT"
fi

git -C "$REPO_ROOT" fetch --all --tags
git -C "$REPO_ROOT" checkout --detach "$TARGET_COMMIT"

HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$TARGET_COMMIT" ]; then
  echo "HEAD mismatch: expected $TARGET_COMMIT got $HEAD_SHA" >&2
  exit 1
fi

# 只检查 tracked 修改；Kaggle 输入的 untracked 文件不影响训练。
STATUS="$(git -C "$REPO_ROOT" status --short --untracked-files=no)"
if [ -n "$STATUS" ]; then
  echo "Repository has tracked modifications after checkout:" >&2
  echo "$STATUS" >&2
  exit 1
fi

echo "repo_root=$REPO_ROOT"
echo "platform=$PLATFORM"
echo "head=$HEAD_SHA"
echo "status=clean"

cat > /kaggle/working/muon_env.sh <<EOF
export REPO_ROOT="$REPO_ROOT"
export PLATFORM="$PLATFORM"
export TARGET_COMMIT="$TARGET_COMMIT"
export DATA_ROOT="/kaggle/working/muon_data"
export SNAPSHOT_CONFIG="/kaggle/working/muon_snapshot_config.json"
export RUN_ROOT="/kaggle/working/runs"
export RUN_ID="muon_30m_fp32_muon_s0_full"
export RUN_DIR="\$RUN_ROOT/\$RUN_ID"
export LOG_DIR="/kaggle/working/logs"
export RELEASE_DIR="/kaggle/working/release/muon_30m_fp32_muon_s0_full_9nodes"
EOF

mkdir -p /kaggle/working/logs
```

说明：

- 固定 Commit 必须包含本轮修改：9 节点 Snapshot 配置、滚动 Checkpoint、更新后的本文档。
- 如果仓库是私有的，使用 Kaggle Secret/PAT；不要把 token 打印到 notebook 输出。
- 如果 `status` 不为空，停止并检查 checkout 是否完整。

---

## 3. Dataset Mount and Fingerprint Validation Cell

本 Cell 不下载、不重新生成 token；只从 Kaggle Input 接入已冻结的 SlimPajama 4x，并验证 manifest/token 文件 SHA-256。

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

mkdir -p "$DATA_ROOT/data"

python - <<'PY'
import os
import sys
from pathlib import Path

REPO_ROOT = os.environ["REPO_ROOT"]
PLATFORM = os.environ["PLATFORM"]
DATA_ROOT = Path(os.environ["DATA_ROOT"])
sys.path.insert(0, str(Path(PLATFORM) / "src"))

from data.frozen_tokens import load_manifest

EXPECTED_FINGERPRINT = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"
EXPECTED_PROTOCOL = "slimpajama-hash-split-v1"
FIXED_DATASET = Path("/kaggle/input/datasets/siennadai/lowp-benchmark-slimpajama-4x-v1")

candidates = []
if FIXED_DATASET.is_dir():
    candidates.extend(sorted(FIXED_DATASET.rglob("slimpajama_4x/manifest.json")))
    candidates.extend(sorted(FIXED_DATASET.rglob("manifest.json")))
# Fallback for mount-name changes; still requires the exact fingerprint.
candidates.extend(sorted(
    p for p in Path("/kaggle/input").rglob("manifest.json")
    if p.parent.name == "slimpajama_4x"
))

seen = []
chosen = None
dedup = []
for candidate in candidates:
    if candidate in dedup:
        continue
    dedup.append(candidate)
    try:
        manifest = load_manifest(candidate)
    except Exception as exc:
        seen.append((str(candidate), f"invalid: {exc}"))
        continue
    seen.append((str(candidate), manifest.get("fingerprint")))
    if manifest.get("fingerprint") == EXPECTED_FINGERPRINT:
        chosen = (candidate, manifest)
        break

if chosen is None:
    print("Candidates:")
    for path, info in seen:
        print(" ", path, info)
    raise SystemExit(
        "Mounted manifest fingerprint does not match the approved 4x protocol. "
        "Stop and resolve the dataset identity; do not edit the manifest."
    )

candidate, manifest = chosen
if manifest.get("protocol_id") != EXPECTED_PROTOCOL:
    raise SystemExit(f"protocol mismatch: {manifest.get('protocol_id')}")

link = DATA_ROOT / "data" / "slimpajama_4x"
if link.is_symlink() or link.exists():
    if link.resolve() != candidate.parent.resolve():
        raise SystemExit(f"{link} already exists and points elsewhere; refusing to overwrite")
    link.unlink()
link.symlink_to(candidate.parent, target_is_directory=True)

print("selected_manifest:", candidate)
print("link:", link, "->", os.readlink(link))
print("protocol_id:", manifest["protocol_id"])
print("fingerprint:", manifest["fingerprint"])
print("max_token_id:", manifest["max_token_id"])
for name, split in manifest["splits"].items():
    print(name, "tokens=", split["tokens"], "sha256=", split["sha256"])

expected = Path(DATA_ROOT) / "data/slimpajama_4x/manifest.json"
assert expected.is_file(), expected
assert load_manifest(expected)["fingerprint"] == EXPECTED_FINGERPRINT
print("DATA_ROOT:", DATA_ROOT)
print("manifest path:", expected)
print("data validation: PASS")
PY
```

要求：

- fingerprint 必须等于 `30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18`。
- `load_manifest` 会校验 token 文件长度与 SHA-256；任何不一致都停止。
- 不复制大型 token 文件到 `/kaggle/working`，只创建软链接。

---

## 4. Dependency Check Cell

优先使用 Kaggle 已有环境。**不要重装 PyTorch/CUDA，不要执行 `pip install -r requirements/kaggle.txt`。**

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

python - <<'PY'
import importlib.util
import sys

import numpy
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("numpy:", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())

# 训练路径只需要 torch + numpy。其余包仅用于可选的辅助分析。
for name in ("tiktoken", "matplotlib", "bitsandbytes", "datasets", "pyarrow"):
    print(name, "available:", importlib.util.find_spec(name) is not None)

assert torch.cuda.is_available(), "CUDA is unavailable; stop and fix the Kaggle GPU runtime"
assert "T4" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print("dependency check: PASS")
PY
```

说明：

- 如果 `import torch, numpy` 失败，停止并检查 Kaggle 镜像；不要用 pip 替换 CUDA 版 torch。
- 如果 `bitsandbytes` 等可选包不存在，不影响本 FP32 Reference Muon 训练。

---

## 5. Snapshot Configuration Cell（9 节点）

从本轮代码仓库复制固定配置到 `/kaggle/working/muon_snapshot_config.json`，并验证 hash。

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

cp "$PLATFORM/configs/muon_snapshot_full_9nodes_s0.json" "$SNAPSHOT_CONFIG"

python - <<'PY'
import os
import sys
from pathlib import Path

PLATFORM = os.environ["PLATFORM"]
sys.path.insert(0, str(Path(PLATFORM) / "src"))
from optim.muon_snapshot_collector import load_snapshot_config

path = os.environ["SNAPSHOT_CONFIG"]
config = load_snapshot_config(path, total_updates=4096)
expected_updates = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
assert config.updates == expected_updates, config.updates
assert config.enabled is True
assert config.on_error == "disable"
assert config.max_total_bytes == 4 * 1024**3
assert config.min_free_bytes == 2 * 1024**3
assert config.config_hash == "d9e7c3df0cec1f048b27489f332c3f5dbe054fd0ddad5cb118138e9e83dcd73e"
print("config_path:", path)
print("config_hash:", config.config_hash)
print("snapshot_updates:", list(config.updates))
print("snapshot_count:", len(config.updates))
print("snapshot config validation: PASS")
PY
```

配置内容：

```json
{
  "schema_version": 1,
  "enabled": true,
  "updates": [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
  "on_error": "disable",
  "max_total_bytes": 4294967296,
  "min_free_bytes": 2147483648
}
```

要求：

- 本轮**只**采集这 9 个绝对 update；不包含 1–128。
- 不把 Stage 1 的 15 个 Snapshot 上传到 Kaggle。
- 新 run_dir 必须是全新目录；不要把旧 run 的 checkpoint/snapshot 放进新 run_dir。
- 配置 hash 必须为 `d9e7c3df0cec1f048b27489f332c3f5dbe054fd0ddad5cb118138e9e83dcd73e`。

---

## 6. Recipe, Parameter Grouping, Seed and Precision Cell

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

PYTHONPATH=src python - <<'PY'
import json
import os
from pathlib import Path

import torch

from config.recipe import load_recipe
from train_platform import build_model, parameter_groups
from optim.muon_snapshot_collector import load_snapshot_config

RECIPE = "recipes/mini_fp32_reference_muon_4x_s0.json"
EXPECTED_RECIPE_FINGERPRINT = "fda8bd1101831b1d9b54d7bbd389d762e8dae353a4fbd7dfb3dce79518081870"
EXPECTED_DATA_FINGERPRINT = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"

cfg = load_recipe(RECIPE)
print("recipe_fingerprint:", cfg["fingerprint"])
print("experiment:", cfg["experiment"])
print("derived:", cfg["derived"])
print("checkpoint:", cfg["checkpoint"])

assert cfg["fingerprint"] == EXPECTED_RECIPE_FINGERPRINT
assert cfg["experiment"]["seed"] == 0
assert cfg["experiment"]["data_seed"] == 1337
assert cfg["experiment"]["algorithm_seed"] == 2026
assert cfg["optimizer"]["name"] == "reference_muon"
assert cfg["optimizer"]["state_simulation"] == "none"
assert cfg["precision"]["compute"] == "fp32"
assert cfg["precision"]["parameter_dtype"] == "fp32"
assert cfg["precision"]["gradient_dtype"] == "fp32"
assert cfg["precision"]["tf32"] is False
assert cfg["precision"]["compile"] is False
assert cfg["precision"]["deterministic"] is True
assert cfg["derived"]["total_updates"] == 4096
assert cfg["derived"]["schedule_total_updates"] == 4096
assert cfg["derived"]["tokens_per_update"] == 8192
assert cfg["train"]["target_tokens"] == 33554432
assert cfg["checkpoint"]["every_updates"] == 128

model = build_model(cfg, torch.device("cpu"))
total_params = sum(p.numel() for p in model.parameters())
groups, records = parameter_groups(
    model, cfg["optimizer"]["weight_decay"], cfg["optimizer"]["name"])
muon_records = [r for r in records if r["group"] == "muon"]
muon_params = sum(r["numel"] for r in muon_records)
print("total_params:", total_params)
print("muon_matrices:", len(muon_records))
print("muon_params:", muon_params)
assert total_params == 29_938_560
assert len(muon_records) == 30
assert muon_params == 10_616_832
expected_shapes = {(1152, 384), (384, 384), (1024, 384), (384, 1024)}
for record in muon_records:
    assert tuple(record["shape"]) in expected_shapes, record
    assert record["name"].startswith("transformer.h."), record
del model

manifest_path = Path(os.environ["DATA_ROOT"]) / "data/slimpajama_4x/manifest.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["fingerprint"] == EXPECTED_DATA_FINGERPRINT
print("data_fingerprint:", manifest["fingerprint"])

snapshot_config = load_snapshot_config(os.environ["SNAPSHOT_CONFIG"], total_updates=4096)
assert snapshot_config.updates == (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
assert snapshot_config.config_hash == "d9e7c3df0cec1f048b27489f332c3f5dbe054fd0ddad5cb118138e9e83dcd73e"
print("snapshot_config_hash:", snapshot_config.config_hash)
print("recipe/precision/seed validation: PASS")
PY
```

说明：

- Recipe 文件、模型、优化器、学习率、训练参数全部冻结；不要为了让 Checkpoint 滚动保存而修改 recipe。
- `checkpoint.save_initial` / `save_final` 仍保留在 recipe 中，但新代码只写 `latest.pt`；`save_final` 不再产生 `final.pt`。
- `eval.attention_backend` 在当前实现中不生效；训练/评估使用 `precision.attention_backend`。这是已知配置问题，本轮不修改。

---

## 7. Preflight and Dry Run Cell

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

mkdir -p "$LOG_DIR"
DRY_LOG="$LOG_DIR/dry_run_$(date -u +%Y%m%dT%H%M%SZ).log"

python src/main.py \
  --recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --data-root "$DATA_ROOT" \
  --to-device cuda:0 \
  --dry-run 2>&1 | tee "$DRY_LOG"

python - <<'PY'
import os
import shutil
from pathlib import Path

free = shutil.disk_usage("/kaggle/working").free
snapshot_payload = 297_271_296
snapshot_count = 9
checkpoint_bytes = 320 * 1024**2
release_copy = snapshot_count * (snapshot_payload + 2 * 1024**2)
working_estimate = snapshot_count * (snapshot_payload + 2 * 1024**2) + checkpoint_bytes
pack_estimate = working_estimate + release_copy
print("working_free_GiB:", free / 1024**3)
print("run_estimate_GiB:", working_estimate / 1024**3)
print("with_release_copy_GiB:", pack_estimate / 1024**3)
assert pack_estimate + 2 * 1024**3 < free, (
    "not enough /kaggle/working for 9 snapshots + latest.pt + release copy + 2 GiB reserve")
print("storage preflight: PASS")
PY
```

Dry Run 输出必须包含：

```json
{
  "total_updates": 4096,
  "schedule_total_updates": 4096,
  "tokens_per_update": 8192,
  "optimizer": "reference_muon",
  "compute": "fp32",
  "resolved_device": "cuda:0",
  "manifest_exists": true
}
```

注意：

- `--dry-run` 不加载 `--snapshot-config`；Snapshot 配置已在 Cell 5/6 单独验证。
- 如果 `manifest_exists=false` 或 `capacity_error` 出现，先修复数据挂载/容量问题。
- 不传递 `--stop-at-update`。

---

## 8. Formal Training Cell（完整 4096 updates）

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  echo "RUN_DIR is not empty: $RUN_DIR" >&2
  echo "This run requires a fresh run directory. Do not mix Stage 1 artifacts." >&2
  exit 65
fi
mkdir -p "$RUN_DIR"

LOG_FILE="$LOG_DIR/${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

set +e
python src/main.py \
  --recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --to-device cuda:0 \
  --snapshot-config "$SNAPSHOT_CONFIG" \
  2>&1 | tee "$LOG_FILE" | grep --line-buffered -E '^\[(run|eval|ckpt|pause|done)\]|WARNING|ERROR|Traceback|Error|warning|error'
rc=${PIPESTATUS[0]}
set -e

echo "python_exit=$rc"
echo "log_file=$LOG_FILE"
exit "$rc"
```

说明：

- **没有 `--stop-at-update`**；本次从头训练到 4096。
- 没有任何编号 Checkpoint；每 128 updates 覆盖 `checkpoints/latest.pt`。
- update 4096 正常完成时状态为 `completed`，`latest.pt` 即最终 Checkpoint。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 是 CUDA 确定性 cuBLAS 所需的环境变量；recipe 已启用 `deterministic=true`。
- 预计运行时间：按 Stage 1 实测约 1.2 秒/update（含 eval），4096 updates 约 **1–1.5 小时**；以 Kaggle T4 实际为准。
- 不要使用 `platform/scripts/run_kaggle.sh`（不传递 `--snapshot-config`），也不要使用旧 notebook launcher。

预期关键行：

```text
[run] muon_30m_fp32_muon_s0_full | reference_muon | cuda:0 | fp32 | 4096 updates | schedule horizon 4096 | 33554432 tokens
[eval] step 0 | ...
[ckpt] latest.pt @ step 0
[ckpt] latest.pt @ step 128
[eval] step 128 | ...
[ckpt] latest.pt @ step 256
...
[ckpt] latest.pt @ step 4096
[eval] step 4096 | ...
[done] completed | 4096/4096 updates | 33554432 tokens | ...
```

---

## 9. Monitoring and Log Filtering Cell

训练期间可反复执行：

```bash
%%bash
source /kaggle/working/muon_env.sh

LOG_FILE="$LOG_DIR/latest.log"

echo "=== training process ==="
pgrep -af "python.*src/main.py" || true

echo "=== filtered log tail ==="
grep -E '^\[(run|eval|ckpt|pause|done)\]|WARNING|ERROR|Traceback|Error|warning|error' "$LOG_FILE" 2>/dev/null | tail -n 40 || true

echo "=== checkpoints (must contain latest.pt only) ==="
ls -lh "$RUN_DIR/checkpoints" 2>/dev/null || true

echo "=== snapshots ==="
find "$RUN_DIR/snapshots" -maxdepth 1 -name 'update_*.pt' 2>/dev/null | sort
echo "snapshot_pt_count=$(find "$RUN_DIR/snapshots" -maxdepth 1 -name 'update_*.pt' 2>/dev/null | wc -l)"
echo "snapshot_manifest_count=$(find "$RUN_DIR/snapshots" -maxdepth 1 -name 'update_*.manifest.json' 2>/dev/null | wc -l)"
du -sh "$RUN_DIR/snapshots" 2>/dev/null || true

echo "=== snapshot errors ==="
tail -n 20 "$RUN_DIR/snapshot_errors.jsonl" 2>/dev/null || echo "(none)"

echo "=== summary (if finished) ==="
cat "$RUN_DIR/summary.json" 2>/dev/null || echo "(not finished)"

echo "=== disk ==="
df -h /kaggle/working
```

过滤规则保留 `[run]`、`[eval]`、`[ckpt]`、`[pause]`、`[done]` 以及 warning/error，过滤逐 update 的 `[train]` 输出。

---

## 10. Post-Run Snapshot and Checkpoint Acceptance Cell

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

run_dir = Path(os.environ["RUN_DIR"])
expected_updates = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
expected_hash = "d9e7c3df0cec1f048b27489f332c3f5dbe054fd0ddad5cb118138e9e83dcd73e"
expected_recipe = "fda8bd1101831b1d9b54d7bbd389d762e8dae353a4fbd7dfb3dce79518081870"
expected_data = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"

summary = json.loads((run_dir / "summary.json").read_text())
print("status:", summary["status"])
print("completed_updates:", summary["completed_updates"])
print("final_validation_nll:", summary["final_validation_nll"])
assert summary["status"] == "completed"
assert summary["completed_updates"] == 4096
assert summary["total_updates"] == 4096
assert summary["schedule_total_updates"] == 4096
assert summary["divergence"] is None
assert summary["recipe_fingerprint"] == expected_recipe
assert summary["data_fingerprint"] == expected_data
assert summary["optimizer_name"] == "reference_muon"
assert summary["compute_precision"] == "fp32"
assert summary["git_commit"] == os.environ["TARGET_COMMIT"]
assert summary["final_validation_nll"] is not None

status = json.loads((run_dir / "snapshot_status.json").read_text())
assert status["committed_updates"] == expected_updates
assert status["conflicted_updates"] == []
assert status["missed_updates"] == []
assert status["error_count"] == 0
assert status["enabled"] is True and status["disabled"] is False

stored = json.loads((run_dir / "snapshot_config.json").read_text())
assert stored["config_hash"] == expected_hash
assert stored["config"]["updates"] == expected_updates

snapshots = run_dir / "snapshots"
pt_updates = sorted(int(path.stem.split("_")[1]) for path in snapshots.glob("update_*.pt"))
manifest_updates = sorted(
    int(path.name.split(".")[0].split("_")[1])
    for path in snapshots.glob("update_*.manifest.json"))
assert pt_updates == expected_updates, pt_updates
assert manifest_updates == expected_updates, manifest_updates
assert len(list(snapshots.glob("update_*.pt"))) == 9
assert len(list(snapshots.glob("update_*.manifest.json"))) == 9

for update in expected_updates:
    pt = snapshots / f"update_{update:06d}.pt"
    mp = snapshots / f"update_{update:06d}.manifest.json"
    manifest = json.loads(mp.read_text())
    digest = hashlib.sha256(pt.read_bytes()).hexdigest()
    assert digest == manifest["tensors_file"]["sha256"], update
    assert pt.stat().st_size == manifest["tensors_file"]["nbytes"], update
    assert manifest["status"] == "committed"
    assert manifest["update"] == update
    assert manifest["identities"]["snapshot_config_hash"] == expected_hash
    assert manifest["identities"]["recipe_fingerprint"] == expected_recipe
    assert manifest["identities"]["data_fingerprint"] == expected_data
    assert manifest["source"]["commit"] == os.environ["TARGET_COMMIT"]
    assert manifest["integrity"]["expected_parameter_count"] == 30
    assert manifest["integrity"]["observed_parameter_count"] == 30
    assert manifest["integrity"]["expected_role_slots"] == 240
    assert manifest["integrity"]["observed_role_slots"] == 240
    assert manifest["integrity"]["unique_tensor_count"] == 210
    assert manifest["integrity"]["alias_count"] == 30
    assert manifest["integrity"]["tensors_file_sha256_ok"] is True
    assert manifest["integrity"]["no_orphan_tensors"] is True
    assert all(entry["first_step"] is False for entry in manifest["parameters"].values())
    print(f"snapshot {update}: OK", manifest["tensors_file"]["sha256"])

checkpoints = run_dir / "checkpoints"
names = sorted(path.name for path in checkpoints.iterdir())
assert names == ["latest.pt"], names
assert not (checkpoints / "initial.pt").exists()
assert not (checkpoints / "final.pt").exists()
assert not list(checkpoints.glob("update_*.pt"))
assert not list(checkpoints.glob("*.tmp"))

latest = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=False)
assert latest["completed_updates"] == 4096
assert latest["processed_target_tokens"] == 4096 * 8192
assert latest["scientific_fingerprint"] == expected_recipe
assert latest["data_fingerprint"] == expected_data
assert {"model", "optimizer", "sampler", "rng"} <= set(latest)

# Final snapshot must agree exactly with the final rolling checkpoint.
manifest = json.loads((snapshots / "update_004096.manifest.json").read_text())
payload = torch.load(snapshots / "update_004096.pt", map_location="cpu", weights_only=True)
for name in manifest["model"]["muon_parameter_order"]:
    entry = manifest["parameters"][name]
    wa = entry["roles"]["W_after"]
    w = payload["tensors"][wa["tensor_id"]]
    assert torch.equal(w, latest["model"][name]), name
    persisted = entry["roles"]["M_t_persisted"]
    canonical = entry["roles"][persisted["alias_of"]]
    momentum = payload["tensors"][canonical["tensor_id"]]
    state_index = entry["optimizer_param_index"]
    assert torch.equal(momentum,
                       latest["optimizer"]["state"][state_index]["muon_momentum"]), name

events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
train_updates = [e["completed_updates"] for e in events if e["event_type"] == "train"]
checkpoint_updates = [e["completed_updates"] for e in events
                      if e["event_type"] == "checkpoint"]
lifecycles = [e for e in events if e["event_type"] == "lifecycle"]
assert train_updates == list(range(1, 4097))
assert checkpoint_updates == list(range(128, 4097, 128)), checkpoint_updates
assert len(checkpoint_updates) == 32
assert lifecycles[-1]["phase"] == "completed"
assert lifecycles[-1]["completed_updates"] == 4096
assert not any(e["event_type"] == "divergence" for e in events)

errors = run_dir / "snapshot_errors.jsonl"
if errors.exists():
    print("snapshot_errors tail:")
    print("\n".join(errors.read_text().splitlines()[-20:]))
    raise SystemExit("snapshot_errors.jsonl must be absent or empty")
print("snapshot_errors: none")
print("post-run acceptance: PASS")
print("snapshot_count:", len(pt_updates))
print("checkpoint_policy: single rolling latest.pt")
PY

echo "=== run tree ==="
find "$RUN_DIR" -maxdepth 2 -type f | sort
```

要求：

- 9 个 `.pt` + 9 个 Manifest，节点精确匹配。
- 每个 Manifest：30 参数、240 role slot、210 唯一 Tensor、30 alias、文件 SHA 正确。
- `checkpoints/` 只有 `latest.pt`；`completed_updates == 4096`。
- update 4096 Snapshot 的 `W_after` 与 `M_t_persisted` 与 `latest.pt` 逐元素一致。
- `metrics.jsonl` 中 checkpoint 事件恰好 32 个（128, 256, ..., 4096），没有重复的 4096 写入。
- 没有 Snapshot 采集错误。

---

## 11. Packaging and Release Preparation Cell

本 Cell 只准备打包文件，**不执行上传**，不改动本地 Stage 1 归档。

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/package_metadata" "$RELEASE_DIR/runs/$RUN_ID/snapshots" \
         "$RELEASE_DIR/runs/$RUN_ID/checkpoints"

cp "$RUN_DIR/summary.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/snapshot_config.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/snapshot_status.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/metrics.jsonl" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/resolved_config.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/parameters.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/data_manifest.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/source.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/environment.json" "$RELEASE_DIR/runs/$RUN_ID/"
cp "$RUN_DIR/checkpoints/latest.pt" "$RELEASE_DIR/runs/$RUN_ID/checkpoints/"

for update in 256 384 512 768 1024 1536 2048 3072 4096; do
  cp "$RUN_DIR/snapshots/update_$(printf '%06d' "$update").pt" \
     "$RELEASE_DIR/runs/$RUN_ID/snapshots/"
  cp "$RUN_DIR/snapshots/update_$(printf '%06d' "$update").manifest.json" \
     "$RELEASE_DIR/runs/$RUN_ID/snapshots/"
done

cp "$SNAPSHOT_CONFIG" "$RELEASE_DIR/package_metadata/snapshot_config.json"
cp "$DATA_ROOT/data/slimpajama_4x/manifest.json" "$RELEASE_DIR/package_metadata/dataset_manifest.json"

python - <<'PY'
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch

release = Path(os.environ["RELEASE_DIR"])
run_dir = Path(os.environ["RUN_DIR"])
updates = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

summary = json.loads((run_dir / "summary.json").read_text())
source = json.loads((run_dir / "source.json").read_text())
environment = json.loads((run_dir / "environment.json").read_text())
config = json.loads((run_dir / "snapshot_config.json").read_text())
data_manifest = json.loads((release / "package_metadata/dataset_manifest.json").read_text())

provenance = {
    "archive_schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": {
        "name": os.environ["RUN_ID"],
        "completed_updates": summary["completed_updates"],
        "target_updates": summary["total_updates"],
        "optimizer": summary["optimizer_name"],
        "compute": summary["compute_precision"],
        "seed": summary["seed"],
        "data_seed": summary["data_seed"],
        "algorithm_seed": summary["algorithm_seed"],
    },
    "git": {
        "repository": "https://github.com/SiennaDai/muon_research",
        "commit": source["upstream_commit"],
        "working_tree_status": "" if source["dirty"] is False else "dirty",
    },
    "recipe": {
        "path": "muon_research/platform/recipes/mini_fp32_reference_muon_4x_s0.json",
        "fingerprint": summary["recipe_fingerprint"],
    },
    "snapshot": {
        "config_hash": config["config_hash"],
        "snapshot_updates": updates,
        "snapshot_count": len(updates),
        "config_file": "package_metadata/snapshot_config.json",
        "stage1_config_hash_for_merge": "9a073ba2e5ed8d9e00d857c2d0f64892633fcb2994c994ea4bc1d06b97c24b0e",
        "merge_note": "Stage 1 updates 1-128 were collected under the 24-node config hash 9a073ba2...; this stage's updates 256-4096 use d9e7c3df.... Both are valid and must be preserved per manifest.",
    },
    "dataset": {
        "manifest_fingerprint": data_manifest["fingerprint"],
        "manifest_sha256": hashlib.sha256(
            (release / "package_metadata/dataset_manifest.json").read_bytes()).hexdigest(),
    },
    "environment": {
        "python": environment["python"],
        "pytorch": environment["torch"],
        "cuda": environment["torch_cuda"],
        "numpy": environment["numpy"],
        "gpu": environment["cuda_devices"][0] if environment["cuda_devices"] else None,
    },
    "training_summary": summary,
    "local_merge_target": {
        "stage1_archive": "D:/muon_research/artifacts/stage1_30m_fp32_muon_s0",
        "expected_merged_nodes": 24,
        "stage1_updates": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128],
        "stage2_updates": updates,
    },
}
(release / "package_metadata/provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n")

files = sorted(p for p in release.rglob("*") if p.is_file())
inventory = [{"path": str(p.relative_to(release)), "size_bytes": p.stat().st_size}
             for p in files]
(release / "package_metadata/run_inventory.json").write_text(
    json.dumps(inventory, indent=2, sort_keys=True) + "\n")

sums = []
for path in sorted(release.rglob("*")):
    if path.is_file() and path.name != "SHA256SUMS":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.relative_to(release)}")
(release / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print("release_dir:", release)
print("release_files:", len(inventory))
print("provenance_readme: package_metadata/provenance.json")
print("SHA256SUMS entries:", len(sums))
PY

tar -C "$(dirname "$RELEASE_DIR")" -cf "${RELEASE_DIR}.tar" "$(basename "$RELEASE_DIR")"
ls -lh "${RELEASE_DIR}.tar"
echo "Packaging complete. Do not upload until the merged 24-node validation passes."
```

合并与发布要求：

- 本轮 Release 只包含 update 256–4096 的 9 个 Snapshot、`latest.pt` 和 metadata；**不要**包含 Stage 1 的 15 个 Snapshot。
- 本地合并时，把 Stage 1 归档的 15 个 Snapshot 与本轮 9 个放在同一逻辑数据集中，保持原始文件名和 Manifest 不变。
- 合并后应有 **24 个 `.pt` + 24 个 Manifest**，节点为：
  `1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096`。
- Stage 1 Manifest 的 `snapshot_config_hash = 9a073ba2...`；本轮 Manifest 的 `snapshot_config_hash = d9e7c3df...`。两者都正确，不要为了统一而修改 Manifest。
- 上传前在本地重新运行 Stage 1 验收脚本，并额外验证本轮 9 个 Snapshot 的 Tensor/alias/checkpoint 一致性。

---

## 12. Resume and Failure Handling

如果训练被 Kaggle 超时或 `KeyboardInterrupt` 中断：

- `checkpoints/latest.pt` 仍是最后一个完整 update 边界（update 0 或 128 的倍数）。
- 不保存不完整 update 的模型/优化器状态。
- 使用同一个 `RUN_DIR`、同一个 `SNAPSHOT_CONFIG`、同一个 recipe 恢复：

```bash
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

python src/main.py \
  --recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --to-device cuda:0 \
  --resume "$RUN_DIR/checkpoints/latest.pt" \
  --snapshot-config "$SNAPSHOT_CONFIG"
```

- 不传递 `--stop-at-update`。
- Collector 会扫描已有 Manifest 并跳过已提交节点。若某个配置节点早于恢复点却没有 Manifest，Collector 会把它记录为 `missed_updates`（不会伪造 Snapshot）；正常按 128 边界保存 `latest.pt` 时不会出现这种情况。
- 不要删除、重命名或手工替换 `latest.pt`；不要生成编号 Checkpoint 来“补齐”。
- 如果 Snapshot 配置在 resume 时 hash 不一致，Collector 会拒绝启动；不要修改配置来绕过。
- 如果出现 `snapshot_errors.jsonl` 或 `snapshot_status.conflicted_updates`，停止并保留现场，不要覆盖已有 Snapshot。

---

## 13. Frozen Configuration and Known Issues

1. **Recipe 冻结**：不要修改模型、优化器、学习率、schedule、precision、batch、seed 或 checkpoint 间隔；Recipe fingerprint 必须保持 `fda8bd11...`。
2. **数据冻结**：Dataset fingerprint 必须为 `30152c9b...`；不要下载或重新生成 token。
3. **只采集 9 个节点**：本轮配置不包含 update 1–128；不要把这些节点重新加入配置或上传旧 Snapshot。
4. **单文件 Checkpoint**：只允许 `checkpoints/latest.pt`；不要额外生成 `initial.pt`、`update_XXXXXX.pt`、`final.pt`。
5. **T4 确定性**：`CUBLAS_WORKSPACE_CONFIG=:4096:8`；若出现 deterministic CUDA 错误，停止并报告，不要关闭确定性。
6. **`eval.attention_backend` 不生效**：当前实现使用 `precision.attention_backend`；这是已审计的配置问题，本轮不修改。
7. **辅助 AdamW weight decay**：1D RMSNorm/tied embedding 也承受 0.1 weight decay；这是已审计的配置问题，本轮不修改。
8. **旧 notebook / `run_kaggle.sh` 不可用**：使用本文件的直接 CLI 命令。
9. **不要上传结果**：本文件只准备 release；上传由研究者确认合并验收后手动执行。
