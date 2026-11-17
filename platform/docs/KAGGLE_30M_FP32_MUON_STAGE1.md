# Kaggle 30M FP32 Muon Reference Training — Stage 1 Runbook

> 目标：在 Kaggle T4 上运行 30M LLaMA + Reference Muon + FP32，完整 horizon 4096 updates，第一阶段在 update 128 正常暂停，并生成 update 1–128 的 15 份研究 Snapshot。
>
> 本文件只提供训练前准备与可复制 Cells；**不实际启动训练、不下载数据、不修改源码/Recipe、不提交或推送 GitHub**。
>
> 当前项目根目录：`/mnt/d/muon_research`；训练平台：`/mnt/d/muon_research/platform`。

---

## 0. 当前项目状态与必须先完成的 GitHub 发布

### 0.1 实际检查结果

| 项目 | 当前状态 |
|---|---|
| Git 根目录 | `/mnt/d/muon_research` |
| 分支 | `main` |
| HEAD | `2c46add3f84703dd91e73bfc89b7f31b02cd6b62` |
| HEAD subject | `Initialize Muon research platform from upstream baseline` |
| Git remote | **无**（`git remote -v` 为空） |
| Snapshot Collector | **已在工作树实现，但尚未提交** |
| ReferenceMuon hook | **已在工作树实现，但尚未提交** |
| 训练循环集成 | **已在工作树实现，但尚未提交** |
| 当前 HEAD 是否包含 Collector | **否**；Kaggle 不能用 `2c46add` 运行 Snapshot |
| 30M Recipe | `platform/recipes/mini_fp32_reference_muon_4x_s0.json`（6 层 × 384，29,938,560 参数） |
| 数据契约 | `data/slimpajama_4x/manifest.json`；期望 fingerprint `30152c9b...` |
| 数据准备脚本 | `platform/scripts/prepare_data.py`（本阶段不运行；数据来自 Kaggle Input） |
| 现有 Kaggle 文档 | `platform/docs/KAGGLE_EXPERIMENT_GUIDE.zh-CN.md`、`platform/docs/OPERATION_MANUAL.zh-CN.md`（其中旧 repo/路径需按本文更新） |

### 0.2 未提交的语义修改（`git diff --ignore-space-at-eol --stat`）

```text
platform/src/experiment_io.py
platform/src/main.py
platform/src/optim/muon_reference.py
platform/src/train_platform.py
platform/tests/conftest.py
```

### 0.3 未跟踪的新文件

```text
platform/src/optim/muon_snapshot_collector.py
platform/tests/test_experiment_io_provenance.py
platform/tests/test_snapshot_collector.py
platform/tests/test_snapshot_invariance.py
platform/tests/test_snapshot_io.py
platform/tests/test_snapshot_resume.py
platform/docs/SNAPSHOT_COLLECTOR_DESIGN.md
platform/docs/KAGGLE_30M_FP32_MUON_STAGE1.md  ← 本文档
```

### 0.4 换行差异警告

`git status` 会把大量 `platform/` 文件显示为 `M`，但 `git diff --ignore-space-at-eol --stat` 只显示上述 5 个语义修改文件。其余大量 `M` 是 Windows/WSL checkout 的 CRLF/LF 差异。

**不要直接执行 `git add -A` 或 `git add platform/`。** 这会把所有换行差异一起提交，掩盖真实修改。请只 add 上面列出的语义修改文件和新文件。

### 0.5 必须先发布新的 GitHub 仓库

当前项目**没有 remote**。Kaggle 无法从 `/mnt/d` 读取代码，因此必须先由研究者创建新仓库并推送。

**不要使用旧项目的 `https://github.com/SiennaDai/benchmark-experiments.git`。** 那是上游旧仓库，不是本项目的新仓库。

研究者手动执行（把 `<NEW_EMPTY_REPO_URL>` 替换为新建的空 GitHub 仓库 URL）：

```bash
cd /mnt/d/muon_research

# 1. 确认当前状态
git -c safe.directory=/mnt/d/muon_research status --short
git -c safe.directory=/mnt/d/muon_research diff --ignore-space-at-eol --stat

# 2. 只 stage 本项目相关文件
git add \
  platform/src/experiment_io.py \
  platform/src/main.py \
  platform/src/optim/muon_reference.py \
  platform/src/optim/muon_snapshot_collector.py \
  platform/src/train_platform.py \
  platform/tests/conftest.py \
  platform/tests/test_experiment_io_provenance.py \
  platform/tests/test_snapshot_collector.py \
  platform/tests/test_snapshot_invariance.py \
  platform/tests/test_snapshot_io.py \
  platform/tests/test_snapshot_resume.py \
  platform/docs/SNAPSHOT_COLLECTOR_DESIGN.md \
  platform/docs/KAGGLE_30M_FP32_MUON_STAGE1.md

# 3. 提交
git commit -m "Add FP32 Muon snapshot collector and Kaggle stage-1 runbook"

# 4. 绑定新仓库（仅当 remote 为空时）
git remote add origin <NEW_EMPTY_REPO_URL>
git push -u origin main

# 5. 记录固定 commit SHA，供 Kaggle Cell 使用
git rev-parse HEAD
```

发布后：

- 把 `git rev-parse HEAD` 的输出填入 Kaggle Cell 2 的 `TARGET_COMMIT`。
- `TARGET_COMMIT` 必须是 **包含 Snapshot Collector 的新 commit**，不是 `2c46add`。
- 如果 GitHub 仓库是私有的，Kaggle clone 需要 PAT/Secret；建议本阶段使用受控的 private repo，并不要把 token 打印到 notebook 输出中。
- 如果 `origin` 已存在但指向旧仓库，先确认它不是本项目要用的仓库；不要直接 push 到旧项目。

> Kaggle 代码路径将是 `/kaggle/working/muon_research`，训练平台在其下的 `platform/`。后续所有命令都必须先进入 `platform/`。

---

## 1. Kaggle Environment Cell

```bash
%%bash
set -euo pipefail
echo "=== nvidia-smi ==="
nvidia-smi
echo "=== python / torch ==="
python - <<'PY'
import sys, shutil
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
print("working free bytes:", shutil.disk_usage("/kaggle/working").free)
print("working free GiB:", shutil.disk_usage("/kaggle/working").free / 1024**3)
PY
echo "=== disk ==="
df -h /kaggle/working /kaggle/input || true
echo "=== input mounts ==="
ls -la /kaggle/input
```

要求：

- `torch.cuda.is_available() == True`
- GPU 名称包含 `T4`
- `/kaggle/working` 可用空间足够（预检 Cell 会计算）
- `/kaggle/input` 至少能看到一个数据集挂载点

---

## 2. Code Acquisition Cell

先在 Kaggle Notebook 设置中开启 **GPU** 和 **Internet**。

```bash
%%bash
set -euo pipefail

# ===== 研究者必须填写 =====
NEW_REPO="<NEW_EMPTY_REPO_URL>"
TARGET_COMMIT="<COMMIT_SHA_FROM_GIT_REV_PARSE_HEAD>"
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

# Ignore untracked caches/artifacts; tracked modifications must be empty.
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

# 写一个可被后续 %%bash cells source 的环境文件
cat > /kaggle/working/muon_env.sh <<EOF
export REPO_ROOT="$REPO_ROOT"
export PLATFORM="$PLATFORM"
export TARGET_COMMIT="$TARGET_COMMIT"
export DATA_ROOT="/kaggle/working/muon_data"
export SNAPSHOT_CONFIG="/kaggle/working/muon_snapshot_config.json"
export RUN_ROOT="/kaggle/working/runs"
export RUN_ID="muon_30m_fp32_s0_stage1"
export RUN_DIR="\$RUN_ROOT/\$RUN_ID"
export LOG_DIR="/kaggle/working/logs"
EOF

mkdir -p /kaggle/working/logs

# 训练路径只需要 torch/numpy。不要重装 torch/CUDA。
python - <<'PY'
import importlib.util, sys
import torch, numpy
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("numpy:", numpy.__version__)
missing = [m for m in ("tiktoken", "matplotlib", "bitsandbytes")
           if importlib.util.find_spec(m) is None]
print("optional missing:", missing)
PY
```

说明：

- 本 FP32 Muon 训练路径不需要 `bitsandbytes`、`tiktoken` 或 `matplotlib`。不要执行 `pip install -r requirements/kaggle.txt`，它包含 `bitsandbytes` 并可能改动 Kaggle 镜像的 numpy；**不要重装 PyTorch/CUDA**。
- 如果 `import torch, numpy` 失败，先停止并检查 Kaggle 环境；不要用 pip 替换 CUDA 版 torch。
- 如果 `git clone` 因 private repo 失败，使用 Kaggle Secret/PAT，并避免把 token 写进输出。

---

## 3. Dataset Mount and Validation Cell

本 Cell 不下载、不重新生成 token；只在 `/kaggle/input` 中寻找已挂载的 `slimpajama_4x/manifest.json`，验证 manifest/hash，然后创建软链接。

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

mkdir -p "$DATA_ROOT/data"

python - <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = os.environ["REPO_ROOT"]
PLATFORM = os.environ["PLATFORM"]
DATA_ROOT = Path(os.environ["DATA_ROOT"])
sys.path.insert(0, str(Path(PLATFORM) / "src"))

from data.frozen_tokens import load_manifest

EXPECTED_DATA_FINGERPRINT = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"
EXPECTED_PROTOCOL = "slimpajama-hash-split-v1"

candidates = sorted(
    p for p in Path("/kaggle/input").rglob("manifest.json")
    if p.parent.name == "slimpajama_4x"
)
if not candidates:
    raise SystemExit(
        "No /kaggle/input/**/slimpajama_4x/manifest.json found. "
        "Add the approved frozen SlimPajama 4x Kaggle dataset as Input; "
        "do not download or regenerate tokens."
    )

chosen = None
seen = []
for candidate in candidates:
    try:
        manifest = load_manifest(candidate)
    except Exception as exc:
        seen.append((str(candidate), f"invalid: {exc}"))
        continue
    seen.append((str(candidate), manifest.get("fingerprint")))
    if manifest.get("fingerprint") == EXPECTED_DATA_FINGERPRINT:
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
if link.is_symlink():
    link.unlink()
elif link.exists():
    raise SystemExit(
        f"{link} exists and is not a symlink; refusing to overwrite. "
        "Remove it manually only after confirming it is not the mounted data."
    )
link.symlink_to(candidate.parent, target_is_directory=True)

print("selected_manifest:", candidate)
print("link:", link, "->", os.readlink(link))
print("protocol_id:", manifest["protocol_id"])
print("fingerprint:", manifest["fingerprint"])
print("max_token_id:", manifest.get("max_token_id"))
for name, split in manifest["splits"].items():
    print(name, "tokens=", split["tokens"], "sha256=", split["sha256"])

expected = Path(DATA_ROOT) / "data/slimpajama_4x/manifest.json"
assert expected.is_file(), expected
assert load_manifest(expected)["fingerprint"] == EXPECTED_DATA_FINGERPRINT
print("DATA_ROOT:", DATA_ROOT)
print("manifest path:", expected)
print("data validation: PASS")
PY
```

要求：

- 必须找到已挂载的 `slimpajama_4x/manifest.json`。
- fingerprint 必须等于 `30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18`（来自 `platform/tests/test_4x_protocol.py`）。
- `load_manifest` 会校验 token 文件长度和 SHA-256；任何不一致都停止。
- 如果实际挂载路径是 `/kaggle/input/<dataset>/data/slimpajama_4x/manifest.json` 或 `/kaggle/input/<dataset>/slimpajama_4x/manifest.json`，本 Cell 都会找到并建立统一的 `DATA_ROOT=/kaggle/working/muon_data`。
- 不复制大型 token 文件到 `/kaggle/working`；只创建软链接。

---

## 4. Snapshot Configuration Cell

生成完整 24 节点配置，覆盖未来 4096 updates；第一阶段只运行到 128，因此预期前 15 份。

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

cat > "$SNAPSHOT_CONFIG" <<'JSON'
{
  "schema_version": 1,
  "enabled": true,
  "updates": [
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56,
    64, 80, 96, 112, 128,
    256, 384, 512, 768, 1024,
    1536, 2048, 3072, 4096
  ],
  "on_error": "disable",
  "max_total_bytes": 8589934592,
  "min_free_bytes": 2147483648
}
JSON

python - <<'PY'
import os
import sys
from pathlib import Path

PLATFORM = os.environ["PLATFORM"]
sys.path.insert(0, str(Path(PLATFORM) / "src"))
from optim.muon_snapshot_collector import load_snapshot_config

path = os.environ["SNAPSHOT_CONFIG"]
config = load_snapshot_config(path, total_updates=4096)
expected_updates = (
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56,
    64, 80, 96, 112, 128,
    256, 384, 512, 768, 1024,
    1536, 2048, 3072, 4096,
)
assert config.updates == expected_updates
assert len(config.updates) == 24
assert len([u for u in config.updates if u <= 128]) == 15
assert len([u for u in config.updates if u > 128]) == 9
assert config.max_total_bytes == 8 * 1024**3
assert config.min_free_bytes == 2 * 1024**3
assert config.config_hash == "9a073ba2e5ed8d9e00d857c2d0f64892633fcb2994c994ea4bc1d06b97c24b0e"
print("config_path:", path)
print("config_hash:", config.config_hash)
print("total_landmarks:", len(config.updates))
print("stage1_landmarks:", len([u for u in config.updates if u <= 128]))
print("stage1_updates:", [u for u in config.updates if u <= 128])
print("snapshot config validation: PASS")
PY
```

**不要为了第一阶段只跑 128 而删掉 256–4096 的节点。** 后续 resume 必须使用同一个 24 节点配置和同一个 config hash；否则 Collector 会在启动时拒绝不一致的配置。

---

## 5. Recipe and Preflight Verification Cell

### 5.1 当前 `src/main.py` 实际支持的 CLI 参数

```text
--recipe PATH                 required
--dry-run
--run-dir PATH
--resume PATH
--max-wall-seconds FLOAT
--stop-at-update INT
--snapshot-config PATH
--data-root PATH              default: platform root
--to-device {auto,cpu,cuda,cuda:N}
--to_device                   alias of --to-device
```

本阶段使用：

```text
--recipe recipes/mini_fp32_reference_muon_4x_s0.json
--data-root /kaggle/working/muon_data
--run-dir /kaggle/working/runs/muon_30m_fp32_s0_stage1
--to-device cuda:0
--stop-at-update 128
--snapshot-config /kaggle/working/muon_snapshot_config.json
```

`--stop-at-update` 只限制执行长度；它不会改变 recipe 的 `schedule_total_updates=4096`。


```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

PYTHONPATH=src python - <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from config.recipe import load_recipe
from train_platform import build_model, parameter_groups

RECIPE = "recipes/mini_fp32_reference_muon_4x_s0.json"
EXPECTED_RECIPE_FINGERPRINT = "fda8bd1101831b1d9b54d7bbd389d762e8dae353a4fbd7dfb3dce79518081870"
EXPECTED_DATA_FINGERPRINT = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"

cfg = load_recipe(RECIPE)
print("recipe:", RECIPE)
print("recipe_fingerprint:", cfg["fingerprint"])
print("experiment:", cfg["experiment"])
print("derived:", cfg["derived"])
print("precision:", cfg["precision"])
print("optimizer:", cfg["optimizer"])
print("schedule:", cfg["schedule"])
print("checkpoint:", cfg["checkpoint"])
print("logging:", cfg["logging"])

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
assert cfg["data"]["manifest"] == "data/slimpajama_4x/manifest.json"
assert cfg["data"]["allow_repeated_epochs"] is False
assert cfg["checkpoint"]["every_updates"] == 128
assert cfg["checkpoint"]["save_initial"] is True
assert cfg["checkpoint"]["save_final"] is True

# 30M model and Muon group
model = build_model(cfg, torch.device("cpu"))
total_params = sum(p.numel() for p in model.parameters())
groups, records = parameter_groups(model, cfg["optimizer"]["weight_decay"], cfg["optimizer"]["name"])
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
print("muon_matrix_names:")
for record in muon_records:
    print(" ", record["name"], record["shape"])
del model

# T4 and environment
assert torch.cuda.is_available(), "CUDA is not available"
gpu_name = torch.cuda.get_device_name(0)
assert "T4" in gpu_name, f"expected Kaggle T4, got {gpu_name}"
print("gpu:", gpu_name)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)

# Data fingerprint
manifest_path = Path(os.environ["DATA_ROOT"]) / "data/slimpajama_4x/manifest.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["fingerprint"] == EXPECTED_DATA_FINGERPRINT
print("data_fingerprint:", manifest["fingerprint"])

# Snapshot config
sys.path.insert(0, str(Path(os.getcwd()) / "src"))
from optim.muon_snapshot_collector import load_snapshot_config
snapshot_config = load_snapshot_config(os.environ["SNAPSHOT_CONFIG"], total_updates=4096)
assert len(snapshot_config.updates) == 24
assert len([u for u in snapshot_config.updates if u <= 128]) == 15
print("snapshot_config_hash:", snapshot_config.config_hash)

# First-stage disk estimate (conservative, actual 30M snapshot payload 283.5 MiB)
payload_bytes = 297_271_296
snapshot_overhead = 2 * 1024**2
checkpoint_estimate = 400 * 1024**2
first_stage_estimate = 15 * (payload_bytes + snapshot_overhead) + 4 * checkpoint_estimate
full_snapshot_estimate = 24 * (payload_bytes + snapshot_overhead)
free = shutil.disk_usage("/kaggle/working").free
print("estimated_stage1_bytes:", first_stage_estimate)
print("estimated_stage1_GiB:", first_stage_estimate / 1024**3)
print("estimated_full_snapshot_GiB:", full_snapshot_estimate / 1024**3)
print("working_free_GiB:", free / 1024**3)
assert full_snapshot_estimate < 8 * 1024**3, "24 snapshots exceed max_total_bytes"
assert free > first_stage_estimate + 2 * 1024**3, (
    "not enough /kaggle/working space for 15 snapshots + checkpoints + 2 GiB reserve"
)
print("preflight: PASS")
PY
```

说明：

- 当前 recipe 是 `platform/recipes/mini_fp32_reference_muon_4x_s0.json`；它不是文件名里的“30M”，但实际模型是 **29,938,560 参数**，Muon 组 **30 个矩阵 / 10,616,832 参数**。
- `schedule_total_updates == 4096`；`stop_at_update=128` 只是运行时 gate，不能改 recipe/scheduler horizon。
- `eval.attention_backend` 在当前实现中不生效；训练/评估都使用 `precision.attention_backend`。这是已知配置问题，不修改 recipe；预检只记录。
- 辅助 AdamW 的 weight decay 会作用于 1D RMSNorm/tied embedding（审计遗留问题）；本阶段不改 recipe。

---

## 6. Dry Run Cell

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

mkdir -p "$LOG_DIR"
DRY_LOG="$LOG_DIR/dry_run_$(date -u +%Y%m%dT%H%M%SZ).log"

set +e
python src/main.py \
  --recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --data-root "$DATA_ROOT" \
  --to-device cuda:0 \
  --dry-run \
  2>&1 | tee "$DRY_LOG"
rc=${PIPESTATUS[0]}
set -e

echo "dry_run_exit=$rc"
echo "dry_run_log=$DRY_LOG"
exit "$rc"
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

如果 `dry_run_exit != 0`，停止，不要启动训练。

注意：

- `main.py --dry-run` 不会加载 `--snapshot-config`；Snapshot 配置已由 Cell 4/5 单独验证。
- Dry Run 不会创建 run directory，也不会训练。
- 如果 `manifest_exists=false` 或 `capacity_error` 出现，先修复数据挂载/容量问题。

---

## 7. Formal Training Cell (Stage 1, stop at update 128)

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh
cd "$PLATFORM"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  echo "RUN_DIR is not empty: $RUN_DIR" >&2
  echo "Stage 1 expects a fresh run directory. Do not overwrite an existing run." >&2
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
  --stop-at-update 128 \
  --snapshot-config "$SNAPSHOT_CONFIG" \
  2>&1 | tee "$LOG_FILE" | grep --line-buffered -E '^\[(run|eval|ckpt|pause|done)\]|WARNING|ERROR|Traceback|Error|status|warning|error'
rc=${PIPESTATUS[0]}
set -e

echo "python_exit=$rc"
echo "log_file=$LOG_FILE"
exit "$rc"
```

预期屏幕关键行：

```text
[run] muon_30m_fp32_s0_stage1 | reference_muon | cuda:0 | fp32 | 4096 updates | schedule horizon 4096 | 33554432 tokens
[eval] step 0 | ...
...
[ckpt] latest.pt @ step 128
[pause] staged gate reached at update 128; resume explicitly from checkpoints/latest.pt
[done] paused_staged | 128/4096 updates | 1048576 tokens | ...
```

说明：

- `--stop-at-update 128` 在完整 update 128 后保存 checkpoint 并暂停；不会自动继续到 256。
- 没有设置 `--max-wall-seconds`；如果 Kaggle 会话时间风险较高，可在确认 128 updates 能完成后自行加一个大于预计耗时的上限，但不要让它早于 128 触发。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 是 CUDA 确定性 cuBLAS 所需的环境变量；recipe 已启用 `deterministic=true`。
- 不要使用 `platform/scripts/run_kaggle.sh`：该脚本当前不传递 `--snapshot-config`；本 Cell 直接调用真实 CLI。
- 不要使用旧 `notebooks/kaggle_launcher.ipynb`：它指向旧仓库且没有 Snapshot 参数。

---

## 8. Monitoring Cell

可在训练期间反复执行。

```bash
%%bash
source /kaggle/working/muon_env.sh

LOG_FILE="$LOG_DIR/latest.log"

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== training process ==="
pgrep -af "python.*src/main.py" || true

echo "=== log tail ==="
tail -n 50 "$LOG_FILE" 2>/dev/null || true

echo "=== checkpoints ==="
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

---

## 9. Post-Stage Verification Cell

训练结束后执行。

```bash
%%bash
set -euo pipefail
source /kaggle/working/muon_env.sh

python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
expected_updates = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56,
                    64, 80, 96, 112, 128]

summary = json.loads((run_dir / "summary.json").read_text())
print("status:", summary["status"])
print("completed_updates:", summary["completed_updates"])
print("total_updates:", summary["total_updates"])
print("schedule_total_updates:", summary["schedule_total_updates"])
print("data_fingerprint:", summary["data_fingerprint"])
print("git_commit:", summary["git_commit"])
print("optimizer_name:", summary["optimizer_name"])
print("compute_precision:", summary["compute_precision"])

assert summary["status"] == "paused_staged"
assert summary["completed_updates"] == 128
assert summary["total_updates"] == 4096
assert summary["schedule_total_updates"] == 4096
assert summary["optimizer_name"] == "reference_muon"
assert summary["compute_precision"] == "fp32"
assert summary["git_commit"] == os.environ["TARGET_COMMIT"]

stored_config = json.loads((run_dir / "snapshot_config.json").read_text())
assert stored_config["config_hash"] == "9a073ba2e5ed8d9e00d857c2d0f64892633fcb2994c994ea4bc1d06b97c24b0e"
assert stored_config["config"]["updates"] == expected_updates + [
    256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
]

snapshots = run_dir / "snapshots"
pt_updates = []
for path in sorted(snapshots.glob("update_*.pt")):
    pt_updates.append(int(path.stem.split("_")[1]))
assert pt_updates == expected_updates, pt_updates
assert len(list(snapshots.glob("update_*.manifest.json"))) == 15

for update in expected_updates:
    manifest = json.loads((snapshots / f"update_{update:06d}.manifest.json").read_text())
    assert manifest["status"] == "committed"
    assert manifest["update"] == update
    assert manifest["integrity"]["expected_parameter_count"] == 30
    assert manifest["integrity"]["observed_parameter_count"] == 30
    assert manifest["integrity"]["expected_role_slots"] == 240
    assert manifest["integrity"]["observed_role_slots"] == 240
    assert manifest["integrity"]["unique_tensor_count"] == 210
    assert manifest["integrity"]["alias_count"] == 30
    assert manifest["integrity"]["tensors_file_sha256_ok"] is True
    assert manifest["identities"]["snapshot_config_hash"] == "9a073ba2e5ed8d9e00d857c2d0f64892633fcb2994c994ea4bc1d06b97c24b0e"
    assert manifest["training"]["train_nll_pre_update"] is not None

checkpoints = run_dir / "checkpoints"
for name in ("initial.pt", "update_000000.pt", "latest.pt", "update_000128.pt"):
    assert (checkpoints / name).is_file(), name
assert not (checkpoints / "final.pt").exists(), "paused_staged should not write final.pt"

errors = run_dir / "snapshot_errors.jsonl"
if errors.exists():
    print("snapshot_errors tail:")
    print("\n".join(errors.read_text().splitlines()[-20:]))
else:
    print("snapshot_errors: none")

print("post-stage validation: PASS")
print("snapshot_count:", len(pt_updates))
print("snapshot_dir_bytes:", sum(p.stat().st_size for p in snapshots.glob("update_*.pt")))
PY

echo "=== tree ==="
find "$RUN_DIR" -maxdepth 2 -type f | sort | sed -n '1,80p'
```

---

## 10. Expected Directory Structure

```text
/kaggle/working/
├── muon_research/                         # new GitHub repo clone
│   └── platform/                          # training platform
├── muon_data/
│   └── data/
│       └── slimpajama_4x -> /kaggle/input/<dataset>/.../slimpajama_4x
├── muon_snapshot_config.json              # full 24-node config, hash 9a073b...
├── logs/
│   ├── latest.log -> ...                  # full training log
│   └── dry_run_....log
└── runs/
    └── muon_30m_fp32_s0_stage1/
        ├── checkpoints/
        │   ├── initial.pt
        │   ├── update_000000.pt
        │   ├── latest.pt
        │   └── update_000128.pt
        ├── snapshots/
        │   ├── update_000001.pt
        │   ├── update_000001.manifest.json
        │   ├── update_000002.pt
        │   ├── update_000002.manifest.json
        │   ├── update_000004.pt
        │   ├── update_000004.manifest.json
        │   ├── ...
        │   ├── update_000112.pt
        │   ├── update_000112.manifest.json
        │   ├── update_000128.pt
        │   └── update_000128.manifest.json
        ├── snapshot_config.json
        ├── snapshot_status.json
        ├── snapshot_errors.jsonl            # absent or only warnings
        ├── metrics.jsonl
        ├── resolved_config.json
        ├── parameters.json
        ├── precision.json
        ├── data_manifest.json
        ├── source.json
        ├── environment.json
        └── summary.json
```

预期：

- `snapshots/` 中 **15 个 `.pt` + 15 个 Manifest**。
- `summary.json.status == "paused_staged"`，`completed_updates == 128`。
- `latest.pt` 和 `update_000128.pt` 存在；`final.pt` 不存在。
- `snapshot_config.json` 和每个 Manifest 的 `snapshot_config_hash` 都是 `9a073ba2e5ed8d9e00d857c2d0f64892633fcb2994c994ea4bc1d06b97c24b0e`。
- `source.json` 应记录新仓库的 commit；`git status` 在 Kaggle clone 中应为 clean。

---

## 11. Storage Budget Check

30M Muon-only snapshot 理论 payload：

```text
10,616,832 Muon parameters × 4 bytes × 7 unique roles = 297,271,296 bytes = 283.5 MiB
```

30 个矩阵 × 7 个 unique role = 210 个 tensor；`M_t_persisted` 与 `M_t_candidate` alias（state_simulation=none）。`torch.save` 容器开销另计，实际略高于 283.5 MiB。

保守估算：

| 项目 | 估算 |
|---|---:|
| 第一阶段 15 snapshots | 15 × (283.5 MiB + 2 MiB) ≈ 4.18 GiB |
| 4 个 checkpoint 文件 | 4 × 400 MiB ≈ 1.56 GiB |
| 2 GiB 预留 | 2 GiB |
| 第一阶段合计 | 约 7.75 GiB |
| 完整 24 snapshots | 24 × (283.5 MiB + 2 MiB) ≈ 6.69 GiB（< 8 GiB 上限） |

预检 Cell 会要求 `/kaggle/working` 的 free 空间大于 `first_stage_estimate + 2 GiB`。当前 Kaggle T4 常见 working 空间约 20 GB；若实际 free 不足约 8 GiB，预检会失败，必须停止并调整存储/清理策略。

注意：`max_total_bytes=8 GiB` 只统计 `run_dir/snapshots/`，不统计 Checkpoint。第一阶段 4 个 checkpoint 很小；若未来继续到 4096，checkpoint 文件会显著增长，需要单独的保留/清理/私有 Dataset 策略。

---

## 12. Resume / Later Stages (do not execute now)

第一阶段完成后，`latest.pt` 是 complete update 128 的 checkpoint，15 个 Snapshot 已提交。

后续阶段必须：

- 继续使用同一个 `run_dir`（或把 checkpoint/run artifacts 恢复回同一路径）；
- 继续使用同一个 `muon_snapshot_config.json`（24 节点，hash 不变）；
- 使用 `--resume "$RUN_DIR/checkpoints/latest.pt"`；
- 不要修改 recipe、config hash 或数据 fingerprint；
- 不要因为分段训练而把 24 节点改成 15 节点。

示意命令（后续阶段，不是本阶段执行）：

```bash
python src/main.py \
  --recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --to-device cuda:0 \
  --resume "$RUN_DIR/checkpoints/latest.pt" \
  --snapshot-config "$SNAPSHOT_CONFIG" \
  --stop-at-update 1024
```

Resume 时 Collector 会扫描已有 Manifest，跳过 1–128 的已提交节点，并在 256、384、512、768、1024 继续采集；不会重复写 1–128。

如果 Kaggle runtime 结束，需要先把 `run_dir`（至少 `checkpoints/latest.pt` 和 `snapshots/`）保存为私有 Kaggle Dataset，再在新 session 挂载回同一路径；不要让 Snapshot 成为恢复模型的输入，恢复只依赖 Checkpoint。

---

## 13. Known Risks / Do Not Change

1. **旧仓库/旧 notebook 不可用**：`notebooks/kaggle_launcher.ipynb` 和旧文档指向 `SiennaDai/benchmark-experiments`；本项目必须使用新仓库。
2. **`run_kaggle.sh` 不传递 `--snapshot-config`**：本阶段使用直接 `python src/main.py` 命令。
3. **T4 确定性**：recipe 启用 `deterministic=true`；训练 Cell 设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。如果出现 deterministic CUDA 错误，停止并报告，不要通过关闭确定性来“绕过”。
4. **`eval.attention_backend` 不生效**：当前实现使用 `precision.attention_backend`；这是已审计的配置问题，本任务不修改。
5. **辅助 AdamW weight decay**：1D RMSNorm/tied embedding 也承受 0.1 weight decay；这是已审计的配置问题，本任务不修改。
6. **数据 fingerprint 必须匹配**：不要修改 manifest 来绕过校验；如果 Kaggle Input 数据集 fingerprint 不是 `30152c9b...`，停止并解决数据版本问题。
7. **不要启动 4096 全量训练**：本阶段只运行到 128；正式启动 Cell 已设置 `--stop-at-update 128`。
8. **不要提交 `data/`、`analysis/`、`results/`、`research/`**：它们不是 Kaggle 训练代码；只提交 `platform/` 相关文件。`research/audit_snapshot_collector.md` 可选保留在本地。
