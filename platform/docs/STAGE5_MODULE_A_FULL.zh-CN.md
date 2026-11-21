# Stage 5 — Module A 正式 Runbook（Gradient & Representation Dynamics）

> 在全新 Kaggle Notebook 中，用 Stage 4 的 31 个 pre-update checkpoint 采集 Module A：
> 每层的激活 `X/A`、反向信号 `dY/dO`、每 Head 梯度 `G_h`，以及梯度份额、谱和
> `X/ΔY/G` 方向对齐。
>
> 只做一阶 forward/backward；不计算 Hessian，不需要 math SDPA，不修改任何 checkpoint。
>
> 旧 notebook / 旧输出不要复用。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，Accelerator 选 **GPU T4**（一张即可）。
2. 打开 **Internet**（Cell 1 clone 仓库）。
3. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
4. 依次粘贴 Cell 1–5，`Save & Run All`。
5. **Cell 2 必须通过**；Cell 3 会检查 `PREFLIGHT_OK`，缺失时会自动补跑预检。
6. Cell 4 每 5 分钟阻塞监控；Cell 5 只在 `COMPLETE` 存在时验收。

预计：

| 项目 | 预计 |
|---|---:|
| 预检 | 2–4 分钟 |
| 正式采集 | 15–30 分钟（T4） |
| 显存峰值 | 约 2–4 GiB |
| 输出 | 约 200 MB（CSV + 谱） |

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 本文代码 commit | `c237f28257382e95b2fd88c0bbc29be8527155db` |
| 采集脚本 | `analysis/structural_emergence/se_stage5_a_collect.py` |
| Checkpoint 数据集 | `siennadai/fp32-muon-ckpt-1536` |
| 数据 | `siennadai/lowp-benchmark-slimpajama-8x-v1` |
| 数据 fingerprint | `ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489` |
| 节点 | 全部 31 个 Stage 4 pre-update 节点（见下） |
| Seeds | seed0 / seed1 |
| L_train | 节点 update 的 32 个训练窗口，8×4 micro-batch |
| L_probe | validation windows 0–7（2048 token） |
| Head 覆盖 | Q/K/V/O 全部 6 层 ×6 Heads |
| 预检节点 | seed0: `0,784` |

31 个节点：

```text
0,8,16,32,48,56,64,80,96,112,128,192,256,320,384,448,512,
576,640,704,768,784,832,896,960,1024,1152,1280,1312,1408,1536
```

节点 `u` 是 update u 开始前的状态；节点 0 使用 update 1 的 sidecar 窗口。

## 2. A 采集什么

### 2.1 前向

每层 `l`：

- `X_l`：`attn.c_attn` 的输入（attention 前 RMSNorm 输出），`[B,T,384]`；
- `A_l`：`attn.c_proj` 的输入（attention 输出），`[B,T,384]`。

不保存原始 `Q/K/V`；attention 矩阵也不保存。

### 2.2 反向

- `dY_l`：loss 对 `attn.c_attn` 输出的梯度，`[B,T,1152]`；
  - 按 Q/K/V 和 Head 切成 `dY^Q_h,dY^K_h,dY^V_h`，每个 `[B,T,64]`。
- `dO_l`：loss 对 `attn.c_proj` 输出的梯度，`[B,T,384]`；
  - 按输出列切成 6 个 Head，每个 `[B,T,64]`。

### 2.3 每 Head 梯度

```text
Q/K/V:  G_h = (dY_h)^T X_l              # [64,384]
O:      G_h = (dO)^T A_l[:, head]        # [384,64]  (c_proj 列块)
```

- L_train：8 个 micro-batch 的 `G_h` 相加，得到该 update 的实际梯度；
- L_probe：固定 validation batch 的 `G_h`；
- 保存每 micro-batch 的 `g_fro`、`cos_to_total`、`g_share_mb`。

### 2.4 保存的指标

`head_dynamics.csv` 每行对应
`(seed,node,loss,layer,type,head)`：

```text
g_fro, g_share, g_top1_share, g_eff_rank, g_stable_rank,
dy_fro, dy_top1, dy_eff_rank,
input_fro, input_top1, input_eff_rank,
cos_g_dy, cos_g_x, angle_g_dy4, angle_g_x4,
clip_coef, total_grad_norm
```

其中 `input` 对 Q/K/V 是 `X_l`，对 O 是 `A_l` 的对应 64 维 Head 列块。

`microbatch.csv` 保存 8 个 micro-batch 的：
`g_fro, cos_to_total, g_share_mb`。

`spectra/{seed}_u{node}.npz` 保存：

- `X/A` top-16 奇异值和右奇异向量；
- `dY/dO` top-8 奇异值和左奇异向量；
- `G_h` top-8 奇异值、左/右奇异向量。

**不保存**原始 `X`（约 75 MB/节点）、`dY`（约 226 MB/节点）或完整 `G_h`。

## 3. Notebook Cells

### Cell 1 — 代码、数据、Checkpoint 路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="c237f28257382e95b2fd88c0bbc29be8527155db"

if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO"
  git clone https://github.com/SiennaDai/muon_research "$REPO"
fi
git -C "$REPO" fetch --all --tags
git -C "$REPO" checkout --detach "$CODE_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"

export CKPT_ROOT="$(python - <<'PY'
from pathlib import Path
hits = [p for p in Path('/kaggle/input').rglob('pre_u000784.pt')
        if 'fp32-muon-ckpt-1536' in str(p)
        and p.parent.name == 'node_checkpoints'
        and p.parents[1].name == 'seed0']
assert hits, 'no seed0 pre_u000784.pt under fp32-muon-ckpt-1536'
print(hits[0].parents[2])
PY
)"

export DATA_ROOT="$(python - <<'PY'
import json
from pathlib import Path
hits = list(Path('/kaggle/input').rglob('slimpajama_8x/manifest.json'))
chosen = None
for p in hits:
    try:
        m = json.loads(p.read_text())
    except Exception:
        continue
    if m.get('fingerprint') == 'ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489':
        chosen = p
        break
assert chosen is not None, [str(p) for p in hits]
print(chosen.parent)
PY
)"

echo "CKPT_ROOT=$CKPT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u000000.pt"
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u000784.pt"
test -f "$DATA_ROOT/manifest.json"
python - <<'PY'
import json, os
from pathlib import Path
m = json.loads((Path(os.environ['DATA_ROOT']) / 'manifest.json').read_text())
assert m['fingerprint'] == 'ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489'
print('data fingerprint OK')
PY
```

### Cell 2 — 预检（node 0 + 784）

检查初始化节点、Role 节点、CSV 行数、G 重放和谱文件。

预期：

- `head_dynamics.csv` **576 行**（2 nodes × 288）；
- `microbatch.csv` **2304 行**（2 × 1152）；
- `replay_validation.csv` **12 行**（node 784 × 6 层 × 2 参数）；
- `spectra/*.npz` **2 个**；
- `COMPLETE` 存在；
- G 相对误差 <1e-4。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_a_preflight
rm -rf "$PRE"

export CKPT_ROOT="$(python - <<'PY'
from pathlib import Path
p = next(p for p in Path('/kaggle/input').rglob('pre_u000784.pt')
         if 'fp32-muon-ckpt-1536' in str(p)
         and p.parent.name == 'node_checkpoints'
         and p.parents[1].name == 'seed0')
print(p.parents[2])
PY
)"
export DATA_ROOT="$(python - <<'PY'
import json
from pathlib import Path
hits = list(Path('/kaggle/input').rglob('slimpajama_8x/manifest.json'))
for p in hits:
    try:
        m = json.loads(p.read_text())
    except Exception:
        continue
    if m.get('fingerprint') == 'ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489':
        print(p.parent)
        break
else:
    raise SystemExit('data manifest not found')
PY
)"

cd "$REPO"
python -u analysis/structural_emergence/se_stage5_a_collect.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --nodes 0,784

python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_a_preflight')
assert (out / 'COMPLETE').is_file(), 'missing COMPLETE'
def rows(name):
    p = out / name
    assert p.is_file(), p
    return list(csv.DictReader(p.open()))
head = rows('head_dynamics.csv')
mb = rows('microbatch.csv')
replay = rows('replay_validation.csv')
spectra = sorted((out / 'spectra').glob('*.npz'))
assert len(head) == 576, len(head)
assert len(mb) == 2304, len(mb)
assert len(replay) == 12, len(replay)
assert len(spectra) == 2, len(spectra)
for r in replay:
    assert float(r['g_rel_error']) < 1e-4, r
# every head group's share must sum to 1
groups = {}
for r in head:
    key = (r['seed'], r['node'], r['loss'], r['layer'], r['type'])
    groups.setdefault(key, 0.0)
    groups[key] += float(r['g_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
assert all(math.isfinite(float(r['g_fro'])) for r in head)
print('A preflight OK:', len(head), len(mb), len(replay), len(spectra))
PY

touch "$PRE/PREFLIGHT_OK"
echo "PREFLIGHT_OK written"
```

### Cell 3 — 正式采集（后台；哨兵缺失时自动预检）

> Kaggle commit 从新的 `/kaggle/working` 开始；交互式 Cell 2 的哨兵不会继承。
> 这里在哨兵缺失时自动跑一遍 node 0 + 784 预检。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_a
PRE=/kaggle/working/stage5_a_preflight

export CKPT_ROOT="$(python - <<'PY'
from pathlib import Path
p = next(p for p in Path('/kaggle/input').rglob('pre_u000784.pt')
         if 'fp32-muon-ckpt-1536' in str(p)
         and p.parent.name == 'node_checkpoints'
         and p.parents[1].name == 'seed0')
print(p.parents[2])
PY
)"
export DATA_ROOT="$(python - <<'PY'
import json
from pathlib import Path
hits = list(Path('/kaggle/input').rglob('slimpajama_8x/manifest.json'))
for p in hits:
    try:
        m = json.loads(p.read_text())
    except Exception:
        continue
    if m.get('fingerprint') == 'ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489':
        print(p.parent)
        break
else:
    raise SystemExit('data manifest not found')
PY
)"

cd "$REPO"
if [ ! -f "$PRE/PREFLIGHT_OK" ]; then
  echo "PREFLIGHT_OK missing; running A preflight now"
  rm -rf "$PRE"
  python -u analysis/structural_emergence/se_stage5_a_collect.py \
    --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
    --out "$PRE" --seeds seed0 --nodes 0,784
  python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file(), 'preflight missing COMPLETE'
def rows(name):
    return list(csv.DictReader((out / name).open()))
head = rows('head_dynamics.csv')
mb = rows('microbatch.csv')
replay = rows('replay_validation.csv')
assert len(head) == 576, len(head)
assert len(mb) == 2304, len(mb)
assert len(replay) == 12, len(replay)
assert len(list((out / 'spectra').glob('*.npz'))) == 2
for r in replay:
    assert float(r['g_rel_error']) < 1e-4, r
groups = {}
for r in head:
    key = (r['seed'], r['node'], r['loss'], r['layer'], r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['g_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
print('A preflight OK')
PY
  touch "$PRE/PREFLIGHT_OK"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_a_collect.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --seeds seed0,seed1 \
  --nodes 0,8,16,32,48,56,64,80,96,112,128,192,256,320,384,448,512,576,640,704,768,784,832,896,960,1024,1152,1280,1312,1408,1536 \
  > "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "pid=$(cat "$OUT/run.pid")"
sleep 20
tail -n 5 "$OUT/run.log"
```

### Cell 4 — 阻塞监控（每 5 分钟）

```bash
%%bash
set -uo pipefail
OUT=/kaggle/working/stage5_a
PID=$(cat "$OUT/run.pid")
echo "monitor started for pid=$PID"
while kill -0 "$PID" 2>/dev/null; do
  N=$(grep -c '^\[stage5a\]' "$OUT/run.log" 2>/dev/null || true)
  echo "[$(date -u +%H:%M:%S)] completed_nodes=$N"
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
  sleep 300
done
echo "[$(date -u +%H:%M:%S)] process exited"
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "Module A run did not complete"; exit 1; }
echo "COMPLETE marker present"
```

### Cell 5 — 验收

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_a')
assert (out / 'COMPLETE').is_file(), 'missing COMPLETE'
def rows(name):
    p = out / name
    assert p.is_file(), p
    return list(csv.DictReader(p.open()))
head = rows('head_dynamics.csv')
mb = rows('microbatch.csv')
replay = rows('replay_validation.csv')
spectra = sorted((out / 'spectra').glob('*.npz'))
assert len(head) == 17856, len(head)
assert len(mb) == 71424, len(mb)
assert len(replay) == 480, len(replay)
assert len(spectra) == 62, len(spectra)
for r in replay:
    assert float(r['g_rel_error']) < 1e-4, r
groups = {}
for r in head:
    key = (r['seed'], r['node'], r['loss'], r['layer'], r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['g_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
assert all(math.isfinite(float(r['g_fro'])) for r in head)
assert all(math.isfinite(float(r['cos_to_total'])) for r in mb)
summary = json.loads((out / 'run_summary.json').read_text())
print(json.dumps({
    'head_rows': len(head), 'microbatch_rows': len(mb),
    'replay_rows': len(replay), 'spectra_files': len(spectra),
    'max_g_rel_error': max(float(r['g_rel_error']) for r in replay),
    'nodes': sorted({int(r['node']) for r in head}),
    'runs': summary['runs'],
}, indent=2))
PY
du -sh /kaggle/working/stage5_a
```

## 4. 产物

```text
stage5_a/
  COMPLETE
  head_dynamics.csv     17856 行
  microbatch.csv        71424 行
  replay_validation.csv 480 行
  run_summary.json
  spectra/
    seed0_u000000.npz ... seed1_u001536.npz  (62 个)
  run.log / run.pid
```

## 5. 行数与语义

- 每个节点每个 loss：6 层 ×4 类型 ×6 Heads =144 行；train+probe=288；62 个 `(seed,node)` →17856。
- `microbatch.csv`：144 Heads ×8 micro-batch =1152/节点；62 →71424。
- `replay_validation.csv`：20 个 Role 节点 ×2 seed ×6 层 ×2 参数 =480。
- `spectra`：31 节点 ×2 seed =62。

## 6. A 能回答什么

- `g_share` 随节点的变化：Head 梯度优势何时出现；
- `X` 与 `ΔY` 的谱和 `cos_g_x/cos_g_dy`：优势来自输入表示还是反向信号；
- `g_eff_rank`、`g_stable_rank`：梯度是否是少数方向主导；
- `microbatch.csv`：早期优势是否跨 micro-batch 稳定；
- 与 B 的 `κ` / `ρ` 对齐：梯度优势是否对应局部曲率；
- seed0/seed1 在节点 0–32 的初始化差异。

## 7. 失败时

- Cell 2 失败：不要跑 Cell 3；贴 `stage5_a_preflight/run.log` 最后 30 行。
- Cell 4 报 `Module A run did not complete`：检查 `stage5_a/run.log`。
- OOM：记录 `[stage5a]` 最后一行、节点和 `nvidia-smi`；不要自行改 batch。
- 行数不符：先看 `run_summary.json` 和 `run.log`，不要直接进入分析。
