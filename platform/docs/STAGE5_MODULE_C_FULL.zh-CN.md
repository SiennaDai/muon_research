# Stage 5 — Module C 正式 Runbook（Muon 动力学与 Head 级损失贡献）

> 用 Stage 4 的 role 文件和 checkpoint，分析 `G → M → D → U → ΔW`，并在结构事件节点
> 784/1312 上把选定 Head 子空间的实际更新 `ΔW_S` 放回损失中：
>
> ```text
> ΔL_actual = L(W + ΔW_S) - L(W)
> ΔL_Taylor = g_SᵀΔW_S + ½ ΔW_Sᵀ H_SS ΔW_S
> ```
>
> 只做一阶 role 代数和 forward-only 损失评估；没有 HVP，没有 double backward。
>
> 旧 notebook / 旧输出不要复用。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，Accelerator 选 **GPU T4**（forward-only，一张足够）。
2. 打开 **Internet**（Cell 1 clone 仓库）。
3. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
4. 依次粘贴 Cell 1–5，`Save & Run All`。
5. Cell 2 是预检；Cell 3 在 `PREFLIGHT_OK` 缺失时会自动补跑预检。
6. Cell 4 每 5 分钟阻塞监控；Cell 5 只在 `COMPLETE` 存在时验收。

预计：

| 项目 | 预计 |
|---|---:|
| 预检 | 1–2 分钟 |
| 正式运行 | 5–10 分钟 |
| 显存峰值 | 约 2–3 GiB |
| 输出 | 几 MB（CSV） |

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 本文代码 commit | `53ab79925eb8de6147c1d57e24392cb4e691410e` |
| 分析脚本 | `analysis/structural_emergence/se_stage5_c_analyze.py` |
| B 指标（随仓库） | `analysis/structural_emergence/stage5_b_metrics/` |
| Checkpoint 数据集 | `siennadai/fp32-muon-ckpt-1536` |
| 数据 | `siennadai/lowp-benchmark-slimpajama-8x-v1` |
| 数据 fingerprint | `ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489` |
| Role 动力学节点 | 全部 20 个新增 role 节点/seed |
| 损失评估节点 | 784 与 1312（seed0/1 各两个） |
| 层 | seed0 `0,2,3`；seed1 `0,3,4` |
| Probe | validation windows 0–7（2048 token） |
| 子空间 | all Q/K/V/O、all QKV、all attn、top Q/K/V/O（按 B κ） |

## 2. C 计算什么

### 2.1 Role 动力学（CPU）

对 20 个 role 节点、每层、每类型、每 Head：

- `||G||, ||M||, ||D||, ||U||, ||ΔW||`；
- `g_share, u_share, dw_share`；
- `cos(G,U), cos(U,ΔW), cos(G,ΔW)`；
- `||U||/||G||`、`||ΔW||/||U||`。

输出 `role_dynamics.csv`。

### 2.2 Head 子空间损失（forward-only）

在事件节点 784/1312 上，对每一层定义 10 个子空间：

```text
all_Q, all_K, all_V, all_O, all_QKV, all_attn,
top_Q, top_K, top_V, top_O
```

`top_*` 使用 B 的 probe κ 最大 Head。

- `g_S`：**L_probe 在当前节点的梯度**在子空间中的 block（一次普通 backward，不是 Role G）；
  一阶项 `g_SᵀΔW_S`；
- `H_SS`：B 的 `q_h/q_hk` 子矩阵；二阶项 `½Σ q_hk`；
- 实际项：把 role 的 `W_after-W_before` 只加到该 Head 子空间，前向计算
  `L(W+ΔW_S)-L(W)`；
- 输出 `first_order / second_order / delta_L_taylor / delta_L_actual / residual`。

`L_probe` 使用 math SDPA，和 B 的 Hessian 计算保持一致。

> 这是**选定 Head 子空间**的损失变化，不是完整 optimizer step：
> 辅助 AdamW 参数和其他 Muon block 保持不动。
>
> 重要：`first_order` 使用 **L_probe 梯度**，与 B 的 probe `q` 和 `actual L_probe` 同一口径；
> 不能用 Role G（L_train 梯度）代替，否则 Taylor 与 actual 不可比。

## 3. Notebook Cells

### Cell 1 — 代码、数据、Checkpoint 路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="53ab79925eb8de6147c1d57e24392cb4e691410e"

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
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u000784.pt"
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u000784_roles.pt"
test -f "$DATA_ROOT/manifest.json"
test -f "$REPO/analysis/structural_emergence/stage5_b_metrics/direction_curvature.csv"
python - <<'PY'
import json, os
from pathlib import Path
m = json.loads((Path(os.environ['DATA_ROOT']) / 'manifest.json').read_text())
assert m['fingerprint'] == 'ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489'
print('data fingerprint OK')
PY
```

### Cell 2 — 预检（seed0 / event 784）

预期：

- `role_dynamics.csv` **2880 行**（seed0 的 20 个 role 节点 ×6 层 ×4 类型 ×6 Heads）；
- `head_loss.csv` **30 行**（3 层 ×10 子空间）；
- `COMPLETE` 存在；
- role 份额每组合和为 1；
- `delta_L_actual`、`first_order`、`second_order`、`delta_L_taylor` 有限。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_c_preflight
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
python -u analysis/structural_emergence/se_stage5_c_analyze.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --events 784

python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_c_preflight')
assert (out / 'COMPLETE').is_file(), 'missing COMPLETE'
def rows(name):
    p = out / name
    assert p.is_file(), p
    return list(csv.DictReader(p.open()))
role = rows('role_dynamics.csv')
loss = rows('head_loss.csv')
assert len(role) == 2880, len(role)
assert len(loss) == 30, len(loss)
groups = {}
for r in role:
    key = (r['seed'], r['node'], r['layer'], r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['g_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
for r in loss:
    for key in ('delta_L_actual', 'first_order', 'second_order', 'delta_L_taylor'):
        assert math.isfinite(float(r[key])), (r, key)
assert all(math.isfinite(float(r['dw_fro'])) for r in role)
print('C preflight OK:', len(role), len(loss))
PY

touch "$PRE/PREFLIGHT_OK"
echo "PREFLIGHT_OK written"
```

### Cell 3 — 正式运行（后台；哨兵缺失时自动预检）

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_c
PRE=/kaggle/working/stage5_c_preflight

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
  echo "PREFLIGHT_OK missing; running C preflight now"
  rm -rf "$PRE"
  python -u analysis/structural_emergence/se_stage5_c_analyze.py \
    --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
    --out "$PRE" --seeds seed0 --events 784
  python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file(), 'preflight missing COMPLETE'
role = list(csv.DictReader((out/'role_dynamics.csv').open()))
loss = list(csv.DictReader((out/'head_loss.csv').open()))
assert len(role) == 2880, len(role)
assert len(loss) == 30, len(loss)
groups = {}
for r in role:
    key = (r['seed'], r['node'], r['layer'], r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['g_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
for r in loss:
    for key in ('delta_L_actual', 'first_order', 'second_order', 'delta_L_taylor'):
        assert math.isfinite(float(r[key])), (r, key)
print('C preflight OK')
PY
  touch "$PRE/PREFLIGHT_OK"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_c_analyze.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --seeds seed0,seed1 --events 784,1312 \
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
OUT=/kaggle/working/stage5_c
PID=$(cat "$OUT/run.pid")
echo "monitor started for pid=$PID"
while kill -0 "$PID" 2>/dev/null; do
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
  sleep 300
done
echo "[$(date -u +%H:%M:%S)] process exited"
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "Module C run did not complete"; exit 1; }
echo "COMPLETE marker present"
```

### Cell 5 — 验收

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_c')
assert (out / 'COMPLETE').is_file(), 'missing COMPLETE'
def rows(name):
    p = out / name
    assert p.is_file(), p
    return list(csv.DictReader(p.open()))
role = rows('role_dynamics.csv')
loss = rows('head_loss.csv')
assert len(role) == 5760, len(role)
assert len(loss) == 120, len(loss)
groups = {}
for r in role:
    key = (r['seed'], r['node'], r['layer'], r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['g_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
for r in loss:
    for key in ('delta_L_actual', 'first_order', 'second_order', 'delta_L_taylor'):
        assert math.isfinite(float(r[key])), (r, key)
summary = json.loads((out/'run_summary.json').read_text())
print(json.dumps({
    'role_rows': len(role), 'head_loss_rows': len(loss),
    'events': summary['events'], 'seeds': summary['seeds'],
    'n_subspaces': sorted({r['subspace'] for r in loss}),
}, indent=2))
PY
du -sh /kaggle/working/stage5_c
```

## 4. 产物

```text
stage5_c/
  COMPLETE
  role_dynamics.csv     5760 行
  head_loss.csv         120 行
  run_summary.json
  run.log / run.pid
```

## 5. 解释口径

- `g_share/u_share/dw_share`：`G/U/ΔW` 在 6 个 Head 中的能量份额；
- `cos(G,U)`、`cos(U,ΔW)`：Muon 的 Nesterov 方向和 NS 输出、最终更新之间的对齐；
- `first_order`：实际梯度沿 Head 子空间更新的一阶项；
- `second_order`：B 的 `q_h/q_hk` 给出的二阶项；
- `delta_L_taylor`：两者之和；
- `delta_L_actual`：只施加该 Head 子空间的 `ΔW_S` 后的真实 loss 变化；
- `residual`：`actual - Taylor`，用于判断局部二阶近似是否足够。

C 的正式范围是**事件节点 784/1312** 的 Head 子空间损失变化；不把这个结果外推为
完整 optimizer step 的损失变化（辅助 AdamW 和其他 Muon block 没动）。

## 6. 失败时

- Cell 2 失败：不要运行 Cell 3；贴 `stage5_c_preflight/run.log` 最后 30 行。
- Cell 4 报 `Module C run did not complete`：检查 `stage5_c/run.log`。
- 行数不符：先看 `run_summary.json` 和 `run.log`。
