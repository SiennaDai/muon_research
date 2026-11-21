# Stage 5 — A-2 Attention 反向信号来源 Runbook

> 目标：解释为什么 `ΔY_h` 在不同 Head 之间不均匀。对每个节点、每层、每 Head 采集
> attention 概率、Head 输出梯度，并把 `ΔY` 分解为 attention/score/Q/K/V 分量。
>
> 这是 forward/backward-only 采集；没有 HVP，不训练。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，Accelerator 选 **GPU T4**。
2. 打开 Internet。
3. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
4. 粘贴 Cell 1–5，`Save & Run All`。

预计正式运行 10–20 分钟；显存约 3–4 GiB。

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 代码 commit | `916a91732534dff78d357a4f2fc2308fb007c004` |
| 脚本 | `analysis/structural_emergence/se_stage5_a2_attention.py` |
| 节点 | 全部 31 个 Stage 4 节点 |
| Seeds | seed0 / seed1 |
| Loss | L_train（8×4 micro-batch）+ L_probe（windows 0–7） |
| 行数 | 预检 432；正式 13,392 |

## 2. 采集与分解

对每层、每 Head：

```text
A_h   = softmax(Q_h K_hᵀ/√d + causal)
dA_h  = dO W_proj[:, head columns]
dV_h  = A_hᵀ dA_h
dP_h  = dA_h V_hᵀ
dS_h  = A_h * (dP_h - Σ dP_h A_h)
dQ_h  = dS_h K_h / √d
dK_h  = dS_hᵀ Q_h / √d
```

计算出的 `dQ/dK/dV` 会与 c_attn 输出梯度的 Q/K/V 分块对照，记录：

- `rel`：相对误差；
- `cos`：方向 cosine；
- `dy_fro`、`d_fro`；
- `da/dp/ds` 范数；
- attention entropy/max；
- Q/K/V 范数；
- `dy_share`、`da_share`。

字段：

```text
seed,node,loss,layer,type,head,
dy_fro,d_fro,rel,cos,
da_fro,dp_fro,ds_fro,a_entropy,a_max,
q_fro,k_fro,v_fro,dy_share,da_share
```

每个 Head 会输出 Q/K/V 三行。

## 3. Notebook Cells

### Cell 1 — 代码与路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="916a91732534dff78d357a4f2fc2308fb007c004"

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
assert hits
print(hits[0].parents[2])
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
echo "CKPT_ROOT=$CKPT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u000784_roles.pt"
```

### Cell 2 — 预检（seed0 / node 0 + 784）

预期：

- `attention_dynamics.csv` **432 行**（2 nodes ×6 layers ×18 Head-type ×2 losses）；
- `COMPLETE` 存在；
- `rel` 最大 <1e-3；
- `cos` 有限；
- 每个 `(seed,node,loss,layer,type)` 的 `dy_share` 和为 1。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_a2_preflight
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
python -u analysis/structural_emergence/se_stage5_a2_attention.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --nodes 0,784

python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'attention_dynamics.csv').open()))
assert len(rows) == 432, len(rows)
assert max(float(r['rel']) for r in rows) < 1e-3
assert all(math.isfinite(float(r['cos'])) for r in rows)
groups = {}
for r in rows:
    key = (r['seed'],r['node'],r['loss'],r['layer'],r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['dy_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
print('A-2 preflight OK:', len(rows))
PY
touch "$PRE/PREFLIGHT_OK"
```

### Cell 3 — 正式运行（后台；哨兵缺失时自动预检）

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_a2
PRE=/kaggle/working/stage5_a2_preflight

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
  echo "PREFLIGHT_OK missing; running A-2 preflight"
  rm -rf "$PRE"
  python -u analysis/structural_emergence/se_stage5_a2_attention.py \
    --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
    --out "$PRE" --seeds seed0 --nodes 0,784
  python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'attention_dynamics.csv').open()))
assert len(rows) == 432, len(rows)
assert max(float(r['rel']) for r in rows) < 1e-3
print('A-2 preflight OK')
PY
  touch "$PRE/PREFLIGHT_OK"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_a2_attention.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --seeds seed0,seed1 \
  --nodes 0,8,16,32,48,56,64,80,96,112,128,192,256,320,384,448,512,576,640,704,768,784,832,896,960,1024,1152,1280,1312,1408,1536 \
  > "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "pid=$(cat "$OUT/run.pid")"
sleep 20
tail -n 5 "$OUT/run.log"
```

### Cell 4 — 阻塞监控

```bash
%%bash
set -uo pipefail
OUT=/kaggle/working/stage5_a2
PID=$(cat "$OUT/run.pid")
while kill -0 "$PID" 2>/dev/null; do
  N=$(grep -c '^\[stage5a2\]' "$OUT/run.log" 2>/dev/null || true)
  echo "[$(date -u +%H:%M:%S)] nodes=$N"
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
  sleep 300
done
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "A-2 did not complete"; exit 1; }
echo "COMPLETE"
```

### Cell 5 — 验收

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_a2')
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'attention_dynamics.csv').open()))
assert len(rows) == 13392, len(rows)
assert max(float(r['rel']) for r in rows) < 1e-3
assert all(math.isfinite(float(r['cos'])) for r in rows)
groups = {}
for r in rows:
    key = (r['seed'],r['node'],r['loss'],r['layer'],r['type'])
    groups[key] = groups.get(key, 0.0) + float(r['dy_share'])
for key, value in groups.items():
    assert abs(value - 1.0) < 1e-4, (key, value)
print(json.dumps({'rows': len(rows), 'nodes': len({r['node'] for r in rows})}, indent=2))
PY
du -sh /kaggle/working/stage5_a2
```

## 4. 产物

```text
stage5_a2/
  COMPLETE
  attention_dynamics.csv  13392 行
  run_summary.json
  run.log / run.pid
```

## 5. 解释口径

- `rel/cos`：手工 attention backward 与 c_attn 输出梯度的分解是否一致；不通过就不要用分解。
- `dy_share`：该 Head 的 Q/K/V 梯度在类型内的份额。
- `da_share`：Head 输出误差 `dA_h` 的份额。
- `a_entropy/a_max`：attention 集中度。
- 若 `dy_share` 与 `da_share` 高度一致，说明上游输出误差已经决定梯度分配；
- 若 `dy_share` 与 attention entropy/`dS` 更相关，说明 attention/score 传播是来源；
- 这能回答“为什么 ΔY 先在 Head 间不均匀”，而不需要 Hessian。

## 6. 失败时

- Cell 2 失败：不要跑 Cell 3；贴 `stage5_a2_preflight/run.log`。
- `rel` 大于 1e-3：先检查 rotary/causal mask/softmax backward，不要继续分析。
- Cell 4 没有 `COMPLETE`：检查 `stage5_a2/run.log`。
