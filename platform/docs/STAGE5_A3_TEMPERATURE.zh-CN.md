# Stage 5 — A-3 Attention Temperature Counterfactual Runbook

> 目标：固定上游 `dA`，只改变 attention temperature `T`，检查 attention 聚焦程度如何改变 Q/K 的反向梯度份额。
>
> 这是局部反向公式干预，不重新训练，不做 HVP。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，GPU T4，Internet；
2. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
3. 粘贴 Cell 1–5，`Save & Run All`。

预计 5–10 分钟；显存约 2–3 GiB。

## 1. 方法

对每个节点、每层、每 Head：

```text
A_h(T) = softmax(Q_h K_hᵀ / (√d · T) + causal)
dA_h   = dO W_proj[:, head columns]        # 与 T 无关，固定
dV_h   = A_h(T)ᵀ dA_h
dP_h   = dA_h V_hᵀ
dS_h   = A_h(T) · (dP_h − Σ dP_h A_h(T))
dQ_h   = (1/T) dS_h K_h / √d
dK_h   = (1/T) dS_hᵀ Q_h / √d
```

温度：

```text
T = 0.5, 0.75, 1.0, 1.5, 2.0
```

`T<1` 使 attention 更聚焦；`T>1` 使 attention 更平坦。  
固定 `dA` 是为了控制“上游反向信号”这一项，只看 attention/score 几何的影响。

## 2. 代码与输出

代码 commit：`f71326f8bfabc2e1c2f5aaba5ee7e51507e52c1c`  
脚本：`analysis/structural_emergence/se_stage5_a3_temperature.py`

输出：

```text
temperature_metrics.csv
  seed,node,layer,type,head,temperature,
  entropy,a_max,dq_fro,dk_fro,dv_fro,
  share_q,share_k,share_v
```

预检：1 个节点 ×6 层 ×3 类型 ×6 Head ×5 温度 = **540 行**；  
正式：62 个 `(seed,node)` → **33,480 行**。

## 3. Notebook Cells

### Cell 1 — 代码与路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="f71326f8bfabc2e1c2f5aaba5ee7e51507e52c1c"

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
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u000784.pt"
```

### Cell 2 — 预检（seed0 / node 784）

预期：540 行；`share_q/share_k/share_v` 每组和为 1。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_a3_preflight
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
python -u analysis/structural_emergence/se_stage5_a3_temperature.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --nodes 784

python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'temperature_metrics.csv').open()))
assert len(rows) == 540, len(rows)
assert all(math.isfinite(float(r['share_q'])) for r in rows)
for key in (('seed','node','layer','head','temperature'),):
    pass
print('A-3 preflight OK:', len(rows))
PY
touch "$PRE/PREFLIGHT_OK"
```

### Cell 3 — 正式运行（后台；哨兵缺失时自动预检）

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_a3
PRE=/kaggle/working/stage5_a3_preflight

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
  echo "PREFLIGHT_OK missing; running A-3 preflight"
  rm -rf "$PRE"
  python -u analysis/structural_emergence/se_stage5_a3_temperature.py \
    --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
    --out "$PRE" --seeds seed0 --nodes 784
  python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'temperature_metrics.csv').open()))
assert len(rows) == 540, len(rows)
assert all(math.isfinite(float(r['share_q'])) for r in rows)
print('A-3 preflight OK')
PY
  touch "$PRE/PREFLIGHT_OK"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_a3_temperature.py \
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
OUT=/kaggle/working/stage5_a3
PID=$(cat "$OUT/run.pid")
while kill -0 "$PID" 2>/dev/null; do
  N=$(grep -c '^\[stage5a3\]' "$OUT/run.log" 2>/dev/null || true)
  echo "[$(date -u +%H:%M:%S)] nodes=$N"
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
  sleep 300
done
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "A-3 did not complete"; exit 1; }
echo "COMPLETE"
```

### Cell 5 — 验收

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_a3')
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'temperature_metrics.csv').open()))
assert len(rows) == 33480, len(rows)
assert all(math.isfinite(float(r['share_q'])) for r in rows)
print(json.dumps({'rows': len(rows)}, indent=2))
PY
du -sh /kaggle/working/stage5_a3
```

## 4. 解释

- 对每个温度，比较：
  - attention entropy；
  - Q/K `share_q/share_k`；
  - Head 排序；
  - 相对 `T=1` 的份额变化。
- `T<1` 使 attention 更聚焦；如果 Q/K 的 Head 集中随 `T` 单调增强，说明 softmax/score 几何是直接放大器。
- `T>1` 使 attention 更平坦；如果 Q/K 份额明显下降，说明 Head 集中依赖 attention 聚焦。
- `T=1` 是真实模型；其他温度是局部反向公式反事实，不是重新训练。

## 5. 失败时

- Cell 2 失败：不要跑 Cell 3；贴 `stage5_a3_preflight/run.log`。
- 行数不符：检查每节点 540 行（6 层 ×3 类型 ×6 Head ×5 温度）。
- Cell 4 没有 `COMPLETE`：检查 `stage5_a3/run.log`。
