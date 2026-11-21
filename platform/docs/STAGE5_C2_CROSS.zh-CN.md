# Stage 5 — C-2 QK–VO 交叉 Hessian 与实际交互 Runbook

> 目标：在事件节点 784/1312、B 覆盖的 L0/L3 上，比较：
>
> ```text
> QK = all Q/K heads
> VO = all V/O heads
> joint = QK + VO = all attention
> ```
>
> 计算交叉 Hessian 项和实际损失交互：
>
> ```text
> I_H      = ΔW_QKᵀ H ΔW_VO = Σ_{h∈QK,k∈VO} q_hk
> I_actual = ΔL_joint - ΔL_QK - ΔL_VO
> I_taylor = Taylor_joint - Taylor_QK - Taylor_VO
> ```
>
> 这是 forward-only 的小分析；不计算 HVP，不训练。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，Accelerator 选 **GPU T4**。
2. 打开 Internet。
3. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
4. 粘贴 Cell 1–5，`Save & Run All`。

预计运行 < 2 分钟；显存约 2–3 GiB。

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 代码 commit | `fac99cde9dbae71684377b6f8702d7fe6d1cdd6b` |
| 脚本 | `analysis/structural_emergence/se_stage5_c2_cross.py` |
| B 指标 | `analysis/structural_emergence/stage5_b_metrics/` |
| 节点 | 784, 1312 |
| 层 | L0, L3（B 有 V/O Hessian） |
| Probe | validation windows 0–7 |
| 预期行数 | 预检 1；正式 8 |

## 2. 输出字段

```text
seed,node,layer,L_base,
q_QK,q_VO,I_H,rho_H,
first_QK,first_VO,first_joint,
second_QK,second_VO,second_joint,
taylor_QK,taylor_VO,taylor_joint,I_taylor,
actual_QK,actual_VO,actual_joint,I_actual,I_residual,
I_actual_norm
```

- `q_QK`：QK 内部所有 `q_hk` 之和；
- `q_VO`：VO 内部所有 `q_hk` 之和；
- `I_H`：QK×VO 交叉项；
- `rho_H = I_H / sqrt(|q_QK q_VO|)`；
- `I_taylor`：Taylor 的交互项，应与 `I_H` 一致；
- `I_actual`：实际损失交互；
- `I_actual_norm = I_actual / (|ΔL_QK|+|ΔL_VO|)`。

## 3. Notebook Cells

### Cell 1 — 代码与路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="fac99cde9dbae71684377b6f8702d7fe6d1cdd6b"

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
test -f "$REPO/analysis/structural_emergence/stage5_b_metrics/cross_terms.csv"
```

### Cell 2 — 预检（seed0 / 784 / L0）

预期：`cross_interaction.csv` **1 行**，`COMPLETE` 存在。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_c2_preflight
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
python -u analysis/structural_emergence/se_stage5_c2_cross.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --events 784 --layers 0

python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file()
rows = list(csv.DictReader((out/'cross_interaction.csv').open()))
assert len(rows) == 1, len(rows)
r = rows[0]
for key in ('I_H','I_taylor','I_actual','actual_QK','actual_VO','actual_joint'):
    assert math.isfinite(float(r[key])), (key, r[key])
assert abs(float(r['I_taylor']) - float(r['I_H'])) < 1e-8, (r['I_taylor'], r['I_H'])
print(r)
PY

touch "$PRE/PREFLIGHT_OK"
echo "PREFLIGHT_OK written"
```

### Cell 3 — 正式运行（后台；哨兵缺失时自动预检）

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_c2
PRE=/kaggle/working/stage5_c2_preflight

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
  echo "PREFLIGHT_OK missing; running C-2 preflight"
  rm -rf "$PRE"
  python -u analysis/structural_emergence/se_stage5_c2_cross.py \
    --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
    --out "$PRE" --seeds seed0 --events 784 --layers 0
  python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file()
rows = list(csv.DictReader((out/'cross_interaction.csv').open()))
assert len(rows) == 1, len(rows)
r = rows[0]
for key in ('I_H','I_taylor','I_actual'):
    assert math.isfinite(float(r[key]))
assert abs(float(r['I_taylor']) - float(r['I_H'])) < 1e-8
print('C-2 preflight OK')
PY
  touch "$PRE/PREFLIGHT_OK"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_c2_cross.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --seeds seed0,seed1 --events 784,1312 --layers 0,3 \
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
OUT=/kaggle/working/stage5_c2
PID=$(cat "$OUT/run.pid")
while kill -0 "$PID" 2>/dev/null; do
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  sleep 60
done
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "C-2 did not complete"; exit 1; }
echo "COMPLETE"
```

### Cell 5 — 验收

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_c2')
assert (out/'COMPLETE').is_file()
rows = list(csv.DictReader((out/'cross_interaction.csv').open()))
assert len(rows) == 8, len(rows)
for r in rows:
    for key in ('I_H','I_taylor','I_actual','actual_QK','actual_VO','actual_joint'):
        assert math.isfinite(float(r[key])), (r, key)
    assert abs(float(r['I_taylor']) - float(r['I_H'])) < 1e-8, r
print(json.dumps({"rows": len(rows), "summary": json.loads((out/'run_summary.json').read_text())}, indent=2))
PY
```

## 4. 解释

- `I_H` 是二阶 Hessian 意义上的 QK–VO 交叉项；
- `I_taylor` 与 `I_H` 应一致；
- `I_actual` 是实际损失中的交互；`I_residual = I_actual - I_taylor` 衡量高阶项；
- `I_actual < 0` 表示 QK+VO 联合更新比单独相加带来更多下降（协同）；
- `I_actual > 0` 表示抵消/干扰；
- `rho_H` 给出归一化交叉强度，但 Q/K 的 q 很小时可能不稳定，需结合 `q_QK/q_VO`。

## 5. 失败时

- Cell 2 失败：不要跑 Cell 3；贴 `stage5_c2_preflight/run.log`。
- `I_taylor` 与 `I_H` 不一致：检查 B 的 q 覆盖是否包含 L0/L3 全部 QK/VO 对。
- Cell 4 没有 `COMPLETE`：检查 `stage5_c2/run.log`。
