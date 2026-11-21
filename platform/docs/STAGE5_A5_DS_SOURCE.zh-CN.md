# Stage 5 — A-5 dS Source Decomposition Runbook

> 目标：把 `dS_h` 拆成 `dP`、softmax 中心化、attention 加权三步；再把 `dP` 拆成 `dA`、`V`、乘法效率。
>
> 只做 forward/backward；不训练，不做 Hessian。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，GPU T4，Internet；
2. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
3. 粘贴 Cell 1–5，`Save & Run All`。

预计 15–30 分钟；显存约 2–3 GiB。

## 1. 方法

对每个 attention Head：

```text
dP_h = dA_h V_hᵀ
m_h(i) = Σ_j A_h(i,j) dP_h(i,j)
C_h = dP_h − m_h
dS_h = A_h ⊙ C_h
```

精确恒等式：

```text
||dS_h||² = ||dP_h||² · c_ratio_h · g_h
c_ratio_h = ||C_h||² / ||dP_h||²
g_h       = ||dS_h||² / ||C_h||²

||dP_h||² = ||dA_h||² · ||V_h||² · eta_p,h
eta_p,h   = ||dA_h V_hᵀ||² / (||dA_h||² ||V_h||²)
```

三项含义：

- `dP`：上游 `dA` 与 `V` 的乘积，表示原始 score 梯度信号；
- `c_ratio`：减去 A 加权行均值后的能量比。A 接近均匀时通常略小于 1；A 加权均值与行和反号时可能大于 1；
- `g`：`A` 对中心化信号的加权保留比例，`0 < g ≤ 1`；
- `eta_p`：`dA` 与 `V` 的乘法效率，`0 < eta_p ≤ 1`。

再把 `dP` 拆成 `dA` 范数、`V` 范数、`eta_p`。  
这样 `dS` 的 Head 差异可以分配到五个具体量：

```text
||dS||² = ||dA||² · ||V||² · eta_p · c_ratio · g
```

采集粒度：每个 `(seed,node,loss,layer,head,window)` 一行。  
probe 8 个窗口；train 使用 checkpoint sidecar 的 4 个 micro-batch 组、每组 4 个窗口，共 16 个窗口。

## 2. 代码与输出

脚本由 Cell 1 检出：

```text
commit 5198d06
analysis/structural_emergence/se_stage5_a5_ds_source.py
```

输出：

```text
ds_source_metrics.csv
  seed,node,loss,layer,head,window,
  da2,v2,dp2,eta_p,m2,dot_dp_m,c2,ds2,
  c_ratio,g,attn_entropy,attn_max_excl0,a2_fro,
  frac_cancel_gt50,ds_top10_share,
  dp_row_cv,c_row_cv,ds_row_cv,corr_ent_ds
```

行数：

- 预检（seed0 / node 784）：6 层 ×6 Head ×（8 probe +16 train）= **864 行**
- 正式：62 个 `(seed,node)` → **53,568 行**（probe 17,856；train 35,712）

## 3. Notebook Cells

### Cell 1 — 代码与路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="5198d06e474011d8daace1dd0e491c20335e3ece"

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

### Cell 2 — 预检（seed0 / node 784，probe+train）

预期：864 行；probe 288、train 576；恒等式与边界通过。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_a5_preflight
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
python -u analysis/structural_emergence/se_stage5_a5_ds_source.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --nodes 784 --losses probe,train

python - "$PRE" <<'PY'
import csv, math, sys
from collections import Counter, defaultdict
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file()
rows = list(csv.DictReader((out / 'ds_source_metrics.csv').open()))
assert len(rows) == 864, len(rows)
counts = Counter(r['loss'] for r in rows)
assert counts['probe'] == 288, counts
assert counts['train'] == 576, counts
nums = ['da2', 'v2', 'dp2', 'eta_p', 'm2', 'dot_dp_m', 'c2', 'ds2',
        'c_ratio', 'g', 'attn_entropy', 'attn_max_excl0', 'a2_fro',
        'frac_cancel_gt50', 'ds_top10_share',
        'dp_row_cv', 'c_row_cv', 'ds_row_cv', 'corr_ent_ds']
max_id = 0.0
for r in rows:
    vals = {k: float(r[k]) for k in nums}
    assert all(math.isfinite(v) for v in vals.values()), r
    assert 0 < vals['eta_p'] <= 1 + 1e-5, vals['eta_p']
    assert 0 <= vals['g'] <= 1 + 1e-5, vals['g']
    assert vals['c_ratio'] >= 0
    assert vals['ds2'] <= vals['c2'] * (1 + 1e-5)
    err = abs(vals['c2'] - (vals['dp2'] + vals['m2'] - 2 * vals['dot_dp_m'])) / max(vals['dp2'], 1e-30)
    max_id = max(max_id, err)
assert max_id < 1e-4, max_id
win = defaultdict(set)
for r in rows:
    win[(r['loss'], r['layer'], r['head'])].add(int(r['window']))
for key, s in win.items():
    assert len(s) == (8 if key[0] == 'probe' else 16), (key, len(s))
print('A-5 preflight OK:', len(rows), 'rows; max c2 identity err', max_id)
PY
touch "$PRE/PREFLIGHT_OK"
```

### Cell 3 — 正式运行（后台；哨兵缺失时自动预检）

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_a5
PRE=/kaggle/working/stage5_a5_preflight

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
  echo "PREFLIGHT_OK missing; running A-5 preflight"
  rm -rf "$PRE"
  python -u analysis/structural_emergence/se_stage5_a5_ds_source.py \
    --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
    --out "$PRE" --seeds seed0 --nodes 784 --losses probe,train
  python - "$PRE" <<'PY'
import csv, math, sys
from collections import Counter
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file()
rows = list(csv.DictReader((out / 'ds_source_metrics.csv').open()))
assert len(rows) == 864, len(rows)
counts = Counter(r['loss'] for r in rows)
assert counts['probe'] == 288 and counts['train'] == 576, counts
print('A-5 preflight OK')
PY
  touch "$PRE/PREFLIGHT_OK"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_a5_ds_source.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --seeds seed0,seed1 --losses probe,train \
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
OUT=/kaggle/working/stage5_a5
PID=$(cat "$OUT/run.pid")
while kill -0 "$PID" 2>/dev/null; do
  N=$(grep -c '^\[stage5a5\]' "$OUT/run.log" 2>/dev/null || true)
  echo "[$(date -u +%H:%M:%S)] nodes=$N"
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
  sleep 300
done
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "A-5 did not complete"; exit 1; }
echo "COMPLETE"
```

### Cell 5 — 验收

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from collections import Counter
from pathlib import Path
out = Path('/kaggle/working/stage5_a5')
assert (out / 'COMPLETE').is_file()
rows = list(csv.DictReader((out / 'ds_source_metrics.csv').open()))
assert len(rows) == 53568, len(rows)
counts = Counter(r['loss'] for r in rows)
assert counts['probe'] == 17856, counts
assert counts['train'] == 35712, counts
nums = ['da2', 'v2', 'dp2', 'eta_p', 'm2', 'dot_dp_m', 'c2', 'ds2',
        'c_ratio', 'g', 'attn_entropy', 'attn_max_excl0', 'a2_fro',
        'frac_cancel_gt50', 'ds_top10_share',
        'dp_row_cv', 'c_row_cv', 'ds_row_cv', 'corr_ent_ds']
max_id = 0.0
for r in rows:
    vals = {k: float(r[k]) for k in nums}
    assert all(math.isfinite(v) for v in vals.values())
    assert 0 < vals['eta_p'] <= 1 + 1e-5
    assert 0 <= vals['g'] <= 1 + 1e-5
    assert vals['c_ratio'] >= 0
    assert vals['ds2'] <= vals['c2'] * (1 + 1e-5)
    err = abs(vals['c2'] - (vals['dp2'] + vals['m2'] - 2 * vals['dot_dp_m'])) / max(vals['dp2'], 1e-30)
    max_id = max(max_id, err)
assert max_id < 1e-4, max_id
print(json.dumps({'rows': len(rows), 'probe': counts['probe'],
                  'train': counts['train'], 'max_identity_err': max_id}, indent=2))
PY
du -sh /kaggle/working/stage5_a5
```

## 4. 解释

正式结果按以下顺序看：

1. **主分解**：在每个 `(seed,node,loss,layer,window)` 内跨 6 个 Head 对
   `log||dP||²`、`log c_ratio`、`log g` 做 LMG。回答 `dS` 的 Head 差异主要来自信号幅度、中心化还是 A 加权。
2. **次分解**：对 `log||dA||²`、`log||V||²`、`log eta_p` 做 LMG。回答 `dP` 的差异来自上游 `dA`、V 尺度还是乘法效率。
3. **Head 排序重建**：只用一项或去掉一项，比较重建份额与真实 `dS` 份额的 Spearman 和 top-1。
4. **位置统计**：`frac_cancel_gt50`、`ds_top10_share`、`dp_row_cv/c_row_cv/ds_row_cv`、`corr_ent_ds`。
5. **时间与 seed**：按 node 和 seed 汇总 LMG 与各项中位数。

`c_ratio` 不保证 ≤1：A 加权行均值与行和反号时，中心化后的能量可以大于 `dP` 能量。这个量是能量比，不是“保留比例”。

## 5. 失败时

- Cell 2 行数不是 864：先看 `stage5_a5_preflight/run.log`。train 行数取决于 sidecar 的 micro-batch 结构；若每组窗口数不同，按实际调整预检断言。
- 恒等式失败：检查 `layer_rows` 的 `m2`/`dot_dp_m` 定义；`m2 = T Σ_i m_i²`。
- Cell 4 没有 `COMPLETE`：检查 `stage5_a5/run.log`。
- 显存不足：把 `--probe-windows` 降到 4，或把 train 的 micro 组拆开跑。

## 6. 输出

```text
/kaggle/working/stage5_a5/
  ds_source_metrics.csv
  run_summary.json
  COMPLETE
  run.log
```
