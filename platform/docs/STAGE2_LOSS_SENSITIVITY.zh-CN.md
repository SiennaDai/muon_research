# Stage 2 — 单步 loss 敏感度 Runbook

> 目标：在 pre-update 节点上，对 Muon 状态做**单步、固定 G** 的低精度替换，
> 评价 `loss_gap = L_q − L_ref`。回答：
>
> - 量化 `G / M_prev / D / U / W` 哪个状态最伤 loss；
> - 谱带 Top / Middle / Tail 的误差是否进 loss；
> - QK / VO / Q / K / V / O 的 loss 容忍度；
> - per-tensor / per-head / per-block、8-bit / 4-bit 的 loss 收益。
>
> 不训练、不计算 HVP、不做累积。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，Accelerator 选 **GPU T4**。
2. 打开 Internet。
3. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
4. 粘贴 Cell 1–5，`Save & Run All`。

预计冒烟 < 2 分钟；全量约 **5–15 分钟**；显存约 2–4 GiB。

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 代码 commit | `697ef96a8cd9c4c151c7bd2f3d1c6dbc19f7f8c0` |
| 脚本 | `analysis/lowp_first_batch/stage2_loss.py` |
| 节点 | 784, 1312 |
| 层 | L0, L3 |
| Probe | validation windows 0–7（2048 token） |
| 状态 | `G, M_prev, D, U, W` |
| 格式 | BF16、INT8 per-head（L1）；per-tensor/head/block、8/4-bit（L4） |
| 预期行数 | L1 80；L2 288；L3 576；L4 80；L2abs 192；L3abs 384；合计 1600 |

说明：本轮只用 784/1312，因为这两个节点在 `fp32-muon-ckpt-1536` 里有完整
`pre_u*_roles.pt`。128/1536 的 roles 在旧 Snapshot 数据集里，暂不纳入，避免额外挂载
15 GB 数据。

## 2. 输出

```text
l1_loss.csv
l2_loss.csv
l3_loss.csv
l4_loss.csv
l2abs_loss.csv
l3abs_loss.csv
summary.csv
SUMMARY.md
summary.json
run_summary.json
COMPLETE
```

每行共同字段：

```text
seed,node,layer,module,
L_base,L_ref,L_q,L_w_only,dL_ref,dL_q,loss_gap,rel_gap,sign_ok,sign_match,
rel_state,rel_U,cos_U,share_L1_U,top1_U,
rel_W,share_L1_W,top1_W,
state,format,space,band,group,alpha,noise,granularity,bits,block_size
```

- `L_ref`：FP32 参考步在 `L_probe` 上的 loss；
- `L_q`：量化步的 loss；
- `L_w_only`：仅 W 量化、未加 `ΔW_ref` 的前向 loss（仅 L1-W 行）；
- `loss_gap = L_q − L_ref`，正 = 比 FP32 差；
- `rel_gap = |loss_gap| / |dL_ref|`；
- `sign_match = sign(dL_q) == sign(dL_ref)`：量化步与 FP32 步的 loss 变化方向是否一致；
- `sign_ok = dL_q < 0` 仅作旧字段保留；L_probe 上 `dL_ref` 可能为正，不要用它做判据；
- `rel_state`：被量化状态本身的相对 Frobenius 误差；
- `rel_U / share_L1_U / top1_U`：更新方向/结构指标；
- `rel_W / share_L1_W / top1_W`：仅 W 量化时使用。

## 3. Notebook Cells

### Cell 1 — 代码与路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="697ef96a8cd9c4c151c7bd2f3d1c6dbc19f7f8c0"

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
test -f "$CKPT_ROOT/seed0/node_checkpoints/pre_u001312_roles.pt"
test -f "$DATA_ROOT/manifest.json"
test -f "$REPO/analysis/lowp_first_batch/stage2_loss.py"
```

### Cell 2 — 冒烟（全部模块各一个条件）

预期：`smoke_l1.csv`、`smoke_l2.csv`、`smoke_l3.csv`、`smoke_l4.csv`、
`smoke_l2abs.csv`、`smoke_l3abs.csv` 各 1 行；`COMPLETE` 存在。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage2_loss_preflight
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
python -u analysis/lowp_first_batch/stage2_loss.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --module all --smoke

python - "$PRE" <<'PY'
import csv, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file()
for name in ('smoke_l1.csv', 'smoke_l2.csv', 'smoke_l3.csv',
             'smoke_l4.csv', 'smoke_l2abs.csv', 'smoke_l3abs.csv'):
    rows = list(csv.DictReader((out / name).open()))
    assert len(rows) == 1, (name, len(rows))
    r = rows[0]
    for key in ('L_base', 'L_ref', 'L_q', 'dL_ref', 'dL_q', 'loss_gap'):
        assert math.isfinite(float(r[key])), (name, key, r[key])
    print(name, {k: r[k] for k in ('state', 'format', 'space', 'band',
                                   'group', 'alpha', 'granularity',
                                   'bits', 'loss_gap', 'rel_gap',
                                   'sign_match')})
print('smoke OK')
PY

touch "$PRE/PREFLIGHT_OK"
echo "PREFLIGHT_OK written"
```

### Cell 3 — 正式运行（后台）

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage2_loss
PRE=/kaggle/working/stage2_loss_preflight

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
  echo "PREFLIGHT_OK missing; refusing to run full"
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
nohup python -u analysis/lowp_first_batch/stage2_loss.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --module all --seeds seed0,seed1 --nodes 784,1312 --layers 0,3 \
  > "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "pid=$(cat "$OUT/run.pid")"
sleep 20
tail -n 10 "$OUT/run.log"
```

### Cell 4 — 阻塞监控

```bash
%%bash
set -uo pipefail
OUT=/kaggle/working/stage2_loss
PID=$(cat "$OUT/run.pid")
while kill -0 "$PID" 2>/dev/null; do
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  sleep 60
done
tail -n 30 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "Stage 2 did not complete"; exit 1; }
echo "COMPLETE"
```

### Cell 5 — 验收与 Kaggle 内分析

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage2_loss
python - "$OUT" <<'PY'
import csv, json, math, sys
from pathlib import Path
out = Path(sys.argv[1])
assert (out / 'COMPLETE').is_file()
expected = {'l1_loss.csv': 80, 'l2_loss.csv': 288,
            'l3_loss.csv': 576, 'l4_loss.csv': 80,
            'l2abs_loss.csv': 192, 'l3abs_loss.csv': 384}
for name, n in expected.items():
    rows = list(csv.DictReader((out / name).open()))
    assert len(rows) == n, (name, len(rows), n)
    for r in rows:
        for key in ('L_base', 'L_ref', 'L_q', 'dL_ref', 'dL_q', 'loss_gap'):
            assert math.isfinite(float(r[key])), (name, key, r)
    print(name, len(rows), 'OK')
print(json.dumps(json.loads((out / 'run_summary.json').read_text()), indent=2))
PY

cd "$REPO"
python -u analysis/lowp_first_batch/analyze_stage2.py --dir "$OUT"
test -f "$OUT/SUMMARY.md"
test -f "$OUT/summary.csv"
echo "--- SUMMARY.md —"
head -80 "$OUT/SUMMARY.md"
```

## 4. 模块解释

- **L1**：`G / M_prev / D / U / W` × BF16 / INT8 per-head。看哪个状态的 `rel_gap` 最大。
  - `W` 用固定 `ΔW_ref`，只做前向；不是完整 backward 重算。
- **L2**：D/U × Top/Middle/Tail × α=5%/10%。看 Tail 误差是否进 loss。
  - 扰动是 band 内匹配相对误差；`rel_state` 是相对整个矩阵的误差，通常小于 α。
- **L3**：D/U × QK/VO/Q/K/V/O × α=5%/10%。看类型 loss 容忍度。
  - QK 只扰动 c_attn 的 Q/K 行；VO 扰动 V 行 + O 列；其余组只扰动对应矩阵。
- **L4**：D/U × per-tensor / per-head / per-block / 8-bit / 4-bit。看粒度与位宽的 loss 收益。
  - `block` 默认 head 对齐；块大小 64/128。
- **L2abs**：同 L2，但用 **matched-absolute** 扰动：`‖E_band‖ = α_abs·‖D‖`，所有 band 相同；α_abs=3%/7%。
  - 用来回答“Tail 与 Top/Middle 的误差在同等绝对大小下，谁更伤 loss”。
- **L3abs**：同 L3，但用 **matched-absolute**：所有 group 的扰动总范数相同。
  - 用来确认 QK/VO 的差异不是组大小造成的。

## 5. 解释

- `loss_gap > 0`：量化步比 FP32 差；`rel_gap` 是相对 FP32 一步 loss 变化的比例。
- `sign_match = 0`：量化步与 FP32 步的 loss 变化方向不一致；这是比旧 `sign_ok` 更合适的判据。
- `rel_U` 高但 `loss_gap` 小：误差落在低曲率方向；`rel_U` 低但 `loss_gap` 大：误差落在高曲率方向。
- `share_L1_U` 大但 `loss_gap` 小：结构被破坏但短期 loss 不明显，累积阶段要重点看。
- L2 是 band 内相对误差；L2abs 是 band 间绝对误差。两者不能混着比较。

## 6. 失败时

- Cell 2 失败：不要跑 Cell 3；贴 `stage2_loss_preflight/run.log`。
- `pre_u000784_roles.pt` 缺失：确认 Add Input 了 `siennadai/fp32-muon-ckpt-1536`。
- `data manifest not found`：确认 Add Input 了 `siennadai/lowp-benchmark-slimpajama-8x-v1`。
- Cell 4 没有 `COMPLETE`：检查 `stage2_loss/run.log`。
- `loss_gap` 为 NaN：检查该节点的 roles 是否完整、`W_after/W_before` 是否有限。
