# Stage 5 — Module B 正式 Runbook（全新 Notebook 版）

> 用途：在 **全新的 Kaggle Notebook** 中，以 commit-and-run-all 方式运行 Module B：
> 用 Stage 4 的 pre-update 节点 checkpoint 重放优化器更新，计算每个 Head 方向上的
> Hessian 曲率 `q_h`、归一化曲率 `κ_h`、Head 间交叉项 `q_hk/ρ_hk`，以及代表 Head 的
> Lanczos 顶部特征值。
>
> 这是分析，不是训练；不修改 checkpoint，不写回模型。
>
> 旧的 notebook / 旧输出不要复用；本 runbook 从全新 `/kaggle/working` 开始。

## 0. 新 Notebook 设置

1. 新建 Kaggle Notebook，Accelerator 选 **GPU T4**（一张即可；脚本用 `cuda:0`）。
2. 打开 **Internet**（Cell 1 要 clone 仓库）。
3. Add Input 添加两个数据集：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
4. 把下面 Cell 1–6 依次粘进去；`Save & Run All`。
5. **Cell 2 必须通过**。Cell 3 会检查 `PREFLIGHT_OK` 哨兵；预检失败时不会进入正式运行。
6. Cell 4 是阻塞监控，每 5 分钟打印一次；它保证 commit 会话不会在后台进程结束前关闭。

预计：

| 项目 | 预计 |
|---|---:|
| 预检 | 2–3 分钟 |
| 正式运行 | 1.5–2.5 小时（T4） |
| 显存峰值 | 约 7–8 GiB |
| 正式输出 | 约几十 MB（CSV + 图） |

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 本文代码 commit | `11eba88277b98426a858ac145a2883d8f31dda47` |
| 分析脚本 | `analysis/structural_emergence/se_stage5_hessian.py` |
| Checkpoint 数据集 | `siennadai/fp32-muon-ckpt-1536` |
| Checkpoint 根目录 | `.../structural_concentration_ckpt` |
| 数据 | `siennadai/lowp-benchmark-slimpajama-8x-v1` |
| 数据 fingerprint | `ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489` |
| Seeds | seed0 / seed1 |
| Hessian 节点 | `32,128,512,768,784,1024,1312,1536` |
| 层 | seed0: `0,2,3`；seed1: `0,3,4` |
| V/O 控制层 | L0 与 L3（Q/K 覆盖所有选定层） |
| L_probe | validation windows `0..7`（2048 token） |
| L_train | 节点 update 的 32 个训练窗口（8192 token），4×8-window 分块 |
| Lanczos | 30 步；每个 type 取 `||ΔW_h||` 最大的 Head，top-4 |
| 预检 | seed0 / node 784 / L3（Q/K/V/O） |

## 2. 关键实现说明

### 2.1 重放 ΔW

节点 `u` 是 **update u 开始前**的模型。正式脚本：

- 用 sidecar 的 `micro_batch_window_ids`（8×4 windows）逐 micro-batch 重放；
- 每个 loss 除以 `accumulation_steps`，再对全部参数做一次 `clip_grad_norm`；
- 读取 checkpoint 的 Muon momentum，按 `M_t=μM_prev+G`、`D=G+μM_t`、`U=NS(D)` 重算；
- `ΔW = -lr·U - lr·wd·W_before`；
- 在节点 784/1312 与 Role 文件的 `G`、`W_after-W_before` 对比，写入
  `replay_validation.csv`。

### 2.2 交叉项必须用完整 HVP

```text
q_h   = ΔW_hᵀ H_probe ΔW_h
κ_h   = q_h / ||ΔW_h||²
q_hk  = ΔW_hᵀ H_probe ΔW_k
ρ_hk  = q_hk / sqrt(|q_h q_k|)
```

- 对每个源 block `k` 做一次**完整模型 HVP**；
- 对每个目标 block `h`，取 `HΔW_k` 在 h 参数块上的分量，flatten 后与 `ΔW_h` 点积；
- 这样 Q 行块、O 列块以及跨参数（`c_attn` vs `c_proj`）的交叉项才正确；
- `cross_terms.csv` 保存完整矩阵（含对角），`rho` 保留符号；
- 旧 pilot 里的 `q_hk` 只用了源 block 分量，已作废；本 commit 是修正版。

### 2.3 L_probe vs L_train

- **L_probe** 固定 validation windows 0–7：用于跨节点比较局部几何，避免数据 batch 混杂；
- **L_train** 使用该 update 的真实 32 个训练窗口，4 个 8-window chunk 的 HVP 平均；
- L_train 只算对角方向曲率，不算完整交叉矩阵（成本过高）；
- 两者并列用于区分“曲率随参数变化”与“曲率只是 batch 统计差异”。

### 2.4 Lanczos

- 每个 `(seed,node,layer)` 对 Q/K（以及 L0/L3 的 V/O）各取 `||ΔW_h||` 最大的一个 Head；
- 30 步，随机初始块向量，全 reorthogonalization；
- 保存 top-4 Ritz 值和 `last_beta`；
- 30 步是 pilot 校准后的选择（20 步的 `last_beta` 相对 top 特征值仍约 10%）。

### 2.5 数值校准

| 观测 | 数值 |
|---|---:|
| 8-window probe HVP | 0.64 s，6.87 GiB |
| replay G 相对误差 | 约 4e-7（预检） |
| ΔW 相对误差 | 2.55e-5 / 5.90e-6（预检） |
| HVP 对称性 | 2.0e-7 |
| 旧 pilot chunk8 峰值 | 14.03 GiB（loss 图保留导致的假象） |

正式脚本的 `hvp()` 不返回 loss，并显式释放图；L_train 只用 8-window 分块。若正式运行
看到 >13 GiB，不要继续扩大节点集，先贴 `run.log`。

## 3. Notebook Cells

### Cell 1 — 代码、数据、Checkpoint 路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="11eba88277b98426a858ac145a2883d8f31dda47"

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
python - <<'PY'
import json, os
from pathlib import Path
m = json.loads((Path(os.environ['DATA_ROOT']) / 'manifest.json').read_text())
assert m['fingerprint'] == 'ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489'
print('data fingerprint OK')
PY
```

### Cell 2 — V/O 预检（必须通过）

覆盖 Q/K/V/O 以及跨参数块交叉项；只写 `stage5_hessian_preflight/`。

预期：

- `direction_curvature.csv` 48 行（24 blocks × probe/train）
- `cross_terms.csv` 576 行（24×24）
- `lanczos.csv` 16 行（4 个代表 Head × top-4）
- `replay_validation.csv` 2 行

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
PRE=/kaggle/working/stage5_hessian_preflight
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
python -u analysis/structural_emergence/se_stage5_hessian.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out "$PRE" --seeds seed0 --nodes 784 --layers 3

python - <<'PY'
import csv, math
from pathlib import Path
out = Path('/kaggle/working/stage5_hessian_preflight')
assert (out / 'COMPLETE').is_file(), 'analysis did not write COMPLETE'
def rows(name):
    p = out / name
    assert p.is_file(), p
    return list(csv.DictReader(p.open()))
direction = rows('direction_curvature.csv')
cross = rows('cross_terms.csv')
lanczos = rows('lanczos.csv')
replay = rows('replay_validation.csv')
assert len(direction) == 48, len(direction)
assert len(cross) == 576, len(cross)
assert len(lanczos) == 16, len(lanczos)
assert len(replay) == 2, len(replay)
for r in replay:
    print(r)
    assert float(r['g_rel_error']) < 1e-4, r
    assert float(r['delta_w_rel_error']) < 1e-4, r
finite = [float(r['kappa']) for r in direction if r['loss'] in {'probe','train'}]
assert all(math.isfinite(x) for x in finite)
print('V/O preflight OK:', len(direction), len(cross), len(lanczos), len(replay))
PY

touch "$PRE/PREFLIGHT_OK"
echo "PREFLIGHT_OK written"
```

### Cell 3 — 正式运行（后台）

先检查预检哨兵；没有通过就不会启动正式 run。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_hessian
PRE=/kaggle/working/stage5_hessian_preflight
test -f "$PRE/PREFLIGHT_OK"
rm -rf "$OUT"
mkdir -p "$OUT"

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
export PYTHONUNBUFFERED=1
nohup python -u analysis/structural_emergence/se_stage5_hessian.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" --out "$OUT" \
  --seeds seed0,seed1 \
  --nodes 32,128,512,768,784,1024,1312,1536 \
  --lanczos-steps 30 \
  > "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "pid=$(cat "$OUT/run.pid")"
sleep 20
tail -n 5 "$OUT/run.log"
```

### Cell 4 — 阻塞监控（每 5 分钟）

进程结束后检查 `COMPLETE` 标记；没有标记就失败退出，Cell 5 不会误报通过。

```bash
%%bash
set -uo pipefail
OUT=/kaggle/working/stage5_hessian
PID=$(cat "$OUT/run.pid")
echo "monitor started for pid=$PID"
while kill -0 "$PID" 2>/dev/null; do
  N=$(grep -c '^\[stage5\]' "$OUT/run.log" 2>/dev/null || true)
  echo "[$(date -u +%H:%M:%S)] completed_layer_nodes=$N"
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
  sleep 300
done
echo "[$(date -u +%H:%M:%S)] process exited"
echo "--- final log tail ---"
tail -n 20 "$OUT/run.log"
test -f "$OUT/COMPLETE" || { echo "Module B run did not complete"; exit 1; }
echo "COMPLETE marker present"
```

### Cell 5 — 验收与行数检查

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_hessian')
assert (out / 'COMPLETE').is_file(), 'missing COMPLETE'

def rows(name):
    p = out / name
    assert p.is_file(), p
    return list(csv.DictReader(p.open()))

direction = rows('direction_curvature.csv')
cross = rows('cross_terms.csv')
lanczos = rows('lanczos.csv')
replay = rows('replay_validation.csv')

assert len(direction) == 1920, len(direction)
assert len(cross) == 57600, len(cross)
assert len(lanczos) == 640, len(lanczos)
assert len(replay) == 8, len(replay)

for r in replay:
    assert float(r['g_rel_error']) < 1e-4, r
    assert float(r['delta_w_rel_error']) < 1e-4, r

finite = [float(r['kappa']) for r in direction if r['loss'] in {'probe','train'}]
assert all(math.isfinite(x) for x in finite)
assert max(abs(x) for x in finite) > 0

summary = json.loads((out/'run_summary.json').read_text())
print(json.dumps({
    'direction_rows': len(direction),
    'cross_rows': len(cross),
    'lanczos_rows': len(lanczos),
    'replay_rows': len(replay),
    'max_replay_g_rel_error': max(float(r['g_rel_error']) for r in replay),
    'max_replay_dw_rel_error': max(float(r['delta_w_rel_error']) for r in replay),
    'last_beta_max': max(float(r['last_beta']) for r in lanczos),
    'runs': summary['runs'],
}, indent=2))
PY
du -sh /kaggle/working/stage5_hessian
```

行数来源：每个 `(seed,node)` 60 个 block；probe+train 120 行，cross 60²=3600 行；
Lanczos 每 `(seed,node)` 10 个代表 Head ×4=40 行；16 个 `(seed,node)` 得到
1920/57600/640。Role 文件只在新节点 784/1312，两个参数共 8 行。

### Cell 6 — 从 CSV 生成图（可选）

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('/kaggle/working/stage5_hessian')
fig = out / 'figures'
fig.mkdir(exist_ok=True)
rows = list(csv.DictReader((out/'direction_curvature.csv').open()))

series = defaultdict(list)
for r in rows:
    if r['loss'] not in {'probe','train'}:
        continue
    key = (r['seed'], int(r['layer']), r['type'], int(r['head']), r['loss'])
    series[key].append((int(r['node']), float(r['kappa'])))

for (seed, layer, typ, head, loss), vals in sorted(series.items()):
    if typ not in {'Q','K'} or head not in {0,1,2}:
        continue
    vals.sort()
    plt.plot([v[0] for v in vals], [v[1] for v in vals],
             marker='o', ms=3, label=f'{typ}H{head} {loss}')
plt.xlabel('node (pre-update)')
plt.ylabel('kappa_h')
plt.yscale('symlog', linthresh=1e-6)
plt.legend(fontsize=7, ncol=2)
plt.title('Stage 5 B: head direction curvature')
plt.tight_layout()
plt.savefig(fig/'kappa_curves.png', dpi=160)
plt.close()

cross = list(csv.DictReader((out/'cross_terms.csv').open()))
sel = [r for r in cross if r['seed']=='seed0' and int(r['node'])==1536 and int(r['layer'])==3]
keys = sorted({(r['h_type'], int(r['h_head'])) for r in sel})
idx = {k:i for i,k in enumerate(keys)}
mat = [[0.0]*len(keys) for _ in keys]
for r in sel:
    mat[idx[(r['h_type'], int(r['h_head']))]][idx[(r['k_type'], int(r['k_head']))]] = float(r['rho'])
plt.figure(figsize=(8,7))
im = plt.imshow(mat, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='rho_hk (L_probe)')
plt.xticks(range(len(keys)), [f'{a}{b}' for a,b in keys], rotation=90, fontsize=6)
plt.yticks(range(len(keys)), [f'{a}{b}' for a,b in keys], fontsize=6)
plt.title('Stage 5 B: cross-head coupling, seed0 node 1536 L3')
plt.tight_layout()
plt.savefig(fig/'cross_seed0_node1536_L3.png', dpi=160)
plt.close()
print('figures:', sorted(p.name for p in fig.iterdir()))
PY
```

## 4. 产物

正式输出在 `/kaggle/working/stage5_hessian/`：

```text
COMPLETE
direction_curvature.csv   1920 行
cross_terms.csv           57600 行
lanczos.csv               640 行
replay_validation.csv     8 行
run_summary.json
run.log / run.pid
figures/                  Cell 6 生成
```

预检输出在 `/kaggle/working/stage5_hessian_preflight/`，不要与正式输出混用。

## 5. 解释口径

| 量 | 含义 | 关注点 |
|---|---|---|
| `κ_h` (probe) | 固定验证集上沿该 Head 更新的曲率 | 哪些 Head 在形成期变陡/变平 |
| `κ_h` (train) | 当前 update batch 上的曲率 | 与 probe 的差异 = batch 特异性 |
| `ρ_hk` | 两 Head 方向的 Hessian 耦合（归一化，带符号） | 同层 head block 的正/负耦合；Q/K 块结构 |
| Lanczos top-4 | 该 Head block 内部的主曲率谱 | 是否出现少数主方向；`last_beta` 是否仍大 |
| `replay_validation` | ΔW / G 重放误差 | 必须 <1e-4；否则后续数值不可信 |

B 只回答“更新方向上的局部曲率与 Head 间耦合”，不做 ΔL / cross-projection；
C 再用这些结果做损失变化与 Muon 动力学解释。

## 6. 失败时

- **Cell 2 失败**：不要运行 Cell 3；把 `stage5_hessian_preflight/run.log` 最后 30 行贴回来。
- **Cell 4 报 `Module B run did not complete`**：说明后台进程失败；检查 `stage5_hessian/run.log`。
- **OOM**：记录最后 `[stage5]` 行、节点/层和 `nvidia-smi`；不要自行扩大 chunk。
- **行数不符**：先看 `run_summary.json` 的 `runs` 和 `run.log`，不要直接进入图分析。
