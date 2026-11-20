# Stage 5 — Module B 正式 Runbook（Head Hessian 曲率与交叉耦合）

> 本 runbook 在 Kaggle T4 上以 **commit-and-run-all** 方式运行 Module B：
> 用 Stage 4 的 pre-update 节点 checkpoint 重放优化器更新，计算每个 Head 方向上的
> Hessian 曲率 `q_h`、归一化曲率 `κ_h`、Head 间交叉项 `q_hk/ρ_hk`，以及代表 Head 的
> Lanczos 顶部特征值。
>
> 这是 **分析**，不是训练；不修改任何 checkpoint，不写回模型。

## 0. 冻结事实

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/SiennaDai/muon_research` |
| 本 runbook 的代码 commit | `e60546e83ce6d965cf77868a5564dcae32ed4418` |
| 分析脚本 | `analysis/structural_emergence/se_stage5_hessian.py` |
| Checkpoint 数据集 | `siennadai/fp32-muon-ckpt-1536` |
| Checkpoint 根目录 | `.../structural_concentration_ckpt` |
| 数据 | `siennadai/lowp-benchmark-slimpajama-8x-v1` |
| 数据 fingerprint | `ba3ac442ab488991beaa93a65652a52fbe56899509fa46e5edfc288b86173489` |
| Seeds | seed0 / seed1 |
| Hessian 节点 | `32,128,512,768,784,1024,1312,1536` |
| Lanczos 层 | seed0: `0,2,3`;seed1: `0,3,4` |
| V/O 控制层 | `0` 与 `3`(Q/K 所有层) |
| L_probe | validation windows `0..7`(2048 token) |
| L_train | 节点 update 的 32 个训练窗口(8192 token),4×8-window 分块 |
| Lanczos | 30 步;每个 type 取 `||ΔW_h||` 最大的 Head |
| 预期运行时间 | T4 约 **1.5–2.5 小时**(模型加载 + Hessian) |
| 预期显存峰值 | 约 **7–8 GiB**(不要恢复 pilot 的 14 GiB 写法) |

## 1. Pilot 校准结论（必须保留）

| 观测 | 数值 | 正式设置 |
|---|---|---|
| `replay_G_rel_error` | `1.52e-5` | 容差 `1e-4`;超限停止 |
| HVP 对称性 | `2.0e-7` | 容差 `1e-5`(正式脚本不单独算随机对称性) |
| 8-window probe HVP | `0.64 s`,峰值 `6.87 GiB` | L_probe 固定 8 windows |
| `4×8` L_train HVP | `1.97 s`,pilot 峰值 `14.03 GiB` | **这是假象**:旧 pilot 的 `_ = hvp_time(...)` 保留了上一条 loss 图 |
| 16-window L_train HVP | OOM | 正式脚本只用 8-window 分块 |
| Lanczos 20 步 | `last_beta=0.010`,top≈`0.098` | 正式用 **30 步**;仍报告 `last_beta` |

> 正式脚本的 `HessianEngine.hvp()` **不返回 loss**,并显式 `del` 中间图;每个 8-window
> HVP 后释放。因此正式峰值应在 7–8 GiB 量级。若仍看到 >13 GiB,不要继续跑完整节点集,
> 先把 `run.log` 的节点/层位置贴回来。

## 2. 本轮计算什么

对每个 `(seed, node, layer)`:

### 2.1 重放该 update 的 ΔW

- 节点 `u` 是 **update u 开始前**的模型;
- 用 sidecar 的 `micro_batch_window_ids`(8×4 windows)重放梯度,做全局 `clip_grad_norm`
  (`grad_clip_norm=1.0`);
- 读取 checkpoint 的 Muon momentum,按 `M_t=μM_prev+G`、`D=G+μM_t`、`U=NS(D)` 重算;
- `ΔW = -lr·U - lr·wd·W_before`;
- 在有 Role 文件的新节点(784、1312)对比 `W_after-W_before` 和 `G`,写入
  `replay_validation.csv`。

### 2.2 L_probe 方向曲率与交叉项

对每个 Head block `h`(Q/K 所有层;V/O 在 L0、L3):

```text
q_h   = ΔW_hᵀ H_probe ΔW_h
κ_h   = q_h / ||ΔW_h||²
q_hk  = ΔW_hᵀ H_probe ΔW_k
ρ_hk  = q_hk / sqrt(|q_h q_k|)
```

- 每个 block 做一次 HVP,同时得到该 block 与所有 block 的交叉项;
- `cross_terms.csv` 保存完整矩阵(含对角),`rho` 保留符号;
- 曲率本身很小是正常的(`ΔW_h` 约 1e-4 量级);解释时看 `κ_h` 和 `ρ_hk`,不要只看 `q_h`。

### 2.3 L_train 方向曲率

- 用该 update 的 32 个训练窗口,4 个 8-window chunk;
- 对每个 block 的 `ΔW_h` 计算 4 次 HVP,取均值;
- 只保存对角方向曲率(`loss=train`),不在 L_train 上算完整交叉矩阵(成本过高);
- L_train 使用真实 update 数据,L_probe 使用固定验证集;两者并列用于判断
  “曲率来自当前 batch 还是数据分布”。

### 2.4 Lanczos

- 每个 `(seed,node,layer)` 对 Q/K(以及 L0/L3 的 V/O)各取 `||ΔW_h||` 最大的一个 Head;
- 30 步,随机初始块向量,全 reorthogonalization;
- 保存 top-4 Ritz 值和 `last_beta`;
- 只对代表 Head 做,不给所有 60 个 block 做。

## 3. Notebook Cells

### Cell 1 — 代码、数据、Checkpoint 路径

```bash
%%bash
set -euo pipefail
BASE=/kaggle/working
REPO="$BASE/muon_research"
CODE_COMMIT="e60546e83ce6d965cf77868a5564dcae32ed4418"

[ -d "$REPO/.git" ] || git clone https://github.com/SiennaDai/muon_research "$REPO"
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
p = hits[0]
# .../structural_concentration_ckpt/seed0/node_checkpoints/pre_u000784.pt
print(p.parents[2])
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

### Cell 2 — 单节点预检(建议保留)

先用 seed0、节点 784、Layer 2 跑 ~1–2 分钟,确认脚本路径、replay 和 HVP 都正常。
预检只写 `stage5_hessian_preflight/`,不污染正式输出。

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
export CKPT_ROOT="$(python - <<'PY'
from pathlib import Path
p = next(p for p in Path('/kaggle/input').rglob('pre_u000784.pt')
         if 'fp32-muon-ckpt-1536' in str(p) and p.parent.name == 'node_checkpoints')
print(p.parents[2])
PY
)"
export DATA_ROOT="$(python - <<'PY'
from pathlib import Path
print(next(Path('/kaggle/input').rglob('slimpajama_8x/manifest.json')).parent)
PY
)"
cd "$REPO"
python -u analysis/structural_emergence/se_stage5_hessian.py \
  --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT" \
  --out /kaggle/working/stage5_hessian_preflight \
  --seeds seed0 --nodes 784 --layers 2
python - <<'PY'
import csv
from pathlib import Path
out = Path('/kaggle/working/stage5_hessian_preflight')
rows = list(csv.DictReader((out/'replay_validation.csv').open()))
assert rows, 'no replay validation rows: role file missing?'
for r in rows:
    print(r)
    assert float(r['g_rel_error']) < 1e-4, r
    assert float(r['delta_w_rel_error']) < 1e-4, r
d = list(csv.DictReader((out/'direction_curvature.csv').open()))
assert len(d) == 24, len(d)  # 12 blocks x (probe, train)
print('preflight direction rows', len(d))
PY
```

### Cell 3 — 正式运行(后台)

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/muon_research
OUT=/kaggle/working/stage5_hessian
mkdir -p "$OUT"
export CKPT_ROOT="$(python - <<'PY'
from pathlib import Path
p = next(p for p in Path('/kaggle/input').rglob('pre_u000784.pt')
         if 'fp32-muon-ckpt-1536' in str(p) and p.parent.name == 'node_checkpoints')
print(p.parents[2])
PY
)"
export DATA_ROOT="$(python - <<'PY'
from pathlib import Path
print(next(Path('/kaggle/input').rglob('slimpajama_8x/manifest.json')).parent)
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

### Cell 4 — 阻塞监控(每 5 分钟)

commit 会话会一直保持,直到后台进程退出;每 5 分钟打印进度、显存和最后一行日志。

```bash
%%bash
set -uo pipefail
OUT=/kaggle/working/stage5_hessian
PID=$(cat "$OUT/run.pid")
echo "monitor started for pid=$PID"
while true; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] process exited"
    break
  fi
  N=$(grep -c '^\[stage5\]' "$OUT/run.log" 2>/dev/null || true)
  echo "[$(date -u +%H:%M:%S)] completed_layer_nodes=$N"
  tail -n 2 "$OUT/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
  sleep 300
done
wait "$PID" 2>/dev/null || true
echo "--- final log tail ---"
tail -n 20 "$OUT/run.log"
```

### Cell 5 — 验收与行数检查

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from pathlib import Path
out = Path('/kaggle/working/stage5_hessian')

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

> 行数按设计层/节点计算:每个 `(seed,node)` 60 个 block;probe+train 120 行,
> cross 60²=3600 行;Lanczos 每 (seed,node) 10 个代表 Head ×4 =40 行;
> 16 个 `(seed,node)` 得到 1920/57600/640。Role 文件只在新节点 784/1312 存在,
> 每个节点 2 个参数,共 8 行。

### Cell 6 — 从 CSV 生成图(可选,但建议)

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
plt.xlabel('node (pre-update)'); plt.ylabel('kappa_h'); plt.yscale('symlog', linthresh=1e-6)
plt.legend(fontsize=7, ncol=2); plt.title('Stage 5 B: head direction curvature')
plt.tight_layout(); plt.savefig(fig/'kappa_curves.png', dpi=160); plt.close()

# cross heatmap for the largest node, layer 3, seed0
cross = list(csv.DictReader((out/'cross_terms.csv').open()))
labels, mat = [], []
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
plt.tight_layout(); plt.savefig(fig/'cross_seed0_node1536_L3.png', dpi=160); plt.close()
print('figures:', sorted(p.name for p in fig.iterdir()))
PY
```

## 4. 运行后如何解释

| 量 | 含义 | 关注点 |
|---|---|---|
| `κ_h` (probe) | 固定验证集上沿该 Head 更新的曲率 | 哪些 Head 在形成期变陡/变平 |
| `κ_h` (train) | 当前 update batch 上的曲率 | 与 probe 的差异 = batch 特异性 |
| `ρ_hk` | 两 Head 方向的 Hessian 耦合(归一化,带符号) | 同层 head block 的正/负耦合;Q/K 块结构 |
| Lanczos top-4 | 该 Head block 内部的主曲率谱 | 是否出现少数主方向;`last_beta` 是否仍大 |
| `replay_validation` | ΔW / G 重放误差 | 必须 <1e-4;否则后续数值不可信 |

这不是 Module A/C 的损失分解;不要在本轮计算 `ΔL` 或 cross-projection。B 只回答
“更新方向上的局部曲率与 Head 间耦合”,C 再用这些结果做损失变化和机制解释。

## 5. commit-and-run-all 说明

- 用 **Save & Run All**;Cell 3 后台启动,Cell 4 阻塞直到进程结束,因此 commit 不会提前关闭。
- Cell 2 预检失败时 commit 直接停止,不会进入 2 小时正式运行;这是有意的。
- 产物在 `/kaggle/working/stage5_hessian/`:
  - `direction_curvature.csv`
  - `cross_terms.csv`
  - `lanczos.csv`
  - `replay_validation.csv`
  - `run_summary.json`
  - `run.log` / `run.pid`
  - `figures/`(Cell 6)
- 预计运行 1.5–2.5 小时;完成后用 `kaggle kernels output <owner>/<notebook> -p <dest>` 下载。
- 若 OOM:记录 `run.log` 最后的 `[stage5]` 行、`nvidia-smi` 和节点/层;不要自行改 batch
  或删节点,先贴回来。正式脚本已经按 8-window 分块和非保留 loss 的写法设计。
