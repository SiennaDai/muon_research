# VQ 曲率加权单步代理 Runbook（analysis only）

> 目的：验证一个便宜的单步指标能否**复现 4096 步的 NLL 排序**。
>
> 对每个候选状态表示（base / C / C3），在 FP32 pre-update 节点上计算
>
> ```
> dL ≈ g·dW + ½ dWᵀH dW ,   dW = −lr·(U_hat − U_ref)
> ```
>
> 其中 H 是 `L_probe`（validation windows 0–7）的 Hessian。
> 把它与已经测得的 4096 步排序比较：**base 4.7584 < C 4.7756 < C3 4.8061**。
>
> 这是分析，不训练、不写回 checkpoint。

## 1. 冻结事实

| 项目 | 固定值 |
|---|---|
| 代码仓库 | `https://github.com/SiennaDai/muon_research` |
| 代码 commit | **`39c7259`** |
| 脚本 | `analysis/structural_emergence/se_vq_curvature_proxy.py` |
| 码本仓库 | `https://github.com/SiennaDai/benchmark-experiments`，commit **`ba2d39d`** |
| Checkpoint 数据集 | `siennadai/fp32-muon-ckpt-1536`（根目录 `structural_concentration_ckpt`） |
| 数据 | `siennadai/lowp-benchmark-slimpajama-8x-v1`（目录 `slimpajama_8x`） |
| Seed / 节点 | `seed0` / `1536` |
| 层 | 全部 6 层 |
| L_probe | validation windows 0–7（2048 token） |
| 候选 | `base`（uniform r8 + 64 词）、`C`（逐矩阵 rank + 64 词）、`C3`（同 rank + 32 词） |
| 输出 | `/kaggle/working/vq_curvature_proxy/{curvature_proxy.csv,summary.json}` |

预计：T4 单卡 **10–25 分钟**，显存约 6–8 GiB（每个候选/层一次 HVP）。

## 2. Notebook 设置

1. 新建 Kaggle Notebook，Accelerator **GPU T4**（一张），开启 Internet。
2. Add Input：
   - `siennadai/fp32-muon-ckpt-1536`
   - `siennadai/lowp-benchmark-slimpajama-8x-v1`
3. 顺序运行 Cell 1–4。

## 3. Cell 1 — 拉取两个仓库

```python
import os, subprocess

MUON = "https://github.com/SiennaDai/muon_research.git"
MUON_REF = "39c7259"
BENCH = "https://github.com/SiennaDai/benchmark-experiments.git"
BENCH_REF = "ba2d39d"

for repo, ref, path in ((MUON, MUON_REF, "/kaggle/working/muon_research"),
                        (BENCH, BENCH_REF, "/kaggle/working/benchmark-experiments")):
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(["git", "clone", repo, path], check=True)
    subprocess.run(["git", "-C", path, "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", path, "checkout", "--detach", ref], check=True)
    print(path, subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip())
```

## 4. Cell 2 — 预检路径

```python
import glob
from pathlib import Path

def resolve(pattern, suffix=""):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise SystemExit(f"not found: {pattern}")
    return Path(hits[0]) / suffix if suffix else Path(hits[0])

CKPT = resolve("/kaggle/input/**/fp32-muon-ckpt-1536", "structural_concentration_ckpt")
DATA = resolve("/kaggle/input/**/lowp-benchmark-slimpajama-8x-v1", "slimpajama_8x")
BOOK = Path("/kaggle/working/benchmark-experiments/reports/"
            "muon_vector_int3_robustness/calibration_codebooks.pt")

need = [CKPT / "seed0" / "node_checkpoints" / "pre_u001536.pt",
        CKPT / "seed0" / "node_checkpoints" / "pre_u001536.json",
        DATA / "manifest.json",
        BOOK]
for item in need:
    assert item.exists(), f"missing: {item}"
    print("ok:", item)
print("CKPT", CKPT)
print("DATA", DATA)
```

## 5. Cell 3 — 运行代理

```python
import subprocess

cmd = [
    "python", "-u", "analysis/structural_emergence/se_vq_curvature_proxy.py",
    "--ckpt-root", str(CKPT),
    "--data-root", str(DATA),
    "--codebook", str(BOOK),
    "--seed", "seed0",
    "--nodes", "1536",
    "--layers", "0", "1", "2", "3", "4", "5",
    "--out", "/kaggle/working/vq_curvature_proxy",
]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd="/kaggle/working/muon_research", check=True)
```

## 6. Cell 4 — 结果

```python
import json
from pathlib import Path

out = Path("/kaggle/working/vq_curvature_proxy")
summary = json.loads((out / "summary.json").read_text())
print(f"{'cand':6s} {'pred_gap':>12s} {'rel_u':>8s} {'NLL4096':>9s} {'vs base':>9s}")
base = summary["base"]["predicted_gap_mean"]
for cand in ("base", "C", "C3"):
    s = summary[cand]
    print(f"{cand:6s} {s['predicted_gap_mean']-base:+12.6f} {s['rel_u_mean']:8.4f} "
          f"{s['nll_4096_s0']:9.4f} {s['nll_4096_s0']-4.7584:+9.4f}")
```

## 7. 判读

| 现象 | 结论 |
|---|---|
| 预测排序 = base < C < C3，且 rel_u 排序不同 | **曲率加权代理有效**，可作为便宜的长期筛子 |
| 预测排序与 rel_u 一致、但与 NLL 不一致 | H 加权没有带来额外信息，单步代理到此为止 |
| 预测排序与 NLL 一致但幅度差很多 | 只能当**序数**代理，用于排除候选 |
| 预测排序都不对 | 长程效应不在探针点的一阶/二阶里，需要短程 rollout |

额外看点：

- `first_order_mean` 与 `second_order_mean` 各自的方向——如果一阶项就能排序，说明主要靠梯度对齐；
- `rel_u_mean` 的顺序是 base < C < C3（比特越多越好），与 NLL 同向，所以真正的测试是
  **预测出来的差距比例**是否与 NLL 的 (+0.0172, +0.0477) 接近。

## 8. 跑完之后

```bash
kaggle kernels output <owner>/<kernel> -p /tmp/kout-proxy \
  --file-pattern '(summary\.json|curvature_proxy\.csv|\.log)$'
```

我会把预测差距与 4096 NLL 差距做相关，并给出“是否值得把它作为标准筛选指标”的结论。
