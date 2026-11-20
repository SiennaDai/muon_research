# Stage 5 — Module B Hessian Pilot（交互式 Runbook）

> 本轮只在 Kaggle GPU Notebook 的交互界面手动执行,目标是测出真实 HVP 时间、显存、对称性容差、replay 误差和 Lanczos 代价。跑完把最后的 `pilot_B_summary.json` 贴回来,再据此生成正式的 commit-and-run-all runbook。
>
> **不要在本机运行这些 HVP**(WSL 只有 3.8 GiB 内存,会 OOM)。

## 前置

- Kaggle Notebook 开启 GPU(T4 即可);
- Add Input 挂载两个数据集:
  - `siennadai/fp32-muon-ckpt-1536`
  - `siennadai/lowp-benchmark-slimpajama-8x-v1`
- Pilot 范围:Seed 0、节点 **784**(有 base checkpoint + QKV/O Role)、Layer **2**、Q Head 2;
  L_probe = validation windows 0–7(2048 token);L_train = 节点 784 的 32 个训练窗口(8192 token)。

## Cell 1 — 路径与环境检查

```python
import os, sys, json, time, math, shutil
from pathlib import Path

INPUT = Path("/kaggle/input")

# Kaggle 新挂载布局是 /kaggle/input/datasets/<owner>/<slug>；
# 旧布局是 /kaggle/input/<slug>。两种都探测。
CKPT_CANDIDATES = [
    Path("/kaggle/input/datasets/siennadai/fp32-muon-ckpt-1536"),
    Path("/kaggle/input/fp32-muon-ckpt-1536"),
    *sorted(p for p in INPUT.rglob("fp32-muon-ckpt-1536") if p.is_dir()),
]
CKPT_DATASET = next(
    p for p in CKPT_CANDIDATES
    if (p / "structural_concentration_ckpt/seed0/node_checkpoints/pre_u000784.pt").is_file()
)
REPO = CKPT_DATASET / "muon_research"
RUN = CKPT_DATASET / "structural_concentration_ckpt/seed0"
NODE = RUN / "node_checkpoints"
DATA = next(p.parent for p in INPUT.rglob("slimpajama_8x/manifest.json"))

for p in (REPO / "platform/src", NODE / "pre_u000784.pt",
          NODE / "pre_u000784.json", NODE / "pre_u000784_roles.pt",
          DATA / "manifest.json"):
    print(p, p.exists())
assert (REPO / "platform/src/main.py").is_file()
assert (NODE / "pre_u000784_roles.pt").exists()
assert (DATA / "manifest.json").is_file()

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(REPO / "platform/src"))

import torch, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0),
      torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
```

## Cell 2 — 加载节点 784 的模型、数据与 sidecar

```python
from config.recipe import load_recipe
from data.frozen_tokens import load_manifest, FrozenWindows
from train_platform import build_model

device = torch.device("cuda:0")
cfg = load_recipe(REPO / "platform/recipes/structural_emergence_30m_fp32_muon_s0.json")
manifest = load_manifest(DATA / "manifest.json")
train = FrozenWindows(manifest, "train", 256)
val = FrozenWindows(manifest, "validation", 256)

model = build_model(cfg, device)
ck = torch.load(NODE / "pre_u000784.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"], strict=True)
model.train()
meta = json.loads((NODE / "pre_u000784.json").read_text())
print("completed", ck["completed_updates"], "windows", len(meta["window_ids"]))

def batch(window_ids, split):
    source = train if split == "train" else val
    pairs = [source.window(i) for i in window_ids]
    return (torch.stack([p[0] for p in pairs]).to(device),
            torch.stack([p[1] for p in pairs]).to(device))

x_probe, y_probe = batch(list(range(8)), "val")        # 2048 tokens
x_train, y_train = batch(meta["window_ids"], "train")  # 8192 tokens
print(x_probe.shape, x_train.shape)

roles = torch.load(NODE / "pre_u000784_roles.pt", map_location="cpu", weights_only=False)
NAME = "transformer.h.2.attn.c_attn.weight"
w_before = roles["tensors"][NAME + "|W_before"].to(device)
w_after = roles["tensors"][NAME + "|W_after"].to(device)
saved_g = roles["tensors"][NAME + "|G"].to(device)
dw = w_after - w_before
param = model.transformer.h[2].attn.c_attn.weight
params = [p for p in model.parameters() if p.requires_grad]
print("dw", tuple(dw.shape), "saved_g", tuple(saved_g.shape))
```

## Cell 3 — Replay G 与裁剪系数

```python
def train_loss():
    return model(x_train, y_train)["loss"]

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
model.zero_grad(set_to_none=True)
loss = train_loss()
grads = torch.autograd.grad(loss, params, allow_unused=True)
torch.cuda.synchronize()
t_grad = time.time() - t0
total_norm = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads if g is not None))
coef = min(1.0, 1.0 / float(total_norm))
idx = next(i for i, p in enumerate(params) if p is param)
g_opt = coef * grads[idx].detach()
rel = float((g_opt - saved_g).norm() / saved_g.norm())
print({"loss": float(loss.detach()), "grad_seconds": t_grad, "total_norm": float(total_norm),
       "clip_coef": coef, "replay_G_rel_error": rel,
       "peak_MiB": torch.cuda.max_memory_allocated() / 1024**2})
del loss, grads, g_opt
torch.cuda.empty_cache()
```

## Cell 4 — HVP 工具

> Hessian 需要 double backward;Flash/efficient SDPA 未实现该导数,因此 HVP 必须使用
> math attention。Cell 3 的 replay G 仍用训练时的 auto 后端;这里只影响后续 HVP。

```python
# Force the math attention path: required for second-order autograd.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def zero_vectors():
    return {id(p): torch.zeros_like(p) for p in params}

def block_vector(sl, generator=None):
    v = zero_vectors()
    block = torch.zeros_like(param)
    block[sl, :] = generator if generator is not None else torch.randn_like(block[sl, :])
    v[id(param)] = block
    return v

def hvp(loss_fn, v_by_id):
    model.zero_grad(set_to_none=True)
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    used = [(p, g) for p, g in zip(params, grads) if g is not None]
    hv = torch.autograd.grad([g for _, g in used], [p for p, _ in used],
                             grad_outputs=[v_by_id[id(p)] for p, _ in used],
                             allow_unused=True)
    hv_by_id = {id(p): h for (p, _), h in zip(used, hv) if h is not None}
    return loss, hv_by_id

def hvp_time(loss_fn, v):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    loss, hv = hvp(loss_fn, v)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return loss, hv, dt, peak
```

## Cell 5 — L_probe HVP、对称性、q_h、交叉项

```python
probe_loss = lambda: model(x_probe, y_probe)["loss"]
SL_H = slice(128, 192)   # Q Head 2
SL_K = slice(192, 256)   # Q Head 3

v = block_vector(SL_H)
w = block_vector(SL_H)
loss_v, hv_v, dt_v, peak_v = hvp_time(probe_loss, v)
probe_loss_value = float(loss_v.detach())
_, hv_w, dt_w, _ = hvp_time(probe_loss, w)
hv_v_block = hv_v[id(param)][SL_H, :]
hv_w_block = hv_w[id(param)][SL_H, :]
v_block = v[id(param)][SL_H, :]
w_block = w[id(param)][SL_H, :]
sym = float(abs((v_block * hv_w_block).sum() - (w_block * hv_v_block).sum())
            / max(abs((v_block * hv_w_block).sum()), 1e-30))

# 沿实际 ΔW_h 的方向曲率
dv = zero_vectors(); dv[id(param)][SL_H, :] = dw[SL_H, :]
_, hv_d, dt_d, peak_d = hvp_time(probe_loss, dv)
q_h = float((dw[SL_H, :] * hv_d[id(param)][SL_H, :]).sum())

# 与 QH3 的交叉项
dk = zero_vectors(); dk[id(param)][SL_K, :] = dw[SL_K, :]
_, hv_k, dt_k, _ = hvp_time(probe_loss, dk)
q_hk = float((dw[SL_H, :] * hv_k[id(param)][SL_H, :]).sum())

print({"probe": {"loss": probe_loss_value, "hvp_seconds": dt_v,
                 "symmetry_rel_error": sym, "q_h": q_h, "q_hk": q_hk,
                 "hvp_actual_seconds": dt_d, "hvp_cross_seconds": dt_k,
                 "peak_MiB": peak_v}})
del loss_v, hv_v, hv_w, hv_d, hv_k, v, w, dv, dk
torch.cuda.empty_cache()
```

## Cell 6 — L_train HVP、分块等价性

```python
train_loss_fn = lambda: model(x_train, y_train)["loss"]
vt = block_vector(SL_H)
_, hvt, dt_full, peak_full = hvp_time(train_loss_fn, vt)
full_block = hvt[id(param)][SL_H, :].detach().clone()
del hvt
torch.cuda.empty_cache()

chunks = [meta["window_ids"][i * 8:(i + 1) * 8] for i in range(4)]
chunk_hv = torch.zeros_like(full_block)
t0 = time.time()
for chunk in chunks:
    xc, yc = batch(chunk, "train")
    fn = lambda xc=xc, yc=yc: model(xc, yc)["loss"]
    bv = zero_vectors(); bv[id(param)][SL_H, :] = vt[id(param)][SL_H, :]
    _, hv, _, _ = hvp_time(fn, bv)
    chunk_hv += hv[id(param)][SL_H, :].detach()
    del hv, bv, xc, yc
    torch.cuda.empty_cache()
chunk_hv /= len(chunks)
chunk_rel = float((chunk_hv - full_block).norm() / full_block.norm())
chunk_seconds = time.time() - t0
print({"train": {"tokens": int(x_train.numel()), "hvp_seconds": dt_full,
                 "peak_MiB": peak_full, "chunk_seconds": chunk_seconds,
                 "chunk_rel_error": chunk_rel}})
```

## Cell 7 — 4 步 vs 20 步 Lanczos

```python
def lanczos_block(loss_fn, sl, steps=20):
    q0 = zero_vectors()
    q0[id(param)][sl, :] = torch.randn_like(param[sl, :])
    q0[id(param)][sl, :] /= q0[id(param)][sl, :].norm()
    Q = [q0[id(param)][sl, :]]
    alphas, betas = [], []
    t0 = time.time()
    for j in range(steps):
        vj = zero_vectors(); vj[id(param)][sl, :] = Q[-1]
        _, hv = hvp(loss_fn, vj)
        w = hv[id(param)][sl, :]
        alpha = float((Q[-1] * w).sum())
        w = w - alpha * Q[-1]
        if j > 0:
            w = w - betas[-1] * Q[-2]
        for qi in Q:
            w = w - (qi * w).sum() * qi
        beta = float(w.norm())
        alphas.append(alpha)
        if j < steps - 1 and beta > 1e-12:
            Q.append(w / beta)
            betas.append(beta)
        else:
            break
        del hv, vj
    torch.cuda.empty_cache()
    T = torch.diag(torch.tensor(alphas))
    for i, b in enumerate(betas):
        T[i, i + 1] = T[i + 1, i] = b
    return time.time() - t0, alphas, betas, torch.linalg.eigvalsh(T)

t4, a4, b4, e4 = lanczos_block(probe_loss, SL_H, 4)
t20, a20, b20, e20 = lanczos_block(probe_loss, SL_H, 20)
print({"lanczos": {"4_seconds": t4, "20_seconds": t20,
                   "top4_from_20": [float(x) for x in e20[-4:]],
                   "last_beta": b20[-1] if b20 else None}})
```

## Cell 8 — 汇总 JSON(贴回给我)

```python
summary = {
    "node": 784, "seed": 0, "layer": 2, "block": "QH2",
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "replay": {"total_norm": float(total_norm), "clip_coef": coef,
               "replay_G_rel_error": rel},
    "probe": {"loss": probe_loss_value, "hvp_seconds": dt_v,
              "symmetry_rel_error": sym, "q_h": q_h, "q_hk": q_hk,
              "peak_MiB": peak_v},
    "train": {"tokens": int(x_train.numel()), "hvp_seconds": dt_full,
              "peak_MiB": peak_full, "chunk_seconds": chunk_seconds,
              "chunk_rel_error": chunk_rel},
    "lanczos": {"4_seconds": t4, "20_seconds": t20,
                "top4": [float(x) for x in e20[-4:]],
                "last_beta": b20[-1] if b20 else None},
}
print(json.dumps(summary, indent=2))
Path("/kaggle/working/pilot_B_summary.json").write_text(json.dumps(summary, indent=2))
```

## 跑完后请贴回

`pilot_B_summary.json` 的内容。重点:

1. `replay_G_rel_error` 是否 ≤1e-4(重放可靠性);
2. `probe/train` 的 `hvp_seconds` 与 `peak_MiB`(确定正式节点数、Lanczos 步数、是否分块);
3. `symmetry_rel_error`(正式验收容差;1e-6 可能过严);
4. `chunk_rel_error`(8192-token 是否必须分块);
5. 20 步 Lanczos 的 `last_beta` / `top4`(是否要 30 步)。

如果某个 Cell 报 CUDA OOM:不要改代码,告诉我 OOM 出现在哪个 Cell、当时的 `peak_MiB` 和 batch shape,我会给更小 batch/分块版本。
