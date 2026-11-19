"""Three-tier structural emergence collectors for FP32 ReferenceMuon.

Tier 1 (every update): matrix scalar statistics, Q/K/V/O head energies and
pairwise role relations.  Tier 2 (dense early, then every N updates): full
row/column L2 norm vectors.  Tier 3 (landmark updates) is the existing
``MuonSnapshotCollector`` and is not reimplemented here.

The collector is read-only.  It is driven by the same ``ReferenceMuon``
observation hook as the snapshot collector and computes every reduction on the
tensor's own device; only small statistic vectors are copied to CPU.  All state
is kept outside the optimizer and checkpoint state dicts.

Output layout under ``run_dir``::

    scalar_stats/      scalar-<NNNNN>.npz     matrix scalar statistics
    head_stats/        head-<NNNNN>.npz       per-head energy / relative energy
    role_relations/    relation-<NNNNN>.npz   pairwise role relations
    row_col_norms/     rowcol-<NNNNN>.npz     row/column L2 norms
    structural_config.json                    sidecar config + hash
    structural_status.json                    final collector status
    metadata/structural_collector.json        schema, index and mapping

Every chunk file is written atomically and registered in that directory's
``index.json`` after it is durable.  ``completed_updates`` is used on resume to
drop rows beyond the restored checkpoint boundary; duplicated rows are never
kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment_io import atomic_write_json, fsync_directory

STRUCTURAL_FORMAT = "muon_structural_stats"
STRUCTURAL_VERSION = 1
CONFIG_FORMAT = "muon_structural_config"
CONFIG_VERSION = 1
COLLECTOR_VERSION = 1
SCHEMA_VERSION = 1

ROLE_ORDER = (
    "W_before", "W_after", "G", "M_prev", "M_t_candidate", "D", "U", "Delta_W",
)
SOURCE_ROLES = ("W_before", "W_after", "G", "M_prev", "M_t_candidate", "D", "U")
DERIVED_ROLE = "Delta_W"
ROLE_INDEX = {role: index for index, role in enumerate(ROLE_ORDER)}

SCALAR_METRICS = (
    "mean", "std", "rms", "fro", "max_abs",
    "positive_ratio", "negative_ratio", "zero_ratio",
    "row_norm_cv", "column_norm_cv",
)

RELATION_PAIRS = (
    ("G_vs_M_prev", "G", "M_prev"),
    ("G_vs_M_t_candidate", "G", "M_t_candidate"),
    ("G_vs_D", "G", "D"),
    ("M_prev_vs_M_t_candidate", "M_prev", "M_t_candidate"),
    ("M_t_candidate_vs_D", "M_t_candidate", "D"),
    ("M_t_candidate_vs_U", "M_t_candidate", "U"),
    ("D_vs_U", "D", "U"),
    ("Delta_W_vs_U", "Delta_W", "U"),
    ("Delta_W_vs_D", "Delta_W", "D"),
    ("W_before_vs_W_after", "W_before", "W_after"),
)

RELATION_METRICS = (
    "frobenius_inner", "cosine", "norm_ratio_A_over_B",
    "relative_frobenius_difference", "optimal_scalar_projection_residual",
)

ROW_COL_ROLES = ("W_before", "G", "M_t_candidate", "D", "U")
ROW_COL_ROLE_INDEX = {role: index for index, role in enumerate(ROW_COL_ROLES)}

AUX_FIELDS = (
    "learning_rate", "weight_decay", "learning_rate_times_weight_decay",
    "muon_momentum", "muon_nesterov", "muon_update_scale",
    "pre_clip_grad_norm", "grad_clip_norm", "clipped",
    "tokens_this_update", "processed_target_tokens", "train_nll",
)

GROUP_DIRS = {
    "scalar_stats": "scalar_stats", "head_stats": "head_stats",
    "role_relations": "role_relations", "row_col_norms": "row_col_norms",
}
GROUP_PREFIXES = {
    "scalar_stats": "scalar", "head_stats": "head",
    "role_relations": "relation", "row_col_norms": "rowcol",
}

DEFAULT_MIN_FREE_BYTES = 1 * 1024**3


class StructuralConfigError(ValueError):
    """Invalid sidecar configuration or inconsistent resume configuration."""


class StructuralIntegrityError(RuntimeError):
    """Collected data failed a completeness or checksum validation."""


class StructuralPauseRequest(RuntimeError):
    """Disk budget reached; the training loop should pause at this boundary."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierConfig:
    enabled: bool = False
    chunk_updates: int = 128


@dataclass(frozen=True)
class RowColConfig:
    enabled: bool = False
    warmup_updates: int = 128
    every_updates: int = 4
    chunk_nodes: int = 32


@dataclass(frozen=True)
class StructuralConfig:
    schema_version: int = SCHEMA_VERSION
    enabled: bool = False
    scalar_stats: TierConfig = TierConfig()
    head_stats: TierConfig = TierConfig()
    role_relations: TierConfig = TierConfig()
    row_col_norms: RowColConfig = RowColConfig()
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    max_total_bytes: int | None = None

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self)


def _config_dict(config: StructuralConfig) -> dict[str, Any]:
    return {
        "schema_version": int(config.schema_version),
        "enabled": bool(config.enabled),
        "scalar_stats": {"enabled": bool(config.scalar_stats.enabled),
                         "chunk_updates": int(config.scalar_stats.chunk_updates)},
        "head_stats": {"enabled": bool(config.head_stats.enabled),
                       "chunk_updates": int(config.head_stats.chunk_updates)},
        "role_relations": {"enabled": bool(config.role_relations.enabled),
                           "chunk_updates": int(config.role_relations.chunk_updates)},
        "row_col_norms": {"enabled": bool(config.row_col_norms.enabled),
                          "warmup_updates": int(config.row_col_norms.warmup_updates),
                          "every_updates": int(config.row_col_norms.every_updates),
                          "chunk_nodes": int(config.row_col_norms.chunk_nodes)},
        "min_free_bytes": int(config.min_free_bytes),
        "max_total_bytes": (None if config.max_total_bytes is None
                            else int(config.max_total_bytes)),
    }


def compute_config_hash(config: StructuralConfig) -> str:
    payload = {
        "structural_config": _config_dict(config),
        "collector_format": STRUCTURAL_FORMAT,
        "collector_version": COLLECTOR_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_int(value, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructuralConfigError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise StructuralConfigError(f"{field} must be >= {minimum}")
    return value


def _parse_config_dict(data: dict, *, source: str) -> StructuralConfig:
    required = {"schema_version", "enabled", "scalar_stats", "head_stats",
                "role_relations", "row_col_norms", "min_free_bytes",
                "max_total_bytes"}
    if not isinstance(data, dict):
        raise StructuralConfigError(f"{source}: configuration root must be an object")
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise StructuralConfigError(f"{source}: missing fields: {missing}")
    if unknown:
        raise StructuralConfigError(f"{source}: unknown fields: {unknown}")
    if _require_int(data["schema_version"], f"{source}.schema_version") != SCHEMA_VERSION:
        raise StructuralConfigError(f"{source}.schema_version must be {SCHEMA_VERSION}")
    if not isinstance(data["enabled"], bool):
        raise StructuralConfigError(f"{source}.enabled must be bool")

    def tier(name: str) -> TierConfig:
        value = data[name]
        expected = {"enabled", "chunk_updates"}
        if not isinstance(value, dict) or set(value) != expected:
            raise StructuralConfigError(
                f"{source}.{name} fields must be {sorted(expected)}")
        if not isinstance(value["enabled"], bool):
            raise StructuralConfigError(f"{source}.{name}.enabled must be bool")
        return TierConfig(enabled=value["enabled"], chunk_updates=_require_int(
            value["chunk_updates"], f"{source}.{name}.chunk_updates", minimum=1))

    scalar, head, relations = tier("scalar_stats"), tier("head_stats"), tier("role_relations")
    enabled_tier1 = [item for item in (scalar, head, relations) if item.enabled]
    if len({item.chunk_updates for item in enabled_tier1}) > 1:
        raise StructuralConfigError(
            f"{source}: enabled tier-1 collectors must share chunk_updates")

    rc = data["row_col_norms"]
    expected_rc = {"enabled", "warmup_updates", "every_updates", "chunk_nodes"}
    if not isinstance(rc, dict) or set(rc) != expected_rc:
        raise StructuralConfigError(
            f"{source}.row_col_norms fields must be {sorted(expected_rc)}")
    if not isinstance(rc["enabled"], bool):
        raise StructuralConfigError(f"{source}.row_col_norms.enabled must be bool")
    row_col = RowColConfig(
        enabled=rc["enabled"],
        warmup_updates=_require_int(rc["warmup_updates"],
                                    f"{source}.row_col_norms.warmup_updates", minimum=0),
        every_updates=_require_int(rc["every_updates"],
                                   f"{source}.row_col_norms.every_updates", minimum=1),
        chunk_nodes=_require_int(rc["chunk_nodes"],
                                 f"{source}.row_col_norms.chunk_nodes", minimum=1),
    )
    min_free = _require_int(data["min_free_bytes"], f"{source}.min_free_bytes", minimum=0)
    max_total = data["max_total_bytes"]
    if max_total is not None:
        max_total = _require_int(max_total, f"{source}.max_total_bytes", minimum=1)

    enabled = data["enabled"]
    if not enabled and any((scalar.enabled, head.enabled, relations.enabled, row_col.enabled)):
        raise StructuralConfigError(
            f"{source}: sub-collectors must be disabled when enabled=false")
    if enabled and not any((scalar.enabled, head.enabled, relations.enabled, row_col.enabled)):
        raise StructuralConfigError(
            f"{source}: at least one sub-collector must be enabled when enabled=true")

    return StructuralConfig(
        schema_version=SCHEMA_VERSION, enabled=enabled, scalar_stats=scalar,
        head_stats=head, role_relations=relations, row_col_norms=row_col,
        min_free_bytes=min_free, max_total_bytes=max_total)


def load_structural_config(path: str | Path) -> StructuralConfig:
    path = Path(path)
    if not path.is_file():
        raise StructuralConfigError(f"structural config not found: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralConfigError(f"cannot read structural config {path}: {exc}") from exc
    return _parse_config_dict(data, source=str(path))


def save_structural_config(run_dir: str | Path, config: StructuralConfig) -> Path:
    path = Path(run_dir) / "structural_config.json"
    atomic_write_json(path, {
        "format": CONFIG_FORMAT, "version": CONFIG_VERSION,
        "config_hash": config.config_hash, "config": _config_dict(config)})
    return path


def load_saved_structural_config(run_dir: str | Path) -> tuple[StructuralConfig, str] | None:
    path = Path(run_dir) / "structural_config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralConfigError(f"cannot read stored structural config {path}: {exc}") from exc
    if (not isinstance(payload, dict) or payload.get("format") != CONFIG_FORMAT
            or payload.get("version") != CONFIG_VERSION):
        raise StructuralConfigError(f"{path}: unsupported stored structural config")
    if "config" not in payload or "config_hash" not in payload:
        raise StructuralConfigError(f"{path}: stored structural config lacks config/config_hash")
    config = _parse_config_dict(payload["config"], source=f"{path}.config")
    if payload["config_hash"] != config.config_hash:
        raise StructuralConfigError(f"{path}: stored structural config hash does not match its content")
    return config, payload["config_hash"]


def resolve_structural_config(path: str | Path | None, *, run_dir: str | Path,
                              resume: bool) -> StructuralConfig:
    """Load, validate and persist the structural sidecar configuration."""
    if resume:
        stored = load_saved_structural_config(run_dir)
        if path is not None:
            provided = load_structural_config(path)
            if stored is None:
                raise StructuralConfigError(
                    "resume was given --structural-config but the run has no stored structural config")
            if provided.config_hash != stored[1]:
                raise StructuralConfigError(
                    f"resume structural config hash mismatch: provided={provided.config_hash} stored={stored[1]}")
            return provided
        return stored[0] if stored is not None else StructuralConfig()
    config = StructuralConfig() if path is None else load_structural_config(path)
    save_structural_config(run_dir, config)
    return config


def row_col_schedule(total_updates: int, warmup_updates: int,
                     every_updates: int) -> tuple[int, ...]:
    """Dense for updates 1..warmup, then every ``every_updates`` multiples."""
    total_updates, warmup_updates = int(total_updates), int(warmup_updates)
    every_updates = int(every_updates)
    warmup = min(warmup_updates, total_updates)
    values = list(range(1, warmup + 1))
    values.extend(u for u in range(warmup + 1, total_updates + 1)
                  if u % every_updates == 0)
    return tuple(values)


# ---------------------------------------------------------------------------
# Run / update contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralRunContext:
    run_id: str
    segment_id: int
    recipe_fingerprint: str
    data_fingerprint: str
    config_hash: str
    total_updates: int
    model_config: dict
    parameter_records: list[dict]
    muon_parameter_order: tuple[str, ...]
    optimizer_param_index: dict[str, int]
    optimizer_metadata: dict
    train_config: dict
    data_config: dict
    seeds: dict
    source: dict
    environment: dict
    device: str
    precision: dict


@dataclass(frozen=True)
class StructuralBeginContext:
    update: int
    segment_id: int
    tokens_this_update: int
    processed_target_tokens: int
    learning_rate: float
    weight_decay: float
    muon_momentum: float
    muon_nesterov: bool
    muon_update_scale: float
    pre_clip_grad_norm: float
    clipped: bool
    grad_clip_norm: float


# ---------------------------------------------------------------------------
# Matrix / head index
# ---------------------------------------------------------------------------

_LAYER_SUFFIXES = (
    ("attn.c_attn", "QKV_fused"), ("attn.c_proj", "O"), ("mlp.w1", "gate"),
    ("mlp.w2", "up"), ("mlp.c_proj", "down"),
)


def build_matrix_index(model_config: dict, parameter_records: list[dict]) -> dict:
    """Build the stable view/role index for a llama model.

    Views 0..29 are the 30 actual Muon matrices in layer-major
    (QKV_fused, O, gate, up, down) order.  Views 30..47 are the Q/K/V row
    slices of each fused QKV matrix, in layer-major Q, K, V order.  Head
    matrices are layer-major (Q, K, V, O); Q/K/V use 64-row head blocks and O
    uses 64-column head blocks.
    """
    import re

    n_layer = int(model_config["n_layer"])
    n_embd = int(model_config["n_embd"])
    n_head = int(model_config["n_head"])
    if n_embd % n_head:
        raise StructuralConfigError("n_embd must be divisible by n_head")
    head_dim = n_embd // n_head

    records = {record["name"]: record for record in parameter_records}
    parsed = {}
    for name in records:
        match = re.match(r"transformer\.h\.(\d+)\.(.+)\.weight$", name)
        if match:
            parsed[(int(match.group(1)), match.group(2))] = name

    actual: list[dict] = []
    for layer in range(n_layer):
        for suffix, kind in _LAYER_SUFFIXES:
            name = parsed.get((layer, suffix))
            if name is None:
                raise StructuralConfigError(
                    f"missing Muon matrix transformer.h.{layer}.{suffix}.weight")
            shape = [int(x) for x in records[name]["shape"]]
            if len(shape) != 2:
                raise StructuralConfigError(f"Muon matrix {name} is not 2D")
            actual.append({
                "view_id": len(actual), "name": name, "layer": layer,
                "type": kind, "derived": False, "rows": shape[0], "cols": shape[1],
                "row_start": 0, "row_end": shape[0], "core": True,
            })

    for view in actual:
        rows, cols, kind = view["rows"], view["cols"], view["type"]
        if kind == "QKV_fused" and (rows, cols) != (3 * n_embd, n_embd):
            raise StructuralConfigError(f"{view['name']} is not a fused QKV matrix")
        if kind == "O" and (rows, cols) != (n_embd, n_embd):
            raise StructuralConfigError(f"{view['name']} is not a square attention output")
        if kind in {"gate", "up"} and cols != n_embd:
            raise StructuralConfigError(f"{view['name']} has an unexpected input width")
        if kind == "down" and rows != n_embd:
            raise StructuralConfigError(f"{view['name']} has an unexpected output height")

    qkv_by_layer = {view["layer"]: view for view in actual if view["type"] == "QKV_fused"}
    derived: list[dict] = []
    for layer in range(n_layer):
        fused = qkv_by_layer[layer]
        for index, kind in enumerate(("Q", "K", "V")):
            start, end = index * n_embd, (index + 1) * n_embd
            derived.append({
                "view_id": len(actual) + len(derived),
                "name": f"{fused['name']}::{kind}", "source_name": fused["name"],
                "layer": layer, "type": kind, "derived": True,
                "rows": n_embd, "cols": n_embd,
                "row_start": start, "row_end": end, "core": False,
            })

    views = actual + derived
    head_matrices: list[dict] = []
    for layer in range(n_layer):
        for kind in ("Q", "K", "V"):
            view = next(v for v in derived if v["layer"] == layer and v["type"] == kind)
            head_matrices.append({"view_id": view["view_id"], "layer": layer,
                                  "type": kind, "axis": "rows",
                                  "blocks": n_head, "head_dim": head_dim})
        view = next(v for v in actual if v["layer"] == layer and v["type"] == "O")
        head_matrices.append({"view_id": view["view_id"], "layer": layer,
                              "type": "O", "axis": "columns",
                              "blocks": n_head, "head_dim": head_dim})

    row_offsets, col_offsets = [], []
    row_cursor = col_cursor = 0
    for view in actual:
        row_offsets.append(row_cursor)
        col_offsets.append(col_cursor)
        row_cursor += view["rows"]
        col_cursor += view["cols"]
    qkv_col_offsets, qkv_cursor = [], 0
    for layer in range(n_layer):
        for kind in ("Q", "K", "V"):
            qkv_col_offsets.append({"layer": layer, "type": kind, "offset": qkv_cursor})
            qkv_cursor += n_embd

    functional = [v["view_id"] for v in views if v["type"] in
                  {"Q", "K", "V", "O", "gate", "up", "down"}]

    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
                  "head_dim": head_dim},
        "views": views,
        "actual_view_ids": [v["view_id"] for v in actual],
        "functional_view_ids": functional,
        "head_matrices": head_matrices,
        "role_order": list(ROLE_ORDER),
        "source_roles": list(SOURCE_ROLES),
        "derived_role": DERIVED_ROLE,
        "scalar_metrics": list(SCALAR_METRICS),
        "relation_pairs": [list(pair) for pair in RELATION_PAIRS],
        "relation_metrics": list(RELATION_METRICS),
        "row_col_roles": list(ROW_COL_ROLES),
        "row_lengths": [v["rows"] for v in actual],
        "row_offsets": row_offsets,
        "col_lengths": [v["cols"] for v in actual],
        "col_offsets": col_offsets,
        "row_total": row_cursor,
        "col_total": col_cursor,
        "qkv_col_offsets": qkv_col_offsets,
        "qkv_col_total": qkv_cursor,
    }


# ---------------------------------------------------------------------------
# Device-side statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def _scalar_stats(tensor: torch.Tensor) -> torch.Tensor:
    """Return the fixed 10-element scalar metric vector for one 2D tensor."""
    numel = tensor.numel()
    inv_n = 1.0 / float(numel)
    square = tensor.mul(tensor)
    s1, s2 = tensor.sum(), square.sum()
    max_abs = tensor.abs().amax()
    positive = (tensor > 0).sum().to(tensor.dtype)
    negative = (tensor < 0).sum().to(tensor.dtype)
    zero = (tensor == 0).sum().to(tensor.dtype)
    row_norms = square.sum(dim=1).sqrt()
    col_norms = square.sum(dim=0).sqrt()
    mean = s1 * inv_n
    variance = (s2 * inv_n - mean * mean).clamp_min(0.0)
    row_mean, row_std = row_norms.mean(), row_norms.std(unbiased=False)
    col_mean, col_std = col_norms.mean(), col_norms.std(unbiased=False)
    nan = torch.full((), float("nan"), device=tensor.device, dtype=tensor.dtype)
    row_cv = torch.where(row_mean > 0, row_std / row_mean, nan)
    col_cv = torch.where(col_mean > 0, col_std / col_mean, nan)
    return torch.stack((
        mean, variance.sqrt(), (s2 * inv_n).sqrt(), s2.sqrt(), max_abs,
        positive * inv_n, negative * inv_n, zero * inv_n, row_cv, col_cv,
    ))


@torch.no_grad()
def _head_energy(tensor: torch.Tensor, axis: str, blocks: int,
                 head_dim: int) -> torch.Tensor:
    energy = tensor.mul(tensor)
    along = energy.sum(dim=1) if axis == "rows" else energy.sum(dim=0)
    return along.reshape(blocks, head_dim).sum(dim=1)


@torch.no_grad()
def _relation_metrics(a: torch.Tensor, b: torch.Tensor, norm_a: torch.Tensor,
                      norm_b: torch.Tensor) -> torch.Tensor:
    """Five relation metrics with A = pair[1], B = pair[2].

    ``relative_frobenius_difference`` is ||A-B||_F / ||B||_F and the residual is
    ||B - alpha A||_F / ||B||_F with alpha = <A,B> / ||A||_F^2.
    """
    dot = a.mul(b).sum()
    na2 = norm_a * norm_a
    valid = (norm_a > 0) & (norm_b > 0)
    nan = torch.full((), float("nan"), device=a.device, dtype=a.dtype)
    cosine = torch.where(valid, dot / (norm_a * norm_b), nan)
    ratio = torch.where(valid, norm_a / norm_b, nan)
    # Direct differences avoid catastrophic cancellation when A ~ B, which is
    # the common case for W_before vs W_after at small learning rates.
    difference = a - b
    rel_diff = torch.where(
        norm_b > 0, difference.mul(difference).sum().sqrt() / norm_b, nan)
    tiny = torch.finfo(a.dtype).tiny
    residual_vector = b - (dot / na2.clamp_min(tiny)) * a
    residual = torch.where(
        valid, residual_vector.mul(residual_vector).sum().sqrt() / norm_b, nan)
    return torch.stack((dot, cosine, ratio, rel_diff, residual))


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(Path(tmp))
        size = os.path.getsize(tmp)
        os.replace(tmp, path)
        fsync_directory(path.parent)
        return {"sha256": digest, "nbytes": int(size)}
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class _GroupStore:
    """One chunked group directory with its index and resume bookkeeping."""

    def __init__(self, run_dir: Path, group: str):
        self.group = group
        self.directory = run_dir / GROUP_DIRS[group]
        self.prefix = GROUP_PREFIXES[group]
        self.index_path = self.directory / "index.json"
        self.orphan_dir = self.directory / "orphaned"
        self.chunks: list[dict] = []
        self.last_update = 0

    def open(self, *, identity: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.index_path.is_file():
            try:
                payload = json.loads(self.index_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise StructuralIntegrityError(f"cannot read {self.index_path}: {exc}") from exc
            if (payload.get("format") != STRUCTURAL_FORMAT
                    or payload.get("version") != STRUCTURAL_VERSION):
                raise StructuralIntegrityError(f"{self.index_path}: unsupported index format")
            if payload.get("identity") != identity:
                raise StructuralIntegrityError(
                    f"{self.index_path}: run identity does not match the current run")
            self.chunks = list(payload.get("chunks", []))
            self.last_update = int(payload.get("last_update", 0))
            for entry in self.chunks:
                if entry.get("pruned"):
                    continue
                path = self.directory / entry["file"]
                if not path.is_file():
                    raise StructuralIntegrityError(f"{path}: indexed chunk is missing")
                if path.stat().st_size != int(entry["nbytes"]):
                    raise StructuralIntegrityError(f"{path}: chunk size does not match its index")
            self._adopt_unindexed(identity)
        else:
            leftovers = self._files()
            if leftovers:
                raise StructuralIntegrityError(
                    f"{self.directory}: chunk files exist without an index")
            self._write_index(identity)

    def _files(self) -> list[Path]:
        return sorted(self.directory.glob(f"{self.prefix}-*.npz"))

    def _write_index(self, identity: dict) -> None:
        atomic_write_json(self.index_path, {
            "format": STRUCTURAL_FORMAT, "version": STRUCTURAL_VERSION,
            "group": self.group, "identity": identity,
            "chunks": self.chunks, "last_update": int(self.last_update)})

    def _adopt_unindexed(self, identity: dict) -> None:
        known = {entry["file"] for entry in self.chunks}
        adopted = False
        for path in self._files():
            if path.name in known:
                continue
            try:
                with np.load(path) as archive:
                    updates = archive["updates"]
            except Exception:
                self._orphan(path)
                continue
            if int(updates[0]) != self.last_update + 1:
                self._orphan(path)
                continue
            self.chunks.append({
                "file": path.name, "first_update": int(updates[0]),
                "last_update": int(updates[-1]), "rows": int(updates.shape[0]),
                "sha256": _sha256_file(path), "nbytes": int(path.stat().st_size)})
            self.last_update = int(updates[-1])
            adopted = True
        if adopted:
            self._write_index(identity)

    def _orphan(self, path: Path) -> None:
        self.orphan_dir.mkdir(parents=True, exist_ok=True)
        target = self.orphan_dir / f"{path.stem}.{time.time_ns()}{path.suffix}"
        if path.exists():
            os.replace(path, target)

    def truncate_after(self, completed_updates: int, *, identity: dict) -> None:
        """Drop every row whose update is greater than ``completed_updates``."""
        if self.last_update <= completed_updates:
            return
        kept: list[dict] = []
        for entry in self.chunks:
            if entry.get("pruned"):
                kept.append(entry)
                continue
            path = self.directory / entry["file"]
            if int(entry["first_update"]) > completed_updates:
                self._orphan(path)
                continue
            if int(entry["last_update"]) <= completed_updates:
                kept.append(entry)
                continue
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            mask = arrays["updates"] <= completed_updates
            if not mask.any():
                self._orphan(path)
                continue
            arrays = {key: value[mask] for key, value in arrays.items()}
            result = _atomic_savez(path, arrays)
            kept.append({"file": path.name, "first_update": int(arrays["updates"][0]),
                         "last_update": int(arrays["updates"][-1]),
                         "rows": int(arrays["updates"].shape[0]), **result})
        self.chunks = kept
        self.last_update = kept[-1]["last_update"] if kept else 0
        self._write_index(identity)

    def append_chunk(self, arrays: dict[str, np.ndarray], *, identity: dict) -> dict:
        used = []
        for entry in self.chunks:
            stem = Path(entry["file"]).stem
            suffix = stem.rsplit("-", 1)[-1]
            if suffix.isdigit():
                used.append(int(suffix))
        index = max(used) + 1 if used else 0
        path = self.directory / f"{self.prefix}-{index:05d}.npz"
        result = _atomic_savez(path, arrays)
        entry = {"file": path.name, "first_update": int(arrays["updates"][0]),
                 "last_update": int(arrays["updates"][-1]),
                 "rows": int(arrays["updates"].shape[0]), **result}
        self.chunks.append(entry)
        self.last_update = entry["last_update"]
        self._write_index(identity)
        return entry

    def total_bytes(self) -> int:
        return sum(int(entry["nbytes"]) for entry in self.chunks
                   if not entry.get("pruned"))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class MuonStructuralCollector:
    """Collect tier-1 and tier-2 statistics from the Muon observation hook."""

    def __init__(self, *, run_dir: str | Path, config: StructuralConfig,
                 context: StructuralRunContext, completed_updates: int = 0):
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.context = context
        if config.config_hash != context.config_hash:
            raise StructuralConfigError("collector config hash does not match run context")
        if not context.muon_parameter_order:
            raise StructuralConfigError("muon_parameter_order is empty")
        if len(context.muon_parameter_order) != len(set(context.muon_parameter_order)):
            raise StructuralConfigError("muon_parameter_order contains duplicates")

        self._names = tuple(context.muon_parameter_order)
        self._name_set = set(self._names)
        self._expected_shapes: dict[str, tuple[int, int]] = {}
        records = {record["name"]: record for record in context.parameter_records}
        for name in self._names:
            record = records.get(name)
            if record is None or record.get("group") != "muon":
                raise StructuralConfigError(f"missing Muon parameter record for {name!r}")
            shape = tuple(int(x) for x in record["shape"])
            if len(shape) != 2:
                raise StructuralConfigError(f"Muon parameter {name!r} is not 2D")
            self._expected_shapes[name] = shape
            if name not in context.optimizer_param_index:
                raise StructuralConfigError(f"missing optimizer index for {name!r}")

        self.index = build_matrix_index(context.model_config, context.parameter_records)
        self._views = self.index["views"]
        self._view_by_id = {view["view_id"]: view for view in self._views}
        self._head_matrices = self.index["head_matrices"]
        self._n_views = len(self._views)
        self._n_heads = len(self._head_matrices)
        self._n_head_blocks = int(self.index["model"]["n_head"])
        self._n_embd = int(self.index["model"]["n_embd"])

        self._param_views: dict[str, list[int]] = {name: [] for name in self._names}
        for view in self._views:
            target = view.get("source_name", view["name"])
            if target in self._param_views:
                self._param_views[target].append(view["view_id"])
        self._param_heads: dict[str, list[int]] = {name: [] for name in self._names}
        for head_index, head in enumerate(self._head_matrices):
            view = self._view_by_id[head["view_id"]]
            target = view.get("source_name", view["name"])
            self._param_heads[target].append(head_index)
        self._qkv_layers = {
            view["name"]: int(view["layer"]) for view in self._views
            if view["type"] == "QKV_fused"}

        self.identity = {
            "run_id": context.run_id,
            "recipe_fingerprint": context.recipe_fingerprint,
            "data_fingerprint": context.data_fingerprint,
            "structural_config_hash": config.config_hash,
            "collector_version": COLLECTOR_VERSION,
        }
        self._completed = int(completed_updates)
        self._stores: dict[str, _GroupStore] = {}
        for group, enabled in (
                ("scalar_stats", config.scalar_stats.enabled),
                ("head_stats", config.head_stats.enabled),
                ("role_relations", config.role_relations.enabled),
                ("row_col_norms", config.row_col_norms.enabled)):
            if enabled:
                store = _GroupStore(self.run_dir, group)
                store.open(identity=self.identity)
                store.truncate_after(self._completed, identity=self.identity)
                self._stores[group] = store

        tier1_chunks = [tier.chunk_updates for tier in
                        (config.scalar_stats, config.head_stats, config.role_relations)
                        if tier.enabled]
        self._tier1_chunk = max(1, min(tier1_chunks)) if tier1_chunks else 0
        self._rc_chunk = int(config.row_col_norms.chunk_nodes)
        self._row_col_updates = (set(row_col_schedule(
            context.total_updates, config.row_col_norms.warmup_updates,
            config.row_col_norms.every_updates))
            if config.row_col_norms.enabled else set())

        self._reset_buffers()
        self._pending = False
        self._pause_reason: str | None = None
        self._failed = False
        self._errors: list[str] = []
        self._bytes_written = 0
        self._closed = False

        self.metadata_path = self.run_dir / "metadata" / "structural_collector.json"
        if not self.metadata_path.is_file():
            atomic_write_json(self.metadata_path, self._metadata())

    # -- public API ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._failed

    @property
    def failed(self) -> bool:
        return self._failed

    def should_collect(self, update: int) -> bool:
        update = int(update)
        return (self.enabled and not self._closed and not self._pending
                and update > self._completed)

    def precheck(self, update: int) -> None:
        """Raise a pause request before an update when the disk budget is low."""
        if not self.enabled:
            return
        if self._pause_reason is not None:
            raise StructuralPauseRequest(self._pause_reason)
        update = int(update)
        if update != 1 and update % 16 != 1:
            return
        try:
            free = shutil.disk_usage(self.run_dir).free
        except OSError:
            return
        if free < int(self.config.min_free_bytes):
            self._pause_reason = (
                f"free disk {free} bytes is below structural min_free_bytes "
                f"{self.config.min_free_bytes} at update {update}")
            raise StructuralPauseRequest(self._pause_reason)
        if (self.config.max_total_bytes is not None
                and self._bytes_written >= int(self.config.max_total_bytes)):
            self._pause_reason = (
                f"structural collector reached max_total_bytes "
                f"{self.config.max_total_bytes} at update {update}")
            raise StructuralPauseRequest(self._pause_reason)

    def begin_update(self, context: StructuralBeginContext) -> None:
        if not self.enabled or self._closed or self._pending:
            return
        if context.update <= self._completed:
            return
        self._begin = context
        self._pending = True
        self._observations: set[str] = set()
        self._seen_views: set[int] = set()
        self._seen_heads: set[int] = set()
        self._stats_acc = None
        self._rel_acc = None
        self._energy_acc = None
        self._relative_acc = None
        self._defined_acc = None
        self._rc_active = (self.config.row_col_norms.enabled
                           and context.update in self._row_col_updates)
        if self._rc_active:
            n_actual = len(self.index["actual_view_ids"])
            self._rc_rows_slots = [[None] * n_actual for _ in ROW_COL_ROLES]
            self._rc_cols_slots = [[None] * n_actual for _ in ROW_COL_ROLES]
            self._rc_qkv_slots = [[None] * len(self.index["qkv_col_offsets"])
                                  for _ in ROW_COL_ROLES]

    @torch.no_grad()
    def observe(self, *, update: int | None, parameter_name: str | None,
                W_before: torch.Tensor, G: torch.Tensor, M_prev: torch.Tensor,
                M_t_candidate: torch.Tensor, M_t_persisted: torch.Tensor,
                D: torch.Tensor, U: torch.Tensor, W_after: torch.Tensor,
                first_step: bool) -> None:
        if not self._pending or self._begin is None or self._failed:
            return
        try:
            if update is None or int(update) != self._begin.update:
                raise StructuralIntegrityError(
                    f"wrong update: expected {self._begin.update}, got {update!r}")
            name = str(parameter_name)
            if name not in self._name_set:
                raise StructuralIntegrityError(f"unknown Muon parameter: {name!r}")
            if name in self._observations:
                raise StructuralIntegrityError(f"duplicate observation: {name!r}")
            if M_t_persisted is not M_t_candidate:
                raise StructuralIntegrityError(
                    "M_t_persisted is not the aliased M_t_candidate tensor; "
                    "structural collection requires state_simulation='none'")
            expected = self._expected_shapes[name]
            tensors = {"W_before": W_before, "W_after": W_after, "G": G,
                       "M_prev": M_prev, "M_t_candidate": M_t_candidate,
                       "D": D, "U": U}
            for role, tensor in tensors.items():
                if not torch.is_tensor(tensor) or tensor.ndim != 2:
                    raise StructuralIntegrityError(f"{name}.{role} is not a 2D tensor")
                if tensor.dtype != torch.float32:
                    raise StructuralIntegrityError(f"{name}.{role} dtype is {tensor.dtype}")
                if tuple(tensor.shape) != expected:
                    raise StructuralIntegrityError(
                        f"{name}.{role} shape {tuple(tensor.shape)} != {expected}")
            tensors[DERIVED_ROLE] = W_after - W_before
            self._ensure_accumulators(W_before.device)

            for view_id in self._param_views[name]:
                view = self._view_by_id[view_id]
                start, end = view["row_start"], view["row_end"]
                pieces = {}
                for role in ROLE_ORDER:
                    value = tensors[role]
                    pieces[role] = value[start:end] if view["derived"] else value
                    self._stats_acc[view_id, ROLE_INDEX[role]] = _scalar_stats(pieces[role])
                for pair_index, (_, role_a, role_b) in enumerate(RELATION_PAIRS):
                    norm_a = self._stats_acc[view_id, ROLE_INDEX[role_a], 3]
                    norm_b = self._stats_acc[view_id, ROLE_INDEX[role_b], 3]
                    self._rel_acc[view_id, pair_index] = _relation_metrics(
                        pieces[role_a], pieces[role_b], norm_a, norm_b)
                if self._rc_active and view["core"]:
                    for role_index, role in enumerate(ROW_COL_ROLES):
                        square = pieces[role].mul(pieces[role])
                        self._rc_rows_slots[role_index][view_id] = square.sum(dim=1).sqrt()
                        self._rc_cols_slots[role_index][view_id] = square.sum(dim=0).sqrt()
                self._seen_views.add(view_id)

            if self._rc_active and name in self._qkv_layers:
                layer = self._qkv_layers[name]
                for role_index, role in enumerate(ROW_COL_ROLES):
                    value = tensors[role]
                    for index in range(3):
                        slice_ = value[index * self._n_embd:(index + 1) * self._n_embd]
                        square = slice_.mul(slice_)
                        self._rc_qkv_slots[role_index][layer * 3 + index] = \
                            square.sum(dim=0).sqrt()

            for head_index in self._param_heads[name]:
                head = self._head_matrices[head_index]
                view = self._view_by_id[head["view_id"]]
                start, end = view["row_start"], view["row_end"]
                for role in ROLE_ORDER:
                    value = tensors[role]
                    piece = value[start:end] if view["derived"] else value
                    energy = _head_energy(piece, head["axis"], head["blocks"],
                                          head["head_dim"])
                    total = energy.sum()
                    defined = bool((total > 0).item())
                    self._energy_acc[head_index, ROLE_INDEX[role]] = energy
                    self._defined_acc[head_index, ROLE_INDEX[role]] = defined
                    if defined:
                        self._relative_acc[head_index, ROLE_INDEX[role]] = energy / total
                self._seen_heads.add(head_index)

            self._observations.add(name)
        except Exception as exc:  # never propagate into optimizer.step()
            self._fail("observe_exception", f"{type(exc).__name__}: {exc}")

    def finish_update(self, *, update: int, train_nll: float) -> None:
        if not self._pending or self._begin is None:
            return
        try:
            if int(update) != self._begin.update:
                raise StructuralIntegrityError(
                    f"finish_update mismatch: expected {self._begin.update}, got {update}")
            missing = sorted(self._name_set - self._observations)
            if missing:
                raise StructuralIntegrityError(f"incomplete parameter observations: {missing}")
            if len(self._seen_views) != self._n_views:
                raise StructuralIntegrityError("incomplete matrix view coverage")
            if len(self._seen_heads) != self._n_heads:
                raise StructuralIntegrityError("incomplete head matrix coverage")

            stats = self._stats_acc.cpu().numpy().astype(np.float32)
            relations = self._rel_acc.cpu().numpy().astype(np.float32)
            valid = np.isfinite(relations[..., 1])
            energy = self._energy_acc.cpu().numpy().astype(np.float32)
            relative = self._relative_acc.cpu().numpy().astype(np.float32)
            defined = self._defined_acc.cpu().numpy().astype(bool)
            if self._tier1_chunk:
                self._buffers["updates"].append(int(update))
                self._buffers["stats"].append(stats)
                self._buffers["relations"].append(relations)
                self._buffers["valid"].append(valid)
                self._buffers["energy"].append(energy)
                self._buffers["relative"].append(relative)
                self._buffers["defined"].append(defined)
                self._buffers["aux"].append(np.asarray(self._aux_row(train_nll),
                                                       dtype=np.float32))
            if self._rc_active:
                self._append_row_col(update)
            self._clear_update()
            self._maybe_flush()
        except Exception as exc:
            self._fail("finish_exception", f"{type(exc).__name__}: {exc}")
            raise StructuralIntegrityError(
                f"structural collector failed at update {update}: {exc}") from exc

    def close(self, *, completed_updates: int) -> None:
        if self._closed:
            return
        self._completed = int(completed_updates)
        if not self._failed:
            try:
                self._trim_buffers()
                self._flush(force=True)
            except Exception as exc:
                self._fail("close_exception", f"{type(exc).__name__}: {exc}")
        status = {
            "format": STRUCTURAL_FORMAT, "version": STRUCTURAL_VERSION,
            "run_id": self.context.run_id, "config_hash": self.config.config_hash,
            "enabled": bool(self.config.enabled), "failed": bool(self._failed),
            "errors": list(self._errors[-20:]), "completed_updates": self._completed,
            "groups": {group: {"chunks": len(store.chunks),
                               "last_update": int(store.last_update),
                               "bytes": int(store.total_bytes())}
                       for group, store in self._stores.items()},
            "bytes_written": int(self._bytes_written),
        }
        try:
            atomic_write_json(self.run_dir / "structural_status.json", status)
        except Exception:
            pass
        self._closed = True

    # -- internals ----------------------------------------------------------

    def _reset_buffers(self) -> None:
        self._buffers: dict[str, list] = {
            "updates": [], "stats": [], "aux": [], "energy": [], "relative": [],
            "defined": [], "relations": [], "valid": [],
        }
        self._rc_updates: list[int] = []
        self._rc_rows: list[np.ndarray] = []
        self._rc_cols: list[np.ndarray] = []
        self._rc_qkv: list[np.ndarray] = []
        self._begin: StructuralBeginContext | None = None
        self._observations: set[str] = set()
        self._seen_views: set[int] = set()
        self._seen_heads: set[int] = set()

    def _clear_update(self) -> None:
        self._begin = None
        self._pending = False
        self._observations = set()
        self._seen_views = set()
        self._seen_heads = set()
        self._stats_acc = self._rel_acc = None
        self._energy_acc = self._relative_acc = self._defined_acc = None
        self._rc_active = False

    def _ensure_accumulators(self, device: torch.device) -> None:
        if self._stats_acc is not None:
            return
        nan = float("nan")
        self._stats_acc = torch.full(
            (self._n_views, len(ROLE_ORDER), len(SCALAR_METRICS)), nan,
            device=device, dtype=torch.float32)
        self._rel_acc = torch.full(
            (self._n_views, len(RELATION_PAIRS), len(RELATION_METRICS)), nan,
            device=device, dtype=torch.float32)
        self._energy_acc = torch.full(
            (self._n_heads, len(ROLE_ORDER), self._n_head_blocks), nan,
            device=device, dtype=torch.float32)
        self._relative_acc = torch.full(
            (self._n_heads, len(ROLE_ORDER), self._n_head_blocks), nan,
            device=device, dtype=torch.float32)
        self._defined_acc = torch.zeros(
            (self._n_heads, len(ROLE_ORDER)), device=device, dtype=torch.bool)

    def _aux_row(self, train_nll: float) -> list[float]:
        assert self._begin is not None
        b = self._begin
        return [float(b.learning_rate), float(b.weight_decay),
                float(b.learning_rate * b.weight_decay), float(b.muon_momentum),
                1.0 if b.muon_nesterov else 0.0, float(b.muon_update_scale),
                float(b.pre_clip_grad_norm), float(b.grad_clip_norm),
                1.0 if b.clipped else 0.0, float(b.tokens_this_update),
                float(b.processed_target_tokens), float(train_nll)]

    def _append_row_col(self, update: int) -> None:
        rows = []
        for role_index in range(len(ROW_COL_ROLES)):
            slots = self._rc_rows_slots[role_index]
            if any(slot is None for slot in slots):
                raise StructuralIntegrityError("row norm slots are incomplete")
            rows.append(torch.cat(slots))
        cols = []
        for role_index in range(len(ROW_COL_ROLES)):
            slots = self._rc_cols_slots[role_index]
            if any(slot is None for slot in slots):
                raise StructuralIntegrityError("column norm slots are incomplete")
            cols.append(torch.cat(slots))
        qkv = []
        for role_index in range(len(ROW_COL_ROLES)):
            slots = self._rc_qkv_slots[role_index]
            if any(slot is None for slot in slots):
                raise StructuralIntegrityError("QKV column norm slots are incomplete")
            qkv.append(torch.stack(slots))
        row_tensor = torch.stack(rows)
        col_tensor = torch.stack(cols)
        qkv_tensor = torch.stack(qkv)
        expected_qkv = (len(ROW_COL_ROLES), 3 * int(self.index["model"]["n_layer"]),
                        self._n_embd)
        if row_tensor.shape != (len(ROW_COL_ROLES), self.index["row_total"]):
            raise StructuralIntegrityError("row norm layout mismatch")
        if col_tensor.shape != (len(ROW_COL_ROLES), self.index["col_total"]):
            raise StructuralIntegrityError("column norm layout mismatch")
        if tuple(qkv_tensor.shape) != expected_qkv:
            raise StructuralIntegrityError("QKV column norm layout mismatch")
        self._rc_updates.append(int(update))
        self._rc_rows.append(row_tensor.cpu().numpy().astype(np.float32))
        self._rc_cols.append(col_tensor.cpu().numpy().astype(np.float32))
        self._rc_qkv.append(qkv_tensor.cpu().numpy().astype(np.float32))

    def _maybe_flush(self) -> None:
        if self._tier1_chunk and len(self._buffers["updates"]) >= self._tier1_chunk:
            self._flush_tier1()
        if self._rc_chunk and len(self._rc_updates) >= self._rc_chunk:
            self._flush_row_col()

    def _flush(self, *, force: bool = False) -> None:
        if force or (self._tier1_chunk
                     and len(self._buffers["updates"]) >= self._tier1_chunk):
            self._flush_tier1()
        if force or (self._rc_chunk and len(self._rc_updates) >= self._rc_chunk):
            self._flush_row_col()

    def _disk_ok(self) -> bool:
        try:
            free = shutil.disk_usage(self.run_dir).free
        except OSError:
            return True
        if free < int(self.config.min_free_bytes):
            self._pause_reason = (
                f"free disk {free} bytes is below structural min_free_bytes "
                f"{self.config.min_free_bytes}")
            return False
        return True

    def _flush_tier1(self) -> None:
        n = len(self._buffers["updates"])
        if n == 0:
            return
        if not self._disk_ok():
            return
        updates = np.asarray(self._buffers["updates"], dtype=np.float32)
        payloads = {}
        if "scalar_stats" in self._stores:
            payloads["scalar_stats"] = {
                "updates": updates, "stats": np.stack(self._buffers["stats"]),
                "aux": np.stack(self._buffers["aux"])}
        if "head_stats" in self._stores:
            payloads["head_stats"] = {
                "updates": updates, "energy": np.stack(self._buffers["energy"]),
                "relative": np.stack(self._buffers["relative"]),
                "defined": np.stack(self._buffers["defined"])}
        if "role_relations" in self._stores:
            payloads["role_relations"] = {
                "updates": updates, "relations": np.stack(self._buffers["relations"]),
                "valid": np.stack(self._buffers["valid"])}
        for group, arrays in payloads.items():
            entry = self._stores[group].append_chunk(arrays, identity=self.identity)
            self._bytes_written += int(entry["nbytes"])
        for key in ("updates", "stats", "aux", "energy", "relative", "defined",
                    "relations", "valid"):
            self._buffers[key].clear()

    def _flush_row_col(self) -> None:
        n = len(self._rc_updates)
        if n == 0 or "row_col_norms" not in self._stores:
            return
        if not self._disk_ok():
            return
        arrays = {"updates": np.asarray(self._rc_updates, dtype=np.float32),
                  "row_norms": np.stack(self._rc_rows),
                  "col_norms": np.stack(self._rc_cols),
                  "qkv_col_norms": np.stack(self._rc_qkv)}
        entry = self._stores["row_col_norms"].append_chunk(arrays, identity=self.identity)
        self._bytes_written += int(entry["nbytes"])
        self._rc_updates.clear()
        self._rc_rows.clear()
        self._rc_cols.clear()
        self._rc_qkv.clear()

    def _trim_buffers(self) -> None:
        keep = [index for index, update in enumerate(self._buffers["updates"])
                if update <= self._completed]
        if len(keep) != len(self._buffers["updates"]):
            for key in self._buffers:
                self._buffers[key] = [self._buffers[key][i] for i in keep]
        keep_rc = [index for index, update in enumerate(self._rc_updates)
                   if update <= self._completed]
        if len(keep_rc) != len(self._rc_updates):
            self._rc_updates = [self._rc_updates[i] for i in keep_rc]
            self._rc_rows = [self._rc_rows[i] for i in keep_rc]
            self._rc_cols = [self._rc_cols[i] for i in keep_rc]
            self._rc_qkv = [self._rc_qkv[i] for i in keep_rc]

    def _metadata(self) -> dict:
        return {
            "format": STRUCTURAL_FORMAT, "version": STRUCTURAL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "identity": self.identity,
            "config": {"config_hash": self.config.config_hash,
                       "config": _config_dict(self.config)},
            "seeds": self.context.seeds,
            "model": self.index["model"],
            "muon_parameter_order": list(self._names),
            "optimizer_param_index": self.context.optimizer_param_index,
            "index": self.index,
            "dtypes": {"stats": "float32", "aux": "float32", "energy": "float32",
                       "relative": "float32", "relations": "float32",
                       "row_norms": "float32", "col_norms": "float32",
                       "qkv_col_norms": "float32"},
            "layout": {
                "scalar_stats": {
                    "keys": ["updates", "stats", "aux"],
                    "stats_shape": [self._n_views, len(ROLE_ORDER),
                                    len(SCALAR_METRICS)],
                    "aux_fields": list(AUX_FIELDS)},
                "head_stats": {
                    "keys": ["updates", "energy", "relative", "defined"],
                    "energy_shape": [self._n_heads, len(ROLE_ORDER),
                                     self._n_head_blocks],
                    "undefined_relative": "NaN with defined=false"},
                "role_relations": {
                    "keys": ["updates", "relations", "valid"],
                    "relations_shape": [self._n_views, len(RELATION_PAIRS),
                                        len(RELATION_METRICS)]},
                "row_col_norms": {
                    "keys": ["updates", "row_norms", "col_norms", "qkv_col_norms"],
                    "row_norms_shape": [len(ROW_COL_ROLES), self.index["row_total"]],
                    "col_norms_shape": [len(ROW_COL_ROLES), self.index["col_total"]],
                    "qkv_col_norms_shape": [len(ROW_COL_ROLES),
                                            len(self.index["qkv_col_offsets"]),
                                            self._n_embd]},
            },
        }

    def _fail(self, code: str, message: str) -> None:
        self._failed = True
        self._errors.append(f"{code}: {message}")
        self._clear_update()
