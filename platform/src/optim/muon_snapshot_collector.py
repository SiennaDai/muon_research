"""Synchronous, read-only research snapshots for FP32 ReferenceMuon.

The collector is deliberately outside optimizer state and checkpoint state.  It
receives the eight role tensors produced by ``ReferenceMuon.step`` before the
in-place parameter write-back, clones them to CPU FP32, validates completeness,
and writes one atomic ``.pt`` payload plus one atomic manifest per landmark
update.  The manifest is the commit marker.

This module also owns the sidecar snapshot configuration so collector settings
never enter the training recipe or its scientific fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from experiment_io import (atomic_torch_save_with_digest, atomic_write_json,
                           sha256_file, sha256_tensor)


SNAPSHOT_FORMAT = "muon_snapshot_manifest"
SNAPSHOT_VERSION = 1
TENSOR_FORMAT = "muon_snapshot_tensors"
CONFIG_FORMAT = "muon_snapshot_config"
CONFIG_VERSION = 1
COLLECTOR_VERSION = 1

ROLE_ORDER = (
    "W_before",
    "W_after",
    "G",
    "M_prev",
    "M_t_candidate",
    "M_t_persisted",
    "D",
    "U",
)
LOGICAL_ROLE_COUNT = 7
ROLE_SLOTS_PER_PARAMETER = len(ROLE_ORDER)  # 8: M_t has two roles

DISABLED_MIN_FREE_BYTES = 5 * 1024**3


class SnapshotConfigError(ValueError):
    """Invalid sidecar configuration or inconsistent resume configuration."""


class SnapshotIntegrityError(RuntimeError):
    """A snapshot failed completeness or checksum validation."""


@dataclass(frozen=True)
class SnapshotConfig:
    schema_version: int = 1
    enabled: bool = False
    updates: tuple[int, ...] = ()
    on_error: str = "disable"
    max_total_bytes: int | None = None
    min_free_bytes: int = DISABLED_MIN_FREE_BYTES

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self)

    @classmethod
    def disabled(cls) -> "SnapshotConfig":
        return cls()


@dataclass(frozen=True)
class SnapshotRunContext:
    run_id: str
    segment_id: int
    recipe_fingerprint: str
    data_fingerprint: str
    snapshot_config_hash: str
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
class SnapshotBeginContext:
    update: int
    segment_id: int
    tokens_this_update: int
    processed_target_tokens: int
    learning_rate: float
    pre_clip_grad_norm: float
    clipped: bool
    grad_clip_norm: float
    window_ids: tuple[int, ...]
    micro_batch_window_ids: tuple[tuple[int, ...], ...]
    sampler_epoch_before: int
    sampler_offset_before: int
    sampler_epoch_after: int
    sampler_offset_after: int


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _config_dict(config: SnapshotConfig) -> dict[str, Any]:
    return {
        "schema_version": int(config.schema_version),
        "enabled": bool(config.enabled),
        "updates": [int(x) for x in config.updates],
        "on_error": str(config.on_error),
        "max_total_bytes": None if config.max_total_bytes is None else int(config.max_total_bytes),
        "min_free_bytes": int(config.min_free_bytes),
    }


def compute_config_hash(config: SnapshotConfig) -> str:
    payload = {
        "snapshot_config": _config_dict(config),
        "collector_format": "muon_snapshot",
        "collector_version": COLLECTOR_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_int(value, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotConfigError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise SnapshotConfigError(f"{field} must be >= {minimum}")
    return value


def _parse_config_dict(data: dict, *, total_updates: int | None, source: str) -> SnapshotConfig:
    required = {"schema_version", "enabled", "updates", "on_error", "max_total_bytes", "min_free_bytes"}
    if not isinstance(data, dict):
        raise SnapshotConfigError(f"{source}: configuration root must be an object")
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise SnapshotConfigError(f"{source}: missing fields: {missing}")
    if unknown:
        raise SnapshotConfigError(f"{source}: unknown fields: {unknown}")

    schema_version = _require_int(data["schema_version"], f"{source}.schema_version")
    if schema_version != 1:
        raise SnapshotConfigError(f"{source}.schema_version must be 1")

    enabled = data["enabled"]
    if not isinstance(enabled, bool):
        raise SnapshotConfigError(f"{source}.enabled must be bool")

    raw_updates = data["updates"]
    if not isinstance(raw_updates, list):
        raise SnapshotConfigError(f"{source}.updates must be a list of integers")
    updates: list[int] = []
    for index, value in enumerate(raw_updates):
        update = _require_int(value, f"{source}.updates[{index}]", minimum=1)
        if total_updates is not None and update > total_updates:
            raise SnapshotConfigError(
                f"{source}.updates[{index}]={update} exceeds training total_updates={total_updates}")
        if updates and update <= updates[-1]:
            raise SnapshotConfigError(f"{source}.updates must be strictly increasing")
        updates.append(update)

    on_error = data["on_error"]
    if on_error != "disable":
        raise SnapshotConfigError(f"{source}.on_error must be 'disable' in v1")

    max_total = data["max_total_bytes"]
    if max_total is not None:
        max_total = _require_int(max_total, f"{source}.max_total_bytes", minimum=1)
    min_free = _require_int(data["min_free_bytes"], f"{source}.min_free_bytes", minimum=0)

    if enabled and not updates:
        raise SnapshotConfigError(f"{source}.updates must be non-empty when enabled=true")
    if not enabled and updates:
        raise SnapshotConfigError(f"{source}.updates must be empty when enabled=false")

    return SnapshotConfig(
        schema_version=schema_version,
        enabled=enabled,
        updates=tuple(updates),
        on_error=on_error,
        max_total_bytes=max_total,
        min_free_bytes=min_free,
    )


def load_snapshot_config(path: str | Path, *, total_updates: int) -> SnapshotConfig:
    path = Path(path)
    if not path.is_file():
        raise SnapshotConfigError(f"snapshot config not found: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotConfigError(f"cannot read snapshot config {path}: {exc}") from exc
    return _parse_config_dict(data, total_updates=total_updates, source=str(path))


def save_snapshot_config(run_dir: str | Path, config: SnapshotConfig) -> Path:
    path = Path(run_dir) / "snapshot_config.json"
    payload = {
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "config_hash": config.config_hash,
        "config": _config_dict(config),
    }
    atomic_write_json(path, payload)
    return path


def load_saved_snapshot_config(run_dir: str | Path, *, total_updates: int | None = None) -> tuple[SnapshotConfig, str] | None:
    path = Path(run_dir) / "snapshot_config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotConfigError(f"cannot read stored snapshot config {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CONFIG_FORMAT or payload.get("version") != CONFIG_VERSION:
        raise SnapshotConfigError(f"{path}: unsupported stored snapshot config")
    if "config" not in payload or "config_hash" not in payload:
        raise SnapshotConfigError(f"{path}: stored snapshot config lacks config/config_hash")
    config = _parse_config_dict(payload["config"], total_updates=total_updates, source=f"{path}.config")
    if payload["config_hash"] != config.config_hash:
        raise SnapshotConfigError(f"{path}: stored snapshot config hash does not match its content")
    return config, payload["config_hash"]


def resolve_snapshot_config(path: str | Path | None, *, run_dir: str | Path,
                            total_updates: int, resume: bool) -> SnapshotConfig:
    """Load, validate and persist the sidecar configuration.

    New runs default to disabled when no path is supplied.  Resumes inherit the
    stored run configuration; an explicitly supplied path must hash-match it.
    Invalid configuration fails before training starts.
    """
    if resume:
        stored = load_saved_snapshot_config(run_dir, total_updates=total_updates)
        if path is not None:
            provided = load_snapshot_config(path, total_updates=total_updates)
            if stored is None:
                raise SnapshotConfigError(
                    "resume was given --snapshot-config but the run has no stored snapshot config")
            if provided.config_hash != stored[1]:
                raise SnapshotConfigError(
                    f"resume snapshot config hash mismatch: provided={provided.config_hash} stored={stored[1]}")
            return provided
        if stored is not None:
            return stored[0]
        return SnapshotConfig.disabled()

    config = SnapshotConfig.disabled() if path is None else load_snapshot_config(path, total_updates=total_updates)
    save_snapshot_config(run_dir, config)
    return config


# ---------------------------------------------------------------------------
# Small JSON helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SnapshotIntegrityError(f"{field} is not finite: {number!r}")
    return number


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class MuonSnapshotCollector:
    """Collect the eight Muon role tensors at configured absolute updates."""

    def __init__(self, *, run_dir: str | Path, config: SnapshotConfig, context: SnapshotRunContext):
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.context = context
        self.snapshots_dir = self.run_dir / "snapshots"
        self.errors_path = self.run_dir / "snapshot_errors.jsonl"
        self.status_path = self.run_dir / "snapshot_status.json"

        if config.config_hash != context.snapshot_config_hash:
            raise SnapshotConfigError("collector config hash does not match run context")
        if len(context.muon_parameter_order) != len(set(context.muon_parameter_order)):
            raise SnapshotConfigError("muon_parameter_order contains duplicates")
        if not context.muon_parameter_order:
            raise SnapshotConfigError("muon_parameter_order is empty")

        records_by_name = {str(record["name"]): record for record in context.parameter_records}
        self._expected_names = tuple(context.muon_parameter_order)
        self._expected_name_set = set(self._expected_names)
        self._name_to_index = {name: index for index, name in enumerate(self._expected_names)}
        self._expected_shapes: dict[str, tuple[int, ...]] = {}
        for name in self._expected_names:
            record = records_by_name.get(name)
            if record is None:
                raise SnapshotConfigError(f"Muon parameter {name!r} has no parameter record")
            if record.get("group") != "muon":
                raise SnapshotConfigError(f"parameter {name!r} is not in the muon group")
            shape = tuple(int(x) for x in record["shape"])
            if len(shape) != 2:
                raise SnapshotConfigError(f"Muon parameter {name!r} is not 2D: {shape}")
            self._expected_shapes[name] = shape
        missing_indices = self._expected_name_set - set(context.optimizer_param_index)
        if missing_indices:
            raise SnapshotConfigError(f"missing optimizer_param_index for: {sorted(missing_indices)}")

        self._config_updates = set(config.updates)
        self._committed: dict[int, dict] = {}
        self._conflicted: dict[int, str] = {}
        self._missed: dict[int, str] = {}
        self._notes: list[str] = []
        self._disabled = False
        self._disabled_reason: str | None = None
        self._closed = False
        self._collecting = False
        self._begin: SnapshotBeginContext | None = None
        self._buffer: dict[str, dict] | None = None
        self._aliases: list[dict] = []
        self._tensor_blobs: dict[str, torch.Tensor] = {}
        self._tensor_descriptors: dict[str, dict] = {}
        self._tensor_owner: dict[str, str] = {}
        self._next_tensor_id = 0
        self._started_at: float | None = None
        self._clone_seconds = 0.0
        self._checksum_seconds = 0.0
        self._write_seconds = 0.0
        self._error_count = 0

        if config.enabled or self.snapshots_dir.exists():
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            (self.snapshots_dir / "orphaned").mkdir(parents=True, exist_ok=True)
            self._cleanup_temp_files()
            self._scan_existing()

    # -- public API ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def disabled(self) -> bool:
        return self._disabled

    def committed_updates(self) -> tuple[int, ...]:
        return tuple(sorted(self._committed))

    def conflicted_updates(self) -> tuple[int, ...]:
        return tuple(sorted(self._conflicted))

    def missed_updates(self) -> tuple[int, ...]:
        return tuple(sorted(self._missed))

    def should_collect(self, update: int) -> bool:
        update = int(update)
        return (
            self.config.enabled
            and not self._disabled
            and not self._closed
            and not self._collecting
            and update in self._config_updates
            and update not in self._committed
            and update not in self._conflicted
        )

    def scan_existing(self, *, completed_updates: int) -> None:
        """Record landmarks already lost before this resume point.

        Existing manifests are scanned in ``__init__``.  This method only
        classifies configured landmarks at or before the resumed checkpoint.
        It never fabricates a snapshot and never changes training state.
        """
        completed_updates = int(completed_updates)
        archived = self._archived_updates()
        for update in sorted(self._config_updates):
            if update in archived:
                continue
            if update <= completed_updates and update not in self._committed and update not in self._conflicted:
                self._missed[update] = "landmark at or before resumed completed_updates"
                self._log_error(
                    "missed_before_resume",
                    update=update,
                    phase="scan",
                    message=f"landmark update {update} <= completed_updates {completed_updates} has no committed snapshot",
                )

    def _archived_updates(self) -> set[int]:
        """Landmarks whose snapshot files were archived and pruned locally."""
        path = self.snapshots_dir / "archived.json"
        if not path.is_file():
            return set()
        try:
            payload = json.loads(path.read_text())
            return {int(value) for value in payload.get("updates", [])}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return set()

    def begin_update(self, context: SnapshotBeginContext) -> None:
        if self._closed or self._disabled:
            return
        if self._collecting:
            self._fail("duplicate_begin_update", update=context.update, phase="begin",
                       message="begin_update called while another update is collecting")
            return
        if not self.should_collect(context.update):
            self._clear_buffer()
            return
        try:
            self._validate_begin_context(context)
        except Exception as exc:
            self._fail("invalid_begin_context", update=context.update, phase="begin", exc=exc)
            return
        self._clear_buffer()
        self._buffer = {}
        self._begin = context
        self._collecting = True
        self._started_at = time.monotonic()
        self._clone_seconds = 0.0
        self._checksum_seconds = 0.0
        self._write_seconds = 0.0

    @torch.no_grad()
    def observe(self, *, update: int | None, parameter_name: str | None,
                W_before: torch.Tensor, G: torch.Tensor, M_prev: torch.Tensor,
                M_t_candidate: torch.Tensor, M_t_persisted: torch.Tensor,
                D: torch.Tensor, U: torch.Tensor, W_after: torch.Tensor,
                first_step: bool) -> None:
        if self._disabled or self._closed or not self._collecting or self._begin is None:
            return
        name = parameter_name
        try:
            if update is None or int(update) != self._begin.update:
                raise SnapshotIntegrityError(f"wrong update: expected {self._begin.update}, got {update!r}")
            if name is None:
                raise SnapshotIntegrityError("parameter_name is None")
            name = str(name)
            if name not in self._expected_name_set:
                raise SnapshotIntegrityError(f"unknown Muon parameter: {name!r}")
            if self._buffer is not None and name in self._buffer:
                raise SnapshotIntegrityError(f"duplicate parameter observation: {name!r}")
            if not isinstance(first_step, bool):
                raise SnapshotIntegrityError("first_step must be bool")

            expected_shape = self._expected_shapes[name]
            role_tensors = (
                ("W_before", W_before), ("W_after", W_after), ("G", G), ("M_prev", M_prev),
                ("M_t_candidate", M_t_candidate), ("M_t_persisted", M_t_persisted),
                ("D", D), ("U", U),
            )
            for role, tensor in role_tensors:
                if not torch.is_tensor(tensor):
                    raise SnapshotIntegrityError(f"{name}.{role} is not a torch.Tensor")
                if tensor.ndim != 2:
                    raise SnapshotIntegrityError(f"{name}.{role} is not 2D: {tuple(tensor.shape)}")
                if tensor.dtype != torch.float32:
                    raise SnapshotIntegrityError(f"{name}.{role} dtype is {tensor.dtype}, expected float32")
                if tuple(tensor.shape) != expected_shape:
                    raise SnapshotIntegrityError(
                        f"{name}.{role} shape {tuple(tensor.shape)} != expected {expected_shape}")

            role_records: dict[str, dict] = {}
            processed: list[tuple[str, torch.Tensor]] = []
            for role, tensor in role_tensors:
                canonical_role = None
                for prior_role, prior_tensor in processed:
                    if tensor is prior_tensor:
                        canonical_role = self._resolve_alias_role(role_records, prior_role)
                        break
                if canonical_role is not None:
                    canonical = role_records[canonical_role]
                    role_records[role] = {"alias_of": canonical_role}
                    self._aliases.append({
                        "parameter_name": name,
                        "alias_role": role,
                        "canonical_role": canonical_role,
                        "tensor_id": canonical["tensor_id"],
                        "reason": "same_tensor_object",
                    })
                else:
                    tensor_id = f"t{self._next_tensor_id:04d}"
                    self._next_tensor_id += 1
                    clone_start = time.monotonic()
                    cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
                    self._clone_seconds += time.monotonic() - clone_start
                    checksum_start = time.monotonic()
                    digest = sha256_tensor(cpu)
                    self._checksum_seconds += time.monotonic() - checksum_start
                    descriptor = {
                        "tensor_id": tensor_id,
                        "shape": [int(x) for x in cpu.shape],
                        "dtype": "float32",
                        "numel": int(cpu.numel()),
                        "nbytes": int(cpu.numel() * 4),
                        "sha256": digest,
                    }
                    self._tensor_blobs[tensor_id] = cpu
                    self._tensor_descriptors[tensor_id] = descriptor
                    self._tensor_owner[tensor_id] = name
                    role_records[role] = descriptor
                processed.append((role, tensor))

            shape = expected_shape
            numel = 1
            for dim in shape:
                numel *= int(dim)
            entry = {
                "parameter_index": int(self._name_to_index[name]),
                "optimizer_param_index": int(self.context.optimizer_param_index[name]),
                "shape": [int(x) for x in shape],
                "dtype": "float32",
                "numel": int(numel),
                "first_step": bool(first_step),
                "roles": role_records,
            }
            assert self._buffer is not None
            self._buffer[name] = entry
        except Exception as exc:  # never propagate into optimizer.step()
            self._fail("observe_exception", update=self._begin.update if self._begin else None,
                       phase="observe", parameter_name=name, exc=exc)

    def finish_update(self, *, update: int, train_nll: float) -> None:
        if self._closed or not self._collecting or self._begin is None or self._buffer is None:
            return
        try:
            if int(update) != self._begin.update:
                raise SnapshotIntegrityError(f"finish_update mismatch: expected {self._begin.update}, got {update}")
            nll = _finite_float(train_nll, "train_nll")
            integrity = self._validate_completeness()
            payload, manifest = self._build_payload_and_manifest(nll=nll, integrity=integrity)
            self._precheck_disk(payload_nbytes=manifest["tensors_file"]["payload_nbytes"])
            self._resolve_existing_targets(self._begin.update)
            write_start = time.monotonic()
            result = atomic_torch_save_with_digest(payload, self.snapshots_dir / f"update_{self._begin.update:06d}.pt")
            final_pt = Path(result["path"])
            if sha256_file(final_pt) != result["sha256"]:
                raise SnapshotIntegrityError("written .pt digest changed during os.replace")
            manifest["tensors_file"] = {
                "path": final_pt.name,
                "sha256": result["sha256"],
                "nbytes": int(result["nbytes"]),
                "payload_nbytes": int(manifest["tensors_file"]["payload_nbytes"]),
                "format": TENSOR_FORMAT,
                "version": SNAPSHOT_VERSION,
            }
            manifest["integrity"]["tensors_file_sha256_ok"] = True
            self._write_seconds = time.monotonic() - write_start
            manifest["collector"].update({
                "clone_seconds": float(self._clone_seconds),
                "checksum_seconds": float(self._checksum_seconds),
                "write_seconds": float(self._write_seconds),
                "total_seconds": float(time.monotonic() - self._started_at) if self._started_at else None,
                "snapshot_bytes": int(result["nbytes"]),
            })
            manifest_path = self.snapshots_dir / f"update_{self._begin.update:06d}.manifest.json"
            atomic_write_json(manifest_path, manifest)
            self._committed[self._begin.update] = {
                "pt_path": str(final_pt),
                "manifest_path": str(manifest_path),
                "pt_nbytes": int(result["nbytes"]),
                "manifest_nbytes": int(os.path.getsize(manifest_path)),
            }
            self._clear_buffer()
        except _SkipSnapshot:
            # _skip already logged the reason and cleared the buffer.
            return
        except Exception as exc:  # never propagate into the training loop
            self._fail("finish_exception", update=self._begin.update if self._begin else None,
                       phase="finish", exc=exc)

    def close(self) -> None:
        if self._closed:
            return
        if self._collecting and self._begin is not None:
            collected = sorted(self._buffer) if self._buffer is not None else []
            missing = sorted(self._expected_name_set - set(collected))
            self._log_error(
                "incomplete_update_at_close",
                update=self._begin.update,
                phase="close",
                message="collector closed with an incomplete update",
                details={"collected_parameters": collected, "missing_parameters": missing},
            )
        self._clear_buffer()
        self._closed = True
        status = {
            "format": "muon_snapshot_status",
            "version": 1,
            "run_id": self.context.run_id,
            "config_hash": self.config.config_hash,
            "enabled": bool(self.config.enabled),
            "committed_updates": list(self.committed_updates()),
            "archived_updates": sorted(self._archived_updates()),
            "conflicted_updates": list(self.conflicted_updates()),
            "missed_updates": list(self.missed_updates()),
            "disabled": bool(self._disabled),
            "disabled_reason": self._disabled_reason,
            "error_count": int(self._error_count),
            "notes": list(self._notes),
        }
        try:
            atomic_write_json(self.status_path, status)
        except Exception:
            # Status is advisory; closing must never raise.
            pass

    # -- completeness and schema -------------------------------------------

    def _validate_begin_context(self, context: SnapshotBeginContext) -> None:
        if isinstance(context.update, bool) or not isinstance(context.update, int) or context.update <= 0:
            raise SnapshotConfigError("begin_update.update must be a positive integer")
        if context.segment_id != self.context.segment_id:
            raise SnapshotConfigError("begin_update.segment_id does not match run context")
        for field in ("tokens_this_update", "processed_target_tokens"):
            value = getattr(context, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SnapshotConfigError(f"begin_update.{field} must be a nonnegative integer")
        if context.tokens_this_update <= 0:
            raise SnapshotConfigError("begin_update.tokens_this_update must be positive")
        _finite_float(context.learning_rate, "learning_rate")
        _finite_float(context.pre_clip_grad_norm, "pre_clip_grad_norm")
        _finite_float(context.grad_clip_norm, "grad_clip_norm")
        if not isinstance(context.clipped, bool):
            raise SnapshotConfigError("begin_update.clipped must be bool")
        if not context.window_ids:
            raise SnapshotConfigError("begin_update.window_ids must be non-empty")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in context.window_ids):
            raise SnapshotConfigError("begin_update.window_ids must contain nonnegative integers")
        flattened = [x for group in context.micro_batch_window_ids for x in group]
        if flattened != list(context.window_ids):
            raise SnapshotConfigError("micro_batch_window_ids do not match window_ids")
        for field in ("sampler_epoch_before", "sampler_offset_before", "sampler_epoch_after", "sampler_offset_after"):
            value = getattr(context, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SnapshotConfigError(f"begin_update.{field} must be a nonnegative integer")

    def _resolve_alias_role(self, role_records: dict[str, dict], role: str) -> str:
        seen = set()
        current = role
        while "alias_of" in role_records.get(current, {}):
            if current in seen:
                raise SnapshotIntegrityError("cyclic alias")
            seen.add(current)
            current = role_records[current]["alias_of"]
        if current not in ROLE_ORDER:
            raise SnapshotIntegrityError(f"invalid canonical role: {current!r}")
        return current

    def _validate_completeness(self) -> dict:
        assert self._buffer is not None
        observed_names = set(self._buffer)
        if observed_names != self._expected_name_set:
            raise SnapshotIntegrityError(_incomplete_message(self._expected_name_set, observed_names))

        referenced: set[str] = set()
        alias_count = 0
        role_slots = 0
        tensor_owner_check: dict[str, str] = {}
        for name, entry in self._buffer.items():
            if entry.get("shape") != [int(x) for x in self._expected_shapes[name]]:
                raise SnapshotIntegrityError(f"{name}: parameter shape mismatch")
            if entry.get("dtype") != "float32":
                raise SnapshotIntegrityError(f"{name}: parameter dtype mismatch")
            if entry.get("parameter_index") != self._name_to_index[name]:
                raise SnapshotIntegrityError(f"{name}: parameter_index mismatch")
            if entry.get("optimizer_param_index") != int(self.context.optimizer_param_index[name]):
                raise SnapshotIntegrityError(f"{name}: optimizer_param_index mismatch")
            roles = entry.get("roles", {})
            if set(roles) != set(ROLE_ORDER):
                raise SnapshotIntegrityError(f"{name}: incomplete roles {sorted(roles)}")
            for role in ROLE_ORDER:
                descriptor = roles[role]
                role_slots += 1
                if "alias_of" in descriptor:
                    canonical_role = self._resolve_alias_role(roles, role)
                    canonical = roles[canonical_role]
                    if "alias_of" in canonical:
                        raise SnapshotIntegrityError(f"{name}.{role}: alias chain is not allowed")
                    if canonical.get("shape") != entry["shape"] or canonical.get("dtype") != "float32":
                        raise SnapshotIntegrityError(f"{name}.{role}: alias target contract mismatch")
                    tensor_id = canonical["tensor_id"]
                    alias_count += 1
                else:
                    tensor_id = descriptor.get("tensor_id")
                    if tensor_id not in self._tensor_blobs:
                        raise SnapshotIntegrityError(f"{name}.{role}: unknown tensor_id {tensor_id!r}")
                    if descriptor.get("shape") != entry["shape"]:
                        raise SnapshotIntegrityError(f"{name}.{role}: descriptor shape mismatch")
                    if descriptor.get("dtype") != "float32":
                        raise SnapshotIntegrityError(f"{name}.{role}: descriptor dtype mismatch")
                    if descriptor.get("nbytes") != int(descriptor.get("numel", -1)) * 4:
                        raise SnapshotIntegrityError(f"{name}.{role}: descriptor nbytes mismatch")
                    blob = self._tensor_blobs[tensor_id]
                    if list(blob.shape) != descriptor["shape"] or blob.dtype != torch.float32:
                        raise SnapshotIntegrityError(f"{name}.{role}: stored tensor contract mismatch")
                    checksum_start = time.monotonic()
                    actual = sha256_tensor(blob)
                    self._checksum_seconds += time.monotonic() - checksum_start
                    if actual != descriptor["sha256"]:
                        raise SnapshotIntegrityError(f"{name}.{role}: tensor checksum mismatch")
                referenced.add(tensor_id)
                owner = self._tensor_owner.get(tensor_id)
                if owner is not None:
                    previous = tensor_owner_check.setdefault(tensor_id, owner)
                    if previous != owner:
                        raise SnapshotIntegrityError(f"tensor {tensor_id} is shared across parameters")

        if referenced != set(self._tensor_blobs):
            missing = sorted(set(self._tensor_blobs) - referenced)
            raise SnapshotIntegrityError(f"orphan stored tensors: {missing}")
        if role_slots != len(self._expected_names) * ROLE_SLOTS_PER_PARAMETER:
            raise SnapshotIntegrityError("role slot count mismatch")

        return {
            "expected_parameter_count": len(self._expected_names),
            "observed_parameter_count": len(observed_names),
            "expected_role_slots_per_parameter": ROLE_SLOTS_PER_PARAMETER,
            "expected_role_slots": len(self._expected_names) * ROLE_SLOTS_PER_PARAMETER,
            "observed_role_slots": role_slots,
            "unique_tensor_count": len(self._tensor_blobs),
            "alias_count": alias_count,
            "all_parameters_present": True,
            "all_roles_present": True,
            "all_shapes_match": True,
            "all_dtypes_float32": True,
            "per_tensor_checksums_ok": True,
            "tensors_file_sha256_ok": False,
            "no_orphan_tensors": True,
            "byte_order": "little",
        }

    def _build_payload_and_manifest(self, *, nll: float, integrity: dict) -> tuple[dict, dict]:
        assert self._begin is not None and self._buffer is not None
        update = self._begin.update
        snapshot_id = f"{self.context.run_id}:{update:06d}"
        payload = {
            "format": TENSOR_FORMAT,
            "version": SNAPSHOT_VERSION,
            "snapshot_id": snapshot_id,
            "run_id": self.context.run_id,
            "update": update,
            "segment_id": self._begin.segment_id,
            "tensors": self._tensor_blobs,
        }
        payload_nbytes = sum(int(tensor.numel()) * 4 for tensor in self._tensor_blobs.values())
        source = self.context.source or {}
        diff = source.get("diff")
        source_manifest = {
            "remote": source.get("remote"),
            "commit": source.get("upstream_commit"),
            "dirty": source.get("dirty"),
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if isinstance(diff, str) else None,
            "provenance_status": source.get("provenance_status", "unknown"),
            "safe_directory_used": bool(source.get("safe_directory_used", False)),
            "provenance_error": (str(source.get("provenance_error"))[:500]
                                 if source.get("provenance_error") is not None else None),
        }
        environment_manifest = dict(_json_safe(self.context.environment))
        environment_manifest["device"] = str(self.context.device)
        environment_manifest["precision"] = _json_safe(self.context.precision)
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "version": SNAPSHOT_VERSION,
            "snapshot_id": snapshot_id,
            "status": "committed",
            "run_id": self.context.run_id,
            "update": update,
            "segment_id": self._begin.segment_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "identities": {
                "recipe_fingerprint": self.context.recipe_fingerprint,
                "data_fingerprint": self.context.data_fingerprint,
                "snapshot_config_hash": self.config.config_hash,
                "collector_version": COLLECTOR_VERSION,
            },
            "source": source_manifest,
            "environment": environment_manifest,
            "seeds": _json_safe(self.context.seeds),
            "model": {
                "config": _json_safe(self.context.model_config),
                "parameter_records": _json_safe(self.context.parameter_records),
                "muon_parameter_order": list(self._expected_names),
                "muon_parameter_count": len(self._expected_names),
                "full_parameter_records_sha256": _sha256_json(self.context.parameter_records),
            },
            "optimizer": _json_safe(self.context.optimizer_metadata),
            "training": {
                "tokens_this_update": self._begin.tokens_this_update,
                "processed_target_tokens": self._begin.processed_target_tokens,
                "train_nll_pre_update": nll,
                "learning_rate": _finite_float(self._begin.learning_rate, "learning_rate"),
                "pre_clip_grad_norm": _finite_float(self._begin.pre_clip_grad_norm, "pre_clip_grad_norm"),
                "clipped": bool(self._begin.clipped),
                "grad_clip_norm": _finite_float(self._begin.grad_clip_norm, "grad_clip_norm"),
            },
            "data": {
                "sequence_length": int(self.context.data_config["sequence_length"]) if "sequence_length" in self.context.data_config else int(self.context.model_config["sequence_length"]),
                "micro_batch_size": int(self.context.train_config["micro_batch_size"]),
                "accumulation_steps": int(self.context.train_config["accumulation_steps"]),
                "window_ids": list(self._begin.window_ids),
                "micro_batch_window_ids": [list(group) for group in self._begin.micro_batch_window_ids],
                "sampler_epoch_before": self._begin.sampler_epoch_before,
                "sampler_offset_before": self._begin.sampler_offset_before,
                "sampler_epoch_after": self._begin.sampler_epoch_after,
                "sampler_offset_after": self._begin.sampler_offset_after,
            },
            "tensors_file": {
                "path": f"update_{update:06d}.pt",
                "sha256": None,
                "nbytes": None,
                "payload_nbytes": int(payload_nbytes),
                "format": TENSOR_FORMAT,
                "version": SNAPSHOT_VERSION,
            },
            "parameters": self._buffer,
            "aliases": list(self._aliases),
            "integrity": integrity,
            "collector": {
                "collector_version": COLLECTOR_VERSION,
                "clone_seconds": None,
                "checksum_seconds": None,
                "write_seconds": None,
                "total_seconds": None,
                "snapshot_bytes": None,
            },
        }
        return payload, manifest

    # -- disk and existing targets -----------------------------------------

    def _committed_bytes(self) -> int:
        return sum(int(info["pt_nbytes"]) + int(info["manifest_nbytes"]) for info in self._committed.values())

    def _precheck_disk(self, *, payload_nbytes: int) -> None:
        estimate_pt = int(payload_nbytes) + 1024 * 1024
        estimate_manifest = 512 * 1024
        required = estimate_pt + estimate_manifest
        try:
            free = shutil.disk_usage(self.snapshots_dir).free
        except OSError as exc:
            raise SnapshotIntegrityError(f"cannot determine free disk space: {exc}") from exc
        if free < required + int(self.config.min_free_bytes):
            self._skip("skipped_no_space", update=self._begin.update if self._begin else None,
                       details={"required_bytes": required, "free_bytes": int(free),
                                "min_free_bytes": int(self.config.min_free_bytes)})
            raise _SkipSnapshot()
        if self.config.max_total_bytes is not None:
            current = self._committed_bytes()
            if current + required > int(self.config.max_total_bytes):
                self._skip("skipped_max_total_bytes", update=self._begin.update if self._begin else None,
                           details={"required_bytes": required, "committed_bytes": current,
                                    "max_total_bytes": int(self.config.max_total_bytes)})
                raise _SkipSnapshot()

    def _resolve_existing_targets(self, update: int) -> None:
        pt_path = self.snapshots_dir / f"update_{update:06d}.pt"
        manifest_path = self.snapshots_dir / f"update_{update:06d}.manifest.json"
        if manifest_path.exists():
            ok, reason = self._validate_manifest_file(manifest_path)
            if ok:
                self._skip("skipped_existing", update=update,
                           details={"manifest": manifest_path.name, "reason": "valid committed snapshot"})
                raise _SkipSnapshot()
            self._conflicted[update] = reason
            self._skip("existing_snapshot_corrupt", update=update,
                       details={"manifest": manifest_path.name, "reason": reason})
            raise _SkipSnapshot()
        if pt_path.exists():
            moved = self._move_orphan(pt_path)
            self._log_error("orphaned_tensor_file", update=update, phase="finish",
                            message=f"tensor file had no manifest; moved to {moved}")

    def _move_orphan(self, pt_path: Path) -> Path:
        orphan_dir = self.snapshots_dir / "orphaned"
        orphan_dir.mkdir(parents=True, exist_ok=True)
        target = orphan_dir / pt_path.name
        if target.exists():
            target = orphan_dir / f"{pt_path.stem}.{time.time_ns()}{pt_path.suffix}"
        os.replace(pt_path, target)
        return target

    # -- startup scan -------------------------------------------------------

    def _cleanup_temp_files(self) -> None:
        for path in list(self.snapshots_dir.glob("*.tmp")):
            try:
                path.unlink()
                self._notes.append(f"removed temporary file {path.name}")
            except OSError:
                pass

    def _scan_existing(self) -> None:
        manifest_paths = sorted(self.snapshots_dir.glob("update_*.manifest.json"))
        for manifest_path in manifest_paths:
            try:
                manifest = json.loads(manifest_path.read_text())
                update = int(manifest["update"])
            except Exception as exc:
                self._conflicted[-1] = f"unreadable manifest {manifest_path.name}: {exc}"
                self._log_error("existing_snapshot_corrupt", phase="scan",
                                message=f"cannot parse {manifest_path.name}: {exc}")
                continue
            ok, reason = self._validate_manifest_file(manifest_path)
            if ok:
                try:
                    pt_path = self.snapshots_dir / f"update_{update:06d}.pt"
                    self._committed[update] = {
                        "pt_path": str(pt_path),
                        "manifest_path": str(manifest_path),
                        "pt_nbytes": int(os.path.getsize(pt_path)),
                        "manifest_nbytes": int(os.path.getsize(manifest_path)),
                    }
                except OSError as exc:
                    self._conflicted[update] = f"cannot stat committed snapshot: {exc}"
                    self._log_error("existing_snapshot_corrupt", update=update, phase="scan", message=str(exc))
            else:
                self._conflicted[update] = reason
                self._log_error("existing_snapshot_corrupt", update=update, phase="scan", message=reason)

        for pt_path in sorted(self.snapshots_dir.glob("update_*.pt")):
            manifest_path = self.snapshots_dir / f"{pt_path.stem}.manifest.json"
            if not manifest_path.exists():
                try:
                    moved = self._move_orphan(pt_path)
                    self._notes.append(f"moved orphan {pt_path.name} to {moved.name}")
                    self._log_error("orphaned_tensor_file", phase="scan",
                                    message=f"{pt_path.name} had no manifest; moved to {moved}")
                except OSError as exc:
                    self._log_error("orphaned_tensor_file", phase="scan",
                                    message=f"could not move {pt_path.name}: {exc}")

    def _validate_manifest_file(self, manifest_path: Path) -> tuple[bool, str | None]:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"cannot read manifest: {exc}"
        try:
            if not isinstance(manifest, dict):
                raise SnapshotIntegrityError("manifest root is not an object")
            if manifest.get("format") != SNAPSHOT_FORMAT or manifest.get("version") != SNAPSHOT_VERSION:
                raise SnapshotIntegrityError("unsupported manifest format/version")
            if manifest.get("status") != "committed":
                raise SnapshotIntegrityError("manifest status is not committed")
            update = manifest.get("update")
            if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
                raise SnapshotIntegrityError("invalid manifest update")
            if manifest_path.name != f"update_{update:06d}.manifest.json":
                raise SnapshotIntegrityError("manifest filename does not match update")
            if manifest.get("run_id") != self.context.run_id:
                raise SnapshotIntegrityError("run_id mismatch")
            if manifest.get("snapshot_id") != f"{self.context.run_id}:{update:06d}":
                raise SnapshotIntegrityError("snapshot_id mismatch")
            identities = manifest.get("identities", {})
            if identities.get("recipe_fingerprint") != self.context.recipe_fingerprint:
                raise SnapshotIntegrityError("recipe fingerprint mismatch")
            if identities.get("data_fingerprint") != self.context.data_fingerprint:
                raise SnapshotIntegrityError("data fingerprint mismatch")
            if identities.get("snapshot_config_hash") != self.config.config_hash:
                raise SnapshotIntegrityError("snapshot config hash mismatch")
            tensors_file = manifest.get("tensors_file", {})
            expected_name = f"update_{update:06d}.pt"
            if tensors_file.get("path") != expected_name:
                raise SnapshotIntegrityError("tensors_file.path is not the expected basename")
            if tensors_file.get("format") != TENSOR_FORMAT or tensors_file.get("version") != SNAPSHOT_VERSION:
                raise SnapshotIntegrityError("tensors_file format/version mismatch")
            pt_path = self.snapshots_dir / expected_name
            if not pt_path.is_file():
                raise SnapshotIntegrityError("tensors file is missing")
            if int(os.path.getsize(pt_path)) != int(tensors_file.get("nbytes", -1)):
                raise SnapshotIntegrityError("tensors file size mismatch")
            if sha256_file(pt_path) != tensors_file.get("sha256"):
                raise SnapshotIntegrityError("tensors file SHA-256 mismatch")
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise SnapshotIntegrityError("payload root is not an object")
            if payload.get("format") != TENSOR_FORMAT or payload.get("version") != SNAPSHOT_VERSION:
                raise SnapshotIntegrityError("payload format/version mismatch")
            for key in ("snapshot_id", "run_id", "update", "segment_id"):
                if payload.get(key) != manifest.get(key):
                    raise SnapshotIntegrityError(f"payload/manifest mismatch for {key}")
            tensors = payload.get("tensors")
            if not isinstance(tensors, dict):
                raise SnapshotIntegrityError("payload tensors is not a dict")
            parameters = manifest.get("parameters")
            if not isinstance(parameters, dict):
                raise SnapshotIntegrityError("manifest parameters is not a dict")
            if set(parameters) != self._expected_name_set:
                raise SnapshotIntegrityError("manifest parameter set does not match the Muon group")
            referenced: set[str] = set()
            alias_count = 0
            role_slots = 0
            for name, entry in parameters.items():
                expected_shape = [int(x) for x in self._expected_shapes[name]]
                if entry.get("shape") != expected_shape:
                    raise SnapshotIntegrityError(f"{name}: shape mismatch")
                if entry.get("dtype") != "float32":
                    raise SnapshotIntegrityError(f"{name}: dtype mismatch")
                if entry.get("parameter_index") != self._name_to_index[name]:
                    raise SnapshotIntegrityError(f"{name}: parameter_index mismatch")
                if entry.get("optimizer_param_index") != int(self.context.optimizer_param_index[name]):
                    raise SnapshotIntegrityError(f"{name}: optimizer_param_index mismatch")
                roles = entry.get("roles", {})
                if set(roles) != set(ROLE_ORDER):
                    raise SnapshotIntegrityError(f"{name}: role set mismatch")
                for role in ROLE_ORDER:
                    descriptor = roles[role]
                    role_slots += 1
                    if "alias_of" in descriptor:
                        canonical_role = descriptor["alias_of"]
                        if canonical_role not in ROLE_ORDER or canonical_role == role:
                            raise SnapshotIntegrityError(f"{name}.{role}: invalid alias target")
                        canonical = roles.get(canonical_role, {})
                        if "alias_of" in canonical:
                            raise SnapshotIntegrityError(f"{name}.{role}: alias chain is not allowed")
                        tensor_id = canonical.get("tensor_id")
                        alias_count += 1
                    else:
                        tensor_id = descriptor.get("tensor_id")
                    if tensor_id not in tensors:
                        raise SnapshotIntegrityError(f"{name}.{role}: missing tensor {tensor_id!r}")
                    referenced.add(tensor_id)
                    tensor = tensors[tensor_id]
                    if not torch.is_tensor(tensor) or tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                        raise SnapshotIntegrityError(f"{name}.{role}: stored tensor is not CPU float32")
                    if list(tensor.shape) != expected_shape:
                        raise SnapshotIntegrityError(f"{name}.{role}: stored tensor shape mismatch")
                    if "alias_of" not in descriptor:
                        if descriptor.get("shape") != expected_shape or descriptor.get("dtype") != "float32":
                            raise SnapshotIntegrityError(f"{name}.{role}: descriptor contract mismatch")
                        if descriptor.get("nbytes") != int(descriptor.get("numel", -1)) * 4:
                            raise SnapshotIntegrityError(f"{name}.{role}: descriptor nbytes mismatch")
                        if sha256_tensor(tensor) != descriptor.get("sha256"):
                            raise SnapshotIntegrityError(f"{name}.{role}: tensor checksum mismatch")
            if referenced != set(tensors):
                raise SnapshotIntegrityError("payload contains orphan tensors")
            integrity = manifest.get("integrity", {})
            expected_slots = len(self._expected_names) * ROLE_SLOTS_PER_PARAMETER
            checks = {
                "expected_parameter_count": len(self._expected_names),
                "observed_parameter_count": len(parameters),
                "expected_role_slots": expected_slots,
                "observed_role_slots": role_slots,
                "unique_tensor_count": len(tensors),
                "alias_count": alias_count,
                "all_parameters_present": True,
                "all_roles_present": True,
                "all_shapes_match": True,
                "all_dtypes_float32": True,
                "per_tensor_checksums_ok": True,
                "tensors_file_sha256_ok": True,
                "no_orphan_tensors": True,
            }
            for key, value in checks.items():
                if integrity.get(key) != value:
                    raise SnapshotIntegrityError(f"integrity.{key} mismatch")
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # -- logging and state helpers -----------------------------------------

    def _log_error(self, code: str, *, update: int | None = None, phase: str,
                   message: str, parameter_name: str | None = None,
                   details: dict | None = None) -> None:
        self._error_count += 1
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.context.run_id,
            "update": update,
            "phase": phase,
            "code": code,
            "message": str(message),
            "parameter_name": parameter_name,
            "details": _json_safe(details or {}),
        }
        try:
            with self.errors_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        except Exception:
            # Error reporting must not affect training.
            pass

    def _clear_buffer(self) -> None:
        self._buffer = None
        self._begin = None
        self._collecting = False
        self._aliases = []
        self._tensor_blobs = {}
        self._tensor_descriptors = {}
        self._tensor_owner = {}
        self._next_tensor_id = 0
        self._started_at = None

    def _skip(self, code: str, *, update: int | None, details: dict | None = None) -> None:
        self._log_error(code, update=update, phase="finish", message=code, details=details)
        self._clear_buffer()

    def _fail(self, code: str, *, update: int | None, phase: str, exc: Exception | None = None,
              message: str | None = None, parameter_name: str | None = None,
              details: dict | None = None) -> None:
        reason = message or code
        if exc is not None:
            reason = f"{type(exc).__name__}: {exc}"
        self._disabled = True
        self._disabled_reason = code
        self._log_error(code, update=update, phase=phase, message=reason,
                        parameter_name=parameter_name, details=details)
        self._clear_buffer()


class _SkipSnapshot(Exception):
    """Internal control flow for a non-fatal skip (disk, existing snapshot)."""


def _incomplete_message(expected: set[str], observed: set[str]) -> str:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return f"incomplete Muon parameter set: missing={missing}, unexpected={unexpected}"
