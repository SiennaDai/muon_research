"""Hessian-ready node checkpoints for the Structural Concentration rerun.

At selected pre-update nodes (``completed_updates = u - 1``) this sidecar writes:

* ``pre_uNNNNNN.pt`` — full model state, Muon momentum, sampler state, RNG,
  update counters and fingerprints.  Auxiliary AdamW moments are deliberately
  not stored (Hessian analysis needs model + data; Muon roles are recoverable
  from model + momentum + the exact update batch).
* ``pre_uNNNNNN_roles.pt`` — QKV_fused and O role tensors for nodes that are not
  already covered by the frozen Stage 1–3 snapshots (7 independent roles;
  ``M_t_persisted`` is recorded as an alias of ``M_t_candidate``).

Everything is written atomically and registered in ``node_checkpoints/index.json``.
The collector never advances the sampler/RNG or mutates training state.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from experiment_io import (atomic_torch_save_with_digest, atomic_write_json,
                           rng_state, sha256_tensor)

NODE_FORMAT = "muon_node_checkpoint"
NODE_VERSION = 1
ROLE_FORMAT = "muon_node_roles"
ROLE_VERSION = 1
CONFIG_FORMAT = "muon_node_checkpoint_config"
CONFIG_VERSION = 1

ROLE_ORDER = ("W_before", "W_after", "G", "M_prev", "M_t_candidate",
              "M_t_persisted", "D", "U")
ALIAS_ROLE = "M_t_persisted"
ROLE_TYPES = ("QKV_fused", "O")


class NodeConfigError(ValueError):
    """Invalid node-checkpoint sidecar configuration."""


@dataclass(frozen=True)
class NodeConfig:
    schema_version: int = CONFIG_VERSION
    enabled: bool = False
    updates: tuple[int, ...] = ()
    role_updates: tuple[int, ...] = ()
    role_matrix_types: tuple[str, ...] = ROLE_TYPES
    output_dir: str = "node_checkpoints"

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self)


def _config_dict(config: NodeConfig) -> dict[str, Any]:
    return {
        "schema_version": int(config.schema_version),
        "enabled": bool(config.enabled),
        "updates": [int(x) for x in config.updates],
        "role_updates": [int(x) for x in config.role_updates],
        "role_matrix_types": list(config.role_matrix_types),
        "output_dir": str(config.output_dir),
    }


def compute_config_hash(config: NodeConfig) -> str:
    payload = {"node_checkpoint_config": _config_dict(config),
               "collector_version": NODE_VERSION}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NodeConfigError(f"{field} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise NodeConfigError(f"{field} is out of range")
    return value


def _parse(data: dict, *, source: str) -> NodeConfig:
    required = {"schema_version", "enabled", "updates", "role_updates",
                "role_matrix_types", "output_dir"}
    if not isinstance(data, dict) or set(data) != required:
        missing = sorted(required - set(data or {}))
        unknown = sorted(set(data or {}) - required)
        raise NodeConfigError(f"{source}: fields mismatch missing={missing} unknown={unknown}")
    if _positive_int(data["schema_version"], f"{source}.schema_version") != CONFIG_VERSION:
        raise NodeConfigError(f"{source}.schema_version must be {CONFIG_VERSION}")
    if not isinstance(data["enabled"], bool):
        raise NodeConfigError(f"{source}.enabled must be bool")
    updates = []
    for index, value in enumerate(data["updates"]):
        update = _positive_int(value, f"{source}.updates[{index}]", allow_zero=True)
        if updates and update <= updates[-1]:
            raise NodeConfigError(f"{source}.updates must be strictly increasing")
        updates.append(update)
    role_updates = []
    for index, value in enumerate(data["role_updates"]):
        update = _positive_int(value, f"{source}.role_updates[{index}]")
        if role_updates and update <= role_updates[-1]:
            raise NodeConfigError(f"{source}.role_updates must be strictly increasing")
        role_updates.append(update)
    types = data["role_matrix_types"]
    if not isinstance(types, list) or any(t not in ROLE_TYPES for t in types):
        raise NodeConfigError(f"{source}.role_matrix_types must be a subset of {ROLE_TYPES}")
    if len(set(types)) != len(types):
        raise NodeConfigError(f"{source}.role_matrix_types contains duplicates")
    if not data["enabled"] and (updates or role_updates):
        raise NodeConfigError(f"{source}: disabled config must have empty updates")
    if data["enabled"] and not updates:
        raise NodeConfigError(f"{source}: enabled config needs at least one update")
    if not set(role_updates) <= set(updates):
        raise NodeConfigError(f"{source}: role_updates must be a subset of updates")
    if 0 in role_updates:
        raise NodeConfigError(f"{source}: roles are not captured at initialization")
    output_dir = data["output_dir"]
    if not isinstance(output_dir, str) or not output_dir or "/" in output_dir:
        raise NodeConfigError(f"{source}.output_dir must be a single path component")
    return NodeConfig(schema_version=CONFIG_VERSION, enabled=data["enabled"],
                      updates=tuple(updates), role_updates=tuple(role_updates),
                      role_matrix_types=tuple(types), output_dir=output_dir)


def load_node_config(path: str | Path) -> NodeConfig:
    path = Path(path)
    if not path.is_file():
        raise NodeConfigError(f"node config not found: {path}")
    try:
        return _parse(json.loads(path.read_text()), source=str(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise NodeConfigError(f"cannot read node config {path}: {exc}") from exc


def load_saved_node_config(run_dir: str | Path) -> NodeConfig | None:
    path = Path(run_dir) / "node_checkpoint_config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NodeConfigError(f"cannot read stored node config {path}: {exc}") from exc
    if (not isinstance(payload, dict) or payload.get("format") != CONFIG_FORMAT
            or payload.get("version") != CONFIG_VERSION):
        raise NodeConfigError(f"{path}: unsupported stored node config")
    if "config" not in payload or "config_hash" not in payload:
        raise NodeConfigError(f"{path}: stored node config lacks config/config_hash")
    config = _parse(payload["config"], source=f"{path}.config")
    if payload["config_hash"] != config.config_hash:
        raise NodeConfigError(f"{path}: stored node config hash mismatch")
    return config


def resolve_node_config(path: str | Path | None, *, run_dir: str | Path,
                        resume: bool) -> NodeConfig:
    run_dir = Path(run_dir)
    stored_path = run_dir / "node_checkpoint_config.json"
    if resume:
        if not stored_path.is_file():
            if path is None:
                return NodeConfig()
            raise NodeConfigError("resume was given --node-checkpoint-config "
                                  "but the run has no stored node config")
        stored = load_saved_node_config(run_dir)
        if stored is None:
            raise NodeConfigError("stored node config disappeared during resume")
        if path is not None:
            provided = load_node_config(path)
            if provided.config_hash != stored.config_hash:
                raise NodeConfigError(
                    f"resume node config hash mismatch: provided={provided.config_hash} "
                    f"stored={stored.config_hash}")
            return provided
        return stored
    config = NodeConfig() if path is None else load_node_config(path)
    atomic_write_json(stored_path, {
        "format": CONFIG_FORMAT, "version": CONFIG_VERSION,
        "config_hash": config.config_hash, "config": _config_dict(config)})
    return config


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class NodeRunContext:
    run_id: str
    segment_id: int
    recipe_fingerprint: str
    data_fingerprint: str
    config_hash: str
    model_config: dict
    parameter_records: list[dict]
    muon_parameter_order: tuple[str, ...]
    train_config: dict
    data_config: dict
    tokens_per_update: int
    seeds: dict
    source: dict
    environment: dict
    device: str
    precision: dict


class MuonNodeCheckpointCollector:
    def __init__(self, *, run_dir: str | Path, config: NodeConfig,
                 context: NodeRunContext, completed_updates: int = 0):
        if not config.enabled:
            raise NodeConfigError("collector requires an enabled config")
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.context = context
        if config.config_hash != context.config_hash:
            raise NodeConfigError("node config hash does not match run context")
        self.directory = self.run_dir / config.output_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.json"
        self.identity = {
            "run_id": context.run_id,
            "recipe_fingerprint": context.recipe_fingerprint,
            "data_fingerprint": context.data_fingerprint,
            "node_config_hash": config.config_hash,
        }
        self._index: dict[str, Any] = {"format": "muon_node_checkpoint_index",
                                       "version": 1, "identity": self.identity,
                                       "nodes": []}
        if self.index_path.is_file():
            payload = json.loads(self.index_path.read_text())
            if payload.get("identity") != self.identity:
                raise NodeConfigError("existing node index identity mismatch")
            self._index = payload
        self._by_update = {int(node["update"]): node
                           for node in self._index.get("nodes", [])}
        self._validate_existing()
        role_types = set(config.role_matrix_types)
        self._role_names = tuple(
            record["name"] for record in context.parameter_records
            if record.get("group") == "muon"
            and record["name"] in self._head_parameter_names(role_types))
        if not self._role_names:
            raise NodeConfigError("no QKV/O Muon parameters available for role capture")
        self._role_update: int | None = None
        self._role_tensors: dict[str, torch.Tensor] = {}
        self._role_descriptors: dict[str, dict] = {}
        self._aliases: list[dict] = []
        self._closed = False
        self._write_index()

    # -- configuration helpers ---------------------------------------------

    def _head_parameter_names(self, role_types: set[str]) -> set[str]:
        names = set()
        for name in self.context.muon_parameter_order:
            if "attn.c_attn.weight" in name and "QKV_fused" in role_types:
                names.add(name)
            elif "attn.c_proj.weight" in name and "O" in role_types:
                names.add(name)
        return names

    def _validate_existing(self) -> None:
        for update, node in self._by_update.items():
            checkpoint = self.directory / node["checkpoint_file"]
            if not checkpoint.is_file():
                raise NodeConfigError(f"indexed node checkpoint is missing: {checkpoint}")
            if checkpoint.stat().st_size != int(node["checkpoint_nbytes"]):
                raise NodeConfigError(f"node checkpoint size mismatch: {checkpoint}")
            roles_file = node.get("roles_file")
            if roles_file and not (self.directory / roles_file).is_file():
                raise NodeConfigError(f"indexed role file is missing: {self.directory / roles_file}")

    def _write_index(self) -> None:
        self._index["nodes"] = sorted(self._by_update.values(),
                                      key=lambda node: int(node["update"]))
        atomic_write_json(self.index_path, self._index)

    # -- base checkpoint ----------------------------------------------------

    def should_save(self, update: int) -> bool:
        update = int(update)
        return (not self._closed and update in self.config.updates
                and update not in self._by_update)

    def save_base(self, *, update: int, model, optimizer, sampler_state: dict,
                  processed_target_tokens: int, window_ids: list[int],
                  micro_batch_window_ids: list[list[int]],
                  sampler_epoch_before: int, sampler_offset_before: int,
                  sampler_epoch_after: int, sampler_offset_after: int) -> dict:
        update = int(update)
        if update not in self.config.updates:
            raise NodeConfigError(f"update {update} is not a configured node")
        if update in self._by_update:
            return self._by_update[update]
        muon_momentum = {}
        missing = []
        for name in self.context.muon_parameter_order:
            parameter = model.get_parameter(name)
            state = optimizer.state.get(parameter, {})
            momentum = state.get("muon_momentum")
            if momentum is None:
                missing.append(name)
            else:
                muon_momentum[name] = momentum.detach().to("cpu", torch.float32).clone()
        completed_updates = 0 if update == 0 else update - 1
        payload = {
            "format": NODE_FORMAT,
            "version": NODE_VERSION,
            "run_id": self.context.run_id,
            "update": update,
            "completed_updates": completed_updates,
            "processed_target_tokens": int(processed_target_tokens),
            "scientific_fingerprint": self.context.recipe_fingerprint,
            "data_fingerprint": self.context.data_fingerprint,
            "node_config_hash": self.config.config_hash,
            "model": model.state_dict(),
            "muon_momentum": muon_momentum,
            "muon_momentum_missing": missing,
            "sampler": sampler_state,
            "rng": rng_state(torch.cuda.is_available()),
            "next_segment_id": self.context.segment_id + 1,
            "parameter_records": _json_safe(self.context.parameter_records),
            "seeds": _json_safe(self.context.seeds),
            "source": _json_safe(self.context.source),
            "seeds_config": _json_safe(self.context.data_config),
        }
        path = self.directory / f"pre_u{update:06d}.pt"
        result = atomic_torch_save_with_digest(payload, path)
        sidecar = {
            "format": NODE_FORMAT,
            "version": NODE_VERSION,
            "update": update,
            "completed_updates": completed_updates,
            "processed_target_tokens": int(processed_target_tokens),
            "checkpoint_file": path.name,
            "checkpoint_sha256": result["sha256"],
            "checkpoint_nbytes": int(result["nbytes"]),
            "window_ids": [int(x) for x in window_ids],
            "micro_batch_window_ids": [[int(x) for x in group]
                                       for group in micro_batch_window_ids],
            "sampler_epoch_before": int(sampler_epoch_before),
            "sampler_offset_before": int(sampler_offset_before),
            "sampler_epoch_after": int(sampler_epoch_after),
            "sampler_offset_after": int(sampler_offset_after),
            "recipe_fingerprint": self.context.recipe_fingerprint,
            "data_fingerprint": self.context.data_fingerprint,
            "node_config_hash": self.config.config_hash,
            "seed": self.context.seeds.get("seed"),
            "code_commit": self.context.source.get("upstream_commit"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "roles_file": None,
        }
        atomic_write_json(self.directory / f"pre_u{update:06d}.json", sidecar)
        node = {
            "update": update, "completed_updates": completed_updates,
            "checkpoint_file": path.name,
            "checkpoint_sha256": result["sha256"],
            "checkpoint_nbytes": int(result["nbytes"]),
            "roles_file": None,
            "window_ids": [int(x) for x in window_ids],
        }
        self._by_update[update] = node
        self._write_index()
        if update in self.config.role_updates:
            self._role_update = update
            self._role_tensors = {}
            self._role_descriptors = {}
            self._aliases = []
        return node

    # -- role capture -------------------------------------------------------

    def observe(self, *, update: int | None, parameter_name: str | None,
                W_before: torch.Tensor, G: torch.Tensor, M_prev: torch.Tensor,
                M_t_candidate: torch.Tensor, M_t_persisted: torch.Tensor,
                D: torch.Tensor, U: torch.Tensor, W_after: torch.Tensor,
                first_step: bool) -> None:
        if self._closed or self._role_update is None:
            return
        if update is None or int(update) != self._role_update:
            return
        name = str(parameter_name)
        if name not in self._role_names or name in self._role_tensors:
            return
        roles = {"W_before": W_before, "W_after": W_after, "G": G,
                 "M_prev": M_prev, "M_t_candidate": M_t_candidate,
                 "M_t_persisted": M_t_persisted, "D": D, "U": U}
        canonical = {}
        for role, tensor in roles.items():
            if role == ALIAS_ROLE and tensor is M_t_candidate:
                self._aliases.append({"parameter_name": name,
                                      "alias_role": ALIAS_ROLE,
                                      "canonical_role": "M_t_candidate"})
                continue
            cpu = tensor.detach().to("cpu", torch.float32).clone()
            key = f"{name}|{role}"
            self._role_tensors[key] = cpu
            self._role_descriptors[key] = {
                "parameter_name": name, "role": role,
                "shape": [int(x) for x in cpu.shape],
                "numel": int(cpu.numel()),
                "nbytes": int(cpu.numel() * 4),
                "sha256": sha256_tensor(cpu),
            }
            canonical[role] = key

    def finish_roles(self, *, update: int) -> None:
        if self._closed or self._role_update is None:
            return
        if int(update) != self._role_update:
            return
        missing = [name for name in self._role_names
                   if not any(key.startswith(name + "|")
                              for key in self._role_tensors)]
        if missing:
            raise RuntimeError(f"node roles incomplete at update {update}: {missing}")
        payload = {
            "format": ROLE_FORMAT,
            "version": ROLE_VERSION,
            "run_id": self.context.run_id,
            "update": int(update),
            "tensors": self._role_tensors,
            "aliases": self._aliases,
        }
        path = self.directory / f"pre_u{int(update):06d}_roles.pt"
        result = atomic_torch_save_with_digest(payload, path)
        sidecar = {
            "format": ROLE_FORMAT,
            "version": ROLE_VERSION,
            "update": int(update),
            "roles_file": path.name,
            "roles_sha256": result["sha256"],
            "roles_nbytes": int(result["nbytes"]),
            "parameters": [name for name in self._role_names
                           if any(key.startswith(name + "|")
                                  for key in self._role_tensors)],
            "role_order": list(ROLE_ORDER),
            "matrix_types": list(self.config.role_matrix_types),
            "tensors": self._role_descriptors,
            "aliases": self._aliases,
            "node_config_hash": self.config.config_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(self.directory / f"pre_u{int(update):06d}_roles.json", sidecar)
        node = self._by_update[int(update)]
        node["roles_file"] = path.name
        node["roles_sha256"] = result["sha256"]
        node["roles_nbytes"] = int(result["nbytes"])
        self._write_index()
        self._role_update = None
        self._role_tensors = {}
        self._role_descriptors = {}
        self._aliases = []

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        status = {
            "format": "muon_node_checkpoint_status",
            "version": 1,
            "run_id": self.context.run_id,
            "config_hash": self.config.config_hash,
            "nodes_saved": [int(node["update"]) for node in
                            sorted(self._by_update.values(),
                                   key=lambda node: int(node["update"]))],
            "nodes_with_roles": [int(node["update"]) for node in
                                 self._by_update.values() if node.get("roles_file")],
            "errors": [],
        }
        try:
            atomic_write_json(self.directory / "status.json", status)
        except Exception:
            pass
