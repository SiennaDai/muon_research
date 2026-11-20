"""Read-only access to Hessian-ready node checkpoints.

Each node ``u`` is the state **before** update ``u`` (``completed_updates=u-1``).
The base checkpoint contains the full model, Muon momentum, sampler state and
RNG.  Role files exist only for nodes not already covered by the frozen Stage
1-3 snapshots (QKV_fused and O, seven independent roles).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_index(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "node_checkpoints/index.json").read_text())


def list_nodes(run_dir: str | Path) -> list[int]:
    return [int(node["update"]) for node in load_index(run_dir)["nodes"]]


def load_node(run_dir: str | Path, update: int) -> tuple[dict, dict]:
    directory = Path(run_dir) / "node_checkpoints"
    checkpoint = torch.load(directory / f"pre_u{update:06d}.pt",
                            map_location="cpu", weights_only=False)
    sidecar = json.loads((directory / f"pre_u{update:06d}.json").read_text())
    return checkpoint, sidecar


def load_roles(run_dir: str | Path, update: int) -> tuple[dict, dict] | None:
    directory = Path(run_dir) / "node_checkpoints"
    path = directory / f"pre_u{update:06d}_roles.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sidecar = json.loads((directory / f"pre_u{update:06d}_roles.json").read_text())
    return payload, sidecar


def role_tensor(payload: dict, parameter_name: str, role: str) -> torch.Tensor:
    key = f"{parameter_name}|{role}"
    if key in payload["tensors"]:
        return payload["tensors"][key]
    for alias in payload.get("aliases", []):
        if (alias["parameter_name"] == parameter_name
                and alias["alias_role"] == role):
            return payload["tensors"][f"{parameter_name}|{alias['canonical_role']}"]
    raise KeyError(key)


def muon_momentum(checkpoint: dict) -> dict:
    return dict(checkpoint.get("muon_momentum", {}))


def model_state(checkpoint: dict) -> dict:
    return dict(checkpoint["model"])


def sampler_state(checkpoint: dict) -> dict:
    return dict(checkpoint["sampler"])
