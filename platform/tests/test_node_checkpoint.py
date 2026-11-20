import json
from pathlib import Path

import pytest
import torch

import train_platform
from optim.muon_node_checkpoint import (
    NodeConfigError, load_node_config, resolve_node_config,
)


def _node_config(tmp_path, *, updates=(0, 1, 2), roles=(1, 2), enabled=True):
    path = tmp_path / "node_config.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "enabled": enabled,
        "updates": list(updates),
        "role_updates": list(roles),
        "role_matrix_types": ["QKV_fused", "O"],
        "output_dir": "node_checkpoints",
    }))
    return path


def _run(tmp_path, cfg, name, node_config, *, resume=None, stop_at_update=None):
    return train_platform.run(
        cfg, tmp_path / name, resume=resume, to_device="cpu",
        stop_at_update=stop_at_update, node_checkpoint_config_path=node_config)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_node_checkpoint_files_and_numerical_invariance(tmp_path, tiny_muon_config):
    config_path = _node_config(tmp_path)
    summary = _run(tmp_path, tiny_muon_config, "run", config_path)
    assert summary["completed_updates"] == 3
    directory = tmp_path / "run/node_checkpoints"
    for update in (0, 1, 2):
        assert (directory / f"pre_u{update:06d}.pt").is_file()
        assert (directory / f"pre_u{update:06d}.json").is_file()
    for update in (1, 2):
        assert (directory / f"pre_u{update:06d}_roles.pt").is_file()
        assert (directory / f"pre_u{update:06d}_roles.json").is_file()
    index = json.loads((directory / "index.json").read_text())
    assert [node["update"] for node in index["nodes"]] == [0, 1, 2]

    init = _load(directory / "pre_u000000.pt")
    assert init["completed_updates"] == 0
    base = _load(directory / "pre_u000002.pt")
    assert base["completed_updates"] == 1
    assert base["update"] == 2
    assert "model" in base and "sampler" in base and "rng" in base
    assert base["muon_momentum"]
    assert not base["muon_momentum_missing"]

    sidecar = json.loads((directory / "pre_u000002.json").read_text())
    assert len(sidecar["window_ids"]) == 2  # micro_batch 2 x accumulation 1
    assert sidecar["checkpoint_sha256"]

    roles = _load(directory / "pre_u000002_roles.pt")
    names = {key.split("|")[0] for key in roles["tensors"]}
    assert names == {"transformer.h.0.attn.c_attn.weight",
                     "transformer.h.0.attn.c_proj.weight"}
    assert len(roles["tensors"]) == 2 * 7
    assert not any(key.endswith("|M_t_persisted") for key in roles["tensors"])
    assert len(roles["aliases"]) == 2

    plain_config = _node_config(tmp_path, enabled=False, updates=(), roles=())
    from config.recipe import load_recipe
    plain_summary = _run(tmp_path, tiny_muon_config, "plain", plain_config)
    assert plain_summary["completed_updates"] == 3
    left = _load(tmp_path / "run/checkpoints/latest.pt")
    right = _load(tmp_path / "plain/checkpoints/latest.pt")
    for key, value in left["model"].items():
        assert torch.equal(value, right["model"][key]), key


def test_resume_skips_existing_nodes(tmp_path, tiny_muon_config):
    config_path = _node_config(tmp_path, updates=(0, 1, 2, 3), roles=(1, 2, 3))
    _run(tmp_path, tiny_muon_config, "run", config_path, stop_at_update=2)
    directory = tmp_path / "run/node_checkpoints"
    first = json.loads((directory / "index.json").read_text())
    assert [node["update"] for node in first["nodes"]] == [0, 1, 2]
    assert (directory / "pre_u000002_roles.pt").is_file()
    assert not (directory / "pre_u000003.pt").exists()

    summary = _run(tmp_path, tiny_muon_config, "run", config_path,
                   resume=tmp_path / "run/checkpoints/latest.pt")
    assert summary["completed_updates"] == 3
    second = json.loads((directory / "index.json").read_text())
    assert [node["update"] for node in second["nodes"]] == [0, 1, 2, 3]
    assert (directory / "pre_u000003_roles.pt").is_file()


def test_config_validation(tmp_path):
    good = json.loads(_node_config(tmp_path).read_text())
    assert load_node_config(tmp_path / "node_config.json").updates == (0, 1, 2)

    bad = dict(good)
    bad["role_updates"] = [4]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(NodeConfigError):
        load_node_config(tmp_path / "bad.json")

    unknown = dict(good)
    unknown["extra"] = 1
    (tmp_path / "unknown.json").write_text(json.dumps(unknown))
    with pytest.raises(NodeConfigError):
        load_node_config(tmp_path / "unknown.json")
