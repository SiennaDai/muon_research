import json
import os
from pathlib import Path

import pytest
import torch

import train_platform
from optim.muon_snapshot_collector import SnapshotConfigError


def _fake_source(_root):
    return {"remote": None, "upstream_commit": "test-commit", "dirty": False,
            "diff": "", "provenance_status": "ok", "provenance_error": None,
            "safe_directory_used": False}


def _snapshot_config(tmp_path, updates=(1, 2, 3), name="snapshot.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema_version": 1, "enabled": True, "updates": list(updates),
        "on_error": "disable", "max_total_bytes": 100_000_000,
        "min_free_bytes": 0,
    }))
    return path


def _train_events(run_dir):
    events = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event_type") == "train":
            events.append({key: value for key, value in event.items()
                           if key not in {"event_index", "segment_id", "run_id",
                                          "elapsed_seconds", "update_elapsed_seconds",
                                          "tokens_per_second", "wall_clock_elapsed_seconds",
                                          "cuda_peak_allocated", "cuda_peak_reserved"}})
    return events


def _state_equal(left, right):
    if set(left) != set(right):
        return False
    for key in left:
        if set(left[key]) != set(right[key]):
            return False
        for item, value in left[key].items():
            other = right[key][item]
            if torch.is_tensor(value):
                if not torch.is_tensor(other) or not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


def _assert_checkpoints_equal(left, right):
    assert set(left["model"]) == set(right["model"])
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])
    assert _state_equal(left["optimizer"]["state"], right["optimizer"]["state"])
    assert left["completed_updates"] == right["completed_updates"]


def _snapshot_file_bytes(run_dir, update):
    snapshot_dir = run_dir / "snapshots"
    return ((snapshot_dir / f"update_{update:06d}.pt").read_bytes(),
            (snapshot_dir / f"update_{update:06d}.manifest.json").read_bytes())


def _role_tensor(manifest, payload, name, role):
    descriptor = manifest["parameters"][name]["roles"][role]
    if "alias_of" in descriptor:
        descriptor = manifest["parameters"][name]["roles"][descriptor["alias_of"]]
    return payload["tensors"][descriptor["tensor_id"]]


def test_staged_resume_matches_uninterrupted_and_snapshots_are_not_rewritten(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    snapshot_config = _snapshot_config(tmp_path)
    staged_dir = tmp_path / "run-staged"
    full_dir = tmp_path / "run-full"

    staged = train_platform.run(tiny_muon_config, staged_dir, to_device="cpu",
                                snapshot_config_path=snapshot_config, stop_at_update=2)
    assert staged["status"] == "paused_staged"
    before = {update: _snapshot_file_bytes(staged_dir, update) for update in (1, 2)}
    checkpoint_at_two = torch.load(staged_dir / "checkpoints/update_000002.pt",
                                   map_location="cpu", weights_only=False)
    assert checkpoint_at_two["completed_updates"] == 2

    resumed = train_platform.run(tiny_muon_config, staged_dir,
                                 resume=staged_dir / "checkpoints/latest.pt",
                                 to_device="cpu", snapshot_config_path=snapshot_config)
    assert resumed["status"] == "completed"
    full = train_platform.run(tiny_muon_config, full_dir, to_device="cpu",
                              snapshot_config_path=snapshot_config)
    assert full["status"] == "completed"

    _assert_checkpoints_equal(torch.load(staged_dir / "checkpoints/final.pt",
                                         map_location="cpu", weights_only=False),
                              torch.load(full_dir / "checkpoints/final.pt",
                                         map_location="cpu", weights_only=False))
    assert _train_events(staged_dir) == _train_events(full_dir)
    for update in (1, 2):
        assert _snapshot_file_bytes(staged_dir, update) == before[update]
    assert (staged_dir / "snapshots/update_000003.manifest.json").exists()

    # Snapshot and checkpoint at the same completed update must agree.
    manifest = json.loads((staged_dir / "snapshots/update_000002.manifest.json").read_text())
    payload = torch.load(staged_dir / "snapshots/update_000002.pt",
                         map_location="cpu", weights_only=True)
    for name, entry in manifest["parameters"].items():
        assert torch.equal(_role_tensor(manifest, payload, name, "W_after"),
                           checkpoint_at_two["model"][name])
        state_index = entry["optimizer_param_index"]
        assert torch.equal(_role_tensor(manifest, payload, name, "M_t_persisted"),
                           checkpoint_at_two["optimizer"]["state"][state_index]["muon_momentum"])

    # Same-update snapshots from the staged and uninterrupted trajectories match.
    for update in (1, 2, 3):
        staged_payload = torch.load(staged_dir / f"snapshots/update_{update:06d}.pt",
                                    map_location="cpu", weights_only=True)
        full_payload = torch.load(full_dir / f"snapshots/update_{update:06d}.pt",
                                  map_location="cpu", weights_only=True)
        assert staged_payload["tensors"].keys() == full_payload["tensors"].keys()
        assert all(torch.equal(staged_payload["tensors"][key], full_payload["tensors"][key])
                   for key in staged_payload["tensors"])


def test_keyboard_interrupt_keeps_last_complete_checkpoint_and_resume_matches(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    snapshot_config = _snapshot_config(tmp_path)
    interrupted_dir = tmp_path / "run-interrupted"
    full_dir = tmp_path / "run-full"

    from optim.muon_reference import ReferenceMuon
    real_step = ReferenceMuon.step
    calls = {"count": 0}

    def flaky_step(self, closure=None):
        calls["count"] += 1
        if calls["count"] == 2:
            # Interrupt after begin_update() and sampler.take(), before the
            # second optimizer step can complete.
            raise KeyboardInterrupt
        return real_step(self, closure)

    with monkeypatch.context() as patch:
        patch.setattr(ReferenceMuon, "step", flaky_step)
        summary = train_platform.run(tiny_muon_config, interrupted_dir,
                                     to_device="cpu", snapshot_config_path=snapshot_config)
    assert summary["status"] == "interrupted"
    latest = torch.load(interrupted_dir / "checkpoints/latest.pt",
                        map_location="cpu", weights_only=False)
    assert latest["completed_updates"] == 1
    assert (interrupted_dir / "snapshots/update_000001.manifest.json").exists()
    assert not (interrupted_dir / "snapshots/update_000002.manifest.json").exists()
    errors = (interrupted_dir / "snapshot_errors.jsonl").read_text()
    assert "incomplete_update_at_close" in errors

    resumed = train_platform.run(tiny_muon_config, interrupted_dir,
                                 resume=interrupted_dir / "checkpoints/latest.pt",
                                 to_device="cpu", snapshot_config_path=snapshot_config)
    assert resumed["status"] == "completed"
    full = train_platform.run(tiny_muon_config, full_dir, to_device="cpu",
                              snapshot_config_path=snapshot_config)
    assert full["status"] == "completed"
    _assert_checkpoints_equal(torch.load(interrupted_dir / "checkpoints/final.pt",
                                         map_location="cpu", weights_only=False),
                              torch.load(full_dir / "checkpoints/final.pt",
                                         map_location="cpu", weights_only=False))
    assert (interrupted_dir / "snapshots/update_000002.manifest.json").exists()
    assert (interrupted_dir / "snapshots/update_000003.manifest.json").exists()


def test_deleting_snapshots_does_not_change_training_resume(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    snapshot_config = _snapshot_config(tmp_path)
    staged_dir = tmp_path / "run-staged"
    full_dir = tmp_path / "run-full"
    train_platform.run(tiny_muon_config, staged_dir, to_device="cpu",
                       snapshot_config_path=snapshot_config, stop_at_update=2)
    # Snapshots are research artifacts, never resume inputs.
    for path in sorted((staged_dir / "snapshots").iterdir(), reverse=True):
        if path.is_file():
            path.unlink()
    resumed = train_platform.run(tiny_muon_config, staged_dir,
                                 resume=staged_dir / "checkpoints/latest.pt",
                                 to_device="cpu", snapshot_config_path=snapshot_config)
    full = train_platform.run(tiny_muon_config, full_dir, to_device="cpu",
                              snapshot_config_path=snapshot_config)
    assert resumed["status"] == full["status"] == "completed"
    _assert_checkpoints_equal(torch.load(staged_dir / "checkpoints/final.pt",
                                         map_location="cpu", weights_only=False),
                              torch.load(full_dir / "checkpoints/final.pt",
                                         map_location="cpu", weights_only=False))
    errors = (staged_dir / "snapshot_errors.jsonl").read_text()
    assert "missed_before_resume" in errors


def test_explicit_resume_config_must_hash_match(tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    original = _snapshot_config(tmp_path, name="original.json")
    run_dir = tmp_path / "run"
    train_platform.run(tiny_muon_config, run_dir, to_device="cpu",
                       snapshot_config_path=original, stop_at_update=1)
    different = _snapshot_config(tmp_path, updates=(1, 3), name="different.json")
    with pytest.raises(SnapshotConfigError, match="hash mismatch"):
        train_platform.run(tiny_muon_config, run_dir,
                           resume=run_dir / "checkpoints/latest.pt",
                           to_device="cpu", snapshot_config_path=different)
