"""Rolling-checkpoint and full-run snapshot-config invariants."""
import json
from pathlib import Path

import pytest
import torch

import experiment_io
import train_platform
from optim.muon_snapshot_collector import (
    MuonSnapshotCollector, SnapshotRunContext, load_snapshot_config,
)

ROOT = Path(__file__).resolve().parents[1]
FULL_RUN_CONFIG = ROOT / "configs/muon_snapshot_full_9nodes_s0.json"
EXPECTED_FULL_RUN_UPDATES = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
EXPECTED_FULL_RUN_HASH = "d9e7c3df0cec1f048b27489f332c3f5dbe054fd0ddad5cb118138e9e83dcd73e"


def _fake_source(_root):
    return {"remote": None, "upstream_commit": "test-commit", "dirty": False,
            "diff": "", "provenance_status": "ok", "provenance_error": None,
            "safe_directory_used": False}


def _snapshot_config(tmp_path, updates):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({
        "schema_version": 1, "enabled": True, "updates": list(updates),
        "on_error": "disable", "max_total_bytes": 100_000_000,
        "min_free_bytes": 0,
    }))
    return path


def _checkpoint_names(run_dir):
    return sorted(path.name for path in (run_dir / "checkpoints").iterdir())


def _context(config):
    records = [{"name": "p0", "group": "muon", "shape": [2, 2],
                "numel": 4, "dtype": "torch.float32"}]
    return SnapshotRunContext(
        run_id="config-test", segment_id=0,
        recipe_fingerprint="recipe", data_fingerprint="data",
        snapshot_config_hash=config.config_hash,
        model_config={"sequence_length": 8}, parameter_records=records,
        muon_parameter_order=("p0",), optimizer_param_index={"p0": 0},
        optimizer_metadata={"name": "reference_muon"},
        train_config={}, data_config={}, seeds={}, source={},
        environment={}, device="cpu", precision={})


def test_full_run_config_selects_only_the_nine_post_stage1_landmarks(tmp_path):
    config = load_snapshot_config(FULL_RUN_CONFIG, total_updates=4096)
    assert config.updates == EXPECTED_FULL_RUN_UPDATES
    assert config.config_hash == EXPECTED_FULL_RUN_HASH
    assert config.max_total_bytes == 4 * 1024**3
    assert config.min_free_bytes == 2 * 1024**3

    collector = MuonSnapshotCollector(
        run_dir=tmp_path / "run", config=config, context=_context(config))
    triggered = {update for update in range(1, 4097) if collector.should_collect(update)}
    assert triggered == set(EXPECTED_FULL_RUN_UPDATES)
    # The stage-1 landmarks must not be collected again.
    assert not triggered & {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128}


def test_each_configured_landmark_writes_a_tensor_file_and_manifest(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    run_dir = tmp_path / "run"
    config = _snapshot_config(tmp_path, (1, 3))
    summary = train_platform.run(tiny_muon_config, run_dir, to_device="cpu",
                                 snapshot_config_path=config)
    assert summary["status"] == "completed"
    assert summary["completed_updates"] == 3

    snapshots = run_dir / "snapshots"
    assert sorted(path.name for path in snapshots.glob("update_*.pt")) == [
        "update_000001.pt", "update_000003.pt"]
    assert sorted(path.name for path in snapshots.glob("update_*.manifest.json")) == [
        "update_000001.manifest.json", "update_000003.manifest.json"]
    assert not (snapshots / "update_000002.pt").exists()

    for update in (1, 3):
        manifest = json.loads(
            (snapshots / f"update_{update:06d}.manifest.json").read_text())
        payload = torch.load(snapshots / f"update_{update:06d}.pt",
                             map_location="cpu", weights_only=True)
        assert manifest["status"] == "committed"
        assert manifest["update"] == update
        assert manifest["identities"]["snapshot_config_hash"] == load_snapshot_config(
            config, total_updates=3).config_hash
        assert manifest["integrity"]["expected_parameter_count"] == 5
        assert manifest["integrity"]["observed_parameter_count"] == 5
        assert manifest["integrity"]["expected_role_slots"] == 40
        assert manifest["integrity"]["observed_role_slots"] == 40
        assert manifest["integrity"]["unique_tensor_count"] == 35
        assert manifest["integrity"]["alias_count"] == 5
        assert len(payload["tensors"]) == 35


def test_rolling_checkpoint_keeps_only_latest_and_no_extra_copies(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)

    # Initial update 0 saves latest.pt only; the explicit stage gate leaves
    # exactly one rolling file, overwritten in place by the last update.
    staged_dir = tmp_path / "run-staged"
    staged = train_platform.run(tiny_muon_config, staged_dir, to_device="cpu",
                                stop_at_update=1)
    assert staged["status"] == "paused_staged"
    assert _checkpoint_names(staged_dir) == ["latest.pt"]
    staged_checkpoint = torch.load(staged_dir / "checkpoints/latest.pt",
                                   map_location="cpu", weights_only=False)
    assert staged_checkpoint["completed_updates"] == 1

    full_dir = tmp_path / "run-full"
    full = train_platform.run(tiny_muon_config, full_dir, to_device="cpu")
    assert full["status"] == "completed"
    assert _checkpoint_names(full_dir) == ["latest.pt"]
    checkpoint = torch.load(full_dir / "checkpoints/latest.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["completed_updates"] == 3
    for name in ("initial.pt", "final.pt", "update_000000.pt",
                 "update_000001.pt", "update_000003.pt"):
        assert not (full_dir / "checkpoints" / name).exists(), name
    assert not list((full_dir / "checkpoints").glob("*.tmp"))

    events = [json.loads(line) for line in
              (full_dir / "metrics.jsonl").read_text().splitlines()]
    checkpoint_events = [event for event in events if event["event_type"] == "checkpoint"]
    assert [event["completed_updates"] for event in checkpoint_events] == [1, 2, 3]
    assert all(event["path"] == "checkpoints/latest.pt"
               and "landmark_path" not in event for event in checkpoint_events)


def test_every_checkpoint_write_targets_the_same_latest_file(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    real_save = train_platform.atomic_torch_save
    targets = []

    def traced_save(value, path):
        targets.append(Path(path).name)
        return real_save(value, path)

    monkeypatch.setattr(train_platform, "atomic_torch_save", traced_save)
    run_dir = tmp_path / "run"
    summary = train_platform.run(tiny_muon_config, run_dir, to_device="cpu")
    assert summary["status"] == "completed"
    # update 0 plus updates 1..3, each overwriting the same file.
    assert targets == ["latest.pt"] * 4
    assert _checkpoint_names(run_dir) == ["latest.pt"]
    checkpoint = torch.load(run_dir / "checkpoints/latest.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["completed_updates"] == 3


def test_atomic_checkpoint_failure_preserves_the_previous_latest(
        tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    run_dir = tmp_path / "run"
    train_platform.run(tiny_muon_config, run_dir, to_device="cpu",
                       stop_at_update=1)
    latest = run_dir / "checkpoints/latest.pt"
    before = latest.read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(experiment_io.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        train_platform.run(tiny_muon_config, run_dir, to_device="cpu",
                           resume=latest, stop_at_update=2)

    assert latest.read_bytes() == before
    restored = torch.load(latest, map_location="cpu", weights_only=False)
    assert restored["completed_updates"] == 1
    assert _checkpoint_names(run_dir) == ["latest.pt"]
