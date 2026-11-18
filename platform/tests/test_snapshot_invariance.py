import json
from pathlib import Path

import torch

import train_platform
from optim.muon_reference import ReferenceMuon
from optim.muon_snapshot_collector import (
    MuonSnapshotCollector, SnapshotBeginContext, SnapshotConfig,
    SnapshotRunContext,
)


def _config(updates=(1, 2)):
    return SnapshotConfig(enabled=True, updates=updates, on_error="disable",
                          max_total_bytes=None, min_free_bytes=0)


def _records(names, shapes):
    return [{"name": name, "group": "muon", "shape": list(shape),
             "numel": int(shape[0] * shape[1]), "dtype": "torch.float32"}
            for name, shape in zip(names, shapes)]


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


def _optimizer_trajectory(tmp_path, *, enabled, state_simulation, nesterov, grads):
    names = ("transformer.h.0.attn.c_attn.weight", "transformer.h.0.mlp.w1.weight")
    shapes = ((4, 2), (2, 4))
    torch.manual_seed(123)
    initial = [torch.randn(*shape) for shape in shapes]
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    optimizer = ReferenceMuon(
        [{"params": params, "optimizer_group": "muon", "weight_decay": 0.1}],
        lr=0.01, state_simulation=state_simulation, muon_nesterov=nesterov)
    collector = None
    if enabled:
        config = _config()
        group = optimizer.param_groups[0]
        metadata = {key: value for key, value in group.items() if key != "params"}
        metadata["name"] = "reference_muon"
        context = SnapshotRunContext(
            run_id="invariance", segment_id=0,
            recipe_fingerprint="recipe", data_fingerprint="data",
            snapshot_config_hash=config.config_hash,
            model_config={"sequence_length": 8}, parameter_records=_records(names, shapes),
            muon_parameter_order=names,
            optimizer_param_index={name: index for index, name in enumerate(names)},
            optimizer_metadata=metadata,
            train_config={"micro_batch_size": 2, "accumulation_steps": 1},
            data_config={}, seeds={"seed": 0, "data_seed": 1, "algorithm_seed": 2},
            source={"remote": None, "upstream_commit": "abc", "dirty": False,
                    "diff": "", "provenance_status": "remote_missing",
                    "safe_directory_used": False},
            environment={}, device="cpu", precision={})
        collector = MuonSnapshotCollector(run_dir=tmp_path / "run", config=config, context=context)
        optimizer.set_snapshot_observer(collector.observe,
                                        {id(p): name for p, name in zip(params, names)})
    trajectory = []
    states = []
    for update, gradient_values in enumerate(grads, start=1):
        for parameter, gradient in zip(params, gradient_values):
            parameter.grad = gradient.clone()
        if collector is not None:
            optimizer.set_training_update(update)
            collector.begin_update(SnapshotBeginContext(
                update=update, segment_id=0, tokens_this_update=16,
                processed_target_tokens=(update - 1) * 16,
                learning_rate=0.01, pre_clip_grad_norm=1.0, clipped=False,
                grad_clip_norm=1.0, window_ids=(update,), micro_batch_window_ids=((update,),),
                sampler_epoch_before=0, sampler_offset_before=update - 1,
                sampler_epoch_after=0, sampler_offset_after=update))
        optimizer.step()
        if collector is not None:
            collector.finish_update(update=update, train_nll=1.0)
        trajectory.append([p.detach().clone() for p in params])
        states.append({key: {item: (value.clone() if torch.is_tensor(value) else value)
                            for item, value in state.items()}
                       for key, state in optimizer.state_dict()["state"].items()})
    return trajectory, states, torch.get_rng_state().clone(), torch.rand(8), optimizer


def test_collector_does_not_perturb_optimizer_state_parameters_or_rng(tmp_path):
    torch.manual_seed(77)
    grads = [[torch.randn(4, 2), torch.randn(2, 4)] for _ in range(2)]
    for simulation in ("none", "bf16_roundtrip"):
        for nesterov in (True, False):
            off = _optimizer_trajectory(tmp_path / f"off-{simulation}-{nesterov}",
                                        enabled=False, state_simulation=simulation,
                                        nesterov=nesterov, grads=grads)
            on = _optimizer_trajectory(tmp_path / f"on-{simulation}-{nesterov}",
                                       enabled=True, state_simulation=simulation,
                                       nesterov=nesterov, grads=grads)
            assert len(off[0]) == len(on[0]) == 2
            for step in range(2):
                assert all(torch.equal(a, b) for a, b in zip(off[0][step], on[0][step]))
                assert _state_equal(off[1][step], on[1][step])
            assert torch.equal(off[2], on[2])
            assert torch.equal(off[3], on[3])
            assert off[4].state_dict()["param_groups"] == on[4].state_dict()["param_groups"]
            # The collector must not become optimizer state.
            assert "_snapshot_observer" not in off[4].state_dict()
            assert "_snapshot_observer" not in on[4].state_dict()


def _fake_source(_root):
    return {"remote": None, "upstream_commit": "test-commit", "dirty": False,
            "diff": "", "provenance_status": "ok", "provenance_error": None,
            "safe_directory_used": False}


def _snapshot_config(tmp_path, updates=(1, 2, 3)):
    path = tmp_path / "snapshot.json"
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


def _load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_checkpoints_equal(left, right):
    assert set(left["model"]) == set(right["model"])
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])
    assert _state_equal(left["optimizer"]["state"], right["optimizer"]["state"])
    assert left["completed_updates"] == right["completed_updates"]


def test_collector_does_not_change_training_loop_or_final_checkpoint(tmp_path, monkeypatch, tiny_muon_config):
    monkeypatch.setattr(train_platform, "source_snapshot", _fake_source)
    off_dir = tmp_path / "run-off"
    on_dir = tmp_path / "run-on"
    snapshot_config = _snapshot_config(tmp_path)
    off_summary = train_platform.run(tiny_muon_config, off_dir, to_device="cpu")
    on_summary = train_platform.run(tiny_muon_config, on_dir, to_device="cpu",
                                    snapshot_config_path=snapshot_config)
    assert off_summary["status"] == on_summary["status"] == "completed"
    assert off_summary["completed_updates"] == on_summary["completed_updates"] == 3
    assert off_summary["processed_target_tokens"] == on_summary["processed_target_tokens"] == 48
    assert _train_events(off_dir) == _train_events(on_dir)
    _assert_checkpoints_equal(_load_checkpoint(off_dir / "checkpoints/latest.pt"),
                              _load_checkpoint(on_dir / "checkpoints/latest.pt"))
    manifests = sorted(p.name for p in (on_dir / "snapshots").glob("update_*.manifest.json"))
    assert manifests == ["update_000001.manifest.json",
                         "update_000002.manifest.json",
                         "update_000003.manifest.json"]
    assert not (off_dir / "snapshots").exists()
