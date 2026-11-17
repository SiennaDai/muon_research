import json
import math
import os
import types
from pathlib import Path

import pytest
import torch

import optim.muon_reference as muon_reference
import optim.muon_snapshot_collector as snapshot_module
from optim.muon_reference import ReferenceMuon
from optim.muon_snapshot_collector import (
    ROLE_ORDER, MuonSnapshotCollector, SnapshotBeginContext, SnapshotConfig,
    SnapshotConfigError, SnapshotRunContext, load_snapshot_config,
    resolve_snapshot_config, save_snapshot_config,
)


def _config(updates=(1,), **overrides):
    values = dict(schema_version=1, enabled=True, updates=tuple(updates),
                  on_error="disable", max_total_bytes=None, min_free_bytes=0)
    values.update(overrides)
    return SnapshotConfig(**values)


def _records(names, shapes):
    return [{"name": name, "group": "muon", "shape": list(shape),
             "numel": int(shape[0] * shape[1]), "dtype": "torch.float32"}
            for name, shape in zip(names, shapes)]


def _setup(tmp_path, names, shapes, *, updates=(1,), state_simulation="none",
           nesterov=True, min_free_bytes=0, max_total_bytes=None):
    config = _config(updates=updates, min_free_bytes=min_free_bytes,
                     max_total_bytes=max_total_bytes)
    params = [torch.nn.Parameter(torch.randn(*shape)) for shape in shapes]
    optimizer = ReferenceMuon(
        [{"params": params, "optimizer_group": "muon", "weight_decay": 0.0}],
        lr=0.01, state_simulation=state_simulation, muon_nesterov=nesterov)
    group = optimizer.param_groups[0]
    optimizer_metadata = {key: value for key, value in group.items() if key != "params"}
    optimizer_metadata["name"] = "reference_muon"
    context = SnapshotRunContext(
        run_id="collector-test", segment_id=0,
        recipe_fingerprint="recipe", data_fingerprint="data",
        snapshot_config_hash=config.config_hash,
        model_config={"sequence_length": 8},
        parameter_records=_records(names, shapes),
        muon_parameter_order=tuple(names),
        optimizer_param_index={name: index for index, name in enumerate(names)},
        optimizer_metadata=optimizer_metadata,
        train_config={"micro_batch_size": 2, "accumulation_steps": 1},
        data_config={}, seeds={"seed": 0, "data_seed": 1, "algorithm_seed": 2},
        source={"remote": None, "upstream_commit": "abc", "dirty": False,
                "diff": "", "provenance_status": "remote_missing",
                "safe_directory_used": False},
        environment={"torch": torch.__version__}, device="cpu",
        precision={"compute": "fp32"})
    collector = MuonSnapshotCollector(run_dir=tmp_path / "run", config=config, context=context)
    optimizer.set_snapshot_observer(collector.observe,
                                    {id(p): name for p, name in zip(params, names)})
    return config, params, optimizer, collector


def _begin(collector, update=1, *, processed=0, tokens=16):
    collector.begin_update(SnapshotBeginContext(
        update=update, segment_id=0, tokens_this_update=tokens,
        processed_target_tokens=processed + tokens,
        learning_rate=0.001, pre_clip_grad_norm=1.0, clipped=False,
        grad_clip_norm=1.0, window_ids=(17, 19),
        micro_batch_window_ids=((17, 19),),
        sampler_epoch_before=0, sampler_offset_before=0,
        sampler_epoch_after=0, sampler_offset_after=2))


def _load_snapshot(collector, update=1):
    manifest = json.loads((collector.snapshots_dir / f"update_{update:06d}.manifest.json").read_text())
    payload = torch.load(collector.snapshots_dir / f"update_{update:06d}.pt",
                         map_location="cpu", weights_only=True)
    return manifest, payload


def _role_tensor(manifest, payload, name, role):
    descriptor = manifest["parameters"][name]["roles"][role]
    if "alias_of" in descriptor:
        descriptor = manifest["parameters"][name]["roles"][descriptor["alias_of"]]
    return payload["tensors"][descriptor["tensor_id"]]


def test_snapshot_contains_all_eight_roles_and_exact_tensors(tmp_path, monkeypatch):
    torch.manual_seed(3)
    names = ("transformer.h.0.attn.c_attn.weight", "transformer.h.0.mlp.w1.weight")
    shapes = ((4, 2), (2, 4))
    config, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,))

    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.randn_like(p)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    clipped_grads = [p.grad.detach().clone() for p in params]
    momentum_prev = [optimizer.state[p].get("muon_momentum", torch.zeros_like(p)).detach().clone()
                     for p in params]

    original_ns = muon_reference.zeropower_newton_schulz
    seen_ns = []

    def traced_ns(matrix, *args, **kwargs):
        output = original_ns(matrix, *args, **kwargs)
        seen_ns.append((matrix.detach().clone(), output.detach().clone()))
        return output

    monkeypatch.setattr(muon_reference, "zeropower_newton_schulz", traced_ns)
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    after = [p.detach().clone() for p in params]
    collector.finish_update(update=1, train_nll=1.25)

    manifest, payload = _load_snapshot(collector)
    assert manifest["format"] == "muon_snapshot_manifest"
    assert manifest["status"] == "committed"
    assert manifest["update"] == 1
    assert manifest["run_id"] == "collector-test"
    assert set(manifest["parameters"]) == set(names)
    assert manifest["identities"]["snapshot_config_hash"] == config.config_hash
    assert manifest["seeds"] == {"seed": 0, "data_seed": 1, "algorithm_seed": 2}
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["environment"]["precision"] == {"compute": "fp32"}
    assert manifest["training"]["train_nll_pre_update"] == pytest.approx(1.25)
    assert manifest["training"]["pre_clip_grad_norm"] == pytest.approx(1.0)
    assert manifest["data"]["window_ids"] == [17, 19]
    assert manifest["data"]["micro_batch_window_ids"] == [[17, 19]]
    assert manifest["data"]["sampler_offset_before"] == 0
    assert manifest["data"]["sampler_offset_after"] == 2
    assert manifest["integrity"]["expected_parameter_count"] == 2
    assert manifest["integrity"]["observed_parameter_count"] == 2
    assert manifest["integrity"]["expected_role_slots"] == 16
    assert manifest["integrity"]["observed_role_slots"] == 16
    assert manifest["integrity"]["unique_tensor_count"] == 14
    assert manifest["integrity"]["alias_count"] == 2

    for index, name in enumerate(names):
        entry = manifest["parameters"][name]
        assert set(entry["roles"]) == set(ROLE_ORDER)
        assert entry["shape"] == list(shapes[index])
        assert entry["dtype"] == "float32"
        assert entry["first_step"] is True
        assert torch.equal(_role_tensor(manifest, payload, name, "W_before"), before[index])
        assert torch.equal(_role_tensor(manifest, payload, name, "W_after"), after[index])
        assert torch.equal(_role_tensor(manifest, payload, name, "G"), clipped_grads[index])
        assert torch.equal(_role_tensor(manifest, payload, name, "M_prev"), momentum_prev[index])
        assert torch.equal(_role_tensor(manifest, payload, name, "D"), seen_ns[index][0])
        assert torch.equal(_role_tensor(manifest, payload, name, "U"), seen_ns[index][1])
        expected_candidate = momentum_prev[index] * 0.95 + clipped_grads[index]
        assert torch.equal(_role_tensor(manifest, payload, name, "M_t_candidate"), expected_candidate)
        # state_simulation="none" makes persisted an alias of candidate.
        assert entry["roles"]["M_t_persisted"] == {"alias_of": "M_t_candidate"}
        for role, descriptor in entry["roles"].items():
            if "alias_of" in descriptor:
                continue
            assert descriptor["shape"] == list(shapes[index])
            assert descriptor["dtype"] == "float32"
            assert descriptor["nbytes"] == descriptor["numel"] * 4
            assert descriptor["sha256"] == snapshot_module.sha256_tensor(
                payload["tensors"][descriptor["tensor_id"]])
    assert manifest["collector"]["snapshot_bytes"] == os.path.getsize(
        collector.snapshots_dir / "update_000001.pt")
    assert manifest["collector"]["write_seconds"] > 0
    assert manifest["collector"]["checksum_seconds"] > 0


def test_second_step_prev_momentum_and_candidate_are_exact(tmp_path):
    torch.manual_seed(4)
    names = ("transformer.h.0.attn.c_proj.weight",)
    shapes = ((3, 3),)
    _, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1, 2))

    for update in (1, 2):
        for p in params:
            p.grad = torch.randn_like(p)
        prev = optimizer.state[params[0]].get("muon_momentum", torch.zeros_like(params[0])).detach().clone()
        expected_candidate = prev * 0.95 + params[0].grad
        optimizer.set_training_update(update)
        _begin(collector, update=update, processed=(update - 1) * 16)
        optimizer.step()
        collector.finish_update(update=update, train_nll=1.0)
        manifest, payload = _load_snapshot(collector, update)
        assert torch.equal(_role_tensor(manifest, payload, names[0], "M_prev"), prev)
        assert torch.equal(_role_tensor(manifest, payload, names[0], "M_t_candidate"),
                           expected_candidate)
        if update == 1:
            assert float(prev.abs().sum()) == 0.0
            assert manifest["parameters"][names[0]]["first_step"] is True
        else:
            assert manifest["parameters"][names[0]]["first_step"] is False


def test_bf16_persisted_momentum_is_stored_separately(tmp_path):
    torch.manual_seed(5)
    names = ("transformer.h.0.mlp.c_proj.weight",)
    shapes = ((2, 5),)
    _, params, optimizer, collector = _setup(
        tmp_path, names, shapes, updates=(1,), state_simulation="bf16_roundtrip")
    params[0].grad = torch.randn_like(params[0])
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    manifest, payload = _load_snapshot(collector)
    entry = manifest["parameters"][names[0]]
    assert "tensor_id" in entry["roles"]["M_t_persisted"]
    assert "alias_of" not in entry["roles"]["M_t_persisted"]
    candidate = _role_tensor(manifest, payload, names[0], "M_t_candidate")
    persisted = _role_tensor(manifest, payload, names[0], "M_t_persisted")
    assert not torch.equal(candidate, persisted)
    assert torch.equal(persisted, candidate.bfloat16().float())
    assert manifest["integrity"]["unique_tensor_count"] == 8
    assert manifest["integrity"]["alias_count"] == 0


def test_nesterov_false_aliases_direction_to_candidate(tmp_path):
    torch.manual_seed(6)
    names = ("transformer.h.0.attn.c_attn.weight",)
    shapes = ((2, 3),)
    _, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,), nesterov=False)
    params[0].grad = torch.randn_like(params[0])
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    manifest, payload = _load_snapshot(collector)
    entry = manifest["parameters"][names[0]]
    assert entry["roles"]["D"] == {"alias_of": "M_t_candidate"}
    assert any(alias["alias_role"] == "D" for alias in manifest["aliases"])
    assert manifest["integrity"]["unique_tensor_count"] == 6


def test_cpu_clone_is_independent_of_later_mutation(tmp_path):
    torch.manual_seed(7)
    names = ("transformer.h.0.mlp.w1.weight",)
    shapes = ((3, 2),)
    _, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,))
    params[0].grad = torch.randn_like(params[0])
    expected_before = params[0].detach().clone()
    expected_grad = params[0].grad.detach().clone()
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    expected_after = params[0].detach().clone()
    # Mutate every optimizer-owned object after observe() has copied it.
    params[0].data.add_(10.0)
    optimizer.state[params[0]]["muon_momentum"].add_(10.0)
    if params[0].grad is not None:
        params[0].grad.add_(10.0)
    collector.finish_update(update=1, train_nll=1.0)
    manifest, payload = _load_snapshot(collector)
    assert torch.equal(_role_tensor(manifest, payload, names[0], "W_before"), expected_before)
    assert torch.equal(_role_tensor(manifest, payload, names[0], "W_after"), expected_after)
    assert torch.equal(_role_tensor(manifest, payload, names[0], "G"), expected_grad)


def test_thirty_muon_matrices_are_required_and_collected(tmp_path):
    suffixes = ("attn.c_attn", "attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj")
    names = tuple(f"transformer.h.{layer}.{suffix}.weight"
                  for layer in range(6) for suffix in suffixes)
    shapes = tuple((2, 2) for _ in names)
    assert len(names) == 30
    _, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,))
    for p in params:
        p.grad = torch.randn_like(p)
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    manifest, _ = _load_snapshot(collector)
    assert manifest["model"]["muon_parameter_count"] == 30
    assert manifest["integrity"]["expected_parameter_count"] == 30
    assert manifest["integrity"]["observed_parameter_count"] == 30
    assert manifest["integrity"]["expected_role_slots"] == 240
    assert manifest["integrity"]["observed_role_slots"] == 240
    assert manifest["integrity"]["unique_tensor_count"] == 210
    assert manifest["integrity"]["alias_count"] == 30
    assert set(manifest["parameters"]) == set(names)
    assert manifest["collector"]["clone_seconds"] >= 0
    assert manifest["collector"]["checksum_seconds"] >= 0
    assert manifest["collector"]["write_seconds"] > 0
    assert manifest["tensors_file"]["nbytes"] > manifest["tensors_file"]["payload_nbytes"]


def test_missing_parameter_prevents_manifest_and_disables(tmp_path):
    names = ("p0", "p1")
    shapes = ((2, 2), (2, 2))
    _, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1, 2))
    _begin(collector)
    # Only one parameter is observed.
    collector.observe(update=1, parameter_name=names[0],
                      W_before=params[0], G=torch.zeros_like(params[0]),
                      M_prev=torch.zeros_like(params[0]),
                      M_t_candidate=torch.zeros_like(params[0]),
                      M_t_persisted=torch.zeros_like(params[0]),
                      D=torch.zeros_like(params[0]), U=torch.zeros_like(params[0]),
                      W_after=torch.zeros_like(params[0]), first_step=True)
    collector.finish_update(update=1, train_nll=1.0)
    assert collector.disabled is True
    assert not (tmp_path / "run" / "snapshots" / "update_000001.manifest.json").exists()
    errors = (tmp_path / "run" / "snapshot_errors.jsonl").read_text()
    assert "incomplete" in errors or "finish_exception" in errors
    assert collector.should_collect(2) is False


def test_duplicate_begin_update_disables_collector(tmp_path):
    names = ("p0",)
    shapes = ((2, 2),)
    _, _, _, collector = _setup(tmp_path, names, shapes, updates=(1,))
    _begin(collector)
    _begin(collector)
    assert collector.disabled is True
    assert "duplicate_begin_update" in (tmp_path / "run" / "snapshot_errors.jsonl").read_text()


def test_disk_insufficient_skips_without_disabling(tmp_path, monkeypatch):
    names = ("p0",)
    shapes = ((2, 2),)
    _, params, optimizer, collector = _setup(
        tmp_path, names, shapes, updates=(1, 2), min_free_bytes=0)
    monkeypatch.setattr(snapshot_module.shutil, "disk_usage",
                        lambda _path: types.SimpleNamespace(free=0))
    params[0].grad = torch.randn_like(params[0])
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    assert collector.disabled is False
    assert not (tmp_path / "run" / "snapshots" / "update_000001.manifest.json").exists()
    assert "skipped_no_space" in (tmp_path / "run" / "snapshot_errors.jsonl").read_text()
    assert collector.should_collect(2) is True


def test_max_total_bytes_skips_without_disabling(tmp_path):
    names = ("p0",)
    shapes = ((2, 2),)
    _, params, optimizer, collector = _setup(
        tmp_path, names, shapes, updates=(1, 2), max_total_bytes=1)
    params[0].grad = torch.randn_like(params[0])
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    assert collector.disabled is False
    assert not (tmp_path / "run" / "snapshots" / "update_000001.manifest.json").exists()
    assert "skipped_max_total_bytes" in (tmp_path / "run" / "snapshot_errors.jsonl").read_text()
    assert collector.should_collect(2) is True


def test_write_exception_disables_but_training_step_completed(tmp_path, monkeypatch):
    names = ("p0",)
    shapes = ((2, 2),)
    _, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,))
    before = params[0].detach().clone()
    params[0].grad = torch.randn_like(params[0])

    def fail_write(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(snapshot_module, "atomic_torch_save_with_digest", fail_write)
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    assert not torch.equal(before, params[0])
    collector.finish_update(update=1, train_nll=1.0)
    assert collector.disabled is True
    assert not (tmp_path / "run" / "snapshots" / "update_000001.manifest.json").exists()
    assert "finish_exception" in (tmp_path / "run" / "snapshot_errors.jsonl").read_text()


def test_corrupt_snapshot_is_a_conflict_not_overwritten(tmp_path):
    names = ("p0",)
    shapes = ((2, 2),)
    config, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,))
    params[0].grad = torch.randn_like(params[0])
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    pt_path = tmp_path / "run" / "snapshots" / "update_000001.pt"
    original = pt_path.read_bytes()
    pt_path.write_bytes(b"corrupt")
    # Rebuild the collector with the same context; scanning must flag conflict.
    context = SnapshotRunContext(
        run_id="collector-test", segment_id=0,
        recipe_fingerprint="recipe", data_fingerprint="data",
        snapshot_config_hash=config.config_hash,
        model_config={"sequence_length": 8}, parameter_records=_records(names, shapes),
        muon_parameter_order=names, optimizer_param_index={names[0]: 0},
        optimizer_metadata={"name": "reference_muon"},
        train_config={"micro_batch_size": 2, "accumulation_steps": 1}, data_config={},
        seeds={"seed": 0, "data_seed": 1, "algorithm_seed": 2},
        source={"remote": None, "upstream_commit": "abc", "dirty": False, "diff": "",
                "provenance_status": "remote_missing", "safe_directory_used": False},
        environment={}, device="cpu", precision={})
    scanned = MuonSnapshotCollector(run_dir=tmp_path / "run", config=config, context=context)
    assert scanned.committed_updates() == ()
    assert scanned.conflicted_updates() == (1,)
    assert scanned.should_collect(1) is False
    assert pt_path.read_bytes() == b"corrupt"


def test_missing_manifest_moves_pt_to_orphaned(tmp_path):
    names = ("p0",)
    shapes = ((2, 2),)
    config, params, optimizer, collector = _setup(tmp_path, names, shapes, updates=(1,))
    params[0].grad = torch.randn_like(params[0])
    optimizer.set_training_update(1)
    _begin(collector)
    optimizer.step()
    collector.finish_update(update=1, train_nll=1.0)
    manifest_path = tmp_path / "run" / "snapshots" / "update_000001.manifest.json"
    manifest_path.unlink()
    context = SnapshotRunContext(
        run_id="collector-test", segment_id=0,
        recipe_fingerprint="recipe", data_fingerprint="data",
        snapshot_config_hash=config.config_hash,
        model_config={"sequence_length": 8}, parameter_records=_records(names, shapes),
        muon_parameter_order=names, optimizer_param_index={names[0]: 0},
        optimizer_metadata={"name": "reference_muon"},
        train_config={"micro_batch_size": 2, "accumulation_steps": 1}, data_config={},
        seeds={"seed": 0, "data_seed": 1, "algorithm_seed": 2},
        source={"remote": None, "upstream_commit": "abc", "dirty": False, "diff": "",
                "provenance_status": "remote_missing", "safe_directory_used": False},
        environment={}, device="cpu", precision={})
    scanned = MuonSnapshotCollector(run_dir=tmp_path / "run", config=config, context=context)
    assert scanned.committed_updates() == ()
    assert not (tmp_path / "run" / "snapshots" / "update_000001.pt").exists()
    assert (tmp_path / "run" / "snapshots" / "orphaned" / "update_000001.pt").exists()


def test_config_validation_and_resume_inheritance(tmp_path):
    valid = {"schema_version": 1, "enabled": True, "updates": [1, 3],
             "on_error": "disable", "max_total_bytes": None, "min_free_bytes": 0}
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(valid))
    config = load_snapshot_config(path, total_updates=4)
    assert config.updates == (1, 3)
    assert config.config_hash == load_snapshot_config(path, total_updates=4).config_hash

    invalid_cases = [
        {**valid, "updates": [3, 1]},        # not increasing
        {**valid, "updates": [1, 1]},        # duplicate
        {**valid, "updates": [0]},           # not positive
        {**valid, "updates": [5]},           # beyond total
        {**valid, "enabled": False},         # updates with disabled
        {**valid, "on_error": "raise"},      # unsupported
        {**valid, "unknown": 1},             # unknown field
        {**valid, "min_free_bytes": -1},     # negative
    ]
    for index, payload in enumerate(invalid_cases):
        bad = tmp_path / f"bad-{index}.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(SnapshotConfigError):
            load_snapshot_config(bad, total_updates=4)

    run_dir = tmp_path / "run"
    save_snapshot_config(run_dir, config)
    inherited = resolve_snapshot_config(None, run_dir=run_dir, total_updates=4, resume=True)
    assert inherited.config_hash == config.config_hash
    explicit = resolve_snapshot_config(path, run_dir=run_dir, total_updates=4, resume=True)
    assert explicit.config_hash == config.config_hash
    different = tmp_path / "different.json"
    different.write_text(json.dumps({**valid, "updates": [2]}))
    with pytest.raises(SnapshotConfigError):
        resolve_snapshot_config(different, run_dir=run_dir, total_updates=4, resume=True)
    new_dir = tmp_path / "new-run"
    disabled = resolve_snapshot_config(None, run_dir=new_dir, total_updates=4, resume=False)
    assert disabled.enabled is False
    assert (new_dir / "snapshot_config.json").exists()
