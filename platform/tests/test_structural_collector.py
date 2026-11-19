import json
from pathlib import Path

import numpy as np
import pytest
import torch

import train_platform
from optim.muon_structural_collector import (
    RELATION_METRICS, RELATION_PAIRS, ROLE_INDEX, ROLE_ORDER, ROW_COL_ROLES,
    build_matrix_index, load_structural_config, row_col_schedule,
)


def _structural_config(tmp_path, *, enabled=True, chunk=2, warmup=2, every=1,
                       chunk_nodes=2):
    values = {
        "schema_version": 1, "enabled": enabled,
        "scalar_stats": {"enabled": enabled, "chunk_updates": chunk},
        "head_stats": {"enabled": enabled, "chunk_updates": chunk},
        "role_relations": {"enabled": enabled, "chunk_updates": chunk},
        "row_col_norms": {"enabled": enabled, "warmup_updates": warmup,
                          "every_updates": every, "chunk_nodes": chunk_nodes},
        "min_free_bytes": 0, "max_total_bytes": None,
    }
    path = tmp_path / "structural.json"
    path.write_text(json.dumps(values))
    return path


def _snapshot_config(tmp_path, updates=(3,), enabled=True):
    path = tmp_path / "snapshots.json"
    path.write_text(json.dumps({
        "schema_version": 1, "enabled": enabled, "updates": list(updates),
        "on_error": "disable", "max_total_bytes": None, "min_free_bytes": 0}))
    return path


def _run(tmp_path, cfg, *, name, structural, snapshots, resume=None,
         stop_at_update=None):
    summary = train_platform.run(
        cfg, tmp_path / name, resume=resume, to_device="cpu",
        stop_at_update=stop_at_update,
        snapshot_config_path=snapshots, structural_config_path=structural)
    return summary, tmp_path / name


def _load_group(run_dir: Path, group: str) -> dict:
    index = json.loads((run_dir / group / "index.json").read_text())
    merged = None
    for entry in index["chunks"]:
        with np.load(run_dir / group / entry["file"]) as archive:
            arrays = {key: archive[key] for key in archive.files}
        merged = arrays if merged is None else {
            key: np.concatenate([merged[key], arrays[key]]) for key in arrays}
    assert merged is not None
    order = np.argsort(merged["updates"])
    return {key: value[order] for key, value in merged.items()}


def _snapshot(run_dir: Path, update: int):
    manifest = json.loads(
        (run_dir / "snapshots" / f"update_{update:06d}.manifest.json").read_text())
    payload = torch.load(run_dir / "snapshots" / f"update_{update:06d}.pt",
                         map_location="cpu", weights_only=True)
    return manifest, payload


def _role_array(manifest, payload, view, role):
    source = view.get("source_name", view["name"])
    descriptor = manifest["parameters"][source]["roles"][role]
    if "alias_of" in descriptor:
        descriptor = manifest["parameters"][source]["roles"][descriptor["alias_of"]]
    array = payload["tensors"][descriptor["tensor_id"]].numpy()
    if view.get("derived"):
        array = array[view["row_start"]:view["row_end"]]
    return array


def _reference_stats(array):
    x = np.asarray(array, dtype=np.float64)
    n = x.size
    row = np.linalg.norm(x, axis=1)
    col = np.linalg.norm(x, axis=0)
    mean = x.mean()
    std = np.sqrt(max(0.0, (x * x).mean() - mean * mean))
    return np.array([
        mean, std, np.sqrt((x * x).mean()), np.linalg.norm(x), np.abs(x).max(),
        (x > 0).mean(), (x < 0).mean(), (x == 0).mean(),
        row.std() / row.mean() if row.mean() > 0 else np.nan,
        col.std() / col.mean() if col.mean() > 0 else np.nan,
    ])


def test_schedule_and_config_hash(tmp_path):
    assert len(row_col_schedule(8192, 128, 4)) == 2144
    assert row_col_schedule(8192, 128, 4)[:3] == (1, 2, 3)
    assert 128 in row_col_schedule(8192, 128, 4)
    assert 132 in row_col_schedule(8192, 128, 4)
    assert 129 not in row_col_schedule(8192, 128, 4)

    production = load_structural_config(
        Path(__file__).parents[1] / "configs/structural_emergence_8192.json")
    assert production.enabled
    assert production.config_hash == load_structural_config(
        Path(__file__).parents[1] / "configs/structural_emergence_8192.json").config_hash

    broken = json.loads(_structural_config(tmp_path).read_text())
    broken["unknown"] = 1
    (tmp_path / "broken.json").write_text(json.dumps(broken))
    with pytest.raises(Exception):
        load_structural_config(tmp_path / "broken.json")


def test_tier1_matches_snapshot_and_training_is_unchanged(tmp_path, tiny_muon_config):
    structural = _structural_config(tmp_path)
    snapshots = _snapshot_config(tmp_path, updates=(3,))
    summary, run_dir = _run(tmp_path, tiny_muon_config, name="run",
                            structural=structural, snapshots=snapshots)
    assert summary["completed_updates"] == 3

    scalar = _load_group(run_dir, "scalar_stats")
    heads = _load_group(run_dir, "head_stats")
    relations = _load_group(run_dir, "role_relations")
    rowcol = _load_group(run_dir, "row_col_norms")
    assert list(scalar["updates"]) == [1, 2, 3]
    assert list(rowcol["updates"]) == [1, 2, 3]

    manifest, payload = _snapshot(run_dir, 3)
    records = [{"name": name, **entry}
               for name, entry in manifest["parameters"].items()]
    index = build_matrix_index(manifest["model"]["config"], records)
    views = index["views"]
    row = int(np.where(scalar["updates"] == 3)[0][0])

    # Scalar statistics: fused QKV, every role.
    fused = views[0]
    expected = _reference_stats(_role_array(manifest, payload, fused, "G"))
    stored = scalar["stats"][row, 0, ROLE_INDEX["G"]]
    np.testing.assert_allclose(stored, expected.astype(np.float32), rtol=2e-4, atol=1e-5)

    # Delta_W is the actual parameter change.
    w_before = _role_array(manifest, payload, fused, "W_before")
    w_after = _role_array(manifest, payload, fused, "W_after")
    expected_delta = _reference_stats(w_after - w_before)
    stored_delta = scalar["stats"][row, 0, ROLE_INDEX["Delta_W"]]
    np.testing.assert_allclose(stored_delta, expected_delta.astype(np.float32),
                               rtol=2e-4, atol=1e-5)

    # Head energies partition each matrix: sum_h E = ||A||_F^2.
    for head_index, head in enumerate(index["head_matrices"]):
        view = views[head["view_id"]]
        for role in ("G", "U"):
            piece = _role_array(manifest, payload, view, role)
            total = float(np.sum(piece.astype(np.float64) ** 2))
            got = float(np.sum(heads["energy"][row, head_index, ROLE_INDEX[role]]))
            assert got == pytest.approx(total, rel=2e-4, abs=1e-6)
            if heads["defined"][row, head_index, ROLE_INDEX[role]]:
                assert float(np.sum(heads["relative"][row, head_index,
                                                     ROLE_INDEX[role]])) == pytest.approx(1.0, rel=1e-5)
            else:
                assert np.isnan(heads["relative"][row, head_index, ROLE_INDEX[role]]).all()

    # Relations: independently recompute one pair on one view.
    pair_index = [name for name, _, _ in RELATION_PAIRS].index("Delta_W_vs_U")
    view = views[1]
    a = _role_array(manifest, payload, view, "W_after") - \
        _role_array(manifest, payload, view, "W_before")
    b = _role_array(manifest, payload, view, "U")
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    dot, nx, ny = x @ y, np.linalg.norm(x), np.linalg.norm(y)
    alpha = dot / (nx * nx)
    expected = np.array([dot, dot / (nx * ny), nx / ny,
                         np.linalg.norm(x - y) / ny,
                         np.linalg.norm(y - alpha * x) / ny])
    stored = relations["relations"][row, view["view_id"], pair_index]
    np.testing.assert_allclose(stored, expected.astype(np.float32), rtol=5e-4, atol=1e-5)

    # Row/column norms match the snapshot tensors.
    rc_row = int(np.where(rowcol["updates"] == 3)[0][0])
    view = views[0]
    g = _role_array(manifest, payload, view, "G")
    role_index = ROW_COL_ROLES.index("G")
    offset, length = index["row_offsets"][0], index["row_lengths"][0]
    np.testing.assert_allclose(
        rowcol["row_norms"][rc_row, role_index, offset:offset + length],
        np.linalg.norm(g, axis=1), rtol=1e-5, atol=1e-6)
    col_offset, col_length = index["col_offsets"][0], index["col_lengths"][0]
    np.testing.assert_allclose(
        rowcol["col_norms"][rc_row, role_index, col_offset:col_offset + col_length],
        np.linalg.norm(g, axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        rowcol["qkv_col_norms"][rc_row, role_index, 0],
        np.linalg.norm(g[:index["model"]["n_embd"]], axis=0), rtol=1e-5, atol=1e-6)

    # The collectors never change the numerical trajectory.
    plain, plain_dir = _run(tmp_path, tiny_muon_config, name="plain",
                            structural=_structural_config(tmp_path, enabled=False),
                            snapshots=_snapshot_config(tmp_path, updates=(), enabled=False))
    assert plain["completed_updates"] == 3
    with torch.no_grad():
        left = torch.load(run_dir / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        right = torch.load(plain_dir / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    for key, value in left["model"].items():
        assert torch.equal(value, right["model"][key]), key


def test_resume_continues_without_duplicates(tmp_path, tiny_muon_config):
    structural = _structural_config(tmp_path)
    snapshots = _snapshot_config(tmp_path, updates=(3,))
    _run(tmp_path, tiny_muon_config, name="run", structural=structural,
         snapshots=snapshots, stop_at_update=2)
    first = _load_group(tmp_path / "run", "scalar_stats")
    assert list(first["updates"]) == [1, 2]

    summary, _ = _run(tmp_path, tiny_muon_config, name="run", structural=structural,
                      snapshots=snapshots, resume=tmp_path / "run/checkpoints/latest.pt")
    assert summary["completed_updates"] == 3
    merged = _load_group(tmp_path / "run", "scalar_stats")
    assert list(merged["updates"]) == [1, 2, 3]
    assert list(_load_group(tmp_path / "run", "row_col_norms")["updates"]) == [1, 2, 3]
