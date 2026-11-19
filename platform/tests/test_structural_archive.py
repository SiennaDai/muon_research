import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from optim.muon_structural_collector import _GroupStore

SCRIPT = Path(__file__).parents[1] / "scripts/archive_structural_run.py"
spec = importlib.util.spec_from_file_location("archive_structural_run", SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def _make_run(tmp_path):
    run = tmp_path / "run"
    store = _GroupStore(run, "scalar_stats")
    identity = {"run_id": "run"}
    store.open(identity=identity)
    for index in range(2):
        arrays = {"updates": np.array([index * 2 + 1, index * 2 + 2], dtype=np.float32),
                  "stats": np.zeros((2, 48, 8, 10), dtype=np.float32),
                  "aux": np.zeros((2, 12), dtype=np.float32)}
        store.append_chunk(arrays, identity=identity)
    snapshots = run / "snapshots"
    snapshots.mkdir()
    (snapshots / "update_000001.pt").write_bytes(b"snapshot-one")
    (snapshots / "update_000001.manifest.json").write_text("{}")
    (run / "metrics.jsonl").write_text("{}\n")
    return run, identity


def test_archive_and_prune(tmp_path):
    run, identity = _make_run(tmp_path)
    manifest_path = archive.build_archive(run, tmp_path / "out", 5000, None, None)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["shards"]) >= 3
    for shard in manifest["shards"]:
        path = tmp_path / "out" / shard["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
    assert manifest["coverage"]["scalar_stats"] == [1, 4]
    assert manifest["coverage"]["snapshots"] == [1, 1]

    with pytest.raises(SystemExit):
        archive.prune(run, manifest_path)

    manifest["verified"] = True
    manifest_path.write_text(json.dumps(manifest))
    removed = archive.prune(run, manifest_path)
    assert removed == 4
    assert not (run / "scalar_stats/scalar-00000.npz").exists()
    assert (run / "metrics.jsonl").exists()
    index = json.loads((run / "scalar_stats/index.json").read_text())
    assert all(entry.get("pruned") for entry in index["chunks"])

    reopened = _GroupStore(run, "scalar_stats")
    reopened.open(identity=identity)
    reopened.append_chunk(
        {"updates": np.array([5.0], dtype=np.float32),
         "stats": np.zeros((1, 48, 8, 10), dtype=np.float32),
         "aux": np.zeros((1, 12), dtype=np.float32)}, identity=identity)
    assert reopened.chunks[-1]["file"] == "scalar-00002.npz"
