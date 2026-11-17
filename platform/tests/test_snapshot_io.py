import json
import math
from pathlib import Path

import pytest
import torch

from experiment_io import (
    atomic_torch_save_with_digest, atomic_write_json, fsync_directory, fsync_file,
    sha256_file, sha256_tensor,
)


def test_sha256_tensor_is_little_endian_fp32_and_deterministic():
    tensor = torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float32)
    expected = torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float32)
    assert sha256_tensor(tensor) == sha256_tensor(expected)
    assert sha256_tensor(tensor) != sha256_tensor(tensor + 1)


def test_atomic_torch_save_with_digest_matches_file_hash(tmp_path):
    path = tmp_path / "payload.pt"
    result = atomic_torch_save_with_digest({"tensor": torch.ones(3, 2)}, path)
    assert path.exists()
    assert result["nbytes"] == path.stat().st_size
    assert result["sha256"] == sha256_file(path)
    assert not list(tmp_path.glob("*.tmp"))
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["tensor"], torch.ones(3, 2))


def test_atomic_write_json_is_strict_and_atomic(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"a": 1, "nested": [True, None, "x"]})
    assert json.loads(path.read_text()) == {"a": 1, "nested": [True, None, "x"]}
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"value": math.nan})


def test_fsync_helpers_do_not_raise_for_temp_files(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("x")
    fsync_file(path)
    fsync_directory(tmp_path)
    fsync_directory(tmp_path / "missing")
