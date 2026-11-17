import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic_config():
    return json.loads((Path(__file__).parents[1] / "recipes/diagnostic_cpu.json").read_text())


@pytest.fixture
def tiny_muon_config(tmp_path):
    """A one-layer, CPU-only ReferenceMuon recipe with synthetic token data.

    The model is deliberately tiny: five Muon matrices, 48 training tokens and
    three updates.  It exercises the real run loop without downloading data or
    creating a large checkpoint.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def write_split(name, count):
        path = data_dir / f"{name}.bin"
        values = np.arange(count, dtype="<u2") % 64
        values.tofile(path)
        return {"file": path.name, "tokens": int(count),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest = {
        "schema_version": 1,
        "fingerprint": "tiny-snapshot-data",
        "dtype": "uint16",
        "endianness": "little",
        "max_token_id": 63,
        "splits": {"train": write_split("train", 256),
                   "validation": write_split("validation", 65)},
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    recipe = {
        "experiment": {"name": "tiny_muon_snapshot", "protocol_id": "tiny",
                       "seed": 0, "data_seed": 1, "algorithm_seed": 2},
        "model": {"family": "llama", "n_layer": 1, "n_embd": 32, "n_head": 2,
                  "ffn_dim": 88, "multiple_of": 8, "vocab_size": 64,
                  "sequence_length": 8, "dropout": 0, "bias": False,
                  "tie_embeddings": True, "init_std": 0.02,
                  "rmsnorm_eps": 1e-5, "rope_theta": 10000},
        "data": {"manifest": str(manifest_path), "train_split": "train",
                 "validation_split": "validation", "allow_repeated_epochs": False},
        "train": {"target_tokens": 48, "micro_batch_size": 2,
                  "accumulation_steps": 1, "grad_clip_norm": 1.0},
        "optimizer": {"name": "reference_muon", "lr": 0.001,
                      "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1,
                      "fused": False, "foreach": False,
                      "state_simulation": "none", "muon_momentum": 0.95,
                      "muon_nesterov": True, "muon_ns_steps": 5,
                      "muon_ns_coefficients": [3.4445, -4.775, 2.0315],
                      "muon_eps": 1e-7},
        "schedule": {"name": "cosine", "warmup_updates": 1,
                     "final_lr_ratio": 0.1, "total_updates": 3},
        "precision": {"compute": "fp32", "parameter_dtype": "fp32",
                      "gradient_dtype": "fp32", "tf32": False,
                      "attention_backend": "math", "deterministic": True,
                      "compile": False},
        "eval": {"every_updates": 0, "max_target_tokens": 8, "batch_size": 1,
                 "compute": "fp32", "attention_backend": "math"},
        "logging": {"every_updates": 1, "diagnostics": False, "wandb": False},
        "checkpoint": {"every_updates": 1, "save_initial": True,
                       "save_final": True},
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    from config.recipe import load_recipe
    return load_recipe(recipe_path)
