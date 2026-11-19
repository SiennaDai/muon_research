"""Read-only access to Structural Emergence collector output.

Every array is decoded through the schema written by the collector itself, so
analysis code never hard-codes matrix order, role order or metric order::

    import structural_io
    meta = structural_io.load_metadata(run_dir)
    scalar = structural_io.load_group(run_dir, "scalar_stats")
    stats = scalar["data"]["stats"]          # (updates, 48, 8, 10)
    views = structural_io.view_table(meta)   # view_id -> name/type/layer

Pruned chunks are skipped.  ``load_group`` reports their update ranges in
``missing_updates`` so a merge with an archived release is explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GROUPS = ("scalar_stats", "head_stats", "role_relations", "row_col_norms")


def load_metadata(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "metadata" / "structural_collector.json"
    return json.loads(path.read_text())


def group_index(run_dir: str | Path, group: str) -> dict:
    return json.loads((Path(run_dir) / group / "index.json").read_text())


def load_group(run_dir: str | Path, group: str) -> dict:
    """Concatenate one group's live chunks in update order.

    Returns ``{"data": {key: ndarray}, "missing_updates": [[first, last], ...],
    "chunks": [...]}``.  The ``updates`` array is sorted ascending.
    """
    if group not in GROUPS:
        raise ValueError(f"unknown structural group: {group}")
    run_dir = Path(run_dir)
    index = group_index(run_dir, group)
    data, missing = None, []
    for entry in index["chunks"]:
        if entry.get("pruned"):
            missing.append([int(entry["first_update"]), int(entry["last_update"])])
            continue
        with np.load(run_dir / group / entry["file"]) as archive:
            arrays = {key: archive[key] for key in archive.files}
        data = arrays if data is None else {
            key: np.concatenate([data[key], arrays[key]]) for key in arrays}
    if data is None:
        data = {}
    if "updates" in data:
        order = np.argsort(data["updates"])
        data = {key: value[order] for key, value in data.items()}
    return {"data": data, "missing_updates": missing, "chunks": index["chunks"]}


def view_table(metadata: dict) -> list[dict]:
    """One row per matrix view (30 actual + 18 derived Q/K/V for the 30M model)."""
    return list(metadata["index"]["views"])


def head_table(metadata: dict) -> list[dict]:
    """One row per head-bearing matrix (layer-major Q, K, V, O)."""
    return list(metadata["index"]["head_matrices"])


def scalar_metrics(metadata: dict) -> list[str]:
    return list(metadata["index"]["scalar_metrics"])


def relation_pairs(metadata: dict) -> list[list[str]]:
    return list(metadata["index"]["relation_pairs"])


def relation_metrics(metadata: dict) -> list[str]:
    return list(metadata["index"]["relation_metrics"])


def row_col_layout(metadata: dict) -> dict:
    index = metadata["index"]
    return {
        "roles": list(index["row_col_roles"]),
        "row_lengths": list(index["row_lengths"]),
        "row_offsets": list(index["row_offsets"]),
        "col_lengths": list(index["col_lengths"]),
        "col_offsets": list(index["col_offsets"]),
        "qkv_col_offsets": list(index["qkv_col_offsets"]),
    }


def coverage(run_dir: str | Path) -> dict:
    """Update ranges present live and in archives for each group."""
    result = {}
    for group in GROUPS:
        index = group_index(run_dir, group)
        live = [entry for entry in index["chunks"] if not entry.get("pruned")]
        pruned = [entry for entry in index["chunks"] if entry.get("pruned")]
        result[group] = {
            "live_chunks": len(live),
            "pruned_chunks": len(pruned),
            "live_updates": ([min(int(e["first_update"]) for e in live),
                              max(int(e["last_update"]) for e in live)]
                             if live else None),
            "pruned_updates": ([min(int(e["first_update"]) for e in pruned),
                                max(int(e["last_update"]) for e in pruned)]
                               if pruned else None),
        }
    return result
