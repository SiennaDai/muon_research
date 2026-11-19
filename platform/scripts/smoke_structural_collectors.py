#!/usr/bin/env python3
"""Validate a structural collector run against one committed snapshot.

Recomputes the tier-1 scalar statistics, head energies, role relations and
tier-2 row/column norms from the raw FP32 snapshot tensors and compares them
with the collector output.  This is the offline half of the smoke test: the
collector must reproduce exactly what a full snapshot would have produced.

Optionally compares per-update wall clock with a run that had no collectors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optim.muon_structural_collector import (  # noqa: E402
    RELATION_METRICS, RELATION_PAIRS, ROLE_INDEX, ROLE_ORDER, ROW_COL_ROLES,
    build_matrix_index, row_col_schedule,
)


def load_group(run_dir: Path, group: str) -> dict:
    index = json.loads((run_dir / group / "index.json").read_text())
    merged = None
    for entry in index["chunks"]:
        with np.load(run_dir / group / entry["file"]) as archive:
            arrays = {key: archive[key] for key in archive.files}
        merged = arrays if merged is None else {
            key: np.concatenate([merged[key], arrays[key]]) for key in arrays}
    order = np.argsort(merged["updates"])
    return {key: value[order] for key, value in merged.items()}


def role_array(manifest: dict, payload: dict, view: dict, role: str) -> np.ndarray:
    source = view.get("source_name", view["name"])
    descriptor = manifest["parameters"][source]["roles"][role]
    if "alias_of" in descriptor:
        descriptor = manifest["parameters"][source]["roles"][descriptor["alias_of"]]
    array = payload["tensors"][descriptor["tensor_id"]].numpy()
    if view.get("derived"):
        array = array[view["row_start"]:view["row_end"]]
    return array


def reference_stats(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float64)
    row, col = np.linalg.norm(x, axis=1), np.linalg.norm(x, axis=0)
    mean = x.mean()
    return np.array([
        mean, math.sqrt(max(0.0, (x * x).mean() - mean * mean)),
        math.sqrt((x * x).mean()), np.linalg.norm(x), np.abs(x).max(),
        (x > 0).mean(), (x < 0).mean(), (x == 0).mean(),
        row.std() / row.mean() if row.mean() > 0 else np.nan,
        col.std() / col.mean() if col.mean() > 0 else np.nan])


def relation_reference(a, b):
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    dot, nx, ny = float(x @ y), float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx > 0 and ny > 0:
        alpha = dot / (nx * nx)
        return np.array([dot, dot / (nx * ny), nx / ny,
                         np.linalg.norm(x - y) / ny,
                         np.linalg.norm(y - alpha * x) / ny])
    return np.full(5, np.nan)


def compare(name: str, got: np.ndarray, want: np.ndarray) -> dict:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    mask = np.isfinite(want)
    if not mask.any():
        return {"name": name, "finite": 0}
    diff = np.abs(got[mask] - want[mask])
    scale = np.maximum(np.abs(want[mask]), 1e-12)
    return {"name": name, "finite": int(mask.sum()),
            "max_abs_error": float(diff.max()),
            "max_relative_error": float((diff / scale).max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--snapshot-update", type=int, required=True)
    parser.add_argument("--plain-run-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads(
        (run_dir / "snapshots" / f"update_{args.snapshot_update:06d}.manifest.json").read_text())
    payload = torch.load(run_dir / "snapshots" / f"update_{args.snapshot_update:06d}.pt",
                         map_location="cpu", weights_only=True)
    metadata = json.loads((run_dir / "metadata/structural_collector.json").read_text())
    index = build_matrix_index(manifest["model"]["config"],
                               [{"name": name, **entry}
                                for name, entry in manifest["parameters"].items()])
    views = index["views"]

    scalar = load_group(run_dir, "scalar_stats")
    heads = load_group(run_dir, "head_stats")
    relations = load_group(run_dir, "role_relations")
    rowcol = load_group(run_dir, "row_col_norms")
    scalar_row = int(np.where(scalar["updates"] == args.snapshot_update)[0][0])
    rc_row = int(np.where(rowcol["updates"] == args.snapshot_update)[0][0])

    checks = []
    stats_want = np.full_like(scalar["stats"][scalar_row], np.nan, dtype=np.float64)
    for view in views:
        for role in ROLE_ORDER:
            if role == "Delta_W":
                a = role_array(manifest, payload, view, "W_after")
                b = role_array(manifest, payload, view, "W_before")
                array = a - b
            else:
                array = role_array(manifest, payload, view, role)
            stats_want[view["view_id"], ROLE_INDEX[role]] = reference_stats(array)
    checks.append(compare("scalar_stats", scalar["stats"][scalar_row], stats_want))

    energy_want = np.full_like(heads["energy"][scalar_row], np.nan, dtype=np.float64)
    for head_index, head in enumerate(index["head_matrices"]):
        view = views[head["view_id"]]
        for role in ROLE_ORDER:
            if role == "Delta_W":
                array = (role_array(manifest, payload, view, "W_after")
                         - role_array(manifest, payload, view, "W_before"))
            else:
                array = role_array(manifest, payload, view, role)
            x = np.asarray(array, dtype=np.float64)
            energy = (x * x).sum(axis=1) if head["axis"] == "rows" else (x * x).sum(axis=0)
            energy = energy.reshape(head["blocks"], head["head_dim"]).sum(axis=1)
            energy_want[head_index, ROLE_INDEX[role]] = energy
    checks.append(compare("head_energy", heads["energy"][scalar_row], energy_want))

    rel_want = np.full_like(relations["relations"][scalar_row], np.nan, dtype=np.float64)
    for view in views:
        for pair_index, (_, role_a, role_b) in enumerate(RELATION_PAIRS):
            a = role_array(manifest, payload, view, "W_after") - \
                role_array(manifest, payload, view, "W_before") \
                if role_a == "Delta_W" else role_array(manifest, payload, view, role_a)
            b = role_array(manifest, payload, view, "W_after") - \
                role_array(manifest, payload, view, "W_before") \
                if role_b == "Delta_W" else role_array(manifest, payload, view, role_b)
            rel_want[view["view_id"], pair_index] = relation_reference(a, b)
    checks.append(compare("role_relations", relations["relations"][scalar_row], rel_want))

    row_want = np.zeros((len(ROW_COL_ROLES), index["row_total"]))
    col_want = np.zeros((len(ROW_COL_ROLES), index["col_total"]))
    qkv_want = np.zeros((len(ROW_COL_ROLES), len(index["qkv_col_offsets"]),
                         index["model"]["n_embd"]))
    for role_index, role in enumerate(ROW_COL_ROLES):
        for row_offset, view in zip(index["row_offsets"], views[:len(index["row_offsets"])]):
            array = role_array(manifest, payload, view, role)
            x = np.asarray(array, dtype=np.float64)
            row_want[role_index, row_offset:row_offset + view["rows"]] = np.linalg.norm(x, axis=1)
        for col_offset, view in zip(index["col_offsets"], views[:len(index["col_offsets"])]):
            array = role_array(manifest, payload, view, role)
            x = np.asarray(array, dtype=np.float64)
            col_want[role_index, col_offset:col_offset + view["cols"]] = np.linalg.norm(x, axis=0)
        for slot, entry in enumerate(index["qkv_col_offsets"]):
            view = next(v for v in views if v["layer"] == entry["layer"]
                        and v["type"] == entry["type"])
            array = role_array(manifest, payload, view, role)
            qkv_want[role_index, slot] = np.linalg.norm(np.asarray(array, dtype=np.float64), axis=0)
    checks.append(compare("row_norms", rowcol["row_norms"][rc_row], row_want))
    checks.append(compare("col_norms", rowcol["col_norms"][rc_row], col_want))
    checks.append(compare("qkv_col_norms", rowcol["qkv_col_norms"][rc_row], qkv_want))

    tiers = {
        "scalar_stats_updates": [int(x) for x in scalar["updates"]],
        "row_col_nodes": int(rowcol["updates"].size),
        "expected_row_col_nodes": len(row_col_schedule(8192, 128, 4)),
        "views": len(views), "head_matrices": len(index["head_matrices"]),
    }
    timing = {}
    if args.plain_run_dir:
        plain = json.loads((Path(args.plain_run_dir) / "summary.json").read_text())
        collected = json.loads((run_dir / "summary.json").read_text())
        timing = {
            "collector_mean_update_seconds": collected["mean_update_seconds"],
            "plain_mean_update_seconds": plain["mean_update_seconds"],
            "ratio": collected["mean_update_seconds"] / plain["mean_update_seconds"],
            "collector_completed_updates": collected["completed_updates"],
            "plain_completed_updates": plain["completed_updates"],
        }
    report = {
        "run_dir": str(run_dir), "snapshot_update": args.snapshot_update,
        "metadata_identity": metadata["identity"], "tiers": tiers,
        "checks": checks, "timing_cpu": timing,
        "max_relative_error": max(item.get("max_relative_error", 0.0) for item in checks),
        "pass_criterion": "max_abs_error < 1e-4 or max_relative_error < 2e-4",
        "all_checks_pass": all(
            item.get("max_abs_error", 0.0) < 1e-4
            or item.get("max_relative_error", 0.0) < 2e-4 for item in checks),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
