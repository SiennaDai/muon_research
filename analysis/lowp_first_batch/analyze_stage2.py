#!/usr/bin/env python3
"""Summarize Stage 2 loss-sensitivity CSVs into CSV + Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MODULES = ("L1", "L2", "L3", "L4", "L2abs", "L3abs")
KEYS = {
    "L1": ("state", "format"),
    "L2": ("space", "band", "alpha"),
    "L3": ("space", "group", "alpha"),
    "L4": ("state", "granularity", "bits", "block_size"),
    "L2abs": ("space", "band", "alpha"),
    "L3abs": ("space", "group", "alpha"),
}


def num(row, key):
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def finite(values):
    return [v for v in values if np.isfinite(v)]


def median(values):
    values = finite(values)
    return float(np.median(values)) if values else float("nan")


def mean(values):
    values = finite(values)
    return float(np.mean(values)) if values else float("nan")


def pick(row, *keys):
    for key in keys:
        if key in row and row[key] not in ("", "nan"):
            return num(row, key)
    return float("nan")


def sign_match(row):
    if "sign_match" in row and row["sign_match"] not in ("", "nan"):
        return float(row["sign_match"])
    dq = num(row, "dL_q")
    dr = num(row, "dL_ref")
    if not (np.isfinite(dq) and np.isfinite(dr)):
        return float("nan")
    return float((dq >= 0) == (dr >= 0))


def summarize(module, rows):
    keys = KEYS[module]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    out = []
    for key, items in sorted(groups.items()):
        rel_u = [pick(r, "rel_U", "rel_W") for r in items]
        share = [pick(r, "share_L1_U", "share_L1_W") for r in items]
        top1 = [pick(r, "top1_U", "top1_W") for r in items]
        out.append({
            "module": module,
            **{k: v for k, v in zip(keys, key)},
            "n": len(items),
            "rel_gap_median": median([num(r, "rel_gap") for r in items]),
            "loss_gap_median": median([num(r, "loss_gap") for r in items]),
            "loss_gap_max": max(finite([num(r, "loss_gap") for r in items]) or [float("nan")]),
            "rel_state_median": median([num(r, "rel_state") for r in items]),
            "rel_U_median": median(rel_u),
            "share_L1_median": median(share),
            "top1_mean": mean(top1),
            "sign_match_mean": mean([sign_match(r) for r in items]),
        })
    return out


def fmt(v):
    if not np.isfinite(v):
        return "-"
    if abs(v) >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.2e}"


def markdown_table(rows, keys, title):
    lines = [f"## {title}", ""]
    header = ["n"] + list(keys) + ["rel_gap", "loss_gap", "loss_gap_max",
                                   "rel_U", "share_L1", "top1", "sign_match"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        cells = [str(r["n"])] + [str(r.get(k, "")) for k in keys]
        cells += [fmt(r["rel_gap_median"]), fmt(r["loss_gap_median"]),
                  fmt(r["loss_gap_max"]), fmt(r["rel_U_median"]),
                  fmt(r["share_L1_median"]), fmt(r["top1_mean"]),
                  fmt(r["sign_match_mean"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = []
    md = ["# Stage 2 loss sensitivity — summary", ""]
    present = {}
    for module in MODULES:
        path = args.dir / f"{module.lower()}_loss.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(path.open()))
        present[module] = len(rows)
        summary = summarize(module, rows)
        all_rows.extend(summary)
        md.append(markdown_table(summary, KEYS[module], module))
        print(f"{module}: {len(rows)} rows")

    fields = ["module", "state", "format", "space", "band", "group", "alpha",
              "granularity", "bits", "block_size", "n",
              "rel_gap_median", "loss_gap_median", "loss_gap_max",
              "rel_state_median", "rel_U_median", "share_L1_median",
              "top1_mean", "sign_match_mean"]
    with (args.dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    (args.dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
    (args.dir / "summary.json").write_text(json.dumps(
        {"rows": present, "summary_rows": len(all_rows)}, indent=2) + "\n")
    print(f"wrote {args.dir/'summary.csv'}")
    print(f"wrote {args.dir/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
