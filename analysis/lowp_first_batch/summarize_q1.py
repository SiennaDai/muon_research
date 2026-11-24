#!/usr/bin/env python3
"""Summarize Q1 metrics by matrix and variant."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NUMERIC = ("rel_U", "cos", "R_norm", "share_L1", "top1_same",
           "rel_U_vs_compare", "cos_vs_compare",
           "R_norm_vs_compare", "share_L1_vs_compare",
           "top1_same_vs_compare", "underflow", "clamp_frac", "zero_frac")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["matrix"], row["variant"])].append(row)

    out = args.out or args.csv_path.with_name("summary.csv")
    fields = ["matrix", "variant", "n", "underflow"]
    for key in NUMERIC:
        if key in ("underflow",):
            continue
        fields += [f"{key}_median", f"{key}_mean"]

    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (matrix, variant), items in sorted(groups.items()):
            record = {"matrix": matrix, "variant": variant, "n": len(items),
                      "underflow": int(sum(int(r["underflow"]) for r in items))}
            for key in NUMERIC:
                if key == "underflow":
                    continue
                vals = [float(r[key]) for r in items if r[key] not in ("", None)
                        and np.isfinite(float(r[key]))]
                if vals:
                    record[f"{key}_median"] = float(np.median(vals))
                    record[f"{key}_mean"] = float(np.mean(vals))
            writer.writerow(record)

    print(f"wrote {out}")
    print(json.dumps({"rows": len(rows), "groups": len(groups)}, indent=2))


if __name__ == "__main__":
    main()
