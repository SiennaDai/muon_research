#!/usr/bin/env python3
"""Summarize Q2a/b/c block-scaled variants."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def med(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["matrix"], row["group"], row["component"])].append(row)

    out = args.out or args.csv_path.with_name("summary.csv")
    fields = ["variant", "matrix", "group", "component", "n",
              "rel_D_total_median", "rel_U_total_median",
              "rel_D_median", "rel_U_median", "A_median", "A_reliable_n",
              "share_D_median", "share_U_median",
              "share_L1_median", "top1_same_mean",
              "underflow_heads_median", "clamp_frac_median", "zero_frac_median"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, items in sorted(groups.items()):
            variant, matrix, group, comp = key
            reliable = [float(r["A"]) for r in items if int(r["A_reliable"])]
            writer.writerow({
                "variant": variant, "matrix": matrix, "group": group,
                "component": comp, "n": len(items),
                "rel_D_total_median": med([float(r["rel_D_total"]) for r in items]),
                "rel_U_total_median": med([float(r["rel_U_total"]) for r in items]),
                "rel_D_median": med([float(r["rel_D"]) for r in items]),
                "rel_U_median": med([float(r["rel_U"]) for r in items]),
                "A_median": med(reliable),
                "A_reliable_n": len(reliable),
                "share_D_median": med([float(r["share_D"]) for r in items]),
                "share_U_median": med([float(r["share_U"]) for r in items]),
                "share_L1_median": med([float(r["share_L1"]) for r in items]),
                "top1_same_mean": float(np.mean([int(r["top1_same"]) for r in items])),
                "underflow_heads_median": med([float(r["underflow_heads"]) for r in items]),
                "clamp_frac_median": med([float(r["clamp_frac"]) for r in items]),
                "zero_frac_median": med([float(r["zero_frac"]) for r in items]),
            })
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
