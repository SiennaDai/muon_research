#!/usr/bin/env python3
"""Summarize Q4 type x band metrics."""
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
        groups[(row["variant"], row["type"], row["band"])].append(row)

    out = args.out or args.csv_path.with_name("summary.csv")
    fields = ["variant", "type", "band", "n", "rank_count_median",
              "energy_share_median", "rel_D_total_median",
              "rel_U_total_median", "A_total_median",
              "rel_D_rel_median", "rel_U_rel_median",
              "A_rel_median", "A_rel_reliable_n",
              "share_D_median", "share_U_median",
              "share_L1_median", "top1_same_mean"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, items in sorted(groups.items()):
            variant, typ, band = key
            a_rel = [float(r["A_rel"]) for r in items if int(r["A_rel_reliable"])]
            writer.writerow({
                "variant": variant, "type": typ, "band": band,
                "n": len(items),
                "rank_count_median": med([float(r["rank_count"]) for r in items]),
                "energy_share_median": med([float(r["energy_share"]) for r in items]),
                "rel_D_total_median": med([float(r["rel_D_total"]) for r in items]),
                "rel_U_total_median": med([float(r["rel_U_total"]) for r in items]),
                "A_total_median": med([float(r["A_total"]) for r in items]),
                "rel_D_rel_median": med([float(r["rel_D_rel"]) for r in items]),
                "rel_U_rel_median": med([float(r["rel_U_rel"]) for r in items]),
                "A_rel_median": med(a_rel),
                "A_rel_reliable_n": len(a_rel),
                "share_D_median": med([float(r["share_D"]) for r in items]),
                "share_U_median": med([float(r["share_U"]) for r in items]),
                "share_L1_median": med([float(r["share_L1"]) for r in items]),
                "top1_same_mean": float(np.mean([int(r["top1_same"]) for r in items])),
            })
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
