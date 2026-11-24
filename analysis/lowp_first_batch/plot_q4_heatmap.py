#!/usr/bin/env python3
"""Figure 4: type x spectral-band NS transfer (Q4)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path("/mnt/d/muon_research")
SUMMARY = ROOT / "results" / "lowp_first_batch" / "q4_type_band" / "summary.csv"
OUT = ROOT / "results" / "lowp_first_batch" / "figures"
VARIANTS = (("bf16", "BF16"), ("int8_head_fp32", "INT8 per-head"))
TYPES = ("Q", "K", "V", "O")
BANDS = ("top", "middle", "tail")
BAND_LABELS = ("Top", "Middle", "Tail")


def load() -> dict:
    data = {}
    for row in csv.DictReader(SUMMARY.open()):
        if row["band"] not in BANDS:
            continue
        key = (row["variant"], row["type"], row["band"])
        data[key] = {
            "share_u": float(row["share_U_median"]),
            "share_d": float(row["share_D_median"]),
            "a_rel": float(row["A_rel_median"])
            if row["A_rel_median"] not in ("", "nan") else np.nan,
        }
    return data


def annotate(ax, matrix, fmt, threshold, norm):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#8a8a8a", fontsize=8.5)
                continue
            text = fmt(val)
            dark = norm(val) > threshold
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if dark else "#1a1a1a", fontsize=8.5,
                    fontweight="medium")


def main() -> None:
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0),
                             constrained_layout=True)
    cmap_u = plt.get_cmap("Blues").copy()
    cmap_a = plt.get_cmap("Oranges").copy()
    cmap_u.set_bad("#e9e9e9")
    cmap_a.set_bad("#e9e9e9")
    norm_u = Normalize(0, 1)
    norm_a = Normalize(0, 5)

    for r, (variant, label) in enumerate(VARIANTS):
        share_u = np.full((len(TYPES), len(BANDS)), np.nan)
        a_rel = np.full((len(TYPES), len(BANDS)), np.nan)
        for i, typ in enumerate(TYPES):
            for j, band in enumerate(BANDS):
                rec = data.get((variant, typ, band))
                if not rec:
                    continue
                share_u[i, j] = rec["share_u"]
                # A_rel is not interpretable when the band carries negligible error.
                if rec["share_u"] >= 0.01 and rec["share_d"] >= 0.01:
                    a_rel[i, j] = rec["a_rel"]

        ax_u = axes[r, 0]
        im_u = ax_u.imshow(share_u, cmap=cmap_u, norm=norm_u, aspect="auto")
        ax_u.set_title(f"{label}: share of U error")
        ax_u.set_xticks(range(len(BANDS)), BAND_LABELS)
        ax_u.set_yticks(range(len(TYPES)), TYPES)
        ax_u.set_xlabel("spectral band")
        ax_u.set_ylabel("type")
        annotate(ax_u, share_u, lambda v: f"{v:.3f}", 0.55, norm_u)

        ax_a = axes[r, 1]
        im_a = ax_a.imshow(a_rel, cmap=cmap_a, norm=norm_a, aspect="auto")
        ax_a.set_title(f"{label}: within-band amplification A_rel")
        ax_a.set_xticks(range(len(BANDS)), BAND_LABELS)
        ax_a.set_yticks(range(len(TYPES)), TYPES)
        ax_a.set_xlabel("spectral band")
        ax_a.set_ylabel("type")
        annotate(ax_a, a_rel, lambda v: f"{v:.2f}", 0.5, norm_a)

        for ax in (ax_u, ax_a):
            for spine in ax.spines.values():
                spine.set_color("#cfcfcf")
            ax.tick_params(length=0)

    cbar_u = fig.colorbar(
        ScalarMappable(norm=norm_u, cmap=cmap_u), ax=axes[:, 0],
        orientation="horizontal", fraction=0.045, pad=0.08)
    cbar_u.set_label("share of U error")
    cbar_a = fig.colorbar(
        ScalarMappable(norm=norm_a, cmap=cmap_a), ax=axes[:, 1],
        orientation="horizontal", fraction=0.045, pad=0.08)
    cbar_a.set_label("A_rel = rel_U / rel_D")

    fig.suptitle("Q4: NS error transfer by type and spectral band "
                 "(median over 96 matrices)", fontsize=11)
    fig.text(0.01, -0.02,
             "A_rel cells with share_U < 1% or share_D < 1% are masked (—): "
             "the band carries negligible error, so A_rel is unstable there.",
             fontsize=7.5, color="#555555")

    png = OUT / "q4_type_band_heatmap.png"
    pdf = OUT / "q4_type_band_heatmap.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
