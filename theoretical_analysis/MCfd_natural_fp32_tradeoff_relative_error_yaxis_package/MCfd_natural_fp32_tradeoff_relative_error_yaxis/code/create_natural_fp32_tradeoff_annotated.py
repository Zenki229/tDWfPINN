#!/usr/bin/env python3
"""
Create annotated natural finite-precision trade-off figures.

Y-axis label is set to "relative error" in all panels.

Input:
    data/natural_fp_raw_stable_gap_curves.csv
    data/natural_fp_tail_slope_summary.csv

Output:
    figures/fig16_natural_fp32_tradeoff_annotated_slopes.png/pdf
    figures/fig16_natural_fp32_mechanism_annotated_slopes.png/pdf

The fitted guide lines are shifted vertically only for readability.
The slope numbers are fitted from the unshifted data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shifted_guide(ax, x, y, slope, fit_range, color, shift_decades=0.42, style=":"):
    xmin, xmax = fit_range
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return
    xc = 10 ** np.mean(np.log10(x[mask]))
    yc = 10 ** np.mean(np.log10(y[mask]))
    xs = np.array([xmin, xmax])
    ys = yc * (xs / xc) ** slope * 10 ** shift_decades
    ax.plot(xs, ys, style, color=color, lw=1.25, alpha=0.95)

def make_figure(df, fitdf, out_dir, include_raw, stem):
    alphas = [1.25, 1.50, 1.75]
    colors = {
        "stable": "#1f77b4",
        "raw32": "#d62728",
        "gap32": "#9467bd",
        "smallguide": "#4d4d4d",
        "largeguide": "#6e6e6e",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.55), sharex=True)

    for row, typ in enumerate(["I", "II"]):
        for col, alpha in enumerate(alphas):
            ax = axes[row, col]
            sub = df[(df["alpha"] == alpha) & (df["type"] == typ)].sort_values("delta")
            x = sub["delta"].values

            ax.loglog(
                x, sub["rel_stable"].values, "-",
                color=colors["stable"], lw=1.95,
                label="stable" if include_raw else "stable bias",
            )
            if include_raw:
                ax.loglog(
                    x, sub["rel_raw32"].values, "-",
                    color=colors["raw32"], lw=1.4,
                    alpha=0.9, label="raw fp32",
                )
            ax.loglog(
                x, sub["rel_gap32"].values, "-.",
                color=colors["gap32"], lw=1.75,
                label="raw-stable gap",
            )

            frow = fitdf[(fitdf["alpha"] == alpha) & (fitdf["type"] == typ)].iloc[0]
            small_range = eval(frow["small_fit_range"])
            large_range = eval(frow["large_fit_range"])
            shifted_guide(
                ax, x, sub["rel_gap32"].values,
                frow["small_expected_slope"], small_range,
                colors["smallguide"], shift_decades=0.45, style=":",
            )
            shifted_guide(
                ax, x, sub["rel_stable"].values,
                frow["large_expected_slope"], large_range,
                colors["largeguide"], shift_decades=-0.43, style="--",
            )

            ax.set_title(rf"Type-{typ}, $\alpha={alpha:.2f}$")
            ax.grid(True, which="major", lw=0.45, alpha=0.35)
            ax.grid(True, which="minor", lw=0.25, alpha=0.18)
            ax.set_xlim(x.min(), x.max())

            vals = [sub["rel_stable"].values, sub["rel_gap32"].values]
            if include_raw:
                vals.append(sub["rel_raw32"].values)
            vals = np.r_[tuple(vals)]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            ax.set_ylim(max(1e-15, vals.min() * 0.25), vals.max() * 3)

            annotation = (
                rf"small-$\delta$: fit {frow['small_fitted_slope']:.2f} "
                rf"(theory {frow['small_expected_slope']:.2f})" + "\n" +
                rf"large-$\delta$: fit {frow['large_fitted_slope']:.2f} "
                rf"(theory {frow['large_expected_slope']:.2f})"
            )
            ax.text(
                0.045, 0.055, annotation,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=6.6,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="0.75",
                    alpha=0.86,
                ),
            )

            if row == 1:
                ax.set_xlabel(r"cutoff $\delta$")
            if col == 0:
                ax.set_ylabel("relative error")
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc="best", handlelength=2.4)

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    data_dir = args.root / "data"
    out_dir = args.root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "natural_fp_raw_stable_gap_curves.csv")
    fitdf = pd.read_csv(data_dir / "natural_fp_tail_slope_summary.csv")

    make_figure(df, fitdf, out_dir, True, "fig16_natural_fp32_tradeoff_annotated_slopes")
    make_figure(df, fitdf, out_dir, False, "fig16_natural_fp32_mechanism_annotated_slopes")

if __name__ == "__main__":
    main()
