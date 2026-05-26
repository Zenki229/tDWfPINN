"""Plot measured saving orders from CSV output."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    coeff = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(coeff[0]), float(coeff[1])


def _drop_first_recorded_pair(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first = df.iloc[0]
    mask = (
        (df["sweep_var"] == first["sweep_var"])
        & (df["sweep_value"] == first["sweep_value"])
        & (df["repeat"] == first["repeat"])
    )
    dropped = int(mask.sum())
    print(
        f"Dropping first recorded pair: sweep_var={first['sweep_var']}, "
        f"sweep_value={first['sweep_value']}, repeat={first['repeat']} ({dropped} rows)",
        flush=True,
    )
    return df.loc[~mask].copy()


def plot_order_csv(
    csv_path: str | Path,
    metric: str,
    kind: str,
    out_dir: str | Path,
    drop_first_pair: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if drop_first_pair:
        df = _drop_first_recorded_pair(df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_cols = ["sweep_var", "sweep_value", "repeat"]
    piv = df.pivot_table(index=idx_cols, columns="method", values=metric, aggfunc="mean").reset_index()
    if "GJ-I" not in piv.columns or "GJ-II" not in piv.columns:
        raise ValueError("CSV must contain both GJ-I and GJ-II rows")
    piv = piv.dropna(subset=["GJ-I", "GJ-II"])
    piv["saving"] = piv["GJ-I"] - piv["GJ-II"]

    processed_path = out_dir / f"{kind}_saving_processed.csv"
    piv.to_csv(processed_path, index=False)
    print(f"Wrote {processed_path}")

    summary_rows = []
    for sweep_var, g in piv.groupby("sweep_var"):
        agg = g.groupby("sweep_value")["saving"].agg(["mean", "median", "std", "count"]).reset_index()
        x = agg["sweep_value"].to_numpy(dtype=float)
        y = agg["median"].to_numpy(dtype=float)
        slope, intercept = _fit_slope(x, y)
        summary_rows.append({
            "sweep_var": sweep_var,
            "slope": slope,
            "intercept": intercept,
            "metric": metric,
            "kind": kind,
            "drop_first_pair": drop_first_pair,
        })

        fig = plt.figure(figsize=(5.8, 4.3))
        yerr = agg["std"].fillna(0).to_numpy(dtype=float)
        plt.errorbar(x, agg["mean"].to_numpy(dtype=float), yerr=yerr, marker="o", linestyle="None", label="mean ± std")
        plt.plot(x, y, marker="s", linestyle="None", label="median")
        if np.isfinite(slope):
            xs = np.linspace(x.min(), x.max(), 200)
            ys = np.exp(intercept) * xs ** slope
            plt.plot(xs, ys, linestyle="--", label=f"median fit slope = {slope:.3f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(sweep_var)
        plt.ylabel(f"{kind} saving: GJ-I minus GJ-II")
        plt.title(f"{kind} saving vs {sweep_var}")
        plt.grid(True, which="both", linestyle=":", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"{kind}_saving_vs_{sweep_var}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Wrote {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"{kind}_slope_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    return summary
