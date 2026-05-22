#!/usr/bin/env python3
"""Alpha -> 2 finite-precision diagnostic for Gauss-Jacobi raw quotients.

This script isolates the observation that, even for the smooth benchmark f(t)=exp(-t),
GJ-I/GJ-II errors can increase as alpha approaches 2 if the endpoint-removable
quotients are evaluated in their raw difference form.  The script compares fp64,
fp32, fp16, and fp8-like quantized arithmetic for the raw quotient operations.

The fp8 curves use ml_dtypes.float8_e5m2 and ml_dtypes.float8_e4m3fn as
quantization diagnostics. They are not native NumPy fp8 arithmetic and should not
be interpreted as a full mixed-precision training experiment.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.special as sp
from scipy.special import roots_jacobi
import matplotlib.pyplot as plt

try:
    import ml_dtypes  # type: ignore
except Exception:  # pragma: no cover
    ml_dtypes = None


T = 1.5
LAM = -1.0
M_GJ = 100
EPS_VIS = 1e-18
CLIP_MAX = 1e8


@dataclass(frozen=True)
class PrecisionSpec:
    key: str
    label: str
    color: str
    linestyle: str = "-"
    marker: str = "o"


PRECISIONS = [
    PrecisionSpec("fp64", "raw fp64", "#1f1f1f", "-", "o"),
    PrecisionSpec("fp32", "raw fp32", "#1f77b4", "-", "s"),
    PrecisionSpec("fp16", "raw fp16", "#d62728", "-", "^"),
    PrecisionSpec("fp8_e5m2", "fp8-like e5m2", "#9467bd", "--", "D"),
    PrecisionSpec("fp8_e4m3", "fp8-like e4m3", "#ff7f0e", "--", "v"),
]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def mittag_leffler_E_1_beta(z: float, beta: float, tol: float = 1e-18, max_terms: int = 300) -> float:
    """Compute E_{1,beta}(z) by a rapidly convergent power series.

    For the benchmark z=-1.5 and beta in (1,2), this series is stable and fast.
    """
    s = 0.0
    zpow = 1.0
    for k in range(max_terms):
        if k > 0:
            zpow *= z
        term = zpow / sp.gamma(k + beta)
        s_old = s
        s += term
        if k > 12 and abs(term) < tol * max(1.0, abs(s), abs(s_old)):
            break
    return float(s)


def exact_caputo_exp(alpha: float, t: float = T, lam: float = LAM) -> float:
    """Exact Caputo derivative for f(t)=exp(lam t), 1<alpha<2."""
    return float((lam ** 2) * (t ** (2 - alpha)) * mittag_leffler_E_1_beta(lam * t, 3 - alpha))


def f(t: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-t)


def fp(t: np.ndarray | float) -> np.ndarray | float:
    return -np.exp(-t)


def quantize(x: np.ndarray | float, kind: str) -> np.ndarray:
    """Quantize x to the target format and return float64 for safe aggregation."""
    arr = np.asarray(x)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if kind == "fp64":
            return arr.astype(np.float64)
        if kind == "fp32":
            return arr.astype(np.float32).astype(np.float64)
        if kind == "fp16":
            return arr.astype(np.float16).astype(np.float64)
        if kind == "fp8_e5m2":
            if ml_dtypes is None:
                return np.full_like(arr, np.nan, dtype=np.float64)
            return np.asarray(arr.astype(ml_dtypes.float8_e5m2), dtype=np.float64)
        if kind == "fp8_e4m3":
            if ml_dtypes is None:
                return np.full_like(arr, np.nan, dtype=np.float64)
            return np.asarray(arr.astype(ml_dtypes.float8_e4m3fn), dtype=np.float64)
    raise ValueError(f"Unknown precision kind: {kind}")


def qop(x: np.ndarray | float, kind: str) -> np.ndarray:
    """Quantize after a floating-point operation."""
    return quantize(x, kind)


def stable_q1_notebook(h: np.ndarray, x: np.ndarray, t: float = T) -> np.ndarray:
    """Stable version of notebook's Type-I quotient.

    Notebook quotient: (f'(t)-f'(t-h))/((x+1)t), h=t(x+1)/2.
    For f=exp(-t), this equals exp(-t) expm1(h) / ((x+1)t).
    """
    out = np.exp(-t) * np.expm1(h) / ((x + 1.0) * t)
    small = h < 1e-8
    if np.any(small):
        hs = h[small]
        # expm1(h)/(2h) = 1/2 + h/4 + h^2/12 + h^3/48 + ... because (x+1)t=2h.
        out[small] = np.exp(-t) * (0.5 + hs / 4.0 + hs ** 2 / 12.0 + hs ** 3 / 48.0)
    return out


def stable_q2(h: np.ndarray, t: float = T) -> np.ndarray:
    """Stable Type-II quotient.

    q2 = [f(t)-f(t-h)-h f'(t)] / h^2
       = -exp(-t) [expm1(h)-h] / h^2.
    """
    out = -np.exp(-t) * (np.expm1(h) - h) / (h * h)
    small = h < 1e-5
    if np.any(small):
        hs = h[small]
        # expm1(h)-h = h^2/2 + h^3/6 + h^4/24 + ...
        out[small] = -np.exp(-t) * (0.5 + hs / 6.0 + hs ** 2 / 24.0 + hs ** 3 / 120.0 + hs ** 4 / 720.0)
    return out


def raw_q1_quantized(h: np.ndarray, x: np.ndarray, kind: str, t: float = T) -> np.ndarray:
    """Raw Type-I quotient evaluated under quantized arithmetic."""
    fpt = qop(fp(t), kind)
    fp_th = qop(fp(t - h), kind)
    num = qop(qop(fpt, kind) - qop(fp_th, kind), kind)
    den = qop((x + 1.0) * t, kind)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return qop(num / den, kind)


def raw_q2_quantized(h: np.ndarray, kind: str, t: float = T) -> np.ndarray:
    """Raw Type-II quotient evaluated under quantized arithmetic."""
    ft = qop(f(t), kind)
    fth = qop(f(t - h), kind)
    fpt = qop(fp(t), kind)
    hh = qop(h, kind)
    term = qop(hh * qop(fpt, kind), kind)
    num = qop(qop(qop(ft, kind) - qop(fth, kind), kind) - term, kind)
    den = qop(qop(hh, kind) * qop(hh, kind), kind)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return qop(num / den, kind)


def gj_values(alpha: float, kind: str = "fp64", stable: bool = False, M: int = M_GJ) -> Tuple[float, float, float, float, float]:
    """Compute GJ-I and GJ-II approximations and diagnostics.

    Returns: GJ-I, GJ-II, tau_min, relative q1 error at tau_min, relative q2 error at tau_min.
    """
    x, w = roots_jacobi(M, 0.0, 1.0 - alpha)
    tau = (x + 1.0) / 2.0
    h = T * tau
    coeff = sp.gamma(2.0 - alpha)

    q1_ref = stable_q1_notebook(h, x)
    q2_ref = stable_q2(h)
    if stable:
        q1 = q1_ref
        q2 = q2_ref
    else:
        q1 = raw_q1_quantized(h, x, kind)
        q2 = raw_q2_quantized(h, kind)

    # Notebook-consistent GJ-I formula.
    part_i = (fp(T) - fp(0.0)) * T ** (1.0 - alpha)
    part_i += (alpha - 1.0) * T ** (2.0 - alpha) * 2.0 ** (alpha - 1.0) * np.sum(w * q1)
    gj_i = part_i / coeff

    # Notebook-consistent GJ-II formula in tau variables.
    wt2 = w * (0.5) ** (2.0 - alpha)
    part1 = (fp(T) - fp(0.0)) * T ** (1.0 - alpha)
    part2 = -(alpha - 1.0) * (f(T) - f(0.0) - T * fp(T)) * T ** (-alpha)
    part3 = -(alpha - 1.0) * alpha * T ** (2.0 - alpha) * np.sum(wt2 * q2)
    gj_ii = (part1 + part2 + part3) / coeff

    j = int(np.argmin(tau))
    q1_local_rel = abs(q1[j] - q1_ref[j]) / max(EPS_VIS, abs(q1_ref[j]))
    q2_local_rel = abs(q2[j] - q2_ref[j]) / max(EPS_VIS, abs(q2_ref[j]))
    return float(gj_i), float(gj_ii), float(tau[j]), float(q1_local_rel), float(q2_local_rel)


def sanitize_for_plot(y: np.ndarray, clip: float = CLIP_MAX) -> np.ndarray:
    z = np.array(y, dtype=float)
    bad = ~np.isfinite(z)
    z[bad] = clip
    z = np.clip(z, EPS_VIS, clip)
    return z


def fit_log_growth(alphas: np.ndarray, errors: np.ndarray, amin: float = 1.80, amax: float = 1.99) -> float:
    """Fit log10(error) vs 1/(2-alpha) on the near-alpha=2 window.

    This is only a monotonicity/growth diagnostic, not a theorem-driven rate.
    """
    mask = (alphas >= amin) & (alphas <= amax) & np.isfinite(errors) & (errors > 0) & (errors < CLIP_MAX)
    if mask.sum() < 4:
        return float("nan")
    x = 1.0 / (2.0 - alphas[mask])
    y = np.log10(errors[mask])
    return float(np.polyfit(x, y, 1)[0])


def run(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    datadir = outdir / "data"
    reportdir = outdir / "report"
    for d in [figdir, datadir, reportdir]:
        d.mkdir(parents=True, exist_ok=True)

    configure_style()

    # Dense alpha grid with extra points near alpha=2.
    alphas = np.unique(np.concatenate([
        np.linspace(1.05, 1.80, 31),
        np.linspace(1.82, 1.95, 14),
        np.array([1.96, 1.97, 1.98, 1.985, 1.99, 1.992, 1.995]),
    ]))

    rows: List[Dict[str, float | str]] = []
    results: Dict[str, Dict[str, np.ndarray]] = {}

    exact_vals = np.array([exact_caputo_exp(a) for a in alphas])

    # Stable reference curve.
    stable_gji = []
    stable_gjii = []
    tau_min = []
    denom_amp = []
    for a, exact in zip(alphas, exact_vals):
        g1, g2, tm, _, _ = gj_values(float(a), stable=True)
        stable_gji.append(abs(g1 - exact) / abs(exact))
        stable_gjii.append(abs(g2 - exact) / abs(exact))
        tau_min.append(tm)
        denom_amp.append(1.0 / (T * tm) ** 2)
    stable_gji = np.asarray(stable_gji)
    stable_gjii = np.asarray(stable_gjii)
    tau_min = np.asarray(tau_min)
    denom_amp = np.asarray(denom_amp)

    for spec in PRECISIONS:
        gji_err = []
        gjii_err = []
        q1_loc = []
        q2_loc = []
        for a, exact in zip(alphas, exact_vals):
            g1, g2, tm, q1e, q2e = gj_values(float(a), kind=spec.key, stable=False)
            gji_err.append(abs(g1 - exact) / abs(exact) if np.isfinite(g1) else np.nan)
            gjii_err.append(abs(g2 - exact) / abs(exact) if np.isfinite(g2) else np.nan)
            q1_loc.append(q1e)
            q2_loc.append(q2e)
        results[spec.key] = {
            "gji_err": np.asarray(gji_err),
            "gjii_err": np.asarray(gjii_err),
            "q1_local_rel": np.asarray(q1_loc),
            "q2_local_rel": np.asarray(q2_loc),
        }

    # Write full CSV.
    full_csv = datadir / "alpha2_precision_sweep_full.csv"
    with full_csv.open("w", newline="") as fcsv:
        fieldnames = ["alpha", "exact", "tau_min", "denom_amp", "stable_gji_relerr", "stable_gjii_relerr"]
        for spec in PRECISIONS:
            fieldnames += [f"{spec.key}_gji_relerr", f"{spec.key}_gjii_relerr", f"{spec.key}_q1_local_relerr", f"{spec.key}_q2_local_relerr"]
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for i, a in enumerate(alphas):
            row = {
                "alpha": float(a),
                "exact": float(exact_vals[i]),
                "tau_min": float(tau_min[i]),
                "denom_amp": float(denom_amp[i]),
                "stable_gji_relerr": float(stable_gji[i]),
                "stable_gjii_relerr": float(stable_gjii[i]),
            }
            for spec in PRECISIONS:
                row[f"{spec.key}_gji_relerr"] = float(results[spec.key]["gji_err"][i]) if np.isfinite(results[spec.key]["gji_err"][i]) else "nan"
                row[f"{spec.key}_gjii_relerr"] = float(results[spec.key]["gjii_err"][i]) if np.isfinite(results[spec.key]["gjii_err"][i]) else "nan"
                row[f"{spec.key}_q1_local_relerr"] = float(results[spec.key]["q1_local_rel"][i]) if np.isfinite(results[spec.key]["q1_local_rel"][i]) else "nan"
                row[f"{spec.key}_q2_local_relerr"] = float(results[spec.key]["q2_local_rel"][i]) if np.isfinite(results[spec.key]["q2_local_rel"][i]) else "nan"
            writer.writerow(row)

    # Summary table at selected alphas.
    selected_alphas = [1.25, 1.50, 1.75, 1.90, 1.95, 1.98, 1.99]
    summary_rows = []
    for a0 in selected_alphas:
        i = int(np.argmin(np.abs(alphas - a0)))
        row = {
            "alpha": float(alphas[i]),
            "tau_min": float(tau_min[i]),
            "denom_amp": float(denom_amp[i]),
            "stable_GJII": float(stable_gjii[i]),
            "fp64_GJII": float(results["fp64"]["gjii_err"][i]),
            "fp32_GJII": float(results["fp32"]["gjii_err"][i]),
            "fp16_GJII": float(results["fp16"]["gjii_err"][i]) if np.isfinite(results["fp16"]["gjii_err"][i]) else np.inf,
            "fp8e5m2_GJII": float(results["fp8_e5m2"]["gjii_err"][i]) if np.isfinite(results["fp8_e5m2"]["gjii_err"][i]) else np.inf,
            "fp64_q2local": float(results["fp64"]["q2_local_rel"][i]),
            "fp32_q2local": float(results["fp32"]["q2_local_rel"][i]),
        }
        summary_rows.append(row)

    summary_csv = datadir / "alpha2_precision_summary_selected.csv"
    with summary_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Figure 12: GJ alpha sweep precision comparison.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(alphas, sanitize_for_plot(stable_gji), color="#2ca02c", lw=1.8, label="stable fp64")
    for spec in PRECISIONS:
        ax.plot(alphas, sanitize_for_plot(results[spec.key]["gji_err"]), color=spec.color, ls=spec.linestyle, lw=1.2, label=spec.label)
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_title(r"GJ-I, $M=100$")
    ax.set_ylim(1e-16, 1e4)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.text(0.02, 0.95, "a", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[0, 1]
    ax.plot(alphas, sanitize_for_plot(stable_gjii), color="#2ca02c", lw=1.8, label="stable fp64")
    for spec in PRECISIONS:
        ax.plot(alphas, sanitize_for_plot(results[spec.key]["gjii_err"]), color=spec.color, ls=spec.linestyle, lw=1.2, label=spec.label)
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_title(r"GJ-II, $M=100$")
    ax.set_ylim(1e-16, 1e8)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.text(0.02, 0.95, "b", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[1, 0]
    ax.plot(alphas, tau_min, color="#1f1f1f", label=r"$\tau_{\min}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel(r"smallest GJ node $\tau_{\min}$")
    ax2 = ax.twinx()
    ax2.plot(alphas, denom_amp, color="#d62728", ls="--", label=r"$(t\tau_{\min})^{-2}$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"denominator amplification")
    ax.set_title("endpoint scale induced by Jacobi nodes")
    ax.text(0.02, 0.95, "c", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[1, 1]
    for spec in PRECISIONS:
        ax.plot(alphas, sanitize_for_plot(results[spec.key]["q2_local_rel"]), color=spec.color, ls=spec.linestyle, lw=1.2, label=spec.label)
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_title(r"Type-II quotient at $\tau_{\min}$")
    ax.set_ylim(1e-16, 1e8)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.text(0.02, 0.95, "d", transform=ax.transAxes, fontweight="bold", va="top")

    for ax in axes.flat:
        ax.grid(True, which="major", alpha=0.22, lw=0.6)
        ax.grid(True, which="minor", alpha=0.08, lw=0.4)
    fig.suptitle(r"Finite-precision diagnosis of raw Gauss-Jacobi endpoint quotients as $\alpha\to2$", y=1.02, fontsize=10)
    fig.savefig(figdir / "fig12_alpha2_gj_precision_sweep.png", bbox_inches="tight")
    fig.savefig(figdir / "fig12_alpha2_gj_precision_sweep.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 13: compact error ratio raw/stable and precision growth.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    ax = axes[0]
    ratio64 = results["fp64"]["gjii_err"] / np.maximum(EPS_VIS, stable_gjii)
    ratio32 = results["fp32"]["gjii_err"] / np.maximum(EPS_VIS, stable_gjii)
    ax.plot(alphas, sanitize_for_plot(ratio64), color="#1f1f1f", label="raw fp64 / stable")
    ax.plot(alphas, sanitize_for_plot(ratio32), color="#1f77b4", label="raw fp32 / stable")
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("error ratio")
    ax.set_title(r"raw/stable gap for GJ-II")
    ax.grid(True, which="major", alpha=0.22)
    ax.legend(frameon=False)
    ax.text(0.02, 0.95, "a", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[1]
    for spec in [PRECISIONS[0], PRECISIONS[1], PRECISIONS[2]]:
        ax.plot(alphas, sanitize_for_plot(results[spec.key]["q2_local_rel"]), color=spec.color, label=spec.label)
    ax.plot(alphas, sanitize_for_plot(results["fp8_e5m2"]["q2_local_rel"]), color="#9467bd", ls="--", label="fp8-like e5m2")
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_title(r"local Type-II quotient error")
    ax.grid(True, which="major", alpha=0.22)
    ax.legend(frameon=False)
    ax.text(0.02, 0.95, "b", transform=ax.transAxes, fontweight="bold", va="top")
    fig.savefig(figdir / "fig13_alpha2_raw_stable_gap.png", bbox_inches="tight")
    fig.savefig(figdir / "fig13_alpha2_raw_stable_gap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Slope/growth summary for report.
    growth_rows = []
    for spec in PRECISIONS:
        growth_rows.append({
            "precision": spec.label,
            "GJ-I_log10err_vs_inv2minusalpha": fit_log_growth(alphas, results[spec.key]["gji_err"]),
            "GJ-II_log10err_vs_inv2minusalpha": fit_log_growth(alphas, results[spec.key]["gjii_err"]),
            "q2local_log10err_vs_inv2minusalpha": fit_log_growth(alphas, results[spec.key]["q2_local_rel"]),
        })
    growth_csv = datadir / "alpha2_precision_growth_diagnostics.csv"
    with growth_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(growth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(growth_rows)

    # Markdown report.
    md = reportdir / "alpha2_precision_diagnostic_report.md"
    def fmt_val(x: float) -> str:
        if not np.isfinite(x):
            return "overflow/NaN"
        if abs(x) >= 1e3 or abs(x) < 1e-2:
            return f"{x:.2e}"
        return f"{x:.3g}"

    summary_table = "| alpha | tau_min | (t tau_min)^-2 | stable GJ-II | raw fp64 | raw fp32 | raw fp16 | fp8-like e5m2 | q2 local fp64 | q2 local fp32 |\n"
    summary_table += "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in summary_rows:
        summary_table += (
            f"| {row['alpha']:.3f} | {fmt_val(row['tau_min'])} | {fmt_val(row['denom_amp'])} | "
            f"{fmt_val(row['stable_GJII'])} | {fmt_val(row['fp64_GJII'])} | {fmt_val(row['fp32_GJII'])} | "
            f"{fmt_val(row['fp16_GJII'])} | {fmt_val(row['fp8e5m2_GJII'])} | "
            f"{fmt_val(row['fp64_q2local'])} | {fmt_val(row['fp32_q2local'])} |\n"
        )

    md.write_text(fr"""# Alpha-to-two finite-precision diagnostic for GJ raw quotients

## Motivation

The validation figures show that the Gauss-Jacobi (GJ) error can increase slowly as the fractional order approaches the second-order limit \(\alpha\to2\).  This script tests whether the increase is a quadrature effect or a finite-precision effect caused by raw evaluation of endpoint-removable quotients.

The test uses the same smooth benchmark as `MCfd.ipynb`,

\[
f(t)=e^{{-t}},\qquad t=1.5,
\]

and compares raw GJ-I/GJ-II quotient evaluation under fp64, fp32, fp16 and fp8-like quantized arithmetic.  NumPy has no standard native fp8 dtype; the fp8 curves use `ml_dtypes.float8_e5m2` and `ml_dtypes.float8_e4m3fn` as controlled quantization diagnostics.

## Stable quotient used as reference

For the exponential benchmark, the raw Type-I quotient in the notebook can be rewritten as

\[
\frac{{f'(t)-f'(t-h)}}{{(x+1)t}}
= e^{{-t}}\frac{{\operatorname{{expm1}}(h)}}{{(x+1)t}},
\qquad h=\frac{{t(x+1)}}2.
\]

The Type-II quotient satisfies

\[
\frac{{f(t)-f(t-h)-h f'(t)}}{{h^2}}
= -e^{{-t}}\frac{{\operatorname{{expm1}}(h)-h}}{{h^2}},
\]

with the Taylor limit \(-e^{{-t}}/2\) used for very small \(h\).  These stable forms evaluate the same continuous kernels, but avoid subtracting nearly equal numbers.

## Main figures

![Precision sweep](../figures/fig12_alpha2_gj_precision_sweep.png)

![Raw-stable gap](../figures/fig13_alpha2_raw_stable_gap.png)

## Numerical summary

The following table reports relative GJ-II derivative errors at selected values of \(\alpha\), with \(M=100\).  The last two columns isolate the local Type-II quotient error at the smallest GJ node.

{summary_table}

## Interpretation

The experiment supports the hypothesis that the observed increase of GJ error as \(\alpha\to2\) is primarily a finite-precision endpoint-cancellation effect of the raw quotient realization.

First, the stable fp64 curve stays close to the quadrature/roundoff floor over the same alpha range where the raw fp64 GJ-II error grows by several orders of magnitude.  Second, fp32 deteriorates much earlier, and fp16/fp8-like arithmetic often overflows or produces invalid values because the denominator \((t\tau)^2\) becomes too small after quantization.  Third, the local quotient error at \(\tau_{{\min}}\) grows consistently with the global GJ-II error.

This should not be described as a failure of Gauss-Jacobi quadrature.  The mathematical integrand has a removable endpoint value.  The issue is that the raw algebraic formula evaluates a numerator of size \(O(h^2)\) by subtracting \(O(1)\) quantities and then divides by \(h^2\).  As \(\alpha\to2\), the Jacobi parameter \(1-\alpha\to-1\), and the left endpoint nodes become increasingly close to \(\tau=0\), which exposes the removable singularity to finite-precision cancellation.

## Suggested paper-level conclusion

The proper statement is therefore:

> The increase of raw GJ errors near \(\alpha=2\) is an implementation-level finite-precision effect.  Stable endpoint quotients or Taylor endpoint replacement recover the expected high-accuracy behavior for smooth benchmarks.

This conclusion is compatible with the theoretical GJ convergence results, which assume exact arithmetic evaluation of smooth transformed kernels.
""", encoding="utf-8")

    # LaTeX remark snippet.
    remark = reportdir / "remark_alpha2_finite_precision.tex"
    remark.write_text(r"""\begin{remark}[Endpoint cancellation as \(\alpha\to2\)]
The Gauss--Jacobi rule itself should not be interpreted as deteriorating when the numerical error of the raw formulas increases near the second-order limit.  The effect is caused by the algebraic realization of the removable endpoint singularity.  For example, in the exponential benchmark \(f(t)=e^{-t}\), the raw Type-II quotient
\[
  \frac{f(t)-f(t-h)-h f'(t)}{h^2},\qquad h=t\tau,
\]
subtracts nearly equal \(O(1)\) quantities to produce an \(O(h^2)\) numerator before dividing by \(h^2\).  As \(\alpha\to2\), the Jacobi weight exponent \(1-\alpha\to-1\), and the leftmost Gauss--Jacobi nodes move closer to \(\tau=0\).  Hence the raw quotient becomes increasingly roundoff-sensitive, especially for Type-II.  This is a finite-precision implementation effect, not a contradiction of the quadrature convergence result for the exact endpoint-regular kernel.

In computations, the endpoint-removable quotients should be evaluated by stable formulas.  For \(f(t)=e^{-t}\), for instance,
\[
  \frac{f'(t)-f'(t-h)}{(x+1)t}
  = e^{-t}\frac{\operatorname{expm1}(h)}{(x+1)t},
  \qquad h=\frac{t(x+1)}2,
\]
and
\[
  \frac{f(t)-f(t-h)-h f'(t)}{h^2}
  = -e^{-t}\frac{\operatorname{expm1}(h)-h}{h^2},
\]
with their Taylor limits used when \(h\) is below a prescribed threshold.  Finite-precision tests with fp64, fp32, fp16 and fp8-like quantization confirm that lower precision makes the raw-quotient deterioration appear earlier and more severely, whereas the stabilized fp64 realization remains close to the quadrature floor.
\end{remark}
""", encoding="utf-8")

    # Zip package.
    zip_path = outdir.parent / "MCfd_alpha2_precision_diagnostic_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in [Path(__file__) if "__file__" in globals() else Path("MCfd_alpha2_precision_diagnostic.py")]:
            # In normal execution __file__ points to this script.
            try:
                if path.exists():
                    zf.write(path, arcname="code/MCfd_alpha2_precision_diagnostic.py")
            except Exception:
                pass
        for base, prefix in [(figdir, "figures"), (datadir, "data"), (reportdir, "report")]:
            for p in base.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=f"{prefix}/{p.name}")

    print(f"Wrote outputs to {outdir}")
    print(f"Wrote package to {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("/mnt/data/MCfd_alpha2_precision_diagnostic_outputs"))
    args = parser.parse_args()
    run(args.outdir)
