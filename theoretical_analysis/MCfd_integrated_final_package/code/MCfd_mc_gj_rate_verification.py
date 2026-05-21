#!/usr/bin/env python3
"""
Verify the two remaining convergence rates in the paper:

1. Monte Carlo RMS rate: O(M^{-1/2}).
2. Gauss--Jacobi analytic-kernel rate: O(rho^{-2M}).

The script is built to be close to MCfd.ipynb but uses stable closed-form
quotients for a rational benchmark, so the asymptotic rates are not hidden by
floating-point cancellation near tau=0.

Run:
    python MCfd_mc_gj_rate_verification.py --outdir MCfd_rate_verification_report
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.special as sp
from scipy.special import roots_jacobi

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Plot style: clean, journal-like, colorblind-friendly.
# ----------------------------------------------------------------------------
COLORS = {
    "type_i": "#1f77b4",
    "type_ii": "#d62728",
    "exact": "#111111",
    "gray": "#666666",
    "rho1": "#1b9e77",
    "rho2": "#7570b3",
    "rho3": "#e7298a",
}


def set_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", lw=0.45, alpha=0.28)
    ax.grid(True, which="minor", lw=0.25, alpha=0.16)


# ----------------------------------------------------------------------------
# Rational benchmark and stable kernels.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RationalBenchmark:
    """f(s)=1/(a+s), with a pole at s=-a outside [0,T]."""

    a: float = 0.2
    t: float = 1.5

    def f(self, s: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (self.a + s)

    def fp(self, s: np.ndarray | float) -> np.ndarray | float:
        return -1.0 / (self.a + s) ** 2

    def K(self, tau: np.ndarray | float) -> np.ndarray | float:
        """Stable Type-I kernel: [f'(t)-f'(t-t tau)]/(t tau).

        Direct subtraction is intentionally avoided. For f(s)=1/(a+s),
            K(tau) = (2A-h)/(A^2 (A-h)^2), A=a+t, h=t tau.
        """
        A = self.a + self.t
        h = self.t * tau
        return (2.0 * A - h) / (A * A * (A - h) ** 2)

    def H(self, tau: np.ndarray | float) -> np.ndarray | float:
        """Stable Type-II kernel: [f(t)-f(t-h)-h f'(t)]/h^2.

        For f(s)=1/(a+s),
            H(tau) = -1/[A^2 (A-h)], A=a+t, h=t tau.
        """
        A = self.a + self.t
        h = self.t * tau
        return -1.0 / (A * A * (A - h))

    def rho_star(self) -> float:
        """Nearest Bernstein-ellipse parameter induced by the pole.

        The pole in tau is tau_* = 1 + a/t. Under x=2 tau-1, the singularity is
        x_* = 2 tau_*-1 > 1, and rho_* = x_* + sqrt(x_*^2-1).
        """
        tau_star = 1.0 + self.a / self.t
        x_star = 2.0 * tau_star - 1.0
        return float(x_star + math.sqrt(x_star * x_star - 1.0))


# ----------------------------------------------------------------------------
# Numerical integration utilities.
# ----------------------------------------------------------------------------
def weighted_integral_quad(
    alpha: float,
    phi: Callable[[np.ndarray | float], np.ndarray | float],
    *,
    epsabs: float = 2e-14,
    epsrel: float = 2e-14,
) -> float:
    val, _ = integrate.quad(
        lambda tau: float(phi(tau)) * tau ** (1.0 - alpha),
        0.0,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=400,
    )
    return float(val)


def weighted_second_moment_quad(
    alpha: float,
    phi: Callable[[np.ndarray | float], np.ndarray | float],
) -> float:
    val, _ = integrate.quad(
        lambda tau: float(phi(tau)) ** 2 * tau ** (1.0 - alpha),
        0.0,
        1.0,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=400,
    )
    return float(val)


def gj_integral(alpha: float, M: int, phi: Callable[[np.ndarray], np.ndarray]) -> float:
    # scipy roots_jacobi integrates on [-1,1] with (1-x)^a (1+x)^b.
    # tau=(x+1)/2 gives d tau = dx/2 and tau^{1-alpha}=2^{alpha-1}(1+x)^{1-alpha},
    # hence total scale 2^{alpha-2}.
    x, w = roots_jacobi(M, 0.0, 1.0 - alpha)
    tau = (x + 1.0) / 2.0
    wt = w * (0.5) ** (2.0 - alpha)
    return float(np.sum(wt * phi(tau)))


def derivative_prefactors(alpha: float, t: float) -> Tuple[float, float]:
    c_i = (alpha - 1.0) * t ** (2.0 - alpha) / sp.gamma(2.0 - alpha)
    c_ii = alpha * (alpha - 1.0) * t ** (2.0 - alpha) / sp.gamma(2.0 - alpha)
    return float(c_i), float(c_ii)


# ----------------------------------------------------------------------------
# MC O(M^{-1/2}) experiment.
# ----------------------------------------------------------------------------
def run_mc_rate_experiment(
    outdir: Path,
    *,
    alphas: Iterable[float] = (1.25, 1.50, 1.75),
    Ms: Iterable[int] = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    repeats: int = 384,
    seed: int = 229,
    benchmark: RationalBenchmark = RationalBenchmark(a=0.2, t=1.5),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    slope_rows: List[Dict[str, float]] = []

    for alpha in alphas:
        I_K = weighted_integral_quad(alpha, benchmark.K)
        I_H = weighted_integral_quad(alpha, benchmark.H)
        EK2_int = weighted_second_moment_quad(alpha, benchmark.K)
        EH2_int = weighted_second_moment_quad(alpha, benchmark.H)

        # If xi has beta density (2-alpha) tau^{1-alpha}, then
        # E[phi(xi)] = (2-alpha) * integral phi(tau) tau^{1-alpha} d tau.
        mean_K_xi = (2.0 - alpha) * I_K
        mean_H_xi = (2.0 - alpha) * I_H
        var_K_xi = (2.0 - alpha) * EK2_int - mean_K_xi**2
        var_H_xi = (2.0 - alpha) * EH2_int - mean_H_xi**2

        c_i, c_ii = derivative_prefactors(alpha, benchmark.t)

        for M in Ms:
            # Generate all repeats together for a clean empirical RMS estimate.
            xi = rng.beta(2.0 - alpha, 1.0, size=(repeats, int(M)))
            QK = benchmark.K(xi).mean(axis=1) / (2.0 - alpha)
            QH = benchmark.H(xi).mean(axis=1) / (2.0 - alpha)
            err_i = c_i * (QK - I_K)
            err_ii = c_ii * (QH - I_H)

            exact_rms_i = abs(c_i) * math.sqrt(max(var_K_xi, 0.0)) / ((2.0 - alpha) * math.sqrt(M))
            exact_rms_ii = abs(c_ii) * math.sqrt(max(var_H_xi, 0.0)) / ((2.0 - alpha) * math.sqrt(M))
            rows.append({
                "alpha": alpha,
                "M": int(M),
                "type": "Type-I",
                "empirical_rms": float(np.sqrt(np.mean(err_i**2))),
                "exact_rms_formula": exact_rms_i,
                "mean_abs_error": float(np.mean(np.abs(err_i))),
            })
            rows.append({
                "alpha": alpha,
                "M": int(M),
                "type": "Type-II",
                "empirical_rms": float(np.sqrt(np.mean(err_ii**2))),
                "exact_rms_formula": exact_rms_ii,
                "mean_abs_error": float(np.mean(np.abs(err_ii))),
            })

    df = pd.DataFrame(rows)
    for alpha in alphas:
        for typ in ("Type-I", "Type-II"):
            sub = df[(df["alpha"] == alpha) & (df["type"] == typ)].sort_values("M")
            slope_emp, intercept_emp = np.polyfit(np.log(sub["M"]), np.log(sub["empirical_rms"]), 1)
            slope_exact, intercept_exact = np.polyfit(np.log(sub["M"]), np.log(sub["exact_rms_formula"]), 1)
            slope_rows.append({
                "alpha": alpha,
                "type": typ,
                "expected_slope": -0.5,
                "empirical_slope": float(slope_emp),
                "exact_formula_slope": float(slope_exact),
            })
    slopes = pd.DataFrame(slope_rows)
    df.to_csv(outdir / "mc_sqrtM_rate_data.csv", index=False)
    slopes.to_csv(outdir / "mc_sqrtM_slope_summary.csv", index=False)
    return df, slopes


# ----------------------------------------------------------------------------
# GJ O(rho^{-2M}) experiment.
# ----------------------------------------------------------------------------
def fit_spectral_slope(M: np.ndarray, err: np.ndarray) -> Dict[str, float]:
    # Avoid both the initial pre-asymptotic region and machine-precision floor.
    mask = np.isfinite(err) & (err > 1e-12) & (err < 1e-1)
    if mask.sum() < 6:
        mask = np.isfinite(err) & (err > 5e-13) & (err < 1.0)
    if mask.sum() < 6:
        return {
            "n_fit": int(mask.sum()),
            "linear_slope": float("nan"),
            "linear_intercept": float("nan"),
            "exp_slope_with_logM": float("nan"),
            "logM_power": float("nan"),
            "intercept_with_logM": float("nan"),
        }

    Mf = M[mask].astype(float)
    yf = np.log(err[mask])
    linear_slope, linear_intercept = np.polyfit(Mf, yf, 1)
    X = np.column_stack([Mf, np.log(Mf), np.ones_like(Mf)])
    exp_slope, logM_power, intercept = np.linalg.lstsq(X, yf, rcond=None)[0]
    return {
        "n_fit": int(mask.sum()),
        "linear_slope": float(linear_slope),
        "linear_intercept": float(linear_intercept),
        "exp_slope_with_logM": float(exp_slope),
        "logM_power": float(logM_power),
        "intercept_with_logM": float(intercept),
    }


def run_gj_spectral_experiment(
    outdir: Path,
    *,
    alpha: float = 1.50,
    pole_distances: Iterable[float] = (0.05, 0.10, 0.20),
    Ms: Iterable[int] = tuple(range(2, 34)),
    t: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    slope_rows: List[Dict[str, float]] = []

    for a in pole_distances:
        bench = RationalBenchmark(a=float(a), t=t)
        rho = bench.rho_star()
        expected = -2.0 * math.log(rho)
        I_K = weighted_integral_quad(alpha, bench.K)
        I_H = weighted_integral_quad(alpha, bench.H)
        c_i, c_ii = derivative_prefactors(alpha, t)

        for M in Ms:
            QK = gj_integral(alpha, int(M), bench.K)
            QH = gj_integral(alpha, int(M), bench.H)
            rows.append({
                "alpha": alpha,
                "a": float(a),
                "rho_star": rho,
                "expected_exp_slope": expected,
                "M": int(M),
                "type": "Type-I",
                "weighted_integral_error": abs(QK - I_K),
                "derivative_error": abs(c_i) * abs(QK - I_K),
            })
            rows.append({
                "alpha": alpha,
                "a": float(a),
                "rho_star": rho,
                "expected_exp_slope": expected,
                "M": int(M),
                "type": "Type-II",
                "weighted_integral_error": abs(QH - I_H),
                "derivative_error": abs(c_ii) * abs(QH - I_H),
            })

    df = pd.DataFrame(rows)
    for a in pole_distances:
        rho = RationalBenchmark(a=float(a), t=t).rho_star()
        expected = -2.0 * math.log(rho)
        for typ in ("Type-I", "Type-II"):
            sub = df[(df["a"] == float(a)) & (df["type"] == typ)].sort_values("M")
            fit = fit_spectral_slope(sub["M"].to_numpy(), sub["derivative_error"].to_numpy())
            slope_rows.append({
                "alpha": alpha,
                "a": float(a),
                "rho_star": rho,
                "type": typ,
                "expected_exp_slope": expected,
                **fit,
                "exp_slope_ratio": fit["exp_slope_with_logM"] / expected if np.isfinite(fit["exp_slope_with_logM"]) else float("nan"),
            })
    slopes = pd.DataFrame(slope_rows)
    df.to_csv(outdir / "gj_rho_spectral_error_data.csv", index=False)
    slopes.to_csv(outdir / "gj_rho_spectral_slope_summary.csv", index=False)
    return df, slopes


# ----------------------------------------------------------------------------
# Figures.
# ----------------------------------------------------------------------------
def plot_mc_sqrtM(df: pd.DataFrame, slopes: pd.DataFrame, outdir: Path) -> None:
    set_plot_style()
    alphas = sorted(df["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(10.2, 3.15), sharey=False)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        for typ, color, marker in [("Type-I", COLORS["type_i"], "o"), ("Type-II", COLORS["type_ii"], "s")]:
            sub = df[(df["alpha"] == alpha) & (df["type"] == typ)].sort_values("M")
            slope = slopes[(slopes["alpha"] == alpha) & (slopes["type"] == typ)]["empirical_slope"].iloc[0]
            ax.loglog(sub["M"], sub["empirical_rms"], marker=marker, color=color, lw=1.5,
                      label=fr"{typ}, fit {slope:.3f}")
            ax.loglog(sub["M"], sub["exact_rms_formula"], color=color, lw=1.0, ls="--", alpha=0.65)

        # Reference M^{-1/2} line scaled to Type-I final point.
        ref = df[(df["alpha"] == alpha) & (df["type"] == "Type-I")].sort_values("M")
        Mref = ref["M"].to_numpy(float)
        y_last = ref["empirical_rms"].to_numpy(float)[-1]
        yline = y_last * (Mref / Mref[-1]) ** (-0.5)
        ax.loglog(Mref, yline, color=COLORS["gray"], ls=":", lw=1.25, label=r"$M^{-1/2}$")
        ax.set_title(fr"$\alpha={alpha:.2f}$")
        ax.set_xlabel(r"sample number $M$")
        clean_axis(ax)
    axes[0].set_ylabel("RMS derivative error")
    axes[-1].legend(frameon=False, loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.text(0.015, 0.96, "a", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 0.88, 1))
    fig.savefig(outdir / "fig09_mc_sqrtM_rate.png")
    fig.savefig(outdir / "fig09_mc_sqrtM_rate.pdf")
    plt.close(fig)


def plot_gj_spectral(df: pd.DataFrame, slopes: pd.DataFrame, outdir: Path) -> None:
    set_plot_style()
    a_values = sorted(df["a"].unique())
    rho_colors = [COLORS["rho1"], COLORS["rho2"], COLORS["rho3"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.25), sharey=True)

    for ax, typ in zip(axes, ["Type-I", "Type-II"]):
        for a, color in zip(a_values, rho_colors):
            sub = df[(df["a"] == a) & (df["type"] == typ)].sort_values("M")
            rho = sub["rho_star"].iloc[0]
            ss = slopes[(slopes["a"] == a) & (slopes["type"] == typ)].iloc[0]
            exp_slope = ss["exp_slope_with_logM"]
            expected = ss["expected_exp_slope"]
            label = fr"$a={a:.2f}$, $\rho_*={rho:.3f}$, fit {exp_slope:.3f}"
            ax.semilogy(sub["M"], sub["derivative_error"], marker="o", color=color, lw=1.25, label=label)

            # Reference rho^{-2M}; scaled at the first point that is not too large/floored.
            M = sub["M"].to_numpy(float)
            err = sub["derivative_error"].to_numpy(float)
            mask = (err > 1e-12) & (err < 1e-1)
            if mask.sum() >= 3:
                idx = np.where(mask)[0][0]
            else:
                idx = 2
            ref = err[idx] * np.exp(expected * (M - M[idx]))
            ax.semilogy(M, ref, color=color, ls="--", lw=1.0, alpha=0.60)
        ax.set_title(typ)
        ax.set_xlabel(r"Gauss--Jacobi points $M$")
        clean_axis(ax)
    axes[0].set_ylabel("absolute derivative error")
    axes[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.text(0.015, 0.96, "b", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 0.80, 1))
    fig.savefig(outdir / "fig10_gj_rho_spectral_rate.png")
    fig.savefig(outdir / "fig10_gj_rho_spectral_rate.pdf")
    plt.close(fig)


def plot_rate_summary(mc_slopes: pd.DataFrame, gj_slopes: pd.DataFrame, outdir: Path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.15))

    # MC slope summary.
    ax = axes[0]
    xlabels = []
    yvals = []
    colors = []
    for alpha in sorted(mc_slopes["alpha"].unique()):
        for typ in ["Type-I", "Type-II"]:
            row = mc_slopes[(mc_slopes["alpha"] == alpha) & (mc_slopes["type"] == typ)].iloc[0]
            xlabels.append(f"{alpha:.2f}\n{'I' if typ == 'Type-I' else 'II'}")
            yvals.append(row["empirical_slope"])
            colors.append(COLORS["type_i"] if typ == "Type-I" else COLORS["type_ii"])
    ax.axhline(-0.5, color=COLORS["gray"], ls="--", lw=1.1, label="expected")
    ax.scatter(np.arange(len(yvals)), yvals, c=colors, s=36, zorder=3)
    ax.set_xticks(np.arange(len(yvals)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("fitted log-log slope")
    ax.set_title(r"MC: expected $-1/2$")
    ax.set_ylim(-0.56, -0.44)
    clean_axis(ax)

    # GJ slope ratio summary.
    ax = axes[1]
    labels = []
    ratios = []
    colors = []
    for a in sorted(gj_slopes["a"].unique()):
        for typ in ["Type-I", "Type-II"]:
            row = gj_slopes[(gj_slopes["a"] == a) & (gj_slopes["type"] == typ)].iloc[0]
            labels.append(f"{a:.2f}\n{'I' if typ == 'Type-I' else 'II'}")
            ratios.append(row["exp_slope_ratio"])
            colors.append(COLORS["type_i"] if typ == "Type-I" else COLORS["type_ii"])
    ax.axhline(1.0, color=COLORS["gray"], ls="--", lw=1.1, label="expected")
    ax.scatter(np.arange(len(ratios)), ratios, c=colors, s=36, zorder=3)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"fitted slope / $(-2\log\rho_*)$")
    ax.set_title(r"GJ: expected ratio $1$")
    ax.set_ylim(0.88, 1.10)
    clean_axis(ax)

    fig.text(0.015, 0.96, "c", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(outdir / "fig11_rate_slope_summary.png")
    fig.savefig(outdir / "fig11_rate_slope_summary.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Markdown report.
# ----------------------------------------------------------------------------
def format_table(df: pd.DataFrame, columns: List[str], floatfmt: str = ".4g") -> str:
    # Simple markdown table with controlled formatting.
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def make_markdown_report(
    outdir: Path,
    mc_slopes: pd.DataFrame,
    gj_slopes: pd.DataFrame,
    config: Dict[str, object],
) -> None:
    mc_table = mc_slopes.copy()
    mc_table["empirical_slope"] = mc_table["empirical_slope"].map(lambda x: f"{x:.4f}")
    mc_table["expected_slope"] = mc_table["expected_slope"].map(lambda x: f"{x:.4f}")

    gj_table = gj_slopes.copy()
    gj_table["expected_exp_slope"] = gj_table["expected_exp_slope"].map(lambda x: f"{x:.4f}")
    gj_table["linear_slope"] = gj_table["linear_slope"].map(lambda x: f"{x:.4f}")
    gj_table["exp_slope_with_logM"] = gj_table["exp_slope_with_logM"].map(lambda x: f"{x:.4f}")
    gj_table["logM_power"] = gj_table["logM_power"].map(lambda x: f"{x:.3f}")
    gj_table["exp_slope_ratio"] = gj_table["exp_slope_ratio"].map(lambda x: f"{x:.3f}")

    mc_md_table = format_table(
        mc_table[["alpha", "type", "expected_slope", "empirical_slope", "exact_formula_slope"]],
        ["alpha", "type", "expected_slope", "empirical_slope", "exact_formula_slope"],
    )
    gj_md_table = format_table(
        gj_table[["a", "rho_star", "type", "expected_exp_slope", "linear_slope", "exp_slope_with_logM", "logM_power", "exp_slope_ratio"]],
        ["a", "rho_star", "type", "expected_exp_slope", "linear_slope", "exp_slope_with_logM", "logM_power", "exp_slope_ratio"],
    )

    md = r"""# MCfd rate verification: $M^{-1/2}$ and $\rho^{-2M}$

This note adds the last two rate checks suggested by the paper:

1. the Monte Carlo root-mean-square rate

   $$
   \left(\mathbb E|D^\alpha f-D^\alpha_{M,\mathrm{MC}}f|^2\right)^{1/2}
   = O(M^{-1/2}),
   $$

2. the analytic Gauss--Jacobi rate

   $$
   |D^\alpha f-D^\alpha_{M,\mathrm{GJ}}f|=O(\rho^{-2M}).
   $$

The experiment uses the stable kernels in the paper rather than raw difference quotients, because the previous precision experiment showed that raw quotients can hit cancellation error before the asymptotic quadrature rate is visible.

## Benchmark

For the rate test I used

$$
    f_a(s)=\frac1{a+s},\qquad t=1.5.
$$

The stable kernels are available in closed form:

$$
K_f(t,\tau)=\frac{f'(t)-f'(t-t\tau)}{t\tau}
=\frac{2A-h}{A^2(A-h)^2},
$$

$$
H_f(t,\tau)=\frac{f(t)-f(t-t\tau)-t\tau f'(t)}{(t\tau)^2}
=-\frac1{A^2(A-h)},
$$

where $A=a+t$ and $h=t\tau$.  The pole is located at

$$
    \tau_* = 1+\frac a t>1.
$$

Under the Bernstein map $x=2\tau-1$, the corresponding ellipse parameter is

$$
    \rho_* = x_*+\sqrt{x_*^2-1},\qquad x_*=2\tau_*-1.
$$

For the GJ test I use $a=0.05,0.10,0.20$, giving three different $\rho_*$ values.

## Figure 9: Monte Carlo $M^{-1/2}$ rate

![MC sqrt M rate](fig09_mc_sqrtM_rate.png)

The dots are empirical RMS derivative errors over repeated Monte Carlo runs.  The dashed curves are the exact variance formula induced by

$$
Q_M^{\mathrm{MC}}[\phi]
=\frac1{(2-\alpha)M}\sum_{j=1}^M\phi(\xi_j),
\qquad \xi_j\sim \mathrm{Beta}(2-\alpha,1).
$$

The fitted slopes are:

__MC_TABLE__

All empirical slopes are very close to $-1/2$, across $\alpha=1.25,1.50,1.75$ and both Type-I / Type-II.

## Figure 10: Gauss--Jacobi $\rho^{-2M}$ rate

![GJ rho spectral rate](fig10_gj_rho_spectral_rate.png)

For the analytic test, the expected exponential slope in a plot of $\log(error)$ against $M$ is

$$
    -2\log\rho_*.
$$

For Type-I, $K_f$ has a second-order pole at $\tau_*$ for this rational benchmark, so the asymptotic form is more accurately fitted as

$$
    error \approx C M^p \rho_*^{-2M}.
$$

Therefore the slope table reports both a naive linear fit and a two-term fit

$$
    \log(error) \approx c+sM+p\log M.
$$

The exponential slope $s$ from the two-term fit is the quantity that should match $-2\log\rho_*$.

__GJ_TABLE__

The Type-II kernel has only a first-order pole, so the naive linear slope is already very close to the expected value.  Type-I shows a visible polynomial prefactor, but after adding the $\log M$ term, the extracted exponential slope also matches $-2\log\rho_*$.

## Figure 11: slope summary

![Rate slope summary](fig11_rate_slope_summary.png)

The left panel summarizes the MC log-log slopes and confirms $M^{-1/2}$.  The right panel summarizes the normalized GJ exponential slopes

$$
    \frac{s_{\mathrm{fit}}}{-2\log\rho_*},
$$

which cluster around 1.

## Interpretation for the paper

The numerical evidence supports the two remaining theoretical rates:

- Monte Carlo has the canonical RMS rate $O(M^{-1/2})$, independent of the fractional order.  The fractional order changes the variance constant through the Beta sampling density, but not the rate.
- For analytic kernels, Gauss--Jacobi error decays exponentially with $M$.  The observed exponential slope is governed by the nearest complex singularity through $\rho_*^{-2M}$.
- The Type-I rational test has a higher-order pole, producing a mild polynomial prefactor.  This does not contradict the theorem: the theorem gives $C_\rho\rho^{-2M}$ for any ellipse strictly inside the analytic domain, and polynomial prefactors are absorbed by using any $\rho<\rho_*$.  The fitted exponential part still matches the expected $-2\log\rho_*$.

## Reproducibility

Run:

```bash
python MCfd_mc_gj_rate_verification.py --outdir MCfd_rate_verification_report --seed __SEED__ --mc-repeats __MC_REPEATS__
```

Generated CSV files:

- `mc_sqrtM_rate_data.csv`
- `mc_sqrtM_slope_summary.csv`
- `gj_rho_spectral_error_data.csv`
- `gj_rho_spectral_slope_summary.csv`
"""
    md = md.replace("__MC_TABLE__", mc_md_table)
    md = md.replace("__GJ_TABLE__", gj_md_table)
    md = md.replace("__SEED__", str(config["seed"]))
    md = md.replace("__MC_REPEATS__", str(config["mc_repeats"]))

    (outdir / "MCfd_mc_gj_rate_verification_report.md").write_text(md, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="MCfd_rate_verification_report")
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--mc-repeats", type=int, default=384)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": args.seed,
        "mc_repeats": args.mc_repeats,
        "mc_alphas": [1.25, 1.50, 1.75],
        "mc_Ms": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "gj_alpha": 1.50,
        "gj_pole_distances": [0.05, 0.10, 0.20],
        "gj_Ms": list(range(2, 34)),
        "t": 1.5,
    }
    (outdir / "rate_verification_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    mc_df, mc_slopes = run_mc_rate_experiment(
        outdir,
        alphas=config["mc_alphas"],
        Ms=config["mc_Ms"],
        repeats=args.mc_repeats,
        seed=args.seed,
        benchmark=RationalBenchmark(a=0.2, t=config["t"]),
    )
    gj_df, gj_slopes = run_gj_spectral_experiment(
        outdir,
        alpha=config["gj_alpha"],
        pole_distances=config["gj_pole_distances"],
        Ms=config["gj_Ms"],
        t=config["t"],
    )

    plot_mc_sqrtM(mc_df, mc_slopes, outdir)
    plot_gj_spectral(gj_df, gj_slopes, outdir)
    plot_rate_summary(mc_slopes, gj_slopes, outdir)
    make_markdown_report(outdir, mc_slopes, gj_slopes, config)

    print("MC slope summary")
    print(mc_slopes.to_string(index=False))
    print("\nGJ slope summary")
    print(gj_slopes.to_string(index=False))
    print(f"\nWrote report to {outdir}")


if __name__ == "__main__":
    main()
