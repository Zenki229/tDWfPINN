"""
MCfd_refactored_validation.py

A reproducible validation suite built from the formulas used in the uploaded
MCfd.ipynb notebook.  It keeps the four estimators used there

    MC-I, MC-II, GJ-I, GJ-II

and adds diagnostics for three questions:

1. How alpha affects the estimators, especially alpha -> 2;
2. How M affects the error, and why Gauss--Jacobi (GJ) may get worse when M is
   too large under raw difference quotients;
3. What changes when the test function is not smooth.

Run:
    python MCfd_refactored_validation.py --outdir MCfd_refactored_outputs --seed 229 --mc-repeats 8

Dependencies:
    numpy, scipy, matplotlib

No pymittagleffler dependency is required.  For f(t)=exp(lambda t), the exact
Caputo derivative is evaluated using an equivalent confluent-hypergeometric
formula for E_{1,3-alpha}(lambda t).
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaln, hyp1f1, roots_jacobi

MethodMC = Literal["MC-I", "MC-II"]
MethodGJ = Literal["GJ-I", "GJ-II"]
KernelMode = Literal["raw", "stable"]

# -----------------------------------------------------------------------------
# Figure style: compact, clean, colorblind-friendly, journal-like.
# -----------------------------------------------------------------------------

MM_TO_IN = 1.0 / 25.4

COLORS = {
    "exact": "#111111",
    "MC-I": "#0072B2",
    "MC-II": "#D55E00",
    "GJ-I": "#009E73",
    "GJ-II": "#E69F00",
    "GJ-I-stable": "#56B4E9",
    "GJ-II-stable": "#CC79A7",
    "smooth": "#111111",
    "endpoint": "#7A7A7A",
    "interior": "#984EA3",
    "diagnostic": "#4D4D4D",
}
LINESTYLES = {
    "MC-I": (0, (5, 2)),
    "MC-II": (0, (5, 2)),
    "GJ-I": "-",
    "GJ-II": "-",
    "exact": "-",
}
MARKERS = {"MC-I": "o", "MC-II": "s", "GJ-I": "^", "GJ-II": "D"}
ERR_FLOOR = 1e-16


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    # Place labels inside the axes to avoid clashes with tick labels after tight export.
    ax.text(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
        zorder=10,
    )


def finish_axis(ax: plt.Axes, *, legend: bool = False) -> None:
    ax.grid(True, which="major", color="0.88", linewidth=0.6)
    ax.grid(True, which="minor", color="0.93", linewidth=0.4)
    if legend:
        ax.legend(handlelength=2.2, borderaxespad=0.35)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


def rel_err(approx: np.ndarray | float, exact: np.ndarray | float) -> np.ndarray | float:
    return np.abs(np.asarray(approx) - np.asarray(exact)) / np.maximum(np.abs(exact), 1e-300)

# -----------------------------------------------------------------------------
# Test functions with exact Caputo derivatives.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExponentialFunction:
    """f(t)=exp(lambda t), the same benchmark as the uploaded MCfd.ipynb."""

    lam: float = -1.0

    def f(self, s):
        return np.exp(self.lam * np.asarray(s))

    def fp(self, s):
        return self.lam * np.exp(self.lam * np.asarray(s))

    def exact_caputo(self, t: float, alpha: float) -> float:
        """Exact D_C^alpha exp(lambda t) for alpha in (1,2).

        MCfd.ipynb uses
            lambda^2 t^(2-alpha) E_{1,3-alpha}(lambda t).
        This implementation evaluates the equivalent expression
            lambda^2 t^(2-alpha) 1F1(1; 3-alpha; lambda t) / Gamma(3-alpha),
        using Kummer's transformation for numerical stability.
        """
        lam = self.lam
        return float(
            lam**2
            * t ** (2.0 - alpha)
            * math.exp(lam * t)
            * hyp1f1(2.0 - alpha, 3.0 - alpha, -lam * t)
            / gamma(3.0 - alpha)
        )

    def kernel_K_stable(self, t: float, tau: np.ndarray) -> np.ndarray:
        """Stable K=(f'(t)-f'(t-t tau))/(t tau) for the exponential benchmark."""
        tau = np.asarray(tau, dtype=float)
        r = t * tau
        lam = self.lam
        e = math.exp(lam * t)
        out = lam * e * (-np.expm1(-lam * r)) / r
        small = np.abs(r) < 1e-7
        if np.any(small):
            out = np.asarray(out, dtype=float)
            out[small] = (
                lam**2 * e
                - 0.5 * r[small] * lam**3 * e
                + (r[small] ** 2) * lam**4 * e / 6.0
                - (r[small] ** 3) * lam**5 * e / 24.0
            )
        return out

    def kernel_H_stable(self, t: float, tau: np.ndarray) -> np.ndarray:
        """Stable H=(f(t)-f(t-t tau)-t tau f'(t))/(t tau)^2."""
        tau = np.asarray(tau, dtype=float)
        r = t * tau
        lam = self.lam
        e = math.exp(lam * t)
        out = e * (-np.expm1(-lam * r) - lam * r) / (r * r)
        small = np.abs(r) < 1e-5
        if np.any(small):
            out = np.asarray(out, dtype=float)
            out[small] = (
                -0.5 * lam**2 * e
                + r[small] * lam**3 * e / 6.0
                - (r[small] ** 2) * lam**4 * e / 24.0
                + (r[small] ** 3) * lam**5 * e / 120.0
            )
        return out


@dataclass(frozen=True)
class ShiftedPowerFunction:
    r"""Non-smooth benchmark f(t)=(t-t_c)_+^beta.

    For 1<beta<2, f is C^1 but not C^2 at t=t_c.  Its Caputo derivative is
    closed form:
        D_C^alpha f(t) = Gamma(beta+1)/Gamma(beta+1-alpha) * (t-t_c)_+^(beta-alpha).
    """

    beta: float = 1.35
    tc: float = 0.25

    def f(self, s):
        s = np.asarray(s)
        return np.maximum(s - self.tc, 0.0) ** self.beta

    def fp(self, s):
        s = np.asarray(s)
        x = np.maximum(s - self.tc, 0.0)
        # derivative is zero to the left of the kink.
        return self.beta * x ** (self.beta - 1.0) * (s > self.tc)

    def exact_caputo(self, t: float, alpha: float) -> float:
        if t <= self.tc:
            return 0.0
        return float(
            math.exp(gammaln(self.beta + 1.0) - gammaln(self.beta + 1.0 - alpha))
            * (t - self.tc) ** (self.beta - alpha)
        )

# -----------------------------------------------------------------------------
# Four estimators: direct function form of the MCfd.ipynb cells.
# -----------------------------------------------------------------------------


def beta_samples(alpha: float, M: int, rng: np.random.Generator) -> np.ndarray:
    """tau ~ Beta(2-alpha, 1), matching scipy.stats.beta.rvs(2-alpha, 1)."""
    return rng.beta(2.0 - alpha, 1.0, size=int(M)).astype(float)


def gauss_jacobi_rule(alpha: float, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss--Jacobi rule for int_0^1 phi(tau) tau^(1-alpha) d tau.

    MCfd.ipynb uses roots_jacobi(M, 0, 1-alpha).  With tau=(x+1)/2,
    the transformed weights are 2^(alpha-2)*w.
    """
    x, w = roots_jacobi(int(M), 0.0, 1.0 - alpha)
    tau = 0.5 * (x + 1.0)
    weights = (2.0 ** (alpha - 2.0)) * w
    return tau.astype(float), weights.astype(float)


def kernel_K(fun, t: float, tau: np.ndarray, mode: KernelMode = "raw") -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    r = t * tau
    if mode == "stable" and isinstance(fun, ExponentialFunction):
        return fun.kernel_K_stable(t, tau)
    return (fun.fp(t) - fun.fp(t - r)) / r


def kernel_H(fun, t: float, tau: np.ndarray, mode: KernelMode = "raw") -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    r = t * tau
    if mode == "stable" and isinstance(fun, ExponentialFunction):
        return fun.kernel_H_stable(t, tau)
    return (fun.f(t) - fun.f(t - r) - r * fun.fp(t)) / (r * r)


def caputo_mc(
    fun,
    t: float,
    alpha: float,
    M: int,
    method: MethodMC,
    *,
    rng: Optional[np.random.Generator] = None,
    tau: Optional[np.ndarray] = None,
    eps_abs: float = 1e-7,
) -> float:
    """MC-I / MC-II from MCfd.ipynb.

    The notebook uses tau_max=max(tau, eps/t) in the denominator while the
    numerator is still evaluated at the original tau.  We keep that behavior.
    """
    if tau is None:
        if rng is None:
            rng = np.random.default_rng(0)
        tau = beta_samples(alpha, int(M), rng)
    else:
        tau = np.asarray(tau, dtype=float)
        M = tau.size

    delta = eps_abs / t
    tau_delta = np.maximum(tau, delta)
    r = t * tau
    r_delta = t * tau_delta

    pref = 1.0 / gamma(2.0 - alpha)
    term_derivative = (fun.fp(t) - fun.fp(0.0)) * t ** (1.0 - alpha)

    if method == "MC-I":
        K_delta = (fun.fp(t) - fun.fp(t - r)) / r_delta
        val = term_derivative + (alpha - 1.0) / (2.0 - alpha) * t ** (2.0 - alpha) * float(np.mean(K_delta))
    elif method == "MC-II":
        endpoint = (fun.f(t) - fun.f(0.0) - t * fun.fp(t)) * t ** (-alpha)
        H_delta = (fun.f(t) - fun.f(t - r) - r * fun.fp(t)) / (r_delta * r_delta)
        val = term_derivative - (alpha - 1.0) * endpoint - alpha * (alpha - 1.0) / (2.0 - alpha) * t ** (2.0 - alpha) * float(np.mean(H_delta))
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(pref * val)


def caputo_gj(
    fun,
    t: float,
    alpha: float,
    M: int,
    method: MethodGJ,
    *,
    kernel_mode: KernelMode = "raw",
    eps_abs: float = 0.0,
) -> float:
    """GJ-I / GJ-II from MCfd.ipynb.

    eps_abs can be used to drop extremely small GJ nodes in the raw kernel, as
    in the original M-sweep cell.  For the stable exponential kernel, no node is
    dropped.
    """
    tau, weights = gauss_jacobi_rule(alpha, int(M))
    if eps_abs > 0.0 and kernel_mode == "raw":
        keep = tau > eps_abs / t
        tau = tau[keep]
        weights = weights[keep]

    pref = 1.0 / gamma(2.0 - alpha)
    term_derivative = (fun.fp(t) - fun.fp(0.0)) * t ** (1.0 - alpha)

    if method == "GJ-I":
        integral = float(np.sum(weights * kernel_K(fun, t, tau, mode=kernel_mode)))
        val = term_derivative + (alpha - 1.0) * t ** (2.0 - alpha) * integral
    elif method == "GJ-II":
        endpoint = (fun.f(t) - fun.f(0.0) - t * fun.fp(t)) * t ** (-alpha)
        integral = float(np.sum(weights * kernel_H(fun, t, tau, mode=kernel_mode)))
        val = term_derivative - (alpha - 1.0) * endpoint - alpha * (alpha - 1.0) * t ** (2.0 - alpha) * integral
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(pref * val)

# -----------------------------------------------------------------------------
# Experiments.
# -----------------------------------------------------------------------------


def experiment_alpha_sweep(
    outdir: Path,
    *,
    seed: int = 229,
    t: float = 1.5,
    lam: float = -1.0,
    M_mc: int = 10_000,
    M_gj: int = 100,
    eps_mc1: float = 1e-16,
    eps_mc2: float = 1e-7,
) -> Dict:
    fun = ExponentialFunction(lam=lam)
    alphas = np.linspace(1.01, 1.99, 100)
    exact = np.array([fun.exact_caputo(t, float(a)) for a in alphas])
    approx: Dict[str, np.ndarray] = {m: np.empty_like(alphas) for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]}

    for i, a in enumerate(alphas):
        rng = np.random.default_rng(seed + i)
        tau = beta_samples(float(a), M_mc, rng)
        approx["MC-I"][i] = caputo_mc(fun, t, float(a), M_mc, "MC-I", tau=tau, eps_abs=eps_mc1)
        approx["MC-II"][i] = caputo_mc(fun, t, float(a), M_mc, "MC-II", tau=tau, eps_abs=eps_mc2)
        approx["GJ-I"][i] = caputo_gj(fun, t, float(a), M_gj, "GJ-I")
        approx["GJ-II"][i] = caputo_gj(fun, t, float(a), M_gj, "GJ-II")

    err = {m: rel_err(approx[m], exact) for m in approx}
    # Diagnostic: mass of the MC sampling distribution below the cutoff.
    # For tau~Beta(2-alpha,1), P(tau<delta)=delta^(2-alpha).
    cutoff_mass_mc2 = (eps_mc2 / t) ** (2.0 - alphas)
    one_over_M = np.full_like(alphas, 1.0 / M_mc)

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM_TO_IN, 58 * MM_TO_IN), constrained_layout=True)

    ax = axes[0]
    ax.plot(alphas, exact, color=COLORS["exact"], lw=1.7, label="exact")
    for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        ax.plot(alphas, approx[m], color=COLORS[m], lw=1.15, ls=LINESTYLES[m], label=m)
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel(r"$D_t^\alpha f(t)$")
    ax.set_title(r"$f(t)=e^{-t}$, $t=1.5$")
    ax.set_xlim(1.0, 2.0)
    finish_axis(ax, legend=True)
    panel_label(ax, "a")

    ax = axes[1]
    for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        ax.semilogy(alphas, np.maximum(err[m], ERR_FLOOR), color=COLORS[m], lw=1.25, ls=LINESTYLES[m], label=m)
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(1e-16, 2)
    finish_axis(ax, legend=False)
    panel_label(ax, "b")

    ax = axes[2]
    ax.semilogy(alphas, cutoff_mass_mc2, color=COLORS["diagnostic"], lw=1.5, label=r"$P(\tau<\epsilon/t)$")
    ax.semilogy(alphas, one_over_M, color="0.65", lw=1.0, ls=(0, (3, 2)), label=r"$1/M_{MC}$")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("endpoint mass")
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(1e-8, 2)
    finish_axis(ax, legend=True)
    panel_label(ax, "c")

    save_figure(fig, outdir, "fig01_alpha_sweep_exp")

    return {
        "alphas": alphas.tolist(),
        "exact": exact.tolist(),
        "approx": {k: v.tolist() for k, v in approx.items()},
        "relative_error": {k: v.tolist() for k, v in err.items()},
        "cutoff_mass_mc2": cutoff_mass_mc2.tolist(),
        "settings": {"t": t, "lambda": lam, "M_mc": M_mc, "M_gj": M_gj, "eps_mc1": eps_mc1, "eps_mc2": eps_mc2},
    }


def experiment_M_sweep(
    outdir: Path,
    *,
    seed: int = 229,
    t: float = 1.5,
    alpha: float = 1.5,
    lam: float = -1.0,
    eps_mc1: float = 1e-10,
    eps_mc2: float = 1e-7,
    eps_gj2_drop: float = 1e-10,
    mc_repeats: int = 8,
) -> Dict:
    fun = ExponentialFunction(lam=lam)
    exact = fun.exact_caputo(t, alpha)

    Ms = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240], dtype=int)
    M_gj = np.array([10, 20, 40, 80, 160, 320, 640, 1280], dtype=int)

    mc_stats = {}
    for method, eps in [("MC-I", eps_mc1), ("MC-II", eps_mc2)]:
        errs = np.zeros((len(Ms), mc_repeats))
        for i, M in enumerate(Ms):
            for r in range(mc_repeats):
                rng = np.random.default_rng(seed + 10_000 * (method == "MC-II") + 100 * i + r)
                val = caputo_mc(fun, t, alpha, int(M), method, rng=rng, eps_abs=eps)
                errs[i, r] = rel_err(val, exact)
        mc_stats[method] = {
            "median": np.median(errs, axis=1),
            "q25": np.quantile(errs, 0.25, axis=1),
            "q75": np.quantile(errs, 0.75, axis=1),
        }

    gj_err: Dict[str, np.ndarray] = {}
    tau_min = np.zeros_like(M_gj, dtype=float)
    for mode in ["raw", "stable"]:
        for method in ["GJ-I", "GJ-II"]:
            vals = []
            for i, M in enumerate(M_gj):
                tau, _ = gauss_jacobi_rule(alpha, int(M))
                tau_min[i] = tau.min()
                eps_drop = eps_gj2_drop if (mode == "raw" and method == "GJ-II") else 0.0
                val = caputo_gj(fun, t, alpha, int(M), method, kernel_mode=mode, eps_abs=eps_drop)
                vals.append(rel_err(val, exact))
            gj_err[f"{method}-{mode}"] = np.array(vals)

    eps_machine = np.finfo(float).eps
    cancellation_I = eps_machine / np.maximum(tau_min, eps_machine)
    cancellation_II = eps_machine / np.maximum(tau_min, eps_machine) ** 2

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM_TO_IN, 58 * MM_TO_IN), constrained_layout=True)

    ax = axes[0]
    for method in ["MC-I", "MC-II"]:
        med = np.maximum(mc_stats[method]["median"], ERR_FLOOR)
        ax.loglog(Ms, med, color=COLORS[method], marker=MARKERS[method], ms=3.0, lw=1.2, ls=LINESTYLES[method], label=method)
        ax.fill_between(Ms, np.maximum(mc_stats[method]["q25"], ERR_FLOOR), np.maximum(mc_stats[method]["q75"], ERR_FLOOR), color=COLORS[method], alpha=0.15, lw=0)
    for method in ["GJ-I", "GJ-II"]:
        ax.loglog(M_gj, np.maximum(gj_err[f"{method}-raw"], ERR_FLOOR), color=COLORS[method], marker=MARKERS[method], ms=3.1, lw=1.25, label=method)
    ax.set_xlabel(r"samples/nodes $M$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-16, 2e-1)
    ax.set_title(r"original setting: $\alpha=1.5$, $t=1.5$")
    finish_axis(ax, legend=True)
    panel_label(ax, "a")

    ax = axes[1]
    for method in ["GJ-I", "GJ-II"]:
        ax.loglog(M_gj, np.maximum(gj_err[f"{method}-raw"], ERR_FLOOR), color=COLORS[method], marker=MARKERS[method], ms=3.0, lw=1.2, label=f"{method}, raw")
        ax.loglog(M_gj, np.maximum(gj_err[f"{method}-stable"], ERR_FLOOR), color=COLORS[f"{method}-stable"], marker=MARKERS[method], ms=2.8, lw=1.05, ls=(0, (2, 2)), label=f"{method}, stable")
    ax.set_xlabel(r"Gauss--Jacobi nodes $M$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-16, 1e-4)
    ax.set_title("raw quotient vs stabilized quotient")
    finish_axis(ax, legend=True)
    panel_label(ax, "b")

    ax = axes[2]
    ax.loglog(M_gj, tau_min, color="0.25", marker="o", ms=3.0, lw=1.2, label=r"$\tau_{\min}$")
    ax.loglog(M_gj, cancellation_I, color=COLORS["GJ-I"], lw=1.1, label=r"$\epsilon_{mach}/\tau_{\min}$")
    ax.loglog(M_gj, cancellation_II, color=COLORS["GJ-II"], lw=1.1, label=r"$\epsilon_{mach}/\tau_{\min}^2$")
    ref = tau_min[0] * (M_gj / M_gj[0]) ** (-2.0)
    ax.loglog(M_gj, ref, color="0.65", lw=1.0, ls=(0, (3, 2)), label=r"$M^{-2}$")
    ax.set_xlabel(r"Gauss--Jacobi nodes $M$")
    ax.set_ylabel("diagnostic scale")
    ax.set_title("endpoint-node diagnostic")
    finish_axis(ax, legend=True)
    panel_label(ax, "c")

    save_figure(fig, outdir, "fig02_M_sweep_exp_diagnostic")

    return {
        "M_mc": Ms.tolist(),
        "M_gj": M_gj.tolist(),
        "exact": exact,
        "mc_stats": {k: {kk: vv.tolist() for kk, vv in d.items()} for k, d in mc_stats.items()},
        "gj_relative_error": {k: v.tolist() for k, v in gj_err.items()},
        "tau_min": tau_min.tolist(),
        "cancellation_I": cancellation_I.tolist(),
        "cancellation_II": cancellation_II.tolist(),
        "settings": {"t": t, "alpha": alpha, "lambda": lam, "eps_mc1": eps_mc1, "eps_mc2": eps_mc2, "eps_gj2_drop": eps_gj2_drop, "mc_repeats": mc_repeats},
    }


def experiment_nonsmooth(
    outdir: Path,
    *,
    seed: int = 229,
    t: float = 1.5,
    alpha: float = 1.5,
    beta_power: float = 1.35,
    tc_interior: float = 0.7,
) -> Dict:
    smooth = ExponentialFunction(lam=-1.0)
    endpoint = ShiftedPowerFunction(beta=beta_power, tc=0.0)
    interior = ShiftedPowerFunction(beta=beta_power, tc=tc_interior)
    funcs = {
        "smooth exponential": smooth,
        rf"endpoint power $t^{{{beta_power:.2f}}}$": endpoint,
        rf"interior kink $(t-t_c)_+^{{{beta_power:.2f}}}$": interior,
    }
    colors = [COLORS["smooth"], COLORS["endpoint"], COLORS["interior"]]
    M_gj = np.array([10, 20, 40, 80, 160, 320, 640, 1280], dtype=int)

    gjI_err = {}
    gjII_err = {}
    for label, fun in funcs.items():
        exact = fun.exact_caputo(t, alpha)
        gjI_err[label] = np.array([rel_err(caputo_gj(fun, t, alpha, int(M), "GJ-I"), exact) for M in M_gj])
        gjII_err[label] = np.array([rel_err(caputo_gj(fun, t, alpha, int(M), "GJ-II"), exact) for M in M_gj])

    alphas = np.linspace(1.05, 1.95, 80)
    M_fixed_gj = 80
    M_fixed_mc = 8000
    err_alpha = {m: [] for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]}
    for i, a in enumerate(alphas):
        exact = interior.exact_caputo(t, float(a))
        tau = beta_samples(float(a), M_fixed_mc, np.random.default_rng(seed + i))
        err_alpha["MC-I"].append(rel_err(caputo_mc(interior, t, float(a), M_fixed_mc, "MC-I", tau=tau, eps_abs=1e-10), exact))
        err_alpha["MC-II"].append(rel_err(caputo_mc(interior, t, float(a), M_fixed_mc, "MC-II", tau=tau, eps_abs=1e-7), exact))
        err_alpha["GJ-I"].append(rel_err(caputo_gj(interior, t, float(a), M_fixed_gj, "GJ-I"), exact))
        err_alpha["GJ-II"].append(rel_err(caputo_gj(interior, t, float(a), M_fixed_gj, "GJ-II"), exact))
    err_alpha = {k: np.asarray(v) for k, v in err_alpha.items()}

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM_TO_IN, 58 * MM_TO_IN), constrained_layout=True)

    grid = np.linspace(0.0, 1.7, 600)
    ax = axes[0]
    ax.plot(grid, smooth.f(grid), color=COLORS["smooth"], lw=1.35, label=r"$e^{-t}$")
    ax.plot(grid, endpoint.f(grid), color=COLORS["endpoint"], lw=1.35, label=rf"$t^{{{beta_power:.2f}}}$")
    ax.plot(grid, interior.f(grid), color=COLORS["interior"], lw=1.35, label=rf"$(t-t_c)_+^{{{beta_power:.2f}}}$")
    ax.axvline(tc_interior, color=COLORS["interior"], lw=0.85, ls=(0, (2, 2)))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"test function $f(t)$")
    ax.set_title("smooth vs non-smooth tests")
    finish_axis(ax, legend=True)
    panel_label(ax, "a")

    ax = axes[1]
    for (label, vals), color in zip(gjI_err.items(), colors):
        ax.loglog(M_gj, np.maximum(vals, ERR_FLOOR), marker="o", ms=3.0, lw=1.2, color=color, label=label)
    ax.set_xlabel(r"Gauss--Jacobi nodes $M$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-16, 3e-1)
    ax.set_title(r"GJ-I regularity controls convergence")
    finish_axis(ax, legend=True)
    panel_label(ax, "b")

    ax = axes[2]
    for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        ax.semilogy(alphas, np.maximum(err_alpha[m], ERR_FLOOR), color=COLORS[m], lw=1.2, ls=LINESTYLES.get(m, "-"), label=m)
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-8, 2)
    ax.set_title(r"interior non-smooth point")
    finish_axis(ax, legend=True)
    panel_label(ax, "c")

    save_figure(fig, outdir, "fig03_nonsmooth_tests")

    return {
        "M_gj": M_gj.tolist(),
        "gjI_relative_error": {k: v.tolist() for k, v in gjI_err.items()},
        "gjII_relative_error": {k: v.tolist() for k, v in gjII_err.items()},
        "alpha_sweep_interior_kink": {
            "alphas": alphas.tolist(),
            "relative_error": {k: v.tolist() for k, v in err_alpha.items()},
        },
        "settings": {"t": t, "alpha": alpha, "beta": beta_power, "tc_interior": tc_interior, "M_fixed_gj": M_fixed_gj, "M_fixed_mc": M_fixed_mc},
    }


def write_analysis_notes(outdir: Path) -> None:
    notes = r"""# MCfd refactored validation notes

This package is generated from the estimator structure in the uploaded `MCfd.ipynb`.

## What is kept from MCfd.ipynb

- Smooth benchmark: `f(t)=exp(lambda*t)` with `lambda=-1`, `t=1.5`.
- Exact derivative: `lambda^2 t^(2-alpha) E_{1,3-alpha}(lambda*t)`.
- MC sampling: `tau ~ Beta(2-alpha, 1)`.
- GJ nodes: `roots_jacobi(M, 0, 1-alpha)` and `tau=(x+1)/2`.
- Four formulas: MC-I, MC-II, GJ-I, GJ-II.

## What is added

1. Alpha diagnostic: `P(tau < eps/t) = (eps/t)^(2-alpha)` shows why MC estimators degrade as alpha approaches 2.
2. M diagnostic: GJ nodes satisfy roughly `tau_min ~ M^{-2}`.  Direct quotients contain removable singularities, so round-off can grow when M is too large.
3. Stable exponential kernels: raw GJ quotients are compared with `expm1`/Taylor-stabilized quotients.
4. Non-smooth test: `f(t)=(t-t_c)_+^beta`, with closed-form Caputo derivative, to test the role of regularity assumptions.

## Small correction relative to the original plotting cells

The original log-error cell used `plt.ylim(1e-16, 0.0)`, which is invalid for a logarithmic axis.  The new plots use positive upper limits and a small visual floor.
"""
    (outdir / "analysis_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCfd refactored validation experiments.")
    parser.add_argument("--outdir", type=Path, default=Path("MCfd_refactored_outputs"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=229, help="Random seed.")
    parser.add_argument("--mc-repeats", type=int, default=8, help="Monte Carlo repeats for M-sweep bands.")
    args = parser.parse_args()

    set_nature_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = {
        "alpha_sweep_exp": experiment_alpha_sweep(args.outdir, seed=args.seed),
        "M_sweep_exp_diagnostic": experiment_M_sweep(args.outdir, seed=args.seed, mc_repeats=args.mc_repeats),
        "nonsmooth_tests": experiment_nonsmooth(args.outdir, seed=args.seed),
    }
    with open(args.outdir / "MCfd_refactored_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_analysis_notes(args.outdir)

    print(f"Saved figures, data and notes to: {args.outdir.resolve()}")
    for stem in ["fig01_alpha_sweep_exp", "fig02_M_sweep_exp_diagnostic", "fig03_nonsmooth_tests"]:
        print(f"  {args.outdir / (stem + '.pdf')}")
        print(f"  {args.outdir / (stem + '.png')}")
    print(f"  {args.outdir / 'MCfd_refactored_results.json'}")
    print(f"  {args.outdir / 'analysis_notes.md'}")


if __name__ == "__main__":
    main()
