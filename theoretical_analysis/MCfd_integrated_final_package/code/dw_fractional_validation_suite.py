"""
Validation suite for Type-I/Type-II Caputo derivative approximations in the
alpha in (1,2) diffusion-wave setting.

The code is designed to reproduce and extend the MCfd.ipynb-style experiments:
  1. alpha sweep, especially alpha -> 2;
  2. M sweep and the large-M Gauss-Jacobi conditioning issue;
  3. non-smooth f tests where the C^2 / analytic assumptions break down.

Run:
    python dw_fractional_validation_suite.py --outdir dw_validation_outputs --seed 229

Dependencies:
    numpy scipy matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaln, hyp1f1, roots_jacobi

Method = Literal["MC-I", "MC-II", "GJ-I", "GJ-II"]
KernelMode = Literal["raw", "stable"]

# -----------------------------------------------------------------------------
# Plot style: clean, compact, color-blind friendly, journal-like.
# -----------------------------------------------------------------------------

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
}

LINESTYLES = {
    "MC-I": (0, (5, 2)),
    "MC-II": (0, (5, 2)),
    "GJ-I": "-",
    "GJ-II": "-",
    "exact": "-",
}

MARKERS = {
    "MC-I": "o",
    "MC-II": "s",
    "GJ-I": "^",
    "GJ-II": "D",
}


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.14,
        1.06,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Test functions with exact Caputo derivatives.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ExponentialFunction:
    lam: float = -1.0
    name: str = "exp"

    def f(self, s):
        return np.exp(self.lam * np.asarray(s))

    def fp(self, s):
        return self.lam * np.exp(self.lam * np.asarray(s))

    def fpp(self, s):
        return self.lam**2 * np.exp(self.lam * np.asarray(s))

    def exact_caputo(self, t: float, alpha: float) -> float:
        """Caputo D^alpha exp(lambda t), alpha in (1,2).

        D_C^alpha e^{lambda t} = lambda^2 t^{2-alpha} E_{1,3-alpha}(lambda t).
        Numerically evaluated via the equivalent confluent-hypergeometric form.
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
        """Stable K=(f'(t)-f'(t-t*tau))/(t*tau) for exponential f."""
        r = t * tau
        lam = self.lam
        out = -lam * math.exp(lam * t) * np.expm1(-lam * r) / r
        small = np.abs(r) < 1e-7
        if np.any(small):
            # K = f''(t) - r f'''(t)/2 + r^2 f''''(t)/6 + O(r^3)
            out = np.asarray(out, dtype=float)
            out[small] = (
                lam**2 * math.exp(lam * t)
                - 0.5 * r[small] * lam**3 * math.exp(lam * t)
                + (r[small] ** 2) * lam**4 * math.exp(lam * t) / 6.0
            )
        return out

    def kernel_H_stable(self, t: float, tau: np.ndarray) -> np.ndarray:
        """Stable H=(f(t)-f(t-r)-r f'(t))/r^2 for exponential f."""
        r = t * tau
        lam = self.lam
        # 1 - exp(-lam*r) - lam*r = -expm1(-lam*r) - lam*r
        out = math.exp(lam * t) * (-np.expm1(-lam * r) - lam * r) / (r * r)
        small = np.abs(r) < 1e-5
        if np.any(small):
            # H = -f''(t)/2 + r f'''(t)/6 - r^2 f''''(t)/24 + O(r^3)
            out = np.asarray(out, dtype=float)
            out[small] = (
                -0.5 * lam**2 * math.exp(lam * t)
                + r[small] * lam**3 * math.exp(lam * t) / 6.0
                - (r[small] ** 2) * lam**4 * math.exp(lam * t) / 24.0
            )
        return out


@dataclass(frozen=True)
class ShiftedPowerFunction:
    """f(t)=(t-tc)_+^beta.

    For 1<beta<2 this is C^1 but not C^2 at t=tc.  If tc=0, the
    non-smooth point is the initial endpoint; if 0<tc<t_eval, it lies inside
    the memory interval.  Its Caputo derivative is known exactly:

        D_C^alpha (t-tc)_+^beta = Gamma(beta+1)/Gamma(beta+1-alpha)
                                  * (t-tc)_+^{beta-alpha}.
    """

    beta: float = 1.35
    tc: float = 0.25
    name: str = "shifted_power"

    def f(self, s):
        s = np.asarray(s)
        return np.maximum(s - self.tc, 0.0) ** self.beta

    def fp(self, s):
        s = np.asarray(s)
        x = np.maximum(s - self.tc, 0.0)
        return self.beta * x ** (self.beta - 1.0) * (s > self.tc)

    def exact_caputo(self, t: float, alpha: float) -> float:
        if t <= self.tc:
            return 0.0
        return float(
            math.exp(gammaln(self.beta + 1.0) - gammaln(self.beta + 1.0 - alpha))
            * (t - self.tc) ** (self.beta - alpha)
        )


# -----------------------------------------------------------------------------
# Quadrature rules and four schemes.
# -----------------------------------------------------------------------------

def relative_error(approx: float, exact: float) -> float:
    return float(abs(approx - exact) / (abs(exact) + 1e-300))


def gauss_jacobi_rule(alpha: float, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """M-point Gauss-Jacobi rule for int_0^1 phi(tau) tau^{1-alpha} d tau.

    scipy.special.roots_jacobi integrates over [-1,1] with weight
    (1-x)^a(1+x)^b.  With tau=(x+1)/2, choose a=0, b=1-alpha and multiply
    the weights by 2^(alpha-2).
    """
    if not (1.0 < alpha < 2.0):
        raise ValueError("alpha must be in (1,2).")
    x, w = roots_jacobi(M, 0.0, 1.0 - alpha)
    tau = 0.5 * (x + 1.0)
    weights = (2.0 ** (alpha - 2.0)) * w
    return tau.astype(float), weights.astype(float)


def beta_samples(alpha: float, M: int, rng: np.random.Generator) -> np.ndarray:
    """Samples tau~Beta(2-alpha,1), density (2-alpha) tau^{1-alpha}."""
    if not (1.0 < alpha < 2.0):
        raise ValueError("alpha must be in (1,2).")
    return rng.beta(2.0 - alpha, 1.0, size=M).astype(float)


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


def caputo_gj(
    fun,
    t: float,
    alpha: float,
    M: int,
    method: Literal["GJ-I", "GJ-II"],
    kernel_mode: KernelMode = "raw",
) -> float:
    tau, weights = gauss_jacobi_rule(alpha, M)
    pref = 1.0 / gamma(2.0 - alpha)
    term_derivative = (fun.fp(t) - fun.fp(0.0)) / (t ** (alpha - 1.0))
    if method == "GJ-I":
        integral = float(np.sum(weights * kernel_K(fun, t, tau, mode=kernel_mode)))
        val = term_derivative + (alpha - 1.0) * t ** (2.0 - alpha) * integral
    elif method == "GJ-II":
        endpoint = (fun.f(t) - fun.f(0.0) - t * fun.fp(t)) / (t**alpha)
        integral = float(np.sum(weights * kernel_H(fun, t, tau, mode=kernel_mode)))
        val = (
            term_derivative
            - (alpha - 1.0) * endpoint
            - alpha * (alpha - 1.0) * t ** (2.0 - alpha) * integral
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(pref * val)


def caputo_mc(
    fun,
    t: float,
    alpha: float,
    M: int,
    method: Literal["MC-I", "MC-II"],
    rng: Optional[np.random.Generator] = None,
    tau: Optional[np.ndarray] = None,
    eps_abs: float = 1e-7,
) -> float:
    """Monte Carlo Type-I / Type-II estimator with denominator cutoff.

    eps_abs is a physical cutoff in the memory length r=t*tau.  Thus the
    dimensionless cutoff is delta=eps_abs/t and tau_delta=max(tau,delta).
    The numerator is still evaluated at the original random tau.
    """
    if tau is None:
        if rng is None:
            rng = np.random.default_rng(0)
        tau = beta_samples(alpha, M, rng)
    else:
        tau = np.asarray(tau, dtype=float)
        M = tau.size

    delta = eps_abs / t
    tau_delta = np.maximum(tau, delta)
    r = t * tau
    r_delta = t * tau_delta

    pref = 1.0 / gamma(2.0 - alpha)
    term_derivative = (fun.fp(t) - fun.fp(0.0)) / (t ** (alpha - 1.0))

    if method == "MC-I":
        K_delta = (fun.fp(t) - fun.fp(t - r)) / r_delta
        val = (
            term_derivative
            + (alpha - 1.0) / (2.0 - alpha) * t ** (2.0 - alpha) * float(np.mean(K_delta))
        )
    elif method == "MC-II":
        endpoint = (fun.f(t) - fun.f(0.0) - t * fun.fp(t)) / (t**alpha)
        H_delta = (fun.f(t) - fun.f(t - r) - r * fun.fp(t)) / (r_delta * r_delta)
        val = (
            term_derivative
            - (alpha - 1.0) * endpoint
            - alpha
            * (alpha - 1.0)
            / (2.0 - alpha)
            * t ** (2.0 - alpha)
            * float(np.mean(H_delta))
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(pref * val)


# -----------------------------------------------------------------------------
# Experiments.
# -----------------------------------------------------------------------------

def experiment_alpha_sweep(
    outdir: Path,
    seed: int = 229,
    t: float = 1.5,
    lam: float = -1.0,
    M_mc: int = 10_000,
    M_gj: int = 100,
    eps_abs: float = 1e-7,
) -> Dict:
    fun = ExponentialFunction(lam=lam)
    alphas = np.linspace(1.01, 1.99, 100)
    exact = np.array([fun.exact_caputo(t, a) for a in alphas])
    approx: Dict[str, np.ndarray] = {m: np.empty_like(alphas) for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]}

    for i, a in enumerate(alphas):
        rng = np.random.default_rng(seed + i)
        tau = beta_samples(a, M_mc, rng)
        approx["MC-I"][i] = caputo_mc(fun, t, a, M_mc, "MC-I", tau=tau, eps_abs=eps_abs)
        approx["MC-II"][i] = caputo_mc(fun, t, a, M_mc, "MC-II", tau=tau, eps_abs=eps_abs)
        approx["GJ-I"][i] = caputo_gj(fun, t, a, M_gj, "GJ-I", kernel_mode="raw")
        approx["GJ-II"][i] = caputo_gj(fun, t, a, M_gj, "GJ-II", kernel_mode="raw")

    err = {m: np.abs(approx[m] - exact) / np.maximum(np.abs(exact), 1e-300) for m in approx}
    delta = eps_abs / t
    cutoff_mass = delta ** (2.0 - alphas)
    one_over_M = np.full_like(alphas, 1.0 / M_mc)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
    ax = axes[0]
    ax.plot(alphas, exact, color=COLORS["exact"], lw=1.8, label="exact")
    for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        ax.plot(alphas, approx[m], color=COLORS[m], lw=1.2, ls=LINESTYLES[m], label=m)
    ax.set_xlabel(r"Fractional order $\alpha$")
    ax.set_ylabel(r"$D_t^\alpha f(t)$")
    ax.set_xlim(alphas.min(), alphas.max())
    ax.legend(ncol=1, loc="lower left")
    panel_label(ax, "a")

    ax = axes[1]
    for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        ax.semilogy(alphas, err[m], color=COLORS[m], lw=1.3, ls=LINESTYLES[m], label=m)
    ax.set_xlabel(r"Fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_xlim(alphas.min(), alphas.max())
    ax.set_ylim(1e-16, 2)
    panel_label(ax, "b")

    ax = axes[2]
    ax.semilogy(alphas, cutoff_mass, color="#4D4D4D", lw=1.6, label=r"$P(\tau<\epsilon/t)$")
    ax.semilogy(alphas, one_over_M, color="#999999", lw=1.1, ls=(0, (3, 2)), label=r"$1/M_{\rm MC}$")
    ax.set_xlabel(r"Fractional order $\alpha$")
    ax.set_ylabel("Endpoint mass")
    ax.set_xlim(alphas.min(), alphas.max())
    ax.set_ylim(1e-8, 2)
    ax.legend(loc="lower right")
    panel_label(ax, "c")

    save_figure(fig, outdir, "fig_alpha_sweep")

    return {
        "alphas": alphas.tolist(),
        "exact": exact.tolist(),
        "approx": {k: v.tolist() for k, v in approx.items()},
        "relative_error": {k: v.tolist() for k, v in err.items()},
        "cutoff_mass": cutoff_mass.tolist(),
        "one_over_M": one_over_M.tolist(),
        "settings": {"t": t, "lambda": lam, "M_mc": M_mc, "M_gj": M_gj, "eps_abs": eps_abs},
    }


def experiment_M_sweep(
    outdir: Path,
    seed: int = 229,
    t: float = 0.75,
    alpha: float = 1.5,
    lam: float = -1.0,
    eps_abs: float = 1e-7,
    mc_repeats: int = 32,
) -> Dict:
    fun = ExponentialFunction(lam=lam)
    exact = fun.exact_caputo(t, alpha)

    M_mc = (10 * 2 ** np.arange(1, 11)).astype(int)  # 20,...,10240
    M_gj = (10 * 2 ** np.arange(0, 9)).astype(int)   # 10,...,2560

    mc_stats = {}
    for method in ["MC-I", "MC-II"]:
        errs = np.zeros((len(M_mc), mc_repeats))
        for i, M in enumerate(M_mc):
            for r in range(mc_repeats):
                rng = np.random.default_rng(seed + 10_000 * (method == "MC-II") + 100 * i + r)
                val = caputo_mc(fun, t, alpha, int(M), method, rng=rng, eps_abs=eps_abs)
                errs[i, r] = relative_error(val, exact)
        mc_stats[method] = {
            "median": np.median(errs, axis=1),
            "q25": np.quantile(errs, 0.25, axis=1),
            "q75": np.quantile(errs, 0.75, axis=1),
            "all": errs,
        }

    gj_err = {}
    min_tau = np.zeros_like(M_gj, dtype=float)
    for mode in ["raw", "stable"]:
        for method in ["GJ-I", "GJ-II"]:
            vals = []
            for i, M in enumerate(M_gj):
                tau, _ = gauss_jacobi_rule(alpha, int(M))
                min_tau[i] = tau.min()
                vals.append(relative_error(caputo_gj(fun, t, alpha, int(M), method, kernel_mode=mode), exact))
            gj_err[f"{method}-{mode}"] = np.array(vals)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)

    ax = axes[0]
    for method in ["MC-I", "MC-II"]:
        med = mc_stats[method]["median"]
        ax.loglog(M_mc, med, color=COLORS[method], marker=MARKERS[method], ms=3.2, lw=1.3, label=method)
        ax.fill_between(M_mc, mc_stats[method]["q25"], mc_stats[method]["q75"], color=COLORS[method], alpha=0.16, lw=0)
    for method in ["GJ-I", "GJ-II"]:
        ax.loglog(M_gj, gj_err[f"{method}-raw"], color=COLORS[method], marker=MARKERS[method], ms=3.2, lw=1.3, label=method)
    ax.set_xlabel(r"Quadrature points $M$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-16, 2e-1)
    ax.legend(loc="lower left", ncol=1)
    panel_label(ax, "a")

    ax = axes[1]
    for method in ["GJ-I", "GJ-II"]:
        ax.loglog(M_gj, gj_err[f"{method}-raw"], color=COLORS[method], marker=MARKERS[method], ms=3.2, lw=1.25, label=f"{method}, raw")
        ax.loglog(
            M_gj,
            gj_err[f"{method}-stable"],
            color=COLORS[f"{method}-stable"],
            marker=MARKERS[method],
            ms=3.0,
            lw=1.15,
            ls=(0, (2, 2)),
            label=f"{method}, stable",
        )
    ax.set_xlabel(r"Gauss--Jacobi points $M$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-16, 1e-4)
    ax.legend(loc="upper left", ncol=1)
    panel_label(ax, "b")

    ax = axes[2]
    ax.loglog(M_gj, min_tau, color="#4D4D4D", marker="o", ms=3.0, lw=1.3, label=r"$\min_j\tau_j$")
    ref = min_tau[0] * (M_gj / M_gj[0]) ** (-2.0)
    ax.loglog(M_gj, ref, color="#9A9A9A", lw=1.1, ls=(0, (3, 2)), label=r"$M^{-2}$ reference")
    ax.set_xlabel(r"Gauss--Jacobi points $M$")
    ax.set_ylabel("Smallest node")
    ax.legend(loc="lower left")
    panel_label(ax, "c")

    save_figure(fig, outdir, "fig_M_sweep")

    return {
        "M_mc": M_mc.tolist(),
        "M_gj": M_gj.tolist(),
        "exact": exact,
        "mc_stats": {
            k: {kk: vv.tolist() for kk, vv in d.items() if kk != "all"}
            for k, d in mc_stats.items()
        },
        "gj_error": {k: v.tolist() for k, v in gj_err.items()},
        "min_tau": min_tau.tolist(),
        "settings": {"t": t, "alpha": alpha, "lambda": lam, "eps_abs": eps_abs, "mc_repeats": mc_repeats},
    }


def experiment_nonsmooth(
    outdir: Path,
    seed: int = 229,
    t: float = 0.75,
    alpha: float = 1.5,
    beta: float = 1.35,
    tc: float = 0.25,
) -> Dict:
    smooth = ExponentialFunction(lam=-1.0)
    endpoint = ShiftedPowerFunction(beta=beta, tc=0.0, name="endpoint_power")
    interior = ShiftedPowerFunction(beta=beta, tc=tc, name="interior_power")

    funcs = {
        "smooth exponential": smooth,
        rf"endpoint power, $\beta={beta}$": endpoint,
        rf"interior kink, $t_c={tc}$": interior,
    }

    M_gj = (10 * 2 ** np.arange(0, 8)).astype(int)  # 10,...,1280
    err_gj_I = {}
    err_gj_II = {}
    for label, fun in funcs.items():
        exact = fun.exact_caputo(t, alpha)
        err_gj_I[label] = np.array([relative_error(caputo_gj(fun, t, alpha, int(M), "GJ-I"), exact) for M in M_gj])
        err_gj_II[label] = np.array([relative_error(caputo_gj(fun, t, alpha, int(M), "GJ-II"), exact) for M in M_gj])

    alphas = np.linspace(1.05, 1.95, 80)
    M_fixed = 80
    err_alpha = {"GJ-I": [], "GJ-II": [], "MC-I": [], "MC-II": []}
    exact_alpha = []
    for i, a in enumerate(alphas):
        exact = interior.exact_caputo(t, float(a))
        exact_alpha.append(exact)
        tau = beta_samples(float(a), 5000, np.random.default_rng(seed + i))
        err_alpha["MC-I"].append(relative_error(caputo_mc(interior, t, float(a), 5000, "MC-I", tau=tau), exact))
        err_alpha["MC-II"].append(relative_error(caputo_mc(interior, t, float(a), 5000, "MC-II", tau=tau), exact))
        err_alpha["GJ-I"].append(relative_error(caputo_gj(interior, t, float(a), M_fixed, "GJ-I"), exact))
        err_alpha["GJ-II"].append(relative_error(caputo_gj(interior, t, float(a), M_fixed, "GJ-II"), exact))
    err_alpha = {k: np.array(v) for k, v in err_alpha.items()}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)

    grid = np.linspace(0.0, 1.0, 500)
    ax = axes[0]
    ax.plot(grid, smooth.f(grid), color=COLORS["smooth"], lw=1.4, label=r"$e^{-t}$")
    ax.plot(grid, endpoint.f(grid), color=COLORS["endpoint"], lw=1.4, label=rf"$t^{{{beta:.2f}}}$")
    ax.plot(grid, interior.f(grid), color=COLORS["interior"], lw=1.4, label=rf"$(t-t_c)_+^{{{beta:.2f}}}$")
    ax.axvline(tc, color=COLORS["interior"], lw=0.9, ls=(0, (2, 2)))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Test function $f(t)$")
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    labels_order = list(funcs.keys())
    colors_order = [COLORS["smooth"], COLORS["endpoint"], COLORS["interior"]]
    ax = axes[1]
    for label, color in zip(labels_order, colors_order):
        ax.loglog(M_gj, err_gj_I[label], marker="o", ms=3, lw=1.25, color=color, label=label)
    ax.set_xlabel(r"Gauss--Jacobi points $M$")
    ax.set_ylabel("relative error")
    ax.set_title("GJ-I regularity test")
    ax.set_ylim(1e-16, 3e-1)
    ax.legend(loc="lower left")
    panel_label(ax, "b")

    ax = axes[2]
    for m in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        ax.semilogy(alphas, err_alpha[m], color=COLORS[m], lw=1.25, ls=LINESTYLES.get(m, "-"), label=m)
    ax.set_xlabel(r"Fractional order $\alpha$")
    ax.set_ylabel("relative error")
    ax.set_title(r"interior kink $(t-t_c)_+^\beta$")
    ax.set_ylim(1e-8, 2)
    ax.legend(loc="upper left")
    panel_label(ax, "c")

    save_figure(fig, outdir, "fig_nonsmooth")

    return {
        "M_gj": M_gj.tolist(),
        "gj_I_error": {k: v.tolist() for k, v in err_gj_I.items()},
        "gj_II_error": {k: v.tolist() for k, v in err_gj_II.items()},
        "alpha_sweep_nonsmooth": {
            "alphas": alphas.tolist(),
            "exact": exact_alpha,
            "errors": {k: v.tolist() for k, v in err_alpha.items()},
        },
        "settings": {"t": t, "alpha": alpha, "beta": beta, "tc": tc, "M_fixed_alpha": M_fixed},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fractional-derivative validation experiments.")
    parser.add_argument("--outdir", type=Path, default=Path("dw_validation_outputs"), help="Directory for figures and data.")
    parser.add_argument("--seed", type=int, default=229, help="Random seed.")
    parser.add_argument("--mc-repeats", type=int, default=32, help="Monte Carlo repeats for M-sweep bands.")
    args = parser.parse_args()

    set_nature_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = {
        "alpha_sweep": experiment_alpha_sweep(args.outdir, seed=args.seed),
        "M_sweep": experiment_M_sweep(args.outdir, seed=args.seed, mc_repeats=args.mc_repeats),
        "nonsmooth": experiment_nonsmooth(args.outdir, seed=args.seed),
    }

    with open(args.outdir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved figures and data to: {args.outdir.resolve()}")
    print("Figures:")
    for name in ["fig_alpha_sweep", "fig_M_sweep", "fig_nonsmooth"]:
        print(f"  - {args.outdir / (name + '.pdf')}")
        print(f"  - {args.outdir / (name + '.png')}")
    print(f"Data: {args.outdir / 'validation_results.json'}")


if __name__ == "__main__":
    main()
