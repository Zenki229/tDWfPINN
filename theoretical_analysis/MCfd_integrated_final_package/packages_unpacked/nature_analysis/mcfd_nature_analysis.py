"""
MCfd_nature_analysis.py

A function-based, reproducible analysis suite for the four fractional-derivative
approximations used in MCfd.ipynb:
    MC-I, MC-II, GJ-I, GJ-II.

The script covers three questions:
1. alpha-dependence for a smooth exponential test function;
2. M-dependence of the error, including a diagnostic for GJ round-off growth;
3. a non-smooth shifted power / hinge test function with closed-form Caputo derivative.

Run:
    python mcfd_nature_analysis.py

Outputs are saved under ./mcfd_figures by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.special as sp
from scipy.special import roots_jacobi
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 0. Figure style: clean, journal-like, colorblind-friendly
# -----------------------------------------------------------------------------

MM_TO_IN = 1.0 / 25.4


def set_nature_style() -> None:
    """Matplotlib settings close to a Nature-style figure: compact, clean, readable."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


COLORS: Mapping[str, str] = {
    "exact": "#1A1A1A",
    "MC-I": "#0072B2",
    "MC-II": "#D55E00",
    "GJ-I": "#009E73",
    "GJ-II": "#CCB974",
    "GJ-I stable": "#56B4E9",
    "GJ-II stable": "#9467BD",
}

LINESTYLES: Mapping[str, str] = {
    "exact": "-",
    "MC-I": "--",
    "MC-II": "--",
    "GJ-I": "-.",
    "GJ-II": "-.",
    "GJ-I stable": ":",
    "GJ-II stable": ":",
}

MARKERS: Mapping[str, str] = {
    "MC-I": "o",
    "MC-II": "s",
    "GJ-I": "^",
    "GJ-II": "D",
    "GJ-I stable": "v",
    "GJ-II stable": "P",
}

METHODS: Tuple[str, ...] = ("MC-I", "MC-II", "GJ-I", "GJ-II")
ERR_FLOOR = 1e-16  # visual floor for log-error plots; avoids meaningless 1e-300 axes.


def _finish_axes(ax: mpl.axes.Axes, *, legend: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color="0.88", linewidth=0.6)
    ax.grid(True, which="minor", color="0.93", linewidth=0.4)
    if legend:
        ax.legend(handlelength=2.5, borderaxespad=0.4)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# -----------------------------------------------------------------------------
# 1. Test functions and exact Caputo derivatives
# -----------------------------------------------------------------------------


def mittag_leffler_series(
    z: float | complex,
    alpha: float,
    beta: float,
    *,
    tol: float = 2e-15,
    max_terms: int = 4000,
) -> complex:
    """
    Lightweight Mittag-Leffler E_{alpha,beta}(z) by direct series.

    For the present smooth benchmark z = lambda * t = -1.5, alpha = 1,
    beta in (1, 2), this converges rapidly. If pymittagleffler is available,
    ExpFunction below will use it automatically.
    """
    zc = complex(z)
    total = 0.0 + 0.0j
    zk = 1.0 + 0.0j
    for k in range(max_terms):
        if k > 0:
            zk *= zc
        term = zk / sp.gamma(alpha * k + beta)
        total += term
        if abs(term) <= tol * max(1.0, abs(total)):
            return total
    return total


try:  # optional dependency already used in the original notebook
    from pymittagleffler import mittag_leffler as _pymittag_leffler
except Exception:  # pragma: no cover
    _pymittag_leffler = None


@dataclass(frozen=True)
class ExpFunction:
    """f(t) = exp(lambda * t), with the same exact formula as MCfd.ipynb."""

    lam: float = -1.0

    def value(self, t: np.ndarray | float) -> np.ndarray | float:
        return np.exp(self.lam * t)

    def d1(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.lam * np.exp(self.lam * t)

    def exact_caputo(self, t: float, alpha: float) -> float:
        if np.isclose(alpha, 1.0):
            return float(self.d1(t))
        if np.isclose(alpha, 2.0):
            return float((self.lam**2) * np.exp(self.lam * t))
        z = self.lam * t
        beta = 3.0 - alpha
        if _pymittag_leffler is not None:
            val = _pymittag_leffler(z, 1.0, beta)
        else:
            val = mittag_leffler_series(z, 1.0, beta)
        return float(np.real((self.lam**2) * t ** (2.0 - alpha) * val))

    def k1_stable(self, t: float, h: np.ndarray) -> np.ndarray:
        """Stable [f'(t)-f'(t-h)]/h for small h."""
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h)
        small = np.abs(h) < 1e-5
        e = math.exp(self.lam * t)
        # Taylor: f''(t) - h f'''(t)/2 + h^2 f''''(t)/6 - h^3 f'''''(t)/24 + ...
        out[small] = (
            (self.lam**2) * e
            - 0.5 * (self.lam**3) * e * h[small]
            + (self.lam**4) * e * h[small] ** 2 / 6.0
            - (self.lam**5) * e * h[small] ** 3 / 24.0
        )
        hs = h[~small]
        out[~small] = self.lam * e * (-np.expm1(-self.lam * hs)) / hs
        return out

    def k2_stable(self, t: float, h: np.ndarray) -> np.ndarray:
        """Stable [f(t)-f(t-h)-h f'(t)]/h^2 for small h."""
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h)
        small = np.abs(h) < 1e-4
        e = math.exp(self.lam * t)
        # Taylor: -f''(t)/2 + h f'''(t)/6 - h^2 f''''(t)/24 + h^3 f'''''(t)/120 + ...
        out[small] = (
            -0.5 * (self.lam**2) * e
            + (self.lam**3) * e * h[small] / 6.0
            - (self.lam**4) * e * h[small] ** 2 / 24.0
            + (self.lam**5) * e * h[small] ** 3 / 120.0
        )
        hs = h[~small]
        out[~small] = e * (-np.expm1(-self.lam * hs) - self.lam * hs) / (hs**2)
        return out


@dataclass(frozen=True)
class HingePowerFunction:
    r"""
    Non-smooth benchmark:
        f(t) = (t - c)_+^beta, beta in (1, 2).

    It is C^1 but not C^2 at t=c. For alpha in (1,2), its Caputo derivative is
        D_C^alpha f(t) = Gamma(beta+1) / Gamma(beta+1-alpha) * (t-c)_+^{beta-alpha}, t>c,
    and zero for t<=c. This makes it a clean non-smooth test with an exact answer.
    """

    beta: float = 1.25
    c: float = 0.7

    def value(self, t: np.ndarray | float) -> np.ndarray | float:
        x = np.maximum(np.asarray(t) - self.c, 0.0)
        y = x**self.beta
        return float(y) if np.ndim(y) == 0 else y

    def d1(self, t: np.ndarray | float) -> np.ndarray | float:
        x = np.maximum(np.asarray(t) - self.c, 0.0)
        y = self.beta * x ** (self.beta - 1.0)
        return float(y) if np.ndim(y) == 0 else y

    def exact_caputo(self, t: float, alpha: float) -> float:
        if t <= self.c:
            return 0.0
        return float(
            sp.gamma(self.beta + 1.0)
            / sp.gamma(self.beta + 1.0 - alpha)
            * (t - self.c) ** (self.beta - alpha)
        )


# -----------------------------------------------------------------------------
# 2. MC-I / MC-II / GJ-I / GJ-II formulas in function form
# -----------------------------------------------------------------------------


def _as_array(h: np.ndarray | float) -> np.ndarray:
    return np.asarray(h, dtype=float)


def kernel_type1(case, t: float, h: np.ndarray, *, stable: bool = False) -> np.ndarray:
    """K1(h) = [f'(t) - f'(t-h)] / h."""
    h = _as_array(h)
    if stable and hasattr(case, "k1_stable"):
        return case.k1_stable(t, h)
    return (case.d1(t) - case.d1(t - h)) / h


def kernel_type2(case, t: float, h: np.ndarray, *, stable: bool = False) -> np.ndarray:
    """K2(h) = [f(t) - f(t-h) - h f'(t)] / h^2."""
    h = _as_array(h)
    if stable and hasattr(case, "k2_stable"):
        return case.k2_stable(t, h)
    return (case.value(t) - case.value(t - h) - h * case.d1(t)) / (h**2)


def caputo_mc_type1(
    case,
    t: float,
    alpha: float,
    M: int,
    rng: np.random.Generator,
    *,
    taus: Optional[np.ndarray] = None,
    eps: float = 1e-16,
    stable: bool = False,
) -> float:
    """MC-I from MCfd.ipynb, written with K1 and Beta(2-alpha,1) sampling."""
    if taus is None:
        taus = rng.beta(2.0 - alpha, 1.0, size=M)
    else:
        taus = np.asarray(taus[:M], dtype=float)
    h = t * np.maximum(taus, eps / t)
    coeff = sp.gamma(2.0 - alpha)
    part1 = (case.d1(t) - case.d1(0.0)) * t ** (1.0 - alpha)
    integral = np.mean(kernel_type1(case, t, h, stable=stable)) / (2.0 - alpha)
    return float((part1 + (alpha - 1.0) * t ** (2.0 - alpha) * integral) / coeff)


def caputo_mc_type2(
    case,
    t: float,
    alpha: float,
    M: int,
    rng: np.random.Generator,
    *,
    taus: Optional[np.ndarray] = None,
    eps: float = 1e-7,
    stable: bool = False,
) -> float:
    """MC-II from MCfd.ipynb, written with K2 and Beta(2-alpha,1) sampling."""
    if taus is None:
        taus = rng.beta(2.0 - alpha, 1.0, size=M)
    else:
        taus = np.asarray(taus[:M], dtype=float)
    h = t * np.maximum(taus, eps / t)
    coeff = sp.gamma(2.0 - alpha)
    part1 = (case.d1(t) - case.d1(0.0)) * t ** (1.0 - alpha)
    part2 = -(alpha - 1.0) * (case.value(t) - case.value(0.0) - t * case.d1(t)) * t ** (-alpha)
    integral = np.mean(kernel_type2(case, t, h, stable=stable)) / (2.0 - alpha)
    return float((part1 + part2 - (alpha - 1.0) * alpha * t ** (2.0 - alpha) * integral) / coeff)


def _gj_nodes(alpha: float, M: int) -> Tuple[np.ndarray, np.ndarray]:
    # roots_jacobi returns weights for ∫_{-1}^{1} (1-x)^0 (1+x)^(1-alpha) g(x) dx.
    x, w = roots_jacobi(M, 0.0, 1.0 - alpha)
    tau = (x + 1.0) / 2.0
    # Therefore ∫_0^1 tau^(1-alpha) F(tau)d tau = 2^(alpha-2) Σ w_i F((x_i+1)/2).
    w_tau = (2.0 ** (alpha - 2.0)) * w
    return tau, w_tau


def caputo_gj_type1(
    case,
    t: float,
    alpha: float,
    M: int,
    *,
    stable: bool = False,
) -> float:
    """GJ-I from MCfd.ipynb, using Gauss-Jacobi quadrature."""
    tau, w_tau = _gj_nodes(alpha, M)
    h = t * tau
    coeff = sp.gamma(2.0 - alpha)
    part1 = (case.d1(t) - case.d1(0.0)) * t ** (1.0 - alpha)
    integral = np.sum(w_tau * kernel_type1(case, t, h, stable=stable))
    return float((part1 + (alpha - 1.0) * t ** (2.0 - alpha) * integral) / coeff)


def caputo_gj_type2(
    case,
    t: float,
    alpha: float,
    M: int,
    *,
    eps: float = 1e-14,
    stable: bool = False,
) -> float:
    """GJ-II from MCfd.ipynb, using Gauss-Jacobi quadrature."""
    tau, w_tau = _gj_nodes(alpha, M)
    # Keep nodes strictly away from zero if the generic kernel is used.
    if not stable and eps is not None and eps > 0.0:
        keep = tau > eps / t
        tau = tau[keep]
        w_tau = w_tau[keep]
    h = t * tau
    coeff = sp.gamma(2.0 - alpha)
    part1 = (case.d1(t) - case.d1(0.0)) * t ** (1.0 - alpha)
    part2 = -(alpha - 1.0) * (case.value(t) - case.value(0.0) - t * case.d1(t)) * t ** (-alpha)
    integral = np.sum(w_tau * kernel_type2(case, t, h, stable=stable))
    return float((part1 + part2 - (alpha - 1.0) * alpha * t ** (2.0 - alpha) * integral) / coeff)


def evaluate_one(
    case,
    t: float,
    alpha: float,
    *,
    M_mc: int,
    M_gj: int,
    rng: np.random.Generator,
    stable_gj: bool = False,
) -> Dict[str, float]:
    exact = case.exact_caputo(t, alpha)
    return {
        "exact": exact,
        "MC-I": caputo_mc_type1(case, t, alpha, M_mc, rng),
        "MC-II": caputo_mc_type2(case, t, alpha, M_mc, rng),
        "GJ-I": caputo_gj_type1(case, t, alpha, M_gj, stable=stable_gj),
        "GJ-II": caputo_gj_type2(case, t, alpha, M_gj, stable=stable_gj),
    }


# -----------------------------------------------------------------------------
# 3. Experiment drivers
# -----------------------------------------------------------------------------


def alpha_sweep(
    case,
    *,
    t: float = 1.5,
    alphas: Optional[np.ndarray] = None,
    M_mc: int = 10_000,
    M_gj: int = 100,
    mc_repeats: int = 8,
    seed: int = 202601,
    stable_gj: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Sweep alpha. MC values are averaged over repeats; GJ is deterministic.
    Returns arrays for approximations and absolute errors.
    """
    if alphas is None:
        alphas = np.linspace(1.01, 1.99, 100)
    alphas = np.asarray(alphas, dtype=float)
    rng = np.random.default_rng(seed)

    exact = np.array([case.exact_caputo(t, a) for a in alphas], dtype=float)
    out: Dict[str, np.ndarray] = {"alpha": alphas, "exact": exact}

    # GJ deterministic
    out["GJ-I"] = np.array([caputo_gj_type1(case, t, a, M_gj, stable=stable_gj) for a in alphas])
    out["GJ-II"] = np.array([caputo_gj_type2(case, t, a, M_gj, stable=stable_gj) for a in alphas])

    # MC repeats
    mc_vals_1 = np.zeros((mc_repeats, len(alphas)))
    mc_vals_2 = np.zeros((mc_repeats, len(alphas)))
    for r in range(mc_repeats):
        for j, a in enumerate(alphas):
            taus = rng.beta(2.0 - a, 1.0, size=M_mc)
            mc_vals_1[r, j] = caputo_mc_type1(case, t, a, M_mc, rng, taus=taus)
            mc_vals_2[r, j] = caputo_mc_type2(case, t, a, M_mc, rng, taus=taus)
    out["MC-I"] = np.mean(mc_vals_1, axis=0)
    out["MC-II"] = np.mean(mc_vals_2, axis=0)
    out["MC-I_abs_err_q25"] = np.quantile(np.abs(mc_vals_1 - exact), 0.25, axis=0)
    out["MC-I_abs_err_q75"] = np.quantile(np.abs(mc_vals_1 - exact), 0.75, axis=0)
    out["MC-II_abs_err_q25"] = np.quantile(np.abs(mc_vals_2 - exact), 0.25, axis=0)
    out["MC-II_abs_err_q75"] = np.quantile(np.abs(mc_vals_2 - exact), 0.75, axis=0)

    for m in METHODS:
        out[f"{m}_abs_err"] = np.abs(out[m] - exact)
        out[f"{m}_rel_err"] = np.abs(out[m] - exact) / np.maximum(np.abs(exact), np.finfo(float).tiny)
    return out


def m_sweep(
    case,
    *,
    t: float = 1.5,
    alpha: float = 1.5,
    Ms: Optional[np.ndarray] = None,
    mc_repeats: int = 32,
    max_gj_M: int = 640,
    seed: int = 202602,
    include_stable_gj: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Sweep M. MC curves use nested samples and are summarized by median absolute error.
    GJ curves are deterministic and computed up to max_gj_M by default.
    """
    if Ms is None:
        Ms = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240], dtype=int)
    Ms = np.asarray(Ms, dtype=int)
    rng = np.random.default_rng(seed)
    exact = case.exact_caputo(t, alpha)

    out: Dict[str, np.ndarray] = {"M": Ms, "exact": np.full_like(Ms, exact, dtype=float)}

    maxM = int(np.max(Ms))
    mc1_err = np.zeros((mc_repeats, len(Ms)))
    mc2_err = np.zeros((mc_repeats, len(Ms)))
    for r in range(mc_repeats):
        taus_max = rng.beta(2.0 - alpha, 1.0, size=maxM)
        for j, M in enumerate(Ms):
            tau = taus_max[:M]
            mc1_err[r, j] = abs(caputo_mc_type1(case, t, alpha, M, rng, taus=tau) - exact)
            mc2_err[r, j] = abs(caputo_mc_type2(case, t, alpha, M, rng, taus=tau) - exact)
    for key, arr in [("MC-I", mc1_err), ("MC-II", mc2_err)]:
        out[f"{key}_abs_err"] = np.median(arr, axis=0)
        out[f"{key}_abs_err_q25"] = np.quantile(arr, 0.25, axis=0)
        out[f"{key}_abs_err_q75"] = np.quantile(arr, 0.75, axis=0)

    # GJ deterministic. Large roots_jacobi can be slow/ill-conditioned, so use max_gj_M.
    for method in ("GJ-I", "GJ-II"):
        out[f"{method}_abs_err"] = np.full(len(Ms), np.nan)
    if include_stable_gj:
        out["GJ-I stable_abs_err"] = np.full(len(Ms), np.nan)
        out["GJ-II stable_abs_err"] = np.full(len(Ms), np.nan)

    out["GJ_tau_min"] = np.full(len(Ms), np.nan)
    out["roundoff_indicator_I"] = np.full(len(Ms), np.nan)
    out["roundoff_indicator_II"] = np.full(len(Ms), np.nan)

    eps_machine = np.finfo(float).eps
    for j, M in enumerate(Ms):
        if M > max_gj_M:
            continue
        try:
            tau, _ = _gj_nodes(alpha, int(M))
            tau_min = float(np.min(tau))
            out["GJ_tau_min"][j] = tau_min
            out["roundoff_indicator_I"][j] = eps_machine / max(tau_min, eps_machine)
            out["roundoff_indicator_II"][j] = eps_machine / max(tau_min, eps_machine) ** 2
            out["GJ-I_abs_err"][j] = abs(caputo_gj_type1(case, t, alpha, int(M), stable=False) - exact)
            out["GJ-II_abs_err"][j] = abs(caputo_gj_type2(case, t, alpha, int(M), stable=False) - exact)
            if include_stable_gj:
                out["GJ-I stable_abs_err"][j] = abs(caputo_gj_type1(case, t, alpha, int(M), stable=True) - exact)
                out["GJ-II stable_abs_err"][j] = abs(caputo_gj_type2(case, t, alpha, int(M), stable=True) - exact)
        except Exception:
            # Keep NaN to make failure visible in the plot/table.
            pass
    return out


# -----------------------------------------------------------------------------
# 4. Plotting
# -----------------------------------------------------------------------------


def plot_alpha_sweep(
    data: Mapping[str, np.ndarray],
    *,
    title: str,
    savepath: Optional[Path] = None,
) -> mpl.figure.Figure:
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM_TO_IN, 72 * MM_TO_IN))
    alpha = data["alpha"]

    ax = axes[0]
    ax.plot(alpha, data["exact"], color=COLORS["exact"], linestyle="-", linewidth=1.8, label="exact")
    for m in METHODS:
        ax.plot(alpha, data[m], color=COLORS[m], linestyle=LINESTYLES[m], linewidth=1.25, label=m)
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel(r"$D_t^\alpha f(t)$")
    ax.set_title(title)
    _panel_label(ax, "a")
    _finish_axes(ax, legend=True)

    ax = axes[1]
    for m in METHODS:
        y = np.maximum(data[f"{m}_abs_err"], ERR_FLOOR)
        ax.plot(alpha, y, color=COLORS[m], linestyle=LINESTYLES[m], linewidth=1.35, label=m)
        if m in ("MC-I", "MC-II") and f"{m}_abs_err_q25" in data:
            lo = np.maximum(data[f"{m}_abs_err_q25"], ERR_FLOOR)
            hi = np.maximum(data[f"{m}_abs_err_q75"], ERR_FLOOR)
            ax.fill_between(alpha, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("absolute error")
    _panel_label(ax, "b")
    _finish_axes(ax, legend=True)

    fig.tight_layout(w_pad=2.0)
    if savepath is not None:
        save_figure(fig, savepath)
    return fig


def plot_m_sweep(
    data: Mapping[str, np.ndarray],
    *,
    alpha: float,
    title: str,
    savepath: Optional[Path] = None,
    include_stable_gj: bool = True,
) -> mpl.figure.Figure:
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM_TO_IN, 72 * MM_TO_IN))
    M = data["M"]

    ax = axes[0]
    plot_methods = list(METHODS)
    if include_stable_gj:
        plot_methods += ["GJ-I stable", "GJ-II stable"]
    for m in plot_methods:
        key = f"{m}_abs_err"
        if key not in data:
            continue
        y = np.maximum(np.asarray(data[key], dtype=float), ERR_FLOOR)
        ax.plot(
            M,
            y,
            marker=MARKERS.get(m, "o"),
            markersize=3.2,
            linewidth=1.15,
            linestyle=LINESTYLES.get(m, "-"),
            color=COLORS.get(m, "0.4"),
            label=m,
        )
        if m in ("MC-I", "MC-II") and f"{m}_abs_err_q25" in data:
            lo = np.maximum(np.asarray(data[f"{m}_abs_err_q25"], dtype=float), ERR_FLOOR)
            hi = np.maximum(np.asarray(data[f"{m}_abs_err_q75"], dtype=float), ERR_FLOOR)
            ax.fill_between(M, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of samples/nodes $M$")
    ax.set_ylabel("absolute error")
    ax.set_title(title + rf", $\alpha={alpha:.2f}$")
    _panel_label(ax, "a")
    _finish_axes(ax, legend=True)

    ax = axes[1]
    for key, label, color in [
        ("GJ_tau_min", r"min GJ node $\tau_{\min}$", "0.25"),
        ("roundoff_indicator_I", r"$\epsilon/\tau_{\min}$", COLORS["GJ-I"]),
        ("roundoff_indicator_II", r"$\epsilon/\tau_{\min}^2$", COLORS["GJ-II"]),
    ]:
        if key in data:
            ax.plot(M, data[key], marker="o", markersize=3.0, linewidth=1.15, label=label, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"GJ nodes $M$")
    ax.set_ylabel("scale")
    ax.set_title("endpoint-node and cancellation diagnostic")
    _panel_label(ax, "b")
    _finish_axes(ax, legend=True)

    fig.tight_layout(w_pad=2.0)
    if savepath is not None:
        save_figure(fig, savepath)
    return fig


def save_figure(fig: mpl.figure.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    fig.savefig(path_no_suffix.with_suffix(".png"))


def save_npz(data: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in data.items()})


# -----------------------------------------------------------------------------
# 5. Main run
# -----------------------------------------------------------------------------


def main(output_dir: str | Path = "mcfd_figures") -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    smooth = ExpFunction(lam=-1.0)
    nonsmooth = HingePowerFunction(beta=1.25, c=0.7)

    # Smooth alpha sweep: this reproduces the MCfd.ipynb alpha experiment, but
    # with repeated MC draws and IQR bands.
    alpha_data = alpha_sweep(
        smooth,
        t=1.5,
        alphas=np.linspace(1.01, 1.99, 100),
        M_mc=10_000,
        M_gj=100,
        mc_repeats=8,
        seed=202601,
        stable_gj=False,
    )
    save_npz(alpha_data, outdir / "smooth_alpha_sweep.npz")
    plot_alpha_sweep(alpha_data, title="smooth benchmark: $f(t)=e^{-t}$", savepath=outdir / "smooth_alpha_sweep")

    # M-dependence: GJ is shown only up to max_gj_M by default, because large M
    # can be slower and often dominated by cancellation near tau=0.
    m_data = m_sweep(
        smooth,
        t=1.5,
        alpha=1.5,
        Ms=np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]),
        mc_repeats=32,
        max_gj_M=640,
        seed=202602,
        include_stable_gj=True,
    )
    save_npz(m_data, outdir / "smooth_M_sweep.npz")
    plot_m_sweep(m_data, alpha=1.5, title="smooth benchmark", savepath=outdir / "smooth_M_sweep")

    # Non-smooth alpha sweep: a single exact-function test of failure/degradation
    # when f is not C^2.
    nonsmooth_alpha_data = alpha_sweep(
        nonsmooth,
        t=1.5,
        alphas=np.linspace(1.01, 1.95, 90),
        M_mc=20_000,
        M_gj=120,
        mc_repeats=8,
        seed=202603,
        stable_gj=False,
    )
    save_npz(nonsmooth_alpha_data, outdir / "nonsmooth_alpha_sweep.npz")
    plot_alpha_sweep(
        nonsmooth_alpha_data,
        title=r"non-smooth benchmark: $f(t)=(t-0.7)_+^{1.25}$",
        savepath=outdir / "nonsmooth_alpha_sweep",
    )

    nonsmooth_m_data = m_sweep(
        nonsmooth,
        t=1.5,
        alpha=1.5,
        Ms=np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560]),
        mc_repeats=32,
        max_gj_M=640,
        seed=202604,
        include_stable_gj=False,
    )
    save_npz(nonsmooth_m_data, outdir / "nonsmooth_M_sweep.npz")
    plot_m_sweep(
        nonsmooth_m_data,
        alpha=1.5,
        title=r"non-smooth benchmark",
        savepath=outdir / "nonsmooth_M_sweep",
        include_stable_gj=False,
    )

    plt.close("all")
    print(f"Saved figures and data to: {outdir.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCfd alpha/M/non-smooth experiments and save Nature-style figures.")
    parser.add_argument("--output-dir", type=str, default="mcfd_figures", help="Directory for figures and NPZ data files.")
    args = parser.parse_args()
    main(args.output_dir)
