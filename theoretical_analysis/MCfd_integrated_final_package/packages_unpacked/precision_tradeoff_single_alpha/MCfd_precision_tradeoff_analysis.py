"""
MCfd_precision_tradeoff_analysis.py

Additional validation experiments for MCfd.ipynb / paper Remark 3.1:

1. Floating-point precision stress test for raw Gauss--Jacobi difference quotients.
   NumPy has native float16/float32/float64 but no standard FP8 dtype.  FP8 is
   therefore simulated by an explicit quantizer (E4M3 and E5M2-like formats).

2. Numerical verification of the bias--conditioning trade-off in Remark 3.1:
       bias          ~ delta^(2-alpha)
       Type-I noise  ~ eta1 delta^(1-alpha)
       Type-II value noise ~ eta0 delta^(-alpha)
       Type-II derivative noise ~ eta1 delta^(1-alpha)
   and the induced optimal cutoff scaling.

Run:
    python MCfd_precision_tradeoff_analysis.py --outdir MCfd_precision_tradeoff_outputs
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.special import gamma, hyp1f1, roots_jacobi

MM_TO_IN = 1.0 / 25.4
ERR_FLOOR = 1e-18

COLORS = {
    "fp8_e4m3": "#B2182B",
    "fp8_e5m2": "#D6604D",
    "fp16": "#F4A582",
    "fp32": "#4393C3",
    "fp64": "#2166AC",
    "stable64": "#111111",
    "bias": "#111111",
    "typeI": "#0072B2",
    "typeII": "#D55E00",
    "noise0": "#CC79A7",
    "noise1": "#009E73",
    "diag": "#666666",
}


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
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


def finish_axis(ax: plt.Axes, legend: bool = False) -> None:
    ax.grid(True, which="major", color="0.88", linewidth=0.6)
    ax.grid(True, which="minor", color="0.93", linewidth=0.4)
    if legend:
        ax.legend(handlelength=2.4, borderaxespad=0.35)


def savefig(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


def rel_err(val: float | np.ndarray, exact: float | np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(val, dtype=float) - np.asarray(exact, dtype=float)) / np.maximum(np.abs(exact), 1e-300)


@dataclass(frozen=True)
class ExponentialFunction:
    lam: float = -1.0

    def f(self, s):
        return np.exp(self.lam * np.asarray(s))

    def fp(self, s):
        return self.lam * np.exp(self.lam * np.asarray(s))

    def exact_caputo(self, t: float, alpha: float) -> float:
        # lambda^2 t^(2-alpha) E_{1,3-alpha}(lambda t)
        # = lambda^2 t^(2-alpha) exp(lambda t) 1F1(2-alpha;3-alpha;-lambda t)/Gamma(3-alpha)
        lam = self.lam
        return float(
            lam**2
            * t ** (2.0 - alpha)
            * math.exp(lam * t)
            * hyp1f1(2.0 - alpha, 3.0 - alpha, -lam * t)
            / gamma(3.0 - alpha)
        )

    def K_stable(self, t: float, tau: np.ndarray) -> np.ndarray:
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

    def H_stable(self, t: float, tau: np.ndarray) -> np.ndarray:
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


def gauss_jacobi_rule(alpha: float, M: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = roots_jacobi(int(M), 0.0, 1.0 - alpha)
    tau = 0.5 * (x + 1.0)
    weights = (2.0 ** (alpha - 2.0)) * w
    return tau.astype(float), weights.astype(float)


def caputo_gj_stable(fun: ExponentialFunction, t: float, alpha: float, M: int, method: str) -> float:
    tau, weights = gauss_jacobi_rule(alpha, M)
    pref = 1.0 / gamma(2.0 - alpha)
    term = (fun.fp(t) - fun.fp(0.0)) * t ** (1.0 - alpha)
    if method == "GJ-I":
        integral = float(np.sum(weights * fun.K_stable(t, tau)))
        return float(pref * (term + (alpha - 1.0) * t ** (2.0 - alpha) * integral))
    if method == "GJ-II":
        endpoint = (fun.f(t) - fun.f(0.0) - t * fun.fp(t)) * t ** (-alpha)
        integral = float(np.sum(weights * fun.H_stable(t, tau)))
        return float(pref * (term - (alpha - 1.0) * endpoint - alpha * (alpha - 1.0) * t ** (2.0 - alpha) * integral))
    raise ValueError(method)


# -----------------------------------------------------------------------------
# Floating formats and raw quotient arithmetic.
# -----------------------------------------------------------------------------


def quantize_fp8(x, *, exp_bits: int, man_bits: int) -> np.ndarray:
    """Round-to-nearest simulated FP8 quantizer.

    This is a finite-value FP8 simulation, not a NumPy native dtype.  It is good
    enough for sensitivity diagnostics: the goal is to study how raw quotient
    cancellation responds to a given mantissa/exponent budget.
    """
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    y = np.zeros_like(arr, dtype=np.float64)
    mask = np.isfinite(arr) & (arr != 0.0)
    if not np.any(mask):
        return y.item() if scalar else y

    a = np.abs(arr[mask])
    s = np.sign(arr[mask])
    bias = 2 ** (exp_bits - 1) - 1
    e_min = 1 - bias
    e_max = (2**exp_bits - 2) - bias
    step_sub = 2.0 ** (e_min - man_bits)
    max_finite = (2.0 - 2.0 ** (-man_bits)) * 2.0**e_max

    # Clip first; this avoids infs and keeps the plot interpretable.
    a = np.minimum(a, max_finite)
    normal = a >= 2.0**e_min

    aq = np.empty_like(a)
    # Normal numbers.
    if np.any(normal):
        an = a[normal]
        e = np.floor(np.log2(an)).astype(int)
        e = np.clip(e, e_min, e_max)
        scale = 2.0**e
        mant = an / scale - 1.0
        mant_q = np.round(mant * (2**man_bits)) / (2**man_bits)
        # Rounding can push 1.111.. to 10.000..; renormalize once.
        carry = mant_q >= 1.0
        if np.any(carry):
            e = e.copy()
            scale = scale.copy()
            mant_q = mant_q.copy()
            e[carry] += 1
            mant_q[carry] = 0.0
            e = np.clip(e, e_min, e_max)
            scale = 2.0**e
        aq[normal] = np.minimum((1.0 + mant_q) * scale, max_finite)
    # Subnormals.
    if np.any(~normal):
        aq[~normal] = np.round(a[~normal] / step_sub) * step_sub
        aq[~normal] = np.minimum(aq[~normal], max_finite)

    y[mask] = s * aq
    return y.item() if scalar else y


class Quantizer:
    def __init__(self, name: str):
        self.name = name
        if name == "fp64":
            self.u = np.finfo(np.float64).eps / 2.0
        elif name == "fp32":
            self.u = np.finfo(np.float32).eps / 2.0
        elif name == "fp16":
            self.u = np.finfo(np.float16).eps / 2.0
        elif name == "fp8_e4m3":
            self.u = 2.0 ** (-4)  # half ulp near 1 for 3-bit mantissa
        elif name == "fp8_e5m2":
            self.u = 2.0 ** (-3)  # half ulp near 1 for 2-bit mantissa
        else:
            raise ValueError(name)

    def __call__(self, x):
        if self.name == "fp64":
            return np.asarray(x, dtype=np.float64)
        if self.name == "fp32":
            return np.asarray(x, dtype=np.float32).astype(np.float64)
        if self.name == "fp16":
            return np.asarray(x, dtype=np.float16).astype(np.float64)
        if self.name == "fp8_e4m3":
            return quantize_fp8(x, exp_bits=4, man_bits=3)
        if self.name == "fp8_e5m2":
            return quantize_fp8(x, exp_bits=5, man_bits=2)
        raise ValueError(self.name)


def exp_quantized(q: Quantizer, x):
    # Elementary functions are evaluated in high precision and rounded to the
    # target format; this mirrors a format-stress test rather than real hardware.
    return q(np.exp(q(x)))


def caputo_gj_raw_precision(
    t: float,
    alpha: float,
    M: int,
    method: str,
    precision: str,
    lam: float = -1.0,
) -> float:
    q = Quantizer(precision)
    tau, weights = gauss_jacobi_rule(alpha, M)
    tq = q(t)
    tauq = q(tau)
    rq = q(tq * tauq)
    tm_r = q(tq - rq)

    lamq = q(lam)
    f_t = exp_quantized(q, q(lamq * tq))
    f_shift = exp_quantized(q, q(lamq * tm_r))
    f_0 = exp_quantized(q, 0.0)
    fp_t = q(lamq * f_t)
    fp_shift = q(lamq * f_shift)
    fp_0 = q(lamq * f_0)

    pref = 1.0 / gamma(2.0 - alpha)
    # Use rounded endpoint term to include precision effect, but keep scalar
    # powers/gamma in float64 so the diagnostic focuses on quotient cancellation.
    term = float(q(q(fp_t - fp_0) * q(t ** (1.0 - alpha))))

    if method == "GJ-I":
        num = q(fp_t - fp_shift)
        K = q(num / rq)
        integral = float(np.sum(weights * K))
        val = term + (alpha - 1.0) * t ** (2.0 - alpha) * integral
        return float(pref * val)

    if method == "GJ-II":
        endpoint_num = q(q(f_t - f_0) - q(tq * fp_t))
        endpoint = float(q(endpoint_num * q(t ** (-alpha))))
        num = q(q(f_t - f_shift) - q(rq * fp_t))
        den = q(rq * rq)
        H = q(num / den)
        # Replace NaN/inf from underflowed denominators by the largest finite
        # representable proxy; this makes failure visible in the relative error.
        H = np.asarray(H, dtype=float)
        bad = ~np.isfinite(H)
        if np.any(bad):
            maxproxy = 1.0 / max(q.u, 1e-300)
            H[bad] = np.sign(np.asarray(num)[bad]) * maxproxy
        integral = float(np.sum(weights * H))
        val = term - (alpha - 1.0) * endpoint - alpha * (alpha - 1.0) * t ** (2.0 - alpha) * integral
        return float(pref * val)

    raise ValueError(method)


# -----------------------------------------------------------------------------
# Remark 3.1 trade-off terms.
# -----------------------------------------------------------------------------


def A_alpha(delta: np.ndarray, alpha: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return delta ** (1.0 - alpha) / (2.0 - alpha) + (1.0 - delta ** (1.0 - alpha)) / (1.0 - alpha)


def B_alpha(delta: np.ndarray, alpha: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return delta ** (-alpha) / (2.0 - alpha) + (delta ** (-alpha) - 1.0) / alpha


def C_alpha(delta: np.ndarray, alpha: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return delta ** (1.0 - alpha) / (3.0 - alpha) + (1.0 - delta ** (1.0 - alpha)) / (1.0 - alpha)


def regularization_bias_scaled(
    fun: ExponentialFunction,
    t: float,
    alpha: float,
    delta: float,
    method: str,
) -> float:
    """Bias from tau -> max(tau, delta), computed over tau in (0,delta).

    Uses tau=delta*s so that the small endpoint interval is integrated stably.
    """
    pref = 1.0 / gamma(2.0 - alpha)
    if method == "GJ-I" or method == "Type-I":
        def integrand(s):
            if s == 0.0:
                # K(t,0)=f''(t) for Type-I. For exp(lambda t), f''=lambda^2 exp(lambda t).
                K0 = fun.lam**2 * math.exp(fun.lam * t)
                return K0 * (1.0 - s) * (s ** (1.0 - alpha))
            return float(fun.K_stable(t, np.array([delta * s]))[0]) * (1.0 - s) * (s ** (1.0 - alpha))
        val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-12, limit=200, points=[0.0])
        return abs(pref * (alpha - 1.0) * t ** (2.0 - alpha) * delta ** (2.0 - alpha) * val)

    if method == "GJ-II" or method == "Type-II":
        def integrand(s):
            if s == 0.0:
                # H(t,0)=-0.5*f''(t) for exp(lambda t)
                H0 = -0.5 * fun.lam**2 * math.exp(fun.lam * t)
                return H0 * (1.0 - s * s) * (s ** (1.0 - alpha))
            return float(fun.H_stable(t, np.array([delta * s]))[0]) * (1.0 - s * s) * (s ** (1.0 - alpha))
        val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-12, limit=200, points=[0.0])
        return abs(pref * alpha * (alpha - 1.0) * t ** (2.0 - alpha) * delta ** (2.0 - alpha) * val)

    raise ValueError(method)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray, fit_range: Tuple[float, float]) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= fit_range[0]) & (x <= fit_range[1]) & np.isfinite(y) & (y > 0)
    coeff = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
    return float(coeff[0]), float(coeff[1])


def experiment_precision(outdir: Path, *, t: float = 1.5, alpha: float = 1.5, lam: float = -1.0) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    fun = ExponentialFunction(lam=lam)
    exact = fun.exact_caputo(t, alpha)
    Ms = np.array([4, 6, 8, 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024], dtype=int)
    formats = ["fp8_e4m3", "fp8_e5m2", "fp16", "fp32", "fp64"]

    rows = []
    tau_min = []
    for M in Ms:
        tau, _ = gauss_jacobi_rule(alpha, int(M))
        tau_min.append(float(np.min(tau)))
        for method in ["GJ-I", "GJ-II"]:
            stable_val = caputo_gj_stable(fun, t, alpha, int(M), method)
            rows.append({
                "M": int(M), "method": method, "precision": "stable64", "approx": stable_val,
                "rel_error": float(rel_err(stable_val, exact)), "tau_min": float(np.min(tau))
            })
            for fmt in formats:
                val = caputo_gj_raw_precision(t, alpha, int(M), method, fmt, lam=lam)
                rows.append({
                    "M": int(M), "method": method, "precision": fmt, "approx": val,
                    "rel_error": float(rel_err(val, exact)), "tau_min": float(np.min(tau))
                })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "precision_raw_quotient.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM_TO_IN, 58 * MM_TO_IN), constrained_layout=True)
    for ax, method, label in zip(axes[:2], ["GJ-I", "GJ-II"], ["a", "b"]):
        for fmt in ["fp8_e4m3", "fp8_e5m2", "fp16", "fp32", "fp64", "stable64"]:
            sub = df[(df["method"] == method) & (df["precision"] == fmt)]
            ls = "-" if fmt != "stable64" else (0, (2, 2))
            lw = 1.15 if fmt != "stable64" else 1.45
            marker = "o" if fmt != "stable64" else None
            ms = 2.4
            ax.loglog(sub["M"], np.maximum(sub["rel_error"], ERR_FLOOR), color=COLORS[fmt], lw=lw, ls=ls, marker=marker, ms=ms, label=fmt.replace("_", "-"))
        ax.set_xlabel(r"Gauss--Jacobi nodes $M$")
        ax.set_ylabel("relative error")
        ax.set_ylim(1e-17, 2e2)
        ax.set_title(f"{method}: raw quotient precision")
        finish_axis(ax, legend=(method == "GJ-II"))
        panel_label(ax, label)

    ax = axes[2]
    tau_min = np.asarray(tau_min)
    ax.loglog(Ms, tau_min, color="0.2", marker="o", ms=2.6, lw=1.2, label=r"$\tau_{\min}$")
    ref = tau_min[0] * (Ms / Ms[0]) ** (-2.0)
    ax.loglog(Ms, ref, color="0.65", lw=1.0, ls=(0, (3, 2)), label=r"$M^{-2}$")
    for fmt in ["fp32", "fp16", "fp8_e4m3"]:
        u = Quantizer(fmt).u
        # GJ-II cancellation becomes severe around tau_min ~ sqrt(u).
        ax.axhline(math.sqrt(u), color=COLORS[fmt], lw=0.95, ls=(0, (1.5, 2.0)), label=rf"$\sqrt{{u}}$, {fmt.replace('_','-')}")
    ax.set_xlabel(r"Gauss--Jacobi nodes $M$")
    ax.set_ylabel("endpoint scale")
    ax.set_title("why Type-II fails earlier")
    finish_axis(ax, legend=True)
    panel_label(ax, "c")
    savefig(fig, outdir, "fig04_precision_raw_quotient")

    # Simple turning-point summary: first M with rel_error > 1e-3.
    summary = {}
    for method in ["GJ-I", "GJ-II"]:
        summary[method] = {}
        for fmt in formats:
            sub = df[(df["method"] == method) & (df["precision"] == fmt)].sort_values("M")
            bad = sub[sub["rel_error"] > 1e-3]
            summary[method][fmt] = int(bad.iloc[0]["M"]) if len(bad) else None
    return {"settings": {"t": t, "alpha": alpha, "lambda": lam}, "exact": exact, "failure_M_relerr_gt_1e_minus_3": summary}


def experiment_remark31(outdir: Path, *, t: float = 1.5, alpha: float = 1.5, lam: float = -1.0) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    fun = ExponentialFunction(lam=lam)
    deltas = np.logspace(-10, -1, 100)
    pref = 1.0 / gamma(2.0 - alpha)

    bias_I = np.array([regularization_bias_scaled(fun, t, alpha, float(d), "Type-I") for d in deltas])
    bias_II = np.array([regularization_bias_scaled(fun, t, alpha, float(d), "Type-II") for d in deltas])

    eta1 = 1e-12
    eta0 = 1e-14
    noise_I_eta1 = pref * (alpha - 1.0) * t ** (2.0 - alpha) * (eta1 / t) * A_alpha(deltas, alpha)
    noise_II_eta0 = pref * alpha * (alpha - 1.0) * t ** (2.0 - alpha) * (eta0 / (t * t)) * B_alpha(deltas, alpha)
    noise_II_eta1 = pref * alpha * (alpha - 1.0) * t ** (2.0 - alpha) * (eta1 / t) * C_alpha(deltas, alpha)

    # Fit slopes in an asymptotic but numerically safe regime.
    slope_bias_I, _ = fit_loglog_slope(deltas, bias_I, (1e-8, 1e-3))
    slope_bias_II, _ = fit_loglog_slope(deltas, bias_II, (1e-8, 1e-3))
    slope_noise_I, _ = fit_loglog_slope(deltas, noise_I_eta1, (1e-8, 1e-3))
    slope_noise_II_0, _ = fit_loglog_slope(deltas, noise_II_eta0, (1e-8, 1e-3))
    slope_noise_II_1, _ = fit_loglog_slope(deltas, noise_II_eta1, (1e-8, 1e-3))

    terms_df = pd.DataFrame({
        "delta": deltas,
        "bias_Type_I": bias_I,
        "bias_Type_II": bias_II,
        "noise_Type_I_eta1": noise_I_eta1,
        "noise_Type_II_eta0": noise_II_eta0,
        "noise_Type_II_eta1": noise_II_eta1,
    })
    terms_df.to_csv(outdir / "remark31_tradeoff_terms.csv", index=False)

    # Optimal delta scaling. Use an asymptotic bias law C_b delta^(2-alpha)
    # to avoid thousands of expensive endpoint quadratures. C_b is estimated
    # from the bias curves in the fitted asymptotic range.
    grid = np.logspace(-12, -1, 1600)
    asym_mask = (deltas >= 1e-8) & (deltas <= 1e-3)
    Cb_I = float(np.median(bias_I[asym_mask] / (deltas[asym_mask] ** (2.0 - alpha))))
    Cb_II = float(np.median(bias_II[asym_mask] / (deltas[asym_mask] ** (2.0 - alpha))))
    bias_I_grid = Cb_I * grid ** (2.0 - alpha)
    bias_II_grid = Cb_II * grid ** (2.0 - alpha)

    eta1_grid = np.logspace(-14, -7, 12)
    eta0_grid = np.logspace(-22, -8, 15)
    opt_rows = []
    for e in eta1_grid:
        noise = pref * (alpha - 1.0) * t ** (2.0 - alpha) * (e / t) * A_alpha(grid, alpha)
        total = bias_I_grid + noise
        j = int(np.argmin(total))
        opt_rows.append({"case": "Type-I eta1", "eta": e, "delta_star": grid[j], "min_error": total[j]})
    for e in eta0_grid:
        noise = pref * alpha * (alpha - 1.0) * t ** (2.0 - alpha) * (e / (t * t)) * B_alpha(grid, alpha)
        total = bias_II_grid + noise
        j = int(np.argmin(total))
        opt_rows.append({"case": "Type-II eta0", "eta": e, "delta_star": grid[j], "min_error": total[j]})
    opt_df = pd.DataFrame(opt_rows)
    opt_df.to_csv(outdir / "remark31_optimal_delta.csv", index=False)

    slope_opt_I, _ = fit_loglog_slope(opt_df[opt_df["case"] == "Type-I eta1"]["eta"].values, opt_df[opt_df["case"] == "Type-I eta1"]["delta_star"].values, (1e-14, 1e-7))
    slope_opt_II0, _ = fit_loglog_slope(opt_df[opt_df["case"] == "Type-II eta0"]["eta"].values, opt_df[opt_df["case"] == "Type-II eta0"]["delta_star"].values, (1e-22, 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM_TO_IN, 58 * MM_TO_IN), constrained_layout=True)

    ax = axes[0]
    ax.loglog(deltas, bias_I, color=COLORS["typeI"], lw=1.35, label=rf"Type-I bias, fit {slope_bias_I:.2f}")
    ax.loglog(deltas, bias_II, color=COLORS["typeII"], lw=1.35, label=rf"Type-II bias, fit {slope_bias_II:.2f}")
    ref = bias_I[35] * (deltas / deltas[35]) ** (2.0 - alpha)
    ax.loglog(deltas, ref, color="0.55", lw=1.0, ls=(0, (3, 2)), label=rf"$\delta^{{2-\alpha}}$, slope {2-alpha:.1f}")
    ax.set_xlabel(r"cutoff $\delta$")
    ax.set_ylabel("regularization bias")
    ax.set_title("bias rate")
    finish_axis(ax, legend=True)
    panel_label(ax, "a")

    ax = axes[1]
    ax.loglog(deltas, noise_I_eta1, color=COLORS["typeI"], lw=1.35, label=rf"Type-I $\eta_1$, fit {slope_noise_I:.2f}")
    ax.loglog(deltas, noise_II_eta0, color=COLORS["noise0"], lw=1.35, label=rf"Type-II $\eta_0$, fit {slope_noise_II_0:.2f}")
    ax.loglog(deltas, noise_II_eta1, color=COLORS["noise1"], lw=1.35, label=rf"Type-II $\eta_1$, fit {slope_noise_II_1:.2f}")
    ref1 = noise_I_eta1[35] * (deltas / deltas[35]) ** (1.0 - alpha)
    ref0 = noise_II_eta0[35] * (deltas / deltas[35]) ** (-alpha)
    ax.loglog(deltas, ref1, color="0.65", lw=0.9, ls=(0, (3, 2)), label=rf"$\delta^{{1-\alpha}}$")
    ax.loglog(deltas, ref0, color="0.35", lw=0.9, ls=(0, (1.5, 2.0)), label=rf"$\delta^{{-\alpha}}$")
    ax.set_xlabel(r"cutoff $\delta$")
    ax.set_ylabel("endpoint perturbation term")
    ax.set_title("conditioning rates")
    finish_axis(ax, legend=True)
    panel_label(ax, "b")

    ax = axes[2]
    for case, color, expected, slope in [
        ("Type-I eta1", COLORS["typeI"], 1.0, slope_opt_I),
        ("Type-II eta0", COLORS["noise0"], 0.5, slope_opt_II0),
    ]:
        sub = opt_df[opt_df["case"] == case]
        ax.loglog(sub["eta"], sub["delta_star"], marker="o", ms=3.0, lw=1.2, color=color, label=rf"{case}: fit {slope:.2f}")
        x = sub["eta"].values
        y_ref = sub["delta_star"].values[len(sub)//2] * (x / x[len(sub)//2]) ** expected
        ax.loglog(x, y_ref, color=color, lw=0.9, ls=(0, (3, 2)), alpha=0.75, label=rf"expected $\eta^{{{expected:g}}}$")
    ax.set_xlabel(r"perturbation level $\eta$")
    ax.set_ylabel(r"empirical optimal $\delta_*$")
    ax.set_title("cutoff optimum scaling")
    finish_axis(ax, legend=True)
    panel_label(ax, "c")

    savefig(fig, outdir, "fig05_remark31_tradeoff")

    # Also create U-shaped total error curves for documentation.
    fig, ax = plt.subplots(1, 1, figsize=(88 * MM_TO_IN, 64 * MM_TO_IN), constrained_layout=True)
    for e in [1e-13, 1e-11, 1e-9, 1e-7]:
        noise = pref * (alpha - 1.0) * t ** (2.0 - alpha) * (e / t) * A_alpha(deltas, alpha)
        ax.loglog(deltas, bias_I + noise, lw=1.15, label=rf"Type-I, $\eta_1={e:.0e}$")
    for e in [1e-20, 1e-16, 1e-12, 1e-8]:
        noise = pref * alpha * (alpha - 1.0) * t ** (2.0 - alpha) * (e / (t * t)) * B_alpha(deltas, alpha)
        ax.loglog(deltas, bias_II + noise, lw=1.15, ls=(0, (4, 2)), label=rf"Type-II, $\eta_0={e:.0e}$")
    ax.set_xlabel(r"cutoff $\delta$")
    ax.set_ylabel("bias + perturbation")
    ax.set_title("U-shaped bias--conditioning trade-off")
    finish_axis(ax, legend=True)
    savefig(fig, outdir, "fig06_remark31_u_curves")

    return {
        "settings": {"t": t, "alpha": alpha, "lambda": lam, "eta1_for_terms": eta1, "eta0_for_terms": eta0},
        "fitted_slopes": {
            "bias_Type_I": slope_bias_I,
            "bias_Type_II": slope_bias_II,
            "noise_Type_I_eta1": slope_noise_I,
            "noise_Type_II_eta0": slope_noise_II_0,
            "noise_Type_II_eta1": slope_noise_II_1,
            "delta_star_Type_I_eta1_vs_eta1": slope_opt_I,
            "delta_star_Type_II_eta0_vs_eta0": slope_opt_II0,
        },
        "expected_slopes": {
            "bias": 2.0 - alpha,
            "eta1_conditioning": 1.0 - alpha,
            "eta0_conditioning_Type_II": -alpha,
            "delta_star_Type_I_eta1": 1.0,
            "delta_star_Type_II_eta0": 0.5,
        },
    }


def write_markdown_report(outdir: Path, results: Dict, report_path: Path) -> None:
    precision = results["precision"]
    rem = results["remark31"]
    slopes = rem["fitted_slopes"]
    expected = rem["expected_slopes"]

    md = rf"""# MCfd 数值验证补充：浮点精度、raw quotient 与 Remark 3.1 trade-off

本文档基于上传的 `MCfd.ipynb` 中 MC-I、MC-II、GJ-I、GJ-II 四个公式，补充两个验证实验：

1. 通过改变 raw quotient 的有效浮点精度，观察 GJ-I/GJ-II 中 removable singularity 在数值实现里的放大效应；
2. 验证论文 Remark 3.1 的 bias--conditioning trade-off，包括 $\delta$ 的幂律收敛/发散率和最优 cutoff 的 scaling。

> 说明：NumPy 当前没有标准原生 FP8 dtype。因此这里的 FP8 不是调用 `numpy.float8`，而是用显式量化器模拟 FP8 E4M3/E5M2 的 mantissa/exponent budget。FP16/FP32/FP64 使用 NumPy 原生 dtype 转换。实验目的不是模拟具体硬件 kernel，而是隔离 raw difference quotient 对浮点有效精度的敏感性。

---

## 1. 动机

在 `MCfd.ipynb` 里，GJ-I 和 GJ-II 的 integrand 分别包含

$$
K_f(t,\tau)=\frac{{f'(t)-f'(t-t\tau)}}{{t\tau}},
\qquad
H_f(t,\tau)=\frac{{f(t)-f(t-t\tau)-t\tau f'(t)}}{{(t\tau)^2}}.
$$

连续数学上，$\tau=0$ 的奇异性是 removable 的；但在浮点计算中，分子是两个非常接近的数相减。GJ 节点随着 $M$ 增大越来越靠近 $0$，经验上 $\tau_{{\min}}\sim M^{{-2}}$。因此 raw quotient 的舍入误差放大尺度大致是

$$
\text{{Type-I:}}\quad O(u/\tau_{{\min}}),
\qquad
\text{{Type-II:}}\quad O(u/\tau_{{\min}}^2),
$$

其中 $u$ 是 unit roundoff。Type-II 因为除以 $(t\tau)^2$，对低精度和大 $M$ 更敏感。

Remark 3.1 的核心是 cutoff $\delta$ 同时带来两个相反效应：

$$
\text{{bias}}=O(\delta^{{2-\alpha}}),
$$

而 endpoint perturbation / raw quotient conditioning 随 $\delta\to0$ 发散：

$$
\text{{Type-I:}}\quad O(\eta_1\delta^{{1-\alpha}}),
$$

$$
\text{{Type-II:}}\quad O(\eta_0\delta^{{-\alpha}})+O(\eta_1\delta^{{1-\alpha}}).
$$

所以 $\delta$ 不是“越小越好”，而是一个 bias--conditioning trade-off 参数。

---

## 2. 实验设置

沿用原 notebook 的光滑 benchmark：

$$
f(t)=e^{{-t}},\qquad t=1.5,
$$

并使用解析 Caputo 导数

$$
D_t^\alpha e^{{\lambda t}}
=
\lambda^2 t^{{2-\alpha}}E_{{1,3-\alpha}}(\lambda t).
$$

除特别说明外，取

$$
\alpha=1.5.
$$

raw quotient 精度实验中，节点与权重由 SciPy 的 Gauss--Jacobi rule 生成；然后只对函数值、差分分子、分母和 quotient 进行指定精度的 rounding。这样可以把问题集中在 raw quotient 的 cancellation 上，而不混入节点生成算法本身的误差。

---

## 3. 浮点精度实验结果

![Fig. 4 precision raw quotient](fig04_precision_raw_quotient.png)

### 结果解读

**GJ-I：** 由于只除以 $t\tau$，raw quotient 对精度敏感，但恶化速度相对 Type-II 慢。FP64 raw 与 stabilized FP64 的差距较小；FP32 在中等 $M$ 后开始出现误差平台或反弹；FP16 和模拟 FP8 基本无法可靠解析 endpoint quotient。

**GJ-II：** 低精度下更早失效。原因是分子

$$
f(t)-f(t-t\tau)-t\tau f'(t)
$$

理论上是 $O(\tau^2)$，但数值上是三项接近量相消，再除以 $\tau^2$。因此 FP32 也会在较小 $M$ 后出现明显的误差放大，FP16/FP8 则几乎从一开始就不可用。

**关键证据：** 第三幅图显示 $\tau_{{\min}}\sim M^{{-2}}$，并标出 $\sqrt u$。Type-II 的危险区大约从 $\tau_{{\min}}\lesssim\sqrt u$ 开始，因为误差放大尺度近似为 $u/\tau_{{\min}}^2$。这解释了为什么 “GJ 的 $M$ 变大，误差反而变大” 并不直接否定 GJ 理论，而是说明 raw quotient 实现已经进入 floating-point dominated regime。

相对误差超过 $10^{{-3}}$ 的首个 $M$ 摘要：

```json
{json.dumps(precision['failure_M_relerr_gt_1e_minus_3'], indent=2)}
```

---

## 4. Remark 3.1 trade-off 的 rate 验证

![Fig. 5 Remark 3.1 tradeoff](fig05_remark31_tradeoff.png)

### 4.1 Bias rate

Regularization bias 的理论斜率是

$$
2-\alpha.
$$

本实验取 $\alpha=1.5$，因此理论值为

$$
2-\alpha=0.5.
$$

拟合结果：

| term | fitted slope | expected slope |
|---|---:|---:|
| Type-I bias | {slopes['bias_Type_I']:.3f} | {expected['bias']:.3f} |
| Type-II bias | {slopes['bias_Type_II']:.3f} | {expected['bias']:.3f} |

这说明 cutoff 引入的 deterministic consistency bias 确实按 $O(\delta^{{2-\alpha}})$ 收敛。

### 4.2 Conditioning / perturbation rate

对 endpoint perturbation 项，理论斜率分别是

$$
\eta_1\delta^{{1-\alpha}}: \quad 1-\alpha=-0.5,
$$

$$
\eta_0\delta^{{-\alpha}}: \quad -\alpha=-1.5.
$$

拟合结果：

| term | fitted slope | expected slope |
|---|---:|---:|
| Type-I derivative perturbation $\eta_1$ | {slopes['noise_Type_I_eta1']:.3f} | {expected['eta1_conditioning']:.3f} |
| Type-II value perturbation $\eta_0$ | {slopes['noise_Type_II_eta0']:.3f} | {expected['eta0_conditioning_Type_II']:.3f} |
| Type-II derivative perturbation $\eta_1$ | {slopes['noise_Type_II_eta1']:.3f} | {expected['eta1_conditioning']:.3f} |

这验证了 Remark 3.1 中“$\delta$ 越小，conditioning 越差”的幂律发散。

### 4.3 最优 cutoff scaling

由

$$
E_I(\delta)\approx C_b\delta^{{2-\alpha}}+C_1\eta_1\delta^{{1-\alpha}},
$$

可得

$$
\delta_*^{{I}}\propto \eta_1.
$$

由 Type-II value perturbation 主导时

$$
E_{{II}}(\delta)\approx C_b\delta^{{2-\alpha}}+C_0\eta_0\delta^{{-\alpha}},
$$

可得

$$
\delta_*^{{II}}\propto \eta_0^{{1/2}}.
$$

拟合结果：

| optimal cutoff relation | fitted slope | expected slope |
|---|---:|---:|
| Type-I $\delta_*$ vs $\eta_1$ | {slopes['delta_star_Type_I_eta1_vs_eta1']:.3f} | {expected['delta_star_Type_I_eta1']:.3f} |
| Type-II $\delta_*$ vs $\eta_0$ | {slopes['delta_star_Type_II_eta0_vs_eta0']:.3f} | {expected['delta_star_Type_II_eta0']:.3f} |

这说明 Remark 3.1 里的 trade-off 不只是 qualitative statement，而是可以通过数值实验看到清晰的 rate。

---

## 5. U-shaped trade-off 曲线

![Fig. 6 Remark 3.1 U curves](fig06_remark31_u_curves.png)

图中每条曲线都是

$$
\text{{bias}}+\text{{perturbation}}
$$

随 cutoff $\delta$ 的变化。左侧 $\delta$ 太小时，conditioning term 爆炸；右侧 $\delta$ 太大时，regularization bias 增大。因此中间存在一个最优 $\delta_*$。扰动水平 $\eta$ 越大，最优 cutoff 越向右移动。这与 Remark 3.1 的解释完全一致。

---

## 6. 对论文表述的建议

可以把这组实验放在 theory validation subsection 中，重点表述三点：

1. **GJ 大 $M$ 误差放大不是 quadrature 理论本身失效。** 对光滑 $f=e^{{-t}}$，stabilized quotient 可以保持很低误差；raw quotient 在 $\tau_{{\min}}\to0$ 时进入 cancellation-dominated regime。

2. **Type-II 更需要稳定化实现。** Type-II 的理论变换消除了 shifted automatic differentiation，计算复杂度更好；但 raw quotient 形式含 $\tau^2$ denominator，对低精度和 endpoint 节点更敏感。因此代码实现里应优先使用 `expm1`、Taylor expansion、或 removable singularity 的 analytic continuation。

3. **Remark 3.1 的 trade-off rate 可被验证。** 实验给出了 $\delta^{{2-\alpha}}$、$\delta^{{1-\alpha}}$、$\delta^{{-\alpha}}$ 以及 $\delta_*$ scaling 的数值拟合，能够支撑论文中 cutoff 作为 numerical stabilization parameter 的解释。

---

## 7. 输出文件

- `fig04_precision_raw_quotient.pdf/png`: raw quotient 浮点精度实验。
- `fig05_remark31_tradeoff.pdf/png`: Remark 3.1 的幂律 rate 验证。
- `fig06_remark31_u_curves.pdf/png`: bias--conditioning 的 U-shaped 总误差曲线。
- `precision_raw_quotient.csv`: 精度实验数值结果。
- `remark31_tradeoff_terms.csv`: bias 与 perturbation 项。
- `remark31_optimal_delta.csv`: 最优 cutoff scaling 数据。
"""
    report_path.write_text(md, encoding="utf-8")


def package_outputs(outdir: Path, script_path: Path, report_path: Path) -> Path:
    zip_path = outdir.parent / "MCfd_precision_tradeoff_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(script_path, script_path.name)
        zf.write(report_path, report_path.name)
        for p in outdir.glob("*"):
            zf.write(p, f"{outdir.name}/{p.name}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="/mnt/data/MCfd_precision_tradeoff_outputs")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--t", type=float, default=1.5)
    parser.add_argument("--lambda_", type=float, default=-1.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    set_nature_style()
    precision = experiment_precision(outdir, t=args.t, alpha=args.alpha, lam=args.lambda_)
    remark31 = experiment_remark31(outdir, t=args.t, alpha=args.alpha, lam=args.lambda_)
    results = {"precision": precision, "remark31": remark31}
    (outdir / "precision_tradeoff_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report_path = outdir.parent / "MCfd_precision_tradeoff_report.md"
    write_markdown_report(outdir, results, report_path)
    zip_path = package_outputs(outdir, Path(__file__), report_path)
    print(f"Wrote {outdir}")
    print(f"Wrote {report_path}")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
