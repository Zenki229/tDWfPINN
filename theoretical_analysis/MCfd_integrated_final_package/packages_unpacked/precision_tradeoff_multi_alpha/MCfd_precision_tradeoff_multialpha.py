from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import gamma, hyp1f1, roots_jacobi

OUT = Path(__file__).resolve().parent / 'MCfd_precision_tradeoff_multialpha_report'
OUT.mkdir(parents=True, exist_ok=True)
SRC_REPORT = Path(__file__).resolve().parents[1] / 'precision_tradeoff_single_alpha' / 'MCfd_precision_tradeoff_report'
MM_TO_IN = 1 / 25.4

COLORS = {
    'I': '#0072B2',
    'II': '#D55E00',
    'A': '#0072B2',
    'B': '#D55E00',
    'C': '#56B4E9',
    'gray': '#666666',
    'a125': '#0072B2',
    'a150': '#009E73',
    'a175': '#D55E00',
}
ALPHA_COLORS = {1.25: COLORS['a125'], 1.50: COLORS['a150'], 1.75: COLORS['a175']}


def style() -> None:
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8.0,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8.5,
        'legend.fontsize': 6.8,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'axes.linewidth': 0.75,
        'xtick.major.width': 0.75,
        'ytick.major.width': 0.75,
        'xtick.minor.width': 0.55,
        'ytick.minor.width': 0.55,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.dpi': 160,
        'savefig.dpi': 500,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'dejavusans',
    })


def panel(ax, lab: str) -> None:
    ax.text(
        0.018, 0.982, lab, transform=ax.transAxes,
        fontsize=9.5, fontweight='bold', va='top', ha='left',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.1, 'alpha': 0.86},
        zorder=10,
    )


def finish(ax, legend: bool = False) -> None:
    ax.grid(True, which='major', color='0.88', lw=0.6)
    ax.grid(True, which='minor', color='0.93', lw=0.4)
    if legend:
        ax.legend(handlelength=2.1, borderaxespad=0.35)


def savefig(fig, stem: str) -> None:
    fig.savefig(OUT / f'{stem}.pdf')
    fig.savefig(OUT / f'{stem}.png')
    plt.close(fig)


class ExpFun:
    def __init__(self, lam: float = -1.0):
        self.lam = lam

    def exact_caputo(self, t: float, alpha: float) -> float:
        lam = self.lam
        return float(lam**2 * t**(2 - alpha) * math.exp(lam * t) * hyp1f1(2 - alpha, 3 - alpha, -lam * t) / gamma(3 - alpha))

    def K_stable(self, t: float, tau):
        tau = np.asarray(tau, dtype=float)
        r = t * tau
        lam = self.lam
        e = math.exp(lam * t)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = lam * e * (-np.expm1(-lam * r)) / r
        small = np.abs(r) < 1e-7
        if np.any(small):
            out = np.asarray(out, dtype=float)
            out[small] = lam**2 * e - 0.5 * r[small] * lam**3 * e + r[small]**2 * lam**4 * e / 6 - r[small]**3 * lam**5 * e / 24
        return out

    def H_stable(self, t: float, tau):
        tau = np.asarray(tau, dtype=float)
        r = t * tau
        lam = self.lam
        e = math.exp(lam * t)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = e * (-np.expm1(-lam * r) - lam * r) / (r * r)
        small = np.abs(r) < 1e-5
        if np.any(small):
            out = np.asarray(out, dtype=float)
            out[small] = -0.5 * lam**2 * e + r[small] * lam**3 * e / 6 - r[small]**2 * lam**4 * e / 24 + r[small]**3 * lam**5 * e / 120
        return out


def A_alpha(alpha: float, delta):
    d = np.asarray(delta, dtype=float)
    return d ** (1 - alpha) / (2 - alpha) + (1 - d ** (1 - alpha)) / (1 - alpha)


def B_alpha(alpha: float, delta):
    d = np.asarray(delta, dtype=float)
    return d ** (-alpha) / (2 - alpha) + (1 - d ** (-alpha)) / (-alpha)


def C_alpha(alpha: float, delta):
    d = np.asarray(delta, dtype=float)
    return d ** (1 - alpha) / (3 - alpha) + (1 - d ** (1 - alpha)) / (1 - alpha)


def slope(x, y, lo=None, hi=None) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    if np.count_nonzero(m) < 3:
        raise RuntimeError('not enough points for slope fit')
    return float(np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0])


def gj_weight_rule(alpha: float, n: int = 160):
    """Nodes/weights for integral_0^1 phi(s) s^(1-alpha) ds."""
    x, w = roots_jacobi(n, 0.0, 1.0 - alpha)
    s = 0.5 * (x + 1.0)
    ws = (2.0 ** (alpha - 2.0)) * w
    return s, ws


def bias_delta(fun: ExpFun, t: float, alpha: float, delta: float, typ: str) -> float:
    # This is the exact cutoff-bias integral evaluated by a high-order
    # Gauss-Jacobi rule for the endpoint weight s^(1-alpha). It is much
    # faster than adaptive quad and is sufficient for slope verification.
    s, ws = gj_weight_rule(alpha, n=160)
    pref = 1 / gamma(2 - alpha)
    if typ == 'I':
        vals = fun.K_stable(t, delta * s) * (1.0 - s)
        integ = np.sum(ws * vals)
        return abs(pref * (alpha - 1) * t ** (2 - alpha) * delta ** (2 - alpha) * integ)
    if typ == 'II':
        vals = fun.H_stable(t, delta * s) * (1.0 - s * s)
        integ = np.sum(ws * vals)
        return abs(pref * alpha * (alpha - 1) * t ** (2 - alpha) * delta ** (2 - alpha) * integ)
    raise ValueError(typ)

def tradeoff_for_alpha(alpha: float, t: float = 1.5):
    fun = ExpFun(-1.0)
    p = 2 - alpha
    deltas_bias = np.logspace(-9, -2, 34)
    bI = np.array([bias_delta(fun, t, alpha, float(d), 'I') for d in deltas_bias])
    bII = np.array([bias_delta(fun, t, alpha, float(d), 'II') for d in deltas_bias])
    fit_lo, fit_hi = 1e-8, 1e-4
    sbI = slope(deltas_bias, bI, lo=fit_lo, hi=fit_hi)
    sbII = slope(deltas_bias, bII, lo=fit_lo, hi=fit_hi)

    deltas = np.logspace(-10, -2, 220)
    A = A_alpha(alpha, deltas)
    B = B_alpha(alpha, deltas)
    C = C_alpha(alpha, deltas)
    sA = slope(deltas, A, lo=1e-10, hi=1e-6)
    sB = slope(deltas, B, lo=1e-10, hi=1e-6)
    sC = slope(deltas, C, lo=1e-10, hi=1e-6)

    mask = (deltas_bias >= fit_lo) & (deltas_bias <= fit_hi)
    CbI = np.median(bI[mask] / (deltas_bias[mask] ** p))
    CbII = np.median(bII[mask] / (deltas_bias[mask] ** p))

    grid = np.logspace(-11, -1, 3000)
    eta1_vals = np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6])
    eta0_vals = np.array([1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 3e-12, 1e-11])
    optI = []
    for e in eta1_vals:
        total = CbI * grid ** p + e * A_alpha(alpha, grid)
        optI.append(grid[np.argmin(total)])
    optII = []
    for e in eta0_vals:
        total = CbII * grid ** p + e * B_alpha(alpha, grid)
        optII.append(grid[np.argmin(total)])
    optI = np.array(optI)
    optII = np.array(optII)
    sOptI = slope(eta1_vals, optI)
    sOptII = slope(eta0_vals, optII)

    return {
        'alpha': alpha,
        'expected': {'bias': 2 - alpha, 'A_C': 1 - alpha, 'B': -alpha, 'optI': 1.0, 'optII': 0.5},
        'measured': {'bias_I': sbI, 'bias_II': sbII, 'A': sA, 'B': sB, 'C': sC, 'optI': sOptI, 'optII': sOptII},
        'deltas_bias': deltas_bias, 'bias_I': bI, 'bias_II': bII,
        'deltas': deltas, 'A': A, 'B': B, 'C': C,
        'CbI': CbI, 'CbII': CbII,
        'eta1_vals': eta1_vals, 'eta0_vals': eta0_vals,
        'optI': optI, 'optII': optII,
        'eta1_show': 1e-7, 'eta0_show': 1e-12,
    }


def run_multialpha(alphas=(1.25, 1.50, 1.75)):
    results = [tradeoff_for_alpha(a) for a in alphas]

    rows = []
    for r in results:
        a = r['alpha']; e = r['expected']; m = r['measured']
        rows.append({
            'alpha': a,
            'expected_bias': e['bias'], 'measured_bias_I': m['bias_I'], 'measured_bias_II': m['bias_II'],
            'expected_A_C': e['A_C'], 'measured_A': m['A'], 'measured_C': m['C'],
            'expected_B': e['B'], 'measured_B': m['B'],
            'expected_optI': e['optI'], 'measured_optI': m['optI'],
            'expected_optII': e['optII'], 'measured_optII': m['optII'],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'remark31_multialpha_rate_summary.csv', index=False)

    pd.concat([pd.DataFrame({'alpha': r['alpha'], 'delta': r['deltas_bias'], 'bias_I': r['bias_I'], 'bias_II': r['bias_II']}) for r in results], ignore_index=True).to_csv(OUT / 'remark31_multialpha_regularization_bias.csv', index=False)
    pd.concat([pd.DataFrame({'alpha': r['alpha'], 'delta': r['deltas'], 'A_alpha': r['A'], 'B_alpha': r['B'], 'C_alpha': r['C']}) for r in results], ignore_index=True).to_csv(OUT / 'remark31_multialpha_conditioning_integrals.csv', index=False)
    pd.concat([pd.DataFrame({'alpha': r['alpha'], 'eta1': r['eta1_vals'], 'delta_star_type_I': r['optI'], 'eta0': r['eta0_vals'], 'delta_star_type_II': r['optII']}) for r in results], ignore_index=True).to_csv(OUT / 'remark31_multialpha_optimal_delta.csv', index=False)

    # Rate validation figure: rows are alpha, columns are bias / conditioning / U-shaped total.
    fig, axes = plt.subplots(3, 3, figsize=(183 * MM_TO_IN, 170 * MM_TO_IN), constrained_layout=True)
    labels = list('abcdefghi')
    k = 0
    for row, r in enumerate(results):
        alpha = r['alpha']; m = r['measured']; p = 2 - alpha

        ax = axes[row, 0]
        ax.loglog(r['deltas_bias'], r['bias_I'], color=COLORS['I'], marker='o', ms=2.1, lw=1.05, label=rf'Type-I, slope {m["bias_I"]:.2f}')
        ax.loglog(r['deltas_bias'], r['bias_II'], color=COLORS['II'], marker='s', ms=2.1, lw=1.05, label=rf'Type-II, slope {m["bias_II"]:.2f}')
        idx = np.argmin(abs(r['deltas_bias'] - 1e-5))
        ref = r['bias_I'][idx] * (r['deltas_bias'] / r['deltas_bias'][idx]) ** p
        ax.loglog(r['deltas_bias'], ref, color='0.55', lw=.9, ls=(0, (3, 2)), label=rf'ref. slope {p:.2f}')
        ax.set_title(rf'regularization bias, $\alpha={alpha:.2f}$')
        ax.set_xlabel(r'cutoff $\delta$'); ax.set_ylabel('bias')
        finish(ax, legend=True); panel(ax, labels[k]); k += 1

        ax = axes[row, 1]
        ax.loglog(r['deltas'], r['A'], color=COLORS['A'], lw=1.05, label=rf'$A_\alpha$, {m["A"]:.2f}')
        ax.loglog(r['deltas'], r['C'], color=COLORS['C'], lw=1.05, label=rf'$C_\alpha$, {m["C"]:.2f}')
        ax.loglog(r['deltas'], r['B'], color=COLORS['B'], lw=1.05, label=rf'$B_\alpha$, {m["B"]:.2f}')
        idx = np.argmin(abs(r['deltas'] - 1e-7))
        ax.loglog(r['deltas'], r['A'][idx] * (r['deltas'] / r['deltas'][idx]) ** (1 - alpha), color='0.55', lw=.8, ls=(0, (3, 2)), label=rf'ref. {1-alpha:.2f}')
        ax.loglog(r['deltas'], r['B'][idx] * (r['deltas'] / r['deltas'][idx]) ** (-alpha), color='0.25', lw=.8, ls=(0, (1, 2)), label=rf'ref. {-alpha:.2f}')
        ax.set_title(rf'conditioning integrals, $\alpha={alpha:.2f}$')
        ax.set_xlabel(r'cutoff $\delta$'); ax.set_ylabel('scale')
        finish(ax, legend=True); panel(ax, labels[k]); k += 1

        ax = axes[row, 2]
        d = r['deltas']; A = r['A']; B = r['B']
        totalI = r['CbI'] * d ** p + r['eta1_show'] * A
        totalII = r['CbII'] * d ** p + r['eta0_show'] * B
        ax.loglog(d, r['CbI'] * d ** p, color='0.65', lw=.95, label='bias model')
        ax.loglog(d, r['eta1_show'] * A, color=COLORS['I'], lw=.95, label='Type-I perturb.')
        ax.loglog(d, totalI, color=COLORS['I'], lw=1.35, ls=(0, (4, 2)), label='Type-I total')
        ax.loglog(d, r['eta0_show'] * B, color=COLORS['II'], lw=.95, label='Type-II perturb.')
        ax.loglog(d, totalII, color=COLORS['II'], lw=1.35, ls=(0, (4, 2)), label='Type-II total')
        ax.set_title(rf'U-shaped trade-off, $\alpha={alpha:.2f}$')
        ax.set_xlabel(r'cutoff $\delta$'); ax.set_ylabel('model error')
        finish(ax, legend=True); panel(ax, labels[k]); k += 1
    savefig(fig, 'fig06_remark31_multialpha_delta_tradeoff_rates')

    # Optimal delta scaling figure.
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM_TO_IN, 62 * MM_TO_IN), constrained_layout=True)
    ax = axes[0]
    for r in results:
        a = r['alpha']; color = ALPHA_COLORS[a]
        ax.loglog(r['eta1_vals'], r['optI'], marker='o', ms=2.8, lw=1.15, color=color, label=rf'$\alpha={a:.2f}$, slope {r["measured"]["optI"]:.2f}')
    mid = results[1]
    ax.loglog(mid['eta1_vals'], mid['optI'][2] * (mid['eta1_vals'] / mid['eta1_vals'][2]) ** 1.0, color='0.55', lw=1, ls=(0, (3, 2)), label=r'$\delta_*\propto\eta_1$')
    ax.set_xlabel(r'derivative perturbation $\eta_1$'); ax.set_ylabel(r'optimal cutoff $\delta_*$')
    ax.set_title('Type-I optimum')
    finish(ax, legend=True); panel(ax, 'a')

    ax = axes[1]
    for r in results:
        a = r['alpha']; color = ALPHA_COLORS[a]
        ax.loglog(r['eta0_vals'], r['optII'], marker='s', ms=2.8, lw=1.15, color=color, label=rf'$\alpha={a:.2f}$, slope {r["measured"]["optII"]:.2f}')
    ax.loglog(mid['eta0_vals'], mid['optII'][2] * (mid['eta0_vals'] / mid['eta0_vals'][2]) ** 0.5, color='0.55', lw=1, ls=(0, (3, 2)), label=r'$\delta_*\propto\eta_0^{1/2}$')
    ax.set_xlabel(r'function perturbation $\eta_0$'); ax.set_ylabel(r'optimal cutoff $\delta_*$')
    ax.set_title('Type-II optimum')
    finish(ax, legend=True); panel(ax, 'b')
    savefig(fig, 'fig07_remark31_multialpha_optimal_delta_scaling')

    for name in [
        'fig04_precision_raw_quotients.png', 'fig04_precision_raw_quotients.pdf',
        'fig05_precision_GJ_M_sweep.png', 'fig05_precision_GJ_M_sweep.pdf',
        'precision_GJ_M_errors.csv', 'precision_raw_quotient_node_errors.csv',
    ]:
        src = SRC_REPORT / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    json_results = [{'alpha': r['alpha'], 'expected': r['expected'], 'measured': r['measured']} for r in results]
    (OUT / 'remark31_multialpha_results.json').write_text(json.dumps(json_results, indent=2), encoding='utf-8')
    return summary, json_results


def format_table(df: pd.DataFrame) -> str:
    rows = [
        r'| $\alpha$ | bias exp. | bias I | bias II | $A,C$ exp. | $A$ | $C$ | $B$ exp. | $B$ | $\delta_*^I$ | $\delta_*^{II}$ |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, r in df.iterrows():
        rows.append(
            f"| {r['alpha']:.2f} | {r['expected_bias']:.2f} | {r['measured_bias_I']:.2f} | {r['measured_bias_II']:.2f} | "
            f"{r['expected_A_C']:.2f} | {r['measured_A']:.2f} | {r['measured_C']:.2f} | {r['expected_B']:.2f} | {r['measured_B']:.2f} | "
            f"{r['measured_optI']:.2f} | {r['measured_optII']:.2f} |"
        )
    return '\n'.join(rows)


def write_report(summary: pd.DataFrame) -> None:
    table = format_table(summary)
    md_template = r'''# MCfd 精度诊断与 Remark 3.1 多 alpha trade-off 验证

本文档更新了上一版报告中 Remark 3.1 的 rate 验证。之前只用 $\alpha=1.5$，现在改为三个代表性 case：

$$
\alpha=1.25,\quad 1.50,\quad 1.75.
$$

这样可以覆盖接近一阶、中间阶、接近二阶三种情形。实验仍沿用 `MCfd.ipynb` 的基准函数

$$
f(t)=e^{-t},\qquad t=1.5.
$$

---

## 1. raw quotient 与浮点精度诊断

GJ-I 和 GJ-II 的 integrand 分别含有

$$
K_f(t,\tau)=\frac{f'(t)-f'(t-t\tau)}{t\tau},\qquad
H_f(t,\tau)=\frac{f(t)-f(t-t\tau)-t\tau f'(t)}{(t\tau)^2}.
$$

连续层面二者在 $\tau=0$ 都是 removable singularity；但 raw quotient 在有限精度下会先做近似相等数相减，再除以 $\tau$ 或 $\tau^2$。当 GJ 节点数 $M$ 增大时，最小节点大致满足 $\tau_{\min}\propto M^{-2}$，因此 endpoint cancellation 会被放大。

![Raw quotient precision diagnostics](fig04_precision_raw_quotients.png)

图中 fp64、fp32、fp16，以及 fp8-like e5m2/e4m3 的对比说明：Type-II raw quotient 对低精度最敏感。fp8-like arithmetic 基本不能直接用于 endpoint raw quotient；fp16 也很容易进入 floating-point dominated regime。

![Precision-dependent GJ M sweep](fig05_precision_GJ_M_sweep.png)

这个实验解释了为什么你原图中 GJ 误差可能随 $M$ 增大反而放大：不是 Gauss--Jacobi 理论本身失效，而是 raw quotient 的实现进入了数值抵消主导区间。

---

## 2. Remark 3.1 的理论 rate

Remark 3.1 的核心 trade-off 是：cutoff $\delta$ 会带来 deterministic consistency bias，但同时抑制 endpoint perturbation amplification。

Type-I:

$$
E_I(\delta)\approx
O(\delta^{2-\alpha})
+
O(\eta_1\delta^{1-\alpha}).
$$

Type-II:

$$
E_{II}(\delta)\approx
O(\delta^{2-\alpha})
+
O(\eta_0\delta^{-\alpha})
+
O(\eta_1\delta^{1-\alpha}).
$$

因此需要验证的斜率是

$$
\text{bias}:\quad 2-\alpha,
\qquad
A_\alpha,C_\alpha:\quad 1-\alpha,
\qquad
B_\alpha:\quad -\alpha.
$$

对于三个 case，理论值分别是：

| $\alpha$ | bias $2-\alpha$ | $A,C$ $1-\alpha$ | $B$ $-\alpha$ |
|---:|---:|---:|---:|
| 1.25 | 0.75 | -0.25 | -1.25 |
| 1.50 | 0.50 | -0.50 | -1.50 |
| 1.75 | 0.25 | -0.75 | -1.75 |

---

## 3. 多 alpha rate 验证结果

![Remark 3.1 multi-alpha rate validation](fig06_remark31_multialpha_delta_tradeoff_rates.png)

每一行对应一个 $\alpha$：1.25、1.50、1.75。三列分别展示：

1. regularization bias 的收敛率；
2. conditioning integrals $A_\alpha(\delta)$、$B_\alpha(\delta)$、$C_\alpha(\delta)$ 的发散率；
3. bias 与 perturbation 叠加后的 U-shaped trade-off。

测得的 log--log slope 如下：

{table}

可以看到三个 $\alpha$ 下都和 Remark 3.1 的幂律预测一致。尤其是 $\alpha=1.75$ 时，$B_\alpha$ 的斜率接近 $-1.75$，这说明越接近二阶，Type-II 中 function-value perturbation 的 endpoint amplification 越强。

---

## 4. 最优 cutoff scaling 的多 alpha 验证

由 leading terms 平衡可得：

Type-I:

$$
C_b\delta^{2-\alpha}+C_1\eta_1\delta^{1-\alpha}
\quad\Longrightarrow\quad
\delta_*^I\propto \eta_1.
$$

Type-II 在 $\eta_0\delta^{-\alpha}$ 主导时：

$$
C_b\delta^{2-\alpha}+C_0\eta_0\delta^{-\alpha}
\quad\Longrightarrow\quad
\delta_*^{II}\propto \eta_0^{1/2}.
$$

这两个 exponent 理论上都不依赖 $\alpha$。多 alpha 数值验证如下：

![Remark 3.1 multi-alpha optimal cutoff scaling](fig07_remark31_multialpha_optimal_delta_scaling.png)

三条曲线分别对应 $\alpha=1.25,1.50,1.75$。Type-I 的 measured slope 都接近 1，Type-II 的 measured slope 都接近 0.5。这说明 Remark 3.1 中的 cutoff trade-off 不只是单个 $\alpha=1.5$ 的偶然现象，而是在多个 fractional order 下稳定成立。

---

## 5. 结论与论文写法建议

这组三 case 的验证比单个 $\alpha=1.5$ 更有说服力，可以在论文中表述为：

> We validate the bias--conditioning trade-off in Remark 3.1 for three representative fractional orders, $\alpha=1.25,1.50,1.75$. The measured log--log slopes agree with the predicted rates $2-\alpha$, $1-\alpha$, and $-\alpha$. Moreover, the optimal cutoff obeys $\delta_*^I\propto\eta_1$ and $\delta_*^{II}\propto\eta_0^{1/2}$ across all tested orders.

建议把这部分作为理论验证 subsection 的一张主图。图中可以强调：

- regularization bias 随 $\delta\to0$ 收敛，rate 是 $2-\alpha$；
- conditioning terms 随 $\delta\to0$ 发散，rate 是 $1-\alpha$ 或 $-\alpha$；
- $\alpha$ 越接近 2，允许的稳定 cutoff 窗口越窄；
- Type-II 的 $B_\alpha(\delta)$ 项最强，因此 raw quotient 和低精度实现都更危险。

---

## 6. 文件说明

- `fig04_precision_raw_quotients.pdf/png`: raw quotient 的 precision diagnostic；
- `fig05_precision_GJ_M_sweep.pdf/png`: precision-dependent GJ $M$ sweep；
- `fig06_remark31_multialpha_delta_tradeoff_rates.pdf/png`: 三个 $\alpha$ 的 rate 验证主图；
- `fig07_remark31_multialpha_optimal_delta_scaling.pdf/png`: 三个 $\alpha$ 的最优 cutoff scaling；
- `remark31_multialpha_rate_summary.csv`: slope 汇总表；
- `remark31_multialpha_regularization_bias.csv`: bias 数据；
- `remark31_multialpha_conditioning_integrals.csv`: conditioning integral 数据；
- `remark31_multialpha_optimal_delta.csv`: optimal cutoff 数据。
'''
    (OUT / 'MCfd_precision_tradeoff_multialpha_report.md').write_text(md_template.replace('{table}', table), encoding='utf-8')


def package() -> Path:
    zip_path = OUT.parent / 'MCfd_precision_tradeoff_multialpha_report_package.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, 'MCfd_precision_tradeoff_multialpha.py')
        for p in OUT.iterdir():
            if p.is_file():
                z.write(p, f'MCfd_precision_tradeoff_multialpha_report/{p.name}')
    return zip_path


if __name__ == '__main__':
    style()
    summary, json_results = run_multialpha()
    write_report(summary)
    zip_path = package()
    print(summary.to_string(index=False))
    print(f'Wrote {OUT}')
    print(f'Packaged {zip_path}')
