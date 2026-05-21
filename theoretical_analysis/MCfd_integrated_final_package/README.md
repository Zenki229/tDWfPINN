# MCfd Integrated Validation Package

This package contains the main derivative-level numerical validation workflow
for the transformed Caputo estimators used in the paper. It is organized around
the four estimator families introduced by Corollary `coro:4`:

```text
MC-I, MC-II, GJ-I, GJ-II
```

The scripts here test fractional derivative evaluation directly. They are not
PINN training runs.

## Main Scripts

| Script | Purpose | Default output |
| --- | --- | --- |
| `code/MCfd_refactored_validation.py` | Alpha sweep, M sweep, endpoint-mass diagnostics, and nonsmooth checks. | `data/refactored_outputs/` |
| `code/dw_fractional_validation_suite.py` | Companion diffusion-wave derivative validation suite. | `data/dw_validation_outputs/` |
| `code/MCfd_mc_gj_rate_verification.py` | MC `M^-1/2` and GJ `rho^(-2M)` rate checks. | `reports/MCfd_rate_verification_report/` |
| `code/MCfd_precision_tradeoff_final.py` | Single-alpha precision and cutoff trade-off archive script. | archived `/mnt/data` workflow |
| `code/MCfd_precision_tradeoff_multialpha.py` | Multi-alpha Remark `rem:delta-tradeoff` archive script. | archived `/mnt/data` workflow |

The precision trade-off scripts are preserved from the original report
generation workflow and contain hard-coded `/mnt/data` archive paths. Their
generated figures and CSV summaries are already stored under `figures/` and
`packages_unpacked/`. Use the first three scripts below for normal local
regeneration.

## Regenerate Main Validation Outputs

From this directory:

```bash
python code/MCfd_refactored_validation.py --outdir data/refactored_outputs --seed 229 --mc-repeats 24
python code/dw_fractional_validation_suite.py --outdir data/dw_validation_outputs --seed 229 --mc-repeats 32
python code/MCfd_mc_gj_rate_verification.py --outdir reports/MCfd_rate_verification_report --seed 229 --mc-repeats 384
```

The first command writes the alpha sweep, M sweep, and nonsmooth diagnostic
data. The third command writes the MC and GJ rate figures used for the paper.

## Checked Rates and Formulas

Monte Carlo convergence from Proposition `prop:mc-convergence`:

```text
(E |D^alpha f - D^alpha_{M,MC} f|^2)^(1/2) = O(M^-1/2)
```

Gauss-Jacobi finite-smoothness behavior from Theorem `thm:GJ-convergence`:

```text
|I_alpha[phi] - Q_M^GJ[phi]| <= C_{alpha,r} M^-r ||phi||_{C^r([0,1])}
```

Analytic-kernel GJ behavior:

```text
|I_alpha[phi] - Q_M^GJ[phi]| = O(rho^(-2M))
```

Endpoint cutoff trade-off from Propositions `prop:delta-conditioning`,
`prop:regularization-bias`, and Remark `rem:delta-tradeoff`:

```text
regularization bias:  O(delta^(2-alpha))
Type-I perturbation:  O(eta_1 delta^(1-alpha))
Type-II perturbation: O(eta_0 delta^(-alpha)) + O(eta_1 delta^(1-alpha))
```

The MC endpoint concentration used in the alpha-sweep explanation is:

```text
P(xi < delta) = delta^(2-alpha),  xi ~ Beta(2-alpha, 1)
```

## Figure Map

| Figure | Meaning |
| --- | --- |
| `figures/alpha/fig01_alpha_sweep_exp.png` | Alpha sensitivity of MC-I, MC-II, GJ-I, and GJ-II. |
| `figures/M_effect/fig02_M_sweep_exp_diagnostic.png` | M sensitivity and raw/stable endpoint quotient separation. |
| `figures/nonsmooth/fig03_nonsmooth_tests.png` | Loss of high-order GJ behavior when transformed kernels are nonsmooth. |
| `figures/precision/fig04_precision_raw_quotients.png` | Raw quotient precision diagnostics. |
| `figures/M_effect/fig05_precision_GJ_M_sweep.png` | Precision-dependent GJ M sweep. |
| `figures/tradeoff/fig06_remark31_multialpha_delta_tradeoff_rates.png` | Multi-alpha cutoff trade-off slope check. |
| `figures/tradeoff/fig07_remark31_multialpha_optimal_delta_scaling.png` | Optimal cutoff scaling across alpha values. |
| `figures/rate/fig09_mc_sqrtM_rate.png` | MC `M^-1/2` RMS convergence. |
| `figures/rate/fig10_gj_rho_spectral_rate.png` | GJ `rho^(-2M)` analytic-kernel convergence. |
| `figures/rate/fig11_rate_slope_summary.png` | Summary of fitted rate slopes. |

## Representative Outputs

![Alpha sweep](figures/alpha/fig01_alpha_sweep_exp.png)

![M sweep](figures/M_effect/fig02_M_sweep_exp_diagnostic.png)

![Cutoff tradeoff](figures/tradeoff/fig06_remark31_multialpha_delta_tradeoff_rates.png)

![GJ spectral rate](figures/rate/fig10_gj_rho_spectral_rate.png)
