# Theoretical Analysis and Validation Artifacts

This directory collects the derivative-level numerical validation artifacts used
to support the theoretical part of the paper. These scripts do not train PINNs.
They isolate the quadrature, endpoint regularization, and finite-precision
behavior of the transformed Caputo derivative estimators for `alpha in (1, 2)`.

## Directory Layout

| Path | Purpose |
| --- | --- |
| `MCfd_integrated_final_package/` | Main integrated validation package: alpha sweep, M sweep, nonsmooth tests, precision diagnostics, cutoff trade-off checks, MC rate, and GJ spectral rate. |
| `reviewer_ready_validation_section/` | Reviewer-ready Section 3.2 replacement with paper-ready figures, tables, CSV data, and response notes. |
| `alpha2_precision_remark/` | Focused diagnostic for the `alpha -> 2` finite-precision behavior of raw Gauss-Jacobi endpoint quotients. |
| `gj_mr_rate_validation/` | Additional Gauss-Jacobi finite-smoothness `M^{-r}` rate validation. |

## Main Validation Chain

The main package is `MCfd_integrated_final_package/`. It contains the complete
source scripts, notebooks, generated CSV/JSON data, and figures. The most useful
figures are already staged under `MCfd_integrated_final_package/figures/`.

Representative commands from inside `theoretical_analysis/MCfd_integrated_final_package`:

```bash
conda run -n sciml python code/MCfd_refactored_validation.py --outdir data/refactored_outputs --seed 229 --mc-repeats 24
conda run -n sciml python code/dw_fractional_validation_suite.py --outdir data/dw_validation_outputs --seed 229
conda run -n sciml python code/MCfd_mc_gj_rate_verification.py --outdir reports/MCfd_rate_verification_report --seed 229 --mc-repeats 384
```

The precision trade-off scripts in `code/MCfd_precision_tradeoff_final.py` and
`code/MCfd_precision_tradeoff_multialpha.py` were originally run in a `/mnt/data`
workspace. Their generated outputs are preserved in `figures/`, `reports/`, and
`packages_unpacked/`.

## Reviewer-Ready Section

`reviewer_ready_validation_section/reviewer_ready_validation_section/` contains
the final manuscript-facing Section 3.2 replacement:

- `section_3_2_revised.tex`: revised paper section.
- `response_to_reviewers_section_3_2.md`: concern-by-concern response map.
- `figures/`: paper-ready PNG/PDF figures.
- `data/` and `tables/`: CSV and table inputs used by the revised section.

## Focused Add-On Packages

`alpha2_precision_remark/` generates the finite-precision diagnostic near
`alpha = 2`:

```bash
conda run -n sciml python theoretical_analysis/alpha2_precision_remark/code/MCfd_alpha2_precision_diagnostic.py --outdir theoretical_analysis/alpha2_precision_remark
```

`gj_mr_rate_validation/` generates the finite-smoothness Gauss-Jacobi rate
validation:

```bash
conda run -n sciml python theoretical_analysis/gj_mr_rate_validation/MCfd_gj_Mr_rate_validation.py --outdir theoretical_analysis/gj_mr_rate_validation/MCfd_GJ_Mr_rate_outputs --alpha 1.5
```

## Figures

![Alpha sweep](MCfd_integrated_final_package/figures/alpha/fig01_alpha_sweep_exp.png)

![M sweep](MCfd_integrated_final_package/figures/M_effect/fig02_M_sweep_exp_diagnostic.png)

![Raw quotient precision](MCfd_integrated_final_package/figures/precision/fig04_precision_raw_quotients.png)

![Cutoff tradeoff](MCfd_integrated_final_package/figures/tradeoff/fig06_remark31_multialpha_delta_tradeoff_rates.png)

![MC rate](MCfd_integrated_final_package/figures/rate/fig09_mc_sqrtM_rate.png)

![GJ spectral rate](MCfd_integrated_final_package/figures/rate/fig10_gj_rho_spectral_rate.png)
