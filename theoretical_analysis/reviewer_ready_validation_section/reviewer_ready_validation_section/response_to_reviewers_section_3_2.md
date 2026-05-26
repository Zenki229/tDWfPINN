# Section 3.2 revision plan addressing reviewer concerns

This package contains a reviewer-ready replacement for Section 3.2, based on the latest PDF version of the manuscript and on the derivative-level numerical diagnostics already generated.

## Main reviewer concerns and corresponding fixes

| Reviewer concern | Revision made in `section_3_2_revised.tex` | Evidence added |
|---|---|---|
| The original wording says “validate theoretical statements,” which is too broad. | The section is retitled as **Derivative-level validation and implementation diagnostics**. The opening paragraph states that the theorems are already proved and that the experiments isolate quadrature and implementation effects. | Opening paragraph of revised Section 3.2. |
| It is unclear what “stable quotient” means. | Stable quotient formulas are explicitly defined for the exponential benchmark using `expm1` and Taylor endpoint replacement. | Equations for `K_f^{stab}` and `H_f^{stab}`. |
| The GJ large-`M` deterioration may be misread as failure of Gauss-Jacobi quadrature. | The text now separates mathematical quadrature error from raw floating-point endpoint cancellation. | Table `M` effect + Figure `fig_valid_M_sweep_exp_diagnostic`. |
| The fp8/fp16 experiment could be overclaimed. | The text explicitly says fp8 is a controlled quantization diagnostic, not native NumPy arithmetic or full mixed-precision PINN training. | Finite-precision paragraph + figure caption. |
| Remark 3.1 was described too strongly as validation. | The wording is changed to **asymptotic numerical check** of the leading bias and conditioning components, not full training-error validation. | Trade-off paragraph and Table `tab:valid-tradeoff-slopes`. |
| MC rate verification lacked reproducibility detail. | The text states that empirical RMS errors are computed over repeated independent runs using exact bounded kernels. | Rate-check paragraph + slope table. |
| GJ `rho^{-2M}` slope needs pre-saturation fitting and singularity explanation. | The rational benchmark, nearest singularity, Bernstein ellipse parameter, and polynomial prefactor are described explicitly. The fit is stated to exclude the floating-point plateau. | GJ rate paragraph + Table `tab:valid-rate-summary`. |
| Nonsmooth benchmark contained a mathematical typo and lacked mapping of interior kink. | The text now says `f \in C^1` but `f \notin C^2`, gives parameter values, and identifies the interior memory point `tau_c=1-t_c/t`. | Nonsmooth paragraph. |
| Section 3.2 should connect to Section 4. | A final implications paragraph explains why Section 4 uses moderate GJ sizes and MC cutoffs. | Final paragraph of revised Section 3.2. |

## Suggested usage

Replace the current `\subsection{Validation}` block with `section_3_2_revised.tex`. The figures are placed in `figures/`; the tables are embedded directly in the LaTeX section.

The finite-precision figure can be moved to an appendix if the main text becomes too long, but the definition of stable quotient should remain in the main text.
