# Reviewer-ready Section 3.2 revision package

This package revises the manuscript's validation section in response to reviewer-style concerns about wording, numerical evidence, and implementation diagnostics.

## Files

- `section_3_2_revised.tex`: full replacement for Section 3.2.
- `response_to_reviewers_section_3_2.md`: mapping from reviewer concern to revision.
- `figures/`: paper-ready figures in PNG and PDF formats.
- `data/`: CSV files used to generate the numerical tables and figures.

## Main changes

1. The section no longer claims that numerical tests “prove” the theory. It is now positioned as derivative-level validation and implementation diagnostics.
2. Stable quotient evaluation is explicitly defined.
3. The effect of quadrature size `M` is separated into Monte Carlo sampling behavior and Gauss-Jacobi endpoint-cancellation behavior.
4. The fp16/fp8-like experiment is clearly framed as a quotient-level precision diagnostic.
5. Remark 3.1 is presented as an asymptotic numerical check of the leading bias and conditioning components.
6. The MC `M^{-1/2}` and GJ `rho^{-2M}` rates are summarized in a numerical table.
7. The nonsmooth example states the precise regularity loss and maps the interior kink to the memory variable.
