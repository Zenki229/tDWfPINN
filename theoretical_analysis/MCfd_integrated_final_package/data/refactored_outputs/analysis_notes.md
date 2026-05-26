# MCfd refactored validation notes

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
