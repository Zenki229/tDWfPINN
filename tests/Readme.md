# Testing the Fractional Derivative Reference Formula

According to Equation (19) of the paper,

```math
u(t,x)=\left(
2E_{\alpha,1}(-\pi^2t^\alpha)
-tE_{\alpha,2}(-\pi^2t^\alpha)
\right)\sin(\pi x).
```

we have

```math
\partial_t^\alpha u(t,x)=-\pi^2u(t,x).
```

`tests/test_physics.py` currently evaluates the `MC-I` transformed formula
through `frac_diff_exact()` and compares it with the identity above. Coverage
of the network-based `_mc_i`, `_mc_ii`, `_gj_i`, and `_gj_ii` paths is provided
separately by the residual/backpropagation tests for the registered PDE cases.
