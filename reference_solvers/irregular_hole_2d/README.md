# 2D Circular-Hole Reference Solver

This folder contains the numerical reference solver for the two-dimensional
circular-hole benchmark. The PDE currently used by the JAX case is:

```text
Omega = (-1, 1)^2 \ B_0.25((-0.3, 0.2))

{}^C D_t^alpha u - div(a(x, y) grad u)
    + b(x, y) . grad u + lambda u^3 = f(t, x, y),
    (t, x, y) in (0, 1] x Omega

u = 0 on partial Omega
u(0, x, y) = 0
u_t(0, x, y) = 0

a(x, y) = 1 + 0.3 sin(pi x) cos(pi y)
b(x, y) = (1 + y, x - 1)
lambda = 1
```

The generator `generate_irregular_hole_reference.py` builds a masked finite
difference operator on the circular-hole grid and advances the Caputo term with
backward-Euler convolution quadrature. It also evaluates the analytic
manufactured solution

```text
u*(t, x, y) = t^2 (1 - t)^2
              (1 - x^2)(1 - y^2)
              ((x + 0.3)^2 + (y - 0.2)^2 - 0.25^2)
```

on the same grid for diagnostics and for an exact-solution GIF.

Default output:

```text
data/irregular_hole/irregular_hole_reference.npz
data/irregular_hole/irregular_hole_reference.gif
data/irregular_hole/irregular_hole_exact.gif
```

Run:

```bash
conda run -n sciml python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py
```
