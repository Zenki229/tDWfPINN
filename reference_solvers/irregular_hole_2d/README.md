# 2D Circular-Hole Reference Solver

This folder contains the reference workflow for the two-dimensional
circular-hole benchmark. It writes both a numerical reference and an analytic
manufactured-solution GIF.

## PDE

```text
Omega = (-1, 1)^2 \ B_0.25((-0.3, 0.2))

{}^C D_t^alpha u - div(a(x, y) grad u)
    + b(x, y) . grad u + lambda u^3 = f(t, x, y),
    (t, x, y) in (0, 1] x Omega

u = 0 on partial Omega
u(0, x, y) = 0
u_t(0, x, y) = 0

alpha = 1.5
a(x, y) = 1 + 0.3 sin(pi x) cos(pi y)
b(x, y) = (1 + y, x - 1)
lambda = 1
```

The manufactured solution used for diagnostics is

```text
u*(t, x, y) = t^2 (1 - t)^2
              (1 - x^2)(1 - y^2)
              ((x + 0.3)^2 + (y - 0.2)^2 - 0.25^2)
```

The numerical reference uses a masked finite-difference operator on the
circular-hole grid and advances the fractional time derivative with
backward-Euler convolution quadrature. The nonlinear cubic term is treated
explicitly with the previous time level.

## Generate

From the repository root:

```bash
python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py
```

Use `--alpha` to generate another fractional order.

Default numerical settings:

```text
n_grid = 128
n_steps = 600
n_frames = 81
T = 1
```

For a higher-accuracy reference, increase the grid and time resolution:

```bash
python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py --n-grid 192 --n-steps 1200 --n-frames 101
```

Outputs:

```text
data/irregular_hole/irregular_hole_reference.npz
data/irregular_hole/irregular_hole_reference.gif
data/irregular_hole/irregular_hole_exact.gif
```

To keep the paper orders side by side without overwriting files:

```bash
python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py --alpha 1.25 --tag-alpha
python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py --alpha 1.50 --tag-alpha
python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py --alpha 1.75 --tag-alpha
```
