# Reference Solvers

This directory collects numerical reference-solution code. The PINN training
code stays in `libs/`, `jax_*.py`, and `conf/`; this directory is for data
generation and independent reference calculations.

| Path | Role |
| --- | --- |
| `burgers_1d/` | Backward-Euler convolution-quadrature and WENO-style code used to produce one-dimensional fractional Burgers numerical references. |
| `forward_1d/` | Analytic Mittag-Leffler reference GIF generator for the one-dimensional forward problem. |
| `lshape_2d/` | L-shaped-domain finite-difference/FEM reference package plus the current backward-Euler CQ reference generator. |
| `irregular_hole_2d/` | Circular-hole reference solver plus analytic manufactured-solution GIF generation. |

All reference GIF generation code should live in the PDE-specific folder here,
not in the top-level `scripts/` directory.
