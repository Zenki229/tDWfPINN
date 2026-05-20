# Reference Solvers

This directory collects numerical reference-solution code. The PINN training
code stays in `libs/`, `jax_*.py`, and `conf/`; this directory is for data
generation and independent reference calculations.

| Path | Role |
| --- | --- |
| `burgers_1d/` | Backward-Euler convolution-quadrature and WENO-style code used to produce one-dimensional fractional Burgers numerical references. |
| `lshape_2d/` | L-shaped-domain finite-difference/FEM reference package plus the current backward-Euler CQ reference generator. |
| `irregular_hole_2d/` | Placeholder for the circular-hole two-dimensional numerical reference generator that will replace the current manufactured-reference evaluation path. |

The GIF-generation scripts and exact solver commands will be documented after
the reference-data workflow is finalized.
