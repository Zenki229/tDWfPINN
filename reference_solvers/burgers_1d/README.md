# 1D Burgers Reference GIFs

This folder contains the GIF generator for the stored one-dimensional
time-fractional Burgers numerical references.

## PDE

```text
{}^C D_t^alpha u + u u_x - (0.01 / pi) u_xx = 0,
    (t, x) in (0, 1.2] x (-1, 1)
u(t, -1) = u(t, 1) = 0
u(0, x) = -sin(pi x)
u_t(0, x) = beta sin(pi x)
```

The currently stored references in `data/burgers_125.npz`,
`data/burgers_150.npz`, and `data/burgers_175.npz` are numerical references,
not analytic solutions. They use zero initial velocity, `beta=0.0`, on a
`200 x 200` `(t, x)` grid.

The reference solver code kept here uses backward-Euler convolution quadrature
for the Caputo term, with WENO/Lax-Friedrichs components retained for the
Burgers spatial discretization.

## Generate

From the repository root:

```bash
python reference_solvers/burgers_1d/generate_burgers_reference_gifs.py
```

To specify the source arrays explicitly:

```bash
python reference_solvers/burgers_1d/generate_burgers_reference_gifs.py --burgers-data data/burgers_125.npz data/burgers_150.npz data/burgers_175.npz
```

Outputs:

```text
data/reference_1d/burgers_alpha1p25_reference.gif
data/reference_1d/burgers_alpha1p50_reference.gif
data/reference_1d/burgers_alpha1p75_reference.gif
data/reference_1d/burgers_reference_summary.csv
```

The generator detects the correct array orientation from
`u(0,x)=-sin(pi x)` before drawing frames. This prevents square-array transpose
mistakes when reading the stored `npz` files.
