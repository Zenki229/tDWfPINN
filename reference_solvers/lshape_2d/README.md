# 2D L-Shaped Reference Solver

This folder contains the current L-shaped-domain reference generator used by
the JAX irregular-domain benchmark. The implementation wraps the grid builder
from `lshape_case_package/` and uses backward-Euler convolution quadrature for
the time-fractional modal equations.

## PDE

```text
Omega_L = [-1, 1]^2 \ [0, 1]^2

{}^C D_t^1.8 u - 0.25 Delta u = 0,
    (t, x, y) in (0, 5] x Omega_L

u = 0 on partial Omega_L
u(0, x, y) = g(x, y)
u_t(0, x, y) = 0.2 g(x, y)
```

The default initial profile is

```text
g(x, y) =
    exp(-((x + 0.55)^2 + (y + 0.45)^2) / 0.08)
  - 0.85 exp(-((x + 0.55)^2 + (y - 0.45)^2) / 0.06)
  + 0.60 exp(-((x - 0.45)^2 + (y + 0.55)^2) / 0.06)
```

## Generate

From the repository root:

```bash
python reference_solvers/lshape_2d/generate_lshape_reference.py
```

Default numerical settings:

```text
alpha = 1.8
diffusion_scale = 0.25
n_grid = 128
n_modes = 36
n_steps = 2000
n_frames = 81
T = 5
```

The grid size `128` is used as the current balance between reference quality
and CPU runtime. Increase `--n-modes` and `--n-steps` before increasing the grid
if the temporal or modal truncation error is the dominant concern.

Example high-resolution command:

```bash
python reference_solvers/lshape_2d/generate_lshape_reference.py --n-grid 128 --n-modes 64 --n-steps 4000 --n-frames 101
```

Outputs:

```text
data/lshape/lshape_reference.npz
data/lshape/lshape_reference.gif
```

The nested [`lshape_case_package/`](lshape_case_package/README.md) directory is
the original package used for the L-shaped mesh and preview figures.
