# 2D L-Shaped Reference Solver

This folder contains the current L-shaped-domain reference generator used by
the PyTorch irregular-domain benchmark. The implementation wraps the grid builder
from `lshape_case_package/` and uses backward-Euler convolution quadrature for
the time-fractional modal equations.

## PDE

```math
\Omega_L = [-1,1]^2 \setminus [0,1]^2.
```

```math
{}^C D_t^{1.8}u - 0.25\Delta u = 0,
\qquad
(t,x,y) \in (0,1] \times \Omega_L,
```

with conditions

```math
\begin{gathered}
u = 0 \quad \text{on } \partial\Omega_L, \\
u(0,x,y) = g(x,y),
\qquad
u_t(0,x,y) = 0.2g(x,y).
\end{gathered}
```

The default initial profile is

```math
\begin{split}
g(x,y) ={}&
\exp\!\left(-\frac{(x+0.55)^2+(y+0.45)^2}{0.08}\right) \\
&-0.85\exp\!\left(-\frac{(x+0.55)^2+(y-0.45)^2}{0.06}\right) \\
&+0.60\exp\!\left(-\frac{(x-0.45)^2+(y+0.55)^2}{0.06}\right).
\end{split}
```

## Generate

Install the repository requirements, which include Matplotlib and Pillow, then
run the generator from the repository root:

```bash
python -m pip install -r requirements.txt
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
T = 1
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

The default `T = 1` archive is the reference used by training. The project
README uses a longer GIF-only preview:

```bash
python reference_solvers/lshape_2d/generate_lshape_reference.py --t-final 5.0 --output-prefix lshape_reference_T5 --gif-only
```

Output:

```text
data/lshape/lshape_reference_T5.gif
```

To keep several fractional orders side by side, append an alpha token:

```bash
python reference_solvers/lshape_2d/generate_lshape_reference.py --alpha 1.25 --tag-alpha
python reference_solvers/lshape_2d/generate_lshape_reference.py --alpha 1.50 --tag-alpha
python reference_solvers/lshape_2d/generate_lshape_reference.py --alpha 1.75 --tag-alpha
```

These commands write:

```text
data/lshape/lshape_reference_alpha1p25.npz
data/lshape/lshape_reference_alpha1p25.gif
data/lshape/lshape_reference_alpha1p50.npz
data/lshape/lshape_reference_alpha1p50.gif
data/lshape/lshape_reference_alpha1p75.npz
data/lshape/lshape_reference_alpha1p75.gif
```

The nested [`lshape_case_package/`](lshape_case_package/README.md) directory is
the original package used for the L-shaped mesh and preview figures.
