# Reference Solvers

This directory contains the independent reference-solution workflows used by
the JAX benchmark cases. PINN training code stays in `jax_*.py`, `libs/`, and
`conf/`; this directory is only for analytic references, numerical reference
solvers, and GIF generation.

Install the repository requirements from the project root before running these
commands:

```bash
python -m pip install -r requirements.txt
```

## Layout

| Path | Role |
| --- | --- |
| [`forward_1d/`](forward_1d/README.md) | Analytic Mittag-Leffler reference GIF for the Section 4.3 forward problem. |
| [`burgers_1d/`](burgers_1d/README.md) | GIF generation from stored one-dimensional Burgers numerical references. |
| [`irregular_hole_2d/`](irregular_hole_2d/README.md) | Circular-hole finite-difference reference plus manufactured exact GIF. |
| [`lshape_2d/`](lshape_2d/README.md) | L-shaped-domain backward-Euler convolution-quadrature reference generator. |

## Regenerate All Current GIFs

From the repository root:

```bash
python reference_solvers/forward_1d/generate_forward_reference_gif.py
python reference_solvers/burgers_1d/generate_burgers_reference_gifs.py
python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py
python reference_solvers/lshape_2d/generate_lshape_reference.py
python reference_solvers/lshape_2d/generate_lshape_reference.py --t-final 5.0 --output-prefix lshape_reference_T5 --gif-only
```

Default outputs:

| Case | Output |
| --- | --- |
| Forward 1D | `data/reference_1d/forward_alpha1p75_reference.gif` |
| Burgers 1D | `data/reference_1d/burgers_alpha1p25_reference.gif`, `burgers_alpha1p50_reference.gif`, `burgers_alpha1p75_reference.gif` |
| Circular-hole 2D | `data/irregular_hole/irregular_hole_reference.npz`, `irregular_hole_reference.gif`, `irregular_hole_exact.gif` |
| L-shape 2D | `data/lshape/lshape_reference.npz`, `lshape_reference.gif`; README preview `lshape_reference_T5.gif` |

Use the PDE-specific README files for grid sizes, command options, and PDE
definitions.
