# L-Shaped Case Package

This package contains the original L-shaped-domain helper code used by
`reference_solvers/lshape_2d/generate_lshape_reference.py`.

## Domain

```text
Omega_L = [-1, 1]^2 \ [0, 1]^2
```

## Contents

| Path | Role |
| --- | --- |
| `code/lshape_fd_reference_and_gif.py` | Self-contained finite-difference grid builder, eigenmode reference preview, and GIF generator. |
| `code/lshape_fractional_dw_fem_reference.py` | Higher-fidelity scikit-fem reference prototype on the built-in L-shaped mesh. |
| `figures/` | Preview heatmaps, surface plots, and GIFs from the original package. |
| `tex/lshape_case_description.tex` | Manuscript-ready LaTeX case description. |

## Install Package Extras

From this directory, install the package-specific optional dependencies when
running the original preview scripts:

```bash
python -m pip install -r requirements.txt
```

## Run Original Preview Scripts

From this directory:

```bash
python code/lshape_fd_reference_and_gif.py
python code/lshape_fractional_dw_fem_reference.py
```

For the repository's current reference data used by JAX training, prefer the
root-level command:

```bash
python reference_solvers/lshape_2d/generate_lshape_reference.py
```
