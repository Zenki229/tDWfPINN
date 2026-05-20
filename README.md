# tDWfPINNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Pytorch 1.13](https://img.shields.io/badge/pytorch-1.13-blue.svg)](https://pytorch.org/)

This repository contains the code and generated artifacts for transformed
diffusion-wave fPINNs. The paper studies time-fractional diffusion-wave
equations with Caputo order `alpha in (1, 2)`, develops transformed Type-I and
Type-II representations of the fractional derivative, validates the numerical
quadrature behavior at derivative level, and provides PINN benchmark cases in
PyTorch and JAX.

The current active implementation work is in JAX. The legacy PyTorch entry
points are retained for comparison. External reference repositories such as
`pinns-jax/`, `jaxpi/`, and `dw_weno/` are ignored by git and are used only as
local reference material; curated numerical reference solvers live under
`reference_solvers/`.

## Repository Layout

| Path | Description |
| --- | --- |
| `libs/` | Shared PINN, sampler, PDE, evaluator, and timing utilities. |
| `conf/` | Hydra configs for model, optimizer, training, and PDE cases. |
| `jax_forward.py` | JAX Section 4.3 forward benchmark. |
| `jax_burgers.py` | JAX fractional Burgers benchmark. |
| `jax_irregular.py` | JAX irregular-domain cases, including circular-hole and L-shape domains. |
| `scripts/` | GIF generation, smoke plotting, and visualization scripts. |
| `data/` | Reference datasets and reference GIFs. |
| `outputs/smoke_results/` | CPU smoke figures: independent `true`, `sol`, and `abs_error` panels. |
| `theoretical_analysis/` | Derivative-level theory validation packages and paper-ready figures. |
| `reference_solvers/` | Numerical reference solvers for Burgers and two-dimensional reference cases. |
| `tests/` | Focused JAX and generation tests. |

## Theory Validation

The theoretical part of the paper proves equivalent transformed representations
of the Caputo derivative and analyzes consistency, endpoint stability,
Monte-Carlo error, Gauss-Jacobi error, and finite-precision behavior. The
corresponding numerical evidence is organized under
[`theoretical_analysis/`](theoretical_analysis/README.md).

The key point of these scripts is that they are derivative-level diagnostics,
not PINN training experiments. They isolate effects that would otherwise be
mixed with neural-network approximation and optimization:

- alpha sensitivity: MC sampling concentrates near the endpoint as
  `alpha -> 2`, increasing exposure to denominator cutoffs and endpoint
  cancellation.
- M sensitivity: MC median errors follow the expected stochastic decay, while
  raw GJ endpoint quotients can deteriorate for very large `M` because the
  leftmost node approaches zero.
- raw versus stable endpoint quotients: stable `expm1`/Taylor forms recover
  the smooth endpoint value that raw floating-point differences can lose.
- cutoff trade-off: the expected powers
  `delta^(2-alpha)`, `delta^(1-alpha)`, and `delta^(-alpha)` are checked
  numerically.
- convergence rates: MC `M^{-1/2}` RMS behavior and GJ spectral/finite
  smoothness rates are verified on controlled scalar benchmarks.
- nonsmooth tests: when the memory interval contains a kink, GJ convergence
  degrades as predicted by the smoothness assumptions.

The validation material is split into four subpackages:

| Package | Role |
| --- | --- |
| `theoretical_analysis/MCfd_integrated_final_package/` | Main integrated package for alpha sweep, M sweep, nonsmooth examples, precision diagnostics, cutoff trade-off, and MC/GJ rate checks. |
| `theoretical_analysis/reviewer_ready_validation_section/` | Revised Section 3.2 text, reviewer-response notes, paper-ready figures, CSV data, and tables. |
| `theoretical_analysis/alpha2_precision_remark/` | Focused diagnostic showing that raw GJ error growth near `alpha=2` is an implementation-level finite-precision effect. |
| `theoretical_analysis/gj_mr_rate_validation/` | Additional GJ finite-smoothness `M^{-r}` rate validation. |

Representative regeneration commands:

```bash
cd theoretical_analysis/MCfd_integrated_final_package
conda run -n sciml python code/MCfd_refactored_validation.py --outdir data/refactored_outputs --seed 229 --mc-repeats 24
conda run -n sciml python code/dw_fractional_validation_suite.py --outdir data/dw_validation_outputs --seed 229
conda run -n sciml python code/MCfd_mc_gj_rate_verification.py --outdir reports/MCfd_rate_verification_report --seed 229 --mc-repeats 384
```

Additional focused packages:

```bash
conda run -n sciml python theoretical_analysis/alpha2_precision_remark/code/MCfd_alpha2_precision_diagnostic.py --outdir theoretical_analysis/alpha2_precision_remark
conda run -n sciml python theoretical_analysis/gj_mr_rate_validation/MCfd_gj_Mr_rate_validation.py --outdir theoretical_analysis/gj_mr_rate_validation/MCfd_GJ_Mr_rate_outputs --alpha 1.5
```

Alpha sweep and endpoint mass:

![Alpha sweep](theoretical_analysis/MCfd_integrated_final_package/figures/alpha/fig01_alpha_sweep_exp.png)

M-sweep and endpoint cancellation:

![M sweep](theoretical_analysis/MCfd_integrated_final_package/figures/M_effect/fig02_M_sweep_exp_diagnostic.png)

Raw quotient precision:

![Raw quotient precision](theoretical_analysis/MCfd_integrated_final_package/figures/precision/fig04_precision_raw_quotients.png)

Cutoff trade-off:

![Cutoff tradeoff](theoretical_analysis/MCfd_integrated_final_package/figures/tradeoff/fig06_remark31_multialpha_delta_tradeoff_rates.png)

MC and GJ rate checks:

![MC rate](theoretical_analysis/MCfd_integrated_final_package/figures/rate/fig09_mc_sqrtM_rate.png)

![GJ spectral rate](theoretical_analysis/MCfd_integrated_final_package/figures/rate/fig10_gj_rho_spectral_rate.png)

Focused alpha-to-two diagnostic:

![Alpha-to-two diagnostic](theoretical_analysis/alpha2_precision_remark/figures/fig12_alpha2_gj_precision_sweep.png)

Finite-smoothness GJ rate:

![GJ finite-smoothness rate](theoretical_analysis/gj_mr_rate_validation/MCfd_GJ_Mr_rate_outputs/figures/fig14_gj_Mr_abstract_kernel.png)

## JAX Benchmark Cases

| Case | Entry | PDE / reference |
| --- | --- | --- |
| Burgers | `jax_burgers.py pde=burgers` | Reference arrays in `data/burgers_125.npz`, `data/burgers_150.npz`, and `data/burgers_175.npz`. |
| Section 4.3 forward | `jax_forward.py pde=forward` | Analytic Mittag-Leffler solution. |
| Circular-hole irregular domain | `jax_irregular.py pde=irregular_hole` | Two-dimensional circular-hole PDE; numerical reference workflow to be finalized. |
| L-shaped domain | `jax_irregular.py pde=lshape` | Constant-coefficient two-dimensional L-shaped reference case. |

### Two-Dimensional PDEs

Both two-dimensional benchmarks are treated as numerical-reference cases. The
PDE-specific reference and GIF generators live under `reference_solvers/`.

#### Circular-Hole Irregular Domain

The circular-hole case uses

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

Here `f(t, x, y)` is the prescribed source term for this benchmark. The
manufactured analytic solution used for diagnostics and the exact GIF is
`u*(t,x,y)=t^2(1-t)^2 phi(x,y)`, where `phi` is the boundary-vanishing spatial
factor in `reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py`.

#### L-Shaped Domain

The current L-shaped case uses

```text
Omega_L = [-1, 1]^2 \ [0, 1]^2

{}^C D_t^1.8 u - 0.25 Delta u = 0,
    (t, x, y) in (0, 5] x Omega_L

u = 0 on partial Omega_L
u(0, x, y) = g(x, y)
u_t(0, x, y) = 0.2 g(x, y)

g(x, y) =
    exp(-((x + 0.55)^2 + (y + 0.45)^2) / 0.08)
  - 0.85 exp(-((x + 0.55)^2 + (y - 0.45)^2) / 0.06)
  + 0.60 exp(-((x - 0.45)^2 + (y + 0.55)^2) / 0.06)
```

## Reference GIFs

### One-Dimensional Reference Solutions

All one-dimensional reference GIF generation code lives in the corresponding
PDE folders under `reference_solvers/`:

```bash
conda run -n sciml python reference_solvers/burgers_1d/generate_burgers_reference_gifs.py
conda run -n sciml python reference_solvers/forward_1d/generate_forward_reference_gif.py
```

These commands write GIFs and summary CSVs under `data/reference_1d/`.

#### Burgers Numerical References

The fractional Burgers benchmark solves

```text
{}^C D_t^alpha u + u u_x - (0.01 / pi) u_xx = 0,
    (t, x) in (0, 1.2] x (-1, 1)
u(t, -1) = u(t, 1) = 0
u(0, x) = -sin(pi x)
u_t(0, x) = beta sin(pi x)
```

There is no closed-form solution used for this case. The reference solution is
the stored numerical solution in `data/burgers_125.npz`,
`data/burgers_150.npz`, and `data/burgers_175.npz`, each containing `float64`
arrays `(t, x, u)` on a `200 x 200` grid. These stored reference arrays use
zero initial velocity, `beta=0.0`; keep `pde.beta` consistent when training
directly against them. The Burgers numerical reference code is collected in
`reference_solvers/burgers_1d/`. Regenerate the Burgers reference GIFs from
those numerical reference arrays with:

```bash
conda run -n sciml python reference_solvers/burgers_1d/generate_burgers_reference_gifs.py --burgers-data data/burgers_125.npz data/burgers_150.npz data/burgers_175.npz
```

The generator detects the correct `(t, x)` orientation from
`u(0, x)=-sin(pi x)`, which avoids the ambiguous square-array transpose issue.

| Alpha | Numerical reference data | GIF |
| --- | --- | --- |
| `1.25` | `data/burgers_125.npz` | `data/reference_1d/burgers_alpha1p25_reference.gif` |
| `1.50` | `data/burgers_150.npz` | `data/reference_1d/burgers_alpha1p50_reference.gif` |
| `1.75` | `data/burgers_175.npz` | `data/reference_1d/burgers_alpha1p75_reference.gif` |

![Burgers alpha 1.25](data/reference_1d/burgers_alpha1p25_reference.gif)

![Burgers alpha 1.50](data/reference_1d/burgers_alpha1p50_reference.gif)

![Burgers alpha 1.75](data/reference_1d/burgers_alpha1p75_reference.gif)

#### Section 4.3 Forward Analytic Reference

The forward diffusion-wave benchmark solves

```text
{}^C D_t^alpha u - lambda / (k^2 pi^2) u_xx = 0,
    (t, x) in (0, 2] x (0, 1)
u(t, 0) = u(t, 1) = 0
u(0, x) = a sin(k pi x)
u_t(0, x) = b sin(k pi x)
```

The analytic solution is

```text
u(t, x) = sin(k pi x)
          [a E_{alpha,1}(-lambda t^alpha)
           + b t E_{alpha,2}(-lambda t^alpha)].
```

For the paper Section 4.3 setting, `alpha=1.75`, `lambda=1`, `k=1`,
`a=1`, and `b=-0.5`. Regenerate only this analytic reference GIF with:

```bash
conda run -n sciml python reference_solvers/forward_1d/generate_forward_reference_gif.py --alpha 1.75 --lam 1.0 --k 1 --a 1.0 --b -0.5 --t-max 2.0 --x-points 401
```

![Forward alpha 1.75](data/reference_1d/forward_alpha1p75_reference.gif)

The generated one-dimensional references are summarized by:

| Reference | Source |
| --- | --- |
| `data/reference_1d/burgers_alpha1p25_reference.gif` | `data/burgers_125.npz` |
| `data/reference_1d/burgers_alpha1p50_reference.gif` | `data/burgers_150.npz` |
| `data/reference_1d/burgers_alpha1p75_reference.gif` | `data/burgers_175.npz` |
| `data/reference_1d/forward_alpha1p75_reference.gif` | analytic Mittag-Leffler solution on a dense 401-point x-grid |

### Two-Dimensional Reference GIFs

L-shaped-domain GIF generation is handled by:

```bash
conda run -n sciml python reference_solvers/lshape_2d/generate_lshape_reference.py
```

For the circular-hole case, the same PDE-specific generator writes both the
finite-difference numerical reference GIF and the analytic manufactured-solution
GIF:

```bash
conda run -n sciml python reference_solvers/irregular_hole_2d/generate_irregular_hole_reference.py
```

Outputs:

| Reference | Source |
| --- | --- |
| `data/irregular_hole/irregular_hole_reference.gif` | masked finite-difference reference with backward-Euler CQ |
| `data/irregular_hole/irregular_hole_exact.gif` | analytic manufactured solution `u*(t,x,y)=q(t)phi(x,y)` |

## Training

The default JAX config points to the Section 4.3 forward case:

```bash
conda run -n sciml python jax_forward.py
```

Run the irregular-domain cases with:

```bash
conda run -n sciml python jax_irregular.py pde=irregular_hole
conda run -n sciml python jax_irregular.py pde=lshape
```

For CPU-only smoke checks in the `sciml` conda environment, keep the batch and
quadrature sizes small:

```bash
conda run -n sciml python jax_forward.py wandb.mode=disabled training.max_steps=1 training.batch.in=4 training.batch.bd=2 training.batch.init=2 pde.GJ.nums=3 weighting.scheme=none
```

## Smoke Figures

Smoke figures are generated by training for a few CPU steps and plotting
`true`, `sol`, and `abs_error` panels independently. The heatmaps use the
paper-style `jet` colormap. These figures validate the data and plotting
pipeline; they are not convergence results.

```bash
conda run -n sciml python scripts/generate_smoke_results.py --case burgers
conda run -n sciml python scripts/generate_smoke_results.py --case forward
conda run -n sciml python scripts/generate_smoke_results.py --case irregular_hole
conda run -n sciml python scripts/generate_smoke_results.py --case lshape
```

The summary is written to:

```text
outputs/smoke_results/summary.csv
```

For two-dimensional domains, the script writes separate `x-y` heatmaps at
`t=T/2` and `t=T`, and each time slice has independent `true`, `sol`, and
`abs_error` files.

## Timing

Training scripts record paper-style timing by default. One timing epoch is
`5000` optimizer steps, matching the original PyTorch experiments where one
outer epoch ran 5000 Adam updates.

The timing CSV is written to the Hydra output directory as `timing.csv` and
contains:

```text
epoch, step, epoch_steps, elapsed_seconds, total_seconds, average_epoch_seconds, loss
```

Configure timing with:

```yaml
training:
  timing:
    enabled: true
    epoch_steps: 5000
    filename: timing.csv
```

## Tests

Run the focused JAX tests with:

```bash
conda run -n sciml pytest -q tests/test_jax_forward.py tests/test_jax_irregular.py tests/test_jax_pde.py tests/test_sampler.py
```

Run generation-related tests with:

```bash
conda run -n sciml pytest -q tests/test_lshape_reference.py tests/test_smoke_plot_outputs.py
```
