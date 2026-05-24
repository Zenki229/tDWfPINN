# tDWfPINN PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-active-ee4c2c.svg)](https://pytorch.org/)

This branch contains the PyTorch implementation of transformed
diffusion-wave fractional PINNs for Caputo orders `alpha in (1, 2)`. The
training code supports the transformed Type-I and Type-II formulas with either
Monte Carlo or Gauss-Jacobi quadrature:

| Method | Formula type | Quadrature |
| --- | --- | --- |
| `MC-I` | Type-I | Monte Carlo |
| `MC-II` | Type-II | Monte Carlo |
| `GJ-I` | Type-I | Gauss-Jacobi |
| `GJ-II` | Type-II | Gauss-Jacobi |

## Precision Requirement

This implementation is intended to run in `torch.float64`. The training entry
point sets

```python
torch.set_default_dtype(torch.float64)
```

inside `main()` before constructing the model, PDE objects, samplers, and
optimizers. Keep this default for all reported runs.

This is not only a quality setting. The transformed Type-II formulas
(`MC-II` and `GJ-II`) subtract nearby network values and divide by small
time-increment powers such as `tau^2` and `t^alpha`. In fp32, the cancellation
and small denominators can amplify roundoff enough to make the residual and
loss blow up, especially near `t = 0` or when the quadrature resolution is
increased. Type-I is usually less sensitive, but mixed precision between the
model, sampled points, quadrature nodes, and reference tensors can still create
incorrect comparisons or runtime dtype errors.

Do not switch this branch to fp32 unless the fractional operators and Type-II
stability are re-audited end to end.

The current PyTorch branch includes two one-dimensional benchmarks plus two
registered two-dimensional irregular-domain cases:

| Case | Config | PDE class | Reference used for plots |
| --- | --- | --- | --- |
| 1D forward | `pde=dw_forward` | `src.physics.dw_pde.DWForward` | Analytic Mittag-Leffler solution |
| 1D Burgers | `pde=burgers` | `src.physics.burgers.TimeFracBurgers1D` | `data/burgers_*.npz` |
| 2D circular hole | `pde=irregular_hole` | `src.physics.irregular_2d.IrregularHole2D` | Manufactured analytic solution |
| 2D L-shape | `pde=lshape` | `src.physics.irregular_2d.LShape2D` | `data/lshape/lshape_reference.npz` |

## Paper Context

For `alpha in (1, 2)`, the Caputo diffusion-wave operator is evaluated through
transformed representations rather than by differentiating the network twice in
time under a singular convolution kernel. In this branch:

| Paper ingredient | PyTorch implementation |
| --- | --- |
| Transformed Type-I representation | `_gj_i` and `_mc_i` in `src.physics.pde.TimeFracCaputoDiffusionWaveTwoDimPDE` and `src.physics.irregular_2d.Irregular2DBase` |
| Transformed Type-II representation | `_gj_ii` and `_mc_ii` in the same PDE bases |
| Gauss-Jacobi quadrature for the endpoint singular kernel | `src.physics.fractional.roots_jacobi` and `pde.gj_params.nums` |
| Monte Carlo quadrature for the same transformed integrals | `pde.monte_carlo_params.nums` and `pde.monte_carlo_params.eps` |
| Type-I/Type-II timing comparison | `timing.csv` with one row per `trainer.timing.epoch_steps` optimizer steps |

The practical distinction is that Type-I evaluates shifted time derivatives,
whereas Type-II evaluates shifted network values. The timing scripts expose
both choices for MC and GJ so the cost can be compared with the same model,
batch sizes, and quadrature counts.

## Install

Use a normal Python environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The local smoke tests were run with Python `3.13.2` and PyTorch `2.10.0`.
For GPU training, install the PyTorch build that matches the server CUDA stack
before installing the remaining requirements.

Check the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Repository Layout

| Path | Description |
| --- | --- |
| [`conf/`](conf/) | Hydra configuration for model, optimizer, trainer, plotter, and PDE cases. |
| [`src/train.py`](src/train.py) | Main PyTorch training entry point. |
| [`src/models/`](src/models/) | MLP network used by the PINN. |
| [`src/physics/`](src/physics/) | Fractional operators and PDE residual definitions. |
| [`src/data/`](src/data/) | 1D, circular-hole, and L-shaped domain samplers. |
| [`src/vis/`](src/vis/) | Matplotlib/Plotly plotting helpers and raw `.npz` export. |
| [`data/`](data/) | Stored 1D Burgers arrays and 2D irregular-domain reference data. |
| [`scripts/`](scripts/) | Shell launchers for full, debug, and 2D method timing runs. |
| [`tests/`](tests/) | Unit tests for config loading, fractional operators, and registered 2D cases. |

## PDE Cases

### 1D Forward Problem

The forward benchmark solves

```text
{}^C D_t^alpha u - lambda / (k^2 pi^2) u_xx = 0,
    (t, x) in (0, 2] x (0, 1),

u(t, 0) = u(t, 1) = 0,
u(0, x) = a sin(k pi x),
u_t(0, x) = b sin(k pi x).
```

The analytic solution used for evaluation is

```text
u(t, x) = sin(k pi x)
          [a E_{alpha,1}(-lambda t^alpha)
           + b t E_{alpha,2}(-lambda t^alpha)].
```

The default config is `alpha=1.75`, `k=2`, `lambda=1`, `a=1`, and `b=1`.

### 1D Burgers Problem

The Burgers benchmark solves

```text
{}^C D_t^alpha u + u u_x - (0.01 / pi) u_xx = 0,
    (t, x) in (0, 1.2] x (-1, 1),

u(t, -1) = u(t, 1) = 0,
u(0, x) = -sin(pi x),
u_t(0, x) = beta sin(pi x).
```

The default config is `alpha=1.5`, `beta=2`, and `pde.datafile=data/burgers_150.npz`.
Reference arrays are available for `alpha=1.25`, `1.5`, and `1.75`.

### 2D Circular-Hole Case

The circular-hole benchmark uses

```text
Omega = (-1, 1)^2 \ B_0.25((-0.3, 0.2)),

{}^C D_t^alpha u - div(a(x, y) grad u)
    + b(x, y) . grad u + lambda u^3 = f(t, x, y),

alpha = 1.5,
a(x, y) = 1 + 0.3 sin(pi x) cos(pi y),
b(x, y) = (1 + y, x - 1),
lambda = 1.
```

The manufactured exact solution is

```text
u(t, x, y) = q(t) phi(x, y),
q(t) = t^2 (1 - t)^2,
phi(x, y) = (1 - x^2)(1 - y^2)
            ((x + 0.3)^2 + (y - 0.2)^2 - 0.25^2).
```

Preview reference data:

![Circular-hole exact reference](data/irregular_hole/irregular_hole_exact.gif)

### 2D L-Shaped Case

The L-shaped benchmark uses

```text
Omega_L = [-1, 1]^2 \ [0, 1]^2,

{}^C D_t^1.8 u - 0.25 Delta u = 0,
    (t, x, y) in (0, 1] x Omega_L,

u = 0 on partial Omega_L,
u(0, x, y) = g(x, y),
u_t(0, x, y) = 0.2 g(x, y).
```

The initial profile is the three-bump profile implemented in
[`src/physics/irregular_2d.py`](src/physics/irregular_2d.py). Training uses
`data/lshape/lshape_reference.npz` for plotting the reference solution at
`t=T/2` and `t=T`. The README preview below uses the longer `T=5` GIF only to
make the wave-like evolution easier to inspect.

![L-shaped reference, T=5](data/lshape/lshape_reference_T5.gif)

## Training

Run the default one-dimensional forward case:

```bash
python src/train.py
```

Run a short CPU smoke test:

```bash
python src/train.py pde=dw_forward plot=matplotlib wandb.mode=disabled trainer.max_steps=10 trainer.timing.epoch_steps=5 trainer.batch_size.domain=8 trainer.batch_size.boundary=4 trainer.batch_size.initial=4 trainer.rad.use=false pde.gj_params.nums=8 pde.gauss_jacobi_params.nums=8 model.hidden_dim=16 model.num_layers=2
```

Run the two-dimensional cases:

```bash
python src/train.py pde=irregular_hole plot=matplotlib wandb.mode=disabled
python src/train.py pde=lshape plot=matplotlib wandb.mode=disabled
```

For a tiny smoke run that quickly proves the image and timing pipeline:

```bash
python src/train.py pde=irregular_hole plot=matplotlib wandb.mode=disabled trainer.max_steps=1 trainer.timing.epoch_steps=1 trainer.batch_size.domain=4 trainer.batch_size.boundary=2 trainer.batch_size.initial=2 trainer.rad.use=false pde.gj_params.nums=3 pde.gauss_jacobi_params.nums=3 pde.monte_carlo_params.nums=3 model.hidden_dim=8 model.num_layers=1 pde.plot_grid=24
python src/train.py pde=lshape plot=matplotlib wandb.mode=disabled trainer.max_steps=1 trainer.timing.epoch_steps=1 trainer.batch_size.domain=4 trainer.batch_size.boundary=2 trainer.batch_size.initial=2 trainer.rad.use=false pde.gj_params.nums=3 pde.gauss_jacobi_params.nums=3 pde.monte_carlo_params.nums=3 model.hidden_dim=8 model.num_layers=1
```

Enable the Adam + L-BFGS hybrid schedule with `optimizer.lbfgs.use=true`:

```bash
python src/train.py pde=irregular_hole plot=matplotlib wandb.mode=disabled optimizer.lbfgs.use=true trainer.max_steps=10000 trainer.steps_per_epoch=5000 optimizer.lbfgs.max_iter=10 optimizer.lbfgs.lr=0.01 trainer.rad.use=true
python src/train.py pde=lshape plot=matplotlib wandb.mode=disabled optimizer.lbfgs.use=true trainer.max_steps=10000 trainer.steps_per_epoch=5000 optimizer.lbfgs.max_iter=10 optimizer.lbfgs.lr=0.01 trainer.rad.use=true
```

For this path, `epochs = trainer.max_steps // trainer.steps_per_epoch`. Each
epoch samples one batch, applies RAD when enabled, runs Adam for
`trainer.steps_per_epoch` updates on that batch, then runs one PyTorch L-BFGS
phase using `optimizer.lbfgs.max_iter`. Timing records only the Adam phase, and
both Adam and L-BFGS losses are logged.

Training loss is logged every `trainer.loss_log_every_steps` Adam steps
(default `100`). Relative error, plotting, and checkpoint storage happen at
evaluation boundaries: every `trainer.eval_every_steps` in Adam-only runs and
every `trainer.eval_every_epochs` hybrid epoch by default.

For 2D cases, the `abs_error` figure title reports the relative L2 error for
that plotted time slice. W&B `eval/relative_error` and `L2_Relative_Error`
record the combined relative L2 error over all evaluated 2D points by
accumulating the global numerator and denominator before taking the ratio.

## Unified Script Runs

Use the unified PyTorch script for both 1D and 2D cases. Case aliases are
`1d`, `2d`, and `all`; concrete cases are `dw_forward`, `burgers`,
`irregular_hole`, and `lshape`.

Run both 2D irregular-domain cases across Type-I, Type-II, MC, and GJ variants:

```bash
STEPS=5000 GJ_QUAD=64 MC_QUAD=640 bash scripts/run_pytorch_cases.sh 2d GJ-I,GJ-II,MC-I,MC-II
```

Run only L-shape with Gauss-Jacobi Type-II:

```bash
STEPS=5000 GJ_QUAD=64 bash scripts/run_pytorch_cases.sh lshape GJ-II
```

Run Burgers over the available reference alphas:

```bash
ALPHAS=1.25,1.5,1.75 STEPS=5000 GJ_QUAD=80 MC_QUAD=80 bash scripts/run_pytorch_cases.sh burgers GJ-I,GJ-II
```

Run the L-shape benchmark over the matching reference alphas:

```bash
ALPHAS=1.25,1.5,1.75 STEPS=50000 GJ_QUAD=64 bash scripts/run_pytorch_cases.sh lshape GJ-II
```

Run a small RAD-enabled smoke case:

```bash
RAD_USE=1 RAD_RATIO=0.5 RAD_DOMAIN_BATCH=32 STEPS=5 GJ_QUAD=4 DOMAIN_BATCH=4 BOUNDARY_BATCH=4 INITIAL_BATCH=4 HIDDEN_DIM=8 NUM_LAYERS=1 bash scripts/run_pytorch_cases.sh burgers GJ-II
```

Important script parameters:

| Variable | Meaning | Default |
| --- | --- | --- |
| `CASES` | Default case selection if positional `$1` is omitted | `2d` |
| `METHODS` | Default method selection if positional `$2` is omitted | `GJ-I,GJ-II,MC-I,MC-II` |
| `ALPHAS` | Burgers / L-shape reference alpha list | `1.25,1.5,1.75` |
| `FORWARD_ALPHAS` | Optional `dw_forward` alpha sweep | unset |
| `STEPS` | Adam update steps | `5000` |
| `STEPS_PER_EPOCH` | Adam updates per hybrid epoch | `5000` |
| `USE_LBFGS` | Enable hybrid Adam + L-BFGS | `0` |
| `LBFGS_MAX_ITER` | L-BFGS iterations per hybrid epoch | `10` |
| `TIMING_EPOCH_STEPS` | Steps per timing row | `5000` |
| `LOSS_LOG_EVERY` | Adam steps between loss logs | `100` |
| `EVAL_EVERY_STEPS` | Adam-only evaluation interval | `5000` |
| `EVAL_EVERY_EPOCHS` | Hybrid evaluation interval | `1` |
| `DOMAIN_BATCH` | Interior collocation batch size | case default |
| `BOUNDARY_BATCH` | Boundary batch size | case default |
| `INITIAL_BATCH` | Initial-condition batch size | case default |
| `RAD_USE` | Enable residual-adaptive sampling (domain only) | `0` |
| `RAD_RATIO` | Fraction of training domain points replaced by RAD | `0.8` |
| `RAD_DOMAIN_BATCH` | Interior candidate batch for RAD | `1000` |
| `GJ_QUAD` | Gauss-Jacobi nodes | case default |
| `MC_QUAD` | Monte Carlo samples | case default |
| `HIDDEN_DIM` | MLP hidden width | `64` |
| `NUM_LAYERS` | MLP hidden layers | `4` |
| `PLOT_GRID` | 2D plot grid | `80` |
| `WANDB_MODE` | W&B mode | `disabled` |
| `PYTHON` | Python executable | `python` |

The old launchers remain available as compatibility wrappers:
`scripts/run_pytorch_2d_cases.sh` and `scripts/run_pytorch_burgers.sh`.

### Interactive Selector

For guided parameter entry, run `scripts/interactive_run.sh`. It validates each
input, exports the same environment variables documented in the table above,
and then invokes `scripts/run_pytorch_cases.sh` with the chosen cases and
methods:

```bash
bash scripts/interactive_run.sh
```

Interaction conventions:

- Press **Enter** (or type `pass`) at any prompt to accept the displayed default.
- At single-choice prompts, type either the option number (`1`, `2`, ...) or the option name.
- At the methods prompt, use a comma list (`1,2` or `GJ-I,MC-II`) or `all`.
- Press **Ctrl+C** to cancel at any time.
- At the final confirmation, **Enter** runs immediately, `p` prints a replay shell command, and `q` quits.

The selector walks through four sections. Each row below shows the prompt, the
environment variable it sets, and the Hydra override that
`scripts/run_pytorch_cases.sh` ultimately emits.

**1. Target.** Selects the PDE, the optional alpha list, and the
quadrature methods. The estimated run count is `(case units) x (number of methods)`,
where `burgers` and `lshape` each contribute one unit per alpha and the other
cases contribute one each.

| Prompt | Variable | Effect |
| --- | --- | --- |
| Select PDE target | `CASES_ARG` (positional `$1`) | Picks `pde=<case>` (or expands `1d`, `2d`, `all`). |
| Alpha list (burgers/lshape) | `ALPHAS` | One run per alpha. For `burgers` sets `pde.alpha=<a>` and `pde.datafile=data/burgers_<token>.npz`; for `lshape` sets `pde.alpha=<a>` and `pde.reference_data=data/lshape/lshape_reference_alpha<token>.npz`. |
| Select integration methods | `METHODS_CSV` (positional `$2`) | Sets `pde.method=<m>` for each chosen method. |

**2. Training Schedule.** Adam, L-BFGS hybrid, and logging cadence.

| Prompt | Variable | Hydra override |
| --- | --- | --- |
| Adam update steps | `STEPS` | `trainer.max_steps` |
| Enable Adam + L-BFGS hybrid training? | `USE_LBFGS` | `optimizer.lbfgs.use` |
| Adam steps per hybrid epoch | `STEPS_PER_EPOCH` | `trainer.steps_per_epoch` |
| L-BFGS max_iter per hybrid epoch | `LBFGS_MAX_ITER` | `optimizer.lbfgs.max_iter` (only emitted when hybrid is on) |
| Timing row interval in Adam steps | `TIMING_EPOCH_STEPS` | `trainer.timing.epoch_steps` |
| Loss logging interval in Adam steps | `LOSS_LOG_EVERY` | `trainer.loss_log_every_steps` (0 disables periodic logs) |
| Adam-only evaluation interval in steps | `EVAL_EVERY_STEPS` | `trainer.eval_every_steps` (0 keeps only the final evaluation) |
| Hybrid evaluation interval in epochs | `EVAL_EVERY_EPOCHS` | `trainer.eval_every_epochs` (0 keeps only the final evaluation) |

**3. Sampling and RAD.** Batch sizes and residual-adaptive sampling.

| Prompt | Variable | Hydra override |
| --- | --- | --- |
| Interior collocation batch size | `DOMAIN_BATCH` | `trainer.batch_size.domain` |
| Boundary batch size | `BOUNDARY_BATCH` | `trainer.batch_size.boundary` |
| Initial-condition batch size | `INITIAL_BATCH` | `trainer.batch_size.initial` |
| Enable residual-adaptive sampling (RAD, domain only) | `RAD_USE` | `trainer.rad.use` |
| RAD replacement ratio (domain points) | `RAD_RATIO` | `trainer.rad.ratio` |
| RAD interior candidate batch size | `RAD_DOMAIN_BATCH` | `trainer.rad.batch.domain` |
| Gauss-Jacobi quadrature nodes | `GJ_QUAD` | `pde.gj_params.nums` and `pde.gauss_jacobi_params.nums` |
| Monte Carlo samples | `MC_QUAD` | `pde.monte_carlo_params.nums` |

**4. Model and Output.** Network shape, plotting, and runtime backends.

| Prompt | Variable | Hydra override |
| --- | --- | --- |
| MLP hidden width | `HIDDEN_DIM` | `model.hidden_dim` |
| MLP hidden layers | `NUM_LAYERS` | `model.num_layers` |
| 2D plot grid | `PLOT_GRID` | `pde.plot_grid` (only for `irregular_hole` and `lshape`) |
| W&B mode | `WANDB_MODE` | `wandb.mode` (`disabled`, `offline`, `online`) |
| Python executable | `PYTHON` | Process launched as `${PYTHON} src/train.py ...` |

Case-dependent defaults: when the selected case is `burgers`, the batch and
quadrature defaults shift to the 1D Burgers regime (`DOMAIN_BATCH=1000`,
`BOUNDARY_BATCH=100`, `INITIAL_BATCH=100`, `GJ_QUAD=80`, `MC_QUAD=80`);
otherwise the defaults stay at the 2D regime (`64 / 16 / 16 / 64 / 640`). The
same case-dependent fallback also applies to non-interactive
`run_pytorch_cases.sh` invocations through `value_or_case_default`.

## Outputs

Hydra creates one timestamped run directory per command. Normal direct runs use

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

The unified timing scripts use case-specific directories:

```text
outputs/pytorch_2d/<case>/<method>/<YYYY-MM-DD_HH-MM-SS>/
outputs/pytorch_2d/lshape/alpha<alpha>/<method>/<YYYY-MM-DD_HH-MM-SS>/
outputs/pytorch_burgers/alpha<alpha>/<method>/<YYYY-MM-DD_HH-MM-SS>/
outputs/pytorch_1d/dw_forward/<method>/<YYYY-MM-DD_HH-MM-SS>/
```

Each run directory contains:

| Output | Description |
| --- | --- |
| `.hydra/` | Resolved Hydra config for the run. |
| `checkpoint_<step>.pt` | Model checkpoint saved at evaluation steps. |
| `timing.csv` | Paper-style timing rows. |
| `results/plots/*.jpg` | Independent `true`, `sol`, and `abs_error` figures. |
| `results/raw_data/*.npz` | Raw arrays used by each plotted panel. |

The timing file has this schema:

```text
epoch,step,epoch_steps,elapsed_seconds,total_seconds,average_epoch_seconds,loss,adam_loss,lbfgs_loss
```

By default, one timing epoch is `5000` Adam steps, matching the paper's timing
convention. For smoke tests, override `trainer.timing.epoch_steps=1`.

## Weights & Biases

W&B is disabled by default for local smoke tests. To log online:

```bash
wandb login
python src/train.py pde=lshape wandb.mode=online wandb.project=tDWfPINN wandb.entity=<your-entity>
```

Set `wandb.mode=offline` on machines without network access but where you still
want local W&B logs.

Hybrid runs log `train/loss_continuous` against `train/loss_event`. Adam loss
events are emitted every `trainer.loss_log_every_steps` Adam updates, and one
L-BFGS event is emitted at the end of each hybrid epoch. L-BFGS is treated as a
phase event instead of a fake optimizer-step sequence because PyTorch L-BFGS
may call its closure fewer or more times than a normal step loop. The raw phase
metrics are also logged as `train/adam_loss` and `train/lbfgs_loss`, while
`train/adam_step` preserves the Adam-update axis.

## Tests

Run all focused tests:

```bash
pytest -q tests
```

The 2D registration test checks that `irregular_hole` and `lshape` both support
`GJ-I`, `GJ-II`, `MC-I`, and `MC-II` residual evaluation and backpropagation.
