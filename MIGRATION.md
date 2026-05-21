# Migration Guide: PyTorch to JAX

This project has been migrated from PyTorch to JAX. This guide outlines the new structure and how to run experiments.

## New Dependencies
- `jax`, `jaxlib`
- `flax` (Neural Networks)
- `optax` (Optimization)
- `hydra-core` (Configuration)
- `wandb` (Logging)
- `pymittagleffler` (Section 4.3 analytical solution)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Installing JAX with CUDA (GPU) Support

**Note:** The default `requirements.txt` installs the CPU-only version of JAX on Windows/Linux if not specified otherwise. To run on an NVIDIA GPU, you need to install the CUDA-enabled version of `jax` and `jaxlib`.

#### Linux (Recommended)
Run the following command to install JAX with CUDA 12 support:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
*If you are using CUDA 11, replace `cuda12_pip` with `cuda11_pip`.*

#### Windows
JAX's official GPU support on Windows is experimental. However, you can try:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
*Note: You might need to use WSL2 (Windows Subsystem for Linux) for the best experience and performance on Windows.*

Verify GPU detection:
```bash
python -c "import jax; print(jax.devices())"
```
Should output something like `[GpuDevice(id=0, process_index=0), ...]`.

## Directory Structure
- `libs/`
  - `jax_pinn.py`: Flax-based MLP implementation.
  - `jax_pde_forward.py`: JAX implementation of the Section 4.3 forward diffusion-wave case.
  - `jax_pde_irregular.py`: JAX implementations of the two irregular-domain diffusion-wave cases.
  - `jax_pde_burgers.py`: JAX implementation of fractional Burgers PDE.
  - `jax_sample.py`: NumPy-based data sampler.
- `conf/`: Hydra configuration files.
  - `config.yaml`: Main config.
  - `model/`: Model configs.
  - `pde/`: PDE configs.
  - `training/`: Training configs.
- `jax_forward.py`: Main Section 4.3 forward training script using JAX, Hydra, and WandB.
- `jax_irregular.py`: Main two-dimensional irregular-domain training script.
- `jax_burgers.py`: Burgers training script retained for the later case.
- `tests/`: Unit tests.

## Running Experiments
To run the training script with default configuration:
```bash
python jax_forward.py
```

To override configuration parameters:
```bash
python jax_forward.py training.learning_rate=0.01 pde.method=MC-I
```

To run the two irregular-domain cases:
```bash
python jax_irregular.py pde=irregular_hole
python jax_irregular.py pde=lshape
```

The L-shaped reference solution is stored in `data/lshape/lshape_reference.npz`,
with an accompanying GIF at `data/lshape/lshape_reference.gif`. Regenerate both
with:
```bash
python reference_solvers/lshape_2d/generate_lshape_reference.py
```
The generator uses `alpha=1.8`, `t in [0, 5]`, constant diffusion scale `0.25`,
L-shaped finite-difference eigenmodes, and backward-Euler convolution quadrature
for the fractional modal coefficients. The matching training PDE is
`D_t^1.8 u - 0.25 Delta u = 0`.

To regenerate the four smoke figures under `outputs/smoke_results/`, run:
```bash
python scripts/generate_smoke_results.py --case burgers
python scripts/generate_smoke_results.py --case forward
python scripts/generate_smoke_results.py --case irregular_hole
python scripts/generate_smoke_results.py --case lshape
```
The smoke generator writes each `true`, `sol`, and `abs_error` panel as an
independent PNG. The two-dimensional cases use separate `x-y` heatmaps at
`t=T/2` and `t=T` by default.

## Timing
JAX training scripts record paper-style timing by default. A timing epoch is
configured as `training.timing.epoch_steps`, defaulting to 5000 optimizer steps.
Each completed timing epoch is appended to `timing.csv` in the Hydra output
directory with elapsed seconds, total seconds, running average epoch time, and
loss.

To run on multiple GPUs, simply run the script on a machine with multiple GPUs. The script automatically detects available devices and uses `pmap` for data parallelism.

## Key Changes
1.  **Framework**: PyTorch `nn.Module` -> Flax `nn.Module`.
2.  **Differentiation**: `torch.autograd` -> `jax.grad`, `jax.vmap`.
3.  **Optimization**: `torch.optim` -> `optax`.
4.  **Config**: `ml_collections` -> `hydra`.
5.  **Logging**: Custom logging -> `wandb`.

## Verification
Run tests to verify the installation:
```bash
pytest tests/test_jax_forward.py tests/test_jax_irregular.py tests/test_jax_pde.py tests/test_sampler.py
```
