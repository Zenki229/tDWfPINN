# Migration Guide: PyTorch to JAX

This project has been migrated from PyTorch to JAX. This guide outlines the new structure and how to run experiments.

## New Dependencies
- `jax`, `jaxlib`
- `flax` (Neural Networks)
- `optax` (Optimization)
- `hydra-core` (Configuration)
- `wandb` (Logging)

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
  - `jax_pde_burgers.py`: JAX implementation of fractional Burgers PDE.
  - `jax_sample.py`: NumPy-based data sampler.
- `conf/`: Hydra configuration files.
  - `config.yaml`: Main config.
  - `model/`: Model configs.
  - `pde/`: PDE configs.
  - `training/`: Training configs.
- `jax_burgers.py`: Main training script using JAX, Hydra, and WandB.
- `tests/`: Unit tests.

## Running Experiments
To run the training script with default configuration:
```bash
python jax_burgers.py
```

To override configuration parameters:
```bash
python jax_burgers.py training.lr=0.01 pde.method=MC-I
```

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
python tests/test_jax_pde.py
python tests/test_sampler.py
```
