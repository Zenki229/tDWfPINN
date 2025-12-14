# JAX Migration & Enhancement Plan for PINN Project

This plan outlines the steps to migrate the existing PyTorch-based Physics-Informed Neural Network (PINN) project to JAX, integrating modern MLOps tools (Hydra, WandB) and ensuring scalability and correctness.

## Phase 1: Foundation & Dependencies
**Objective**: Set up the environment and migrate basic components.
1.  **Dependency Management**:
    -   Create `requirements.txt` including `jax`, `jaxlib`, `flax`, `optax`, `hydra-core`, `wandb`, `matplotlib`, `numpy`, `scipy`.
2.  **Data Sampling (`libs/easy_sample.py`)**:
    -   Refactor `TimeSpaceEasySampler` to return JAX arrays or NumPy arrays.
    -   Replace `torch.rand` with `jax.random` or `numpy.random`.
    -   Optimize `rad_sampler` using `jax.random.choice` for efficiency.
3.  **Model Architecture (`libs/pinn.py`)**:
    -   Replace `torch.nn.Module` MLP with `flax.linen.Module`.
    -   Implement weight initialization using JAX keys.
    -   Ensure activation functions match the original (Tanh, etc.).

## Phase 2: Core Physics & Numerics (The "First Principles" Part)
**Objective**: Re-implement physics constraints with strict mathematical correctness using JAX.
1.  **Fractional Derivatives (`libs/pde_burgers.py`)**:
    -   Replace `torch.autograd.grad` with `jax.grad` and `jax.jacfwd/jacrev`.
    -   Implement high-order derivatives (dt, dx, dxx) using JAX's functional transformations.
2.  **Integration Schemes**:
    -   Port `MC-I`, `MC-II` (Monte Carlo) and `GJ-I`, `GJ-II` (Gauss-Jacobi) methods.
    -   Use `jax.vmap` to vectorize operations over quadrature points, replacing manual loops.
    -   Ensure numerical stability and verify against PyTorch baseline (unit tests).

## Phase 3: Configuration & Experiment Tracking
**Objective**: Upgrade parameter management and logging.
1.  **Hydra Integration**:
    -   Create a hierarchical config structure in `conf/` (e.g., `conf/config.yaml`, `conf/model/mlp.yaml`, `conf/pde/burgers.yaml`).
    -   Refactor `burgers.py` to use `@hydra.main`.
2.  **WandB Integration**:
    -   Initialize WandB in the training script.
    -   Log loss components (boundary, initial, domain), error metrics, and gradients.
    -   Implement artifact logging for saved models and plots.

## Phase 4: Training Loop & Parallelization
**Objective**: Implement a high-performance training loop supporting multi-GPU.
1.  **Functional Training Loop**:
    -   Define a pure function `train_step(state, batch)` using `optax` for optimization.
    -   Use `jax.jit` to compile the update step.
2.  **Multi-GPU Support (`pmap`/`pjit`)**:
    -   Implement data sharding for the sampler.
    -   Use `jax.pmap` to replicate the training step across devices.
    -   Implement gradient synchronization (via `jax.lax.pmean`).
3.  **Evaluation & Checkpointing**:
    -   Port `evaluator` to use JAX for inference.
    -   Use `flax.training.checkpoints` for model persistence.

## Phase 5: Quality Assurance
**Objective**: Ensure correctness and maintainability.
1.  **Unit Tests**:
    -   Create `tests/` directory.
    -   Write tests comparing JAX derivative computations vs. analytical solutions.
    -   Write regression tests comparing JAX output vs. PyTorch output (for fixed weights).
2.  **Documentation**:
    -   Add docstrings to all new functions.
    -   Create `MIGRATION.md` detailing the changes and how to run experiments.
