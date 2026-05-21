# Tests

The active test suite targets the JAX implementation and the reference-solver
generation utilities.

Run the JAX training/PDE checks with:

```bash
pytest -q tests/test_jax_forward.py tests/test_jax_irregular.py tests/test_jax_pde.py tests/test_sampler.py
```

Run the reference and plotting checks with:

```bash
pytest -q tests/test_reference_gif_generators.py tests/test_lshape_reference.py tests/test_smoke_plot_outputs.py
```
