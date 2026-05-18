import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from libs.jax_pde_forward import JAXDWForward
from libs.jax_pinn import create_model, create_train_state
from libs.jax_sample import TimeSpaceEasySampler
from libs.jax_utils import replicate


def _pde_cfg(method="GJ-II"):
    return OmegaConf.create({
        "al": 1.5,
        "k": 2,
        "lam": 4.0,
        "a": 1.0,
        "b": -0.5,
        "tlim": [0, 2],
        "xlim": [[0, 1]],
        "method": method,
        "GJ": {"nums": 5},
        "MC": {"nums": 5, "eps": 1e-8},
    })


def _model_cfg():
    return OmegaConf.create({
        "arch_name": "mlp",
        "input_dim": 2,
        "num_layers": 2,
        "hidden_dim": 10,
        "out_dim": 1,
        "activation": "tanh",
    })


def test_forward_losses_all_fractional_methods():
    key = jax.random.PRNGKey(0)
    model, params = create_model(key, _model_cfg())
    batch = {
        "in": jnp.ones((6, 2)),
        "bd": jnp.ones((2, 2)),
        "init": jnp.ones((2, 2)),
    }

    for method in ("GJ-I", "GJ-II", "MC-I", "MC-II"):
        pde = JAXDWForward(_pde_cfg(method))
        losses = pde.losses(model.apply, params, batch, key)
        assert set(losses) == {"in", "bd", "init", "init_dt"}
        assert losses["in"].shape == (6,), f"{method} returned {losses['in'].shape}"
        assert jnp.all(jnp.isfinite(losses["in"]))


def test_forward_exact_initial_conditions():
    pde = JAXDWForward(_pde_cfg("GJ-II"))
    points = np.array([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]])
    exact = pde.exact(points).reshape(-1)
    expected = np.sin(pde.k * np.pi * points[:, 1])
    np.testing.assert_allclose(exact, expected, atol=1e-10)


def test_forward_pmapped_train_step_single_cpu():
    n_devices = jax.local_device_count()
    key = jax.random.PRNGKey(1)
    optim_cfg = OmegaConf.create({"learning_rate": 1e-3, "grad_accum_steps": 1})
    weighting_cfg = OmegaConf.create({
        "scheme": "none",
        "init_weights": {"in": 1.0, "bd": 1.0, "init": 1.0, "init_dt": 1.0},
        "momentum": 0.9,
    })
    state = replicate(create_train_state(key, _model_cfg(), optim_cfg, weighting_cfg))
    pde = JAXDWForward(_pde_cfg("GJ-II"), weighting_cfg)
    sampler = TimeSpaceEasySampler(
        [[0, 1]],
        [0, 2],
        {"in": 4 * n_devices, "bd": 2 * n_devices, "init": 2 * n_devices},
        n_devices=n_devices,
        shard=True,
    )
    batch = next(iter(sampler))
    keys = jax.random.split(key, n_devices)
    state, loss_val, aux, keys = pde.step(state, batch, keys)
    assert loss_val.shape == (n_devices,)
    assert jnp.all(jnp.isfinite(loss_val))
    assert state.step.shape == (n_devices,)
