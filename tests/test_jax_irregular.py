import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from libs.jax_pde_irregular import JAXIrregularHoleDW, JAXLShapeDW
from libs.jax_pinn import create_model, create_train_state
from libs.jax_sample import IrregularHoleSampler, LShapeSampler
from libs.jax_utils import replicate


def _model_cfg():
    return OmegaConf.create({
        "arch_name": "mlp",
        "input_dim": 3,
        "num_layers": 2,
        "hidden_dim": 8,
        "out_dim": 1,
        "activation": "tanh",
    })


def _hole_cfg(method="GJ-II"):
    return OmegaConf.create({
        "al": 1.5,
        "tlim": [0, 1],
        "method": method,
        "center": [-0.3, 0.2],
        "r0": 0.25,
        "lam": 1.0,
        "diffusion_amp": 0.3,
        "GJ": {"nums": 3},
        "MC": {"nums": 3, "eps": 1e-8},
    })


def _lshape_cfg(method="GJ-II"):
    return OmegaConf.create({
        "al": 1.5,
        "tlim": [0, 1],
        "method": method,
        "diffusion_amp": 0.25,
        "velocity_scale": 0.2,
        "GJ": {"nums": 3},
        "MC": {"nums": 3, "eps": 1e-8},
    })


def _weighting_cfg():
    return OmegaConf.create({
        "scheme": "none",
        "init_weights": {"in": 1.0, "bd": 1.0, "init": 1.0, "init_dt": 1.0},
        "momentum": 0.9,
    })


def test_irregular_hole_sampler_domain_and_boundary():
    sampler = IrregularHoleSampler(
        [0, 1],
        {"in": 100, "bd": 100, "init": 100},
        center=(-0.3, 0.2),
        r0=0.25,
    )
    batch = sampler.sample()
    center = np.array([-0.3, 0.2])

    for key in ("in", "init"):
        xy = batch[key][:, 1:3]
        assert np.all(np.abs(xy) < 1.0)
        assert np.all(np.sum((xy - center) ** 2, axis=1) > 0.25 ** 2)

    bd = batch["bd"][:, 1:3]
    on_square = np.isclose(np.abs(bd), 1.0).any(axis=1)
    on_circle = np.isclose(np.sum((bd - center) ** 2, axis=1), 0.25 ** 2)
    assert np.all(on_square | on_circle)


def test_lshape_sampler_domain_and_boundary():
    sampler = LShapeSampler([0, 1], {"in": 100, "bd": 100, "init": 100})
    batch = sampler.sample()

    for key in ("in", "init"):
        x = batch[key][:, 1]
        y = batch[key][:, 2]
        assert np.all((-1.0 < x) & (x < 1.0) & (-1.0 < y) & (y < 1.0))
        assert np.all((x < 0.0) | (y < 0.0))

    x = batch["bd"][:, 1]
    y = batch["bd"][:, 2]
    on_boundary = (
        np.isclose(x, -1.0)
        | np.isclose(y, -1.0)
        | (np.isclose(x, 1.0) & (y <= 0.0))
        | (np.isclose(y, 1.0) & (x <= 0.0))
        | (np.isclose(x, 0.0) & (y >= 0.0))
        | (np.isclose(y, 0.0) & (x >= 0.0))
    )
    assert np.all(on_boundary)


def test_irregular_hole_exact_initial_and_boundary_values():
    pde = JAXIrregularHoleDW(_hole_cfg())
    points = np.array([
        [0.0, -0.5, -0.5],
        [0.0, 0.5, -0.5],
        [0.5, -1.0, 0.2],
        [0.5, 0.0, 1.0],
        [0.5, -0.3 + 0.25, 0.2],
    ])
    exact = pde.exact(points).reshape(-1)
    np.testing.assert_allclose(exact, np.zeros_like(exact), atol=1e-10)


def _assert_losses_finite(pde, batch, key):
    model, params = create_model(key, _model_cfg())
    losses = pde.losses(model.apply, params, batch, key)
    assert set(losses) == {"in", "bd", "init", "init_dt"}
    for value in losses.values():
        assert value.ndim == 1
        assert jnp.all(jnp.isfinite(value))


def test_irregular_losses_all_fractional_methods():
    key = jax.random.PRNGKey(0)
    hole_batch = IrregularHoleSampler(
        [0, 1],
        {"in": 4, "bd": 4, "init": 4},
        center=(-0.3, 0.2),
        r0=0.25,
    ).sample()
    lshape_batch = LShapeSampler([0, 1], {"in": 4, "bd": 4, "init": 4}).sample()

    for method in ("GJ-I", "GJ-II", "MC-I", "MC-II"):
        _assert_losses_finite(JAXIrregularHoleDW(_hole_cfg(method)), hole_batch, key)
        _assert_losses_finite(JAXLShapeDW(_lshape_cfg(method)), lshape_batch, key)


def test_irregular_pmapped_train_step_single_cpu():
    n_devices = jax.local_device_count()
    key = jax.random.PRNGKey(1)
    optim_cfg = OmegaConf.create({"learning_rate": 1e-3, "grad_accum_steps": 1})
    state = replicate(create_train_state(key, _model_cfg(), optim_cfg, _weighting_cfg()))
    pde = JAXIrregularHoleDW(_hole_cfg("GJ-II"), _weighting_cfg())
    sampler = IrregularHoleSampler(
        [0, 1],
        {"in": 4 * n_devices, "bd": 2 * n_devices, "init": 2 * n_devices},
        center=(-0.3, 0.2),
        r0=0.25,
        n_devices=n_devices,
        shard=True,
    )
    batch = next(iter(sampler))
    keys = jax.random.split(key, n_devices)
    state, loss_val, aux, keys = pde.step(state, batch, keys)
    assert loss_val.shape == (n_devices,)
    assert jnp.all(jnp.isfinite(loss_val))
    assert state.step.shape == (n_devices,)
