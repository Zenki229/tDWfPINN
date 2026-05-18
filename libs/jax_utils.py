import os
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
import jax.flatten_util as fu


def flatten_pytree(pytree):
    """Flatten a pytree into a 1D array."""
    flat, _ = fu.ravel_pytree(pytree)
    return flat


def l2_norm(pytree):
    """Compute L2 norm of a pytree."""
    return jnp.sqrt(jnp.sum(jnp.array([jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(pytree)])))


@partial(jax.jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    """JIT-compiled Jacobian of a scalar function w.r.t. params."""
    def fn(p):
        return apply_fn(p, *args)
    jac = jax.jacrev(fn)(params)
    return flatten_pytree(jac)


@partial(jax.jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    """JIT-compiled NTK: J @ J^T for a scalar function."""
    jac = jacobian_fn(apply_fn, params, *args)
    return jnp.dot(jac, jac.T)


def replicate(state, devices=None):
    """Replicate a TrainState across devices."""
    if devices is None:
        devices = jax.local_devices()
    return jax.device_put_replicated(state, devices)


def unreplicate(state):
    """Extract first replica from a replicated TrainState."""
    return jax.tree_util.tree_map(
        lambda x: x[0] if hasattr(x, "ndim") and x.ndim > 0 else x,
        state,
    )


def save_checkpoint(state, workdir, keep=5, name="checkpoint"):
    """Save checkpoint (only process 0). Uses orbax via flax checkpoints."""
    if jax.process_index() == 0:
        ckpt_dir = os.path.join(workdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Unreplicate state if needed
        if hasattr(state, 'step'):
            s = unreplicate(state) if state.step.ndim > 0 else state
        else:
            s = state
        checkpoints.save_checkpoint(ckpt_dir, s, step=s.step, keep=keep, prefix=name)


def restore_checkpoint(state, workdir, step=None, name="checkpoint"):
    """Restore checkpoint. Handles pmap → single-device transition."""
    ckpt_dir = os.path.join(workdir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return state
    target = unreplicate(state) if hasattr(state, 'step') and state.step.ndim > 0 else state
    restored = checkpoints.restore_checkpoint(ckpt_dir, target=target,
                                              step=step, prefix=name)
    if restored is target:
        return state
    return restored
