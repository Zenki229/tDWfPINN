import os
import csv
import time
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


def _cfg_get(config, name, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    try:
        return getattr(config, name)
    except (AttributeError, KeyError):
        return default


def _format_seconds(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


class EpochTimer:
    """Record paper-style timing every fixed number of optimizer steps."""

    def __init__(self, output_dir, enabled=True, epoch_steps=5000,
                 filename="timing.csv"):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.epoch_steps = int(epoch_steps)
        self.filename = filename
        self.path = os.path.join(output_dir, filename)
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_times = []

        if self.enabled and self.epoch_steps <= 0:
            raise ValueError("timing epoch_steps must be positive")

    @classmethod
    def from_config(cls, training_config, output_dir):
        timing_config = _cfg_get(training_config, "timing", None)
        enabled = _cfg_get(timing_config, "enabled", True)
        epoch_steps = _cfg_get(
            timing_config,
            "epoch_steps",
            _cfg_get(training_config, "epoch_steps", 5000),
        )
        filename = _cfg_get(timing_config, "filename", "timing.csv")
        return cls(output_dir, enabled=enabled, epoch_steps=epoch_steps,
                   filename=filename)

    def start(self):
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self.start_time = time.perf_counter()
        self.epoch_start_time = self.start_time
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "step",
                    "epoch_steps",
                    "elapsed_seconds",
                    "total_seconds",
                    "average_epoch_seconds",
                    "loss",
                ],
            )
            writer.writeheader()
        print(
            f"[*] Timing every {self.epoch_steps} steps; "
            f"writing {self.path}"
        )

    def maybe_record(self, step, loss_val):
        if not self.enabled or step % self.epoch_steps != 0:
            return None

        jax.block_until_ready(loss_val)
        now = time.perf_counter()
        elapsed = now - self.epoch_start_time
        total = now - self.start_time
        self.epoch_times.append(elapsed)
        avg = float(np.mean(self.epoch_times))
        loss = float(jnp.mean(loss_val))
        epoch = len(self.epoch_times)

        row = {
            "epoch": epoch,
            "step": step,
            "epoch_steps": self.epoch_steps,
            "elapsed_seconds": f"{elapsed:.8f}",
            "total_seconds": f"{total:.8f}",
            "average_epoch_seconds": f"{avg:.8f}",
            "loss": f"{loss:.8e}",
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        self.epoch_start_time = now
        print(
            f"  Timing epoch {epoch:>3d} | step {step:>7d} | "
            f"elapsed {_format_seconds(elapsed)} | "
            f"avg {_format_seconds(avg)}"
        )
        return {
            "timing/epoch": epoch,
            "timing/elapsed_seconds": elapsed,
            "timing/average_epoch_seconds": avg,
        }

    def finish(self, max_steps):
        if not self.enabled:
            return
        total = time.perf_counter() - self.start_time
        if self.epoch_times:
            avg = float(np.mean(self.epoch_times))
            print(
                f"[*] Timing summary: {len(self.epoch_times)} full timing "
                f"epoch(s), average {_format_seconds(avg)}, total "
                f"{_format_seconds(total)}"
            )
        else:
            print(
                f"[*] Timing summary: no complete {self.epoch_steps}-step "
                f"timing epoch in {max_steps} step(s); total "
                f"{_format_seconds(total)}"
            )
