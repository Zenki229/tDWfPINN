import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from libs.jax_evaluator import BaseEvaluator
from libs.jax_pde_irregular import JAXIrregularHoleDW, JAXLShapeDW
from libs.jax_pinn import create_train_state
from libs.jax_run_metadata import prepare_wandb_run, wandb_init_kwargs
from libs.jax_sample import IrregularHoleSampler, LShapeSampler
from libs.jax_utils import (
    EpochTimer,
    replicate,
    restore_checkpoint,
    save_checkpoint,
    unreplicate,
)


def _cfg_get(config, name, default=None):
    if config is None:
        return default
    try:
        return getattr(config, name)
    except (AttributeError, KeyError):
        return default


def _build_case(cfg, n_devices):
    batch = OmegaConf.to_container(cfg.training.batch)
    if cfg.pde.name == "irregular_hole":
        pde = JAXIrregularHoleDW(cfg.pde, cfg.weighting)
        sampler = IrregularHoleSampler(
            tlim=cfg.pde.tlim,
            batch=batch,
            center=cfg.pde.center,
            r0=cfg.pde.r0,
            n_devices=n_devices,
            seed=cfg.seed,
            shard=True,
        )
        label = "circular-hole manufactured diffusion-wave"
    elif cfg.pde.name == "lshape":
        pde = JAXLShapeDW(cfg.pde, cfg.weighting)
        sampler = LShapeSampler(
            tlim=cfg.pde.tlim,
            batch=batch,
            n_devices=n_devices,
            seed=cfg.seed,
            shard=True,
        )
        label = "L-shaped reference-comparison diffusion-wave"
    else:
        raise ValueError(
            "jax_irregular.py expects pde.name to be irregular_hole or lshape"
        )
    return pde, sampler, label


def _rad_enabled(cfg):
    return bool(_cfg_get(_cfg_get(cfg.pde, "RAD"), "use", False))


def _rad_resample_batch(cfg, pde, sampler, state, batch_host, step):
    rad_cfg = cfg.pde.RAD
    ratio = float(_cfg_get(rad_cfg, "ratio", 0.3))
    ratio = min(max(ratio, 0.0), 1.0)
    batch_in = int(batch_host["in"].shape[0])
    num_rad = min(batch_in, max(1, int(round(batch_in * ratio))))
    candidate_cfg = _cfg_get(rad_cfg, "batch")
    candidate_in = int(_cfg_get(candidate_cfg, "in", max(batch_in, num_rad)))
    candidate_in = max(candidate_in, num_rad)

    candidates = sampler.sample_interior(candidate_in)
    eval_state = unreplicate(state)
    key = jax.random.PRNGKey(int(cfg.seed) + int(step) + 7919)
    residual = pde.r_net(
        eval_state.apply_fn,
        eval_state.params,
        jnp.asarray(candidates),
        key,
    )
    residual = np.asarray(jax.device_get(residual)).reshape(-1)
    err = np.square(residual)
    err_sum = float(np.sum(err))
    if not np.isfinite(err_sum) or err_sum <= 1e-12:
        prob = None
    else:
        prob = err / err_sum

    replace = num_rad <= len(candidates)
    indices = np.random.default_rng(int(cfg.seed) + int(step)).choice(
        len(candidates),
        size=num_rad,
        replace=not replace,
        p=prob,
    )
    selected = candidates[indices]
    keep_count = batch_in - num_rad
    if keep_count > 0:
        kept = batch_host["in"][:keep_count]
        batch_host["in"] = np.concatenate([kept, selected], axis=0)
    else:
        batch_host["in"] = selected
    return batch_host


def _next_training_batch(cfg, pde, sampler, state, step):
    batch_host = sampler.sample()
    if _rad_enabled(cfg):
        batch_host = _rad_resample_batch(cfg, pde, sampler, state, batch_host, step)
    return sampler._shard(batch_host)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg.model.input_dim = 3
    prepare_wandb_run(cfg)

    if cfg.wandb.mode != "disabled":
        wandb.init(**wandb_init_kwargs(cfg))

    n_devices = jax.local_device_count()
    print(f"[*] Running on {n_devices} device(s): {jax.devices()}")
    print(f"[*] Run name: {cfg.wandb.name}")
    print(f"[*] W&B group: {cfg.wandb.group}")

    root_key = jax.random.PRNGKey(cfg.seed)
    root_key, model_key = jax.random.split(root_key)

    state = create_train_state(model_key, cfg.model, cfg.training, cfg.weighting)
    print(
        f"[*] Model: {cfg.model.num_layers} layers x "
        f"{cfg.model.hidden_dim} ({cfg.model.activation}), input_dim={cfg.model.input_dim}"
    )

    pde, sampler, label = _build_case(cfg, n_devices)
    evaluator = BaseEvaluator(cfg, pde)
    print(f"[*] PDE: {label}")
    print(f"[*] Method: {cfg.pde.method}, alpha = {cfg.pde.al}")
    print(f"[*] Optimizer: {cfg.training.optimizer}")
    print(f"[*] RAD: {'enabled' if _rad_enabled(cfg) else 'disabled'}")

    state = replicate(state)
    pmap_keys = jax.random.split(root_key, n_devices)

    max_steps = cfg.training.max_steps
    log_every = getattr(cfg.training, "log_every_steps", 100)
    save_every = getattr(
        cfg.saving, "save_every_steps", getattr(cfg.training, "save_every_steps", 1000)
    )
    keep_ckpts = getattr(cfg.saving, "num_keep_ckpts", 5)
    update_weights_every = getattr(
        cfg.weighting,
        "update_every_steps",
        getattr(cfg.training, "update_weights_every_steps", 500),
    )
    weighting_scheme = getattr(cfg.weighting, "scheme", "none")
    workdir = HydraConfig.get().runtime.output_dir
    os.makedirs(workdir, exist_ok=True)
    timer = EpochTimer.from_config(cfg.training, workdir)

    print(f"[*] Training {max_steps} steps, log every {log_every}")
    print(f"[*] Workdir: {workdir}")

    if getattr(cfg.training, "restore", False):
        restored = restore_checkpoint(state, workdir)
        if restored is not state:
            state = replicate(restored)

    timer.start()
    for step in range(max_steps):
        batch_sharded = _next_training_batch(cfg, pde, sampler, state, step)
        state, loss_val, aux, pmap_keys = pde.step(state, batch_sharded, pmap_keys)
        timing_log = timer.maybe_record(step + 1, loss_val)
        if timing_log and cfg.wandb.mode != "disabled":
            wandb.log(timing_log, step=step + 1)

        if weighting_scheme != "none" and (step + 1) % update_weights_every == 0:
            state = pde.update_weights(state, batch_sharded, pmap_keys)

        if step % log_every == 0 or step == max_steps - 1:
            batch_eval = {k: v[0] for k, v in batch_sharded.items()}
            eval_state = unreplicate(state)
            log_dict = evaluator(eval_state, batch_eval, pmap_keys[0], step=step)

            avg_loss = float(jnp.mean(loss_val))
            log_dict["train/loss"] = avg_loss
            log_dict["step"] = step

            terms = " ".join(
                f"{k}={v:.3e}" for k, v in log_dict.items() if k.endswith("_loss")
            )
            print(f"  Step {step:>6d} | Loss {avg_loss:.4e} | {terms}")

            if cfg.wandb.mode != "disabled":
                wandb.log(log_dict, step=step)

        if (step + 1) % save_every == 0:
            save_checkpoint(unreplicate(state), workdir, keep=keep_ckpts)

    timer.finish(max_steps)
    save_checkpoint(unreplicate(state), workdir, keep=keep_ckpts, name="final")
    print("[*] Training finished. Model saved.")


if __name__ == "__main__":
    main()
