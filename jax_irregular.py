import os

import hydra
import jax
import jax.numpy as jnp
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from libs.jax_evaluator import BaseEvaluator
from libs.jax_pde_irregular import JAXIrregularHoleDW, JAXLShapeDW
from libs.jax_pinn import create_train_state
from libs.jax_sample import IrregularHoleSampler, LShapeSampler
from libs.jax_utils import (
    EpochTimer,
    replicate,
    restore_checkpoint,
    save_checkpoint,
    unreplicate,
)


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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg.model.input_dim = 3

    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    n_devices = jax.local_device_count()
    print(f"[*] Running on {n_devices} device(s): {jax.devices()}")

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
        batch_sharded = next(sampler)
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
