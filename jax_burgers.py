import os
import hydra
import wandb
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state
from omegaconf import DictConfig, OmegaConf
from libs.jax_pinn import create_model
from libs.jax_pde_burgers import JAXDWBurgers
from libs.jax_sample import TimeSpaceEasySampler
from functools import partial

class TrainState(train_state.TrainState):
    pass

def get_pde_loss(params, apply_fn, pde, batch, key, weights):
    # pde.residual returns dict of losses (vectors)
    # We need to compute scalar loss
    losses = pde.residual(apply_fn, params, batch, key)
    
    total_loss = 0.0
    loss_dict = {}
    
    for k, v in losses.items():
        # MSE loss
        l = jnp.mean(v ** 2)
        loss_dict[k] = l
        if k in weights:
            total_loss += weights[k] * l
            
    return total_loss, loss_dict

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5))
def train_step(state, batch, key, pde, weights, gradient_accumulation_steps=1):
    # Split key for MC sampling inside PDE
    key, subkey = jax.random.split(key)
    
    def loss_fn(params):
        loss, aux = get_pde_loss(params, state.apply_fn, pde, batch, subkey, weights)
        return loss, aux
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    
    # Sync gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Update state
    state = state.apply_gradients(grads=grads)
    return state, loss, aux, key

def shard(data, n_devices):
    # data is a dict of numpy arrays
    # We want to reshape each array from (B, ...) to (n_devices, B/n_devices, ...)
    sharded_data = {}
    for k, v in data.items():
        if v.shape[0] % n_devices != 0:
            # Pad or trim if not divisible (simplification: trim)
            new_size = (v.shape[0] // n_devices) * n_devices
            v = v[:new_size]
        
        B = v.shape[0]
        shape = (n_devices, B // n_devices) + v.shape[1:]
        sharded_data[k] = v.reshape(shape)
    return sharded_data

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Device setup
    n_devices = jax.local_device_count()
    print(f"Running on {n_devices} devices: {jax.devices()}")

    # Initialize model
    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    
    model, params = create_model(
        model_key, 
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
        num_layers=cfg.model.num_layers,
        activation=cfg.model.activation
    )

    # Optimizer
    optimizer = optax.adam(learning_rate=cfg.training.lr)
    
    # TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Replicate state for pmap
    state = jax.device_put_replicated(state, jax.local_devices())
    
    # Keys for pmap
    # keys = jax.random.split(key, n_devices)
    # We pass a single key to pmap and split it inside, or pass different keys?
    # Usually we pass different keys.
    pmap_keys = jax.random.split(key, n_devices)

    # PDE and Sampler
    pde = JAXDWBurgers(cfg.pde)
    
    # Adjust batch size for multi-device? 
    # Usually config batch size is global.
    sampler = TimeSpaceEasySampler(
        axeslim=cfg.pde.xlim, 
        tlim=cfg.pde.tlim, 
        batch=OmegaConf.to_container(cfg.training.batch)
    )

    # Weights
    weights = OmegaConf.to_container(cfg.pde.weighting)

    # Training Loop
    iterator = iter(sampler)
    
    for step in range(cfg.training.max_steps):
        # Sample data (CPU)
        batch_cpu = next(iterator)
        
        # Shard data
        batch_sharded = shard(batch_cpu, n_devices)
        
        # Train step
        state, loss, aux, pmap_keys = train_step(state, batch_sharded, pmap_keys, pde, weights)
        
        # Logging (take first device output)
        if step % 100 == 0 or step == cfg.training.max_steps - 1:
            # loss and aux are sharded, take mean or first
            avg_loss = jnp.mean(loss).item()
            
            log_dict = {"train/loss": avg_loss, "step": step}
            
            # Unpack aux losses
            # aux is a dict of sharded arrays
            for k, v in aux.items():
                log_dict[f"train/{k}"] = jnp.mean(v).item()
            
            print(f"Step {step}: Loss = {avg_loss:.4e}")
            if cfg.wandb.mode != "disabled":
                wandb.log(log_dict)

    # Save model (taking first device params)
    # state.params is replicated
    params_cpu = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
    
    # Save code here...
    
    print("Training finished.")

if __name__ == "__main__":
    main()
