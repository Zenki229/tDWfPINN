import os
import random
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Random seed set to {seed}")

def setup_wandb(cfg: DictConfig, model: torch.nn.Module = None) -> wandb.run:
    """
    Initialize Weights & Biases logging.

    Args:
        cfg (DictConfig): The full Hydra configuration.
        model (torch.nn.Module, optional): Model to watch.
    """
    name = cfg.wandb.get("name") if hasattr(cfg.wandb, "get") else None
    tags_cfg = cfg.wandb.get("tags") if hasattr(cfg.wandb, "get") else None
    if tags_cfg is None:
        tags = None
    else:
        tags = [str(t) for t in OmegaConf.to_container(tags_cfg, resolve=True)]

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        name=name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        dir=os.getcwd(),
    )
    wandb.define_metric("train/loss_event")
    wandb.define_metric("train/loss_continuous", step_metric="train/loss_event")
    wandb.define_metric("train/adam_loss", step_metric="train/loss_event")
    wandb.define_metric("train/lbfgs_loss", step_metric="train/loss_event")
    wandb.define_metric("train/adam_step")
    wandb.define_metric("loss_*", step_metric="train/adam_step")
    wandb.define_metric("eval/*", step_metric="train/adam_step")
    wandb.define_metric("L2_Relative_Error", step_metric="train/adam_step")
    wandb.define_metric("timing/*", step_metric="train/adam_step")
    log.info(f"WandB initialized for project {cfg.wandb.project}")
    return run 
