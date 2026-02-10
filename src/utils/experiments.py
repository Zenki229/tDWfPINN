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
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        dir=os.getcwd(),
    )
    log.info(f"WandB initialized for project {cfg.wandb.project}")
    return run 
