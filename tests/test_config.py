import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf

def test_config_loading():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
        assert cfg.model.input_dim == 2
        assert cfg.pde.alpha == 1.75
        assert cfg.optimizer.lr > 0
        assert cfg.trainer.steps_per_epoch > 0
        assert cfg.trainer.loss_log_every_steps == 100
        assert cfg.trainer.eval_every_epochs == 1
