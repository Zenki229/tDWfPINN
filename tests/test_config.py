from hydra import initialize, compose


def test_config_loading():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
        assert cfg.model.input_dim == 2
        assert cfg.pde.name == "forward"
        assert cfg.pde.al == 1.75
        assert cfg.training.learning_rate > 0
