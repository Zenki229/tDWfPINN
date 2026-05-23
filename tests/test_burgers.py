from pathlib import Path
import sys

import hydra
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.sampler import TimeSpaceSampler
from src.physics.burgers import TimeFracBurgers1D


def _compose_burgers(method: str):
    return compose(
        config_name="config",
        overrides=[
            "pde=burgers",
            f"pde.method={method}",
            "pde.alpha=1.5",
            "pde.datafile=data/burgers_150.npz",
            "model.hidden_dim=8",
            "model.num_layers=1",
            "trainer.batch_size.domain=4",
            "trainer.batch_size.boundary=2",
            "trainer.batch_size.initial=2",
            "pde.gauss_jacobi_params.nums=3",
            "pde.gj_params.nums=3",
            "pde.monte_carlo_params.nums=3",
        ],
    )


def test_burgers_residual_backward_and_exact():
    methods = ["GJ-I", "GJ-II", "MC-I", "MC-II"]
    device = torch.device("cpu")

    with initialize(version_base=None, config_path="../conf"):
        for method in methods:
            cfg = _compose_burgers(method)
            cfg.work_dir = str(Path.cwd())
            cfg.model.input_dim = int(cfg.pde.input_dim)
            model = hydra.utils.instantiate(cfg.model).to(device)
            pde = TimeFracBurgers1D(cfg, device)
            batch_size = OmegaConf.to_container(cfg.trainer.batch_size, resolve=True)
            sampler = TimeSpaceSampler(
                [list(cfg.pde.x_lim)],
                list(cfg.pde.t_lim),
                device,
                batch_size,
            )

            points = sampler.sample()
            residuals = pde.residual(model, points)
            loss = sum(torch.square(value).mean() for value in residuals.values())
            assert torch.isfinite(loss)
            loss.backward()

            exact = pde.exact(points["domain"])
            assert exact.shape == (points["domain"].shape[0], 1)
            assert torch.isfinite(exact).all()
