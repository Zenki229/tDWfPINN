from pathlib import Path
import sys

import hydra
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.sampler import IrregularHoleSampler, LShapeSampler
from src.physics.irregular_2d import IrregularHole2D, LShape2D


def _compose_case(case_name: str, method: str):
    return compose(
        config_name="config",
        overrides=[
            f"pde={case_name}",
            f"pde.method={method}",
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


def test_irregular_2d_cases_residual_backward():
    methods = ["GJ-I", "GJ-II", "MC-I", "MC-II"]
    device = torch.device("cpu")
    cases = {
        "irregular_hole": (IrregularHole2D, IrregularHoleSampler),
        "lshape": (LShape2D, LShapeSampler),
    }

    with initialize(version_base=None, config_path="../conf"):
        for case_name, (pde_cls, sampler_cls) in cases.items():
            for method in methods:
                cfg = _compose_case(case_name, method)
                cfg.work_dir = str(Path.cwd())
                cfg.model.input_dim = int(cfg.pde.input_dim)
                model = hydra.utils.instantiate(cfg.model).to(device)
                pde = pde_cls(cfg, device)
                batch_size = OmegaConf.to_container(cfg.trainer.batch_size, resolve=True)
                if case_name == "irregular_hole":
                    sampler = sampler_cls(cfg.pde.t_lim, batch_size, device, cfg.pde.center, cfg.pde.r0)
                else:
                    sampler = sampler_cls(cfg.pde.t_lim, batch_size, device)

                residuals = pde.residual(model, sampler.sample())
                loss = sum(torch.square(value).mean() for value in residuals.values())
                assert torch.isfinite(loss)
                loss.backward()
