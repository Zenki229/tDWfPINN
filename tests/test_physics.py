import os
import sys
import pytest
import numpy as np
import torch
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.physics.fractional import mitlef, roots_jacobi
from src.physics.dw_eg1 import *
from src.vis.plotter import PlotlyPlotter, PltPlotter



def test_fraction_operator_with_plot():
    alpha = 1.50
    work_dirs = "tests/test_fractional"
    os.makedirs(work_dirs, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_eval = np.linspace(1e-3, 1.0, 50)
    x_eval = np.linspace(0.0, 1.0, 50)
    T, X = np.meshgrid(t_eval, x_eval)
    points_np = np.stack([T.flatten(), X.flatten()], axis=1)
    points = torch.from_numpy(points_np).to(device=device, dtype=torch.float64)
    methods = ["MC-I"]
    for idx, method in enumerate(methods):
        cfg = OmegaConf.create({
            "pde": {
                "alpha": alpha,
                "method": method,
                "monte_carlo_params": {"nums": 80, "eps": 1e-10},
                "gj_params": {"nums": 120}
            },
            "plot": {
                "backend": "matplotlib",
                "font_size": 14,
                "jpg": True
            }
        })
        plotter = PlotlyPlotter(work_dirs, cfg) if cfg.plot.backend == "plotly" else PltPlotter(work_dirs, cfg)
        pde = DWForwardEg1(cfg, device)
        dt_alpha = pde.frac_diff_exact(points)
        u_val = pde.exact(points)
        dt_alpha_true = -pde.lam * u_val
        rel_err = torch.linalg.norm(dt_alpha - dt_alpha_true) / torch.linalg.norm(dt_alpha_true)
        if method in ["MC-I", "MC-II"]:
            assert rel_err.item() < 0.6
        else:
            assert rel_err.item() < 0.4
        dt_alpha_np = dt_alpha.detach().cpu().numpy().reshape(T.shape)
        dt_alpha_true_np = dt_alpha_true.detach().cpu().numpy().reshape(T.shape)
        plotter.plot_solution(T, X, dt_alpha_np, f"{method} dt_alpha", f"dt_alpha_{method}")
    
