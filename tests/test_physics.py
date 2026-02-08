import os
import sys
import pytest
import numpy as np
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.physics.fractional import mitlef, roots_jacobi
from src.physics.dw_eg1 import *
from src.vis.plotter import PlotlyPlotter, PltPlotter
from libs.pde_burgers import DWBurgers



def test_fraction_operator_with_plot():
    alpha = 1.50
    work_dirs = "tests/test_fractional-0206"
    os.makedirs(work_dirs, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_eval = np.linspace(1e-3, 1.0, 50)
    x_eval = np.linspace(0.0, 1.0, 50)
    T, X = np.meshgrid(t_eval, x_eval)
    points_np = np.stack([T.flatten(), X.flatten()], axis=1)
    points = torch.from_numpy(points_np).to(device=device, dtype=torch.float64)
    methods = ["MC-I", "MC-II", "GJ-I", "GJ-II"]
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
        # rel_err = torch.linalg.norm(dt_alpha - dt_alpha_true) / torch.linalg.norm(dt_alpha_true)
        # if method in ["MC-I", "MC-II", "GJ-I", "GJ-II"]:
        #     assert rel_err.item() < 0.6
        # else:
        #     assert rel_err.item() < 0.4
        dt_alpha_np = dt_alpha.detach().cpu().numpy().reshape(T.shape)
        dt_alpha_true_np = dt_alpha_true.detach().cpu().numpy().reshape(T.shape)
        plotter.plot_solution(t=T, x=X, values=dt_alpha_np, title=f"{method} dt_alpha", name=f"dt_alpha_{method}")
        plotter.plot_solution(t=T, x=X, values=dt_alpha_true_np, title=f"{method} dt_alpha_true", name=f"dt_alpha_true_{method}")
        abs_err = np.abs(dt_alpha_true_np-dt_alpha_np)
        l2_err = np.linalg.norm(abs_err) / np.linalg.norm(dt_alpha_true_np)
        plotter.plot_solution(t=T, x=X, values=abs_err, title=f"L2 Error is {l2_err:.4f}", name=f"abs_err_{method}")

def test_burgers_residual_shapes():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimpleNamespace(
        al=1.5,
        datafile="",
        beta=2.0,
        xlim=[[0, 1]],
        tlim=[0, 1],
        method="MC-I",
        MC=SimpleNamespace(nums=5, eps=1e-6),
        GJ=SimpleNamespace(nums=5),
        dev=device
    )
    pde = DWBurgers(cfg)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1)
    ).to(device=device)
    n_in, n_bd, n_init = 6, 4, 5
    points_in = torch.rand(n_in, 2, device=device)
    points_bd = torch.rand(n_bd, 2, device=device)
    points_init = torch.rand(n_init, 2, device=device)
    points_init[:, 0] = 0.0
    points_all = {"in": points_in, "bd": points_bd, "init": points_init}
    losses = pde.residual(net, points_all)
    assert set(losses.keys()) == {"in", "bd", "init", "init_dt"}
    assert losses["in"].shape == (n_in, 1)
    assert losses["bd"].shape == (n_bd, 1)
    assert losses["init"].shape == (n_init, 1)
    assert losses["init_dt"].shape == (n_init, 1)
    for val in losses.values():
        assert torch.isfinite(val).all()
    
