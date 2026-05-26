"""Construct autograd losses for generalized Gauss--Jacobi Type-I and Type-II residuals.

All N, M, d, L, and H sweeps use the same generalized GJ PDE implementation in
``generalized_gj.py``.
"""
from __future__ import annotations

from typing import Any, Literal

import torch

from .config import BenchConfig, dtype_from_name
from .generalized_gj import GeneralizedGJOperator
from .models import NoBiasMLP
from .points import make_domain_points

MethodName = Literal["GJ-I", "GJ-II"]


def make_model_and_points(cfg: BenchConfig, device: torch.device):
    dtype = dtype_from_name(cfg.dtype)
    torch.manual_seed(cfg.seed)
    model = NoBiasMLP(input_dim=cfg.d, hidden_dim=cfg.H, num_hidden_layers=cfg.L).to(device=device, dtype=dtype)
    points = make_domain_points(cfg.N, cfg.d, device=device, dtype=dtype, seed=cfg.seed + 17)
    points = points.detach().clone().requires_grad_(False)
    return model, points


def make_operator(cfg: BenchConfig, device: torch.device) -> GeneralizedGJOperator:
    dtype = dtype_from_name(cfg.dtype)
    return GeneralizedGJOperator(alpha=cfg.alpha, M=cfg.M, device=device, dtype=dtype)


def prepare_gj_components(cfg: BenchConfig, device: torch.device):
    """Allocate static objects before graph construction.

    Storage measurements take their CUDA baseline after this function returns,
    so model parameters, collocation points, and quadrature nodes are not counted
    as retained graph storage.
    """
    model, points = make_model_and_points(cfg, device)
    operator = make_operator(cfg, device=device)
    return model, points, operator


def _value_and_time_derivative(model: torch.nn.Module, points: torch.Tensor):
    pts_grad = points.detach().clone().requires_grad_(True)
    val = model(pts_grad)
    grads = torch.autograd.grad(
        outputs=val,
        inputs=pts_grad,
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True,
    )[0]
    return val, grads[:, 0:1]


def build_gj_loss_from_components(
    cfg: BenchConfig,
    method: MethodName,
    model: torch.nn.Module,
    points: torch.Tensor,
    operator: Any,
):
    if points.requires_grad:
        raise RuntimeError("points must have requires_grad=False; generalized_gj.py creates grad-enabled copies internally")

    val, dt = _value_and_time_derivative(model, points)
    if method == "GJ-I":
        frac_dt = operator._gj_i(model, points, dt)
    elif method == "GJ-II":
        frac_dt = operator._gj_ii(model, points, val, dt)
    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.sum(frac_dt ** 2)


def build_gj_loss(cfg: BenchConfig, method: MethodName, device: torch.device):
    model, points, operator = prepare_gj_components(cfg, device=device)
    loss = build_gj_loss_from_components(cfg, method=method, model=model, points=points, operator=operator)
    return loss, model, points, operator
