"""Generalized Gauss--Jacobi Type-I/Type-II operators.

This module implements the Gauss--Jacobi Type-I and Type-II transformed
Caputo operators for arbitrary input ``d = 1 + d_x``.
"""
from __future__ import annotations

import scipy.special as sp
import torch
from torch import nn


class GeneralizedGJOperator:
    """Gauss--Jacobi Type-I and Type-II transformed Caputo operators."""

    def __init__(self, alpha: float, M: int, device: torch.device, dtype: torch.dtype):
        self.alpha = float(alpha)
        self.M = int(M)
        self.device = device
        self.dtype = dtype
        quad_t_np, quad_w_np = sp.roots_jacobi(self.M, 0.0, 1.0 - self.alpha)
        quad_t = (quad_t_np + 1.0) / 2.0
        quad_w = quad_w_np * (1.0 / 2.0) ** (2.0 - self.alpha)
        self.quad_t = torch.tensor(quad_t, device=device, dtype=dtype)
        self.quad_w = torch.tensor(quad_w, device=device, dtype=dtype)

    def u_net(self, net: nn.Module, points: torch.Tensor) -> torch.Tensor:
        return net(points)

    def _time_grad(self, net: nn.Module, points: torch.Tensor) -> torch.Tensor:
        pts = points.detach().clone().requires_grad_(True)
        val = self.u_net(net, pts)
        grads = torch.autograd.grad(
            outputs=val,
            inputs=pts,
            grad_outputs=torch.ones_like(val),
            retain_graph=True,
            create_graph=True,
        )[0]
        return grads[..., 0:1]

    def _quad_points(self, points: torch.Tensor, taus: torch.Tensor):
        t = points[:, 0:1]
        coords = points[:, 1:]
        t_tau = taus.reshape(-1, 1) @ t.reshape(1, -1)
        t_eval = t.reshape(1, -1) - t_tau
        coords_eval = coords.unsqueeze(0).expand(taus.numel(), points.shape[0], coords.shape[1])
        new_points = torch.cat([t_eval.unsqueeze(-1), coords_eval], dim=2)
        return new_points, t_tau

    def _quad_dt(self, net: nn.Module, new_points: torch.Tensor) -> torch.Tensor:
        new_points = new_points.detach().clone().requires_grad_(True)
        val_tau = self.u_net(net, new_points)
        dt_tau = torch.autograd.grad(
            outputs=val_tau,
            inputs=new_points,
            grad_outputs=torch.ones_like(val_tau),
            retain_graph=True,
            create_graph=True,
        )[0][..., 0:1]
        return dt_tau

    def _dt0(self, net: nn.Module, points: torch.Tensor) -> torch.Tensor:
        t0 = torch.cat([torch.zeros_like(points[:, 0:1]), points[:, 1:]], dim=1)
        return self._time_grad(net, t0)

    def _val0(self, net: nn.Module, points: torch.Tensor) -> torch.Tensor:
        t0 = torch.cat([torch.zeros_like(points[:, 0:1]), points[:, 1:]], dim=1)
        return self.u_net(net, t0)

    def _gj_i(self, net: nn.Module, points: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        coeff = sp.gamma(2.0 - self.alpha)
        taus = self.quad_t
        quad_w = self.quad_w.reshape(-1, 1, 1)
        new_points, t_tau = self._quad_points(points, taus)
        dt_tau = self._quad_dt(net, new_points)
        dt0 = self._dt0(net, points)
        integral = torch.sum(quad_w * (dt.unsqueeze(0) - dt_tau) / t_tau.unsqueeze(-1), dim=0)
        t = points[:, 0:1]
        part1 = (self.alpha - 1.0) * t ** (2.0 - self.alpha) * integral
        part2 = (dt - dt0) * t ** (1.0 - self.alpha)
        return (part1 + part2) / coeff

    def _gj_ii(self, net: nn.Module, points: torch.Tensor, val: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        coeff = sp.gamma(2.0 - self.alpha)
        taus = self.quad_t
        quad_w = self.quad_w.reshape(-1, 1, 1)
        new_points, t_tau = self._quad_points(points, taus)
        val2 = self.u_net(net, new_points)
        val0 = self._val0(net, points)
        dt0 = self._dt0(net, points)
        val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0)
        integral = torch.sum(
            quad_w * (val.unsqueeze(0) - val2 - val3) / (t_tau ** 2).unsqueeze(-1),
            dim=0,
        )
        t = points[:, 0:1]
        part1 = self.alpha * (self.alpha - 1.0) * t ** (2.0 - self.alpha) * integral
        part2 = (self.alpha - 1.0) * (val - val0 - t * dt) / t ** self.alpha
        part3 = (dt - dt0) / t ** (self.alpha - 1.0)
        return (part3 - part2 - part1) / coeff
