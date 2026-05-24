import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.special as sp
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.physics.fractional import roots_jacobi
from src.physics.pde import PDE
from src.utils.typing import Tensor


def _cfg_value(cfg, name, default=None):
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def _quadrature_nums(cfg):
    if hasattr(cfg.pde, "gauss_jacobi_params"):
        return cfg.pde.gauss_jacobi_params.nums
    if hasattr(cfg.pde, "gj_params"):
        return cfg.pde.gj_params.nums
    return 32


class Irregular2DBase(PDE):
    """Base utilities for two-dimensional time-fractional diffusion-wave cases."""

    spatial_dim = 2

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.alpha = float(cfg.pde.alpha)
        self.method = str(cfg.pde.method)
        self.t_lim = list(cfg.pde.t_lim)

        if "GJ" in self.method:
            nums = int(_quadrature_nums(cfg))
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.alpha)
            self.quad_t = (quad_t + 1) / 2
            self.quad_w = quad_wt * (1 / 2) ** (2 - self.alpha)

    def u_net(self, net: nn.Module, points: Tensor) -> Tensor:
        return net(points)

    def _time_grad(self, net: nn.Module, points: Tensor) -> Tensor:
        pts = points.detach().clone().requires_grad_(True)
        val = self.u_net(net, pts)
        grads = torch.autograd.grad(
            outputs=val,
            inputs=pts,
            grad_outputs=torch.ones_like(val),
            retain_graph=True,
            create_graph=True,
        )[0]
        return grads[:, 0:1]

    def _value_time_grad_laplacian(
        self,
        net: nn.Module,
        points: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pts = points.detach().clone().requires_grad_(True)
        val = self.u_net(net, pts)
        grads = torch.autograd.grad(
            outputs=val,
            inputs=pts,
            grad_outputs=torch.ones_like(val),
            retain_graph=True,
            create_graph=True,
        )[0]
        dt = grads[:, 0:1]
        ux = grads[:, 1:2]
        uy = grads[:, 2:3]
        uxx = torch.autograd.grad(
            outputs=ux,
            inputs=pts,
            grad_outputs=torch.ones_like(ux),
            retain_graph=True,
            create_graph=True,
        )[0][:, 1:2]
        uyy = torch.autograd.grad(
            outputs=uy,
            inputs=pts,
            grad_outputs=torch.ones_like(uy),
            retain_graph=True,
            create_graph=True,
        )[0][:, 2:3]
        return val, dt, ux, uy, uxx + uyy

    def _quad_points(self, points: Tensor, taus: Tensor) -> Tuple[Tensor, Tensor]:
        t = points[:, 0:1]
        coords = points[:, 1:3]
        t_tau = taus.reshape(-1, 1) @ t.reshape(1, -1)
        t_eval = t.reshape(1, -1) - t_tau
        coords_eval = coords.unsqueeze(0).expand(taus.numel(), points.shape[0], 2)
        new_points = torch.cat([t_eval.unsqueeze(-1), coords_eval], dim=2)
        return new_points, t_tau

    def _quad_dt(self, net: nn.Module, new_points: Tensor) -> Tensor:
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

    def frac_diff(self, net: nn.Module, points: Tensor, val: Tensor, dt: Tensor) -> Tensor:
        if self.method == "GJ-I":
            return self._gj_i(net, points, dt)
        if self.method == "GJ-II":
            return self._gj_ii(net, points, val, dt)
        if self.method == "MC-I":
            return self._mc_i(net, points, dt)
        if self.method == "MC-II":
            return self._mc_ii(net, points, val, dt)
        raise ValueError(f"Unknown method: {self.method}")

    def _dt0(self, net: nn.Module, points: Tensor) -> Tensor:
        t0 = torch.cat([torch.zeros_like(points[:, 0:1]), points[:, 1:3]], dim=1)
        return self._time_grad(net, t0)

    def _val0(self, net: nn.Module, points: Tensor) -> Tensor:
        t0 = torch.cat([torch.zeros_like(points[:, 0:1]), points[:, 1:3]], dim=1)
        return self.u_net(net, t0)

    def _mc_taus(self) -> Tensor:
        nums = int(self.cfg.pde.monte_carlo_params.nums)
        eps = float(self.cfg.pde.monte_carlo_params.eps)
        dist = torch.distributions.Beta(
            torch.tensor(2 - self.alpha, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        taus = dist.sample((nums,))
        return eps + (1 - 2 * eps) * taus

    def _mc_i(self, net: nn.Module, points: Tensor, dt: Tensor) -> Tensor:
        eps = float(self.cfg.pde.monte_carlo_params.eps)
        coeff = sp.gamma(2 - self.alpha)
        taus = self._mc_taus()
        new_points, t_tau = self._quad_points(points, taus)
        dt_tau = self._quad_dt(net, new_points)
        dt0 = self._dt0(net, points)
        den = torch.clamp(t_tau, min=eps).unsqueeze(-1)
        integral = torch.mean((dt.unsqueeze(0) - dt_tau) / den, dim=0)
        t = points[:, 0:1]
        part1 = ((self.alpha - 1) / (2 - self.alpha)) * t ** (2 - self.alpha) * integral
        part2 = (dt - dt0) * t ** (1 - self.alpha)
        return (part1 + part2) / coeff

    def _mc_ii(self, net: nn.Module, points: Tensor, val: Tensor, dt: Tensor) -> Tensor:
        eps = float(self.cfg.pde.monte_carlo_params.eps)
        coeff = sp.gamma(2 - self.alpha)
        taus = self._mc_taus()
        new_points, t_tau = self._quad_points(points, taus)
        val2 = self.u_net(net, new_points)
        val0 = self._val0(net, points)
        dt0 = self._dt0(net, points)
        den = torch.clamp(t_tau ** 2, min=eps).unsqueeze(-1)
        val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0)
        integral = torch.mean((val.unsqueeze(0) - val2 - val3) / den, dim=0)
        t = points[:, 0:1]
        part1 = self.alpha * (self.alpha - 1) / (2 - self.alpha) * t ** (2 - self.alpha) * integral
        part2 = (self.alpha - 1) * (val - val0 - t * dt) / t ** self.alpha
        part3 = (dt - dt0) / t ** (self.alpha - 1)
        return (part3 - part2 - part1) / coeff

    def _gj_i(self, net: nn.Module, points: Tensor, dt: Tensor) -> Tensor:
        coeff = sp.gamma(2 - self.alpha)
        taus = torch.tensor(self.quad_t.tolist(), device=self.device)
        quad_w = torch.tensor(self.quad_w.tolist(), device=self.device).reshape(-1, 1, 1)
        new_points, t_tau = self._quad_points(points, taus)
        dt_tau = self._quad_dt(net, new_points)
        dt0 = self._dt0(net, points)
        integral = torch.sum(quad_w * (dt.unsqueeze(0) - dt_tau) / t_tau.unsqueeze(-1), dim=0)
        t = points[:, 0:1]
        part1 = (self.alpha - 1) * t ** (2 - self.alpha) * integral
        part2 = (dt - dt0) * t ** (1 - self.alpha)
        return (part1 + part2) / coeff

    def _gj_ii(self, net: nn.Module, points: Tensor, val: Tensor, dt: Tensor) -> Tensor:
        coeff = sp.gamma(2 - self.alpha)
        taus = torch.tensor(self.quad_t.tolist(), device=self.device)
        quad_w = torch.tensor(self.quad_w.tolist(), device=self.device).reshape(-1, 1, 1)
        new_points, t_tau = self._quad_points(points, taus)
        val2 = self.u_net(net, new_points)
        val0 = self._val0(net, points)
        dt0 = self._dt0(net, points)
        val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0)
        integral = torch.sum(quad_w * (val.unsqueeze(0) - val2 - val3) / (t_tau ** 2).unsqueeze(-1), dim=0)
        t = points[:, 0:1]
        part1 = self.alpha * (self.alpha - 1) * t ** (2 - self.alpha) * integral
        part2 = (self.alpha - 1) * (val - val0 - t * dt) / t ** self.alpha
        part3 = (dt - dt0) / t ** (self.alpha - 1)
        return (part3 - part2 - part1) / coeff

    def plot_time_slices(self) -> List[float]:
        if hasattr(self.cfg.pde, "plot_time_slices"):
            return [float(v) for v in self.cfg.pde.plot_time_slices]
        return [0.5 * float(self.t_lim[1]), float(self.t_lim[1])]

    def domain_mask_np(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IrregularHole2D(Irregular2DBase):
    """Circular-hole manufactured two-dimensional diffusion-wave benchmark."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.center = tuple(float(v) for v in cfg.pde.center)
        self.r0 = float(cfg.pde.r0)
        self.lam = float(_cfg_value(cfg.pde, "lam", 1.0))
        self.diffusion_amp = float(_cfg_value(cfg.pde, "diffusion_amp", 0.3))

    def q(self, t: Tensor) -> Tensor:
        return t ** 2 * (1 - t) ** 2

    def dtalpha_q(self, t: Tensor) -> Tensor:
        safe_t = torch.clamp(t, min=1e-10)
        return (
            2.0 / sp.gamma(3 - self.alpha) * safe_t ** (2 - self.alpha)
            - 12.0 / sp.gamma(4 - self.alpha) * safe_t ** (3 - self.alpha)
            + 24.0 / sp.gamma(5 - self.alpha) * safe_t ** (4 - self.alpha)
        )

    def phi_terms(self, x: Tensor, y: Tensor):
        cx, cy = self.center
        psi = (1 - x ** 2) * (1 - y ** 2)
        chi = (x - cx) ** 2 + (y - cy) ** 2 - self.r0 ** 2
        phi = psi * chi
        psi_x = -2 * x * (1 - y ** 2)
        psi_y = -2 * y * (1 - x ** 2)
        lap_psi = 2 * x ** 2 + 2 * y ** 2 - 4
        chi_x = 2 * (x - cx)
        chi_y = 2 * (y - cy)
        lap_chi = 4.0
        phi_x = chi * psi_x + psi * chi_x
        phi_y = chi * psi_y + psi * chi_y
        lap_phi = chi * lap_psi + 2 * (psi_x * chi_x + psi_y * chi_y) + psi * lap_chi
        return phi, phi_x, phi_y, lap_phi

    def diffusion_coeff(self, x: Tensor, y: Tensor) -> Tensor:
        return 1.0 + self.diffusion_amp * torch.sin(math.pi * x) * torch.cos(math.pi * y)

    def grad_diffusion_coeff(self, x: Tensor, y: Tensor):
        ax = self.diffusion_amp * math.pi * torch.cos(math.pi * x) * torch.cos(math.pi * y)
        ay = -self.diffusion_amp * math.pi * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        return ax, ay

    def source(self, points: Tensor) -> Tensor:
        t = points[:, 0:1]
        x = points[:, 1:2]
        y = points[:, 2:3]
        phi, phi_x, phi_y, lap_phi = self.phi_terms(x, y)
        q = self.q(t)
        dtalpha = self.dtalpha_q(t)
        a = self.diffusion_coeff(x, y)
        ax, ay = self.grad_diffusion_coeff(x, y)
        bx = 1.0 + y
        by = x - 1.0
        return (
            dtalpha * phi
            - q * (ax * phi_x + ay * phi_y + a * lap_phi)
            + q * (bx * phi_x + by * phi_y)
            + self.lam * q ** 3 * phi ** 3
        )

    def residual(self, net: nn.Module, points_all: Dict[str, Tensor]) -> Dict[str, Tensor]:
        losses = {}
        points_in = points_all["domain"]
        val, dt, ux, uy, lap = self._value_time_grad_laplacian(net, points_in)
        dt_alpha = self.frac_diff(net, points_in, val, dt)
        x = points_in[:, 1:2]
        y = points_in[:, 2:3]
        a = self.diffusion_coeff(x, y)
        ax, ay = self.grad_diffusion_coeff(x, y)
        bx = 1.0 + y
        by = x - 1.0
        div_term = ax * ux + ay * uy + a * lap
        convection = bx * ux + by * uy
        losses["domain"] = dt_alpha - div_term + convection + self.lam * val ** 3 - self.source(points_in)
        losses["boundary"] = self.u_net(net, points_all["boundary"])
        points_init = points_all["initial"]
        losses["initial"] = self.u_net(net, points_init)
        losses["initial_dt"] = self._time_grad(net, points_init)
        return losses

    def exact(self, points: Tensor) -> Tensor:
        pts = points.to(self.device)
        t = pts[:, 0:1]
        x = pts[:, 1:2]
        y = pts[:, 2:3]
        phi, _, _, _ = self.phi_terms(x, y)
        return self.q(t) * phi

    def domain_mask_np(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        cx, cy = self.center
        return ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) >= self.r0 ** 2


class LShape2D(Irregular2DBase):
    """L-shaped constant-coefficient reference-comparison benchmark."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.diffusion_scale = float(_cfg_value(cfg.pde, "diffusion_scale", 0.25))
        self.velocity_scale = float(_cfg_value(cfg.pde, "velocity_scale", 0.2))
        self.reference_data = Path(str(_cfg_value(cfg.pde, "reference_data", "")))
        if self.reference_data and not self.reference_data.is_absolute():
            root = Path(str(_cfg_value(cfg, "work_dir", Path.cwd())))
            self.reference_data = root / self.reference_data
        self._reference = None
        if self.reference_data and self.reference_data.exists():
            self._reference = np.load(self.reference_data)

    def initial_profile(self, x: Tensor, y: Tensor) -> Tensor:
        return (
            torch.exp(-((x + 0.55) ** 2 + (y + 0.45) ** 2) / 0.08)
            - 0.85 * torch.exp(-((x + 0.55) ** 2 + (y - 0.45) ** 2) / 0.06)
            + 0.60 * torch.exp(-((x - 0.45) ** 2 + (y + 0.55) ** 2) / 0.06)
        )

    def residual(self, net: nn.Module, points_all: Dict[str, Tensor]) -> Dict[str, Tensor]:
        losses = {}
        points_in = points_all["domain"]
        val, dt, _, _, lap = self._value_time_grad_laplacian(net, points_in)
        dt_alpha = self.frac_diff(net, points_in, val, dt)
        losses["domain"] = dt_alpha - self.diffusion_scale * lap
        losses["boundary"] = self.u_net(net, points_all["boundary"])
        points_init = points_all["initial"]
        x0 = points_init[:, 1:2]
        y0 = points_init[:, 2:3]
        g = self.initial_profile(x0, y0)
        losses["initial"] = self.u_net(net, points_init) - g
        losses["initial_dt"] = self._time_grad(net, points_init) - self.velocity_scale * g
        return losses

    def exact(self, points: Tensor) -> Tensor:
        if self._reference is None:
            raise FileNotFoundError(
                f"L-shape reference file not found: {self.reference_data}. "
                "Generate or restore data/lshape/lshape_reference.npz first."
            )
        pts = points.detach().cpu().numpy()
        times = self._reference["times"]
        xs = self._reference["x"]
        ys = self._reference["y"]
        snapshots = self._reference["snapshots"]
        ti = np.abs(times[:, None] - pts[:, 0][None, :]).argmin(axis=0)
        xi = np.abs(xs[:, None] - pts[:, 1][None, :]).argmin(axis=0)
        yi = np.abs(ys[:, None] - pts[:, 2][None, :]).argmin(axis=0)
        vals = snapshots[ti, yi, xi].reshape(-1, 1)
        return torch.tensor(vals.tolist(), device=self.device)

    def reference_slices(self, time_values: Iterable[float]):
        if self._reference is None:
            raise FileNotFoundError(
                f"L-shape reference file not found: {self.reference_data}. "
                "Generate or restore data/lshape/lshape_reference.npz first."
            )
        times = self._reference["times"]
        selected = []
        for value in time_values:
            idx = int(np.argmin(np.abs(times - float(value))))
            selected.append((float(times[idx]), self._reference["snapshots"][idx]))
        return self._reference["x_grid"], self._reference["y_grid"], selected

    def domain_mask_np(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        return ((x_grid >= -1.0) & (x_grid <= 1.0) & (y_grid >= -1.0) & (y_grid <= 1.0)
                & ((x_grid <= 0.0) | (y_grid <= 0.0)))
