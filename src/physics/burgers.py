import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from omegaconf import DictConfig
from pathlib import Path

from src.physics.pde import TimeFracCaputoDiffusionWaveTwoDimPDE
from src.utils.typing import Dict, Tensor


class TimeFracBurgers1D(TimeFracCaputoDiffusionWaveTwoDimPDE):
    """One-dimensional time-fractional Burgers benchmark.

    PDE:
        D_t^alpha u + u u_x - nu u_xx = 0

    Boundary and initial data match the stored reference arrays under
    data/burgers_*.npz.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device, cfg.pde.alpha)
        self.beta = float(cfg.pde.beta)
        self.nu = float(cfg.pde.nu)
        self.x_lim = list(cfg.pde.x_lim)
        self.t_lim = list(cfg.pde.t_lim)
        self.datafile = Path(str(cfg.pde.datafile))
        if self.datafile and not self.datafile.is_absolute():
            root = Path(str(getattr(cfg, "work_dir", Path.cwd())))
            self.datafile = root / self.datafile
        self._load_reference()

    def _load_reference(self):
        data = np.load(self.datafile)
        self.ref_t = np.asarray(data["t"])
        self.ref_x = np.asarray(data["x"])
        self.ref_u = np.asarray(data["u"])
        self._interp = RegularGridInterpolator(
            (self.ref_t, self.ref_x),
            self.ref_u,
            bounds_error=False,
            fill_value=None,
        )

    def _frac_dt(self, net: torch.nn.Module, points: Tensor, val: Tensor, dt: Tensor) -> Tensor:
        if self.method == "GJ-II":
            return self._gj_ii(net, points, val, dt)
        if self.method == "MC-I":
            return self._mc_i(net, points, dt)
        if self.method == "MC-II":
            return self._mc_ii(net, points, val, dt)
        if self.method == "GJ-I":
            return self._gj_i(net, points, dt)
        raise ValueError(f"Unknown method: {self.method}")

    def residual(self, net: torch.nn.Module, points_all: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}

        points_in = points_all["domain"]
        points_in.requires_grad = True
        val = self.u_net(net, points_in)
        grads = torch.autograd.grad(
            outputs=val,
            inputs=points_in,
            grad_outputs=torch.ones_like(val),
            retain_graph=True,
            create_graph=True,
        )[0]
        dt = grads[:, 0:1]
        dx = grads[:, 1:2]
        dxx = torch.autograd.grad(
            outputs=dx,
            inputs=points_in,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True,
        )[0][:, 1:2]
        points_in.detach()
        points_in.requires_grad = False
        dt_alpha = self._frac_dt(net, points_in, val, dt)
        losses["domain"] = dt_alpha + val * dx - self.nu * dxx

        points_bd = points_all["boundary"]
        losses["boundary"] = self.u_net(net, points_bd)

        points_init = points_all["initial"]
        x_init = points_init[:, 1:2]
        target_init = -torch.sin(torch.pi * x_init)
        losses["initial"] = self.u_net(net, points_init) - target_init

        points_init.requires_grad = True
        val_init = self.u_net(net, points_init)
        dt_init = torch.autograd.grad(
            outputs=val_init,
            inputs=points_init,
            grad_outputs=torch.ones_like(val_init),
            retain_graph=True,
            create_graph=True,
        )[0][:, 0:1]
        target_dt = self.beta * torch.sin(torch.pi * x_init)
        losses["initial_dt"] = dt_init - target_dt

        return losses

    def exact(self, points: Tensor) -> Tensor:
        points_np = points.detach().cpu().numpy()
        values = self._interp(points_np[:, :2]).reshape(-1, 1)
        return torch.tensor(values.tolist(), device=self.device)
