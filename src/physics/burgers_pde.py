import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from omegaconf import DictConfig
from src.physics.pde import TimeFracCaputoDiffusionWaveTwoDimPDE
from src.utils.typing import Tensor, Dict


class BurgersPDE(TimeFracCaputoDiffusionWaveTwoDimPDE):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device, cfg.pde.alpha)
        self.beta = cfg.pde.beta
        self.nu = getattr(cfg.pde, "nu", 0.01 / np.pi)
        self.datafile = getattr(cfg.pde, "datafile", None)
        self.x_lim = cfg.pde.x_lim
        self.t_lim = cfg.pde.t_lim
        self._exact_cache = None
        self._exact_interpolator = None

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
            create_graph=True
        )[0]
        dt = grads[:, 0:1]
        dx = grads[:, 1:2]
        dxx = torch.autograd.grad(
            inputs=points_in,
            outputs=dx,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True
        )[0][:, 1:2]
        points_in.detach()
        points_in.requires_grad = False

        if self.method == "GJ-II":
            dt_alpha = self._gj_ii(net, points_in, val, dt)
        elif self.method == "MC-I":
            dt_alpha = self._mc_i(net, points_in, dt)
        elif self.method == "MC-II":
            dt_alpha = self._mc_ii(net, points_in, val, dt)
        elif self.method == "GJ-I":
            dt_alpha = self._gj_i(net, points_in, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        losses["domain"] = dt_alpha + val * dx - self.nu * dxx

        points_bd = points_all["boundary"]
        losses["boundary"] = self.u_net(net, points_bd)

        points_init = points_all["initial"]
        x = points_init[:, 1:2]
        target_init = -torch.sin(np.pi * x)
        pred_init = self.u_net(net, points_init)
        losses["initial"] = pred_init - target_init

        points_init.requires_grad = True
        val_init = self.u_net(net, points_init)
        dt_init = torch.autograd.grad(
            outputs=val_init,
            inputs=points_init,
            grad_outputs=torch.ones_like(val_init),
            retain_graph=True,
            create_graph=True
        )[0][:, 0:1]
        target_dt_init = self.beta * torch.sin(np.pi * x)
        losses["initial_dt"] = dt_init - target_dt_init

        return losses

    def _load_exact(self):
        if self.datafile is None:
            raise ValueError("pde.datafile is required for exact solution.")
        data = np.load(self.datafile)
        t = data["t"]
        x = data["x"]
        u = data["u"]
        if u.shape == (len(x), len(t)):
            u = u.T
        self._exact_cache = (t, x, u)
        self._exact_interpolator = RegularGridInterpolator((t, x), u, bounds_error=False, fill_value=None)

    def exact(self, points: Tensor) -> Tensor:
        if self.datafile is None:
            raise ValueError("pde.datafile is required for exact solution.")
        data = np.load(self.datafile)
        t = data["t"]
        x = data["x"]
        u = data["u"]
        if u.shape == (len(x), len(t)):
            u = u.T 
        # institute points with t,x 
        self.exact_points = np.concatenate((t.reshape(1,-1), x.reshape(1,-1)), axis=0).T
        return torch.from_numpy(u).to(self.device, dtype=points.dtype).reshape(-1, 1)

if __name__ == "__main__":
    cfg = OmegaConf.create({
        "pde": {
            "alpha": 1.50,
            "method": "MC-I",
            "datafile": "data/burgers/burgers_150.npz",
        },
    })
    device= "cuda:0" if torch.cuda.is_available() else "cpu"
    pde = BurgersPDE(cfg, torch.device(device))
    points = torch.rand(100, 2)
    u = pde.exact(points)
    print(u.shape)
