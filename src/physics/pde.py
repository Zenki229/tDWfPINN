import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.utils.typing import Tensor
import scipy.special as sp
from scipy.stats import beta
from omegaconf import DictConfig
from src.physics.fractional import roots_jacobi

class PDE(nn.Module):
    """
    Base class for PDE problems.
    """
    def __init__(self):
        super().__init__()

    def residual(self, net: nn.Module, points: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def exact(self, points: Tensor) -> Tensor:
        raise NotImplementedError
    
    def exact_dt(self, points:Tensor) -> Tensor: 
        """
            return the 1st derivative of exact solution with respect to time if it is analytically defined. 
        """
        pass 

class TimeFracCaputoDiffusionWaveTwoDimPDE(PDE):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.alpha = cfg.pde.alpha
        self.method = cfg.pde.method
        if 'GJ' in self.method:
            nums = cfg.pde.gauss_jacobi_params.nums
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.alpha)
            self.quad_t = (quad_t + 1) / 2
            self.quad_w = quad_wt * (1 / 2) ** (2 - self.alpha)

    def u_net(self, net: nn.Module, points: torch.Tensor) -> torch.Tensor:
        return net(points)

    def frac_diff(self, net: nn.Module, points: Tensor) -> Tuple[Tensor, Tensor]:
        points.requires_grad = True
        val = self.u_net(net, points)
        grads = torch.autograd.grad(
            outputs=val,
            inputs=points,
            grad_outputs=torch.ones_like(val),
            retain_graph=True,
            create_graph=True
        )[0]
        dt = grads[:, 0:1]
        dx = grads[:, 1:2]

        dxx = torch.autograd.grad(
            inputs=points,
            outputs=dx,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True
        )[0][:, 1:2]
        points.detach()
        points.requires_grad = False

        if self.method == 'GJ-II':
            return self._gj_ii(net, points, val, dt), dxx
        if self.method == 'MC-I':
            return self._mc_i(net, points, dt), dxx
        if self.method == 'MC-II':
            return self._mc_ii(net, points, val, dt), dxx
        if self.method == 'GJ-I':
            return self._gj_i(net, points, dt), dxx
        raise ValueError(f"Unknown method: {self.method}")

    def _mc_i(self, net: nn.Module, points: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        monte_carlo_nums = self.cfg.pde.monte_carlo_params.nums
        monte_carlo_eps = self.cfg.pde.monte_carlo_params.eps
        alpha = self.alpha
        num_points = len(points)
        coeff_for_frac_dt = sp.gamma(2 - alpha)
        taus = beta.rvs(2 - alpha, 1, size=monte_carlo_nums)
        taus: torch.Tensor = torch.from_numpy(taus).to(self.device)
        t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(device=self.device)
        t0.requires_grad = True
        val0 = self.u_net(net, t0)
        dt0 = torch.autograd.grad(
            outputs=val0,
            inputs=t0,
            grad_outputs=torch.ones_like(val0),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        t0.requires_grad = False
        t = points[:, 0:1].clone()
        taus = taus.unsqueeze(-1)
        t_tau = t.T * taus
        t_minus_t_tau = points[:, 0] - t_tau
        t_tau_clip = torch.maximum(t_tau, torch.tensor(monte_carlo_eps).to(self.device))
        x = points[:, 1].unsqueeze(0).expand(monte_carlo_nums, num_points).clone()
        new_points = torch.stack((t_minus_t_tau, x), dim=2)
        new_points.detach()
        new_points.requires_grad = True
        val_tau = self.u_net(net, new_points)
        dt_tau = torch.autograd.grad(
            outputs=val_tau,
            inputs=new_points,
            grad_outputs=torch.ones_like(val_tau),
            retain_graph=True,
            create_graph=True
        )[0][:, :, 0:1]
        part1 = torch.mean((dt - dt_tau) / t_tau_clip.unsqueeze(-1), dim=0)
        part1 *= (alpha - 1.0) / (2.0 - alpha) * torch.pow(points[:, 0:1], 2 - alpha)
        part2 = (dt - dt0) * torch.pow(points[:, 0:1], 1 - alpha)
        dt_alpha = (part1 + part2) / coeff_for_frac_dt
        new_points.requires_grad = False
        return dt_alpha

    def _mc_ii(self, net: nn.Module, points: torch.Tensor, val: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        monte_carlo_nums = self.cfg.pde.monte_carlo_params.nums
        monte_carlo_eps = self.cfg.pde.monte_carlo_params.eps
        alpha = self.alpha
        num_points = len(points)
        coeff_for_frac_dt = sp.gamma(2 - alpha)
        taus = beta.rvs(2 - alpha, 1, size=monte_carlo_nums)
        taus: torch.Tensor = torch.from_numpy(taus).to(self.device)
        t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(device=self.device)
        t0.requires_grad = True
        val0 = self.u_net(net, t0)
        dt0 = torch.autograd.grad(
            outputs=val0,
            inputs=t0,
            grad_outputs=torch.ones_like(val0),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        t0.requires_grad = False
        t = points[:, 0:1].clone()
        taus = taus.unsqueeze(-1)
        t_tau = t.T * taus
        t_minus_t_tau = points[:, 0] - t_tau
        t_tau_clip = torch.maximum(t_tau, torch.tensor(monte_carlo_eps).to(self.device))
        x = points[:, 1].unsqueeze(0).expand(monte_carlo_nums, num_points).clone()
        new_points = torch.stack((t_minus_t_tau, x), dim=2)
        new_points.detach()
        val2 = self.u_net(net, new_points)
        val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0)
        part1 = alpha * (alpha - 1) / (2 - alpha) * torch.pow(points[:, 0:1], 2 - alpha)
        part1 *= torch.mean((val - val2 - val3) / torch.pow(t_tau_clip.unsqueeze(-1), 2), dim=0)
        part2 = (alpha - 1) * (val - val0 - points[:, 0:1] * dt) / torch.pow(points[:, 0:1], alpha)
        part3 = (dt - dt0) / torch.pow(points[:, 0:1], alpha - 1)
        dt_alpha = (part3 - part2 - part1) / coeff_for_frac_dt
        return dt_alpha

    def _gj_i(self, net: nn.Module, points: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        nums = self.cfg.pde.gauss_jacobi_params.nums
        num_points = len(points)
        taus = torch.from_numpy(self.quad_t).to(self.device).float()
        quad_w = torch.from_numpy(self.quad_w).to(self.device).float()
        quad_w = quad_w.unsqueeze(-1).unsqueeze(-1)
        coeff_for_frac_dt = sp.gamma(2 - alpha)
        t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(device=self.device)
        t0.requires_grad = True
        val0 = self.u_net(net, t0)
        dt0 = torch.autograd.grad(
            outputs=val0,
            inputs=t0,
            grad_outputs=torch.ones_like(val0),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        t0.requires_grad = False
        t = points[:, 0:1].clone()
        taus = taus.unsqueeze(-1)
        t_tau = t.T * taus
        t_minus_t_tau = points[:, 0] - t_tau
        x = points[:, 1].unsqueeze(0).expand(nums, num_points).clone()
        new_points = torch.stack((t_minus_t_tau, x), dim=2)
        new_points.detach()
        new_points.requires_grad = True
        val_tau = self.u_net(net, new_points)
        dt_tau = torch.autograd.grad(
            outputs=val_tau,
            inputs=new_points,
            grad_outputs=torch.ones_like(val_tau),
            create_graph=True,
            retain_graph=True
        )[0][:, :, 0:1]
        part1 = torch.sum(quad_w * (dt - dt_tau) / t_tau.unsqueeze(-1), dim=0)
        part1 *= (alpha - 1.0) * torch.pow(points[:, 0:1], 2 - alpha)
        part2 = (dt - dt0) * torch.pow(points[:, 0:1], 1 - alpha)
        dt_alpha = (part1 + part2) / coeff_for_frac_dt
        new_points.requires_grad = False
        return dt_alpha

    def _gj_ii(self, net: nn.Module, points: torch.Tensor, val: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        coeff_for_frac_dt = sp.gamma(2 - alpha)
        nums = self.cfg.pde.gauss_jacobi_params.nums
        num_points = len(points)
        taus = torch.from_numpy(self.quad_t).to(self.device).float()
        quad_w = torch.from_numpy(self.quad_w).to(self.device).float()
        quad_w = quad_w.unsqueeze(-1).unsqueeze(-1)
        t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(self.device)
        t0.requires_grad = True
        val0 = self.u_net(net, t0)
        dt0 = torch.autograd.grad(
            outputs=val0,
            inputs=t0,
            grad_outputs=torch.ones_like(val0),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        t0.requires_grad = False
        t = points[:, 0:1].clone()
        taus = taus.unsqueeze(-1)
        t_tau = t.T * taus
        t_minus_t_tau = points[:, 0] - t_tau
        x = points[:, 1].unsqueeze(0).expand(nums, num_points).clone()
        new_points = torch.stack((t_minus_t_tau, x), dim=2)
        new_points.detach()
        val2 = self.u_net(net, new_points)
        val3 = t_tau * dt.squeeze(-1)
        part1 = alpha * (alpha - 1) * torch.pow(points[:, 0:1], 2 - alpha)
        part1 *= torch.sum(
            quad_w * ((val - val2 - val3.unsqueeze(-1)) / torch.pow(t_tau.unsqueeze(-1), 2)),
            dim=0
        )
        part2 = (alpha - 1) * (val - val0 - points[:, 0:1] * dt) / torch.pow(points[:, 0:1], alpha)
        part3 = (dt - dt0) / torch.pow(points[:, 0:1], alpha - 1)
        dt_alpha = (part3 - part2 - part1) / coeff_for_frac_dt
        return dt_alpha

    def frac_diff_exact(self, points:torch.Tensor):
        """
            compute the true dtal u with self.exact method
        """
        def eval_exact(pts: torch.Tensor) -> torch.Tensor:
            original_shape = pts.shape[:-1]
            flat = pts.reshape(-1, 2)
            val = self.exact(flat)
            return val.reshape(*original_shape, 1)

        def eval_exact_dt(pts: torch.Tensor) -> torch.Tensor:
            original_shape = pts.shape[:-1]
            flat = pts.reshape(-1, 2)
            val = self.exact_dt(flat)
            return val.reshape(*original_shape, 1)

        val = self.exact(points)
        dt = self.exact_dt(points)
        if self.method == 'MC-I':
            monte_carlo_nums = self.cfg.pde.monte_carlo_params.nums
            monte_carlo_eps = self.cfg.pde.monte_carlo_params.eps
            alpha = self.alpha
            num_points = len(points)
            coeff_for_frac_dt = sp.gamma(2 - alpha)
            taus = beta.rvs(2 - alpha, 1, size=monte_carlo_nums)
            taus = torch.from_numpy(taus).to(self.device)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(device=self.device)
            dt0 = self.exact_dt(t0)
            t = points[:, 0:1].clone()
            taus = taus.unsqueeze(-1)
            t_tau = t.T * taus
            t_minus_t_tau = points[:, 0] - t_tau
            t_tau_clip = torch.maximum(t_tau, torch.tensor(monte_carlo_eps, device=self.device))
            x = points[:, 1].unsqueeze(0).expand(monte_carlo_nums, num_points).clone()
            new_points = torch.stack((t_minus_t_tau, x), dim=2)
            dt_tau = eval_exact_dt(new_points)
            part1 = torch.mean((dt - dt_tau) / t_tau_clip.unsqueeze(-1), dim=0)
            part1 *= (alpha - 1.0) / (2.0 - alpha) * torch.pow(points[:, 0:1], 2 - alpha)
            part2 = (dt - dt0) * torch.pow(points[:, 0:1], 1 - alpha)
            dt_alpha = (part1 + part2) / coeff_for_frac_dt
            return dt_alpha
        if self.method == 'MC-II':
            monte_carlo_nums = self.cfg.pde.monte_carlo_params.nums
            monte_carlo_eps = self.cfg.pde.monte_carlo_params.eps
            alpha = self.alpha
            num_points = len(points)
            coeff_for_frac_dt = sp.gamma(2 - alpha)
            taus = beta.rvs(2 - alpha, 1, size=monte_carlo_nums)
            taus = torch.from_numpy(taus).to(self.device)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(device=self.device)
            val0 = self.exact(t0)
            dt0 = self.exact_dt(t0)
            t = points[:, 0:1].clone()
            taus = taus.unsqueeze(-1)
            t_tau = t.T * taus
            t_minus_t_tau = points[:, 0] - t_tau
            t_tau_clip = torch.maximum(t_tau, torch.tensor(monte_carlo_eps, device=self.device))
            x = points[:, 1].unsqueeze(0).expand(monte_carlo_nums, num_points).clone()
            new_points = torch.stack((t_minus_t_tau, x), dim=2)
            val2 = eval_exact(new_points)
            val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0)
            part1 = alpha * (alpha - 1) / (2 - alpha) * torch.pow(points[:, 0:1], 2 - alpha)
            part1 *= torch.mean((val - val2 - val3) / torch.pow(t_tau_clip.unsqueeze(-1), 2), dim=0)
            part2 = (alpha - 1) * (val - val0 - points[:, 0:1] * dt) / torch.pow(points[:, 0:1], alpha)
            part3 = (dt - dt0) / torch.pow(points[:, 0:1], alpha - 1)
            dt_alpha = (part3 - part2 - part1) / coeff_for_frac_dt
            return dt_alpha
        if self.method == 'GJ-I':
            alpha = self.alpha
            nums = self.cfg.pde.gauss_jacobi_params.nums
            num_points = len(points)
            taus = torch.from_numpy(self.quad_t).to(self.device).float()
            quad_w = torch.from_numpy(self.quad_w).to(self.device).float()
            quad_w = quad_w.unsqueeze(-1).unsqueeze(-1)
            coeff_for_frac_dt = sp.gamma(2 - alpha)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(device=self.device)
            dt0 = self.exact_dt(t0)
            t = points[:, 0:1].clone()
            taus = taus.unsqueeze(-1)
            t_tau = t.T * taus
            t_minus_t_tau = points[:, 0] - t_tau
            x = points[:, 1].unsqueeze(0).expand(nums, num_points).clone()
            new_points = torch.stack((t_minus_t_tau, x), dim=2)
            dt_tau = eval_exact_dt(new_points)
            part1 = torch.sum(quad_w * (dt - dt_tau) / t_tau.unsqueeze(-1), dim=0)
            part1 *= (alpha - 1.0) * torch.pow(points[:, 0:1], 2 - alpha)
            part2 = (dt - dt0) * torch.pow(points[:, 0:1], 1 - alpha)
            dt_alpha = (part1 + part2) / coeff_for_frac_dt
            return dt_alpha
        if self.method == 'GJ-II':
            alpha = self.alpha
            coeff_for_frac_dt = sp.gamma(2 - alpha)
            nums = self.cfg.pde.gauss_jacobi_params.nums
            num_points = len(points)
            taus = torch.from_numpy(self.quad_t).to(self.device).float()
            quad_w = torch.from_numpy(self.quad_w).to(self.device).float()
            quad_w = quad_w.unsqueeze(-1).unsqueeze(-1)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:, 1:]), dim=1).to(self.device)
            val0 = self.exact(t0)
            dt0 = self.exact_dt(t0)
            t = points[:, 0:1].clone()
            taus = taus.unsqueeze(-1)
            t_tau = t.T * taus
            t_minus_t_tau = points[:, 0] - t_tau
            x = points[:, 1].unsqueeze(0).expand(nums, num_points).clone()
            new_points = torch.stack((t_minus_t_tau, x), dim=2)
            val2 = eval_exact(new_points)
            val3 = t_tau * dt.squeeze(-1)
            part1 = alpha * (alpha - 1) * torch.pow(points[:, 0:1], 2 - alpha)
            part1 *= torch.sum(
                quad_w * ((val - val2 - val3.unsqueeze(-1)) / torch.pow(t_tau.unsqueeze(-1), 2)),
                dim=0
            )
            part2 = (alpha - 1) * (val - val0 - points[:, 0:1] * dt) / torch.pow(points[:, 0:1], alpha)
            part3 = (dt - dt0) / torch.pow(points[:, 0:1], alpha - 1)
            dt_alpha = (part3 - part2 - part1) / coeff_for_frac_dt
            return dt_alpha
        raise ValueError(f"Unknown method: {self.method}")
