import torch
import numpy as np
import scipy.special as sp
from scipy.stats import beta
from typing import Tuple
from src.physics.pde import PDE
from src.physics.fractional import roots_jacobi, mitlef
from src.utils.typing import Tensor, Dict, Any
from omegaconf import DictConfig

class DWForward(PDE):
    """
    DWForward PDE problem implementation.
    
    Solves fractional diffusion-wave equation.
    """
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # PDE Parameters
        self.alpha = cfg.pde.alpha
        self.k = cfg.pde.k
        self.lam = cfg.pde.lambda_val
        self.x_lim = cfg.pde.x_lim
        self.t_lim = cfg.pde.t_lim
        
        # Method specific setup
        self.method = cfg.pde.method
        if 'GJ' in self.method:
            nums = cfg.pde.gj_params.nums
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.alpha)
            self.quad_t = (quad_t + 1) / 2
            self.quad_w = quad_wt * (1 / 2) ** (2 - self.alpha)
        
        self.a = getattr(cfg.pde, 'a', 1.0)
        self.b = getattr(cfg.pde, 'b', 1.0)

    def u_net(self, net: torch.nn.Module, points: Tensor) -> Tensor:
        return net(points)

    def residual(self, net: torch.nn.Module, points_all: Dict[str, Tensor]) -> Dict[str, Tensor]:
        losses = {}
        
        # Domain Residual
        points_in = points_all['domain']
        points_in.requires_grad = True
        dt, dxx = self.frac_diff(net, points_in)
        
        residual_in = dt - dxx * self.lam / ((self.k * np.pi) ** 2)
        losses['domain'] = residual_in
        
        # Boundary Residual
        points_bd = points_all['boundary']
        pred_bd = self.u_net(net, points_bd)
        losses['boundary'] = pred_bd 
        
        # Initial Condition Residual (u(0,x))
        points_init = points_all['initial']
        
        # x coordinates
        x = points_init[:, 1:] 
        
        target_init = self.a * torch.sin(self.k * np.pi * x)
        pred_init = self.u_net(net, points_init)
        losses['initial'] = pred_init - target_init
        
        # Initial Derivative Residual (u_t(0,x))
        points_init.requires_grad = True
        val_init = self.u_net(net, points_init)
        dt_init = torch.autograd.grad(
            outputs=val_init, 
            inputs=points_init, 
            grad_outputs=torch.ones_like(val_init),
            retain_graph=True, 
            create_graph=True
        )[0][:, 0:1] 
        
        target_dt_init = self.b * torch.sin(self.k * np.pi * x)
        losses['initial_dt'] = dt_init - target_dt_init
        
        return losses

    def frac_diff(self, net: torch.nn.Module, points: Tensor) -> Tuple[Tensor, Tensor]:
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
        
        if self.method == 'GJ-II':
            return self._gj_ii(net, points, points[:, 0:1], points[:, 1:], val, dt, dxx)
        elif self.method == 'MC-I':
            return self._mc_i(net, points, points[:, 0:1], points[:, 1:], val, dt, dxx)
        else:
            return self._gj_ii(net, points, points[:, 0:1], points[:, 1:], val, dt, dxx)

    def _gj_ii(self, net, points, t, x, val, dt, dxx):
        al = self.alpha
        coeff = sp.gamma(2 - al)
        nums = self.cfg.pde.gj_params.nums
        lens = len(points)
        
        taus = torch.from_numpy(self.quad_t).to(self.device).float() # M
        quad_w = torch.from_numpy(self.quad_w).to(self.device).float() # M
        quad_w = quad_w.unsqueeze(-1).unsqueeze(-1) # M*1*1
        
        t0 = torch.cat((torch.zeros_like(t), x), dim=1).to(self.device)
        # t0.requires_grad = True
        val0 = self.u_net(net, t0)
        dt0 = torch.autograd.grad(outputs=val0, inputs=t0, grad_outputs=torch.ones_like(val0), create_graph=True)[0][:, 0:1]
        
        t_tau = t.T * taus.unsqueeze(-1) # M*N
        t_t_tau = points[:, 0] - t_tau # M*N
        
        new_x = x.unsqueeze(0).expand(nums, lens, -1) 
        new_points = torch.cat((t_t_tau.unsqueeze(-1), new_x), dim=2) # M*N*(1+D)
        
        val2 = self.u_net(net, new_points) # M*N*1
        val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0) # M*N*1
        
        numerator = val.unsqueeze(0) - val2 - val3
        denominator = torch.pow(t_tau.unsqueeze(-1), 2)
        integrand = numerator / denominator
        
        integral = torch.sum(quad_w * integrand, dim=0) # N*1
        
        part1 = al * (al - 1) * torch.pow(t, 2 - al) * integral
        part2 = (al - 1) * (val - val0 - t * dt) / torch.pow(t, al)
        part3 = (dt - dt0) / torch.pow(t, al - 1)
        
        dtal = (part3 - part2 - part1) / coeff
        return dtal, dxx

    def _mc_i(self, net, points, t, x, val, dt, dxx):
        al = self.alpha
        nums = self.cfg.pde.mc_params.nums
        eps = self.cfg.pde.mc_params.eps
        lens = len(points)
        coeff = sp.gamma(2 - al)
        
        taus_np = beta.rvs(2 - al, 1, size=nums)
        taus = torch.from_numpy(taus_np).to(self.device).float()
        
        t0 = torch.cat((torch.zeros_like(t), x), dim=1).to(self.device)
        # t0.requires_grad = True
        val0 = self.u_net(net, t0)
        dt0 = torch.autograd.grad(outputs=val0, inputs=t0, grad_outputs=torch.ones_like(val0), create_graph=True)[0][:, 0:1]
        
        taus = taus.reshape(-1, 1) # M*1
        t_tau = t.T * taus # M*N
        t_t_tau = points[:, 0] - t_tau # M*N
        t_tau_max = torch.max(t_tau, torch.tensor([eps], device=self.device))
        
        new_x = x.unsqueeze(0).expand(nums, lens, -1)
        new_points = torch.cat((t_t_tau.unsqueeze(-1), new_x), dim=2)
        # new_points.requires_grad = True
        
        valttau = self.u_net(net, new_points)
        dttau = torch.autograd.grad(outputs=valttau, inputs=new_points, grad_outputs=torch.ones_like(valttau), create_graph=True)[0][:, :, 0:1]
        
        term = (dt.unsqueeze(0) - dttau) / t_tau_max.unsqueeze(-1)
        part1 = torch.mean(term, dim=0) # N*1
        part1 = part1 * (al - 1.0) / (2.0 - al) * torch.pow(t, 2 - al)
        part2 = (dt - dt0) * torch.pow(t, 1 - al)
        
        dtal = (part1 + part2) / coeff
        return dtal, dxx

    def exact(self, points: Tensor) -> Tensor:
        points_np = points.detach().cpu().numpy()
        t = points_np[:, 0:1]
        x = points_np[:, 1:]
        
        part1 = np.sin(self.k * np.pi * x)
        z = -self.lam * np.power(t, self.alpha)
        part2 = self.a * mitlef(self.alpha, 1.0, z) + self.b * t * mitlef(self.alpha, 2.0, z)
        
        return torch.from_numpy(part1 * part2).to(self.device)
