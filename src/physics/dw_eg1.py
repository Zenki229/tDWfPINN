import torch
import numpy as np
from src.physics.pde import TimeFracCaputoDiffusionWaveTwoDimPDE
from src.physics.fractional import mitlef
from src.utils.typing import Tensor, Dict, Any
from omegaconf import DictConfig

class DWForwardEg1(TimeFracCaputoDiffusionWaveTwoDimPDE):
    """
    DWForward PDE problem implementation.
    
    Solves fractional diffusion-wave equation for example 1.
    """
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device, 1.5)
        self.k = 1.0
        self.lam = np.power(np.pi, 2)
        self.x_lim = [0, 1.0]
        self.t_lim = [0, 1.0]
        self.a = 2.0
        self.b = -1.0

    def residual(self, net: torch.nn.Module, points_all: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    def exact(self, points: Tensor) -> Tensor:
        points_np = points.detach().cpu().numpy()
        t = points_np[:, 0:1]
        x = points_np[:, 1:]
        
        part1 = np.sin(self.k * np.pi * x)
        z = -self.lam * np.power(t, self.alpha)
        part2 = self.a * mitlef(self.alpha, 1.0, z) + self.b * t * mitlef(self.alpha, 2.0, z)
        
        return torch.from_numpy(part1 * part2).to(self.device)

#TODO: 
