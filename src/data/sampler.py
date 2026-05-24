import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from src.utils.typing import Tensor

class BaseSampler(Dataset):
    """
    Base class for sampling points in the domain and boundary.
    """
    def __init__(self, batch_size: Dict[str, int], device: torch.device):
        self.device = device
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Tensor]:
        return self.sample()

    def sample(self) -> Dict[str, Tensor]:
        raise NotImplementedError("Subclasses should implement this!")

    def rad_sampler(self, residual: Tensor, points: Tensor, num_outputs: int) -> Tensor:
        node = points.detach().cpu().numpy()
        res = residual.detach().cpu().numpy()
        err = np.power(res, 2)
        err_sum = np.sum(err)
        if err_sum == 0:
            err_normal = np.ones_like(err) / len(err)
        else:
            err_normal = err / err_sum
        p = err_normal.flatten()
        p = p / p.sum()
        size = node.shape[0]
        idx = np.random.choice(size, num_outputs, replace=False, p=p)
        return points[idx]

class TimeSpaceSampler(BaseSampler):
    """
    Sampler for time-space domains.
    
    Args:
        spatial_lim (List[List[float]]): Limits for spatial dimensions [[x_min, x_max], ...].
        time_lim (List[float]): Limits for time dimension [t_min, t_max].
        device (torch.device): Device to store tensors on.
        batch_size (Dict[str, int]): Batch sizes for 'domain', 'boundary', 'initial'.
    """
    def __init__(self, spatial_lim: List[List[float]], time_lim: List[float], 
                 device: torch.device, batch_size: Dict[str, int]):
        super().__init__(batch_size, device)
        self.spatial_lim = spatial_lim
        self.time_lim = time_lim
        self.dim = len(spatial_lim)
        
    def sample(self) -> Dict[str, Tensor]:
        points = {}
        
        # 1. Sample in the domain (interior)
        # Structure: [t, x1, x2, ...]
        n_domain = self.batch_size['domain']
        domain_points = torch.zeros((n_domain, self.dim + 1), device=self.device)
        
        # Time dimension
        t_len = self.time_lim[1] - self.time_lim[0]
        domain_points[:, 0] = torch.rand(n_domain, device=self.device) * t_len + self.time_lim[0]
        
        # Spatial dimensions
        for i in range(self.dim):
            x_len = self.spatial_lim[i][1] - self.spatial_lim[i][0]
            domain_points[:, i + 1] = torch.rand(n_domain, device=self.device) * x_len + self.spatial_lim[i][0]
            
        points['domain'] = domain_points

        # 2. Sample on the boundary
        # Boundaries: 2 * dim faces (for hypercube)
        n_boundary = self.batch_size['boundary']
        # Randomly choose which face for each point
        face_indices = torch.randint(0, 2 * self.dim, (n_boundary,), device=self.device)
        
        boundary_points_list = []
        
        # We need to generate points for each face type
        # Faces are: x_i = min, x_i = max for i in 0..dim-1
        # Note: Time is dim 0 in storage, but usually we treat time boundary separate (initial/terminal).
        # The original code treats spatial boundaries.
        # Original code logic:
        # dim = spatial dims. 
        # points struct: [t, x...]
        # faces: 2*dim. 
        # For each face, t is random [t0, t1].
        # One x_k is fixed, others random.
        
        # Pre-allocate for efficiency could be hard due to variable counts per face, 
        # but let's follow the logic of generating per point or per face batch.
        # Original code iterates over faces.
        
        bd_points = torch.zeros((n_boundary, self.dim + 1), device=self.device)
        # Set Time for all
        bd_points[:, 0] = torch.rand(n_boundary, device=self.device) * t_len + self.time_lim[0]
        
        for i in range(2 * self.dim):
            # i ranges 0 .. 2*dim - 1
            # spatial dim index: m = i // 2
            # min/max: n = i % 2
            
            mask = (face_indices == i)
            count = mask.sum().item()
            if count == 0:
                continue
                
            m = i // 2
            n = i % 2
            
            # Set random values for all spatial dims first
            for j in range(self.dim):
                if j == m:
                    # Fixed value
                    val = self.spatial_lim[j][n]
                    bd_points[mask, j + 1] = val
                else:
                    # Random value
                    l = self.spatial_lim[j][1] - self.spatial_lim[j][0]
                    bd_points[mask, j + 1] = torch.rand(count, device=self.device) * l + self.spatial_lim[j][0]
                    
        points['boundary'] = bd_points

        # 3. Sample initial condition (t = t_0)
        n_initial = self.batch_size['initial']
        initial_points = torch.zeros((n_initial, self.dim + 1), device=self.device)
        # t is fixed at t_lim[0]
        initial_points[:, 0] = self.time_lim[0]
        
        for i in range(self.dim):
            x_len = self.spatial_lim[i][1] - self.spatial_lim[i][0]
            initial_points[:, i + 1] = torch.rand(n_initial, device=self.device) * x_len + self.spatial_lim[i][0]
            
        points['initial'] = initial_points
        
        return points

    def rad_sampler(self, residual: Tensor, points: Tensor, num_outputs: int) -> Tensor:
        """
        Residual-based Adaptive Distribution (RAD) sampling.
        
        Args:
            residual (Tensor): Residual values at points.
            points (Tensor): Coordinate points.
            num_outputs (int): Number of points to resample.
            
        Returns:
            Tensor: Selected points.
        """
        node = points.detach().cpu().numpy()
        res = residual.detach().cpu().numpy()
        
        err = np.power(res, 2)
        err_sum = np.sum(err)
        if err_sum == 0:
            err_normal = np.ones_like(err) / len(err)
        else:
            err_normal = err / err_sum
            
        # Flatten for choice
        p = err_normal.flatten()
        # Normalize strictly to avoid sum != 1 errors due to float precision
        p = p / p.sum()
        
        size = node.shape[0]
        idx = np.random.choice(size, num_outputs, replace=False, p=p)
        
        points_output = points[idx]
        return points_output


class IrregularHoleSampler(BaseSampler):
    """Sampler for Omega = (-1, 1)^2 minus an off-center circular hole."""

    def __init__(self, time_lim: List[float], batch_size: Dict[str, int],
                 device: torch.device, center=(-0.3, 0.2), r0=0.25):
        super().__init__(batch_size, device)
        self.time_lim = time_lim
        self.center = torch.tensor(list(center), device=device)
        self.r0 = float(r0)

    def _sample_time(self, n: int) -> Tensor:
        t_len = self.time_lim[1] - self.time_lim[0]
        return torch.rand(n, device=self.device) * t_len + self.time_lim[0]

    def _inside_domain(self, xy: Tensor) -> Tensor:
        in_square = torch.all((xy > -1.0) & (xy < 1.0), dim=1)
        outside_hole = torch.sum((xy - self.center) ** 2, dim=1) > self.r0 ** 2
        return in_square & outside_hole

    def _sample_spatial(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            proposal = torch.rand(max(remaining * 2, 16), 2, device=self.device) * 2.0 - 1.0
            accepted = proposal[self._inside_domain(proposal)]
            if accepted.numel() == 0:
                continue
            take = min(remaining, accepted.shape[0])
            chunks.append(accepted[:take])
            remaining -= take
        return torch.cat(chunks, dim=0)

    def _sample_boundary(self, n: int) -> Tensor:
        points = torch.zeros((n, 3), device=self.device)
        points[:, 0] = self._sample_time(n)
        pieces = torch.randint(0, 5, (n,), device=self.device)

        for piece in range(5):
            mask = pieces == piece
            count = int(mask.sum().item())
            if count == 0:
                continue
            if piece < 4:
                vals = torch.rand(count, device=self.device) * 2.0 - 1.0
                if piece == 0:
                    points[mask, 1] = -1.0
                    points[mask, 2] = vals
                elif piece == 1:
                    points[mask, 1] = 1.0
                    points[mask, 2] = vals
                elif piece == 2:
                    points[mask, 1] = vals
                    points[mask, 2] = -1.0
                else:
                    points[mask, 1] = vals
                    points[mask, 2] = 1.0
            else:
                theta = torch.rand(count, device=self.device) * (2.0 * np.pi)
                points[mask, 1] = self.center[0] + self.r0 * torch.cos(theta)
                points[mask, 2] = self.center[1] + self.r0 * torch.sin(theta)
        return points

    def sample(self) -> Dict[str, Tensor]:
        n_domain = self.batch_size["domain"]
        n_initial = self.batch_size["initial"]
        xy_domain = self._sample_spatial(n_domain)
        xy_initial = self._sample_spatial(n_initial)
        return {
            "domain": torch.cat([self._sample_time(n_domain).reshape(-1, 1), xy_domain], dim=1),
            "boundary": self._sample_boundary(self.batch_size["boundary"]),
            "initial": torch.cat([torch.zeros((n_initial, 1), device=self.device), xy_initial], dim=1),
        }


class LShapeSampler(BaseSampler):
    """Sampler for Omega_L = [-1, 1]^2 \\ [0, 1]^2."""

    def __init__(self, time_lim: List[float], batch_size: Dict[str, int],
                 device: torch.device):
        super().__init__(batch_size, device)
        self.time_lim = time_lim

    def _sample_time(self, n: int) -> Tensor:
        t_len = self.time_lim[1] - self.time_lim[0]
        return torch.rand(n, device=self.device) * t_len + self.time_lim[0]

    def _inside_domain(self, xy: Tensor) -> Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        in_square = (x > -1.0) & (x < 1.0) & (y > -1.0) & (y < 1.0)
        outside_removed_quadrant = (x < 0.0) | (y < 0.0)
        return in_square & outside_removed_quadrant

    def _sample_spatial(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            proposal = torch.rand(max(remaining * 2, 16), 2, device=self.device) * 2.0 - 1.0
            accepted = proposal[self._inside_domain(proposal)]
            if accepted.numel() == 0:
                continue
            take = min(remaining, accepted.shape[0])
            chunks.append(accepted[:take])
            remaining -= take
        return torch.cat(chunks, dim=0)

    def _sample_boundary(self, n: int) -> Tensor:
        points = torch.zeros((n, 3), device=self.device)
        points[:, 0] = self._sample_time(n)
        pieces = torch.randint(0, 6, (n,), device=self.device)

        for piece in range(6):
            mask = pieces == piece
            count = int(mask.sum().item())
            if count == 0:
                continue
            vals = torch.rand(count, device=self.device) * 2.0 - 1.0
            vals01 = torch.rand(count, device=self.device)
            vals_neg = -torch.rand(count, device=self.device)
            if piece == 0:
                points[mask, 1] = -1.0
                points[mask, 2] = vals
            elif piece == 1:
                points[mask, 1] = vals
                points[mask, 2] = -1.0
            elif piece == 2:
                points[mask, 1] = 1.0
                points[mask, 2] = vals_neg
            elif piece == 3:
                points[mask, 1] = vals_neg
                points[mask, 2] = 1.0
            elif piece == 4:
                points[mask, 1] = 0.0
                points[mask, 2] = vals01
            else:
                points[mask, 1] = vals01
                points[mask, 2] = 0.0
        return points

    def sample(self) -> Dict[str, Tensor]:
        n_domain = self.batch_size["domain"]
        n_initial = self.batch_size["initial"]
        xy_domain = self._sample_spatial(n_domain)
        xy_initial = self._sample_spatial(n_initial)
        return {
            "domain": torch.cat([self._sample_time(n_domain).reshape(-1, 1), xy_domain], dim=1),
            "boundary": self._sample_boundary(self.batch_size["boundary"]),
            "initial": torch.cat([torch.zeros((n_initial, 1), device=self.device), xy_initial], dim=1),
        }
