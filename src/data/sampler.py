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
