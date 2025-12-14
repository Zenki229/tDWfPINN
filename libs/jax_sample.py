import numpy as np
import jax.numpy as jnp
import jax

class BaseEasySampler:
    def __init__(self, batch):
        self.batch = batch 
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample()
    
    def sample(self, **args):
        raise NotImplementedError("Subclasses should implement this!")

class TimeSpaceEasySampler(BaseEasySampler):
    def __init__(self, axeslim, tlim, batch):
        super().__init__(batch)
        self.axeslim = axeslim
        self.tlim = tlim
        self.dim = len(axeslim)
        
    def sample(self):
        size = self.batch
        points = {}
        
        # Sample in the domain
        size_in = size['in']
        # t, x1, x2, ...
        node_in = np.zeros((size_in, self.dim + 1))
        
        leftlim = self.tlim[0]
        rightlim = self.tlim[1]
        length = rightlim - leftlim
        node_in[:, 0] = np.random.rand(size_in) * length + leftlim
        
        for i in range(self.dim):
            leftlim = self.axeslim[i][0]
            rightlim = self.axeslim[i][1]
            length = rightlim - leftlim
            node_in[:, i+1] = np.random.rand(size_in) * length + leftlim
            
        points['in'] = node_in
        
        # Sample on the boundary
        size_bd = size['bd']
        # We need to pick which boundary face to sample from
        # There are 2 * dim faces (including time boundaries if we consider them, 
        # but usually 'bd' means spatial boundary and 'init' means t=0)
        # The original code includes t boundaries in 'bd' loop?
        # Let's look at original code: 
        # bd_num = torch.randint(low=0, high=2*self.dim, ...)
        # It seems it samples from 2*dim spatial faces? 
        # Wait, self.dim is spatial dimension (len(axeslim)).
        # But node_in has dim+1 columns (time + space).
        # Original code: 
        # for i in range(2*self.dim): ...
        # m, n = i//2, i % 2
        # node_bd[i] = ... random t ...
        # if j != m: random x_j
        # else: fixed x_m (min or max)
        # So it samples spatial boundaries at random times.
        
        bd_num = np.random.randint(0, 2*self.dim, size=size_bd)
        node_bd_list = []
        
        for i in range(2 * self.dim):
            ind = np.where(bd_num == i)[0]
            num = len(ind)
            if num == 0:
                continue
                
            m, n = i // 2, i % 2 # m is the spatial dimension index (0..dim-1), n is 0 or 1 (min or max)
            
            # Initialize with random time and space
            # columns: t, x_0, x_1, ...
            curr_bd = np.zeros((num, self.dim + 1))
            
            # Time column (0)
            curr_bd[:, 0] = np.random.rand(num) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]
            
            # Space columns (1..dim)
            for j in range(self.dim):
                if j != m:
                    # Random sample for other dimensions
                    l = self.axeslim[j][0]
                    r = self.axeslim[j][1]
                    curr_bd[:, j+1] = np.random.rand(num) * (r - l) + l
                else:
                    # Fixed value for the boundary dimension
                    curr_bd[:, j+1] = self.axeslim[m][n]
            
            node_bd_list.append(curr_bd)
            
        if node_bd_list:
            points['bd'] = np.concatenate(node_bd_list, axis=0)
        else:
            points['bd'] = np.zeros((0, self.dim + 1))

        # Sample initial condition (t=0)
        size_init = size['init']
        node_init = np.zeros((size_init, self.dim + 1))
        node_init[:, 0] = 0.0 # t=0
        
        for i in range(self.dim):
            l = self.axeslim[i][0]
            r = self.axeslim[i][1]
            length = r - l
            node_init[:, i+1] = np.random.rand(size_init) * length + l
            
        points['init'] = node_init
        
        return points

    def rad_sampler(self, residual, points, num_outputs, key=None):
        """
        RAD sampling based on the residual.
        residual: array of shape (N, 1) or (N,)
        points: array of shape (N, D)
        """
        # Ensure numpy
        residual = np.array(residual)
        points = np.array(points)
        
        err = np.power(residual, 2)
        # Avoid division by zero
        err_sum = np.sum(err)
        if err_sum < 1e-10:
            p = np.ones_like(err.flatten()) / len(err)
        else:
            p = (err / err_sum).flatten()
            
        size = points.shape[0]
        # Weighted sampling without replacement
        ind = np.random.choice(size, num_outputs, replace=False, p=p)
        return points[ind]
