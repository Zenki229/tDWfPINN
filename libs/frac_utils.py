import numpy as np
from scipy.stats import beta
from scipy.special import roots_jacobi
import scipy.special as sp

class PINN:
    def __init__(self, config):
        self.config = config
    def u_net(self, net , points):
        raise NotImplementedError("Subclasses should implement this!")
        #return net(points)
        # return points[:, 1:2]*(1-points[:,1:2])*net(points)
    def residual(self, net,*args):
        # generate residual
        raise NotImplementedError("Subclasses should implement this!")

def compute_frac_diff(net, x, config):
    """
    Compute Caputo Fractional derivative using our improved algorithm
    config:
        al: order 
        nums: number of samples or quadratures needed in Monte Carlo or Gauss-Jacobi qudrature 
        method: MC-I, MC-II or GJ-I, GJ-II
    """
    # This function previously used PyTorch and needs to be migrated or deprecated.
    pass
