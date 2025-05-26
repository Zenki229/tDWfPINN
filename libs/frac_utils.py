import numpy as np
from torch.distributions.beta import Beta
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
    if config.method == 'MC-I':
        # Monte Carlo Integration
        MC = config.MC
        al = MC.al
        nums = MC.nums
        eps = MC.epsilon 
        coeff = sp.gamma(2-alpha) 
        epsilon = MC.epsilon
        taus = beta.rvs(2-alpha,1,size=nums)
        taus_max = np.maximum(taus,epsilon/t)
        taus = torch.from_numpy(taus).to(device=config.device)
        taus_max = torch.from_numpy(taus_max).to(device=config.device)
        x.requires_grad = True 
        f = net(x)
        part1 = (f.grad(t,1.0)-f.grad(0,1.0))*t**(1-alpha)+(alpha-1)/(2-alpha)*t**(2-alpha)*np.mean((f.grad(t,1.0)-f.grad(t-t*taus,1.0))/(t*taus_max))
        MCI = part1/coeff
