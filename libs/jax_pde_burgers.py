import jax
import jax.numpy as jnp
from jax import grad, hessian, vmap, jacfwd, jacrev
import numpy as np
from scipy.special import roots_jacobi, gamma as sp_gamma
from functools import partial

class JAXDWBurgers:
    def __init__(self, config):
        self.config = config
        self.al = config.al
        self.beta = config.beta
        self.tlim = config.tlim
        self.xlim = config.xlim
        
        # Precompute Gauss-Jacobi nodes and weights if needed
        if 'GJ' in config.method:
            GJ = config.GJ
            nums = GJ.nums
            # roots_jacobi(n, alpha, beta) -> returns roots of P_n^(alpha, beta)
            # We need roots for weight (1-x)^alpha (1+x)^beta on [-1, 1]
            # Mapping to [0, 1] for time integral
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.al)
            
            # Transform from [-1, 1] to [0, 1]
            self.quad_t = jnp.array((quad_t + 1) / 2)
            self.quad_w = jnp.array(quad_wt * (1 / 2) ** (2 - self.al))
            
    def u_net(self, apply_fn, params, points):
        """
        points: (N, 2) array [t, x]
        returns: (N, 1) array
        """
        return apply_fn(params, points)

    def exact(self, datafile):
        data = np.load(datafile)
        # t = data['t'] 
        # x = data['x']
        u = data['u'] 
        return u.reshape(-1, 1)

    @partial(jax.jit, static_argnums=(0, 1))
    def residual(self, apply_fn, params, points_all, key):
        """
        points_all: dict of arrays {'in': ..., 'bd': ..., 'init': ...}
        """
        losses = {}
        
        # 1. Residual in the domain
        points_in = points_all['in'] # (N, 2)
        
        # We need derivatives. Define a wrapper for a single point (t, x)
        def u_single(t, x):
            # t, x are scalars
            inputs = jnp.stack([t, x], axis=0)
            return apply_fn(params, inputs)[0]

        # Derivatives
        u_t = grad(u_single, 0)
        u_x = grad(u_single, 1)
        u_xx = grad(u_x, 1) # Second derivative wrt x

        # Vectorized derivatives
        # vmap over batch dimension
        # points_in[:, 0] is t, points_in[:, 1] is x
        t_in = points_in[:, 0]
        x_in = points_in[:, 1]
        
        # Calculate u, ux, uxx at points_in
        u_val = vmap(u_single)(t_in, x_in).reshape(-1, 1)
        dx = vmap(u_x)(t_in, x_in).reshape(-1, 1)
        dxx = vmap(u_xx)(t_in, x_in).reshape(-1, 1)
        
        # Fractional derivative calculation
        # This requires integrating over history [0, t]
        # We delegate this to a helper function
        dt_frac = self.compute_frac_diff(apply_fn, params, t_in, x_in, u_val, u_t, key)
        
        losses['in'] = dt_frac + u_val * dx - (0.01 / jnp.pi) * dxx
        
        # 2. Residual on boundary
        points_bd = points_all['bd']
        losses['bd'] = self.u_net(apply_fn, params, points_bd)
        
        # 3. Residual on initial condition
        points_init = points_all['init']
        # Exact IC: -sin(pi * x)
        x_init = points_init[:, 1:2]
        pred_init = -jnp.sin(jnp.pi * x_init)
        losses['init'] = self.u_net(apply_fn, params, points_init) - pred_init
        
        # 4. Residual on 1st derivative initial condition (if applicable)
        # The original code calculates gradient of u w.r.t t at t=0
        # dt = autograd.grad(...)
        # For t=0, u_t(0, x)
        t0 = points_init[:, 0] # all zeros
        x0 = points_init[:, 1]
        
        dt_val = vmap(u_t)(t0, x0).reshape(-1, 1)
        pred_dt = self.beta * jnp.sin(jnp.pi * x0).reshape(-1, 1)
        losses['init_dt'] = dt_val - pred_dt
        
        return losses

    def compute_frac_diff(self, apply_fn, params, t, x, u_val, u_t_fn, key):
        """
        Compute fractional derivative using the specified method.
        t: (N,)
        x: (N,)
        u_val: (N, 1) - u(t,x)
        u_t_fn: function to compute du/dt(t, x)
        """
        config = self.config
        method = config.method
        al = self.al
        
        # Helper to compute u(t, x)
        def u_fn(t_scalar, x_scalar):
            inputs = jnp.stack([t_scalar, x_scalar], axis=0)
            return apply_fn(params, inputs)[0]
            
        u_t_vmap = vmap(u_t_fn)
        
        # Pre-calculate u'(0, x)
        dt0 = u_t_vmap(jnp.zeros_like(t), x).reshape(-1, 1)
        
        # Current dt (u'(t, x))
        dt = u_t_vmap(t, x).reshape(-1, 1)
        
        if method == 'GJ-II':
            # Gauss-Jacobi Quadrature
            nums = config.GJ.nums
            coeff = sp_gamma(2 - al)
            
            # Quadrature points: t_tau = t * tau_i
            # taus: (M,)
            taus = self.quad_t
            weights = self.quad_w
            
            # Reshape for broadcasting
            # t: (N, 1), x: (N, 1)
            t = t.reshape(-1, 1)
            x = x.reshape(-1, 1)
            
            # t_tau matrix: (N, M)
            t_tau_mat = t @ taus.reshape(1, -1) # (N, M)
            
            # Arguments for u_net evaluation
            # We need to evaluate at (t - t_tau, x)
            # t_args = t - t_tau
            t_eval = t - t_tau_mat # (N, M)
            x_eval = jnp.broadcast_to(x, (x.shape[0], nums)) # (N, M)
            
            # Flatten to evaluate
            t_flat = t_eval.flatten()
            x_flat = x_eval.flatten()
            
            # u(t - t_tau, x)
            val2 = vmap(u_fn)(t_flat, x_flat).reshape(t.shape[0], nums) # (N, M)
            
            # val3 = t_tau * dt
            # dt is (N, 1)
            val3 = t_tau_mat * dt # (N, M)
            
            # Integrand: (u(t,x) - u(t-t_tau,x) - t_tau*u'(t,x)) / t_tau^2
            # u_val is (N, 1)
            numerator = u_val - val2 - val3
            denominator = t_tau_mat ** 2
            
            # Avoid division by zero if t is close to 0 (though GJ nodes shouldn't include 0 exactly usually)
            # Add small epsilon or mask
            denominator = jnp.where(denominator < 1e-10, 1e-10, denominator)
            
            integrand = numerator / denominator
            
            # Sum weighted integrand
            # weights is (M,)
            integral = jnp.sum(integrand * weights.reshape(1, -1), axis=1, keepdims=True) # (N, 1)
            
            part1 = al * (al - 1) * (t ** (2 - al)) * integral
            part2 = (al - 1) * (u_val - (u_fn(0., 0.) if 0 else 0) - t * dt) / (t ** al + 1e-10) 
            # Note: u(0,x) is needed for part2?
            # Original code: part2 = (al-1)*(val-val0-points[:, 0:1]*dt)/torch.pow(points[:, 0:1],al)
            # val0 is u(0, x)
            val0 = vmap(u_fn)(jnp.zeros_like(t.flatten()), x.flatten()).reshape(-1, 1)
            part2 = (al - 1) * (u_val - val0 - t * dt) / (t ** al + 1e-10)
            
            part3 = (dt - dt0) / (t ** (al - 1) + 1e-10)
            
            dtal = (part3 - part2 - part1) / coeff
            return dtal

        elif method == 'MC-I' or method == 'MC-II':
             # Monte Carlo
             # Need random taus from Beta(2-al, 1)
             nums = config.MC.nums
             eps = config.MC.eps
             coeff = sp_gamma(2 - al)
             
             # Sample taus
             # key is required here
             taus = jax.random.beta(key, 2 - al, 1, shape=(nums,))
             
             # Similar logic to GJ but with mean instead of weighted sum
             # ... implementation ...
             # For brevity, implementing GJ-II logic first as it's default in config_eg1.py
             pass
        
        # Fallback or Todo
        return jnp.zeros_like(u_val)

