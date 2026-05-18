import jax
import jax.numpy as jnp
from jax import grad, vmap
import numpy as np
from pymittagleffler import mittag_leffler
from scipy.special import roots_jacobi

from libs.jax_pde_burgers import JAXDWBurgers
from libs.jax_pinn import ForwardIVP


class JAXDWForward(JAXDWBurgers):
    """Section 4.3 forward diffusion-wave benchmark.

    PDE:
        D_t^alpha u - lambda / (k^2 pi^2) * u_xx = 0
        u(t, 0) = u(t, 1) = 0
        u(0, x) = a sin(k pi x)
        u_t(0, x) = b sin(k pi x)

    In Section 4.3, a = 1 and b = -0.5.
    """

    def __init__(self, config, weighting_config=None):
        ForwardIVP.__init__(self, config, weighting_config)
        self.al = config.al
        self.xlim = config.xlim
        self.tlim = config.tlim
        self.method = config.method
        self.k = float(config.k)
        self.lam = float(config.lam)
        self.a = float(getattr(config, "a", 1.0))
        self.b = float(getattr(config, "b", -0.5))

        if "GJ" in self.method:
            nums = config.GJ.nums
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.al)
            self.quad_t = jnp.array((quad_t + 1) / 2)
            self.quad_w = jnp.array(quad_wt * (1 / 2) ** (2 - self.al))

    def r_net(self, apply_fn, params, points, key=None):
        t = points[..., 0]
        x = points[..., 1]

        def u_single(tt, xx):
            inp = jnp.stack([tt, xx])
            return apply_fn(params, inp)[0]

        u_t_fn = grad(u_single, 0)
        u_x_fn = grad(u_single, 1)
        u_xx_fn = grad(u_x_fn, 1)

        u_val = vmap(u_single)(t, x).reshape(-1)
        dxx = vmap(u_xx_fn)(t, x).reshape(-1)
        dt_frac = self.compute_frac_diff(apply_fn, params, t, x, u_val, u_t_fn, key)
        diffusion_scale = self.lam / (self.k * self.k * jnp.pi * jnp.pi)
        return dt_frac - diffusion_scale * dxx

    def losses(self, apply_fn, params, batch, key):
        losses = {}
        losses["in"] = self.r_net(apply_fn, params, batch["in"], key)

        points_bd = batch["bd"]
        losses["bd"] = self.u_net(apply_fn, params, points_bd).squeeze(-1)

        points_init = batch["init"]
        x_init = points_init[:, 1]
        basis = jnp.sin(self.k * jnp.pi * x_init)
        losses["init"] = (
            self.u_net(apply_fn, params, points_init).squeeze(-1) - self.a * basis
        )

        def u_single(t, x):
            inp = jnp.stack([t, x])
            return apply_fn(params, inp)[0]

        u_t_fn = grad(u_single, 0)
        dt_val = vmap(u_t_fn)(points_init[:, 0], x_init)
        losses["init_dt"] = dt_val - self.b * basis
        return losses

    def exact(self, points):
        points = np.asarray(jax.device_get(points))
        t = points[:, 0:1]
        x = points[:, 1:2]
        z = -self.lam * np.power(t, self.al)
        time_part = (
            self.a * np.real(mittag_leffler(z, self.al, 1.0))
            + self.b * t * np.real(mittag_leffler(z, self.al, 2.0))
        )
        return np.sin(self.k * np.pi * x) * time_part
