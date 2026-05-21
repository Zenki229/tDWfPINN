import jax
import jax.numpy as jnp
from jax import grad, jacrev, vmap
import numpy as np
from scipy.special import gamma as sp_gamma
from scipy.special import roots_jacobi

from libs.jax_pinn import ForwardIVP


class JAXDWBurgers(ForwardIVP):
    """JAX implementation of the time-fractional Burgers equation.

    PDE: D_t^alpha u + u * u_x - nu * u_xx = 0, alpha in (1, 2)
    IC:  u(0, x) = -sin(pi * x)
         u_t(0, x) = beta * sin(pi * x)
    BC:  u(t, x_left) = u(t, x_right) = 0
    """

    def __init__(self, config, weighting_config=None):
        super().__init__(config, weighting_config)
        self.al = config.al
        self.beta = config.beta
        self.tlim = config.tlim
        self.xlim = config.xlim
        self.method = config.method

        if "GJ" in self.method:
            nums = config.GJ.nums
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.al)
            self.quad_t = jnp.array((quad_t + 1) / 2)
            self.quad_w = jnp.array(quad_wt * (1 / 2) ** (2 - self.al))

    def u_net(self, apply_fn, params, points):
        return apply_fn(params, points)

    def r_net(self, apply_fn, params, points, key=None):
        """PDE residual at domain points (t, x), returned as shape (N,)."""
        t = points[..., 0]
        x = points[..., 1]

        def u_single(tt, xx):
            inp = jnp.stack([tt, xx])
            return apply_fn(params, inp)[0]

        u_t_fn = grad(u_single, 0)
        u_x_fn = grad(u_single, 1)
        u_xx_fn = grad(u_x_fn, 1)

        u_val = vmap(u_single)(t, x).reshape(-1)
        dx = vmap(u_x_fn)(t, x).reshape(-1)
        dxx = vmap(u_xx_fn)(t, x).reshape(-1)
        dt_frac = self.compute_frac_diff(apply_fn, params, t, x, u_val, u_t_fn, key)

        return dt_frac + u_val * dx - (0.01 / jnp.pi) * dxx

    def losses(self, apply_fn, params, batch, key):
        losses = {}

        losses["in"] = self.r_net(apply_fn, params, batch["in"], key)

        points_bd = batch["bd"]
        losses["bd"] = self.u_net(apply_fn, params, points_bd).squeeze(-1)

        points_init = batch["init"]
        x_init = points_init[:, 1]
        pred_init = -jnp.sin(jnp.pi * x_init)
        losses["init"] = (
            self.u_net(apply_fn, params, points_init).squeeze(-1) - pred_init
        )

        def u_single(t, x):
            inp = jnp.stack([t, x])
            return apply_fn(params, inp)[0]

        u_t_fn = grad(u_single, 0)
        t0 = points_init[:, 0]
        x0 = points_init[:, 1]
        dt_val = vmap(u_t_fn)(t0, x0)
        pred_dt = self.beta * jnp.sin(jnp.pi * x0)
        losses["init_dt"] = dt_val - pred_dt

        return losses

    def compute_diag_ntk(self, apply_fn, params, batch, key):
        """Compute one scalar NTK proxy per loss term for adaptive weighting."""
        losses = self.losses(apply_fn, params, batch, key)
        ntk = {}
        for k in losses:
            jac_fn = jacrev(lambda p: jnp.mean(self.losses(apply_fn, p, batch, key)[k] ** 2))
            jac = jac_fn(params)
            ntk[k] = jnp.sum(
                jnp.array([jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(jac)])
            )
        return ntk

    def compute_frac_diff(self, apply_fn, params, t, x, u_val, u_t_fn, key=None):
        al = self.al
        method = self.method
        key = jax.random.PRNGKey(0) if key is None else key

        t = jnp.asarray(t).reshape(-1)
        x = jnp.asarray(x).reshape(-1)
        u_val = jnp.asarray(u_val).reshape(-1)

        def u_fn(tt, xx):
            inp = jnp.stack([tt, xx])
            return apply_fn(params, inp)[0]

        u_t_vmap = vmap(u_t_fn)
        dt0 = u_t_vmap(jnp.zeros_like(t), x).reshape(-1)
        dt = u_t_vmap(t, x).reshape(-1)

        if method == "GJ-I":
            return self._gj_i(t, x, dt, dt0, u_t_fn, al)
        if method == "GJ-II":
            return self._gj_ii(t, x, u_val, dt, dt0, u_fn, al)
        if method == "MC-I":
            return self._mc_i(key, t, x, dt, dt0, u_t_fn, al)
        if method == "MC-II":
            return self._mc_ii(key, t, x, u_val, dt, dt0, u_fn, al)
        raise ValueError(f"Unknown method: {method}")

    def _quadrature_grid(self, t, x, taus):
        t_col = t.reshape(-1, 1)
        x_col = x.reshape(-1, 1)
        t_tau = t_col * taus.reshape(1, -1)
        t_eval = t_col - t_tau
        x_eval = jnp.broadcast_to(x_col, t_eval.shape)
        return t_tau, t_eval, x_eval

    def _gj_i(self, t, x, dt, dt0, u_t_fn, al):
        coeff = sp_gamma(2 - al)
        t_tau, t_eval, x_eval = self._quadrature_grid(t, x, self.quad_t)
        n_pts, nums = t_eval.shape

        dttau = vmap(u_t_fn)(t_eval.ravel(), x_eval.ravel()).reshape(n_pts, nums)
        den = jnp.maximum(t_tau, 1e-10)
        integral = jnp.sum(
            ((dt.reshape(-1, 1) - dttau) / den) * self.quad_w.reshape(1, -1),
            axis=1,
        )

        safe_t = jnp.maximum(t, 1e-10)
        part1 = (al - 1.0) * (safe_t ** (2 - al)) * integral
        part2 = (dt - dt0) * (safe_t ** (1 - al))
        return (part1 + part2) / coeff

    def _gj_ii(self, t, x, u_val, dt, dt0, u_fn, al):
        coeff = sp_gamma(2 - al)
        t_tau, t_eval, x_eval = self._quadrature_grid(t, x, self.quad_t)
        n_pts, nums = t_eval.shape

        val2 = vmap(u_fn)(t_eval.ravel(), x_eval.ravel()).reshape(n_pts, nums)
        val3 = t_tau * dt.reshape(-1, 1)
        num = u_val.reshape(-1, 1) - val2 - val3
        den = jnp.maximum(t_tau ** 2, 1e-10)
        integral = jnp.sum((num / den) * self.quad_w.reshape(1, -1), axis=1)

        safe_t = jnp.maximum(t, 1e-10)
        part1 = al * (al - 1) * (safe_t ** (2 - al)) * integral
        val0 = vmap(u_fn)(jnp.zeros_like(t), x).reshape(-1)
        part2 = (al - 1) * (u_val - val0 - t * dt) / (safe_t ** al)
        part3 = (dt - dt0) / (safe_t ** (al - 1))
        return (part3 - part2 - part1) / coeff

    def _mc_taus(self, key, al):
        nums = self.config.MC.nums
        eps = self.config.MC.eps
        taus = jax.random.beta(key, 2 - al, 1, shape=(nums,))
        return eps + (1 - 2 * eps) * taus

    def _mc_i(self, key, t, x, dt, dt0, u_t_fn, al):
        nums = self.config.MC.nums
        eps = self.config.MC.eps
        coeff = sp_gamma(2 - al)
        taus = self._mc_taus(key, al)
        t_tau, t_eval, x_eval = self._quadrature_grid(t, x, taus)
        n_pts = t.shape[0]

        dttau = vmap(u_t_fn)(t_eval.ravel(), x_eval.ravel()).reshape(n_pts, nums)
        den = jnp.maximum(t_tau, eps)
        integral = jnp.mean((dt.reshape(-1, 1) - dttau) / den, axis=1)

        safe_t = jnp.maximum(t, eps)
        part1 = ((al - 1.0) / (2.0 - al)) * (safe_t ** (2 - al)) * integral
        part2 = (dt - dt0) * (safe_t ** (1 - al))
        return (part1 + part2) / coeff

    def _mc_ii(self, key, t, x, u_val, dt, dt0, u_fn, al):
        nums = self.config.MC.nums
        eps = self.config.MC.eps
        coeff = sp_gamma(2 - al)
        taus = self._mc_taus(key, al)
        t_tau, t_eval, x_eval = self._quadrature_grid(t, x, taus)
        n_pts = t.shape[0]

        val2 = vmap(u_fn)(t_eval.ravel(), x_eval.ravel()).reshape(n_pts, nums)
        val3 = t_tau * dt.reshape(-1, 1)
        den = jnp.maximum(t_tau ** 2, eps)
        integral = jnp.mean((u_val.reshape(-1, 1) - val2 - val3) / den, axis=1)

        safe_t = jnp.maximum(t, eps)
        part1 = al * (al - 1) / (2 - al) * (safe_t ** (2 - al)) * integral
        val0 = vmap(u_fn)(jnp.zeros_like(t), x).reshape(-1)
        part2 = (al - 1) * (u_val - val0 - t * dt) / (safe_t ** al)
        part3 = (dt - dt0) / (safe_t ** (al - 1))
        return (part3 - part2 - part1) / coeff

    def exact(self, datafile):
        data = np.load(datafile)
        return data["u"].reshape(-1, 1)
