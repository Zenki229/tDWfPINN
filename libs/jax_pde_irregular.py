import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacrev, vmap
from scipy.special import gamma as sp_gamma
from scipy.special import roots_jacobi

from libs.jax_pinn import ForwardIVP


class JAXIrregularDWBase(ForwardIVP):
    """Shared two-dimensional diffusion-wave utilities for irregular domains."""

    def __init__(self, config, weighting_config=None):
        super().__init__(config, weighting_config)
        self.al = float(config.al)
        self.tlim = config.tlim
        self.method = config.method

        if "GJ" in self.method:
            nums = config.GJ.nums
            quad_t, quad_wt = roots_jacobi(nums, 0, 1 - self.al)
            self.quad_t = jnp.array((quad_t + 1) / 2)
            self.quad_w = jnp.array(quad_wt * (1 / 2) ** (2 - self.al))

    def u_net(self, apply_fn, params, points):
        return apply_fn(params, points)

    def compute_diag_ntk(self, apply_fn, params, batch, key):
        losses = self.losses(apply_fn, params, batch, key)
        ntk = {}
        for loss_key in losses:
            jac_fn = jacrev(
                lambda p: jnp.mean(self.losses(apply_fn, p, batch, key)[loss_key] ** 2)
            )
            jac = jac_fn(params)
            ntk[loss_key] = jnp.sum(
                jnp.array([jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(jac)])
            )
        return ntk

    def _u_time_space(self, apply_fn, params):
        def u_fn(tt, xy):
            inp = jnp.concatenate([jnp.reshape(tt, (1,)), xy])
            return apply_fn(params, inp)[0]

        return u_fn

    def compute_frac_diff(self, apply_fn, params, t, coords, u_val, u_t_fn, key=None):
        al = self.al
        method = self.method
        key = jax.random.PRNGKey(0) if key is None else key

        t = jnp.asarray(t).reshape(-1)
        coords = jnp.asarray(coords).reshape((t.shape[0], -1))
        u_val = jnp.asarray(u_val).reshape(-1)
        u_fn = self._u_time_space(apply_fn, params)

        dt0 = vmap(u_t_fn)(jnp.zeros_like(t), coords).reshape(-1)
        dt = vmap(u_t_fn)(t, coords).reshape(-1)

        if method == "GJ-I":
            return self._gj_i(t, coords, dt, dt0, u_t_fn, al)
        if method == "GJ-II":
            return self._gj_ii(t, coords, u_val, dt, dt0, u_fn, al)
        if method == "MC-I":
            return self._mc_i(key, t, coords, dt, dt0, u_t_fn, al)
        if method == "MC-II":
            return self._mc_ii(key, t, coords, u_val, dt, dt0, u_fn, al)
        raise ValueError(f"Unknown method: {method}")

    def _quadrature_grid(self, t, coords, taus):
        t_col = t.reshape(-1, 1)
        t_tau = t_col * taus.reshape(1, -1)
        t_eval = t_col - t_tau
        coords_eval = jnp.broadcast_to(
            coords[:, None, :],
            (coords.shape[0], taus.shape[0], coords.shape[1]),
        )
        return t_tau, t_eval, coords_eval

    def _gj_i(self, t, coords, dt, dt0, u_t_fn, al):
        coeff = sp_gamma(2 - al)
        t_tau, t_eval, coords_eval = self._quadrature_grid(t, coords, self.quad_t)
        n_pts, nums = t_eval.shape

        dttau = vmap(u_t_fn)(
            t_eval.ravel(),
            coords_eval.reshape((n_pts * nums, coords.shape[1])),
        ).reshape(n_pts, nums)
        den = jnp.maximum(t_tau, 1e-10)
        integral = jnp.sum(
            ((dt.reshape(-1, 1) - dttau) / den) * self.quad_w.reshape(1, -1),
            axis=1,
        )

        safe_t = jnp.maximum(t, 1e-10)
        part1 = (al - 1.0) * (safe_t ** (2 - al)) * integral
        part2 = (dt - dt0) * (safe_t ** (1 - al))
        return (part1 + part2) / coeff

    def _gj_ii(self, t, coords, u_val, dt, dt0, u_fn, al):
        coeff = sp_gamma(2 - al)
        t_tau, t_eval, coords_eval = self._quadrature_grid(t, coords, self.quad_t)
        n_pts, nums = t_eval.shape

        val2 = vmap(u_fn)(
            t_eval.ravel(),
            coords_eval.reshape((n_pts * nums, coords.shape[1])),
        ).reshape(n_pts, nums)
        val3 = t_tau * dt.reshape(-1, 1)
        den = jnp.maximum(t_tau ** 2, 1e-10)
        integral = jnp.sum(
            ((u_val.reshape(-1, 1) - val2 - val3) / den)
            * self.quad_w.reshape(1, -1),
            axis=1,
        )

        safe_t = jnp.maximum(t, 1e-10)
        val0 = vmap(u_fn)(jnp.zeros_like(t), coords).reshape(-1)
        part1 = al * (al - 1) * (safe_t ** (2 - al)) * integral
        part2 = (al - 1) * (u_val - val0 - t * dt) / (safe_t ** al)
        part3 = (dt - dt0) / (safe_t ** (al - 1))
        return (part3 - part2 - part1) / coeff

    def _mc_taus(self, key, al):
        nums = self.config.MC.nums
        eps = self.config.MC.eps
        taus = jax.random.beta(key, 2 - al, 1, shape=(nums,))
        return eps + (1 - 2 * eps) * taus

    def _mc_i(self, key, t, coords, dt, dt0, u_t_fn, al):
        nums = self.config.MC.nums
        eps = self.config.MC.eps
        coeff = sp_gamma(2 - al)
        taus = self._mc_taus(key, al)
        t_tau, t_eval, coords_eval = self._quadrature_grid(t, coords, taus)
        n_pts = t.shape[0]

        dttau = vmap(u_t_fn)(
            t_eval.ravel(),
            coords_eval.reshape((n_pts * nums, coords.shape[1])),
        ).reshape(n_pts, nums)
        den = jnp.maximum(t_tau, eps)
        integral = jnp.mean((dt.reshape(-1, 1) - dttau) / den, axis=1)

        safe_t = jnp.maximum(t, eps)
        part1 = ((al - 1.0) / (2.0 - al)) * (safe_t ** (2 - al)) * integral
        part2 = (dt - dt0) * (safe_t ** (1 - al))
        return (part1 + part2) / coeff

    def _mc_ii(self, key, t, coords, u_val, dt, dt0, u_fn, al):
        nums = self.config.MC.nums
        eps = self.config.MC.eps
        coeff = sp_gamma(2 - al)
        taus = self._mc_taus(key, al)
        t_tau, t_eval, coords_eval = self._quadrature_grid(t, coords, taus)
        n_pts = t.shape[0]

        val2 = vmap(u_fn)(
            t_eval.ravel(),
            coords_eval.reshape((n_pts * nums, coords.shape[1])),
        ).reshape(n_pts, nums)
        val3 = t_tau * dt.reshape(-1, 1)
        den = jnp.maximum(t_tau ** 2, eps)
        integral = jnp.mean((u_val.reshape(-1, 1) - val2 - val3) / den, axis=1)

        safe_t = jnp.maximum(t, eps)
        val0 = vmap(u_fn)(jnp.zeros_like(t), coords).reshape(-1)
        part1 = al * (al - 1) / (2 - al) * (safe_t ** (2 - al)) * integral
        part2 = (al - 1) * (u_val - val0 - t * dt) / (safe_t ** al)
        part3 = (dt - dt0) / (safe_t ** (al - 1))
        return (part3 - part2 - part1) / coeff


class JAXIrregularHoleDW(JAXIrregularDWBase):
    """Manufactured circular-hole benchmark from irregular_2d_diffusion_wave_case.tex."""

    def __init__(self, config, weighting_config=None):
        super().__init__(config, weighting_config)
        self.center = tuple(float(v) for v in config.center)
        self.r0 = float(config.r0)
        self.lam = float(getattr(config, "lam", 1.0))
        self.diffusion_amp = float(getattr(config, "diffusion_amp", 0.3))

    def q(self, t):
        return t ** 2 * (1 - t) ** 2

    def dtalpha_q(self, t):
        al = self.al
        safe_t = jnp.maximum(t, 1e-10)
        return (
            2.0 / sp_gamma(3 - al) * safe_t ** (2 - al)
            - 12.0 / sp_gamma(4 - al) * safe_t ** (3 - al)
            + 24.0 / sp_gamma(5 - al) * safe_t ** (4 - al)
        )

    def phi_terms(self, x, y):
        cx, cy = self.center
        psi = (1 - x ** 2) * (1 - y ** 2)
        chi = (x - cx) ** 2 + (y - cy) ** 2 - self.r0 ** 2
        phi = psi * chi

        psi_x = -2 * x * (1 - y ** 2)
        psi_y = -2 * y * (1 - x ** 2)
        lap_psi = 2 * x ** 2 + 2 * y ** 2 - 4

        chi_x = 2 * (x - cx)
        chi_y = 2 * (y - cy)
        lap_chi = 4.0

        phi_x = chi * psi_x + psi * chi_x
        phi_y = chi * psi_y + psi * chi_y
        lap_phi = chi * lap_psi + 2 * (psi_x * chi_x + psi_y * chi_y) + psi * lap_chi
        return phi, phi_x, phi_y, lap_phi

    def diffusion_coeff(self, x, y):
        return 1.0 + self.diffusion_amp * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)

    def grad_diffusion_coeff(self, x, y):
        ax = self.diffusion_amp * jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
        ay = -self.diffusion_amp * jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
        return ax, ay

    def source(self, points):
        t = points[:, 0]
        x = points[:, 1]
        y = points[:, 2]

        phi, phi_x, phi_y, lap_phi = self.phi_terms(x, y)
        q = self.q(t)
        dtalpha = self.dtalpha_q(t)
        a = self.diffusion_coeff(x, y)
        ax, ay = self.grad_diffusion_coeff(x, y)
        bx = 1.0 + y
        by = x - 1.0

        return (
            dtalpha * phi
            - q * (ax * phi_x + ay * phi_y + a * lap_phi)
            + q * (bx * phi_x + by * phi_y)
            + self.lam * (q ** 3) * (phi ** 3)
        )

    def r_net(self, apply_fn, params, points, key=None):
        t = points[:, 0]
        x = points[:, 1]
        y = points[:, 2]
        coords = points[:, 1:3]

        def u_single(tt, xx, yy):
            inp = jnp.stack([tt, xx, yy])
            return apply_fn(params, inp)[0]

        u_time = self._u_time_space(apply_fn, params)
        u_t_fn = grad(u_time, 0)
        u_x_fn = grad(u_single, 1)
        u_y_fn = grad(u_single, 2)
        u_xx_fn = grad(u_x_fn, 1)
        u_yy_fn = grad(u_y_fn, 2)

        u_val = vmap(u_single)(t, x, y).reshape(-1)
        ux = vmap(u_x_fn)(t, x, y).reshape(-1)
        uy = vmap(u_y_fn)(t, x, y).reshape(-1)
        uxx = vmap(u_xx_fn)(t, x, y).reshape(-1)
        uyy = vmap(u_yy_fn)(t, x, y).reshape(-1)
        dt_frac = self.compute_frac_diff(apply_fn, params, t, coords, u_val, u_t_fn, key)

        a = self.diffusion_coeff(x, y)
        ax, ay = self.grad_diffusion_coeff(x, y)
        bx = 1.0 + y
        by = x - 1.0
        div_term = ax * ux + ay * uy + a * (uxx + uyy)
        convection = bx * ux + by * uy
        return dt_frac - div_term + convection + self.lam * (u_val ** 3) - self.source(points)

    def losses(self, apply_fn, params, batch, key):
        losses = {}
        losses["in"] = self.r_net(apply_fn, params, batch["in"], key)
        losses["bd"] = self.u_net(apply_fn, params, batch["bd"]).squeeze(-1)

        points_init = batch["init"]
        losses["init"] = self.u_net(apply_fn, params, points_init).squeeze(-1)

        u_time = self._u_time_space(apply_fn, params)
        u_t_fn = grad(u_time, 0)
        losses["init_dt"] = vmap(u_t_fn)(
            points_init[:, 0],
            points_init[:, 1:3],
        )
        return losses

    def exact(self, points):
        points = jnp.asarray(points)
        t = points[:, 0]
        x = points[:, 1]
        y = points[:, 2]
        phi, _, _, _ = self.phi_terms(x, y)
        exact = self.q(t) * phi
        return np.asarray(jax.device_get(exact)).reshape(-1, 1)


class JAXLShapeDW(JAXIrregularDWBase):
    """L-shaped-domain constant-coefficient reference-comparison PDE."""

    def __init__(self, config, weighting_config=None):
        super().__init__(config, weighting_config)
        self.diffusion_scale = float(getattr(config, "diffusion_scale", 1.0))
        self.velocity_scale = float(getattr(config, "velocity_scale", 0.2))

    def initial_profile(self, x, y):
        return (
            jnp.exp(-((x + 0.55) ** 2 + (y + 0.45) ** 2) / 0.08)
            - 0.85 * jnp.exp(-((x + 0.55) ** 2 + (y - 0.45) ** 2) / 0.06)
            + 0.60 * jnp.exp(-((x - 0.45) ** 2 + (y + 0.55) ** 2) / 0.06)
        )

    def r_net(self, apply_fn, params, points, key=None):
        t = points[:, 0]
        x = points[:, 1]
        y = points[:, 2]
        coords = points[:, 1:3]

        def u_single(tt, xx, yy):
            inp = jnp.stack([tt, xx, yy])
            return apply_fn(params, inp)[0]

        u_time = self._u_time_space(apply_fn, params)
        u_t_fn = grad(u_time, 0)
        u_x_fn = grad(u_single, 1)
        u_y_fn = grad(u_single, 2)
        u_xx_fn = grad(u_x_fn, 1)
        u_yy_fn = grad(u_y_fn, 2)

        u_val = vmap(u_single)(t, x, y).reshape(-1)
        ux = vmap(u_x_fn)(t, x, y).reshape(-1)
        uy = vmap(u_y_fn)(t, x, y).reshape(-1)
        uxx = vmap(u_xx_fn)(t, x, y).reshape(-1)
        uyy = vmap(u_yy_fn)(t, x, y).reshape(-1)
        dt_frac = self.compute_frac_diff(apply_fn, params, t, coords, u_val, u_t_fn, key)

        div_term = self.diffusion_scale * (uxx + uyy)
        return dt_frac - div_term

    def losses(self, apply_fn, params, batch, key):
        losses = {}
        losses["in"] = self.r_net(apply_fn, params, batch["in"], key)
        losses["bd"] = self.u_net(apply_fn, params, batch["bd"]).squeeze(-1)

        points_init = batch["init"]
        x0 = points_init[:, 1]
        y0 = points_init[:, 2]
        g = self.initial_profile(x0, y0)
        losses["init"] = self.u_net(apply_fn, params, points_init).squeeze(-1) - g

        u_time = self._u_time_space(apply_fn, params)
        u_t_fn = grad(u_time, 0)
        losses["init_dt"] = vmap(u_t_fn)(
            points_init[:, 0],
            points_init[:, 1:3],
        ) - self.velocity_scale * g
        return losses
