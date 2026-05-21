import jax
import jax.numpy as jnp

from libs.jax_utils import flatten_pytree


class BaseEvaluator:
    """Collects host-side metrics during training."""

    def __init__(self, config, pde):
        self.config = config
        self.pde = pde
        self.log_dict = {}

    def log_losses(self, params, apply_fn, batch, key):
        ls = self.pde.losses(apply_fn, params, batch, key)
        self.log_dict.update({f"{k}_loss": jnp.mean(v ** 2) for k, v in ls.items()})

    def log_grads(self, params, apply_fn, batch, key):
        """Log per-loss gradient norms."""

        def per_loss(p, loss_key):
            ls = self.pde.losses(apply_fn, p, batch, key)
            return jnp.mean(ls[loss_key] ** 2)

        ls = self.pde.losses(apply_fn, params, batch, key)
        for k in ls:
            jac_fn = jax.jacrev(lambda p: per_loss(p, k))
            jac = jac_fn(params)
            flat = flatten_pytree(jac)
            self.log_dict[f"{k}_grad_norm"] = jnp.linalg.norm(flat)

    def log_weights(self, state):
        if state.weights is None:
            return
        for k, v in state.weights.items():
            self.log_dict[f"{k}_weight"] = v

    def log_ntk(self, params, apply_fn, batch, key):
        ntk = self.pde.compute_diag_ntk(apply_fn, params, batch, key)
        self.log_dict.update({f"{k}_ntk": v for k, v in ntk.items()})

    def log_l2_error(self, params, apply_fn, u_exact_fn, points):
        """Log L2 error against reference solution. u_exact_fn(points) -> (N, 1)."""
        u_pred = self.pde.u_net(apply_fn, params, points)
        u_exact = u_exact_fn(points)
        err = jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
        rel_err = err / (jnp.sqrt(jnp.mean(u_exact ** 2)) + 1e-10)
        self.log_dict["l2_error"] = err
        self.log_dict["rel_l2_error"] = rel_err

    def __call__(self, state, batch, key, step=None):
        """Main evaluation entry point. Returns Python scalar metrics."""
        self.log_dict = {}
        log_cfg = getattr(self.config, "logging", None)
        if log_cfg is None:
            self.log_losses(state.params, state.apply_fn, batch, key)
        else:
            if getattr(log_cfg, "log_losses", True):
                self.log_losses(state.params, state.apply_fn, batch, key)
            if getattr(log_cfg, "log_weights", True):
                self.log_weights(state)
            if getattr(log_cfg, "log_grads", False):
                self.log_grads(state.params, state.apply_fn, batch, key)
            if getattr(log_cfg, "log_ntk", False):
                self.log_ntk(state.params, state.apply_fn, batch, key)
        return {k: float(v) if hasattr(v, "item") else v for k, v in self.log_dict.items()}
