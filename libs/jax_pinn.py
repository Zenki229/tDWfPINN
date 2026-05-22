import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Sequence, Callable, Dict, Optional
from functools import partial
import optax
import jax.flatten_util as fu

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.swish,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": nn.tanh,
    "sin": jnp.sin,
}


def _get_activation(name: str) -> Callable:
    if name in activation_fn:
        return activation_fn[name]
    raise NotImplementedError(f"Activation {name} not supported.")


def _cfg_get(config, name, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    try:
        return getattr(config, name)
    except (AttributeError, KeyError):
        return default


def _cfg_dict(value):
    if value is None:
        return {}
    return dict(value)


# ---- Weight factorization ----

def _weight_fact(init_fn, mean=1.0, stddev=0.1):
    def init(key, shape, dtype=jnp.float32):
        key_g, key_v = jax.random.split(key)
        w = init_fn(key_v, shape, dtype)
        v_norm = jnp.linalg.norm(w.reshape(-1)) + 1e-10
        g = jnp.exp(mean + stddev * jax.random.normal(key_g, (shape[-1],), dtype))
        g = g.reshape((1,) * (w.ndim - 1) + (-1,))
        return g * (w / v_norm)
    return init


# ---- Feature embedding modules ----

class PeriodEmbs(nn.Module):
    period: Sequence[float]
    axes: Sequence[int]
    trainable: bool = False

    @nn.compact
    def __call__(self, x):
        if self.trainable:
            period = self.param("period", lambda k, s: jnp.array(s), tuple(self.period))
        else:
            period = jnp.array(self.period)
        embs = []
        for p, a in zip(period, self.axes):
            v = 2 * jnp.pi * x[:, a:a + 1] / p
            embs.extend([jnp.cos(v), jnp.sin(v)])
        out = jnp.concatenate(embs, axis=-1)
        kept = [x[:, i:i + 1] for i in range(x.shape[-1]) if i not in self.axes]
        if kept:
            out = jnp.concatenate([out] + kept, axis=-1)
        return out


class FourierEmbs(nn.Module):
    embed_scale: float = 1.0
    embed_dim: int = 128

    @nn.compact
    def __call__(self, x):
        B = nn.Dense(features=self.embed_dim, use_bias=False,
                     kernel_init=nn.initializers.normal(stddev=self.embed_scale))
        x_proj = B(x)
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)


# ---- Dense layer with optional weight factorization ----

class Dense(nn.Module):
    features: int
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.glorot_uniform()
        if self.reparam and self.reparam.get("type") == "weight_fact":
            kernel_init = _weight_fact(kernel_init,
                                       self.reparam.get("mean", 1.0),
                                       self.reparam.get("stddev", 0.1))
        return nn.Dense(features=self.features, kernel_init=kernel_init,
                        bias_init=nn.initializers.zeros)(x)


# ---- Architectures ----

class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str = "tanh"
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        act = _get_activation(self.activation.lower())
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = act(x)
        x = nn.Dense(features=self.out_dim, kernel_init=nn.initializers.glorot_uniform(),
                     bias_init=nn.initializers.zeros)(x)
        return x


class ModifiedMlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str = "tanh"
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        act = _get_activation(self.activation.lower())
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
        u = act(Dense(features=self.hidden_dim, reparam=self.reparam)(x))
        v = act(Dense(features=self.hidden_dim, reparam=self.reparam)(x))
        for _ in range(self.num_layers - 1):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = act(x) * u + (1 - act(x)) * v
        x = nn.Dense(features=self.out_dim, kernel_init=nn.initializers.glorot_uniform(),
                     bias_init=nn.initializers.zeros)(x)
        return x


# ---- Factory functions ----

def _create_arch(config):
    arch_name = str(_cfg_get(config, 'arch_name', 'mlp')).lower()
    out_dim = _cfg_get(config, 'out_dim', _cfg_get(config, 'output_dim'))
    if out_dim is None:
        raise ValueError("Model config must define out_dim or output_dim.")
    base = dict(num_layers=config.num_layers, hidden_dim=config.hidden_dim,
                out_dim=out_dim, activation=config.activation)
    for k in ('periodicity', 'fourier_emb', 'reparam'):
        value = _cfg_get(config, k)
        if value:
            base[k] = _cfg_dict(value)
    if arch_name == "modified_mlp":
        return ModifiedMlp(**base)
    return Mlp(**base)


def create_model(key, config):
    model = _create_arch(config)
    dummy = jnp.ones((1, config.input_dim))
    params = model.init(key, dummy)
    return model, params


def _create_optimizer(config):
    optimizer_name = str(_cfg_get(config, 'optimizer', 'adam')).lower()
    lbfgs_cfg = _cfg_get(config, 'lbfgs')
    use_lbfgs = optimizer_name == 'lbfgs' or bool(_cfg_get(lbfgs_cfg, 'use', False))
    if use_lbfgs:
        lr = _cfg_get(lbfgs_cfg, 'learning_rate',
                      _cfg_get(lbfgs_cfg, 'lr', _cfg_get(config, 'learning_rate', 1e-2)))
        history_size = int(_cfg_get(lbfgs_cfg, 'history_size', 10))
        # Keep line search disabled so the optimizer works with Flax TrainState
        # and the existing stochastic pmapped training step.
        return optax.lbfgs(
            learning_rate=lr,
            memory_size=history_size,
            linesearch=None,
        )

    lr = _cfg_get(config, 'learning_rate', _cfg_get(config, 'lr', 1e-3))
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=_cfg_get(config, 'decay_steps', 5000),
        decay_rate=_cfg_get(config, 'decay_rate', 0.9))
    tx = optax.adam(learning_rate=schedule,
                    b1=_cfg_get(config, 'beta1', 0.9),
                    b2=_cfg_get(config, 'beta2', 0.999),
                    eps=_cfg_get(config, 'eps', 1e-8))
    accum = _cfg_get(config, 'grad_accum_steps', 1)
    if accum > 1:
        tx = optax.MultiSteps(tx, accum)
    return tx


# ---- Train state with adaptive weights ----

class JAXTrainState(train_state.TrainState):
    weights: Optional[Dict] = None
    momentum: float = 0.9

    def apply_weights(self, new_weights, **kwargs):
        if self.weights is None:
            return self.replace(weights=new_weights)
        updated = {}
        for k in self.weights:
            updated[k] = (self.momentum * self.weights[k] +
                          (1 - self.momentum) * jax.lax.stop_gradient(new_weights.get(k, 1.0)))
        return self.replace(weights=updated)


def create_train_state(key, model_config, optim_config, weighting_config=None):
    model, params = create_model(key, model_config)
    tx = _create_optimizer(optim_config)
    init_weights = _cfg_dict(_cfg_get(weighting_config, 'init_weights',
                                      _cfg_get(optim_config, 'init_weights', {})))
    momentum = _cfg_get(weighting_config, 'momentum',
                        _cfg_get(optim_config, 'momentum', 0.9))
    return JAXTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        weights=init_weights if init_weights else None,
        momentum=momentum)


# ---- Base PINN class (instance-method JIT for clean dispatch) ----

class PINN:
    def __init__(self, config, weighting_config=None):
        self.config = config
        self.weighting_config = weighting_config

    def u_net(self, apply_fn, params, points):
        raise NotImplementedError

    def losses(self, apply_fn, params, batch, key):
        raise NotImplementedError

    def r_net(self, apply_fn, params, points):
        raise NotImplementedError

    def compute_diag_ntk(self, apply_fn, params, batch, key):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 3))
    def loss(self, params, weights, apply_fn, batch, key):
        ls = self.losses(apply_fn, params, batch, key)
        total = 0.0
        for k, v in ls.items():
            w = weights.get(k, 1.0) if weights else 1.0
            total = total + w * jnp.mean(v ** 2)
        return total, ls

    @partial(jax.jit, static_argnums=(0, 1))
    def _grad_norms(self, apply_fn, params, batch, key):
        def loss_dict(p):
            ls = self.losses(apply_fn, p, batch, key)
            return {k: jnp.mean(v ** 2) for k, v in ls.items()}
        jac = jax.jacrev(loss_dict)(params)
        norms = {}
        for k, g in jac.items():
            flat, _ = fu.ravel_pytree(g)
            norms[k] = jnp.linalg.norm(flat)
        return norms

    @partial(jax.jit, static_argnums=(0, 1))
    def compute_weights(self, apply_fn, params, batch, key):
        scheme = _cfg_get(self.weighting_config, 'scheme', 'grad_norm')
        if scheme == "ntk":
            norms = self.compute_diag_ntk(apply_fn, params, batch, key)
        else:
            norms = self._grad_norms(apply_fn, params, batch, key)
        norms = {k: jnp.maximum(v, 1e-10) for k, v in norms.items()}
        mean_n = jnp.mean(jnp.array(list(norms.values())))
        return {k: mean_n / norms[k] for k in norms}

    def _optimizer_closure(self, state, batch, key):
        key, subkey = jax.random.split(key)

        def value_fn(p):
            loss_val, _ = self.loss(p, state.weights, state.apply_fn, batch, subkey)
            return jax.lax.pmean(loss_val, axis_name='batch')

        def value_with_aux(p):
            return self.loss(p, state.weights, state.apply_fn, batch, subkey)

        (loss_val, aux), grads = jax.value_and_grad(
            value_with_aux, has_aux=True
        )(state.params)

        grads = jax.lax.pmean(grads, axis_name='batch')
        loss_val = jax.lax.pmean(loss_val, axis_name='batch')
        return loss_val, aux, grads, value_fn, key

    def _apply_lbfgs_gradients(self, state, grads, loss_val, value_fn):
        updates, new_opt_state = state.tx.update(
            grads,
            state.opt_state,
            state.params,
            value=loss_val,
            grad=grads,
            value_fn=value_fn,
        )
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

    # pmap-wrapped training ops

    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def step(self, state, batch, key):
        loss_val, aux, grads, _, key = self._optimizer_closure(state, batch, key)
        return state.apply_gradients(grads=grads), loss_val, aux, key

    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 4))
    def lbfgs_step(self, state, batch, key, max_iter):
        for _ in range(max_iter):
            loss_val, aux, grads, value_fn, key = self._optimizer_closure(
                state, batch, key
            )
            state = self._apply_lbfgs_gradients(state, grads, loss_val, value_fn)
        return state, loss_val, aux, key

    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, key):
        new_w = self.compute_weights(state.apply_fn, state.params, batch, key)
        new_w_pmean = {k: jax.lax.pmean(v, axis_name='batch') for k, v in new_w.items()}
        return state.apply_weights(new_w_pmean)


class ForwardIVP(PINN):
    def __init__(self, config, weighting_config=None):
        super().__init__(config, weighting_config)
        wc = weighting_config
        self.use_causal = bool(_cfg_get(wc, 'use_causal', False))
        if self.use_causal:
            self.causal_tol = _cfg_get(wc, 'causal_tol', 1.0)
            self.num_chunks = _cfg_get(wc, 'num_chunks', 10)
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1)

    def res_and_w(self, apply_fn, params, batch):
        t = batch['in'][..., 0]
        idx = jnp.argsort(t)
        r = self.r_net(apply_fn, params, batch['in'][idx])
        r_chunks = r.reshape(self.num_chunks, -1)
        l = jnp.mean(r_chunks ** 2, axis=1)
        w = jnp.exp(-self.causal_tol * (self.M @ l))
        return r, w


class ForwardBVP(PINN):
    pass
