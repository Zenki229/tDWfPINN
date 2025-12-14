import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable, Any

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": nn.tanh,
}

def _get_activation(name: str) -> Callable:
    if name in activation_fn:
        return activation_fn[name]
    else:
        raise NotImplementedError(f"Activation {name} not supported yet!")

class Mlp(nn.Module):
    features: Sequence[int]
    activation_name: str = "tanh"
    
    @nn.compact
    def __call__(self, x):
        act = _get_activation(self.activation_name.lower())
        
        # Hidden layers
        for feat in self.features[:-1]:
            x = nn.Dense(
                features=feat,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros
            )(x)
            x = act(x)
            
        # Output layer
        x = nn.Dense(
            features=self.features[-1],
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros
        )(x)
        return x

def create_model(key, input_dim, hidden_dim, output_dim, num_layers, activation):
    # Construct features list: [hidden, hidden, ..., output]
    # Note: Input dim is implicit in Flax
    features = [hidden_dim] * num_layers + [output_dim]
    model = Mlp(features=features, activation_name=activation)
    dummy_input = jnp.ones((1, input_dim))
    params = model.init(key, dummy_input)
    return model, params
