import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from libs.jax_pinn import create_model
from libs.jax_pde_burgers import JAXDWBurgers
from omegaconf import OmegaConf

def test_model_shape():
    key = jax.random.PRNGKey(0)
    model, params = create_model(key, 2, 10, 1, 2, "tanh")
    x = jnp.ones((5, 2))
    y = model.apply(params, x)
    assert y.shape == (5, 1)

def test_pde_residual_shape():
    # Setup config
    cfg = OmegaConf.create({
        "al": 1.5,
        "beta": 2.0,
        "tlim": [0, 1],
        "xlim": [[0, 1]],
        "method": "GJ-II",
        "GJ": {"nums": 10},
        "MC": {"nums": 10, "eps": 1e-10}
    })
    
    pde = JAXDWBurgers(cfg)
    key = jax.random.PRNGKey(0)
    model, params = create_model(key, 2, 10, 1, 2, "tanh")
    
    # Fake batch
    batch = {
        "in": jnp.ones((10, 2)),
        "bd": jnp.ones((2, 2)),
        "init": jnp.ones((2, 2))
    }
    
    losses = pde.residual(model.apply, params, batch, key)
    
    assert "in" in losses
    assert "bd" in losses
    assert "init" in losses
    assert losses["in"].shape == (10, 1)
    
if __name__ == "__main__":
    test_model_shape()
    test_pde_residual_shape()
    print("All tests passed!")
