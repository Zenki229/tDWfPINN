import pytest
import numpy as np
import torch
from src.physics.fractional import mitlef, roots_jacobi

def test_mittag_leffler_identity():
    # E_alpha,beta(0) = 1 / Gamma(beta)
    # If beta=1, E(0)=1
    val = mitlef(1.5, 1.0, np.array([0.0]))
    if np.ndim(val) == 0:
        assert np.isclose(val, 1.0)
    else:
        assert np.isclose(val[0], 1.0)

def test_roots_jacobi_shapes():
    n = 10
    roots, weights = roots_jacobi(n, 0, 0.5)
    assert len(roots) == n
    assert len(weights) == n

def test_fractional_operator_shapes():
    # Mocking would be needed for complex integration tests
    pass
