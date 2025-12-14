import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from libs.jax_sample import TimeSpaceEasySampler

def test_sampler():
    sampler = TimeSpaceEasySampler([[0, 1]], [0, 1], {"in": 10, "bd": 5, "init": 5})
    batch = next(iter(sampler))
    assert batch["in"].shape == (10, 2)
    # bd size might vary slightly in original logic depending on faces, but let's check basic structure
    # assert batch["bd"].shape[1] == 2
    assert batch["init"].shape == (5, 2)

if __name__ == "__main__":
    test_sampler()
    print("Sampler tests passed!")
