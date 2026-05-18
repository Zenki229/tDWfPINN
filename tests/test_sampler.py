import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from libs.jax_sample import TimeSpaceEasySampler


def test_sampler():
    sampler = TimeSpaceEasySampler([[0, 1]], [0, 1], {"in": 10, "bd": 5, "init": 5})
    batch = next(iter(sampler))
    assert batch["in"].shape == (10, 2)
    assert batch["init"].shape == (5, 2)
    assert batch["bd"].shape[1] == 2
    print("[PASS] test_sampler")


def test_sampler_multi_device():
    sampler = TimeSpaceEasySampler(
        [[0, 1]], [0, 1],
        {"in": 8, "bd": 4, "init": 4},
        n_devices=2,
    )
    batch = next(iter(sampler))
    assert batch["in"].shape == (2, 4, 2), f"Expected (2,4,2), got {batch['in'].shape}"
    assert batch["init"].shape == (2, 2, 2)
    print("[PASS] test_sampler_multi_device")


def test_sampler_single_device_shard_for_pmap():
    sampler = TimeSpaceEasySampler(
        [[0, 1]], [0, 1],
        {"in": 8, "bd": 4, "init": 4},
        n_devices=1,
        shard=True,
    )
    batch = next(iter(sampler))
    assert batch["in"].shape == (1, 8, 2)
    assert batch["bd"].shape == (1, 4, 2)
    assert batch["init"].shape == (1, 4, 2)
    print("[PASS] test_sampler_single_device_shard_for_pmap")


if __name__ == "__main__":
    test_sampler()
    test_sampler_multi_device()
    test_sampler_single_device_shard_for_pmap()
    print("Sampler tests passed!")
