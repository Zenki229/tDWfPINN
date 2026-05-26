"""Configuration objects for empirical complexity sweeps."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class BenchConfig:
    """One benchmark setting.

    d includes the time coordinate.  For example, d=2 means (t, x).
    """

    N: int = 64
    M: int = 16
    d: int = 2
    L: int = 3
    H: int = 32
    alpha: float = 1.5
    seed: int = 1234
    dtype: str = "float64"

    def with_update(self, **kwargs) -> "BenchConfig":
        return replace(self, **kwargs)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


DEFAULT_SWEEP_VALUES = {
    "N": [16, 32, 64, 128, 256],
    "M": [4, 8, 16, 32, 64],
    "d": [1280,2560,5120,10240],
    "L": [4,8,16,32,64],
    "H": [8, 16, 24, 32, 48, 64],
}


def dtype_from_name(name: str):
    import torch

    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
