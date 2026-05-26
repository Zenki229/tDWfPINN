"""Point generation for fractional residual benchmarks."""
from __future__ import annotations

import torch


def make_domain_points(N: int, d: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Generate interior points with positive time.

    Time is kept away from zero to avoid singular denominators in Type-II.
    Spatial coordinates are random in [-1, 1].
    """
    if d < 2:
        raise ValueError("d must include time plus at least one spatial coordinate, so d >= 2")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    t = 0.2 + 0.8 * torch.rand((N, 1), generator=gen, dtype=dtype)
    x = -1.0 + 2.0 * torch.rand((N, d - 1), generator=gen, dtype=dtype)
    pts = torch.cat([t, x], dim=1).to(device=device)
    return pts
