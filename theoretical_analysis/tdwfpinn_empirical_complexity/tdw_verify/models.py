"""Small no-bias MLP used for empirical complexity sweeps.

The fractional operators in the new-dev branch only require a callable
``torch.nn.Module``.  We use a no-bias MLP so that the measured dense-layer
structure matches the paper's ``A_mac = H d + (L-1) H^2 + H`` convention.
"""
from __future__ import annotations

import torch
from torch import nn


class NoBiasMLP(nn.Module):
    """Fully connected tanh MLP without bias terms.

    Args:
        input_dim: Total input dimension d, including time.
        hidden_dim: Width H.
        num_hidden_layers: Number of hidden layers L.
        output_dim: Output dimension, normally 1 for PINNs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int = 1):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
