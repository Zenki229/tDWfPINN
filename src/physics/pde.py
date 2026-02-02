import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.utils.typing import Tensor

class PDE(nn.Module):
    """
    Base class for PDE problems.
    """
    def __init__(self):
        super().__init__()

    def residual(self, net: nn.Module, points: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def exact(self, points: Tensor) -> Tensor:
        raise NotImplementedError
