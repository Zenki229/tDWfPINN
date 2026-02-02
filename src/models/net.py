import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Type

ACTIVATION_FN: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for PINNs.
    
    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output features.
        hidden_dim (int): Number of neurons in hidden layers.
        num_layers (int): Number of hidden layers.
        activation (str): Activation function name.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, 
                 num_layers: int, activation: str = "tanh"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        act_class = ACTIVATION_FN.get(activation.lower())
        if act_class is None:
            raise NotImplementedError(f"Activation {activation} not supported!")
            
        layers = []
        # Input layer
        layers.append(('input', nn.Linear(input_dim, hidden_dim)))
        layers.append(('input_activation', act_class()))
        
        # Hidden layers
        for i in range(num_layers):
            layers.append((f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim)))
            layers.append((f'activation_{i}', act_class()))
            
        # Output layer
        layers.append(('output', nn.Linear(hidden_dim, output_dim)))
        
        self.net = nn.Sequential(OrderedDict(layers))
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str, *args, **kwargs) -> 'MLP':
        # Note: This requires knowing init args or just loading state dict into existing model
        # For simplicity, we just load state dict if instance exists, or return dict
        # But static method usually implies creating new instance.
        # We'll leave it to the user to instantiate and load_state_dict.
        return torch.load(path)
