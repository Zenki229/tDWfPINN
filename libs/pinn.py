import torch.nn
from collections import OrderedDict
from torch.optim.lr_scheduler import OneCycleLR

activation_fn = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,  # SiLU is the same as Swish
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
}
def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")

class Mlp(torch.nn.Module):
    # NL: the number of hidden layers
    # NN: the number of vertices in each layer
    def __init__(self, arch):
        super(Mlp, self).__init__() # Call super constructor first
        input_dim = arch.input_dim
        output_dim = arch.output_dim
        hidden_dim = arch.hidden_dim
        num_layers = arch.num_layers
        act = _get_activation(arch.activation.lower())
        

        layers = [('input', torch.nn.Linear(input_dim, hidden_dim)), ('input_activation', act())] # Use the instantiated act
        for i in range(num_layers): # Assuming num_layers is the number of hidden layers
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_dim, hidden_dim))
            )
            layers.append(('activation_%d' % i, act())) # Use the same instantiated act
        layers.append(('output', torch.nn.Linear(hidden_dim, output_dim)))

        layer_dict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layer_dict)
        
        # Apply Xavier initialization to all linear layers
        for name, module in self.layers.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        out = self.layers(x)
        return out

    @staticmethod
    def load_model(model_path):
        model = torch.load(model_path)
        return model

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)