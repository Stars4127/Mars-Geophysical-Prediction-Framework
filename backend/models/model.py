import torch
import torch.nn as nn

class ConfigurableANN(nn.Module):
    """
    Configurable Multi-Layer Perceptron (MLP) architecture.
    """
    def __init__(self, input_dim: int, layers: int, units: int, activation: str, dropout: float = 0.0, output_dim: int = 1):
        super().__init__()
        
        act = activation.lower()
        if act == "relu":
            act_fn = nn.ReLU()
        elif act == "tanh":
            act_fn = nn.Tanh()
        elif act == "gelu":
            act_fn = nn.GELU()
        elif act == "elu":
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()

        net = []
        prev_dim = input_dim
        for _ in range(layers):
            net.append(nn.Linear(prev_dim, units))
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            net.append(act_fn)
            prev_dim = units
            
        net.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*net)

    @property
    def model(self):
        return self.net

    @model.setter
    def model(self, value):
        self.net = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_state_dict(self, state_dict, strict=True):
        # Handle checkpoints saved with either "model." or "net." prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k.replace("model.", "net.", 1)] = v
            else:
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict=strict)
