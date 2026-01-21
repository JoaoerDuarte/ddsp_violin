import torch.nn as nn


def MLP(input_size, hidden_size, num_layers):
    """Creates a multi-layer perceptron with LayerNorm and LeakyReLU activations."""
    layer_sizes = [input_size] + (num_layers) * [hidden_size]
    layers = []
    
    for i in range(num_layers):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.LayerNorm(layer_sizes[i + 1]))
        layers.append(nn.LeakyReLU())
    
    return nn.Sequential(*layers)


def GRU(input_features, hidden_size, multiply_input_size=False, bidirectional=False):
    """Creates a GRU layer with optional input size multiplication."""
    actual_input_size = input_features * hidden_size if multiply_input_size else input_features
    
    return nn.GRU(
        actual_input_size, 
        hidden_size, 
        batch_first=True, 
        bidirectional=bidirectional
    )


class Normalize(nn.Module):
    """Instance normalization layer that handles different input dimensions."""
    
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_channels, eps=1e-5)
    
    def forward(self, x):
        original_shape = x.shape
        needs_transpose = len(x.shape) == 3 and original_shape[2] == self.norm.num_features
        
        if needs_transpose:
            x = x.transpose(1, 2)
            
        x = self.norm(x)
        
        if needs_transpose:
            x = x.transpose(1, 2)
            
        return x