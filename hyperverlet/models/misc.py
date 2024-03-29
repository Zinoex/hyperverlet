import math
from typing import Optional

import torch
from torch import nn


ACT = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'identity': nn.Identity
}


class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, output_dim, activation='tanh'):
        super().__init__(
            nn.Linear(input_dim, output_dim, bias=True),
            ACT[activation]()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self[0].weight, a=math.sqrt(5))
        if self[0].bias is not None:
            nn.init.zeros_(self[0].bias)


class NDenseBlock(nn.Sequential):
    def __init__(self, input_dim: int, h_dim: int, output_dim: int, n_dense: int, activate_last=True, activation='tanh'):
        def idim(i):
            return input_dim if i == 0 else h_dim

        def odim(i):
            return output_dim if i == n_dense - 1 else h_dim

        def act(i):
            return activation if i < n_dense - 1 or activate_last else 'identity'

        layers = [DenseBlock(idim(i), odim(i), activation=act(i)) for i in range(n_dense)]

        super().__init__(
            *layers
        )


class MergeNDenseBlock(nn.Module):
    def __init__(self, input_dims, h_dim: int, output_dim: int, n_dense: int, activate_last=True, activation='tanh'):
        super().__init__()

        self.num_inputs = len(input_dims)
        self.first_layer = nn.ModuleList([nn.Linear(input_dim, h_dim, bias=False) for input_dim in input_dims])
        self.bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.bias)
        self.activation = ACT[activation]()

        self.other_layers = NDenseBlock(h_dim, h_dim, output_dim, n_dense - 1, activate_last=activate_last, activation=activation)

    def forward(self, *args):
        x = self.first_layer[0](args[0])
        for i in range(1, self.num_inputs):
            x += self.first_layer[i](args[i])
        x = self.activation(x + self.bias)
        return self.other_layers(x)
