from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(features, features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class SingleAxisMLP(nn.Sequential):
    def __init__(self, input_dim, h_dim):
        super().__init__(
            nn.Linear(input_dim, h_dim),
            nn.Sigmoid(),
            Block(h_dim),
            Block(h_dim),
            Block(h_dim),
            nn.Linear(h_dim, 1)
        )


class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, output_dim, activation='relu', activate=True):
        act = {
            'silu': nn.SiLU,
            'relu': nn.ReLU
        }

        super().__init__(
            nn.Linear(input_dim, output_dim),
            act[activation]() if activate else nn.Identity()
        )


class NDenseBlock(nn.Sequential):
    def __init__(self, input_dim: int, h_dim: int, n_dense: int, *args, activate_last=True, layer_norm=False):
        layers = [DenseBlock(input_dim if i == 0 else h_dim, h_dim, activate=i < n_dense - 1 or activate_last) for i in range(n_dense)]
        layers.extend(args)

        if layer_norm:
            layers.append(nn.LayerNorm(h_dim))

        super().__init__(
            *layers
        )


class MergeNDenseBlock(nn.Module):
    def __init__(self, input_dims, h_dim: int, n_dense: int, *args, activate_last=True, layer_norm=False):
        super().__init__()

        self.num_inputs = len(input_dims)
        self.first_layer = nn.ModuleList([nn.Linear(input_dim, h_dim, bias=False) for input_dim in input_dims])
        self.bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.bias)

        self.other_layers = NDenseBlock(h_dim, h_dim, n_dense - 1, *args, activate_last=activate_last, layer_norm=layer_norm)

    def forward(self, *args):
        x = self.first_layer[0](args[0])
        for i in range(1, self.num_inputs):
            x += self.first_layer[i](args[i])
        x = F.relu(x + self.bias)
        return self.other_layers(x)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)