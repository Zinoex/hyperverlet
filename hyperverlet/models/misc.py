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
        # nn.init.kaiming_uniform_(self[0].weight, a=math.sqrt(5))
        # nn.init.xavier_normal_(self[0].weight)
        # nn.init.orthogonal_(self[0].weight)

        if self[0].bias is not None:
            # nn.init.ones_(self[0].bias)
            nn.init.zeros_(self[0].bias)

            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self[0].weight)
            # bound = 1 / math.sqrt(fan_in)
            # nn.init.uniform_(self[0].bias, -bound, bound)


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


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:

    out = scatter_add(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_add(ones, index, index_dim, None, dim_size)
    count.clamp_(min=1)
    count = broadcast(count, out, dim)
    if torch.is_floating_point(out):
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out
