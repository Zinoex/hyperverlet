import torch
from torch import nn, Tensor


class TimeDecayMSELoss(nn.Module):
    def __init__(self, decay_factor=0.99):
        super().__init__()
        self.decay_factor = decay_factor

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        size = [input.size(0), -1]
        input = input.view(*size)
        target = target.view(*size)

        squared_diff = (input - target) ** 2
        loss = torch.mean(squared_diff, dim=1)
        decay_factor = torch.cumprod(torch.full_like(loss, self.decay_factor), 0)

        return torch.mean(loss * decay_factor)


class MeanNormLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        size = [input.size(0), input.size(1), -1]
        input = input.view(*size)
        target = target.view(*size)

        diff = target - input
        loss = torch.norm(diff, dim=2)

        return torch.mean(loss)
