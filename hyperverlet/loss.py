import torch
from torch import nn, Tensor


class TimeDecayMSELoss(nn.Module):
    def __init__(self, decay_factor=0.99):
        super().__init__()
        self.decay_factor = decay_factor

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.view(-1, 1)
        target = target.view(-1, 1)

        squared_diff = (input - target) ** 2
        loss = torch.mean(squared_diff)
        decay_factor = torch.cumprod(torch.full_like(loss, self.decay_factor), 0)

        return torch.mean(loss * decay_factor)
