import torch
from torch import nn

from hyperverlet.models.utils import SingleAxisMLP


class SpringMassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        self.model_q = SingleAxisMLP(5, self.h_dim)
        self.model_p = SingleAxisMLP(5, self.h_dim)

    def forward(self, q, p, dq, dp, m, t, dt, length, k, include_q=True, include_p=True, **kwargs):
        if include_q:
            hq = torch.cat([q, dq, m, length, k], dim=-1)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([p, dp, m, length, k], dim=-1)
            hp = self.model_p(hp)
        else:
            hp = None

        return hq, hp
