import torch
from torch import nn

from hyperverlet.models.misc import Block


class LennardJonesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        self.model_q = nn.Sequential(
            nn.Linear(12, self.h_dim),
            nn.PReLU(),
            Block(self.h_dim),
            Block(self.h_dim),
            Block(self.h_dim),
            nn.Linear(self.h_dim, 3)
        )

        self.model_p = nn.Sequential(
            nn.Linear(12, self.h_dim),
            nn.PReLU(),
            Block(self.h_dim),
            Block(self.h_dim),
            Block(self.h_dim),
            nn.Linear(self.h_dim, 3)
        )

        # TODO: Try a GNN model for interactions and permutation equivariance

    def forward(self, q, p, dq, dp, m, t, dt, **kwargs):
        hq = torch.cat([q, dq, p, dp], dim=-1)
        hq = self.model_q(hq)

        hp = torch.cat([q, dq, p, dp], dim=-1)
        hp = self.model_p(hp)

        return hq, hp
