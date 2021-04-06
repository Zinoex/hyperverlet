import torch
from torch import nn

from hyperverlet.models.misc import MergeNDenseBlock


class LennardJonesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        kwargs = dict(n_dense=5, activate_last=False, activation='prelu')

        self.model_q = MergeNDenseBlock(torch.ones(4, dtype=torch.int).tolist(), self.h_dim, 3, **kwargs)
        self.model_p = MergeNDenseBlock(torch.ones(4, dtype=torch.int).tolist(), self.h_dim, 3, **kwargs)

        # TODO: Try a GNN model for interactions and permutation equivariance

    def forward(self, q, p, dq, dp, m, t, dt, **kwargs):
        hq = self.model_q(q, dq, p, dp)

        hp = self.model_p(q, dq, p, dp)

        return hq, hp
