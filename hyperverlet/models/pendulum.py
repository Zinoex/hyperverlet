import torch
from torch import nn

from hyperverlet.models.misc import MergeNDenseBlock


class PendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = MergeNDenseBlock(torch.ones(self.q_input_dim, dtype=torch.int).tolist(), self.h_dim, 1, **kwargs)
        self.model_p = MergeNDenseBlock(torch.ones(self.p_input_dim, dtype=torch.int).tolist(), self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq, dp, m, t, dt, length, **kwargs):
        hq = self.model_q(q, dq, m, length)
        hp = self.model_p(p, dp, m, length)

        return hq, hp