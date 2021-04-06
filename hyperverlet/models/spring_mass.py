import torch
from torch import nn

from hyperverlet.models.misc import MergeNDenseBlock, NDenseBlock


class SpringMassModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq, dp, m, t, dt, **kwargs):
        return self.hq(q, dq, m, t, dt, **kwargs), self.hp(p, dp, m, t, dt, **kwargs)

    def hp(self, p, dp,  m, t, dt, length, k, **kwargs):
        hp = torch.cat([p, dp, m, length, k], dim=-1)
        return self.model_p(hp)

    def hq(self, q, dq,  m, t, dt, length, k, **kwargs):
        hq = torch.cat([q, dq, m, length, k], dim=-1)
        return self.model_q(hq)
