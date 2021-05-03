import torch
from torch import nn

from hyperverlet.models.misc import NDenseBlock


class PendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq, dp, m, t, dt, **kwargs):
        return self.hq(q, dq, p, dp, m, t, dt, **kwargs), self.hp(q, dq, p, dp, m, t, dt, **kwargs)

    def hp(self, q, dq, p, dp, m, t, dt, length, **kwargs):
        hp = torch.cat([p, dp, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, q, dq, p, dp, m, t, dt, length, **kwargs):
        hq = torch.cat([q, dq, m, length], dim=-1)
        return self.model_q(hq)