import torch
from torch import nn

from hyperverlet.models.misc import NDenseBlock


class PendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']
        self.output_dim = model_args['output_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, self.output_dim, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, self.output_dim, **kwargs)

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hp = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hq = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        return self.model_q(hq)
