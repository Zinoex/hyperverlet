import torch
from torch import nn

from hyperverlet.models.misc import NDenseBlock
from hyperverlet.models.symplectic import LASymplecticModel


class SpringMassModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs):
        hp = torch.cat([dq1, dq2, dp1, dp2, m, length, k], dim=-1)
        return self.model_p(hp)

    def hq(self, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs):
        hq = torch.cat([dq1, dq2, dp1, dp2, m, length, k], dim=-1)
        return self.model_q(hq)


class SymplecticSpringMassModel(LASymplecticModel):
    def forward(self,  q, p, m, t, dt, length, k, **kwargs):
        cat = torch.cat([m, length, k], dim=-1)
        return super(SymplecticSpringMassModel, self).forward(q, p, cat, m, t, dt)
