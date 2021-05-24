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

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hp = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hq = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        return self.model_q(hq)


class PendulumSharedModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model = NDenseBlock(self.q_input_dim, self.h_dim, 2, **kwargs)

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hx = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        hx = self.model(hx)

        return hx[..., 0], hx[..., 1]


class PostPendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hp = torch.cat([dq2, dp2, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hq = torch.cat([dq2, dp2, m, length], dim=-1)
        return self.model_q(hq)


class StatePostPendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hp = torch.cat([q, dq2, p, dp2, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hq = torch.cat([q, dq2, p, dp2, m, length], dim=-1)
        return self.model_q(hq)


class PrePostPendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hp = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hq = torch.cat([dq1, dq2, dp1, dp2, m, length], dim=-1)
        return self.model_q(hq)


class TimePostPendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 1, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 1, **kwargs)

    def forward(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(q, p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hp = torch.cat([t, dt, dq2, dp2, m, length], dim=-1)
        return self.model_p(hp)

    def hq(self, q, p, dq1, dq2, dp1, dp2, m, t, dt, length, **kwargs):
        hq = torch.cat([t, dt, dq2, dp2, length], dim=-1)
        return self.model_q(hq)
