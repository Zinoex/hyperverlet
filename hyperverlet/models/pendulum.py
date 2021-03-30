import torch
from torch import nn

from hyperverlet.models.misc import SingleAxisMLP


class PendulumModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        self.model_q = SingleAxisMLP(self.q_input_dim, self.h_dim)
        self.model_p = SingleAxisMLP(self.p_input_dim, self.h_dim)

    def forward(self, q, p, dq, dp, m, t, dt, length, include_q=True, include_p=True, **kwargs):
        if include_q:
            hq = torch.cat([q, dq, m, length], dim=-1)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([p, dp, m, length], dim=-1)
            hp = self.model_p(hp)
        else:
            hp = None

        return hq, hp