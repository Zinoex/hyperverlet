import torch
from torch import nn

from hyperverlet.models.utils import SingleAxisMLP


class ThreeBodySpringMassModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        self.model_q = SingleAxisMLP(self.q_input_dim, self.h_dim)
        self.model_p = SingleAxisMLP(self.p_input_dim, self.h_dim)

    def forward(self, q, p, dq, dp, m, t, dt, length, k, include_q=True, include_p=True, **kwargs):
        if len(q.size()) == 3:
            m = m.repeat(1, 1, 2)
        else:
            m = m.repeat(1, 2)

        if include_q:
            hq = torch.stack([q, dq, m], dim=-1)
            hq = self.model_q(hq).squeeze(-1)
        else:
            hq = None

        if include_p:
            hp = torch.stack([p, dp, m], dim=-1)
            hp = self.model_p(hp).squeeze(-1)
        else:
            hp = None

        return hq, hp
