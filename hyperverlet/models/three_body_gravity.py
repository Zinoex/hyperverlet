import torch
from torch import nn

from hyperverlet.models.misc import NDenseBlock
from hyperverlet.models.symplectic import SymplecticActivation


class ThreeBodyGravityLinearModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        layers = model_args['layers']
        self.euclid = model_args['input_dim']

        self.extended = model_args['extended']
        self.power = model_args.get('power', 3)

        if self.extended:
            kwargs = dict(n_dense=2, activate_last=True, activation='tanh')
            cat_dim = model_args['cat_dim'] * 2
            h_dim = model_args['h_dim']

            self.params = nn.ModuleList([NDenseBlock(cat_dim, h_dim, self.euclid * self.euclid, **kwargs) for i in range(layers)])
            self.bq = NDenseBlock(1, h_dim, self.euclid, **kwargs)
            self.bp = NDenseBlock(1, h_dim, self.euclid, **kwargs)
        else:
            # Si is distributed N(0, 0.01), and b is set to zero.
            self.params = nn.ParameterList([nn.Parameter(torch.randn(self.euclid, self.euclid) * 0.01) for i in range(layers)])
            self.bq = nn.Parameter(torch.zeros(self.euclid))
            self.bp = nn.Parameter(torch.zeros(self.euclid))

    def m_pair(self, m):
        num_dims = m.dim()
        num_planets = m.size(-2)
        sizes = [1 for _ in range(num_dims)]
        sizes[-2] = num_planets

        # Replace with cartesian product+unfold
        m1 = m.repeat(sizes).unfold(-2, num_planets, num_planets).transpose(-1, -2)
        m2 = m.repeat_interleave(num_planets, -2).unfold(-2, num_planets, num_planets).transpose(-1, -2)

        return torch.cat([m1, m2], dim=-1)

    def forward(self, q, p, m, dt):
        num_planets = m.size(-2)
        m_pair = self.m_pair(m)

        for i, Si in enumerate(self.params):
            if self.extended:
                S = Si(m_pair)

                # Planet-based expansion
                S = S.view(-1, num_planets, num_planets, self.euclid, self.euclid)
                # Ensure local symmetry
                # S = S + S.transpose(-1, -2)
            else:
                # Local S matrix expanded to a global
                S = Si.view(1, 1, 1, self.euclid, self.euclid)
                S = S.repeat(1, num_planets, num_planets, 1, 1)

            # View as phase space
            S = S.transpose(-2, -3)
            S = S.reshape(-1, num_planets * self.euclid, num_planets * self.euclid)

            # Ensure global symmetry
            S = S + S.transpose(-2, -1)

            p = p.view(-1, num_planets * self.euclid, 1)
            q = q.view(-1, num_planets * self.euclid, 1)

            if i % 2 == 0:
                p = p + torch.matmul(S, q) * dt ** self.power
            else:
                q = q + torch.matmul(S, p) * dt ** self.power

            p = p.view(-1, num_planets, self.euclid)
            q = q.view(-1, num_planets, self.euclid)

        if self.extended:
            bq = self.bq(m).view(-1, num_planets, self.euclid)
            bp = self.bp(m).view(-1, num_planets, self.euclid)
        else:
            bq = self.bq.view(1, 1, self.euclid).repeat(1, num_planets)
            bp = self.bp.view(1, 1, self.euclid).repeat(1, num_planets)

        return q + bq * dt ** self.power, p + bp * dt ** self.power


class ThreeBodyGravityModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.model = nn.ModuleList([
            ThreeBodyGravityLinearModel(model_args),
            SymplecticActivation(model_args, 'sigmoid', 'low'),
            ThreeBodyGravityLinearModel(model_args),
            SymplecticActivation(model_args, 'sigmoid', 'up'),
            ThreeBodyGravityLinearModel(model_args),
            SymplecticActivation(model_args, 'sigmoid', 'low'),
            ThreeBodyGravityLinearModel(model_args),
            SymplecticActivation(model_args, 'sigmoid', 'up'),
        ])

    def forward(self, q, p, m, t, dt, **kwargs):
        for module in self.model:
            q, p = module(q, p, m, dt)

        return q, p
