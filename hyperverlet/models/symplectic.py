import torch
from torch import nn

from hyperverlet.models import misc
from hyperverlet.models.misc import NDenseBlock


class SymplecticLinear(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        layers = model_args['layers']
        dim = model_args['input_dim']
        self.d = dim // 2

        self.extended = model_args['extended']
        self.power = model_args['power']

        if self.extended:
            kwargs = dict(n_dense=2, activate_last=True, activation='tanh')
            cat_dim = model_args['cat_dim']
            h_dim = model_args['h_dim']

            self.params = nn.ModuleList([NDenseBlock(cat_dim, h_dim, self.d * self.d, **kwargs) for i in range(layers)])
            self.bq = NDenseBlock(cat_dim, h_dim, self.d, **kwargs)
            self.bp = NDenseBlock(cat_dim, h_dim, self.d, **kwargs)
        else:
            # Si is distributed N(0, 0.01), and b is set to zero.
            self.params = nn.ParameterList([nn.Parameter(torch.randn(self.d, self.d) * 0.01) for i in range(layers)])
            self.bq = nn.Parameter(torch.zeros(self.d))
            self.bp = nn.Parameter(torch.zeros(self.d))

    def forward(self, q, p, cat, dt):
        for i, Si in enumerate(self.params):
            if self.extended:
                S = Si(cat)
                S = S.view(-1, self.d, self.d)
            else:
                S = Si.unsqueeze(0)

            S = S + S.transpose(-2, -1)

            if i % 2 == 0:
                p = p + torch.matmul(S, q.unsqueeze(-1))[..., 0] * dt ** self.power
            else:
                q = q + torch.matmul(S, p.unsqueeze(-1))[..., 0] * dt ** self.power

        if self.extended:
            bq = self.bq(cat)
            bp = self.bp(cat)
        else:
            bq = self.bq
            bp = self.bp

        return q + bq * dt ** self.power, p + bp * dt ** self.power


class SymplecticActivation(nn.Module):
    def __init__(self, model_args, mode):
        super().__init__()
        activation_function = model_args['activation']
        self.act = misc.ACT[activation_function]()

        assert mode in ['up', 'low']
        self.mode = mode

        dim = model_args['input_dim']
        self.d = dim // 2

        self.extended = model_args['extended']
        self.power = model_args['power']

        if self.extended:
            kwargs = dict(n_dense=2, activate_last=True, activation='tanh')
            cat_dim = model_args['cat_dim']
            h_dim = model_args['h_dim']

            self.a = NDenseBlock(cat_dim, h_dim, self.d, **kwargs)
        else:
            self.a = nn.Parameter(torch.randn(self.d) * 0.01)

    def forward(self, q, p, cat, dt):
        if self.extended:
            a = self.a(cat)
        else:
            a = self.a

        if self.mode == 'up':
            return q, p + self.act(q) * a * dt ** self.power
        elif self.mode == 'low':
            return q + self.act(p) * a * dt ** self.power, p


class LASymplecticModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        layers = []
        repeats = model_args['repeats']

        for _ in range(repeats):
            layers.extend([
                SymplecticLinear(model_args),
                SymplecticActivation(model_args, 'low'),
                SymplecticLinear(model_args),
                SymplecticActivation(model_args, 'up')
            ])

        self.model = nn.ModuleList(layers)

    def forward(self, q, p, cat, m, t, dt):
        for module in self.model:
            q, p = module(q, p, cat, dt)

        return q, p
