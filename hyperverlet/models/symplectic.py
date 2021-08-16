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
        self.power = model_args.get('power', 3)

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
    def __init__(self, model_args, activation, mode):
        super().__init__()
        self.act = misc.ACT[activation]()

        assert mode in ['up', 'low']
        self.mode = mode

        dim = model_args['input_dim']
        self.d = dim // 2

        self.extended = model_args['extended']
        self.power = model_args.get('power', 3)

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


class SymplecticGradient(nn.Module):
    def __init__(self, model_args, activation, mode):
        super().__init__()
        self.act = misc.ACT[activation]()

        assert mode in ['up', 'low']
        self.mode = mode

        self.width = model_args['width']
        dim = model_args['input_dim']
        self.d = dim // 2

        self.extended = model_args['extended']

        if self.extended:
            kwargs = dict(n_dense=2, activate_last=True, activation='sigmoid')
            cat_dim = model_args['cat_dim']
            h_dim = model_args['h_dim']

            self.K = NDenseBlock(cat_dim, h_dim, self.d * self.width, **kwargs)
            self.a = NDenseBlock(cat_dim, h_dim, self.d, **kwargs)
            self.b = NDenseBlock(cat_dim, h_dim, self.d, **kwargs)
        else:
            self.K = nn.Parameter(torch.randn(self.width, self.d) * 0.01)
            self.a = nn.Parameter(torch.randn(self.width) * 0.01)
            self.b = nn.Parameter(torch.zeros(self.width))

    def forward(self, q, p, cat, dt):
        if self.extended:
            K = self.K(cat)
            K = K.view(-1, self.width, self.d)
            a = self.a(cat)
            b = self.b(cat)
        else:
            K = self.K.unsqueeze(0)
            a = self.a
            b = self.b

        if self.mode == 'up':
            gradH = torch.matmul(K.transpose(-2, -1), (self.act(torch.matmul(K, q.unsqueeze(-1))[..., 0] + b) * a).unsqueeze(-1))[..., 0]
            return q, p + gradH * dt ** 3
        elif self.mode == 'low':
            gradH = torch.matmul(K.transpose(-2, -1), (self.act(torch.matmul(K, p.unsqueeze(-1))[..., 0] + b) * a).unsqueeze(-1))[..., 0]
            return q + gradH * dt ** 3, p
