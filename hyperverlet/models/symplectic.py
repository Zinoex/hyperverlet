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

        # Si is distributed N(0, 0.01), and b is set to zero.
        self.params = nn.ParameterList([nn.Parameter(torch.randn(self.d, self.d) * 0.01) for i in range(layers)])
        self.bq = nn.Parameter(torch.zeros(self.d))
        self.bp = nn.Parameter(torch.zeros(self.d))

    def forward(self, q, p, dt):
        for i, Si in enumerate(self.params):
            S = Si.unsqueeze(0)
            S = S + S.transpose(-2, -1)

            if i % 2 == 0:
                p = p + torch.matmul(S, q.unsqueeze(-1))[..., 0] * dt ** self.power
            else:
                q = q + torch.matmul(S, p.unsqueeze(-1))[..., 0] * dt ** self.power

        return q + self.bq * dt ** self.power, p + self.bp * dt ** self.power


class SymplecticActivation(nn.Module):
    def __init__(self, model_args, mode):
        super().__init__()
        activation_function = model_args['activation']
        self.act = misc.ACT[activation_function]()

        assert mode in ['up', 'low']
        self.mode = mode

        dim = model_args['input_dim']
        self.d = dim // 2

        self.power = model_args['power']
        self.a = nn.Parameter(torch.randn(self.d) * 0.01)

    def forward(self, q, p, dt):
        if self.mode == 'up':
            return q, p + self.act(q) * self.a * dt ** self.power
        elif self.mode == 'low':
            return q + self.act(p) * self.a * dt ** self.power, p


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

    def forward(self, q, p, m, t, dt, **kwargs):

        for module in self.model:
            q, p = module(q, p, dt)

        return q, p


class SymplecticGradient(nn.Module):
    def __init__(self, model_args, activation, mode):
        super().__init__()
        self.act = misc.ACT[activation]()

        assert mode in ['up', 'low']
        self.mode = mode

        self.width = model_args['width']
        dim = model_args['input_dim']
        self.d = dim // 2

        self.power = model_args['power']
        self.K = nn.Parameter(torch.randn(self.width, self.d) * 0.01)
        self.a = nn.Parameter(torch.randn(self.width) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.width))

    def forward(self, q, p, dt):
        K = self.K.unsqueeze(0)
        a = self.a
        b = self.b

        if self.mode == 'up':
            gradH = torch.matmul(K.transpose(-2, -1), (self.act(torch.matmul(K, q.unsqueeze(-1))[..., 0] + b) * a).unsqueeze(-1))[..., 0]
            return q, p + gradH * dt ** self.power
        elif self.mode == 'low':
            gradH = torch.matmul(K.transpose(-2, -1), (self.act(torch.matmul(K, p.unsqueeze(-1))[..., 0] + b) * a).unsqueeze(-1))[..., 0]
            return q + gradH * dt ** self.power, p
