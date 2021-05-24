import torch
from torch import nn

from hyperverlet.models.misc import NDenseBlock

from hyperverlet.models.graph_model import GraphNetwork


class ThreeBodySpringMassModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        kwargs = dict(n_dense=5, activate_last=False, activation='sigmoid')

        self.model_q = NDenseBlock(self.q_input_dim, self.h_dim, 2, **kwargs)
        self.model_p = NDenseBlock(self.p_input_dim, self.h_dim, 2, **kwargs)

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs):
        return self.process(self.model_q, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs), self.process(self.model_p, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs)

    def process(self, model, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs):
        num_particles = dq1.size(-2)
        spatial_dim = dq1.size(-1)

        hx = torch.stack([dq1, dq2, dp1, dp2], dim=-1).view(-1, 4 * spatial_dim)
        hx = torch.cat([hx, m.view(-1, 1)], dim=-1)
        hx = model(hx)

        if len(dq1.size()) == 3:
            hx = hx.view(-1, num_particles, spatial_dim)
        else:
            hx = hx.view(num_particles, spatial_dim)

        return hx


class ThreeBodySpringMassGraphModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.node_input_dim = model_args['encoder_args']['node_input_dim']
        self.model_q = GraphNetwork(model_args)
        self.model_p = GraphNetwork(model_args)

    def fully_connected(self, batch_size, num_particles, device):
        particle_ids = torch.arange(num_particles, device=device)

        senders = particle_ids.repeat(num_particles)
        receivers = particle_ids.repeat_interleave(num_particles)

        if batch_size is not None:
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_particles ** 2) * num_particles
            senders = senders.repeat(batch_size) + batch_idx
            receivers = receivers.repeat(batch_size) + batch_idx

        return senders, receivers

    def preprocess_edges(self, num_particles, length, k):
        """
        IMPORTANT: The edge_index returned as the first element of the tuple assumes the following axis order:
        [batch_size * particles] x 2, (sender, receiver) in either a tuple or a tensor with size 2 along the first axis.

        Reason: To be able to split sender and receiver in the graph network layer.
        """
        batch_size = length.size(0) if len(length.size()) == 3 else None

        return self.fully_connected(batch_size, num_particles, length.device), torch.stack([length, k], dim=-1).view(-1, 2)

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.hq(dq1, dq2, dp1, dp2, m, t, dt, **kwargs), self.hp(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hq(self, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.process(self.model_q, dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def hp(self, dq1, dq2, dp1, dp2, m, t, dt, **kwargs):
        return self.process(self.model_p, dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

    def process(self, model, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs):
        num_particles = dq1.size(-2)
        spatial_dim = dq1.size(-1)

        edge_index, edge_attr = self.preprocess_edges(num_particles, length, k)

        node_attr = torch.stack([dq1, dq2, dp1, dp2], dim=-1).view(-1, 4 * spatial_dim)
        node_attr = torch.cat([node_attr, m.view(-1, 1)], dim=-1)
        hx = model(node_attr, edge_attr, edge_index)

        if len(dq1.size()) == 3:
            hx = hx.view(-1, num_particles, spatial_dim)
        else:
            hx = hx.view(num_particles, spatial_dim)

        return hx
