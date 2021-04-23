import torch
from torch import nn

from hyperverlet.models.misc import MergeNDenseBlock, NDenseBlock

from hyperverlet.models.graph_model import GraphNetwork


class ThreeBodySpringMassModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.h_dim = model_args['h_dim']
        self.q_input_dim = model_args['q_input_dim']
        self.p_input_dim = model_args['p_input_dim']

        self.model_q = MergeNDenseBlock(self.q_input_dim, self.h_dim, 1, n_dense=5, activate_last=False, activation='prelu')
        self.model_p = MergeNDenseBlock(self.p_input_dim, self.h_dim, 1, n_dense=5, activate_last=False, activation='prelu')

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

    def forward(self, dq1, dq2, dp1, dp2, m, t, dt, length, k, **kwargs):
        num_particles = dq1.size(-2)
        spatial_dim = dq1.size(-1)

        edge_index, edge_attr = self.preprocess_edges(num_particles, length, k)

        edge_attr = edge_attr.unsqueeze(1).repeat(1, spatial_dim, 1)

        if len(dq1.size()) == 3:
            m = m.repeat(1, 1, 2)
            dt = dt.repeat(1, num_particles, 2)
        else:
            m = m.repeat(1, 2)
            dt = dt.repeat(num_particles, 2)

        node_attr = torch.stack([dq1, dq2, dp1, dp2, m], dim=-1).view(-1, spatial_dim, self.node_input_dim)
        hx = self.model_p(node_attr, edge_attr, edge_index)

        if len(dq1.size()) == 3:
            hx = hx.view(-1, num_particles, spatial_dim, 2)
        else:
            hx = hx.view(num_particles, spatial_dim, 2)

        return hx[..., 0], hx[..., 1]

        # return self.hq(q, dq, m, t, dt, **kwargs), self.hp(p, dp, m, t, dt, **kwargs)

    def hq(self, dq1, dq2, m, t, dt, **kwargs):
        return self.process(self.model_q, dq1, dq2, m, t, dt, **kwargs)

    def hp(self, dp1, dp2, m, t, dt, **kwargs):
        return self.process(self.model_p, dp1, dp2, m, t, dt, **kwargs)

    def process(self, model, dx1, dx2, m, t, dt, length, k, **kwargs):
        num_particles = dx1.size(-2)
        spatial_dim = dx1.size(-1)

        edge_index, edge_attr = self.preprocess_edges(num_particles, length, k)

        edge_attr = edge_attr.unsqueeze(1).repeat(1, spatial_dim, 1)

        if len(dx1.size()) == 3:
            m = m.repeat(1, 1, 2)
        else:
            m = m.repeat(1, 2)

        node_attr = torch.stack([dx1, dx2, m], dim=-1).view(-1, spatial_dim, self.node_input_dim)
        hx = model(node_attr, edge_attr, edge_index)

        if len(dx1.size()) == 3:
            hx = hx.view(-1, num_particles, spatial_dim)
        else:
            hx = hx.view(num_particles, spatial_dim)

        return hx
