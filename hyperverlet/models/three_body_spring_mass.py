import torch
from torch import nn

from hyperverlet.models.misc import MergeNDenseBlock

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

    def forward(self, q, p, dq, dp, m, t, dt, length, k, include_q=True, include_p=True, **kwargs):
        num_particles = q.size(-2)
        spatial_dim = q.size(-1)

        edge_index, edge_attr = self.preprocess_edges(num_particles, length, k)

        edge_attr = edge_attr.unsqueeze(1).repeat(1, spatial_dim, 1)

        if len(q.size()) == 3:
            m = m.repeat(1, 1, 2)
        else:
            m = m.repeat(1, 2)

        if include_q:
            node_attr = torch.stack([q, dq, m], dim=-1).view(-1, spatial_dim, 3)
            hq = self.model_q(node_attr, edge_attr, edge_index)

            if len(q.size()) == 3:
                hq = hq.view(-1, num_particles, spatial_dim)
            else:
                hq = hq.view(num_particles, spatial_dim)
        else:
            hq = None

        if include_p:
            node_attr = torch.stack([p, dp, m], dim=-1).view(-1, spatial_dim, 3)
            hp = self.model_p(node_attr, edge_attr, edge_index)

            if len(q.size()) == 3:
                hp = hp.view(-1, num_particles, spatial_dim)
            else:
                hp = hp.view(num_particles, spatial_dim)
        else:
            hp = None

        return hq, hp
