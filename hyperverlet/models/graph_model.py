import torch
from torch import nn
from torch.nn import Linear

from hyperverlet.models.meta_layer import MetaLayer
from hyperverlet.models.misc import NDenseBlock, MergeNDenseBlock, scatter_add, scatter_mean


class GraphNetwork(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.message_passing_steps = model_args['message_passing_steps']
        assert self.message_passing_steps >= 1

        self.shared = model_args['shared']

        self.graph_encoder = GraphEncoder(model_args['encoder_args'])
        self.graph_decoder = GraphDecoder(model_args['decoder_args'])

        if self.shared:
            self.meta_layer = MetaLayer(EdgeModel(model_args['edge_model']), NodeModel(model_args['node_model']))
            self.meta_layers = [self.meta_layer for _ in range(self.message_passing_steps)]
        else:
            self.meta_layers = nn.ModuleList([
                MetaLayer(EdgeModel(model_args['edge_model']), NodeModel(model_args['node_model']))
                for _ in range(self.message_passing_steps)
            ])

    def forward(self, node_attr, edge_attr, edge_index):
        orig_v = self.graph_encoder.encode_nodes(node_attr)
        orig_e = self.graph_encoder.encode_edges(edge_attr)

        v, e = orig_v, orig_e
        prev_v, prev_e = v, e

        for idx, layer in enumerate(self.meta_layers):
            v, e = layer(v, edge_index, e)
            v, e = v + prev_v, e + prev_e
            prev_v, prev_e = v, e

        return self.graph_decoder(v)


class EdgeModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.h_dim = model_args['h_dim']
        self.n_dense = model_args['n_dense']
        self.activate_last = model_args['activate_last']
        self.layer_norm = model_args['layer_norm']

        self.mlp = NDenseBlock(3 * self.h_dim, self.h_dim, self.h_dim, self.n_dense, activate_last=self.activate_last)

    def forward(self, src, dest, e):
        return self.mlp(torch.cat([e, dest, src], dim=-1))


class NodeModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.h_dim = model_args['h_dim']
        self.n_dense = model_args['n_dense']
        self.activate_last = model_args['activate_last']
        self.layer_norm = model_args['layer_norm']

        self.mlp = NDenseBlock(2 * self.h_dim, self.h_dim, self.h_dim, self.n_dense, activate_last=self.activate_last)

    def forward(self, v, edge_index, e):
        _, receiver = edge_index
        out = scatter_mean(e, receiver, dim=0, dim_size=v.size(0))

        return self.mlp(torch.cat([out, v], dim=-1))


class GraphEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.h_dim = model_args['h_dim']
        self.n_dense = model_args['n_dense']
        self.activate_last = model_args['activate_last']
        self.layer_norm = model_args['layer_norm']
        self.node_input_dim = model_args['node_input_dim']
        self.edge_input_dim = model_args['edge_input_dim']

        self.node_encoder = NDenseBlock(self.node_input_dim, self.h_dim, self.h_dim, self.n_dense, activate_last=self.activate_last)
        self.edge_encoder = NDenseBlock(self.edge_input_dim, self.h_dim, self.h_dim, self.n_dense, activate_last=self.activate_last)

    def forward(self, node_attr, edge_attr):
        return self.node_encoder(node_attr), self.edge_encoder(edge_attr)

    def encode_nodes(self, node_attr):
        return self.node_encoder(node_attr)

    def encode_edges(self, edge_attr):
        return self.edge_encoder(edge_attr)


class GraphDecoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.h_dim = model_args['h_dim']
        self.n_dense = model_args['n_dense']
        self.node_output_dim = model_args['node_output_dim']

        self.node_decoder = NDenseBlock(self.h_dim, self.h_dim, self.node_output_dim, self.n_dense, activate_last=False)

    def forward(self, v):
        return self.node_decoder(v)
