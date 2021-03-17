from torch import nn
from torch.nn import Linear

from hyperverlet.models.meta_layer import MetaLayer
from hyperverlet.models.utils import NDenseBlock, MergeNDenseBlock, scatter_add


class GraphNetwork(nn.Module):
    def __init__(self, model_args):
        super(GraphNetwork, self).__init__()

        self.message_passing_steps = model_args['message_passing_steps']
        assert self.message_passing_steps >= 1

        self.shared = model_args['shared']

        self.graph_encoder = GraphEncoder(model_args)
        self.graph_decoder = GraphDecoder(model_args)

        if self.shared:
            self.meta_layer = MetaLayer(EdgeModel(model_args), NodeModel(model_args))
            self.meta_layers = [self.meta_layer for _ in range(self.message_passing_steps)]
        else:
            self.meta_layers = nn.ModuleList([MetaLayer(EdgeModel(model_args), NodeModel(model_args)) for _ in range(self.message_passing_steps)])

    def forward(self, graph):
        orig_v = self.graph_encoder.encode_nodes(graph.x)
        orig_e = self.graph_encoder.encode_edges(graph.edge_attr)

        v = orig_v
        e = orig_e

        for idx, layer in enumerate(self.meta_layers):
            if idx < self.message_passing_steps:
                v, e = layer(v, graph.edge_index, e)

        return self.graph_decoder(*v)


class GraphEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.euclidean_dim = model_args['euclidean_dim']

        # Node encoder settings
        encoder_args = model_args['encoder_args']
        self.h_dim = encoder_args['h_dim']
        self.n_dense = encoder_args['n_dense']
        self.activate_last = encoder_args['activate_last']
        self.layer_norm = encoder_args['layer_norm']
        self.vel_hdim = encoder_args['velocity_embedding_size']
        self.particle_type_hdim = encoder_args['particle_embedding_size']
        include_distance_to_boundary = encoder_args['include_distance_to_boundary']

        if include_distance_to_boundary:
            self.boundary_dim = self.euclidean_dim * 2
        else:
            self.boundary_dim = 0

        self.edge_encoder = NDenseBlock(self.euclidean_dim, self.h_dim, self.n_dense, activate_last=self.activate_last, layer_norm=self.layer_norm)
        self.node_encoder = NDenseBlock(self.vel_hdim + self.boundary_dim + self.particle_type_hdim, self.h_dim, self.n_dense, activate_last=self.activate_last, layer_norm=self.layer_norm)

        def forward(self, x, edge_attr):
            return self.node_encoder(x), self.edge_encoder(edge_attr)

        def encode_edges(self, edge_attr):
            return self.edge_encoder(edge_attr)

        def encode_nodes(self, x):
            return self.node_encoder(x)


class EdgeModel(nn.Module):
    def __init__(self, model_args):
        super(EdgeModel, self).__init__()

        # EdgeModel settings
        edge_model_args = model_args['edge_model_args']
        self.h_dim = edge_model_args['h_dim']
        self.n_dense = edge_model_args['n_dense']
        self.activate_last = edge_model_args['activate_last']
        self.layer_norm = edge_model_args['layer_norm']

        self.mlp = MergeNDenseBlock((self.h_dim, self.h_dim, self.h_dim), self.h_dim, self.n_dense, activate_last=self.activate_last, layer_norm=self.layer_norm)

        def forward(self, src, dest, e):
            return self.mlp(e, dest, src)


class NodeModel(nn.Module):
    def __init__(self, model_args):
        super(NodeModel, self).__init__()

        # NodeModel settings
        node_model_args = model_args['node_model_args']
        self.h_dim = node_model_args['h_dim']
        self.n_dense = node_model_args['n_dense']
        self.activate_last = node_model_args['activate_last']
        self.layer_norm = node_model_args['layer_norm']

        self.mlp = MergeNDenseBlock((self.h_dim, self.h_dim), self.h_dim, self.n_dense, activate_last=self.activate_last, layer_norm=self.layer_norm)

    def forward(self, v, edge_index, e):
        _, receiver = edge_index
        out = scatter_add(e, receiver, dim=0, dim_size=v.size(0))

        return self.mlp(out, v)


class GraphDecoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.euclidean_dim = model_args['euclidean_dim']

        # Node decoder settings
        decoder_args = model_args['decoder_args']
        self.h_dim = decoder_args['h_dim']
        self.n_dense = decoder_args['n_dense']

        self.node_decoder = NDenseBlock(self.h_dim, self.h_dim, self.n_dense, Linear(self.h_dim, self.euclidean_dim), activate_last=True)

    def forward(self, *v):
        return self.node_decoder(*v)
