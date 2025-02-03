"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset, zeros
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import (add_remaining_self_loops, add_self_loops,
                                   remove_self_loops, softmax)
import torch_geometric.transforms as T


# from dgl.nn.pytorch.conv import GINConv
from torch_geometric.nn.conv import GINConv
# import dgl.function as fn
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
# from dgl.utils import expand_as_pair
# from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch_geometric.nn import global_add_pool as SumPooling
from torch_geometric.nn import global_mean_pool as AvgPooling
from torch_geometric.nn import global_max_pool as MaxPooling

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)

        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 graph_pooling_type, neighbor_pooling_type, edge_feat_dim=0,
                 final_dropout=0.0, learn_eps=False, output_dim=1, **kwargs):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """

        super().__init__()
        def init_weights_orthogonal(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
            elif isinstance(m, MLP):
                if hasattr(m, 'linears'):
                    m.linears.apply(init_weights_orthogonal)
                else:
                    m.linear.apply(init_weights_orthogonal)
            elif isinstance(m, nn.ModuleList):
                pass
            else:
                raise Exception()

        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim + edge_feat_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim + edge_feat_dim, hidden_dim, hidden_dim)
            if kwargs['init'] == 'orthogonal':
                init_weights_orthogonal(mlp)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp),
                        train_eps=True,
                        aggr=neighbor_pooling_type,
                        )
                        )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))


        if kwargs['init'] == 'orthogonal':
            print('orthogonal')
            self.linears_prediction.apply(init_weights_orthogonal)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        # h = self.preprocess_nodes(h)
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer

    def get_graph_embed(self, g, h):
        self.eval()
        with torch.no_grad():
            # return self.forward(g, h).detach().numpy()
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            # perform pooling over all nodes in each graph in every layer
            graph_embed = torch.Tensor([]).to(self.device)
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            return graph_embed

    def get_graph_embed_no_cat(self, g, h):
        self.eval()
        with torch.no_grad():
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            # perform pooling over all nodes in each graph in every layer
            # graph_embed = torch.Tensor([]).to(self.device)
            # for i, h in enumerate(hidden_rep):
            #     pooled_h = self.pool(g, h)
            #     graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            # return graph_embed
            return self.pool(g, hidden_rep[-1]).to(self.device)

    @property
    def edge_feat_loc(self):
        return self.ginlayers[0].edge_feat_loc

    @edge_feat_loc.setter
    def edge_feat_loc(self, loc):
        for layer in self.ginlayers:
            layer.edge_feat_loc = loc
