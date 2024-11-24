# +
# Graph Neural Network architecture
# -

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleList, GELU, Softplus, Dropout
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from Source.constants import *
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add

# Model for updating edge attritbutes
class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels,dropout_rate, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in*2 + edge_in, hid_channels),
                  GELU(),
                  Dropout(dropout_rate),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out

# Model for updating node attritbutes
class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels,dropout_rate, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + 3*edge_out + 1, hid_channels),
                  GELU(),
                  Dropout(dropout_rate),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)
        if self.residuals:
            out = out + x
        return out
    

# First edge model for updating edge attritbutes when no initial node features are provided
class EdgeModelIn(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels,dropout_rate, norm=False):
        super().__init__()

        self.norm = norm

        layers = [Linear(edge_in, hid_channels),
                  GELU(),
                  Dropout(dropout_rate),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):

        out = self.edge_mlp(edge_attr)

        return out

# First node model for updating node attritbutes when no initial node features are provided
class NodeModelIn(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels,dropout_rate, norm=False):
        super().__init__()

        self.norm = norm

        layers = [Linear(3*edge_out + 1, hid_channels),
                  GELU(),
                  Dropout(dropout_rate),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):

        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)

        return out

# Graph Neural Network architecture, based on the Graph Network (arXiv:1806.01261)
# Employing the MetaLayer implementation in Pytorch-Geometric

class GNN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, linkradius, dim_out,dropout_rate, alpha, residuals = True):
        super().__init__()

        self.n_layers = n_layers
        self.link_r = linkradius
        self.dim_out = dim_out
        self.alpha = alpha

        # Number of input node features 
        node_in = node_features
        
        # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        edge_in = 3
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels

        layers = []

        # Encoder graph block
        inlayer = MetaLayer(
            node_model=NodeModel(node_in,
                                   node_out,
                                   edge_in,
                                   edge_out,
                                   hid_channels,
                                 residuals=False,
                                   norm = True,
                                  dropout_rate = dropout_rate),
            edge_model=EdgeModel(node_in,
                                   node_out,
                                   edge_in,
                                   edge_out,
                                   hid_channels,
                                   norm = True,
                                 residuals=False,
                                   dropout_rate = dropout_rate)
            )

        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Hidden graph blocks
        for _ in range(n_layers-1):

            lay = MetaLayer(
                node_model=NodeModel(node_in,
                                     node_out,
                                     edge_in,
                                     edge_out,
                                     hid_channels,
                                     residuals=residuals,
                                     norm =True,
                                     dropout_rate = dropout_rate),
                edge_model=EdgeModel(node_in,
                                     node_out,
                                     edge_in,
                                     edge_out,
                                     hid_channels,
                                     residuals=residuals,
                                     norm = True,
                                     dropout_rate = dropout_rate)
                )

            layers.append(lay)

        # holding submodules in a list
        self.layers = ModuleList(layers)

        # Final aggregation layer 
        self.outlayer = Sequential(
            Linear(3*node_out+1, hid_channels),
            GELU(),
            Dropout(dropout_rate),
            Linear(hid_channels, hid_channels),
            GELU(),
            Dropout(dropout_rate),
            Linear(hid_channels, hid_channels),
            GELU(),
            Dropout(dropout_rate),
            Linear(hid_channels, self.dim_out),
            Softplus()
            )

    def forward(self, data):

        # Retrieving data
        h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u

        # Message passing layers
        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

        # Multipooling layer
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        out = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        # Final linear layer
        # alpha = 0.85     # variance factor to help training
        out = self.outlayer(out)
        out[:, 1] = self.alpha * out[:, 1]
    
        return out

############################################################################################################################
# import torch
# import torch.nn.functional as F
# from torch.nn import Sequential, Linear, ReLU, ModuleList, GELU, Softplus, Dropout
# from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm
# from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
# from Source.constants import *
# from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
# from torch.nn.init import kaiming_normal_, zeros_

# # Model for updating edge attributes
# class EdgeModel(torch.nn.Module):
#     def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, dropout_rate, residuals=True, norm=False):
#         super().__init__()

#         self.residuals = residuals
#         self.norm = norm

#         layers = [Linear(node_in * 2 + edge_in, hid_channels),
#                   GELU(),
#                   Dropout(dropout_rate),
#                   Linear(hid_channels, edge_out)]
#         if self.norm:  
#             layers.append(LayerNorm(edge_out))

#         self.edge_mlp = Sequential(*layers)
        
#         # Apply Kaiming initialization for weights and zero initialization for biases
#         for m in self.edge_mlp:
#             if isinstance(m, Linear):
#                 kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization
#                 if m.bias is not None:
#                     zeros_(m.bias)  # Initialize biases to zero

#     def forward(self, src, dest, edge_attr, u, batch):
#         print(src.shape)
#         print(dest.shape)
#         print(edge_attr.shape)
#         out = torch.cat([src, dest, edge_attr], dim=1)
#         out = self.edge_mlp(out)
      
#         if self.residuals:
#             out = out + edge_attr
#         return out


# # Model for updating node attributes
# class NodeModel(torch.nn.Module):
#     def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, dropout_rate, residuals=True, norm=False):
#         super().__init__()

#         self.residuals = residuals
#         self.norm = norm

#         layers = [Linear(node_in + 3 * edge_out + 1, hid_channels),
#                   GELU(),
#                   Dropout(dropout_rate),
#                   Linear(hid_channels, node_out)]
#         if self.norm:  
#             layers.append(LayerNorm(node_out))

#         self.node_mlp = Sequential(*layers)

#         # Apply Kaiming initialization for weights and zero initialization for biases
#         for m in self.node_mlp:
#             if isinstance(m, Linear):
#                 kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization
#                 if m.bias is not None:
#                     zeros_(m.bias)  # Initialize biases to zero

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         row, col = edge_index
#         out = edge_attr

#         # Multipooling layer
#         out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
#         out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
#         out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
#         out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

#         out = self.node_mlp(out)
#         if self.residuals:
#             out = out + x
#         return out


# # Graph Neural Network architecture, based on the Graph Network (arXiv:1806.01261)
# # Employing the MetaLayer implementation in Pytorch-Geometric

# class GNN(torch.nn.Module):
#     def __init__(self, node_features, n_layers, hidden_channels, linkradius, dim_out, dropout_rate, alpha, residuals=True):
#         super().__init__()

#         self.n_layers = n_layers
#         self.link_r = linkradius
#         self.dim_out = dim_out
#         self.alpha = alpha

#         # Number of input node features 
#         node_in = node_features
        
#         # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
#         edge_in = 3
#         node_out = hidden_channels
#         edge_out = hidden_channels
#         hid_channels = hidden_channels

#         layers = []

#         # Encoder graph block
#         inlayer = MetaLayer(
#             node_model=NodeModel(node_in,
#                                  node_out,
#                                  edge_in,
#                                  edge_out,
#                                  hid_channels,
#                                  residuals=False,
#                                  dropout_rate=dropout_rate),
#             edge_model=EdgeModel(node_in,
#                                  node_out,
#                                  edge_in,
#                                  edge_out,
#                                  hid_channels,
#                                  residuals=False,
#                                  dropout_rate=dropout_rate)
#         )

#         layers.append(inlayer)

#         # Change input node and edge feature sizes
#         node_in = node_out
#         edge_in = edge_out

#         # Hidden graph blocks
#         for _ in range(n_layers - 1):
#             lay = MetaLayer(
#                 node_model=NodeModel(node_in,
#                                      node_out,
#                                      edge_in,
#                                      edge_out,
#                                      hid_channels,
#                                      residuals=residuals,
#                                      dropout_rate=dropout_rate),
#                 edge_model=EdgeModel(node_in,
#                                      node_out,
#                                      edge_in,
#                                      edge_out,
#                                      hid_channels,
#                                      residuals=residuals,
#                                      dropout_rate=dropout_rate)
#             )

#             layers.append(lay)

#         # holding submodules in a list
#         self.layers = ModuleList(layers)

#         # Final aggregation layer 
#         self.outlayer = Sequential(
#             Linear(3 * node_out + 1, hid_channels),
#             GELU(),
#             Dropout(dropout_rate),
#             Linear(hid_channels, hid_channels),
#             GELU(),
#             Dropout(dropout_rate),
#             Linear(hid_channels, hid_channels),
#             GELU(),
#             Dropout(dropout_rate),
#             Linear(hid_channels, self.dim_out),
#             Softplus()
#         )

#         # Apply Kaiming initialization for the final output layer
#         for m in self.outlayer:
#             if isinstance(m, Linear):
#                 kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     zeros_(m.bias)

#     def forward(self, data):
#         h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u

#         # Message passing layers
#         for layer in self.layers:
#             h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

#         # Multipooling layer
#         addpool = global_add_pool(h, data.batch)
#         meanpool = global_mean_pool(h, data.batch)
#         maxpool = global_max_pool(h, data.batch)

#         out = torch.cat([addpool, meanpool, maxpool, u], dim=1)

#         # Final linear layer
#         out = self.outlayer(out)
#         out[:, 1] = self.alpha * out[:, 1]

#         return out