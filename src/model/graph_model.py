from flax import nnx 
from typing import Literal, List, Tuple
from torch_geometric.data import Data
import jax.numpy as jnp 
import torch 
import jax 
from jax.typing import ArrayLike

class GATConvJax(nnx.Module): 
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 heads: int, 
                 edge_dim: int, 
                 rngs: nnx.Rngs):
        return None

    def __call__(self, 
                 x: ArrayLike,
                 jnp_edge_index: ArrayLike, 
                 edge_attr: ArrayLike,
                 ):
        
        raise NotImplementedError 
        return None


class NEGATRegressorJAX(nnx.Module):
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 list_edge_hidden_features: list, 
                 gat_out_features: int, 
                 gat_head: int, 
                 rngs: nnx.Rngs,
                 bias: bool = True,  
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
    ):
        self.name = "NEGATRegressorJAX"
        self.bias = bias 
        self.gnn_param_matrix = []

        ############ GNN: node regression convolution layers #################
        self.node_layers = dict()

        in_feats_n = node_input_features 

        if len(list_node_hidden_features) != 0: 
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers[idx] = [nnx.Linear(in_features=in_feats_n, 
                                                  out_features=hid_feats_n, 
                                                  rngs=rngs) for _ in range(k_hop_node)]
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fcnn_node = nnx.Linear(hid_feats_n, node_out_features, rngs=rngs)

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = dict()
        self.edge_biases = []

        in_feats_e = edge_input_features 

        # add bias to SCNN as a whole following the equations 
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features): 
                self.edge_layers[idx] = [[nnx.Linear(in_features=in_feats_e, 
                                                    out_features=hid_feats_e, 
                                                    rngs=rngs, 
                                                    use_bias=False),
                                        nnx.Linear(in_features=in_feats_e, 
                                                    out_features=hid_feats_e, 
                                                    rngs=rngs, 
                                                    use_bias=False)] for _ in range(k_hop_edge)]
                if bias: 
                    self.edge_biases.append(nnx.Param(jnp.zeros((hid_feats_e,))))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e 
        else: 
            hid_feats_e = in_feats_e

        self.fcnn_edge = nnx.Linear(hid_feats_e, edge_output_features, rngs=rngs)

        ############### Custom GATConv ##########################################
        self.gatconv_jax = GATConvJax(in_channels=node_out_features, 
                                      out_channels=gat_out_features, 
                                      heads=gat_head, 
                                      edge_dim=edge_output_features)









    def __call__(self, 
                 tupleData: Tuple[Data]) -> jnp.array:
        
        node_data, edge_data = tupleData[0], tupleData[1]
        x = jnp.array(node_data.x.detach().cpu().numpy())
        A_x = self.jax_adjacency_matrix(node_data.edge_index)

        x1 = jnp.array(edge_data.x.detach().cpu().numpy())
        L_1l = self.jax_adjacency_matrix(edge_data.edge_index, edge_data.edge_attrs)
        L_1u = self.jax_adjacency_matrix(edge_data.edge_index_u, edge_data.edge_attrs2)



    def jax_adjacency_matrix(self, 
                             edge_index: torch.Tensor, 
                             edge_attr: torch.Tensor) -> jnp.array: 
        jnp_edge_index = jnp.array(edge_index)
        num_simplices = jnp_edge_index.max()

        jnp_A = jnp.zeros((num_simplices+1, num_simplices+1))

        if edge_attr == None: 
            for u, v in jnp_edge_index.T: 
                jnp_A = jnp_A.at(u,v).set(1.0)
        else: 
            jnp_edge_attr = jnp.array(edge_attr)
            for id, (u, v) in enumerate(jnp_edge_index.T):
                jnp_A = jnp_A.at(u,v).set(jnp_edge_attr[id]) 
        
        return jnp_A




        


# class NEGATRegressor(nn.Module):
#     """ Only SE with Node and Edge Regression followed by GATConv."""
#     def __init__(self, 
#                  node_input_features: int, 
#                  list_node_hidden_features: List[int], 
#                  node_out_features: int, 
#                  k_hop_node: int,  
#                  edge_input_features: int, 
#                  edge_output_features: int, 
#                  k_hop_edge: int, 
#                  list_edge_hidden_features: list, 
#                  gat_out_features: int, 
#                  gat_head: int,
#                  bias: bool = True, 
#                  normalize: bool = True, 
#                  adj_norm: bool = True, # normalize the adjacency matrix (recommended)
#                  device: Literal['cuda','cpu','mps'] = 'cpu',
#     ):
#         super().__init__()
#         self.name = "NEGATRegressor" # used in logging 
#         self.bias = bias
#         self.device = device


#         ###### GNN: node regression convolution layers ###### 
#         self.node_layers = nn.ModuleList()
#         in_feats_n = node_input_features 
         
#         if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
#             for idx, hid_feats_n in enumerate(list_node_hidden_features): 
#                 self.node_layers.append(TAGConv(in_channels=in_feats_n, 
#                                                 out_channels=hid_feats_n, 
#                                                 K=k_hop_node, 
#                                                 bias=bias, 
#                                                 normalize=adj_norm))
#                 # no normalization after last layer
#                 if normalize and idx < len(list_node_hidden_features): 
#                     self.node_layers.append(LayerNorm(hid_feats_n))
#                 in_feats_n = hid_feats_n
#         else: 
#             hid_feats_n = in_feats_n
        
#         self.fc_node = nn.Linear(hid_feats_n, node_out_features)

#         ###### SCNN: edge-regression convolution layers ######
#         self.edge_layers = nn.ModuleList()
#         self.edge_biases = nn.ParameterList()

#         in_feats_e = edge_input_features

#         # add bias to SCNN as a whole (rather than individual TAGConv above)
#         if len(list_edge_hidden_features) != 0: 
#             for idx, hid_feats_e in enumerate(list_edge_hidden_features):

#                 self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
#                                                 out_channels=hid_feats_e,
#                                                 K=k_hop_edge,
#                                                 bias=False,
#                                                 normalize=False),
#                                         TAGConv(in_channels=in_feats_e,
#                                                 out_channels=hid_feats_e, 
#                                                 K=k_hop_edge,
#                                                 bias=False, # TODO: True? 
#                                                 normalize=False)]))
#                 if bias:
#                     self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
#                 else: 
#                     self.edge_biases.append(None)
#                 in_feats_e = hid_feats_e
#         else: 
#             hid_feats_e = in_feats_e

#         self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

#         self.gatconv = GATConv(in_channels=node_out_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_output_features)

#         # self.gatconv = GATConv(in_channels=node_input_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_input_features)

#         # since gatconv with multiple heats concatenates outputs, a final regression layer is required. 
#         # mlp 
#         self.mlp_gat = nn.Sequential(
#             nn.Linear(gat_out_features * self.gatconv.heads, 2, bias = True),
#             # nn.ReLU(), # TODO: Dropout required?
#         )

#         self.reset_parameters() 
    
#     def reset_parameters(self):
#         def reset_layer(layer): 
#             """Helper function to avoid redundancy"""
#             if isinstance(layer, MessagePassing):
#                 # using kaiming initialization for each layer in GNN 
#                 for name, param in layer.named_parameters(): # iterator over parameters 
#                     if "weight" in name: 
#                         # if using relu
#                         # nn.init.kaiming_uniform_(param, nonlinearity='relu')
#                         # different from layer.reset_parameters() --> uniform 
                        
#                         # if using tanh 
#                         nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

#                     elif "bias" in name and param is not None: # if bias = False 
#                         nn.init.constant_(param, 0.1)
#             elif isinstance(layer, nn.Linear): 
#                 # uniform_ means the operation modifies tensor in-place
#                 # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#                 nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
#                 if layer.bias is not None: 
#                     # zero initialize prevents learning angles 
#                     nn.init.constant_(layer.bias, 0.01)
#         self.apply(reset_layer)
    
#     def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
#         node_data, edge_data = tupleData[0], tupleData[1]
#         x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
#         x1 = edge_data.x.to(self.device)
#         edge_index_l = edge_data.edge_index.to(self.device)
#         edge_weight_l = edge_data.edge_attr.to(self.device)
#         edge_index_u = edge_data.edge_index_u.to(self.device)
#         edge_weight_u = edge_data.edge_attr2.to(self.device)

#         # node-regression 
#         for layer in self.node_layers:
#             # no activation for LayerNorm 
#             if isinstance(layer, TAGConv): 
#                 x = layer(x, edge_index)
#                 x = F.relu(x)
#                 # torch.tanh_(x)
#             else: 
#                 x = layer(x)

#         x = self.fc_node(x)

#         # edge-regression
#         for layer, e_bias in zip(self.edge_layers, self.edge_biases):
#             x1 = layer[0](x=x1, edge_index=edge_index_l, edge_weight=edge_weight_l) \
#                 #   + layer[1](x=x1, edge_index=edge_index_u, edge_weight=edge_weight_u)
            
#             if self.bias: 
#                 x1 += e_bias
#             torch.relu_(x1) 

#         x1 = self.fc_edge(x1)
        
#         # gatconv
#         alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

#         # agg_mssgs to SE predictions
#         x_o = self.mlp_gat(alpha_gat)

#         # return x_o
#         return x_o