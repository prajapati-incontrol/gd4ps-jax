import os
import sys 
import logging 
from datetime import datetime 
import pandapower as pp 
import jax.numpy as jnp
import jax 
import numpy as np 
import warnings
import copy 
import time 
from sklearn.preprocessing import StandardScaler
import torch 
from typing import Tuple, List
import networkx as nx 
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import yaml


# # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, parent_dir)
parent_dir = os.getcwd()


def load_config(): 
    path = parent_dir + "/config/config.yaml"
    with open(path, "r") as f: 
        return yaml.safe_load(f)

def get_device(preferred_device: str = "auto") -> torch.device:

    """Selects the best available device based on user preference and availability.

    Args:
        preferred_device (str): Preferred device ("cpu", "cuda", "mps", or "auto").

    Returns:
        torch.device: The selected device.
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # macOS Metal Performance Shaders
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    elif preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")  # Default to CPU if the requested device is unavailable

#####################################################################################

def dataset_splitter(dataset:Dataset,
                     batch_size:int,
                     split_list:List[float]=[0.8,0.1,0.1]):
    """Dataset Splitter"""
    # split sizes 
    train_size = int(split_list[0] * len(dataset))
    val_size = int(split_list[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader

#####################################################################################

def construct_hodge_laplacian(G: nx.Graph) -> Tuple[
                                  Tuple[np.array, np.array], 
                                  Tuple[np.array, np.array]]:
    """This function creates Hodge-Laplacian and lower and upper incidence matrix given a Undirected Graph 
    by adding random orientation, making it a directed graph."""
    
    # directed graph 
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(G.nodes())
    G_dir.add_edges_from(G.edges())

    # get the incidence matrices 
    B_1 = nx.incidence_matrix(G_dir, oriented=True).toarray() 
    
    num_edges = G_dir.number_of_edges()
    num_triangles = sum(nx.triangles(nx.Graph(G_dir)).values()) // 3

    B_2 = np.zeros((num_edges, num_triangles))

    # dictionary of edge indices 
    edge_to_index = {edge:i for i, edge in enumerate(G_dir.edges())}

    triangle_idx = 0 
    # find all the triangles and fill the B_2 incidence matrix 
    for triangle in nx.enumerate_all_cliques(nx.Graph(G_dir)): # since enumerate* works for undirected only 
        if len(triangle) == 3:
            sorted_tri = sorted(triangle)
            cyclic_edges = [(sorted_tri[i], sorted_tri[(i+1) % 3]) for i in range(3)]

            for edge in cyclic_edges: 
                if edge in G_dir.edges(): 
                    B_2[edge_to_index[edge], triangle_idx] = 1
                elif (edge[1], edge[0]) in G_dir.edges(): 
                    B_2[edge_to_index[(edge[1], edge[0])], triangle_idx] = -1
            triangle_idx += 1 
    
    # check if boundary condition satisfied 
    bc = B_1 @ B_2 
    if not np.all(bc == 0):
        raise RuntimeError("Boundary Condition not satisfied! Check incidence matrix again.")

    # lower laplacian 
    L_l = B_1.T @ B_1 

    # upper laplacian 
    L_u = B_2 @ B_2.T 

    # check if the L_l and L_u dimensions are equal to number of edges 
    # first check if they are square 
    if not (L_l.shape[0] == L_l.shape[1] and L_u.shape[0] == L_u.shape[1]):
        raise ValueError("Laplacians are not a square matrix!")
    
    if not (L_l.shape[0] == num_edges and L_u.shape[0] == num_edges): 
        raise ValueError("Size of Laplacians is not equal to the number of edges!")

    return (L_l, L_u), (B_1, B_2)


#####################################################################################

def get_edge_index_lg(edge_index: torch.Tensor) -> Tuple[torch.Tensor]: 
    """Calculates the edge-index representing the linegraph laplacian."""
    G = nx.Graph()
    edge_index_np = edge_index.cpu().numpy() # [2, num_edges]
    edges = [(edge_index_np[0,i], edge_index_np[1,i]) for i in range(edge_index_np.shape[1])]
    G.add_edges_from(edges)

    # calculate the B_1 node-edge incidence matrix 
    B_1 = nx.incidence_matrix(G, oriented=True).toarray()

    # linegraph adjacency 
    A_lg = np.abs(B_1.T @ B_1 - 2 * np.eye(B_1.shape[1]))

    # linegraph laplacian 
    L_lg = np.diag(A_lg @ np.ones(B_1.shape[1])) - A_lg

    # get row and col indices 
    row_lg, col_lg = np.nonzero(L_lg)

    # add edge weights 
    edge_weight_lg = []
    for row_id, col_id in zip(row_lg, col_lg):
        edge_weight_lg.append(L_lg[row_id, col_id])

    edge_weight_lg = torch.tensor(edge_weight_lg, dtype=torch.float32)
    if tensor_any_nan(edge_weight_lg)[0]:
        raise ValueError("NaN in Line-graph Laplacian!")


    # create equivalent edge indices 
    edge_index_lg = torch.tensor(np.vstack((row_lg, col_lg)), dtype=torch.long)

    return edge_index_lg, edge_weight_lg


#####################################################################################

def get_edge_index_lu(edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the edge-index representing the lower and upper laplacian."""
    G = nx.Graph()
    edge_index_np = edge_index.cpu().numpy() # [2, num_edges]
    edges = [(edge_index_np[0,i], edge_index_np[1,i]) for i in range(edge_index_np.shape[1])]
    G.add_edges_from(edges)

    (L_l, L_u), _ = construct_hodge_laplacian(G=G)

    row_l, col_l = np.nonzero(L_l)
    row_u, col_u = np.nonzero(L_u)

    # add edge-weights 
    edge_weight_l, edge_weight_u = [], []
    for row_id, col_id in zip(row_l, col_l):
        edge_weight_l.append(L_l[row_id, col_id])

    edge_weight_l = torch.tensor(edge_weight_l, dtype=torch.float32)

    for row_id, col_id in zip(row_u, col_u):
        edge_weight_u.append(L_u[row_id, col_id])        

    edge_weight_u = torch.tensor(edge_weight_u, dtype=torch.float32)

    if tensor_any_nan(edge_weight_l, edge_weight_u)[0]:
        raise ValueError("NaN in Hodge Laplacian!")

    # create equivalent edge indices 
    edge_index_l = torch.tensor(np.vstack((row_l, col_l)), dtype=torch.long)
    edge_index_u = torch.tensor(np.vstack((row_u, col_u)), dtype=torch.long)


    return edge_index_l, edge_index_u, edge_weight_l, edge_weight_u


#####################################################################################

def get_edge_index_from_ppnet(net):
    """
    This function creates an edge-index tensor ([2, num_edges]) from pandapower net,
    considering both lines (from_bus to to_bus) and transformers (hv_bus to lv_bus).

    Args:
        net: Pandapower Net

    Returns:
        torch.Tensor: Edge-index in COO format; Shape [2, num_edges]
    """
    edges = []
    
    # Add lines (from_bus to to_bus)
    for _, line in net.line.iterrows():
        edges.append((line.from_bus, line.to_bus))
    
    # Add transformers (hv_bus to lv_bus)
    for _, trafo in net.trafo.iterrows():
        edges.append((trafo.hv_bus, trafo.lv_bus))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    if tensor_any_nan(edge_index)[0]: 
        print(f"Edge index tensor = {edge_index}")
        raise ValueError("Nan in get_edge_index_from_ppnet")
    return edge_index


#####################################################################################

def load_sampled_input_data(net: pp.pandapowerNet, 
                            num_samples: int, 
                            load_std: float = 0.3, 
                            noise: float = 0.05, 
                            ):
    """
    Load the data.

    Node Features: V, P + Gaussian Noise 
    Edge Features: P_IN, Q_IN, P_OUT, Q_OUT, r, x, b, g, shift, tap_pos (incorrect tap pos)
    Y Label (Node): V, THETA 
    
    Args: 
        net (pp.pandapowerNet): Pandapower Network 
        num_samples (int): Number of samples to collect 
        load_std (float): Gaussian Noise on load for monte-carlo sampling.
        noise (float): Gaussian noise standard deviation to perturb the power flow results 
        
    Returns: 
        sampled_input_data (dict): Dictionary of sampled input data, with keys varying for each scenario. 
    
    """
    sampled_input_data = dict()
    edge_index = get_edge_index_from_ppnet(net=net)
    sampled_input_data['edge_index'] = edge_index

    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    # node input features  
    num_buses = len(net.bus.index)
    num_node_features = 2 # V and P
    # node input features 
    node_input_features = np.zeros((num_samples, num_buses, num_node_features))

    # edge input features 
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos
    num_edge_features = 10 # p_from, q_from, p_to, q_to, r, x, b, g, shift, tap    

    # edge input features 
    edge_input_features = np.zeros((num_samples, num_edges, num_edge_features))

    # node output label features 
    y_label = np.zeros((num_samples, num_buses, 2)) # v and theta 

    # add r, x, b, g, shift, tap to net 
    net = add_branch_parameters(net)

    # variables to permutate in the net 
    pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)

    ### Adding data to the node and edge input feature matrices ###

    # since edge-features for r,x,b,g,shift are not considered to change, assign them here. 
    # Vectorized edge tensor assignment
    
    # lines
    edge_input_features[:,:num_lines,4:8] = np.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32) 
    # trafos
    edge_input_features[:, num_lines:,4:9] = np.array(net.trafo[['r', 'x', 'b', 'g', 'shift_rad']].values, dtype=np.float32)
    
    # maximum retry: since power flow results cannot converge sometimes 
    max_retries = 20 
    seed_perm = 0 

    for iperm in range(num_samples):
        retries = 0 
        while retries < max_retries:
            seed_perm += 1 
            try: 
                rng = np.random.default_rng(seed_perm)
                # permutate variables 
                pload = rng.normal(pload_ref, load_std)
                qload = rng.normal(qload_ref, load_std)

                # modify the net data 
                net.load['p_mw'] = pload 
                net.load['q_mvar'] = qload 

                net['converged'] = False 
                pp.runpp(net, max_iteration=50)

                # store the results for v and p only 
                node_input_features[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                y_label[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_input_features[iperm,:,1] = np.array(net.res_bus.p_mw.values)
                y_label[iperm,:,1] = np.array(net.res_bus.va_degree.values)

                # p_from measurements 
                edge_input_features[iperm,:num_lines,0] = np.array(net.res_line.p_from_mw)
                edge_input_features[iperm,num_lines:,0] = np.array(net.res_trafo.p_hv_mw)

                # q_from measurements 
                edge_input_features[iperm,:num_lines,1] = np.array(net.res_line.q_from_mvar)
                edge_input_features[iperm,num_lines:,1] = np.array(net.res_trafo.q_hv_mvar)

                # p_to measurements 
                edge_input_features[iperm,:num_lines,2] = np.array(net.res_line.p_to_mw)
                edge_input_features[iperm,num_lines:,2] = np.array(net.res_trafo.p_lv_mw)

                # q_to measurements 
                edge_input_features[iperm,:num_lines,3] = np.array(net.res_line.q_to_mvar)
                edge_input_features[iperm,num_lines:,3] = np.array(net.res_trafo.q_lv_mvar)

                break # exit while loop if successful (reaching this line.)

            
            except Exception as e: 
                print(f"\t Error at permutation {iperm}: {e}")
                print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                retries += 1
                continue

        if retries == max_retries:
            print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
            node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure

    # calculate the mask for available measurements at node and edge input features 
    # account for real measurements 
    node_mask = np.zeros((num_samples, num_buses, num_node_features))
    edge_mask = np.zeros((num_samples, num_edges, num_edge_features))
    edge_mask[:,:,4:] = 1 # parameters are considered known.
    
    if noise: 
        edge_input_features_noisy = copy.deepcopy(edge_input_features)
        node_input_features_noisy = copy.deepcopy(node_input_features)
        
        node_input_features_noisy[:,:,0] = np.random.normal(node_input_features[:,:,0], 0.5/100/3) 
        node_input_features_noisy[:,:,1] = np.random.normal(node_input_features[:,:,1], 5/100/3) 

        # add uncertainty to pq measurements only not parameters
        edge_input_features_noisy[:,:,:4] = np.random.normal(edge_input_features[:,:,:4], 5/100)

        # convert all arrays to tensors 
        node_input_features_noisy = torch.tensor(node_input_features_noisy, dtype=torch.float32)
        edge_input_features_noisy = torch.tensor(edge_input_features_noisy, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)

        if tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")
        
        node_input_feat, scaler_n = scale_numeric_columns(node_input_features_noisy)
        edge_input_feat, scaler_e = scale_numeric_columns(edge_input_features_noisy, categorical_cols=[9])
        y_label, scaler_y = scale_numeric_columns(y_label)

        # for only node data 
        sampled_input_data['node_input_feat'] = node_input_feat
        sampled_input_data['edge_input_feat'] = edge_input_feat
        sampled_input_data['y_label'] = y_label 
        sampled_input_data['scaler_node'] = scaler_n
        sampled_input_data['scaler_edge'] = scaler_e
        sampled_input_data['scaler_y_label'] = scaler_y

        
        return sampled_input_data

    else: 

        # convert all arrays to tensors 
        node_input_features = torch.tensor(node_input_features, dtype=torch.float32)
        edge_input_features = torch.tensor(edge_input_features, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)

        if tensor_any_nan(node_input_features, edge_input_features, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")

        node_input_feat, scaler_n = scale_numeric_columns(node_input_features)
        edge_input_feat, scaler_e = scale_numeric_columns(edge_input_features, categorical_cols=[9])
        y_label, scaler_y = scale_numeric_columns(y_label)

        # for only node data 
        sampled_input_data['node_input_feat'] = node_input_feat
        sampled_input_data['edge_input_feat'] = edge_input_feat
        sampled_input_data['y_label'] = y_label 
        sampled_input_data['scaler_node'] = scaler_n
        sampled_input_data['scaler_edge'] = scaler_e
        sampled_input_data['scaler_y_label'] = scaler_y

        return sampled_input_data


#####################################################################################


    # # remove switches PoC 
    # net.switch.drop(net.switch.index, inplace = True)
    # net.res_switch.drop(net.res_switch.index, inplace = True)

    # # node input features  
    # num_buses = len(net.bus.index)
    # num_node_features = 2 # V and P
    # # node input features 
    # node_input_features = jnp.zeros((num_samples, num_buses, num_node_features))
    
    # # edge input features 
    # num_lines = len(net.line.index)
    # num_trafos = len(net.trafo.index)
    # num_edges = num_lines + num_trafos
    # num_edge_features = 10 # p_from, q_from, p_to, q_to, r, x, b, g, shift, tap    

    # # edge input features 
    # edge_input_features = jnp.zeros((num_samples, num_edges, num_edge_features))

    # # node output label features 
    # y_label = jnp.zeros((num_samples, num_buses, 2)) # v and theta 

    # # add r, x, b, g, shift, tap to net 
    # net = add_branch_parameters(net)

    # # variables to permutate in the net 
    # pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)


    # # set constant line parameters 
    # edge_input_features = edge_input_features.at[:, :num_lines, 4:8].set(
    #     jnp.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=jnp.float32)
    # )

    # # set constant trafo parameters 
    # # trafos
    # edge_input_features = edge_input_features.at[:, num_lines:, 4:10].set(
    #     jnp.array(net.trafo[['r', 'x', 'b', 'g', 'shift_rad','tap_pos']].values, dtype=jnp.float32)
    # )

    # # maximum retry: since power flow results cannot converge sometimes 
    # max_retries = 20 
    # seed_perm = 0 

    # for iperm in range(num_samples): 

    #     retries = 0 
    #     while retries < max_retries:

    #         seed_perm += 1 

    #         try: 
    #             rng = np.random.default_rng(seed_perm)
    #             # permutate variables 
    #             pload = rng.normal(pload_ref, load_std)
    #             qload = rng.normal(qload_ref, load_std)

    #             # modify the net data 
    #             net.load['p_mw'] = pload 
    #             net.load['q_mvar'] = qload 

    #             net['converged'] = False 
    #             pp.runpp(net, max_iteration=50)

    #             # Store the results for v and p only
    #             node_input_features = node_input_features.at[iperm, :, 0].set(
    #                 jnp.array(net.res_bus.vm_pu.values, dtype=jnp.float32)
    #             )
    #             y_label = y_label.at[iperm, :, 0].set(
    #                 jnp.array(net.res_bus.vm_pu.values, dtype=jnp.float32)
    #             )
    #             node_input_features = node_input_features.at[iperm, :, 1].set(
    #                 jnp.array(net.res_bus.p_mw.values, dtype=jnp.float32)
    #             )
    #             y_label = y_label.at[iperm, :, 1].set(
    #                 jnp.array(net.res_bus.va_degree.values, dtype=jnp.float32)
    #             )

    #             # p_from measurements
    #             edge_input_features = edge_input_features.at[iperm, :num_lines, 0].set(
    #                 jnp.array(net.res_line.p_from_mw.values, dtype=jnp.float32)
    #             )
    #             edge_input_features = edge_input_features.at[iperm, num_lines:, 0].set(
    #                 jnp.array(net.res_trafo.p_hv_mw.values, dtype=jnp.float32)
    #             )

    #             # q_from measurements
    #             edge_input_features = edge_input_features.at[iperm, :num_lines, 1].set(
    #                 jnp.array(net.res_line.q_from_mvar.values, dtype=jnp.float32)
    #             )
    #             edge_input_features = edge_input_features.at[iperm, num_lines:, 1].set(
    #                 jnp.array(net.res_trafo.q_hv_mvar.values, dtype=jnp.float32)
    #             )

    #             # p_to measurements
    #             edge_input_features = edge_input_features.at[iperm, :num_lines, 2].set(
    #                 jnp.array(net.res_line.p_to_mw.values, dtype=jnp.float32)
    #             )
    #             edge_input_features = edge_input_features.at[iperm, num_lines:, 2].set(
    #                 jnp.array(net.res_trafo.p_lv_mw.values, dtype=jnp.float32)
    #             )

    #             # q_to measurements
    #             edge_input_features = edge_input_features.at[iperm, :num_lines, 3].set(
    #                 jnp.array(net.res_line.q_to_mvar.values, dtype=jnp.float32)
    #             )
    #             edge_input_features = edge_input_features.at[iperm, num_lines:, 3].set(
    #                 jnp.array(net.res_trafo.q_lv_mvar.values, dtype=jnp.float32)
    #             )
            

    #         except Exception as e: 
    #             print(f"\t Error at permutation {iperm}: {e}")
    #             print(f"\t Retry #{retries} at {iperm} with a new random seed...")
    #             retries += 1
    #             continue
        
    #     if retries == max_retries:
    #         print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
    #         node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure
    
    # if noise: 
    #     key = jax.random.PRNGKey(int(time.time()))  # You should manage this key properly in your pipeline

    #     # Copy the original arrays
    #     edge_input_features_noisy = edge_input_features.copy()
    #     node_input_features_noisy = node_input_features.copy()

    #     # Add noise to node features
    #     key, subkey1, subkey2 = jax.random.split(key, 3)
    #     noise_vm_pu = jax.random.normal(subkey1, node_input_features[:, :, 0].shape) * (0.5 / 100 / 3)
    #     noise_p_mw = jax.random.normal(subkey2, node_input_features[:, :, 1].shape) * (5 / 100 / 3)

    #     node_input_features_noisy = node_input_features_noisy.at[:, :, 0].add(noise_vm_pu)
    #     node_input_features_noisy = node_input_features_noisy.at[:, :, 1].add(noise_p_mw)

    #     # Add noise to edge pq measurements only (first 4 features)
    #     key, subkey3 = jax.random.split(key)
    #     noise_edges = jax.random.normal(subkey3, edge_input_features[:, :, :4].shape) * (5 / 100)
    #     edge_input_features_noisy = edge_input_features_noisy.at[:, :, :4].add(noise_edges)

    #     sampled_input_data['node_input_features'] = scale_jnp_array(node_input_features_noisy)
    #     sampled_input_data['edge_input_features'] = scale_jnp_array(edge_input_features_noisy, categorical_index=9)
    #     sampled_input_data['y_label'] = scale_jnp_array(y_label)

    #     return sampled_input_data
    
    # else: 

    #     sampled_input_data['node_input_features'] = scale_jnp_array(node_input_features_noisy)
    #     sampled_input_data['edge_input_features'] = scale_jnp_array(edge_input_features_noisy, categorical_index=9)
    #     sampled_input_data['y_label'] = scale_jnp_array(y_label)

    #     return sampled_input_data


#####################################################################################

def scale_numeric_columns(input_tensor: torch.Tensor, 
                          categorical_cols: int = None):
    """
    Scales only numerical columns in a 3d tensors, keeping categorical columns unchanged. (for tap positions)
   """

    tensor_list = list(input_tensor)

    # Convert tensors to numpy arrays for scaling
    tensor_np_list = [tensor.numpy() for tensor in tensor_list]

    # Identify numerical columns (all columns except categorical ones)
    all_cols = range(tensor_np_list[0].shape[1])
    if categorical_cols is None:
        numerical_cols = all_cols  # If no categorical cols, scale everything
        categorical_cols = []  # Empty categorical list
    else:
        numerical_cols = list(set(all_cols) - set(categorical_cols))  # Exclude categorical cols

    # Fit StandardScaler only on numerical columns
    scaler = StandardScaler()
    flat_numerical_data = np.vstack([tensor[:, numerical_cols] for tensor in tensor_np_list])
    scaler.fit(flat_numerical_data)

    # Transform only numerical columns
    scaled_tensor_list = [
        torch.tensor(
            np.hstack((scaler.transform(tensor[:, numerical_cols]), tensor[:, categorical_cols])), 
            dtype=torch.float32
        ) 
        for tensor in tensor_np_list
    ]
    
    scaled_tensor = torch.stack(scaled_tensor_list) # 3D Tensor

    return scaled_tensor, scaler


#####################################################################################


def inverse_scale(scaled_tensor: torch.Tensor, 
                  scaler: StandardScaler, 
                  categorical_cols: int = None):
    """
    Reverts the scaling of numerical columns to get back the original values.

    """

    # obtain the numerical cols 
    all_cols = range(scaled_tensor.shape[1])
    if categorical_cols is None:
        numerical_cols = all_cols  # If no categorical cols, scale everything
        categorical_cols = []  # Empty categorical list
    else:
        numerical_cols = list(set(all_cols) - set(categorical_cols))  # Exclude categorical cols


    # Apply inverse transform only to numerical columns
    if len(scaled_tensor.shape) == 3: 
        
        scaled_tensor_list = list(scaled_tensor)

        # Convert tensors to numpy arrays for inverse transformation
        scaled_np_list = [tensor.numpy() for tensor in scaled_tensor_list]

        original_tensor_list = [
            torch.tensor(
                np.hstack((scaler.inverse_transform(tensor[:, numerical_cols]), tensor[:, categorical_cols])), 
                dtype=torch.float32
            )
            for tensor in scaled_np_list
        ]
        original_tensor = torch.stack(original_tensor_list)
        return original_tensor
    else: # 2d 
        scaled_np = scaled_tensor.cpu().detach().numpy() 
        original_tensor = torch.tensor(
            np.hstack((scaler.inverse_transform(scaled_np[:, numerical_cols]), scaled_np[:, categorical_cols])), 
                dtype=torch.float32
        )
        return original_tensor


#####################################################################################

def add_branch_parameters(net: pp.pandapowerNet): 
    """Calculates the branch parameters of the bus-branch model. 
    if branch is line, then net.line has added columns as r, x, b, g
    if branch is trafo, then net.trafo has added columns as r, x, b, g, tap, shift 
    
    Assumptions: 
    No mutual inductance between lines. 

    Note: 
    Pandapower equations from documentation for trafo r, x, b and g are ambiguous, and do not match with the 
    internal y-bus matrices. So, branch parameters for trafo are calculated from Y_bus internals and not 
    the pandapower equations. 

    Returns: 
    net: Pandapower Net with added branch parameter for line and trafo. 
    
    """
    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    sys_freq = 50 # [Hz]
    
    # check 1: if the required columns are present in lines dataframe  
    required_columns = {
        'line':['r_ohm_per_km', 'length_km', 'parallel', 'from_bus', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km'],
        'bus':['vn_kv'],
        'trafo': ['lv_bus', 'vn_lv_kv', 'sn_mva', 'vkr_percent', 'vk_percent', 'pfe_kw', 'i0_percent', 'hv_bus', 'vn_hv_kv']
    }

    for element, columns in required_columns.items(): 
        for column in columns:
            if column not in net[element].columns:
                warnings.warn(f"Column '{column}' is missing in '{element}', padding with zeros.")
                net[element][column] = 0    
    
    # line branch parameters 
    sys_freq = 50

    # convert r_ohm_per_km to r_pu
    r_pu = (net.line['r_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values
    net.line.loc[:,'r_pu'] = r_pu

    # convert x_ohm_per_km to x_ohm
    x_pu = (net.line['x_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values
    net.line.loc[:,'x_pu'] = x_pu

    z_pu = r_pu + 1j*x_pu
    y_series = 1 / z_pu

    # convert c_nf_per_km to b_mho
    b_pu = ( 2 * np.pi * sys_freq * net.line['c_nf_per_km'] * 10**(-9) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values 
    net.line.loc[:, 'b_pu'] = b_pu

    # convert g_us_per_km to g_mho
    g_pu = ( 2 * np.pi * sys_freq * net.line['g_us_per_km'] * 10**(-6) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values
    net.line.loc[:, 'g_pu'] = g_pu

    # add zeros for tap and shift degree in lines 
    net.line.loc[:, 'tap_nom'] = 0 
    net.line.loc[:, 'shift_rad'] = 0


    y_sh = g_pu - 1j*b_pu    

    # get Y-bus for lines only 
    a_1 = y_series + y_sh/2
    a_2 = - y_series
    a_3 = - y_series
    a_4 = y_series + y_sh/2

    yb_size = len(net.bus)

    Ybline = np.zeros((yb_size, yb_size)).astype(complex)

    fb_line = net.line.from_bus 
    tb_line = net.line.to_bus 

    line_idx = net.line.index 

    for (idx, fb, tb) in zip(line_idx, fb_line, tb_line): 
        Ybline[fb, fb] = complex(a_1[idx])
        Ybline[fb, tb] = complex(a_2[idx])
        Ybline[tb, fb] = complex(a_3[idx])
        Ybline[tb, tb] = complex(a_4[idx])

    # get the pandpaower internal YBus 
    pp.runpp(net)
    Ybus = np.array(net._ppc["internal"]["Ybus"].todense())

    Ybus_trafo = Ybus - Ybline 

    if sum(net.trafo.tap_pos.isna()) > 0: 
        print("Filling nan as 0 in tap_pos, tap_neutral, tap_step_degree")
        net.trafo.loc[net.trafo.loc[:,'tap_pos'].isna(),'tap_pos'] = 0
        net.trafo.loc[net.trafo.loc[:,'tap_neutral'].isna(),'tap_neutral'] = 0
        net.trafo.loc[net.trafo.loc[:,'tap_step_degree'].isna(),'tap_step_degree'] = 0
        
        

    ps_s = net.trafo.shift_degree * np.pi/180
    net.trafo.loc[:,"shift_rad"] = ps_s
    tap_nom_s = 1 + (net.trafo.tap_pos - net.trafo.tap_neutral) * (net.trafo.tap_step_percent / 100)
    net.trafo.loc[:,"tap_nom"] = tap_nom_s 
    N_tap_s = tap_nom_s * np.exp(1j*ps_s)


    for id, row in net.trafo.iterrows(): 
        a_1t = Ybus_trafo[row.hv_bus, row.hv_bus]
        a_2t = Ybus_trafo[row.hv_bus, row.lv_bus]
        a_3t = Ybus_trafo[row.lv_bus, row.hv_bus]
        a_4t = Ybus_trafo[row.lv_bus, row.lv_bus]

        # series impedance 
        y_series_trafo = - a_3t * N_tap_s[id] 

        # r, x 
        z_s_trafo = 1 / y_series_trafo
        r_trafo, x_trafo = np.real(z_s_trafo), np.imag(z_s_trafo)

        net.trafo.loc[id, 'r'] = r_trafo
        net.trafo.loc[id, 'x'] = x_trafo 

        # shunt impedance 
        y_sh_trafo = 2* (a_4t - y_series_trafo) 
        g_trafo, b_trafo = np.real(y_sh_trafo), np.imag(y_sh_trafo)

        net.trafo.loc[id,'g'] = g_trafo 
        net.trafo.loc[id,'b'] = b_trafo
    
    return net


#####################################################################################

def get_adjacency(net):
    """
    Calculate adjacency matrix for the pandapower net. 

    Args:
        net (pandapower DataFrame)

    Returns: 
        Adjacency Matrix (S) : np.ndarray; Shape: [num_nodes, num_nodes] or [num_buses, num_buses]
    
    """

    S = jnp.zeros([len(net.bus.index), len(net.bus.index)])

    # buses connected by lines
    for i, j in zip(net.line.from_bus.values, net.line.to_bus.values):
        S = S.at[i, j].set(1)
        S = S.at[j, i].set(1)
    
    # buses connected by trafos 
    for i, j in zip(net.trafo.hv_bus.values, net.trafo.lv_bus.values):
        S = S.at[i, j].set(1)
        S = S.at[j, i].set(1)

    return S


def initialize_network(net_name: str, 
                       verbose: bool = True) -> pp.pandapowerNet: 
    match net_name: 
        case 'PP_MV_RING':
            net = pp.networks.simple_pandapower_test_networks.simple_mv_open_ring_net()
        
        case 'TOY':
            # load toy network
            net = pp.from_pickle(parent_dir + '/data/net_TOY.p')
            net.name = net_name 

            
        case 'MVO':
            net = pp.networks.mv_oberrhein(include_substations=True)
            net.trafo.loc[net.trafo.loc[:,'tap_step_percent'].isna(),'tap_step_percent'] = 2.173913
            net.name = net_name

        case _:
            raise NameError("\n Invalid Network Name! ")
    
    net.trafo.shift_degree = 0.0 # because bug in pandapower, sets shift_degree = 150.0 by default
    print(f"Network: {net_name} is selected \n")
    netG = pp.topology.create_nxgraph(net)
    print(f"Net {net_name} has {len(netG.nodes())} nodes and {len(netG.edges())} edges. \n")

    return net 


#####################################################################################

def setup_logging(log_dir="logs", 
                  log_filename="script_log.txt"): 
    """
    Configure logging to write to a file in the specified directory. Includes timestamped 
    logs and handles directory creation. 
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s", 
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler() # also output to console
        ]
    )

    logging.info("Logging initialized.")


#####################################################################################

def tensor_any_nan(*args):
    """Check if any input tensor contains NaN values."""
    nan_indices = [i for i, arg in enumerate(args) if torch.isnan(arg).any().item()]

    return any(torch.isnan(arg).any().item() for arg in args), nan_indices

