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


# # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, parent_dir)
parent_dir = os.getcwd()


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
    adj_matrix = get_adjacency(net)
    sampled_input_data['adj_matrix'] = adj_matrix

    # remove switches PoC 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    # node input features  
    num_buses = len(net.bus.index)
    num_node_features = 2 # V and P
    # node input features 
    node_input_features = jnp.zeros((num_samples, num_buses, num_node_features))
    
    # edge input features 
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos
    num_edge_features = 10 # p_from, q_from, p_to, q_to, r, x, b, g, shift, tap    

    # edge input features 
    edge_input_features = jnp.zeros((num_samples, num_edges, num_edge_features))

    # node output label features 
    y_label = jnp.zeros((num_samples, num_buses, 2)) # v and theta 

    # add r, x, b, g, shift, tap to net 
    net = add_branch_parameters(net)

    # variables to permutate in the net 
    pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)


    # set constant line parameters 
    edge_input_features = edge_input_features.at[:, :num_lines, 4:8].set(
        jnp.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=jnp.float32)
    )

    # set constant trafo parameters 
    # trafos
    edge_input_features = edge_input_features.at[:, num_lines:, 4:10].set(
        jnp.array(net.trafo[['r', 'x', 'b', 'g', 'shift_rad','tap_pos']].values, dtype=jnp.float32)
    )

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

                # Store the results for v and p only
                node_input_features = node_input_features.at[iperm, :, 0].set(
                    jnp.array(net.res_bus.vm_pu.values, dtype=jnp.float32)
                )
                y_label = y_label.at[iperm, :, 0].set(
                    jnp.array(net.res_bus.vm_pu.values, dtype=jnp.float32)
                )
                node_input_features = node_input_features.at[iperm, :, 1].set(
                    jnp.array(net.res_bus.p_mw.values, dtype=jnp.float32)
                )
                y_label = y_label.at[iperm, :, 1].set(
                    jnp.array(net.res_bus.va_degree.values, dtype=jnp.float32)
                )

                # p_from measurements
                edge_input_features = edge_input_features.at[iperm, :num_lines, 0].set(
                    jnp.array(net.res_line.p_from_mw.values, dtype=jnp.float32)
                )
                edge_input_features = edge_input_features.at[iperm, num_lines:, 0].set(
                    jnp.array(net.res_trafo.p_hv_mw.values, dtype=jnp.float32)
                )

                # q_from measurements
                edge_input_features = edge_input_features.at[iperm, :num_lines, 1].set(
                    jnp.array(net.res_line.q_from_mvar.values, dtype=jnp.float32)
                )
                edge_input_features = edge_input_features.at[iperm, num_lines:, 1].set(
                    jnp.array(net.res_trafo.q_hv_mvar.values, dtype=jnp.float32)
                )

                # p_to measurements
                edge_input_features = edge_input_features.at[iperm, :num_lines, 2].set(
                    jnp.array(net.res_line.p_to_mw.values, dtype=jnp.float32)
                )
                edge_input_features = edge_input_features.at[iperm, num_lines:, 2].set(
                    jnp.array(net.res_trafo.p_lv_mw.values, dtype=jnp.float32)
                )

                # q_to measurements
                edge_input_features = edge_input_features.at[iperm, :num_lines, 3].set(
                    jnp.array(net.res_line.q_to_mvar.values, dtype=jnp.float32)
                )
                edge_input_features = edge_input_features.at[iperm, num_lines:, 3].set(
                    jnp.array(net.res_trafo.q_lv_mvar.values, dtype=jnp.float32)
                )
            

            except Exception as e: 
                print(f"\t Error at permutation {iperm}: {e}")
                print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                retries += 1
                continue
        
        if retries == max_retries:
            print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
            node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure
    
    if noise: 
        key = jax.random.PRNGKey(int(time.time()))  # You should manage this key properly in your pipeline

        # Copy the original arrays
        edge_input_features_noisy = edge_input_features.copy()
        node_input_features_noisy = node_input_features.copy()

        # Add noise to node features
        key, subkey1, subkey2 = jax.random.split(key, 3)
        noise_vm_pu = jax.random.normal(subkey1, node_input_features[:, :, 0].shape) * (0.5 / 100 / 3)
        noise_p_mw = jax.random.normal(subkey2, node_input_features[:, :, 1].shape) * (5 / 100 / 3)

        node_input_features_noisy = node_input_features_noisy.at[:, :, 0].add(noise_vm_pu)
        node_input_features_noisy = node_input_features_noisy.at[:, :, 1].add(noise_p_mw)

        # Add noise to edge pq measurements only (first 4 features)
        key, subkey3 = jax.random.split(key)
        noise_edges = jax.random.normal(subkey3, edge_input_features[:, :, :4].shape) * (5 / 100)
        edge_input_features_noisy = edge_input_features_noisy.at[:, :, :4].add(noise_edges)

        sampled_input_data['node_input_features'] = scale_jnp_array(node_input_features_noisy)
        sampled_input_data['edge_input_features'] = scale_jnp_array(edge_input_features_noisy, categorical_index=9)
        sampled_input_data['y_label'] = scale_jnp_array(y_label)

        return sampled_input_data
    
    else: 

        sampled_input_data['node_input_features'] = scale_jnp_array(node_input_features_noisy)
        sampled_input_data['edge_input_features'] = scale_jnp_array(edge_input_features_noisy, categorical_index=9)
        sampled_input_data['y_label'] = scale_jnp_array(y_label)

        return sampled_input_data

def scale_jnp_array(jnp_array_3d: jnp.array, categorical_index: int = None):
    """
    Applies scikit-learn StandardScaler to a 3D jnp.array, excluding the specified categorical column.
    If categorical_index is None, scales all columns.

    """
    shape = jnp_array_3d.shape
    np_array = np.array(jnp_array_3d).reshape(-1, shape[-1])

    all_cols = list(range(shape[2]))

    if categorical_index is None:
        numerical_indices = all_cols
        categorical_data = None
    else:
        numerical_indices = [i for i in all_cols if i != categorical_index]
        categorical_data = np_array[:, categorical_index].reshape(-1, 1)

    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(np_array[:, numerical_indices])

    # Recombine numerical and categorical
    full_data = np.zeros_like(np_array)
    full_data[:, numerical_indices] = scaled_numerical
    if categorical_data is not None:
        full_data[:, categorical_index] = categorical_data.squeeze()

    scaled_jnp = jnp.array(full_data.reshape(shape), dtype=jnp.float32)
    return scaled_jnp, scaler

def inverse_scale_jnp_array(scaled_jnp_array_3d: jnp.array, 
                            scaler: StandardScaler, 
                            categorical_index: int = None):
    """
    Applies inverse transform using the fitted StandardScaler to a 3D jnp.array,
    excluding the specified categorical column. If categorical_index is None, all columns are inverse scaled.

    """
    shape = scaled_jnp_array_3d.shape
    np_array = np.array(scaled_jnp_array_3d).reshape(-1, shape[-1])

    all_cols = list(range(shape[2]))

    if categorical_index is None:
        numerical_indices = all_cols
        categorical_data = None
    else:
        numerical_indices = [i for i in all_cols if i != categorical_index]
        categorical_data = np_array[:, categorical_index].reshape(-1, 1)

    original_numerical = scaler.inverse_transform(np_array[:, numerical_indices])

    # Recombine
    full_data = np.zeros_like(np_array)
    full_data[:, numerical_indices] = original_numerical
    if categorical_data is not None:
        full_data[:, categorical_index] = categorical_data.squeeze()

    original_jnp = jnp.array(full_data.reshape(shape), dtype=jnp.float32)
    return original_jnp



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

