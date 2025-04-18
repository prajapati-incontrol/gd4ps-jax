from torch_geometric.data import Data, Dataset 
from typing import Dict 
import os 
import sys 

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils import get_edge_index_lu, get_edge_index_lg

class CustomDataset(Dataset):
    def __init__(self, 
                 sampled_input_data: Dict): 
        super().__init__()
        self.x = sampled_input_data['node_input_feat'] 
        self.y = sampled_input_data['y_label'] 
        self.num_samples = self.x.shape[0] # num_samples
        self.edge_index = sampled_input_data['edge_index'] # same for all samples 
        
        # hodge-laplacian dataset
        self.edge_attr = sampled_input_data['edge_input_feat'] 
        self.edge_index_l = get_edge_index_lu(self.edge_index)[0]
        self.edge_index_u = get_edge_index_lu(self.edge_index)[1]
        self.edge_weight_l = get_edge_index_lu(self.edge_index)[2]
        self.edge_weight_u = get_edge_index_lu(self.edge_index)[3]
        

        # linegraph laplacian dataset
        self.edge_index_lg = get_edge_index_lg(self.edge_index)[0] # line graph laplacian
        self.edge_weight_lg = get_edge_index_lg(self.edge_index)[1]  

    
    def __getitem__(self, index):
        node_graph_data = Data(x=self.x[index],
                                    edge_index=self.edge_index, 
                                    y=self.y[index])
        edge_HL_graph_data = Data(x=self.edge_attr[index],
                                      edge_index=self.edge_index_l, 
                                      edge_attr=self.edge_weight_l,
                                      edge_index_u=self.edge_index_u, 
                                      edge_attr2=self.edge_weight_u)
        edge_LG_graph_data = Data(x=self.edge_attr[index], 
                                      edge_index=self.edge_index_lg, 
                                      edge_attr=self.edge_weight_lg)
        
        return node_graph_data, edge_HL_graph_data, edge_LG_graph_data
    
    def __len__(self):
        return self.num_samples


    


# class XORDataset(Dataset):
#     def __init__(self, size, seed, std=0.1):
#         """
#         Inputs:
#             size - Number of data points we want to generate
#             seed - The seed to use to create the PRNG state with which we want to generate the data points
#             std - Standard deviation of the noise (see generate_continuous_xor function)
#         """
#         super().__init__()
#         self.size = size 
#         self.np_rng = np.random.RandomState(seed=seed)
#         self.std = std 
#         self.generate_continuous_xor()
    
#     def generate_continuous_xor(self):
#         data = self.np_rng.randint(low=0, high=2, size=(self.size,2)).astype(np.float32)
#         label = (data.sum(axis=1) == 1).astype(np.int32)
        
#         # add some gaussian noise to the datapoints 
#         data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

#         self.data = data 
#         self.label = label 
    
#     def __len__(self): 
#         return self.size 
    
#     def __getitem__(self, idx): 
#         data_point = self.data[idx]
#         data_label = self.label[idx]
#         return data_point, data_label