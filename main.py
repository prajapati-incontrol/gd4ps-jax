import logging 
import argparse 
from flax import nnx 
import jax 

# from src.model.graph_model import NEGATRegressorJAX
from src.dataset.custom_dataset import NodeEdgeDataset
from utils import setup_logging, initialize_network, load_sampled_input_data, dataset_splitter, get_device
from utils import load_config 
from src.model.graph_model import GATConvJax


def main():

    config = load_config()
    device = get_device(config['device']) 

    print(f"Using device: {device}")

    net = initialize_network(config['data']['net_name'])



    sampled_input_data = load_sampled_input_data(sc_type=config['data']['scenario_type'], 
                                                 net=net, 
                                                 num_samples=config['data']['num_samples'],
                                                 p_std=config['data']['net_load_std'],
                                                 noise=config['data']['noise'],
                                                 trafo_ids=config['data']['trafo_ids'],
                                                 scaler=config['data']['scaler'],
                                                 )


    # instantiate the dataset 
    dataset = NodeEdgeDataset(model_name=config['model']['name'], sampled_input_data=sampled_input_data)


    all_loaders, plot_loader = dataset_splitter(dataset,
                                    batch_size=config['loader']['batch_size'], 
                                    split_list=config['loader']['split_list'])

    model = GATConvJax(in_channels=dataset[0][0].x.shape[1], 
                       out_channels=config['model']['node_out_features'], 
                       heads=config['model']['gat_head'], 
                       edge_dim=dataset[0][1].x.shape[1], 
                       rngs=nnx.Rngs(jax.random.key(42)))
    
    print(nnx.display(model))
    
if __name__ == "__main__":
    main()
