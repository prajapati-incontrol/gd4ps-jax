import logging 
import argparse 
from flax import nnx 
import jax 

# from src.model.graph_model import NEGATRegressorJAX
from src.dataset.custom_dataset import CustomDataset
from utils import setup_logging, initialize_network, load_sampled_input_data, dataset_splitter, get_device
from src.model.graph_model import GATConvJax


log_it = True 

def main(): 
    parser = argparse.ArgumentParser(description='Train the NEGATRegressor')
    parser.add_argument("--net_name",type=str, help='Pandapower Network Datastructure',default='TOY')
    parser.add_argument("--net_pstd",type=str, help="Load p_std to generate data (0.0 means identical training data)",default=0.3)
    parser.add_argument("--num_samples",type=int, help='Number of Permutations', default=100)
    parser.add_argument("--noise",type=bool, help='Gaussian noise on PFR?', default=True)
    parser.add_argument("--batch_size",type=int, help="Batch-size for model training",default=64)

    args = parser.parse_args()

    device = get_device("cpu")

    if log_it: 
        setup_logging()
    
    # initialize network 
    net = initialize_network(args.net_name)

    # load sampled input data 
    sampled_input_data = load_sampled_input_data(net=net, 
                                                 num_samples=args.num_samples, 
                                                 load_std=args.net_pstd, 
                                                 noise=args.noise, 
                                                 )

    # print(sampled_input_data['scaler_node'])
    # exit()

    # create Custom Dataset 
    dataset = CustomDataset(sampled_input_data=sampled_input_data)

    # dataloader 
    train_loader, val_loader, test_loader = dataset_splitter(dataset,
                                                             batch_size=args.batch_size)

    # batch = next(iter(train_loader))
    # print(batch[0].edge_index)
    # exit()
    # print(len(test_loader))
    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(dataset))
    # exit()

    # instantiate the model 
    node_out_features = 32
    edge_out_features = 32
    list_node_hidden_features = [32]
    list_edge_hidden_features = [32]
    k_hop_node = 2 
    k_hop_edge = 1
    gat_out_features = 64
    gat_head = 1


    model = GATConvJax(in_channels=dataset[0][0].x.shape[1], 
                       out_channels=node_out_features, 
                       heads=2, 
                       edge_dim=dataset[0][1].x.shape[1], 
                       rngs=nnx.Rngs(jax.random.key(42)))
    
    print(nnx.display(model))





    



if __name__ == "__main__": 
    main()