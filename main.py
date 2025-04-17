import logging 
import argparse 


from utils import setup_logging, initialize_network, load_sampled_input_data

log_it = True 

def main(): 
    parser = argparse.ArgumentParser(description='Train the NEGATRegressor')
    parser.add_argument("--net_name",type=str, help='Pandapower Network Datastructure',default='TOY')
    parser.add_argument("--net_pstd",type=str, help="Load p_std to generate data (0.0 means identical training data)",default=0.3)
    parser.add_argument("--num_samples",type=int, help='Number of Permutations', default=40)
    parser.add_argument("--noise",type=bool, help='Gaussian noise on PFR?', default=True)

    args = parser.parse_args()

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
    
    print(sampled_input_data)
    exit()
    

    
    
    



    



if __name__ == "__main__": 
    main()