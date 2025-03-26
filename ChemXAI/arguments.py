import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    # Utils params
    utils_parser = parser.add_argument_group('utils')
    utils_parser.add_argument('--gpu', default=False, help='Select between GPU or CPU')
    utils_parser.add_argument("--save", type=str, help="True to save the trained model obtained")

    # Training params

    # Explainers params
    parser.add_argument("--model", type=str,
                        help="Name of the model: MLP, GCN or ImprovedGCN")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset among Cora, PubMed, syn1-6, Mutagenicity")
    parser.add_argument("--indexes", type=list, default=[0],
                        help="indexes of the instance/nodes/graphs whose prediction are explained")
    parser.add_argument("--hops", type=int,
                        help="number k for k-hops neighbours considered in an explanation")
    parser.add_argument("--num_samples", type=int,
                        help="number of coalitions sampled and used to approx shapley values")
    parser.add_argument("--feat", type=str,
                        help="method used to determine the features considered")
    parser.add_argument("--multiclass", type=bool,
                        help='False if we consider explanations for the predicted class only')
    parser.add_argument("--info", type=bool,
                        help='True if want to print info')
    
    parser.set_defaults(gpu=False,
                        save=False,
                        model='GCN',
                        indexes=[500,600],
                        hops=3,
                        num_samples=400,
                        feat='Expectation',
                        multiclass=False,
                        info=False,
                        )