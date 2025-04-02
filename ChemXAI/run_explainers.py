import torch

import arguments
from src.data import prepare_data
from src.train import evaluate, test

def main():
    args = arguments.arg_parse()

    # Load the dataset
    data = prepare_data(args.dataset, args.train_ratio,
                        args.input_dim, args.seed)

    # Load the model
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    model = torch.load(model_path)
    
    # Evaluate the model 
    if args.dataset in ['Cora', 'PubMed']:
        _, test_acc = evaluate(data, model, data.test_mask)
    else: 
        test_acc = test(data, model, data.test_mask)
    print('Test accuracy is {:.4f}'.format(test_acc))

if __name__ == '__main__':
    main()
