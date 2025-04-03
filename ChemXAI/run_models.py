import torch
from torch_geometric.datasets import Planetoid
from src.explainers import GNNEx
from src.data import prepare_data_graph
from src.models import GCN, ImprovedGCN

def load_model(model, dataset, device):
    model = torch.load(f'{model}_model_{dataset}.pth', map_location=device)

    return model


def main():
    # Teste GCN com CORA
    # # dataset = Planetoid(root='data/Cora', name='Cora')
    # # data = dataset[0]

    # # model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)
    # # model.load_state_dict(torch.load('models/GCN_model_Cora.pth'))
    # # print(model)

    # # model.eval()
    # # out = model(data.x, data.edge_index)
    # # print(out)

    # # exp = GNNEx(model=model, data=data, epochs=200, mode='multiclass_classification', task_level='node')

    # # exp.explanation(index=10)

    # Teste GCN com QM9
    dataset = prepare_data_graph('QM9')

    model = ImprovedGCN(in_channels=dataset.num_features, hidden_channels=128, out_channels=1)
    model.load_state_dict(torch.load('models/best_qm9_model.pt', map_location=torch.device('cpu')))
    print(model)
    
if __name__ == '__main__':
    main()