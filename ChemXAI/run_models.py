import torch
from torch_geometric.datasets import Planetoid
from src.explainers import GNNEx, Shap
from src.data import prepare_data_graph, qm9_tubular
from src.models import GCN, ImprovedGCN, MLP

def load_model(model, dataset, device):
    model = torch.load(f'{model}_model_{dataset}.pth', map_location=device)

    return model


def main():
    # Teste GCN com CORA e GNNExplainer -> Funcional
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

    # Teste GCN com QM9 e GNNExplainer -> Não Funcional
    # # dataset = prepare_data_graph('QM9')

    # # model = ImprovedGCN(in_channels=dataset.num_features, hidden_channels=128, out_channels=1)
    # # model.load_state_dict(torch.load('models/best_qm9_model.pt', map_location=torch.device('cpu')))
    # # print(model)

    # Teste MLP com QM9 e Shap -> Funcional
    # #  data = qm9_tubular()
    # #  train_loader, _, test_loader, X = data.get_dataloader()
    # #  model = MLP(input_dim=X.shape[1], output_dim=1, layers=[180], device='cpu',lr=0.001)
    # #  print(model)

    # #  shap_explainer = Shap(model=model, train_loader=train_loader, test_loader=test_loader, device='cpu')
    # #  local_explanation = shap_explainer.local_explanation(0)
    # #  print(local_explanation)

    # Teste GCN com QM9 e GNNExplainer ->
    dataset = prepare_data_graph('QM9')

    model = ImprovedGCN(in_channels=dataset.num_features, hidden_channels=128, out_channels=1)
    model.load_state_dict(torch.load('models/ImprovedGCN_model_QM9_2.pth'))
    print(model)

    # model.eval()
    # out = model(dataset.x, dataset.edge_index)
    # print(out)

    exp = GNNEx(model=model, data=dataset, epochs=50, mode='regression', task_level='node')

    exp.explanation(index=10)


if __name__ == '__main__':
    main()