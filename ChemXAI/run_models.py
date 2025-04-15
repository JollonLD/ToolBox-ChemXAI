import torch
from torchinfo import summary
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from src.explainers import GNNEx, Shap, GraphShap, GraphLIME
from src.data import prepare_data_graph, qm9_tubular
from src.models import GCN, MLP


def load_model(model, dataset, device):
    model = torch.load(f'{model}_model_{dataset}.pth', map_location=device)

    return model


def main():
    # Teste MLP com QM9 e Shap -> Funcional
    # #  data = qm9_tubular()
    # #  train_loader, _, test_loader, X = data.get_dataloader()
    # #  model = MLP(input_dim=X.shape[1], output_dim=1, layers=[180], device='cpu',lr=0.001)
    # #  print(model)
    # #  shap_explainer = Shap(model=model, train_loader=train_loader, test_loader=test_loader, device='cpu')
    # #  local_explanation = shap_explainer.local_explanation(0)
    # #  print(local_explanation)



    # Teste GCN com QM9 e GNNExplainer -> Talvez
    # # dataset = prepare_data_graph('QM9')
    # # model = ImprovedGCN(in_channels=dataset.num_features, hidden_channels=128, out_channels=1)
    # # model.load_state_dict(torch.load('models/ImprovedGCN_model_QM9_test.pth', map_location=torch.device('cpu')))
    # # print(model)
    # # # model.eval()
    # # # out = model(dataset.x, dataset.edge_index)
    # # # print(out)
    # # exp = GNNEx(model=model, data=dataset, epochs=50, mode='regression', task_level='node', return_type='raw')
    # # exp.explanation(index=10)

    # Teste GCN com PCQM4 e GraphShap -> Funcional -> Explicação para as características do nó 0 e grafo 0
    # data = prepare_data_graph('PCQM4')
    # model = GCN(num_features=data[0].x.size(1))
    # model.load_state_dict(torch.load('models/gcn_pcqm4.pth'))
    # model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # dataset = data[:100000]
    # explainer = GraphShap(dataset[0], model, device, gpu=torch.cuda.is_available())
    # node_index = 0
    # explanation = explainer.explain(node_index=node_index, hops=2, num_samples=100, multiclass=False)
    # print("Shapley Values for Node Features:")
    # print(explanation['shap_values'])
    # print("\nTop Features per Class (if applicable):")
    # print(explanation['top_features'])
    # print("\nTop Shapley Values per Class (if applicable):")
    # print(explanation['top_values'])

    # Teste GCN com PCQM4 e GraphLIME -> Funcional -> Explicação para as características do nó 0 e grafo 0
    data = prepare_data_graph('PCQM4')
    model = GCN(num_features=data[0].x.size(1))
    model.load_state_dict(torch.load('models/gcn_pcqm4.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataset = data[:100000]
    explainer = GraphLIME(dataset[0], model, device, gpu=torch.cuda.is_available())
    node_index = 0
    explanation = explainer.explain(node_index=node_index, hops=2, num_samples=100, multiclass=False)
    print("Shapley Values for Node Features:")
    print(explanation['shap_values'])
    print("\nTop Features per Class (if applicable):")
    print(explanation['top_features'])
    print("\nTop Shapley Values per Class (if applicable):")
    print(explanation['top_values'])

if __name__ == '__main__':
    main()