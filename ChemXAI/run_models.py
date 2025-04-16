import torch
from torchinfo import summary
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from src.explainers import GNNEx, Shap, GraphShap, GraphLIME, GraphLIMEGraphLevel
from src.data import prepare_data_graph, qm9_tubular
from src.models import GCN, MLP


def main():
    # Teste MLP com QM9 e Shap -> Funcional -> Explicação para as features da instancia 0 do dataset
    # #  data = qm9_tubular()
    # #  train_loader, _, test_loader, X = data.get_dataloader()
    # #  model = MLP(input_dim=X.shape[1], output_dim=1, layers=[180], device='cpu',lr=0.001)
    # #  print(model)
    # #  shap_explainer = Shap(model=model, train_loader=train_loader, test_loader=test_loader, device='cpu')
    # #  local_explanation = shap_explainer.local_explanation(0)
    # #  print(local_explanation)




    # Teste GCN com QM9 e GraphShap -> Funcional -> Explicação para as features do nó 0 do grafo 0
    # data = prepare_data_graph('QM9')
    # dataset = data[:100000]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_qm9.pth'))
    # explainer = GraphShap(dataset[0], model, device, gpu=torch.cuda.is_available())
    # node_index = 0
    # explanation = explainer.explain(node_index=node_index, hops=2, num_samples=100, multiclass=False)
    # print("Shapley Values for Node Features:")
    # print(explanation['shap_values'])
    # print("\nTop Features per Class (if applicable):")
    # print(explanation['top_features'])
    # print("\nTop Shapley Values per Class (if applicable):")
    # print(explanation['top_values'])




    # Teste GCN com QM9 e GrahLIME -> Funcional -> Explicação para as características dos nós do grafo 0
    # data = prepare_data_graph('QM9')
    # dataset = data[:100000]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_qm9.pth'))
    # explainer = GraphLIMEGraphLevel(model=model, device=device, rho=0.1)
    # explanation = explainer.explain(dataset[0], num_samples=100)
    # print("Features Importance:")
    # print(explanation['feature_importance'])
    # print("\nTop Features:")
    # print(explanation['top_features'])
    # print("\nPCA coeficients:")
    # print(explanation['coef_matrix'])




    # Teste GCN com QM9 e GNNExplainer -> Funcional -> Explicação para as features do nó 0 do grafo 0
    # dataset = prepare_data_graph('QM9')
    # data = dataset[0]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_qm9.pth'))
    # exp = GNNEx(model=model, data=data, device=device, epochs=20, mode='regression', task_level='node', return_type='raw')
    # nodes_exp, pred = exp.explanation(index=0)
    # # model.eval()
    # # out = model(data.x, data.edge_index)
    # # print(out)
    # print(f'Explicação para o Nó [0] {nodes_exp[0]}')




    # Teste GCN com PCQM4 e GraphShap -> Funcional -> Explicação para as características do nó 0 do grafo 0
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




    # Teste GCN com PCQM4 e GraphLIME -> Funcional -> Explicação para as características dos nós do grafo 0
    # data = prepare_data_graph('PCQM4')
    # dataset = data[:100000]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_pcqm4.pth'))
    # explainer = GraphLIMEGraphLevel(model=model, device=device, rho=0.1)
    # explanation = explainer.explain(dataset[0], num_samples=100)
    # print("Features Importance:")
    # print(explanation['feature_importance'])
    # print("\nTop Features:")
    # print(explanation['top_features'])
    # print("\nPCA coeficients:")
    # print(explanation['coef_matrix'])




    # Teste GCN com PCQM4 e GNNExplainer -> Funcional -> Explicação para as características do nó 0 do grafo 0
    data = prepare_data_graph('PCQM4')
    model = GCN(num_features=data[0].x.size(1))
    model.load_state_dict(torch.load('models/gcn_pcqm4.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataset = data[:100000]
    explainer = GNNEx(model=model, data=dataset[0], epochs=20, mode='regression', task_level='node', return_type='raw')
    explainer.explanation(index=0)




if __name__ == '__main__':
    main()