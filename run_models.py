import torch
#from torchinfo import summary
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from chemxai.explainers import Shap, GNNExplain, NodeGraphShap, GraphShap, NodeGrapLIME, GraphLIME
from chemxai.data import prepare_data_graph, qm9_tabular
from chemxai.models import GCN, MLP
from chemxai.train import train_gcn_pcqm4, train_gcn_qm9, train_mlp_qm9

def main():

    # Train_models
    # train_gcn_qm9()
    # train_gcn_pcqm4()
    train_mlp_qm9()

    # Teste MLP com QM9 e Shap -> Funcional -> Explicação para as features da instancia 0 do dataset
    # #  data = qm9_tubular()
    # #  train_loader, _, test_loader, X = data.get_dataloader()
    # #  model = MLP(input_dim=X.shape[1], output_dim=1, layers=[180], device='cpu',lr=0.001)
    # #  print(model)
    # #  shap_explainer = Shap(model=model, train_loader=train_loader, test_loader=test_loader, device='cpu')
    # #  local_explanation = shap_explainer.local_explanation(0)
    # #  print(local_explanation)




    # Teste GCN com QM9 e GraphShap -> Funcional -> Explicação para as features do grafo 0
    # data = prepare_data_graph('QM9')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_qm9.pth', map_location=torch.device(device)))
    # model = model.to(device)
    # explainer = GraphShap(data[0], model, device, gpu=torch.cuda.is_available())
    # explanation = explainer.explain()
    # print("Shapley Values for Node Features:")
    # print(explanation['shap_values'])
    # print("\nTop Features (if applicable):")
    # print(explanation['top_features'])
    # print("\nTop Shapley Values (if applicable):")
    # print(explanation['top_values'])




    # Teste GCN com QM9 e GrahLIME -> Funcional -> Explicação para as características do grafo 0
    # data = prepare_data_graph('QM9')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_qm9.pth', map_location=torch.device(device)))
    # model = model.to(device)
    # explainer = GraphLIME(model=model, device=device, rho=0.1)
    # explanation = explainer.explain(data[0], num_samples=100)
    # print("Features Importance:")
    # print(explanation['feature_importance'])
    # print("\nTop Features:")
    # print(explanation['top_features'])




    # Teste GCN com QM9 e GNNExplainer -> Funcional -> Explicação para as features do nó 0 do grafo 0
    # data = prepare_data_graph('QM9')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_qm9.pth', map_location=torch.device(device)))
    # model = model.to(device)
    # exp = GNNExplainer(model=model, data=data[0], device=device, epochs=20, mode='regression', task_level='graph', return_type='raw')
    # graph_exp, pred = exp.explanation(index=0)
    # print(f'Explicação para o Grafo 0 {graph_exp[0]}')




    # Teste GCN com PCQM4 e GraphShap -> Funcional -> Explicação para as características do nó 0 do grafo 0
    # data = prepare_data_graph('PCQM4')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data[0].x.size(1))
    # model.load_state_dict(torch.load('models/gcn_pcqm4.pth', map_location=torch.device(device)))
    # model = model.to(device)
    # explainer = GraphShap(data[0], model, device, gpu=torch.cuda.is_available())
    # explanation = explainer.explain()
    # print("Shapley Values for Node Features:")
    # print(explanation['shap_values'])
    # print("\nTop Features (if applicable):")
    # print(explanation['top_features'])
    # print("\nTop Shapley Values (if applicable):")
    # print(explanation['top_values'])




    # Teste GCN com PCQM4 e GraphLIME -> Funcional -> Explicação para as características dos nós do grafo 0
    # data = prepare_data_graph('PCQM4')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data.num_features).to(device)
    # model.load_state_dict(torch.load('models/gcn_pcqm4.pth', map_location=torch.device(device)))
    # explainer = GraphLIME(model=model, device=device, rho=0.1)
    # explanation = explainer.explain(data[0], num_samples=100)
    # print("Features Importance:")
    # print(explanation['feature_importance'])
    # print("\nTop Features:")
    # print(explanation['top_features'])




    # Teste GCN com PCQM4 e GNNExplainer -> Funcional -> Explicação para as características do grafo 0
    # data = prepare_data_graph('PCQM4')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN(num_features=data[0].x.size(1))
    # model.load_state_dict(torch.load('models/gcn_pcqm4.pth', map_location=torch.device(device)))
    # model = model.to(device)
    # exp = GNNExplain(model=model, data=data[0], device=device, epochs=20, mode='regression', task_level='graph', return_type='raw')
    # graph_exp, pred = exp.explain(index=0)
    # print(f'Explicação para o Grafo 0 {graph_exp[0]}')



if __name__ == '__main__':
    main()                                                                                                                                                                      
