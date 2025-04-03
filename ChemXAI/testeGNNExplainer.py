import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import GNNExplainer, Explainer

# Definindo o modelo GCN (Graph Convolutional Network)
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Carregar dados do conjunto de dados Planetoid (Cora, Citeseer, Pubmed)
dataset = Planetoid(root='data/Cora',name='Cora')
data = dataset[0]

# Criando o modelo
model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)

# Definindo o GNNExplainer
class GNNEx():
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Cria o explainer
        self.explainer = Explainer(
            model=model,
            algorithm= GNNExplainer(epochs=200),  # Especificando o algoritmo GNNExplainer
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',  # O modelo retorna log probabilidades.
            ),
        )

    def explanation(self):
        # Gerar explicação para o nó no índice 10 (pode ser qualquer índice)
        explanation = self.explainer(self.data.x, self.data.edge_index, index=10)
        
        print("Edge Mask:", explanation.edge_mask)
        print("Node Mask:", explanation.node_mask)

        # Visualiza a importância das características
        explanation.visualize_feature_importance(top_k=10)

        # Visualiza o grafo
        explanation.visualize_graph()


def main():
    # Treine o modelo com os dados
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'GCN_model_Cora.pth')
    # Imprime as previsões do modelo
    model.eval()
    out = model(data.x, data.edge_index)
    print(out)

    # Cria o objeto GNNExplainer e gera as explicações
    exp = GNNEx(model=model, data=data)
    exp.explanation()

if __name__ == '__main__':
    main()
