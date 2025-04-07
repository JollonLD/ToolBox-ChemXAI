import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
import torch.optim as optim

#================================================================#
# Tubular Models
#================================================================#

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, device, lr=0.001):
        super(MLP, self).__init__()
        all_layers = []
        prev_dim = input_dim

        # Camadas ocultas
        for layer_dim in layers:
            all_layers.append(nn.Linear(prev_dim, layer_dim))
            all_layers.append(nn.ReLU())
            prev_dim = layer_dim

        # Camada de saída (ativação linear - regressão)
        all_layers.append(nn.Linear(prev_dim, output_dim))

        # Combinando as camadas
        self.layers = nn.Sequential(*all_layers)

        self.criterion = nn.L1Loss() #nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, x):
        return self.layers(x)


#================================================================#
# Graph Based Models
#================================================================#

# GCN (Graph Convolutional Network)
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# More complex GCN model
class ImprovedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        # Aumentar a profundidade da rede
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        self.conv4 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        
        # Batch Normalization para estabilizar o treinamento
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels // 2)
        
        # Camada linear para saída
        self.lin = torch.nn.Linear(hidden_channels // 4, out_channels)
        
        # Dropout para regularização
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        # Primeira camada convolucional
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Segunda camada convolucional
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Terceira camada convolucional
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Quarta camada convolucional
        x = self.conv4(x, edge_index, edge_weight)
        x = F.relu(x)
        
        if batch is None:
          # assume que todos os nós pertencem ao mesmo grafo
          batch = x.new_zeros(x.size(0), dtype=torch.long)


        # Pooling - combinando mean pooling com add pooling
        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)
        x = x1 + x2  # Combinação dos dois tipos de pooling
        
        x = self.lin(x)
        
        return x