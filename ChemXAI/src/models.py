import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.nn import MessagePassing, global_add_pool

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


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=256):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, 1)
        self.dropout = Dropout(0.3)

    def forward(self, x, edge_index, batch=None):
        # Se batch não for passado, assume-se um grafo único
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_add_pool(x, batch)

        x = self.dropout(F.relu(self.lin1(x)))
        x = self.lin2(x)

        return x.view(-1)



class MolecularGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, output_dim=1):
        super(MolecularGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Pooling global para obter representação do grafo inteiro
        self.pool = global_mean_pool
        
        # Camadas de regressão
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)  # output_dim=1 para regressão
        
    def forward(self, x, edge_index, batch=None):
        # Convolução de grafo
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Pooling global (necessário para regressão de grafos)
        x = self.pool(x, batch)
        
        # MLP para regressão
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        
        return x
    

# For Graph Classification
class MoleculeGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super(MoleculeGNN, self).__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Readout layer (graph-level pooling)
        self.pool = global_mean_pool
        
        # Classification layers
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch=None):
        # Node embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Readout (aggregate node features to graph features)
        x = self.pool(x, batch)
        
        # Classification
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
    

class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim):
        super(MPNNLayer, self).__init__(aggr='add')
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.res_connection = nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        inputs = torch.cat([x_j, edge_attr], dim=1)
        return self.node_mlp(inputs)

    def update(self, aggr_out, x):
        return aggr_out + self.res_connection(x)


class ChemicalMPNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, output_dim=1, hidden_dim=64):
        super(ChemicalMPNN, self).__init__()
        self.mp1 = MPNNLayer(num_node_features, num_edge_features, hidden_dim)
        self.mp2 = MPNNLayer(hidden_dim, num_edge_features, hidden_dim)
        self.mp3 = MPNNLayer(hidden_dim, num_edge_features, hidden_dim)
        
        self.readout = global_add_pool
        
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.mp1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.mp2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.mp3(x, edge_index, edge_attr)
        
        x = self.readout(x, batch)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        
        return x