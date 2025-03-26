import argparse
import os.path as osp
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

# Argumentos do Parser
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--target', type=int, default=0, help="Índice da propriedade alvo (0-18)")
parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do batch para treinamento")
parser.add_argument('--patience', type=int, default=20, help="Paciência para early stopping")
args, _ = parser.parse_known_args()

# Definir dispositivo (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Caminho correto para o dataset
path = osp.join(osp.abspath(''), 'data', 'QM9')

# Carregar o dataset QM9 apenas com normalização de features
dataset = QM9(root=path, transform=T.NormalizeFeatures())

# Dividir os dados em treino, validação e teste
torch.manual_seed(42)  # Para reprodutibilidade
perm = torch.randperm(len(dataset))
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))

train_idx = perm[:train_size]
val_idx = perm[train_size:train_size + val_size]
test_idx = perm[train_size + val_size:]

train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Normalização dos valores alvo (y)
y_train = np.array([data.y[:, args.target].item() for data in train_dataset])
scaler = StandardScaler()
scaler.fit(y_train.reshape(-1, 1))

# Aplicar normalização em todos os conjuntos
for dataset_split in [train_dataset, val_dataset, test_dataset]:
    for i in range(len(dataset_split)):
        data = dataset_split[i]
        if data.y is not None:
            y_normalized = scaler.transform(data.y[:, args.target].view(-1, 1).numpy())
            data.y[:, args.target] = torch.FloatTensor(y_normalized).view(-1)

print(f"Dataset QM9 carregado! Tamanho do treino: {len(train_dataset)}, validação: {len(val_dataset)}, teste: {len(test_dataset)}")
print(f"Propriedade alvo escolhida: {args.target}")
print(f"Número de características dos nós: {dataset.num_features}")

# Definir modelo GCN melhorado
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

    def forward(self, x, edge_index, batch, edge_weight=None):
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
        
        # Pooling - combinando mean pooling com add pooling
        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)
        x = x1 + x2  # Combinação dos dois tipos de pooling
        
        # Camada linear final
        x = self.lin(x)
        
        return x

# Inicializar o modelo
model = ImprovedGCN(
    in_channels=dataset.num_features, 
    hidden_channels=args.hidden_channels, 
    out_channels=1
).to(device)

# Otimizador com learning rate scheduling
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# Função de treino
def train(epoch):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Verificar se edge_attr existe e tem pelo menos uma coluna
        if data.edge_attr is not None and data.edge_attr.size(1) > 0:
            edge_weight = data.edge_attr[:, 0]
        else:
            edge_weight = None
            
        pred = model(data.x, data.edge_index, data.batch, edge_weight=edge_weight)
        
        target = data.y[:, args.target].unsqueeze(1)
        loss = F.mse_loss(pred, target)
        
        loss.backward()
        # Clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(train_dataset)

# Função de teste
@torch.no_grad()
def test(loader):
    model.eval()
    total_mae = 0
    total_mse = 0
    
    for data in loader:
        data = data.to(device)
        
        # Verificar se edge_attr existe e tem pelo menos uma coluna
        if data.edge_attr is not None and data.edge_attr.size(1) > 0:
            edge_weight = data.edge_attr[:, 0]
        else:
            edge_weight = None
            
        pred = model(data.x, data.edge_index, data.batch, edge_weight=edge_weight)
        
        target = data.y[:, args.target].unsqueeze(1)
        
        # Calcular MAE e MSE
        total_mae += F.l1_loss(pred, target).item() * data.num_graphs
        total_mse += F.mse_loss(pred, target).item() * data.num_graphs
    
    # Normalizado pelo tamanho do dataset
    mae = total_mae / len(loader.dataset)
    rmse = np.sqrt(total_mse / len(loader.dataset))
    
    return mae, rmse

# Loop de treinamento com early stopping
def execute():
    best_val_mae = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_maes = []
    val_rmses = []
    test_maes = []
    test_rmses = []
    
    print("\nIniciando treinamento...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Treinar modelo
        train_loss = train(epoch)
        train_losses.append(train_loss)
        
        # Avaliar no conjunto de validação
        val_mae, val_rmse = test(val_loader)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        
        # Avaliar no conjunto de teste
        test_mae, test_rmse = test(test_loader)
        test_maes.append(test_mae)
        test_rmses.append(test_rmse)
        
        # Ajustar learning rate com o scheduler
        scheduler.step(val_mae)
        
        # Salvar melhor modelo baseado no MAE de validação
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_test_mae = test_mae
            best_test_rmse = test_rmse
            best_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'test_mae': test_mae
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log do progresso
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, "
                  f"Test MAE: {test_mae:.4f}, LR: {lr:.6f}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered! Sem melhoria por {args.patience} epochs.")
            break
    
    # Carregar o melhor modelo
    model.load_state_dict(best_model['model_state_dict'])
    
    # Tempo total de treinamento
    total_time = time.time() - start_time
    print(f"\nTreinamento concluído em {total_time:.2f} segundos")
    print(f"Melhor modelo na epoch {best_model['epoch']} com Val MAE: {best_val_mae:.4f}, Test MAE: {best_test_mae:.4f}, Test RMSE: {best_test_rmse:.4f}")
    
    # Salvar o modelo
    torch.save(best_model, 'best_qm9_model.pt')
    
    # Plotar resultados
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Loss de Treinamento')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_maes, label='Validation MAE')
    plt.plot(epochs_range, test_maes, label='Test MAE')
    plt.axvline(x=best_model['epoch'], color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('MAE de Validação e Teste')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('qm9_training_results.png')
    plt.show()
    
    return best_model

# Executar o treinamento
if __name__ == '__main__':
    best_model = execute()