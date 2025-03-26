import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
import rdkit
from torch_geometric.datasets import MoleculeNet
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class GCN(torch.nn.Module):
    def __init__(self, data, embedding_size):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, data.num_classes)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = torch.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden
    

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(val_loss)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss):
        # Função para salvar o melhor modelo até agora
        print(f"Validation loss improved to {val_loss}. Saving model...")





def train(loader):
    model.train()  # Define o modelo no modo de treinamento
    epoch_loss = 0
    for batch in loader:
        # Enviar batch para o dispositivo (GPU ou CPU)
        batch = batch.to(device)

        # Resetar gradientes
        optimizer.zero_grad()

        # Passar os dados pelo modelo
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)

        # Calcular a perda
        loss = loss_fn(pred, batch.y.float())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Acumular perda total da época
        epoch_loss += loss.item()

    # Retornar a perda média da época
    return epoch_loss / len(loader)


def validate(loader):
    model.eval()  # Define o modelo no modo de avaliação
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y.float())
            val_loss += loss.item()

    return val_loss / len(loader)
# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")
embedding_size = 64


# Instancing the Model
model = GCN(data, embedding_size)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

# Use GPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader

# Parâmetros de separação
train_ratio = 0.7  # Proporção de treino
val_ratio = 0.15   # Proporção de validação
test_ratio = 0.15  # Proporção de teste

# Calcular os índices de separação
data_size = len(data)
train_size = int(data_size * train_ratio)
val_size = int(data_size * val_ratio)
test_size = data_size - train_size - val_size  # Restante para o teste

# Divisão dos dados
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Tamanhos dos lotes
NUM_GRAPHS_PER_BATCH = 64

# Criar os DataLoaders
train_loader = DataLoader(train_data, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
val_loader = DataLoader(val_data, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
test_loader = DataLoader(test_data, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)

# Exibir tamanhos para conferência
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")


# Early Stopping
early_stopping = EarlyStopping(patience=100)

# Loop de treinamento com validação
print("Starting training...")
losses = []
val_losses = []
for epoch in range(2000):
    train_loss = train(train_loader)
    val_loss = validate(val_loader)
    losses.append(train_loss)
    val_losses.append(val_loss)

    # Feedback
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Verificar early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Training stopped.")
        break


# Certificar-se de que as perdas estão no formato de lista de floats
if isinstance(losses[0], torch.Tensor):  # Verificar se os elementos são tensores
    losses_float = [loss.cpu().detach().item() for loss in losses]
else:
    losses_float = [float(loss) for loss in losses]

# Índices para o eixo x
loss_indices = list(range(len(losses_float)))

# Plotar o gráfico
plt.figure(figsize=(10, 6))
sns.lineplot(x=loss_indices, y=losses_float)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row[0])
df["y_pred"] = df["y_pred"].apply(lambda row: row[0])


plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
plt.plot([-7, 2], [-7, 2], color='red', linestyle='-', label='Reference Line')
plt.set_xlabel("Real Values")
plt.set_ylabel("Predicted Values")
plt.set_title("Scatter Plot with Reference Line")
plt.legend()
plt.set(xlim=(-7, 2))
plt.set(ylim=(-7, 2))


if __name__ == "__main__":
    print("Running the script\n")