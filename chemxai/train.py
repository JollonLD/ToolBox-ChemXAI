import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models import GCN, MLP
from data import prepare_data_graph, qm9_tubular

def train_mlp_qm9(att_index=10, epochs=10, layers=[64, 32], learning_rate=1e-3, batch_size=32):
    """
    Função para treinar um modelo MLP.

    Args:
        model (nn.Module): O modelo MLP a ser treinado.
        train_loader (torch.utils.data.DataLoader): DataLoader para o conjunto de treinamento.
        val_loader (torch.utils.data.DataLoader): DataLoader para o conjunto de validação.
        epochs (int): Número de épocas de treinamento.
        device (torch.device): Dispositivo para o qual os tensores serão movidos ('cuda' ou 'cpu').

    Returns:
        list: Uma lista de tuplas (epoch, train_loss, val_loss) para cada época.
    """

    # Carregar os dados
    qm9 = qm9_tubular()

    train_loader, val_loader, test_loader, X_original = qm9.get_dataloader(
        att_index=att_index,           # Índice da propriedade a ser prevista
        batch_size=batch_size,         # Tamanho do lote
        descriptor_type='CM',   # Usar Coulomb Matrix como descritor
        list_mols=[]            # Lista vazia = todas as moléculas
    )

    # Definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    # Obter a dimensão da entrada (tamanho do descritor CM)
    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = 1  # Previsão de uma única propriedade

    # Definir a arquitetura da MLP
    model = MLP(input_dim, output_dim, layers, device, lr=learning_rate)

    history = []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0] # Xn
            targets = batch[1] # Yn_scaled

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.criterion(outputs, targets)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0] # Xn
                targets = batch[1] # Yn_scaled

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = model.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        history.append((epoch + 1, train_loss, val_loss))
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return history


def train_gcn_qm9(target_idx=3, epochs=10, batch_size=64, lr=0.001, weight_decay=1e-4):
    dirname = os.getcwd()
    # Cria o caminho completo para o diretório 'models' dentro do 'dirname'
    models_dir = os.path.join(dirname, 'models')
    # Cria o diretório 'models', se não existir
    os.makedirs(models_dir, exist_ok=True)
    print(f'Diretório criado: {models_dir}')

    # Detectar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # 1. Carregar e transformar o dataset
    dataset = prepare_data_graph('QM9')

    # 2. Normalizar o alvo
    y = dataset.data.y[:, target_idx]
    mean = y.mean()
    std = y.std()
    dataset.data.y = (y - mean) / std

    # 3. Dividir dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 4. Instanciar modelo para regressão
    model = GCN(
        num_features=dataset.data.x.size(1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 5. Treinamento
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch).view(-1)
            target = batch.y.view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch).view(-1)
                target = batch.y.view(-1)
                loss = criterion(pred, target)
                val_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = dirname + '/models/gcn_qm9.pth'
            torch.save(model.state_dict(), path)

        print(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 6. Avaliação final
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch).view(-1)
            pred = pred * std + mean
            y_true = batch.y * std + mean
            test_loss += F.mse_loss(pred, y_true).item() * batch.num_graphs

    test_loss /= len(test_loader.dataset)
    print(f"\nMSE no teste: {test_loss:.4f}")
    print(f"RMSE no teste: {test_loss ** 0.5:.4f}")


def train_gcn_pcqm4(epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-4):
    dirname = os.getcwd()
    # Cria o caminho completo para o diretório 'models' dentro do 'dirname'
    models_dir = os.path.join(dirname, 'models')
    # Cria o diretório 'models', se não existir
    os.makedirs(models_dir, exist_ok=True)
    print(f'Diretório criado: {models_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # 1. Carregamento e redução do dataset
    dataset = prepare_data_graph('PCQM4')
    dataset = dataset[:100000]  # reduzido para testes rápidos

    # 2. Divisão
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 3. Loaders com drop_last para evitar batches incompletos
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    model = GCN(num_features=dataset[0].x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 4. Treinamento
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.batch).view(-1)
            target = data.y.view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.batch).view(-1)
                target = data.y.view(-1)
                loss = criterion(pred, target)
                val_loss += loss.item() * data.num_graphs

        avg_val_loss = val_loss / len(val_loader.dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = dirname + '/models/gcn_pcqm4.pth'
            torch.save(model.state_dict(), path)

        print(f"[{epoch:02d}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 5. Avaliação final (com drop_last=True, garante mesmo tamanho)
    model.load_state_dict(torch.load(path))
    model.eval()
    final_val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).view(-1)
            target = data.y.view(-1)
            final_val_loss += criterion(pred, target).item() * data.num_graphs

    final_val_loss /= len(val_loader.dataset)
    print(f"\nMSE final na validação: {final_val_loss:.4f}")
    print(f"RMSE final na validação: {final_val_loss ** 0.5:.4f}")



if __name__ == '__main__':
    train_mlp_qm9()