import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models import GCN
from data import prepare_data_graph

def train_gcn_qm9(target_idx=3, epochs=10, batch_size=64, lr=0.001, weight_decay=1e-4):
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
            torch.save(model.state_dict(), '../models/gcn_qm9.pth')

        print(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 6. Avaliação final
    model.load_state_dict(torch.load('../models/gcn_qm9.pth'))
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
            torch.save(model.state_dict(), 'models/gcn_pcqm4.pth')

        print(f"[{epoch:02d}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 5. Avaliação final (com drop_last=True, garante mesmo tamanho)
    model.load_state_dict(torch.load('models/gcn_pcqm4.pth'))
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
    train_gcn_qm9()