import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from .models import GCN, MLP
from .data import graph_datasets, qm9_tabular

def train_mlp_qm9(att_index=10, epochs=10, layers=[64, 32], learning_rate=1e-3, batch_size=32, n_noise=3, descriptor_type='Morgan'):
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

    dirname = os.getcwd()
    models_dir = os.path.join(dirname, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f'Diretório criado: {models_dir}')
    
    if n_noise > 0:
        path = models_dir + '/mlp_qm9_noise_' + descriptor_type + '.pth'
    else:
        path = models_dir + '/mlp_qm9_' + descriptor_type + '.pth'

    # Carregar os dados
    qm9 = qm9_tabular()
    if n_noise > 0:
        _, _, _, train_loader, val_loader, test_loader, _ = qm9.get_paired_dataloaders(
            att_index=att_index,           # Índice da propriedade a ser prevista
            batch_size=batch_size,         # Tamanho do lote
            descriptor_type=descriptor_type,          
            list_mols=[],                  # Lista vazia = todas as moléculas
            n_noise=n_noise,
            morgan_radius=2, morgan_nBits=512             
        )
    else:    
        train_loader, val_loader, test_loader = qm9.get_paired_dataloaders(
            att_index=att_index,           # Índice da propriedade a ser prevista
            batch_size=batch_size,         # Tamanho do lote
            descriptor_type=descriptor_type,          # Usar Coulomb Matrix como descritor
            list_mols=[],                   # Lista vazia = todas as moléculas
            n_noise=n_noise,
            morgan_radius=2, morgan_nBits=512
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

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device) # Xn
            targets = batch[1].to(device) # Yn_scaled

            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.criterion(outputs, targets)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device) # Xn
                targets = batch[1].to(device) # Yn_scaled

                outputs = model(inputs)
                loss = model.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        history.append((epoch + 1, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, path)
    
        print(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Avaliação final
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            preds = model(inputs)
            test_loss += F.mse_loss(preds, targets).item() * inputs.size(0)
            
    test_loss = test_loss / len(test_loader.dataset)
    
    print(f"\nMSE no teste: {test_loss:.4f}")
    print(f"RMSE no teste: {test_loss ** 0.5:.4f}")

    return history

def train_gcn_qm9(target_idx=3, epochs=10, batch_size=64, lr=0.001, weight_decay=1e-4, n_noise=0):
    
    dirname = os.getcwd()
    models_dir = os.path.join(dirname, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f'Diretório criado: {models_dir}')
    
    if n_noise > 0:
        path = models_dir + '/gcn_qm9_noise.pth'
    else:
        path = models_dir + '/gcn_qm9.pth'

    # Detectar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    gd = graph_datasets()

    # Usar get_paired_dataloaders para obter dados alinhados
    loaders = gd.get_paired_dataloaders(
        dataset_name='QM9', 
        batch_size=batch_size,
        seed=42,
        noise_type='gaussian',
        noise_scale=1.0
    )
    
    # Decidir quais loaders usar com base na presença de ruído
    if n_noise > 0:
        _, _, _, train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader, test_loader, _, _, _ = loaders

    # Obter exemplo do primeiro batch para inicializar o modelo
    data_sample = next(iter(train_loader))
    
    # 4. Instanciar modelo para regressão
    model = GCN(
        num_features=data_sample.x.size(1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 5. Treinamento
    best_val_loss = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Selecionar apenas a propriedade específica que queremos prever
            target = batch.y[:, target_idx].view(-1, 1)
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Garantir que as dimensões correspondam
            if out.shape[0] != target.shape[0]:
                min_size = min(out.shape[0], target.shape[0])
                out = out[:min_size]
                target = target[:min_size]
            
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * out.size(0)
            num_samples += out.size(0)

        avg_train_loss = total_loss / num_samples

        # Validação
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Selecionar a propriedade alvo específica
                target = batch.y[:, target_idx].view(-1, 1)
                
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                
                # Garantir que as dimensões correspondam
                if out.shape[0] != target.shape[0]:
                    min_size = min(out.shape[0], target.shape[0])
                    out = out[:min_size]
                    target = target[:min_size]
                
                loss = criterion(out, target)
                val_loss += loss.item() * out.size(0)
                val_samples += out.size(0)

        avg_val_loss = val_loss / val_samples

        history.append((epoch + 1, avg_train_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path)

        print(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 6. Avaliação final
    model.load_state_dict(torch.load(path))
    model.eval()
    test_loss = 0
    test_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Selecionar a propriedade alvo específica
            target = batch.y[:, target_idx].view(-1, 1)
            
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Garantir que as dimensões correspondam
            if out.shape[0] != target.shape[0]:
                min_size = min(out.shape[0], target.shape[0])
                out = out[:min_size]
                target = target[:min_size]
            
            loss = F.mse_loss(out, target).item()
            test_loss += loss * out.size(0)
            test_samples += out.size(0)

    test_loss /= test_samples
    print(f"\nMSE no teste: {test_loss:.4f}")
    print(f"RMSE no teste: {test_loss ** 0.5:.4f}")

    return history

# Update the create_dataloaders_from_augmented_graphs function
def create_dataloaders_from_augmented_graphs(augmented_graphs, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Separa os grafos aumentados em conjuntos de treino, validação e teste,
    e cria dataloaders para cada conjunto.
    """
    # Verificar se as proporções somam 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "As proporções devem somar 1"
    
    # Definir semente para reprodutibilidade
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calcular o número de amostras para cada conjunto
    num_samples = len(augmented_graphs)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    num_test = num_samples - num_train - num_val
    
    # Embaralhar índices
    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:]
    
    # Criar subconjuntos
    train_dataset = [augmented_graphs[i] for i in train_indices]
    val_dataset = [augmented_graphs[i] for i in val_indices]
    test_dataset = [augmented_graphs[i] for i in test_indices]
    
    # Criar dataloaders com o collate function personalizado
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        exclude_keys = ['smiles', 'augmentation_method', 'parent_idx']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        exclude_keys = ['smiles', 'augmentation_method', 'parent_idx']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        exclude_keys = ['smiles', 'augmentation_method', 'parent_idx']
    )
    
    print(f"Dataloaders criados: {len(train_dataset)} treino, {len(val_dataset)} validação, {len(test_dataset)} teste")
    
    return train_loader, val_loader, test_loader

def rebuild_graph_dataset(augmented_graphs):
    """Create a new dataset with consistently structured graphs."""
    from torch_geometric.data import Data
    
    new_graphs = []
    
    for i, graph in enumerate(augmented_graphs):
        try:
            # Extract essential attributes
            x = graph.x.clone() if hasattr(graph, 'x') and graph.x is not None else None
            edge_index = graph.edge_index.clone() if hasattr(graph, 'edge_index') and graph.edge_index is not None else None
            edge_attr = graph.edge_attr.clone() if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
            
            # For y, handle both tensor and primitive values
            y = None
            if hasattr(graph, 'y') and graph.y is not None:
                if torch.is_tensor(graph.y):
                    y = graph.y.clone()
                else:
                    y = graph.y  # Just copy the primitive value
            
            # Calculate num_nodes
            num_nodes = None
            if hasattr(graph, 'x') and graph.x is not None:
                num_nodes = graph.x.size(0)
            elif hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.size(1) > 0:
                num_nodes = int(graph.edge_index.max().item()) + 1
                
            # Create a new Data object with only essential attributes
            new_graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=num_nodes
            )
            
            new_graphs.append(new_graph)
            
        except Exception as e:
            print(f"Error processing graph {i}: {e}")
            continue
            
    print(f"Successfully rebuilt {len(new_graphs)} out of {len(augmented_graphs)} graphs")
    return new_graphs

def normalize_features(graphs):
    """Normalize node features to have zero mean and unit variance."""
    all_features = []
    for graph in graphs:
        if hasattr(graph, 'x') and graph.x is not None:
            all_features.append(graph.x)
    
    if not all_features:
        return graphs
    
    all_features_tensor = torch.cat(all_features, dim=0)
    mean = all_features_tensor.mean(dim=0)
    std = all_features_tensor.std(dim=0)
    std[std < 1e-5] = 1.0  # Prevent division by zero
    
    for graph in graphs:
        if hasattr(graph, 'x') and graph.x is not None:
            graph.x = (graph.x - mean) / std
    
    return graphs

# And normalize target values
def normalize_targets(graphs):
    """Normalize target values."""
    all_targets = []
    for graph in graphs:
        if hasattr(graph, 'y') and graph.y is not None:
            if torch.is_tensor(graph.y):
                all_targets.append(graph.y)
            else:
                # Handle scalar values
                all_targets.append(torch.tensor([graph.y], dtype=torch.float))
    
    if not all_targets:
        return graphs, None, None
    
    all_targets_tensor = torch.cat([t.view(1) for t in all_targets], dim=0)
    mean = all_targets_tensor.mean()
    std = all_targets_tensor.std()
    if std < 1e-5:
        std = 1.0  # Prevent division by zero
    
    for graph in graphs:
        if hasattr(graph, 'y') and graph.y is not None:
            if torch.is_tensor(graph.y):
                graph.y = (graph.y - mean) / std
            else:
                graph.y = (graph.y - mean.item()) / std.item()
    
    return graphs, mean.item(), std.item()

def get_test_loader(test_loader):
    return test_loader

def train_gcn_pcqm4(epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-4, n_noise=0):

    dirname = os.getcwd()
    models_dir = os.path.join(dirname, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f'Diretório criado: {models_dir}')
    
    if n_noise > 0:
        path = models_dir + '/gcn_pcqm4_noise.pth'
    else:
        path = models_dir + '/gcn_pcqm4.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    gd = graph_datasets()

    # 1. Carregar e transformar o dataset
    if n_noise > 0:
        dataset = gd.prepare_data_graph_noise('PCQM4')
    else:
        dataset = gd.prepare_data_graph('PCQM4')

    dataset = dataset[:20000]
    dataset = [data for data in dataset]  

    # for graph in dataset:
    #     print(graph)

    # working_graphs = [graph.clone() for graph in dataset]
    # print(working_graphs)

    aug = Augmentator(seed=42)

    augmented_graphs = aug.Graphs.augment_data(dataset=dataset, augmentation_methods=["node_drop"], edge_drop_rate=0.15, node_drop_rate=0.1, augment_percentage=0.4)

    # print(augmented_graphs[17580])
    # print(augmented_graphs[-1])

    # 3. Loaders com drop_last para evitar batches incompletos
    # train_loader, val_loader, test_loader = gd.get_paired_dataloaders(dataset_name='PCQM4', batch_size=batch_size)

    augmented_graphs = rebuild_graph_dataset(augmented_graphs)
    augmented_graphs = normalize_features(augmented_graphs)
    augmented_graphs, target_mean, target_std = normalize_targets(augmented_graphs)

    train_loader, val_loader, test_loader = create_dataloaders_from_augmented_graphs(
        augmented_graphs, 
        batch_size=batch_size,
        train_ratio=0.7, 
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    mol = next(iter(train_loader))
    print(mol[0].num_nodes)

    model = GCN(num_features=dataset[0].x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 4. Treinamento
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            # print(data)
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.batch).view(-1)
            target = data.y.view(-1)
            min_size = min(pred.size(0), target.size(0))
            if pred.size(0) != target.size(0):
                pred = pred[:min_size]
                target = target[:min_size]
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                # Garantir que pred e target tenham o mesmo tamanho
                min_size = min(pred.size(0), target.size(0))
                if pred.size(0) != target.size(0):
                    pred = pred[:min_size]
                    target = target[:min_size]
                loss = criterion(pred, target)
                val_loss += loss.item() * data.num_graphs

        avg_val_loss = val_loss / len(val_loader.dataset)

        history.append((epoch + 1, avg_train_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path)

        print(f"[{epoch:02d}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    from sklearn.metrics import r2_score

    # 5. Avaliação final
    model.load_state_dict(torch.load(path))
    model.eval()
    final_val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).view(-1)
            target = data.y.view(-1)
            final_val_loss += criterion(pred, target).item() * data.num_graphs
            
            # Collect predictions and targets for R² calculation
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    final_val_loss /= len(test_loader.dataset)

    # Convert lists to tensors
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Calculate R²
    r2 = r2_score(all_targets, all_preds)

    print(f"\nMSE final na validação: {final_val_loss:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE final na validação: {final_val_loss ** 0.5:.4f}")

    return test_loader, history

if __name__ == '__main__':
    # train_gcn_pcqm4()
    pass
