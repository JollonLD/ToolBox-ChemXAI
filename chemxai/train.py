import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models import GCN, MLP
from data import graph_datasets, qm9_tabular

def train_mlp_qm9(att_index=10, epochs=10, layers=[64, 32], learning_rate=1e-3, 
                batch_size=32, n_noise=3, descriptor_type='Morgan', cache_descriptors=True,
                num_workers=4, morgan_radius=2, morgan_nBits=512):
    """
    Função para treinar um modelo MLP com dados moleculares usando descritores otimizados.

    Args:
        att_index (int): Índice da propriedade a ser predita (default: 10)
        epochs (int): Número de épocas de treinamento (default: 10)
        layers (list): Lista de dimensões das camadas ocultas (default: [64, 32])
        learning_rate (float): Taxa de aprendizado (default: 1e-3)
        batch_size (int): Tamanho do lote (default: 32)
        n_noise (int): Número de features de ruído a adicionar (default: 3)
        descriptor_type (str): Tipo de descritor ('CM', 'Morgan', 'Physicochemical', '3D') (default: 'Morgan')
        cache_descriptors (bool): Se deve usar cache para descritores (default: True)
        num_workers (int): Número de workers para carregamento de dados (default: 4)
        morgan_radius (int): Raio para fingerprints Morgan (default: 2)
        morgan_nBits (int): Número de bits para fingerprints Morgan (default: 512)

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

    # Carregar os dados usando a função otimizada
    qm9 = qm9_tabular()
    if n_noise > 0:
        train_loader, val_loader, test_loader, train_loader_noise, val_loader_noise, test_loader_noise, is_noise = qm9.get_paired_dataloaders_tabular(
            att_index=att_index,           # Índice da propriedade a ser prevista
            batch_size=batch_size,         # Tamanho do lote
            descriptor_type=descriptor_type,          
            list_mols=[],                  # Lista vazia = todas as moléculas
            n_noise=n_noise,
            morgan_radius=morgan_radius, 
            morgan_nBits=morgan_nBits,
            cache_descriptors=cache_descriptors
        )
    else:    
        train_loader, val_loader, test_loader, *_ = qm9.get_paired_dataloaders_tabular(
            att_index=att_index,           # Índice da propriedade a ser prevista
            batch_size=batch_size,         # Tamanho do lote
            descriptor_type=descriptor_type,  
            list_mols=[],                   # Lista vazia = todas as moléculas
            n_noise=n_noise,
            morgan_radius=morgan_radius, 
            morgan_nBits=morgan_nBits,
            cache_descriptors=cache_descriptors
        )

    # Definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    # Obter a dimensão da entrada (tamanho do descritor)
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

def main():
    """
    Treina 6 modelos MLP com diferentes configurações:
    - 3 modelos para att_index=10 (com diferentes arquiteturas)
    - 3 modelos para att_index=0 (com diferentes arquiteturas)
    - Todos com n_noise=0 (sem ruído)
    - 30 épocas para todos
    """
    
    print("="*80)
    print("TREINANDO 6 MODELOS MLP COM DIFERENTES CONFIGURAÇÕES")
    print("="*80)
    
    # Configurações dos modelos
    models_config = [
        # Modelos para att_index=10
        {
            'name': 'MLP_att10_small',
            'att_index': 10,
            'layers': [32, 16],
            'descriptor_type': 'Morgan',
            'cache_descriptors': True
        },
        {
            'name': 'MLP_att10_medium', 
            'att_index': 10,
            'layers': [64, 32, 16],
            'descriptor_type': 'CM',
            'cache_descriptors': True
        },
        {
            'name': 'MLP_att10_large',
            'att_index': 10, 
            'layers': [128, 64, 32],
            'descriptor_type': 'Physicochemical',
            'cache_descriptors': True
        },
        
        # Modelos para att_index=0
        {
            'name': 'MLP_att0_small',
            'att_index': 0,
            'layers': [32, 16], 
            'descriptor_type': 'Morgan',
            'cache_descriptors': True
        },
        {
            'name': 'MLP_att0_medium',
            'att_index': 0,
            'layers': [64, 32, 16],
            'descriptor_type': 'CM',
            'cache_descriptors': True
        },
        {
            'name': 'MLP_att0_large',
            'att_index': 0,
            'layers': [128, 64, 32],
            'descriptor_type': 'Physicochemical',
            'cache_descriptors': True
        }
    ]
    
    # Parâmetros fixos para todos os modelos
    epochs = 30
    n_noise = 0  # Sem ruído
    learning_rate = 1e-3
    batch_size = 32
    num_workers = 4  # Utilizar processamento paralelo
    
    results = {}
    
    print(f"Parâmetros fixos:")
    print(f"  Épocas: {epochs}")
    print(f"  Ruído: {n_noise} (sem ruído)")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Workers: {num_workers}")
    print()
    
    for i, config in enumerate(models_config, 1):
        print(f"[{i}/6] Treinando {config['name']}")
        print(f"  att_index: {config['att_index']}")
        print(f"  layers: {config['layers']}")
        print(f"  descriptor_type: {config['descriptor_type']}")
        print(f"  cache_descriptors: {config['cache_descriptors']}")
        print("-" * 50)
        
        try:
            # Treinar o modelo
            history = train_mlp_qm9(
                att_index=config['att_index'],
                epochs=epochs,
                layers=config['layers'],
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_noise=n_noise,
                descriptor_type=config['descriptor_type'],
                cache_descriptors=config['cache_descriptors'],
                num_workers=num_workers
            )
            
            # Armazenar resultados
            results[config['name']] = {
                'config': config,
                'history': history,
                'final_train_loss': history[-1][1] if history else None,
                'final_val_loss': history[-1][2] if history else None,
                'status': 'success'
            }
            
            print(f"✓ {config['name']} treinado com sucesso!")
            if history:
                print(f"  Loss final treino: {history[-1][1]:.4f}")
                print(f"  Loss final validação: {history[-1][2]:.4f}")
            
        except Exception as e:
            print(f"✗ Erro no treinamento de {config['name']}: {e}")
            results[config['name']] = {
                'config': config,
                'error': str(e),
                'status': 'failed'
            }
        
        print()
    
    # Resumo final
    print("="*80)
    print("RESUMO DOS TREINAMENTOS")
    print("="*80)
    
    successful_models = 0
    failed_models = 0
    
    print("MODELOS PARA att_index=10:")
    for name in ['MLP_att10_small', 'MLP_att10_medium', 'MLP_att10_large']:
        result = results[name]
        if result['status'] == 'success':
            print(f"  ✓ {name}: Val Loss = {result['final_val_loss']:.4f}")
            print(f"    Arquitetura: {result['config']['layers']}")
            print(f"    Descritor: {result['config']['descriptor_type']}")
            successful_models += 1
        else:
            print(f"  ✗ {name}: FALHOU - {result['error']}")
            failed_models += 1
        print()
    
    print("MODELOS PARA att_index=0:")
    for name in ['MLP_att0_small', 'MLP_att0_medium', 'MLP_att0_large']:
        result = results[name]
        if result['status'] == 'success':
            print(f"  ✓ {name}: Val Loss = {result['final_val_loss']:.4f}")
            print(f"    Arquitetura: {result['config']['layers']}")
            print(f"    Descritor: {result['config']['descriptor_type']}")
            successful_models += 1
        else:
            print(f"  ✗ {name}: FALHOU - {result['error']}")
            failed_models += 1
        print()
    
    print("-" * 80)
    print(f"ESTATÍSTICAS FINAIS:")
    print(f"  Total de modelos: 6")
    print(f"  Sucessos: {successful_models}")
    print(f"  Falhas: {failed_models}")
    print(f"  Taxa de sucesso: {successful_models/6*100:.1f}%")
    
    # Encontrar o melhor modelo para cada att_index
    print("\nMELHORES MODELOS:")
    
    # Melhor para att_index=10
    att10_models = {name: result for name, result in results.items() 
                    if name.startswith('MLP_att10_') and result['status'] == 'success'}
    if att10_models:
        best_att10 = min(att10_models.items(), key=lambda x: x[1]['final_val_loss'])
        print(f"  att_index=10: {best_att10[0]} (Val Loss: {best_att10[1]['final_val_loss']:.4f})")
    
    # Melhor para att_index=0  
    att0_models = {name: result for name, result in results.items()
                   if name.startswith('MLP_att0_') and result['status'] == 'success'}
    if att0_models:
        best_att0 = min(att0_models.items(), key=lambda x: x[1]['final_val_loss'])
        print(f"  att_index=0:  {best_att0[0]} (Val Loss: {best_att0[1]['final_val_loss']:.4f})")
    
    # Salvar resultados
    results_dir = os.path.join(os.getcwd(), 'training_results')
    os.makedirs(results_dir, exist_ok=True)
    
    import json
    results_file = os.path.join(results_dir, 'six_models_training_results.json')
    
    # Converter resultados para formato serializável
    serializable_results = {}
    for name, result in results.items():
        serializable_results[name] = result.copy()
        if 'history' in result and result['history']:
            # Converter history para formato serializável
            serializable_results[name]['history'] = [
                (int(epoch), float(train_loss), float(val_loss)) 
                for epoch, train_loss, val_loss in result['history']
            ]
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResultados salvos em: {results_file}")
    print("="*80)
    
    return results

if __name__ == '__main__':
    # Executar o treinamento dos 6 modelos
    results = main()