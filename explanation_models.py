#!/usr/bin/env python3
"""
Script para gerar explicações SHAP e LIME para diferentes modelos de descritores moleculares.
Carrega modelos treinados com diferentes descritores (Morgan, CM, Physicochemical) 
e gera explicações para instâncias de teste.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from chemxai.data import qm9_tabular
from chemxai.models import MLP
from chemxai.explainers import Shap, LIME, GraphShap
from chemxai.evaluate import FingerprintAnalyzer
import time

def load_model(descriptor_type, with_noise=False, device=None):
    """
    Carrega um modelo MLP treinado para um determinado tipo de descritor.
    
    Args:
        descriptor_type (str): Tipo de descritor ('Morgan', 'CM', 'Physicochemical')
        with_noise (bool): Se deve carregar o modelo com ruído
        device (torch.device): Dispositivo para executar o modelo
        
    Returns:
        tuple: (modelo carregado, dataloader de teste, dataloader de treino)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Definir nome do arquivo modelo
    if with_noise:
        model_path = f'models/mlp_qm9_noise_{descriptor_type}.pth'
    else:
        model_path = f'models/mlp_qm9_{descriptor_type}.pth'
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
    # Carregar os dados
    print(f"Carregando dados com descritor {descriptor_type}...")
    qm9 = qm9_tabular()
    
    # Configuração específica para Morgan fingerprints
    kwargs = {}
    if descriptor_type == 'Morgan':
        kwargs = {'morgan_radius': 2, 'morgan_nBits': 512}
    
    # Obter os dataloaders
    if with_noise:
        train_loader, val_loader, test_loader, train_loader_noise, val_loader_noise, test_loader_noise, is_noise = qm9.get_paired_dataloaders(
            att_index=10,
            batch_size=32,
            descriptor_type=descriptor_type,
            n_noise=3,
            **kwargs
        )
        active_loader_test = test_loader_noise
        active_loader_train = train_loader_noise
    else:
        train_loader, val_loader, test_loader = qm9.get_paired_dataloaders(
            att_index=10,
            batch_size=32,
            descriptor_type=descriptor_type,
            n_noise=0,
            **kwargs
        )
        active_loader_test = test_loader
        active_loader_train = train_loader
    
    # Obter dimensão da entrada
    input_dim = next(iter(active_loader_test))[0].shape[1]
    output_dim = 1  # Previsão de uma propriedade
    
    # Definir a arquitetura da MLP
    layers = [128, 64]  # Mesmo que no treinamento
    model = MLP(input_dim, output_dim, layers, device, lr=1e-3)
    model.to(device)
    
    # Carregar os pesos treinados
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Modelo carregado de {model_path}")
    return model, active_loader_test, active_loader_train

def load_gcn_model(with_noise=False, device=None):
    """
    Carrega um modelo GCN treinado para o dataset QM9.
    
    Args:
        with_noise (bool): Se deve carregar o modelo com ruído
        device (torch.device): Dispositivo para executar o modelo
        
    Returns:
        tuple: (modelo carregado, dataloader de teste, dataloader de treino)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Definir nome do arquivo modelo
    if with_noise:
        model_path = f'models/gcn_qm9_noise.pth'
    else:
        model_path = f'models/gcn_qm9.pth'
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo GCN não encontrado: {model_path}. Execute 'python train_models.py' para treinar o modelo GCN.")
        
    # Carregar dataset QM9 com grafos
    print("Carregando dataset QM9 para modelo de grafo...")
    from chemxai.data import graph_datasets
    graph_data = graph_datasets()
    dataset = graph_data.prepare_data_graph(dataset_name='QM9')
    
    # Preparar dataloaders com batch_size pequeno para evitar problemas de memória
    # durante as explicações e avaliação do modelo
    from torch_geometric.loader import DataLoader
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Usar random_split do torch padrão para criar os subsets
    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Criar dataloaders com batch_size menor para análise
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Inicializar o modelo
    from chemxai.models import GCN
    input_dim = dataset[0].x.shape[1]
    hidden_dim = 256  # Dimensão oculta usada no treinamento
    
    gcn_model = GCN(num_features=input_dim, hidden_dim=hidden_dim)
    gcn_model.to(device)
    
    # Carregar os pesos treinados
    gcn_model.load_state_dict(torch.load(model_path, map_location=device))
    gcn_model.eval()
    
    print(f"Modelo GCN carregado de {model_path}")
    return gcn_model, test_loader, train_loader

def explain_with_shap_lime(model, test_loader, train_loader, descriptor_type, with_noise=False, device=None):
    """
    Gera explicações SHAP e LIME para um modelo específico e exibe os resultados.
    
    Args:
        model: Modelo treinado
        test_loader: DataLoader para o conjunto de teste
        train_loader: DataLoader para o conjunto de treinamento
        descriptor_type: Tipo de descritor usado
        with_noise: Se o modelo inclui colunas de ruído
        device: Dispositivo para computação
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    noise_str = "com ruído" if with_noise else "sem ruído"
    print(f"\n{'='*80}")
    print(f"Gerando explicações para modelo {descriptor_type} {noise_str}")
    print(f"{'='*80}")
    
    # Obter dados dos batches
    batch_idx = 0
    mol_idx = 0
    batch_data = next(iter(test_loader))[batch_idx]
    background = next(iter(train_loader))[batch_idx]
    
    # 1. Explicação SHAP
    print("\nExecutando explicação SHAP...")
    start_time = time.time()
    explainer_shap = Shap(model=model, background_tensor=background, 
                          test_tensor=batch_data, device=device)
    shap_values = explainer_shap.explain_local(index=mol_idx)
    shap_time = time.time() - start_time
    print(f"Tempo de execução SHAP: {shap_time:.2f} segundos")
    
    # 2. Explicação LIME
    print("\nExecutando explicação LIME...")
    start_time = time.time()
    explainer_lime = LIME(model=model, background_tensor=background, 
                          test_tensor=batch_data, device=device)
    lime_values = explainer_lime.explain_local(index=mol_idx)
    lime_time = time.time() - start_time
    print(f"Tempo de execução LIME: {lime_time:.2f} segundos")
    
    # Visualizar as explicações
    plt.figure(figsize=(14, 6))
    
    # Mostrar apenas as top 20 features mais importantes
    n_features = min(20, len(shap_values))
    
    # Organizar por importância absoluta
    shap_indices = np.argsort(np.abs(shap_values))[-n_features:]
    lime_indices = np.argsort(np.abs(lime_values))[-n_features:]
    
    # SHAP plot
    plt.subplot(1, 2, 1)
    plt.barh(range(n_features), [shap_values[i] for i in shap_indices], color='b')
    plt.yticks(range(n_features), [f"Feature {i}" for i in shap_indices])
    plt.title(f"SHAP - {descriptor_type} {noise_str}")
    plt.xlabel("Valor SHAP")
    
    # LIME plot
    plt.subplot(1, 2, 2)
    plt.barh(range(n_features), [lime_values[i] for i in lime_indices], color='r')
    plt.yticks(range(n_features), [f"Feature {i}" for i in lime_indices])
    plt.title(f"LIME - {descriptor_type} {noise_str}")
    plt.xlabel("Valor LIME")
    
    plt.tight_layout()
    plt.savefig(f"explicacao_{descriptor_type}_{'noise' if with_noise else 'no_noise'}.png")
    plt.show()
    
    # Para Morgan fingerprints, usar o FingerprintAnalyzer
    if descriptor_type == 'Morgan':
        print("\n=== Análise de Fingerprint com SHAP ===")
        analyzer_shap = FingerprintAnalyzer(
            explanation=shap_values,
            batch_idx=batch_idx,
            mol_idx=mol_idx,
            dataset_type='test',
            device=device
        )
        analyzer_shap.analyze()
        
        print("\n=== Análise de Fingerprint com LIME ===")
        analyzer_lime = FingerprintAnalyzer(
            explanation=lime_values,
            batch_idx=batch_idx,
            mol_idx=mol_idx,
            dataset_type='test',
            device=device
        )
        analyzer_lime.analyze()

def exp_SHAP_QM9():
    """Função para exemplos de explicação com MLP (SHAP) e GCN (GraphShap)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Exemplo 1: MLP com SHAP (QM9 tabular com descritores físico-químicos)
    print("\n\n=== Exemplo 1: MLP com SHAP (QM9 tabular) ===")
    try:
        # Carregar modelo MLP treinado com descritor físico-químico
        model, test_loader, train_loader = load_model(
            descriptor_type='Physicochemical', with_noise=False, device=device)
        
        # Calcular erro no conjunto de teste
        print("\nCalculando métricas no conjunto de teste para MLP...")
        model.eval()
        test_loss = 0.0
        test_mse = 0.0
        num_samples = 0

        with torch.no_grad():
            for data, target in test_loader:  # Modificado: desempacotar tupla corretamente
                data, target = data.to(device), target.to(device)  # Modificado: mover tensores para o device
                # Processar o batch corretamente com o modelo MLP
                out = model(data)  # Modificado: usar o modelo MLP, não o GCN
                
                # Calcular erros
                test_loss += torch.nn.L1Loss(reduction='sum')(out, target).item()
                test_mse += torch.nn.MSELoss(reduction='sum')(out, target).item()
                num_samples += target.size(0)

        # Calcular média corretamente
        test_loss /= num_samples
        test_mse /= num_samples
        test_rmse = np.sqrt(test_mse)
        
        print(f"Erro MLP no teste (MAE): {test_loss:.4f}")
        print(f"Erro MLP no teste (RMSE): {test_rmse:.4f}")
        
        # Obter dados dos batches
        batch_idx = 0
        mol_idx = 0
        batch_data = next(iter(test_loader))[0]  # Primeiro elemento é o tensor de dados
        background = next(iter(train_loader))[0]  # Primeiro elemento é o tensor de dados
        
        # Nomes dos descritores físico-químicos
        descriptor_names = [
        "Molecular Weight", "LogP", "TPSA", "H-bond Donors", "H-bond Acceptors",
        "Rotatable Bonds", "Aromatic Rings", "Balaban J Index", "Druglikeness (QED)"
        ]
        
        # Explicação SHAP
        print("\nExecutando explicação SHAP para MLP...")
        explainer_shap = Shap(model=model, background_tensor=background, 
                            test_tensor=batch_data, device=device)
        shap_values = explainer_shap.explain_global()
        
        # Mostrar as top features mais importantes
        n_features = min(len(descriptor_names), len(shap_values))
        top_indices = np.argsort(np.abs(shap_values))[-n_features:][::-1]
        top_values = [shap_values[i] for i in top_indices]
        top_names = [descriptor_names[i] for i in top_indices]
        
        # Gráfico de barras horizontais para SHAP
        plt.figure(figsize=(12, 8))
        colors = ['blue' if val > 0 else 'red' for val in top_values]
        plt.barh(range(n_features), top_values, color=colors)
        plt.yticks(range(n_features), top_names)
        plt.title(f"SHAP - Importância dos Descritores Físico-Químicos (RMSE: {test_rmse:.4f})", fontsize=14)
        plt.xlabel("Valor SHAP", fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("shap_mlp_phychem.png", dpi=300)
        
        # Verificar o ambiente antes de tentar mostrar
        import matplotlib
        if matplotlib.get_backend() != 'agg':
            plt.show()
        
        print(f"Gráfico salvo em shap_mlp_phychem.png")
        
    except Exception as e:
        print(f"ERRO no Exemplo 1: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Exemplo 2: GCN com GraphShap (QM9 grafo)
    print("\n\n=== Exemplo 2: GCN com GraphShap (QM9 grafo) ===")
    try:
        from chemxai.data import graph_datasets
        from chemxai.models import GCN
        import torch_geometric
        
        # Carregar dataset QM9 com grafos
        print("Carregando dataset QM9 para modelo de grafo...")
        graph_data = graph_datasets()
        dataset = graph_data.prepare_data_graph(dataset_name='QM9')
        
        # Verificar modelo GCN
        model_path = 'models/gcn_qm9.pth'
        if not os.path.exists(model_path):
            print(f"Aviso: Modelo GCN não encontrado em {model_path}. Certifique-se de treinar o modelo primeiro.")
            print("Execute 'python train_gcn.py' para treinar o modelo GCN.")
            return
        
        gd = graph_datasets()
        # Inicializar o modelo GCN
        loaders = gd.get_paired_dataloaders(
            dataset_name='QM9', 
            batch_size=32,
            seed=42,
            noise_type='gaussian',
            noise_scale=1.0
        )

        _, _, test_loader, _, _, _ = loaders

        data_normal = next(iter(test_loader))

        gcn_model = GCN(num_features=data_normal.x.size(1)).to(device)
        gcn_model.load_state_dict(torch.load('models/gcn_qm9.pth', map_location=torch.device(device)))
        gcn_model = gcn_model.to(device)
        
        # Calcular erro no conjunto de teste para GCN
        print("\nCalculando métricas no conjunto de teste para GCN...")
        gcn_model.eval()
        test_loss = 0.0
        test_mse = 0.0
        num_samples = 0
        property_idx = 3
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Calcular output para o batch
                out = gcn_model(batch.x, batch.edge_index, batch=batch.batch)
                target = batch.y[:, property_idx:property_idx+1]
                # Calcular erros com dimensões ajustadas
                test_loss += torch.nn.L1Loss(reduction='sum')(out, target).item()
                test_mse += torch.nn.MSELoss(reduction='sum')(out, target).item()
                num_samples += target.shape[0]

        # Calcular média corretamente
        test_loss /= num_samples
        test_mse /= num_samples
        test_rmse = np.sqrt(test_mse)
        
        print(f"Erro GCN no teste (MAE): {test_loss:.4f}")
        print(f"Erro GCN no teste (RMSE): {test_rmse:.4f}")
        
        # Selecionar uma amostra do dataset para explicação
        sample_data = dataset[0].to(device)  # Use o primeiro grafo do dataset

        # Obter dimensão de entrada do nó
        input_dim = sample_data.x.shape[1]
        
        # Nomes das features para grafos QM9
        node_feature_names = [
        "C (Carbon)", "N (Nitrogen)", "O (Oxygen)", "F (Fluorine)", "H (Hydrogen)",
        "Formal Charge", "Aromaticity", "sp Hybridization", "sp2 Hybridization", "sp3 Hybridization",
        "Number of Hydrogens"
        ]
        
        # Completar nomes se necessário
        while len(node_feature_names) < input_dim:
            node_feature_names.append(f"Característica {len(node_feature_names)}")
        
        # Explicar com GraphShap
        print("\nExecutando explicação GraphShap para GCN...")
        explainer = GraphShap(data=sample_data, model=gcn_model, device=device)
        feature_importance = explainer.explain(num_samples=50)
        
        # Mostrar as top features mais importantes
        n_features = min(11, len(feature_importance))  # QM9 normalmente tem 11 características por nó
        top_indices = np.argsort(np.abs(feature_importance))[-n_features:][::-1]
        top_values = [feature_importance[i] for i in top_indices]
        top_names = [node_feature_names[i] for i in top_indices]
        
        # Gráfico de barras horizontais para GraphShap
        plt.figure(figsize=(12, 8))
        colors = ['blue' if val > 0 else 'red' for val in top_values]
        plt.barh(range(n_features), top_values, color=colors)
        plt.yticks(range(n_features), top_names)
        plt.title(f"GraphShap - Importância das Características dos Átomos (RMSE: {test_rmse:.4f})", fontsize=14)
        plt.xlabel("Valor GraphShap", fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("graphshap_gcn.png", dpi=300)
        
        # Verificar o ambiente antes de tentar mostrar
        if matplotlib.get_backend() != 'agg':
            plt.show()
        
        print(f"Gráfico salvo em graphshap_gcn.png")
        
    except Exception as e:
        print(f"ERRO no Exemplo 2: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    # """Função principal para explicar todos os modelos."""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Usando dispositivo: {device}")
    
    # # Lista de descritores a processar
    # descriptors = ['Morgan', 'CM', 'Physicochemical']
    
    # # Explicar cada modelo
    # for descriptor in descriptors:
    #     try:
    #         # Modelo sem ruído
    #         print(f"\n\n--- Processando {descriptor} sem ruído ---")
    #         model, test_loader, train_loader = load_model(
    #             descriptor, with_noise=False, device=device)
    #         explain_with_shap_lime(
    #             model, test_loader, train_loader, descriptor, 
    #             with_noise=False, device=device)
    #     except Exception as e:
    #         print(f"ERRO: Falha ao processar {descriptor} sem ruído: {str(e)}")
            
    #     try:
    #         # Modelo com ruído
    #         print(f"\n\n--- Processando {descriptor} com ruído ---")
    #         model, test_loader, train_loader = load_model(
    #             descriptor, with_noise=True, device=device)
    #         explain_with_shap_lime(
    #             model, test_loader, train_loader, descriptor, 
    #             with_noise=True, device=device)
    #     except Exception as e:
    #         print(f"ERRO: Falha ao processar {descriptor} com ruído: {str(e)}")
    exp_SHAP_QM9()


if __name__ == "__main__":
    main()