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
from chemxai.explainers import Shap, LIME   
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

def main():
    """Função principal para explicar todos os modelos."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Lista de descritores a processar
    descriptors = ['Morgan', 'CM', 'Physicochemical']
    
    # Explicar cada modelo
    for descriptor in descriptors:
        try:
            # Modelo sem ruído
            print(f"\n\n--- Processando {descriptor} sem ruído ---")
            model, test_loader, train_loader = load_model(
                descriptor, with_noise=False, device=device)
            explain_with_shap_lime(
                model, test_loader, train_loader, descriptor, 
                with_noise=False, device=device)
        except Exception as e:
            print(f"ERRO: Falha ao processar {descriptor} sem ruído: {str(e)}")
            
        try:
            # Modelo com ruído
            print(f"\n\n--- Processando {descriptor} com ruído ---")
            model, test_loader, train_loader = load_model(
                descriptor, with_noise=True, device=device)
            explain_with_shap_lime(
                model, test_loader, train_loader, descriptor, 
                with_noise=True, device=device)
        except Exception as e:
            print(f"ERRO: Falha ao processar {descriptor} com ruído: {str(e)}")

if __name__ == "__main__":
    main()