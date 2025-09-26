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

# ...existing code...

def train_all_mlp_descriptors(att_index=10, epochs=10, layers=[64, 32], learning_rate=1e-3, batch_size=32):
    """
    Treina modelos MLP para todos os tipos de descritores disponíveis, com e sem ruído.
    
    Args:
        att_index (int): Índice da propriedade a ser predita
        epochs (int): Número de épocas de treinamento
        layers (list): Arquitetura das camadas ocultas
        learning_rate (float): Taxa de aprendizado
        batch_size (int): Tamanho do batch
    
    Returns:
        dict: Dicionário com os históricos de treinamento
    """
    descriptor_types = ['CM', 'Morgan', 'Physicochemical', '3D']
    noise_configs = [0, 3]  # sem ruído e com 3 features de ruído
    
    results = {}
    
    print("="*60)
    print("TREINANDO MODELOS MLP PARA TODOS OS DESCRITORES")
    print("="*60)
    
    for descriptor_type in descriptor_types:
        print(f"\n--- Treinando com descritor: {descriptor_type} ---")
        
        for n_noise in noise_configs:
            noise_label = "com_ruido" if n_noise > 0 else "sem_ruido"
            model_key = f"MLP_{descriptor_type}_{noise_label}"
            
            print(f"\nConfiguracao: {model_key}")
            print(f"Epocas: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
            print(f"Ruido: {n_noise} features")
            
            try:
                history = train_mlp_qm9(
                    att_index=att_index,
                    epochs=epochs,
                    layers=layers,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    n_noise=n_noise,
                    descriptor_type=descriptor_type
                )
                
                results[model_key] = {
                    'history': history,
                    'descriptor_type': descriptor_type,
                    'n_noise': n_noise,
                    'final_train_loss': history[-1][1] if history else None,
                    'final_val_loss': history[-1][2] if history else None
                }
                
                print(f"✓ Treinamento concluído para {model_key}")
                if history:
                    print(f"  Loss final treino: {history[-1][1]:.4f}")
                    print(f"  Loss final validação: {history[-1][2]:.4f}")
                
            except Exception as e:
                print(f"✗ Erro no treinamento de {model_key}: {e}")
                results[model_key] = {'error': str(e)}
    
    print("\n" + "="*60)
    print("RESUMO DOS TREINAMENTOS MLP")
    print("="*60)
    
    for model_key, result in results.items():
        if 'error' in result:
            print(f"{model_key}: ERRO - {result['error']}")
        else:
            print(f"{model_key}: OK - Val Loss: {result['final_val_loss']:.4f}")
    
    return results

def train_all_gcn_datasets(target_idx=3, epochs=10, batch_size=64, lr=0.001, weight_decay=1e-4):
    """
    Treina modelos GCN para todos os datasets disponíveis (QM9 e PCQM4), com e sem ruído.
    
    Args:
        target_idx (int): Índice da propriedade a ser predita
        epochs (int): Número de épocas de treinamento
        batch_size (int): Tamanho do batch
        lr (float): Taxa de aprendizado
        weight_decay (float): Regularização L2
    
    Returns:
        dict: Dicionário com os históricos de treinamento
    """
    datasets = ['QM9', 'PCQM4']
    noise_configs = [0, 1]  # sem ruído e com ruído
    
    results = {}
    
    print("="*60)
    print("TREINANDO MODELOS GCN PARA TODOS OS DATASETS")
    print("="*60)
    
    for dataset_name in datasets:
        print(f"\n--- Treinando com dataset: {dataset_name} ---")
        
        for n_noise in noise_configs:
            noise_label = "com_ruido" if n_noise > 0 else "sem_ruido"
            model_key = f"GCN_{dataset_name}_{noise_label}"
            
            print(f"\nConfiguracao: {model_key}")
            print(f"Epocas: {epochs}, Batch: {batch_size}, LR: {lr}")
            print(f"Ruido: {'Sim' if n_noise > 0 else 'Não'}")
            
            try:
                if dataset_name == 'QM9':
                    history = train_gcn_qm9(
                        target_idx=target_idx,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        weight_decay=weight_decay,
                        n_noise=n_noise
                    )
                elif dataset_name == 'PCQM4':
                    test_loader, history = train_gcn_pcqm4(
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        weight_decay=weight_decay,
                        n_noise=n_noise
                    )
                
                results[model_key] = {
                    'history': history,
                    'dataset': dataset_name,
                    'n_noise': n_noise,
                    'final_train_loss': history[-1][1] if history else None,
                    'final_val_loss': history[-1][2] if history else None
                }
                
                print(f"✓ Treinamento concluído para {model_key}")
                if history:
                    print(f"  Loss final treino: {history[-1][1]:.4f}")
                    print(f"  Loss final validação: {history[-1][2]:.4f}")
                
            except Exception as e:
                print(f"✗ Erro no treinamento de {model_key}: {e}")
                results[model_key] = {'error': str(e)}
    
    print("\n" + "="*60)
    print("RESUMO DOS TREINAMENTOS GCN")
    print("="*60)
    
    for model_key, result in results.items():
        if 'error' in result:
            print(f"{model_key}: ERRO - {result['error']}")
        else:
            print(f"{model_key}: OK - Val Loss: {result['final_val_loss']:.4f}")
    
    return results

def train_all_models_comprehensive(
    att_index=10, 
    target_idx=3,
    epochs_mlp=10, 
    epochs_gcn=10,
    mlp_layers=[64, 32],
    learning_rate_mlp=1e-3,
    learning_rate_gcn=1e-3,
    batch_size=32,
    weight_decay=1e-4
):
    """
    Função principal que treina TODOS os modelos disponíveis:
    - MLP para todos os descritores tabulares (CM, Morgan, Physicochemical, 3D)
    - GCN para todos os datasets gráficos (QM9, PCQM4)
    Tanto com quanto sem ruído para todos os casos.
    
    Args:
        att_index (int): Índice da propriedade para modelos MLP
        target_idx (int): Índice da propriedade para modelos GCN
        epochs_mlp (int): Épocas para modelos MLP
        epochs_gcn (int): Épocas para modelos GCN
        mlp_layers (list): Arquitetura MLP
        learning_rate_mlp (float): LR para MLP
        learning_rate_gcn (float): LR para GCN
        batch_size (int): Tamanho do batch
        weight_decay (float): Regularização L2 para GCN
    
    Returns:
        dict: Resultados completos de todos os treinamentos
    """
    print("="*80)
    print("TREINAMENTO ABRANGENTE DE TODOS OS MODELOS")
    print("="*80)
    print(f"Configurações:")
    print(f"  MLP: {epochs_mlp} épocas, LR={learning_rate_mlp}, layers={mlp_layers}")
    print(f"  GCN: {epochs_gcn} épocas, LR={learning_rate_gcn}, weight_decay={weight_decay}")
    print(f"  Batch size: {batch_size}")
    print(f"  Propriedade MLP (att_index): {att_index}")
    print(f"  Propriedade GCN (target_idx): {target_idx}")
    
    all_results = {}
    
    # 1. Treinar todos os modelos MLP
    print(f"\n{'='*50}")
    print("FASE 1: MODELOS MLP (DESCRITORES TABULARES)")
    print(f"{'='*50}")
    
    mlp_results = train_all_mlp_descriptors(
        att_index=att_index,
        epochs=epochs_mlp,
        layers=mlp_layers,
        learning_rate=learning_rate_mlp,
        batch_size=batch_size
    )
    
    all_results['MLP'] = mlp_results
    
    # 2. Treinar todos os modelos GCN
    print(f"\n{'='*50}")
    print("FASE 2: MODELOS GCN (DADOS GRÁFICOS)")
    print(f"{'='*50}")
    
    gcn_results = train_all_gcn_datasets(
        target_idx=target_idx,
        epochs=epochs_gcn,
        batch_size=batch_size,
        lr=learning_rate_gcn,
        weight_decay=weight_decay
    )
    
    all_results['GCN'] = gcn_results
    
    # 3. Resumo final
    print(f"\n{'='*80}")
    print("RESUMO FINAL DE TODOS OS TREINAMENTOS")
    print(f"{'='*80}")
    
    total_models = 0
    successful_models = 0
    failed_models = []
    
    for model_type, results in all_results.items():
        print(f"\n--- {model_type} ---")
        for model_key, result in results.items():
            total_models += 1
            if 'error' in result:
                print(f"  ✗ {model_key}: {result['error']}")
                failed_models.append(model_key)
            else:
                print(f"  ✓ {model_key}: Val Loss = {result['final_val_loss']:.4f}")
                successful_models += 1
    
    print(f"\n{'='*80}")
    print(f"ESTATÍSTICAS FINAIS:")
    print(f"  Total de modelos: {total_models}")
    print(f"  Sucessos: {successful_models}")
    print(f"  Falhas: {len(failed_models)}")
    print(f"  Taxa de sucesso: {successful_models/total_models*100:.1f}%")
    
    if failed_models:
        print(f"\nModelos que falharam:")
        for model in failed_models:
            print(f"  - {model}")
    
    # Salvar resultados
    results_dir = os.path.join(os.getcwd(), 'training_results')
    os.makedirs(results_dir, exist_ok=True)
    
    import json
    results_file = os.path.join(results_dir, 'comprehensive_training_results.json')
    
    # Converter resultados para formato serializável
    serializable_results = {}
    for model_type, results in all_results.items():
        serializable_results[model_type] = {}
        for model_key, result in results.items():
            if 'history' in result:
                # Converter history para formato serializável
                result['history'] = [(int(epoch), float(train_loss), float(val_loss)) 
                                   for epoch, train_loss, val_loss in result['history']]
            serializable_results[model_type][model_key] = result
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResultados salvos em: {results_file}")
    
    return all_results

def quick_training_test():
    """
    Função de teste rápido com poucas épocas para verificar se tudo funciona.
    """
    print("EXECUTANDO TESTE RÁPIDO DE TREINAMENTO")
    print("(Poucas épocas apenas para verificar funcionamento)")
    
    return train_all_models_comprehensive(
        att_index=10,
        target_idx=3,
        epochs_mlp=2,  # Poucas épocas para teste
        epochs_gcn=2,
        mlp_layers=[32, 16],  # Arquitetura menor para teste
        learning_rate_mlp=1e-3,
        learning_rate_gcn=1e-3,
        batch_size=64,
        weight_decay=1e-4
    )

if __name__ == '__main__':
    # Para executar o treinamento completo, descomente a linha abaixo:
    # results = train_all_models_comprehensive()
    
    # Para executar apenas um teste rápido:
    results = quick_training_test()