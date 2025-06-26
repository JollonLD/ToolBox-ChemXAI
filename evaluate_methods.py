import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from chemxai.data import qm9_tabular, graph_datasets
from chemxai.models import MLP, GCN
from chemxai.evaluate import Evaluator

def evaluate_mlp_with_physicochemical():
    """
    Avalia os métodos de explicabilidade para o modelo MLP treinado com descritores Physicochemical.
    """
    print("\n" + "="*80)
    print("Avaliando Métodos de Explicabilidade para MLP com descritores Physicochemical")
    print("="*80)
    
    # Configuração de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Diretório onde os modelos estão armazenados
    models_dir = os.path.join(os.getcwd(), 'models')
    mlp_path = os.path.join(models_dir, 'mlp_qm9_Physicochemical.pth')
    mlp_noise_path = os.path.join(models_dir, 'mlp_qm9_noise_Physicochemical.pth')
    
    # Verificar se os modelos existem
    if not os.path.exists(mlp_path) or not os.path.exists(mlp_noise_path):
        raise FileNotFoundError(f"Modelo MLP não encontrado em {mlp_path} ou {mlp_noise_path}")
    
    # Inicializar o dataset QM9 com descritores Physicochemical
    qm9 = qm9_tabular()
    
    # Obter os dataloaders pareados (normal e com ruído)
    loaders = qm9.get_paired_dataloaders(
        batch_size=32,
        seed=42,
        descriptor_type='Physicochemical',
        noise_type='gaussian',
        noise_scale=1.0,
        n_noise=2
    )
    
    train_loader_normal, val_loader_normal, test_loader_normal, train_loader_noise, val_loader_noise, test_loader_noise, is_noise = loaders
    
    # Limitar o número de amostras para avaliação mais rápida
    sample_size = 100
    indices = list(range(sample_size))
    
    # Criar subsets limitados
    train_dataset_normal = train_loader_normal.dataset
    test_dataset_normal = test_loader_normal.dataset
    train_dataset_noise = train_loader_noise.dataset
    test_dataset_noise = test_loader_noise.dataset
    
    limited_train_normal = torch.utils.data.Subset(train_dataset_normal, indices)
    limited_test_normal = torch.utils.data.Subset(test_dataset_normal, indices[:len(indices)//2])
    
    limited_train_noise = torch.utils.data.Subset(train_dataset_noise, indices)
    limited_test_noise = torch.utils.data.Subset(test_dataset_noise, indices[:len(indices)//2])
    
    # Criar novos dataloaders com as amostras limitadas - sem shuffle para garantir alinhamento
    train_loader_normal = DataLoader(limited_train_normal, batch_size=32, shuffle=False)
    test_loader_normal = DataLoader(limited_test_normal, batch_size=32, shuffle=False)
    
    train_loader_noise = DataLoader(limited_train_noise, batch_size=32, shuffle=False)
    test_loader_noise = DataLoader(limited_test_noise, batch_size=32, shuffle=False)
    
    # Carregar o modelo MLP treinado com descritores Physicochemical
    input_dim = next(iter(train_loader_normal))[0].shape[1]
    model_normal = MLP(input_dim=input_dim, output_dim=1, layers=[128,64], device=device).to(device)
    model_normal.load_state_dict(torch.load(mlp_path, map_location=torch.device(device)))
    model_normal.eval()
    print(f"Modelo sem ruído:\n{model_normal}")
    
    # Carregar o modelo MLP com ruído
    input_dim_noise = next(iter(train_loader_noise))[0].shape[1]
    model_noise = MLP(input_dim=input_dim_noise, output_dim=1, layers=[128,64], device=device).to(device)
    model_noise.load_state_dict(torch.load(mlp_noise_path, map_location=torch.device(device)))
    model_noise.eval()
    print(f"Modelo com ruído:\n{model_noise}")
    
    # Lista de explainers para avaliar
    explainers = ['shap_local', 'shap_global', 'lime']
    
    # Avaliar cada explainer
    for explainer_type in explainers:
        print(f"\n--- Avaliando explainer: {explainer_type} ---")
        
        evaluator = Evaluator(
            model_normal=model_normal,
            model_noise=model_noise,
            train_loader_normal=train_loader_normal,
            test_loader_normal=test_loader_normal,
            train_loader_noise=train_loader_noise,
            test_loader_noise=test_loader_noise,
            device=device,
            model_type='tabular',
            explainer_type=explainer_type
        )
        
        # Executar avaliação de robustez
        similarities, l1_differences, l2_differences, spearman_correlations, figs = evaluator.robustness()
        
        # Resumir os resultados
        print(f"Resultados para {explainer_type}:")
        print(f"  - Similaridade média: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
        print(f"  - Diferença L1 média: {np.mean(l1_differences):.4f} ± {np.std(l1_differences):.4f}")
        print(f"  - Diferença L2 média: {np.mean(l2_differences):.4f} ± {np.std(l2_differences):.4f}")
        print(f"  - Correlação Spearman média: {np.mean(spearman_correlations):.4f} ± {np.std(spearman_correlations):.4f}")

def evaluate_gcn_with_qm9():
    """
    Avalia os métodos de explicabilidade para o modelo GCN treinado com QM9.
    """
    print("\n" + "="*80)
    print("Avaliando Métodos de Explicabilidade para GCN com QM9")
    print("="*80)
    
    # Configuração de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Diretório onde os modelos estão armazenados
    models_dir = os.path.join(os.getcwd(), 'models')
    gcn_path = os.path.join(models_dir, 'gcn_qm9.pth')
    gcn_noise_path = os.path.join(models_dir, 'gcn_qm9_noise.pth')
    
    # Verificar se os modelos existem
    if not os.path.exists(gcn_path) or not os.path.exists(gcn_noise_path):
        raise FileNotFoundError(f"Modelo GCN não encontrado em {gcn_path} ou {gcn_noise_path}")
    
    # Inicializar o dataset QM9 como grafo
    gd = graph_datasets()
    
    # Obter os dataloaders pareados (normal e com ruído)
    loaders = gd.get_paired_dataloaders(
        dataset_name='QM9', 
        batch_size=32,
        seed=42,
        noise_type='gaussian',
        noise_scale=1.0
    )
    
    train_loader_normal, val_loader_normal, test_loader_normal, train_loader_noise, val_loader_noise, test_loader_noise = loaders
    
    # Limitar o número de amostras para avaliação mais rápida
    from torch_geometric.loader import DataLoader as GraphDataLoader
    
    sample_size = 100
    indices = list(range(sample_size))
    
    # Criar subsets limitados
    train_dataset_normal = train_loader_normal.dataset
    test_dataset_normal = test_loader_normal.dataset
    train_dataset_noise = train_loader_noise.dataset
    test_dataset_noise = test_loader_noise.dataset
    
    limited_train_normal = torch.utils.data.Subset(train_dataset_normal, indices)
    limited_test_normal = torch.utils.data.Subset(test_dataset_normal, indices[:len(indices)//2])
    
    limited_train_noise = torch.utils.data.Subset(train_dataset_noise, indices)
    limited_test_noise = torch.utils.data.Subset(test_dataset_noise, indices[:len(indices)//2])
    
    # Criar novos dataloaders com as amostras limitadas - sem shuffle para garantir alinhamento
    train_loader_normal = GraphDataLoader(limited_train_normal, batch_size=32, shuffle=False)
    test_loader_normal = GraphDataLoader(limited_test_normal, batch_size=32, shuffle=False)
    
    train_loader_noise = GraphDataLoader(limited_train_noise, batch_size=32, shuffle=False)
    test_loader_noise = GraphDataLoader(limited_test_noise, batch_size=32, shuffle=False)
    
    # Obter a primeira amostra para determinar o número de features
    data_normal = next(iter(train_loader_normal))[0]
    data_noise = next(iter(train_loader_noise))[0]
    
    # Carregar o modelo GCN treinado
    model_normal = GCN(num_features=data_normal.x.size(1)).to(device)
    model_normal.load_state_dict(torch.load(gcn_path, map_location=torch.device(device)))
    model_normal.eval()
    print(f"Modelo sem ruído:\n{model_normal}")
    
    # Carregar o modelo GCN com ruído
    model_noise = GCN(num_features=data_noise.x.size(1)).to(device)
    model_noise.load_state_dict(torch.load(gcn_noise_path, map_location=torch.device(device)))
    model_noise.eval()
    print(f"Modelo com ruído:\n{model_noise}")
    
    # Lista de explainers para avaliar
    explainers = ['gnn_explainer', 'graph_shap']
    
    # Avaliar cada explainer
    for explainer_type in explainers:
        print(f"\n--- Avaliando explainer: {explainer_type} ---")
        
        evaluator = Evaluator(
            model_normal=model_normal,
            model_noise=model_noise,
            train_loader_normal=train_loader_normal,
            test_loader_normal=test_loader_normal,
            train_loader_noise=train_loader_noise,
            test_loader_noise=test_loader_noise,
            device=device,
            model_type='graph',
            explainer_type=explainer_type
        )
        
        # Executar avaliação de robustez
        similarities, l1_differences, l2_differences, spearman_correlations, figs = evaluator.robustness()
        
        # Resumir os resultados
        print(f"Resultados para {explainer_type}:")
        print(f"  - Similaridade média: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
        print(f"  - Diferença L1 média: {np.mean(l1_differences):.4f} ± {np.std(l1_differences):.4f}")
        print(f"  - Diferença L2 média: {np.mean(l2_differences):.4f} ± {np.std(l2_differences):.4f}")
        print(f"  - Correlação Spearman média: {np.mean(spearman_correlations):.4f} ± {np.std(spearman_correlations):.4f}")

def main():
    """
    Função principal para executar todas as avaliações.
    """
    print("Iniciando avaliação de métodos de explicabilidade...")
    
    # Avaliar MLP com Physicochemical
    evaluate_mlp_with_physicochemical()
    
    # Avaliar GCN com QM9
    evaluate_gcn_with_qm9()
    
    print("\nAvaliação de métodos concluída!")

if __name__ == "__main__":
    main()