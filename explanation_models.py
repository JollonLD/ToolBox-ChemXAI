import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

from chemxai.data import qm9_tabular, graph_datasets
from chemxai.models import MLP, GCN
from chemxai.explainers import Shap, GraphShap


def explain_models():
    """Gera explicações para os modelos MLP e GCN treinados."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Diretório onde os modelos estão armazenados
    models_dir = os.path.join(os.getcwd(), 'models')
    mlp_path = os.path.join(models_dir, 'mlp_qm9_Physicochemical.pth')
    gcn_path = os.path.join(models_dir, 'gcn_qm9.pth')
    
    # Verificar se os modelos existem
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"Modelo MLP não encontrado em {mlp_path}")
    if not os.path.exists(gcn_path):
        raise FileNotFoundError(f"Modelo GCN não encontrado em {gcn_path}")
    
    print("\n" + "="*80)
    print("Explicando modelo MLP com SHAP (Physicochemical descriptors)")
    print("="*80)
    
    # 1. Explicação para MLP com descritores Physicochemical
    explain_mlp_with_shap(mlp_path, device)
    
    print("\n" + "="*80)
    print("Explicando modelo GCN com GraphShap")
    print("="*80)
    
    # 2. Explicação para GCN usando GraphShap
    explain_gcn_with_graphshap(gcn_path, device)


def explain_mlp_with_shap(model_path, device):
    """Gera explicação SHAP para o modelo MLP com descritores físico-químicos."""
    
    # Carregar dados
    qm9 = qm9_tabular()
    property_idx = 0  # Propriedade que foi usada no treinamento
    
    # Obter dataloaders
    train_loader, val_loader, test_loader = qm9.get_paired_dataloaders(
        att_index=property_idx, 
        batch_size=32, 
        descriptor_type='Physicochemical',
        n_noise=0
    )
    
    # Obter dimensão de entrada a partir do primeiro batch
    batch = next(iter(train_loader))
    input_dim = batch[0].shape[1]
    
    # Inicializar e carregar o modelo
    model = MLP(input_dim=input_dim, output_dim=1, layers=[128, 64], device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Avaliar o modelo para garantir que funciona corretamente
    test_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_mse += ((outputs - targets) ** 2).sum().item()
            num_samples += targets.size(0)
    
    test_mse /= num_samples
    print(f"MSE no conjunto de teste: {test_mse:.6f}")
    print(f"RMSE no conjunto de teste: {np.sqrt(test_mse):.6f}")
    
    # Obter batches para explicação
    background_data = next(iter(train_loader))[0].to(device)
    test_data = next(iter(test_loader))[0].to(device)
    
    print("\nGerando explicação SHAP...")
    start_time = time.time()
    
    # Criar explainer SHAP
    explainer = Shap(
        model=model,
        background_tensor=background_data,
        test_tensor=test_data,
        device=device
    )
    
    # Obter valores SHAP globais
    shap_values = explainer.explain_global()
    
    end_time = time.time()
    print(f"Tempo para gerar explicação SHAP: {end_time - start_time:.2f} segundos")
    
    # Nomes dos descritores físico-químicos
    descriptor_names = [
        "Molecular Weight", "LogP", "TPSA", "H-bond Donors", "H-bond Acceptors",
        "Rotatable Bonds", "Aromatic Rings", "Balaban J Index", "Drug-likeness (QED)"
    ]
    
    # Visualizar importância global
    plt.figure(figsize=(12, 8))
    
    # Ordenar descritores por importância
    indices = np.argsort(np.abs(shap_values))
    n_features = min(len(descriptor_names), len(shap_values))
    
    plt.barh(range(n_features), 
             [shap_values[i] for i in indices[-n_features:]],
             color=['blue' if v > 0 else 'red' for v in [shap_values[i] for i in indices[-n_features:]]])
    
    plt.yticks(range(n_features), 
               [descriptor_names[i] if i < len(descriptor_names) else f"Feature {i}" 
                for i in indices[-n_features:]])
    
    plt.title("Global Importance of Physicochemical Descriptors (SHAP)", fontsize=14)
    plt.xlabel("Average SHAP Value (Impact Magnitude)", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_mlp_explanation.png", dpi=300)
    plt.show()
    
    print(f"Explicação SHAP salva em 'shap_mlp_explanation.png'")


def explain_gcn_with_graphshap(model_path, device):
    """Gera explicação GraphShap para o modelo GCN."""
    
    # Carregar dataset de grafos
    gd = graph_datasets()
    
    # Obter dataloaders
    train_loader, val_loader, test_loader, _, _, _ = gd.get_paired_dataloaders(
        dataset_name='QM9', 
        batch_size=16,  # Batch menor para explicação
        seed=42,
        n_noise=0,
        noise_scale=0.0
    )
    
    # Obter amostra para inicializar modelo
    data_sample = next(iter(test_loader))
    
    # Inicializar e carregar modelo GCN
    gcn_model = GCN(num_features=data_sample.x.size(1)).to(device)
    gcn_model.load_state_dict(torch.load(model_path, map_location=device))
    gcn_model.eval()
    
    # Avaliar o modelo
    test_mse = 0.0
    num_samples = 0
    property_idx = 0  # Propriedade que foi usada no treinamento
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Selecionar propriedade específica para avaliação
            target = batch.y[:, property_idx].view(-1, 1)
            
            # Forward pass
            out = gcn_model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Garantir dimensões compatíveis
            if out.shape[0] != target.shape[0]:
                min_size = min(out.shape[0], target.shape[0])
                out = out[:min_size]
                target = target[:min_size]
            
            test_mse += ((out - target) ** 2).sum().item()
            num_samples += target.size(0)
    
    test_mse /= num_samples
    print(f"MSE no conjunto de teste: {test_mse:.6f}")
    print(f"RMSE no conjunto de teste: {np.sqrt(test_mse):.6f}")
    
    # Selecionar uma amostra individual para explicação
    sample_data = next(iter(test_loader))
    sample_graph = sample_data.to(device)
    
    # Criar explainer GraphShap
    print("\nGerando explicação GraphShap...")
    start_time = time.time()
    
    explainer = GraphShap(data=sample_graph, model=gcn_model, device=device)
    feature_importance = explainer.explain(num_samples=50)
    
    end_time = time.time()
    print(f"Tempo para gerar explicação GraphShap: {end_time - start_time:.2f} segundos")
    
    # Nomes das características dos nós em QM9 (atômicas)
    node_feature_names = [
        "C atom", "N atom", "O atom", "F atom", "H atom", 
        "Formal charge", "Aromaticity", "sp Hybridization", "sp² Hybridization", 
        "sp³ Hybridization", "Number of Hydrogens"
    ]
    
    # Completar nomes se necessário
    while len(node_feature_names) < len(feature_importance):
        node_feature_names.append(f"Feature {len(node_feature_names)}")
    
    # Visualizar importância global das características
    plt.figure(figsize=(12, 8))
    
    # Ordenar características por importância
    indices = np.argsort(np.abs(feature_importance))
    n_features = min(len(feature_importance), 15)  # Mostrar no máximo 15 features
    
    plt.barh(range(n_features), 
             [feature_importance[i] for i in indices[-n_features:]],
             color=['blue' if v > 0 else 'red' for v in [feature_importance[i] for i in indices[-n_features:]]])
    
    plt.yticks(range(n_features), [node_feature_names[i] for i in indices[-n_features:]])
    plt.title("Importance of Atomic Features (GraphShap)", fontsize=14)
    plt.xlabel("GraphShap Value (Impact Magnitude)", fontsize=12)
    plt.tight_layout()
    plt.savefig("graphshap_gcn_explanation.png", dpi=300)
    plt.show()
    
    print(f"Explicação GraphShap salva em 'graphshap_gcn_explanation.png'")


if __name__ == "__main__":
    explain_models()