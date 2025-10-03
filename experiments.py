# Experimento: Avaliar fidelidade dos modelos por cluster
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import datetime
from tqdm import tqdm
import sys

from chemxai.data import qm9_tabular
from chemxai.models import MLP
from chemxai.train import train_mlp_qm9
from chemxai.explainers import Shap, LIME
from chemxai.evaluate import TabularAnalyzer
from chemxai.plots import radar_plot, horizontal_bar_plot
from chemxai.data import Cluster

def safe_log(message, level="INFO"):
    """
    Função de logging segura e simplificada
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
    except Exception:
        print(f"{message}")  # Fallback simples

def run_single_model_safe(fname, models_dir):
    """
    Executa um único modelo com tratamento robusto de erros
    """
    result = {
        'model': fname,
        'status': 'unknown',
        'duration_minutes': 0,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        # Parse do nome do arquivo
        safe_log(f"Processando: {fname}")
        
        parts = fname.split("_")
        if len(parts) < 4 or not parts[-1].startswith("att"):
            raise ValueError(f"Formato de nome inválido: {fname}")
            
        descriptor_type = parts[2]
        att_index = int(parts[-1].replace("att", "").replace(".pth", ""))
        
        # Validação básica
        is_valid, validation_info = validate_parameters(att_index, descriptor_type)
        if not is_valid:
            raise ValueError(f"Parâmetros inválidos: {'; '.join(validation_info['errors'])}")
        
        # Executar análise principal
        run_specific_model_with_explanations(att_index=att_index, descriptor_type=descriptor_type)
        
        # Sucesso
        result['status'] = 'success'
        result['duration_minutes'] = round((time.time() - start_time) / 60, 2)
        safe_log(f"✅ Sucesso - {fname} ({result['duration_minutes']} min)")
        
    except KeyboardInterrupt:
        result['status'] = 'interrupted'
        result['error'] = 'Interrompido pelo usuário'
        safe_log(f"🛑 Interrompido - {fname}")
        raise  # Re-raise para parar execução geral
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['duration_minutes'] = round((time.time() - start_time) / 60, 2)
        safe_log(f"❌ Erro - {fname}: {str(e)}")
    
    return result

def validate_parameters(att_index, descriptor_type):
    """
    Valida os parâmetros de entrada de forma simplificada
    """
    errors = []
    warnings = []
    
    # Validar att_index
    if not isinstance(att_index, int) or att_index < 0 or att_index >= 19:
        errors.append(f"att_index deve estar entre 0-18, recebido: {att_index}")
    
    # Validar descriptor_type
    valid_descriptors = [
        'CM', 'Morgan', 'Physicochemical', '3D', 'MACCS', 
        'Topological', 'AtomPair', 'EState', 'Pattern', 
        'Avalon', 'MorganCount', 'Autocorr'
    ]
    
    if descriptor_type not in valid_descriptors:
        errors.append(f"descriptor_type inválido. Válidos: {valid_descriptors}")
    
    # Avisos
    if att_index > 15:
        warnings.append(f"att_index {att_index} pode ter menos dados disponíveis")
    
    is_valid = len(errors) == 0
    validation_info = {'valid': is_valid, 'errors': errors, 'warnings': warnings}
    
    return is_valid, validation_info

def run_cluster_fidelity_experiment(att_index=0, descriptor_type='Physicochemical', num_clusters=5, cluster_size=100):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{timestamp}] Starting cluster fidelity experiment...")

    # 1. Carregar dados
    qm9 = qm9_tabular()
    loaders = qm9.get_paired_dataloaders(
        att_index=att_index,
        batch_size=32,
        descriptor_type=descriptor_type,
        n_noise=0,
        add_noise=False
    )
    train_loader, val_loader, test_loader = loaders

    # 2. Criar clusters usando o train_loader
    cluster_manager = Cluster(train_loader)
    clusters = cluster_manager.create_clusters(num_clusters=num_clusters, size_cluster=cluster_size)
    print(f"Clusters criados: {len(clusters)}")

    # 3. Carregar modelo treinado
    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = f'Large/mlp_qm9_{descriptor_type}_att{att_index}'
    model_path = os.path.join(os.getcwd(), 'models', f'{model_name}.pth')
    model = MLP(input_dim, output_dim, layers=[128, 64, 32], device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # 4. Avaliar fidelidade por cluster
    cluster_results = {}
    feature_names = qm9.get_descriptor_names(descriptor_type)
    for cluster_id, cluster_data in clusters.items():
        # Extrair X e y do cluster
        X_cluster = []
        y_cluster = []
        for xb, yb in cluster_data:
            X_cluster.append(xb)
            y_cluster.append(yb)
        if not X_cluster:
            continue
        X_cluster = torch.cat(X_cluster, dim=0)
        y_cluster = torch.cat(y_cluster, dim=0)

        # Limitar tamanho para explicação
        X_cluster = X_cluster[:100]
        y_cluster = y_cluster[:100]

        # Rodar explicadores
        shap_explainer = Shap(model, X_cluster, X_cluster, device)
        shap_explanation = shap_explainer.explain_global()
        lime_explainer = LIME(model, X_cluster, X_cluster, device)
        lime_explanation = lime_explainer.explain_local(index=0)

        # Predições
        with torch.no_grad():
            y_pred = model(X_cluster.to(device)).cpu().numpy()

        # Calcular fidelidade
        analyzer_shap = TabularAnalyzer(
            model=model,
            explainer=shap_explainer,
            explanation=shap_explanation,
            data=X_cluster,
            y_true=y_cluster.numpy(),
            y_pred=y_pred,
            device=device
        )
        fidelity_shap = analyzer_shap.get_metrics()

        analyzer_lime = TabularAnalyzer(
            model=model,
            explainer=lime_explainer,
            explanation=lime_explanation,
            data=X_cluster,
            y_true=y_cluster.numpy(),
            y_pred=y_pred,
            device=device
        )
        fidelity_lime = analyzer_lime.get_metrics()

        cluster_results[cluster_id] = {
            "shap_fidelity": fidelity_shap,
            "lime_fidelity": fidelity_lime
        }
        print(f"Cluster {cluster_id}: SHAP {fidelity_shap}, LIME {fidelity_lime}")

    # 5. Salvar resultados
    results_path = f"experiments/cluster_fidelity_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(cluster_results, f, indent=2)
    print(f"Resultados salvos em {results_path}")

def create_model_and_load_data(qm9, descriptor_type, att_index, device):
    """
    Função auxiliar simplificada para criar modelo e carregar dados
    """
    # Obter dimensões dos dados
    X_sample, _, _ = qm9.compute_descriptors(
        descriptor_type=descriptor_type, 
        att_index=att_index, 
        list_mols=[]
    )
    
    input_dim = X_sample.shape[1]
    model_path = os.path.join(os.getcwd(), 'models', f'Large/mlp_qm9_{descriptor_type}_att{att_index}.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    # Verificar compatibilidade e carregar modelo
    state_dict = torch.load(model_path, map_location=device)
    checkpoint_input_dim = state_dict['layers.0.weight'].shape[1]
    
    # Auto-correção para matriz de Coulomb se necessário
    if checkpoint_input_dim != input_dim:
        if descriptor_type == 'CM' and {checkpoint_input_dim, input_dim} == {29, 841}:
            input_dim = checkpoint_input_dim  # Usar dimensão do modelo
        else:
            raise ValueError(f"Incompatibilidade: modelo={checkpoint_input_dim}, dados={input_dim}")
    
    # Criar e carregar modelo
    model = MLP(input_dim, 1, layers=[128, 64, 32], device=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model, input_dim

def prepare_explanation_data(qm9, descriptor_type, att_index, input_dim):
    """
    Prepara dados para explicação de forma simplificada
    """
    # Carregar dados completos
    X_all, Y_all, _ = qm9.compute_descriptors(
        descriptor_type=descriptor_type, 
        att_index=att_index, 
        list_mols=[]
    )
    
    # Verificar compatibilidade dimensional
    if X_all.shape[1] != input_dim:
        raise ValueError(f"Dimensões incompatíveis: dados={X_all.shape[1]}, modelo={input_dim}")
    
    # Dividir dados para explicação
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )
    
    # Limitar amostras
    X_train = torch.from_numpy(X_train[:500]).float()
    X_test = torch.from_numpy(X_test[:100]).float()
    
    return X_train, X_test

def run_specific_model_with_explanations(att_index=0, descriptor_type='Physicochemical'):
    """
    Executa análise SHAP e LIME para um modelo específico
    """
    
    # Validação simples de parâmetros
    is_valid, validation_info = validate_parameters(att_index, descriptor_type)
    if not is_valid:
        error_msg = f"Parâmetros inválidos: {'; '.join(validation_info['errors'])}"
        safe_log(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Avisos se existirem
    for warning in validation_info['warnings']:
        safe_log(f"⚠️  {warning}")
    
    # Setup básico
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'Large/mlp_qm9_{descriptor_type}_att{att_index}'
    
    safe_log(f"Iniciando análise para: {model_name}")
    
    # Criar diretório de experimento simplificado
    exp_dir = os.path.join(os.getcwd(), "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Encontrar próximo ID
    existing = [int(d.split("_")[-1]) for d in os.listdir(exp_dir) 
               if d.startswith("experiment_") and d.split("_")[-1].isdigit()]
    next_id = max(existing, default=0) + 1
    
    # Criar subpasta específica do modelo
    experiment_id = f"{next_id:03d}"
    model_dir = os.path.join(exp_dir, f"experiment_{experiment_id}", f"{descriptor_type}_att{att_index}")
    os.makedirs(model_dir, exist_ok=True)
    
    safe_log(f"Diretório criado: {model_dir}")

    
    # Sistema de logging simplificado
    log_file = os.path.join(model_dir, "analysis.txt")
    json_log_file = os.path.join(model_dir, "analysis.json")
    
    # Log básico
    experiment_log = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "model": model_name,
        "att_index": att_index,
        "descriptor_type": descriptor_type,
        "results": {}
    }
    
    def save_progress(message, data=None):
        safe_log(message)
        if data:
            experiment_log["results"].update(data)
        with open(json_log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2)
    
    # Inicializar
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_progress(f"Usando dispositivo: {device}")
    
    log_lines = []
    log_lines.append(f"Device: {device}\n")
    log_lines.append(f"Analysis for model: {model_name}\n")
    log_lines.append(f"Experiment directory: {model_dir}\n")

    
    # Carregamento de dados simplificado
    save_progress("Carregando dados...")
    try:
        qm9 = qm9_tabular()
        loaders = qm9.get_paired_dataloaders(
            att_index=att_index,
            batch_size=32,
            descriptor_type=descriptor_type,
            n_noise=0,
            add_noise=False
        )
        train_loader, val_loader, test_loader = loaders
        save_progress(f"Dados carregados - Train: {len(train_loader)} batches")
        log_lines.append("Data loaded.\n")
    except Exception as e:
        error_msg = f"Erro ao carregar dados: {str(e)}"
        save_progress(error_msg)
        raise

    
    # Carregamento do modelo simplificado
    save_progress("Carregando modelo...")
    try:
        # Obter dimensões dos dados
        X_sample, _, _ = qm9.compute_descriptors(
            descriptor_type=descriptor_type, 
            att_index=att_index, 
            list_mols=[]
        )
        
        input_dim = X_sample.shape[1]
        model_path = os.path.join(os.getcwd(), 'models', f'{model_name}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Verificar compatibilidade e carregar modelo
        state_dict = torch.load(model_path, map_location=device)
        checkpoint_input_dim = state_dict['layers.0.weight'].shape[1]
        
        # Auto-correção para matriz de Coulomb se necessário
        if checkpoint_input_dim != input_dim:
            if descriptor_type == 'CM' and {checkpoint_input_dim, input_dim} == {29, 841}:
                save_progress(f"Auto-corrigindo CM: {checkpoint_input_dim} features no modelo")
                input_dim = checkpoint_input_dim  # Usar dimensão do modelo
            else:
                raise ValueError(f"Incompatibilidade: modelo={checkpoint_input_dim}, dados={input_dim}")
        
        # Criar e carregar modelo
        model = MLP(input_dim, 1, layers=[128, 64, 32], device=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        save_progress(f"Modelo carregado: {input_dim} features")
        
    except Exception as e:
        error_msg = f"Erro no modelo: {str(e)}"
        save_progress(error_msg)
        raise

    # Preparação dos dados para explicação - versão simplificada
    save_progress("Preparando dados para explicação...")
    try:
        X_train, X_test = prepare_explanation_data(qm9, descriptor_type, att_index, input_dim)
        
        # Mover dados para o dispositivo
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        
        save_progress(f"Dados preparados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        
    except Exception as e:
        error_msg = f"Erro na preparação: {str(e)}"
        save_progress(error_msg)
        raise

    # Executar explicações - versão simplificada
    save_progress("Executando SHAP...")
    try:
        # Usar explainers do chemxai
        explainer = Shap(model, X_train, X_test, device)
        shap_values = explainer.explain_global()
        save_progress("SHAP concluído")
        
    except Exception as e:
        save_progress(f"Erro SHAP: {str(e)}")
        shap_values = None
    
    save_progress("Executando LIME...")
    try:
        lime_explainer = LIME(model, X_train, X_test, device)
        lime_explanations = lime_explainer.explain_local(index=0)
        save_progress("LIME concluído")
        
    except Exception as e:
        save_progress(f"Erro LIME: {str(e)}")
        lime_explanations = None
    
    # Resultados finais
    results = {
        'model_id': f"mlp_att{att_index}_{descriptor_type}",
        'shap_completed': shap_values is not None,
        'lime_completed': lime_explanations is not None,
        'samples': len(X_test)
    }
    
    save_progress("Análise concluída", results)
    return True


def run_all_large_models():
    """
    Executa análises para todos os modelos em models/Large (versão simplificada)
    """
    models_dir = os.path.join(os.getcwd(), "models", "Large")
    if not os.path.exists(models_dir):
        safe_log(f"❌ Diretório não encontrado: {models_dir}")
        return

    safe_log("🚀 Iniciando análise de todos os modelos Large")
    
    # Listar modelos válidos
    valid_models = []
    for fname in os.listdir(models_dir):
        if fname.startswith("mlp_qm9_") and fname.endswith(".pth"):
            try:
                # Extrair descriptor_type e att_index do nome do arquivo
                parts = fname.split("_")
                if len(parts) < 4:
                    continue
                    
                descriptor_type = parts[2]
                att_part = parts[-1].replace("att", "").replace(".pth", "")
                att_index = int(att_part)
                
                # Validar parâmetros
                is_valid, _ = validate_parameters(att_index, descriptor_type)
                if is_valid:
                    valid_models.append((att_index, descriptor_type))
                    
            except (ValueError, IndexError):
                continue
    
    safe_log(f"📊 Encontrados {len(valid_models)} modelos válidos")
    
    # Executar análises com tratamento de erro
    successful = 0
    failed = 0
    
    for att_index, descriptor_type in valid_models:
        result = run_single_model_safe(att_index, descriptor_type)
        if result:
            successful += 1
        else:
            failed += 1
    
    safe_log(f"✅ Concluído: {successful} sucessos, {failed} falhas")
    return successful, failed



# Função para testar um modelo específico  
def test_single_model():
    """Função de teste para um modelo específico"""
    result = run_single_model_safe(0, 'Physicochemical')
    safe_log(f"Teste concluído: {result}")

if __name__ == "__main__":
    # Para executar todos os modelos: run_all_large_models()
    # Para testar um modelo: test_single_model()
    run_all_large_models()