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

# Classe de cache para otimização de dados
class DataCache:
    def __init__(self):
        self.cached_data = {}
        self.qm9_instance = None
    
    def get_qm9(self):
        if self.qm9_instance is None:
            safe_log("🔄 Carregando QM9 pela primeira vez...")
            self.qm9_instance = qm9_tabular()
        return self.qm9_instance
    
    def get_data(self, descriptor_type, att_index):
        cache_key = f"{descriptor_type}_{att_index}"
        
        if cache_key not in self.cached_data:
            safe_log(f"💾 Cache miss - carregando {cache_key}")
            qm9 = self.get_qm9()
            X, Y, _ = qm9.compute_descriptors(
                descriptor_type=descriptor_type,
                att_index=att_index,
                list_mols=[]
            )
            self.cached_data[cache_key] = (X, Y)
        else:
            safe_log(f"⚡ Cache hit - usando {cache_key}")
            
        return self.cached_data[cache_key]
    
    def clear_cache(self):
        self.cached_data.clear()
        safe_log("🗑️  Cache limpo")

# Instância global do cache
data_cache = DataCache()

def optimize_for_gpu():
    """
    Configura otimizações para GPU e controle de memória
    """
    if torch.cuda.is_available():
        # Configurações de GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Limpeza de cache GPU
        torch.cuda.empty_cache()
        
        # Informações da GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        safe_log(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return torch.device('cuda')
    else:
        safe_log("💻 Usando CPU (GPU não disponível)")
        return torch.device('cpu')

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

def create_experiment_directory():
    """
    Cria diretório único para o experimento
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(os.getcwd(), "experiments", f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    safe_log(f"📁 Experimento criado: {experiment_dir}")
    return experiment_dir

def run_single_model_safe(att_index, descriptor_type):
    """
    Executa um único modelo com tratamento robusto de erros
    """
    model_id = f"mlp_att{att_index}_{descriptor_type}"
    result = {
        'model': model_id,
        'status': 'unknown',
        'duration_minutes': 0,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        # Log do processamento
        safe_log(f"🔄 Processando: {model_id}")
        
        # Validação básica
        is_valid, validation_info = validate_parameters(att_index, descriptor_type)
        if not is_valid:
            raise ValueError(f"Parâmetros inválidos: {'; '.join(validation_info['errors'])}")
        
        # Executar análise principal
        run_specific_model_with_explanations(att_index=att_index, descriptor_type=descriptor_type)
        
        # Sucesso
        result['status'] = 'success'
        result['duration_minutes'] = round((time.time() - start_time) / 60, 2)
        safe_log(f"✅ Sucesso - {model_id} ({result['duration_minutes']} min)")
        return True
        
    except KeyboardInterrupt:
        result['status'] = 'interrupted'
        result['error'] = 'Interrompido pelo usuário'
        safe_log(f"🛑 Interrompido - {model_id}")
        raise  # Re-raise para parar execução geral
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['duration_minutes'] = round((time.time() - start_time) / 60, 2)
        safe_log(f"❌ Erro - {model_id}: {str(e)}")
        return False

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

def load_model_optimized(descriptor_type, att_index, device):
    """
    Carrega modelo de forma otimizada com cache e verificações
    """
    model_path = os.path.join(os.getcwd(), 'models', f'Large/mlp_qm9_{descriptor_type}_att{att_index}.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    # Carregar dados do cache
    safe_att_index = 0 if att_index >= 10 else att_index
    X_data, _ = data_cache.get_data(descriptor_type, safe_att_index)
    input_dim = X_data.shape[1]
    
    # Carregar state dict
    state_dict = torch.load(model_path, map_location='cpu')  # Sempre carregar em CPU primeiro
    checkpoint_input_dim = state_dict['layers.0.weight'].shape[1]
    
    # Auto-correção para CM
    if checkpoint_input_dim != input_dim and descriptor_type == 'CM':
        input_dim = checkpoint_input_dim
        safe_log(f"⚠️  CM auto-correção: usando {input_dim} features")
    
    # Criar modelo
    model = MLP(input_dim, 1, layers=[128, 64, 32], device=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Mover para GPU se disponível
    if device.type == 'cuda':
        model = model.to(device)
        torch.cuda.empty_cache()  # Limpar cache após carregamento
    
    return model, input_dim

def create_model_and_load_data(qm9, descriptor_type, att_index, device):
    """
    Função auxiliar simplificada para criar modelo e carregar dados
    """
    # CORREÇÃO: Usar att_index=0 se o original for problemático
    safe_att_index = 0 if att_index >= 10 else att_index
    
    # Obter dimensões dos dados - SEM list_mols para usar dataset completo
    X_sample, _, _ = qm9.compute_descriptors(
        descriptor_type=descriptor_type, 
        att_index=safe_att_index,  # Usar índice seguro
        list_mols=[]  # Lista vazia = usar todos os dados
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
        if descriptor_type == 'CM':
            # Para CM, sempre usar a dimensão do modelo para evitar incompatibilidades
            input_dim = checkpoint_input_dim  # Usar dimensão do modelo
        else:
            raise ValueError(f"Incompatibilidade: modelo={checkpoint_input_dim}, dados={input_dim}")
    
    # Criar e carregar modelo
    model = MLP(input_dim, 1, layers=[128, 64, 32], device=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model, input_dim

def prepare_data_optimized(descriptor_type, att_index, input_dim, device, batch_size=32):
    """
    Prepara dados de forma otimizada com batching e controle de memória
    """
    safe_att_index = 0 if att_index >= 10 else att_index
    X_all, Y_all = data_cache.get_data(descriptor_type, safe_att_index)
    
    # Verificar compatibilidade
    if X_all.shape[1] != input_dim:
        if descriptor_type == 'CM':
            # Para CM, pegar apenas as features necessárias
            X_all = X_all[:, :input_dim]
        else:
            raise ValueError(f"Incompatibilidade dimensional: {X_all.shape[1]} vs {input_dim}")
    
    # Dividir dados
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )
    
    # Limitar amostras para não sobrecarregar memória
    max_samples = 1000 if device.type == 'cuda' else 500
    X_train = X_train[:max_samples]
    X_test = X_test[:min(200, len(X_test))]
    
    # Converter para tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    
    # Mover para device em batches para economizar memória
    if device.type == 'cuda':
        # Verificar memória disponível
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        tensor_memory = X_train_tensor.numel() * 4  # 4 bytes por float
        
        if tensor_memory > available_memory * 0.8:  # Usar no máximo 80% da memória
            safe_log("⚠️  Memória GPU limitada, reduzindo dados")
            reduction_factor = int(available_memory * 0.8 / tensor_memory)
            X_train_tensor = X_train_tensor[:len(X_train_tensor)//reduction_factor]
            X_test_tensor = X_test_tensor[:len(X_test_tensor)//reduction_factor]
    
    return X_train_tensor.to(device), X_test_tensor.to(device)

def prepare_explanation_data(qm9, descriptor_type, att_index, input_dim):
    """
    Prepara dados para explicação de forma simplificada
    """
    # CORREÇÃO: Usar att_index=0 se o original for problemático
    safe_att_index = 0 if att_index >= 10 else att_index
    
    # Carregar dados completos - SEM list_mols
    X_all, Y_all, _ = qm9.compute_descriptors(
        descriptor_type=descriptor_type, 
        att_index=safe_att_index,  # Usar índice seguro
        list_mols=[]  # Lista vazia = usar todos os dados
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

def run_explanations_optimized(model, X_train, X_test, device, experiment_dir, descriptor_type, att_index):
    """
    Executa explicações de forma otimizada E SALVA os resultados
    """
    results = {}
    
    # Criar subdiretório para este modelo
    model_dir = os.path.join(experiment_dir, f"{descriptor_type}_att{att_index}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Obter nomes das features
    try:
        qm9 = data_cache.get_qm9()
        feature_names = qm9.get_descriptor_names(descriptor_type)
    except:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    # SHAP otimizado
    try:
        safe_log("🔍 Executando SHAP...")
        with torch.cuda.device(device) if device.type == 'cuda' else torch.no_grad():
            explainer = Shap(model, X_train[:100], X_test[:50], device)
            shap_global = explainer.explain_global()
            shap_local = explainer.explain_local(index=0)
            
            # Salvar SHAP
            shap_results = {
                'global_importance': shap_global.tolist() if isinstance(shap_global, np.ndarray) else shap_global,
                'local_explanation': shap_local.tolist() if isinstance(shap_local, np.ndarray) else shap_local,
                'feature_names': feature_names,
                'model_type': f"{descriptor_type}_att{att_index}",
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Salvar JSON
            with open(os.path.join(model_dir, 'shap_results.json'), 'w') as f:
                json.dump(shap_results, f, indent=2)
            
            # Gerar e salvar gráficos SHAP (TOP 15 features)
            try:
                shap_array = np.array(shap_global)
                top_indices = np.argsort(np.abs(shap_array))[-15:]
                top_values = shap_array[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                # Gráfico global SHAP - Bar plot (TOP 15)
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_values)), top_values)
                plt.yticks(range(len(top_values)), top_names)
                plt.xlabel('SHAP Global Importance')
                plt.title(f'SHAP Global Top 15 - {descriptor_type} att{att_index}')
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'shap_global.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Gráfico radar SHAP (TOP 15)
                plt.figure(figsize=(10, 10))
                radar_plot(
                    top_values, 
                    top_names, 
                    title=f"SHAP Radar Top 15 - {descriptor_type} att{att_index}"
                )
                plt.savefig(os.path.join(model_dir, 'shap_radar.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Gráfico horizontal bar SHAP (TOP 15)
                horizontal_bar_plot(
                    top_values, 
                    top_names, 
                    title=f"SHAP Top 15 - {descriptor_type} att{att_index}",
                    save_path=model_dir,
                    filename='shap_horizontal.png'
                )
                
            except Exception as plot_e:
                safe_log(f"⚠️ Erro ao gerar gráficos SHAP: {plot_e}")
            
            results['shap'] = True
            safe_log("✅ SHAP concluído e salvo")
            
    except Exception as e:
        safe_log(f"❌ SHAP falhou: {str(e)}")
        results['shap'] = False
    
    # Limpar cache entre explicadores
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # LIME otimizado
    try:
        safe_log("🔍 Executando LIME...")
        with torch.cuda.device(device) if device.type == 'cuda' else torch.no_grad():
            lime_explainer = LIME(model, X_train[:100], X_test[:50], device)
            lime_explanations = lime_explainer.explain_local(index=0)
            
            # Salvar LIME
            lime_results = {
                'local_explanation': lime_explanations.tolist() if isinstance(lime_explanations, np.ndarray) else lime_explanations,
                'feature_names': feature_names,
                'model_type': f"{descriptor_type}_att{att_index}",
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Salvar JSON
            with open(os.path.join(model_dir, 'lime_results.json'), 'w') as f:
                json.dump(lime_results, f, indent=2)
            
            # Gerar e salvar gráficos LIME (TOP 15 features)
            try:
                lime_array = np.array(lime_explanations)
                top_indices = np.argsort(np.abs(lime_array))[-15:]
                top_values = lime_array[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                # Gráfico LIME - Bar plot (TOP 15)
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_values)), top_values)
                plt.yticks(range(len(top_values)), top_names)
                plt.xlabel('LIME Local Importance')
                plt.title(f'LIME Local Top 15 - {descriptor_type} att{att_index}')
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'lime_local.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Gráfico radar LIME (TOP 15)
                plt.figure(figsize=(10, 10))
                radar_plot(
                    top_values, 
                    top_names, 
                    title=f"LIME Radar Top 15 - {descriptor_type} att{att_index}"
                )
                plt.savefig(os.path.join(model_dir, 'lime_radar.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Gráfico horizontal bar LIME (TOP 15)
                horizontal_bar_plot(
                    top_values, 
                    top_names, 
                    title=f"LIME Top 15 - {descriptor_type} att{att_index}",
                    save_path=model_dir,
                    filename='lime_horizontal.png'
                )
                
            except Exception as plot_e:
                safe_log(f"⚠️ Erro ao gerar gráficos LIME: {plot_e}")
            
            results['lime'] = True
            safe_log("✅ LIME concluído e salvo")
            
    except Exception as e:
        safe_log(f"❌ LIME falhou: {str(e)}")
        results['lime'] = False
    
    # Salvar métricas de avaliação
    try:
        # Fazer predições
        with torch.no_grad():
            y_pred = model(X_test).cpu().numpy()
        
        # Calcular métricas básicas
        metrics = {
            'model_info': {
                'descriptor_type': descriptor_type,
                'att_index': att_index,
                'input_dim': X_train.shape[1],
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            },
            'predictions': {
                'mean_prediction': float(np.mean(y_pred)),
                'std_prediction': float(np.std(y_pred)),
                'min_prediction': float(np.min(y_pred)),
                'max_prediction': float(np.max(y_pred))
            }
        }
        
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except Exception as e:
        safe_log(f"⚠️ Erro ao calcular métricas: {e}")
    
    safe_log(f"💾 Resultados salvos em: {model_dir}")
    return results

def run_specific_model_with_explanations(att_index=0, descriptor_type='Physicochemical', experiment_dir=None):
    """
    Versão otimizada da função principal COM SALVAMENTO
    """
    # Configurar dispositivo e otimizações
    device = optimize_for_gpu()
    
    # Criar diretório se não fornecido
    if experiment_dir is None:
        experiment_dir = create_experiment_directory()
    
    # Validação
    is_valid, validation_info = validate_parameters(att_index, descriptor_type)
    if not is_valid:
        raise ValueError(f"Parâmetros inválidos: {'; '.join(validation_info['errors'])}")
    
    safe_log(f"🚀 Processando {descriptor_type}_att{att_index}")
    
    try:
        # 1. Carregar modelo (otimizado)
        safe_log("📦 Carregando modelo...")
        model, input_dim = load_model_optimized(descriptor_type, att_index, device)
        
        # 2. Preparar dados (otimizado)
        safe_log("📊 Preparando dados...")
        X_train, X_test = prepare_data_optimized(descriptor_type, att_index, input_dim, device)
        
        # 3. Executar explicações E SALVAR (otimizado)
        safe_log("🧠 Executando explicações...")
        results = run_explanations_optimized(model, X_train, X_test, device, experiment_dir, descriptor_type, att_index)
        
        safe_log(f"✅ Concluído e salvo: SHAP={results['shap']}, LIME={results['lime']}")
        return True
        
    except Exception as e:
        safe_log(f"❌ Erro: {str(e)}")
        return False
    
    finally:
        # Limpeza final
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def run_all_large_models():
    """
    Versão otimizada que processa modelos de forma eficiente
    """
    device = optimize_for_gpu()
    
    models_dir = os.path.join(os.getcwd(), "models", "Large")
    if not os.path.exists(models_dir):
        safe_log(f"❌ Diretório não encontrado: {models_dir}")
        return

    # Criar diretório do experimento
    experiment_dir = create_experiment_directory()

    # Verificação rápida primeiro
    compatible = quick_compatibility_check()
    if not compatible:
        safe_log("❌ Nenhum modelo compatível")
        return
    
    safe_log(f"🚀 Processando {len(compatible)} modelos compatíveis")
    safe_log(f"📁 Salvando em: {experiment_dir}")
    
    successful = 0
    failed = 0
    execution_log = []
    
    for i, (fname, descriptor_type, att_index, input_dim) in enumerate(compatible):
        safe_log(f"📊 Progresso: {i+1}/{len(compatible)}")
        
        try:
            result = run_specific_model_with_explanations(att_index, descriptor_type, experiment_dir)
            if result:
                successful += 1
            else:
                failed += 1
                
        except KeyboardInterrupt:
            safe_log("� Interrompido pelo usuário")
            break
        except Exception as e:
            safe_log(f"❌ Erro fatal em {fname}: {str(e)}")
            failed += 1
        
        # Limpeza periódica
        if i % 3 == 0:  # A cada 3 modelos
            data_cache.clear_cache()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    
    # Salvar log de execução
    final_report = {
        'experiment_info': {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_models': len(compatible),
            'successful': successful,
            'failed': failed,
            'experiment_directory': experiment_dir
        },
        'execution_log': execution_log
    }
    
    with open(os.path.join(experiment_dir, 'experiment_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    safe_log(f"🏁 Finalizado: {successful} sucessos, {failed} falhas")
    safe_log(f"📄 Relatório salvo: {os.path.join(experiment_dir, 'experiment_report.json')}")
    return successful, failed

def quick_compatibility_check():
    """
    Função simples para verificar compatibilidade dos modelos sem carregar dados
    """
    models_dir = os.path.join(os.getcwd(), "models", "Large")
    if not os.path.exists(models_dir):
        safe_log(f"❌ Diretório não encontrado: {models_dir}")
        return []

    safe_log("🔍 Verificação rápida de compatibilidade...")
    
    compatible_models = []
    
    for fname in os.listdir(models_dir):
        if fname.startswith("mlp_qm9_") and fname.endswith(".pth"):
            try:
                # Parse básico do nome
                parts = fname.split("_")
                if len(parts) < 4:
                    continue
                    
                descriptor_type = parts[2]
                att_part = parts[-1].replace("att", "").replace(".pth", "")
                att_index = int(att_part)
                
                # Verificar se arquivo existe e pode ser carregado
                model_path = os.path.join(models_dir, fname)
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Verificar estrutura básica
                if 'layers.0.weight' in state_dict:
                    input_dim = state_dict['layers.0.weight'].shape[1]
                    compatible_models.append((fname, descriptor_type, att_index, input_dim))
                    safe_log(f"✅ {fname}: {input_dim} features")
                else:
                    safe_log(f"❌ {fname}: Estrutura inválida")
                    
            except Exception as e:
                safe_log(f"❌ {fname}: Erro - {str(e)}")
    
    safe_log(f"📊 Modelos compatíveis: {len(compatible_models)}")
    return compatible_models

# Função para testar um modelo específico  
def test_single_model():
    """Função de teste para um modelo específico"""
    result = run_single_model_safe(0, 'Physicochemical')
    safe_log(f"Teste concluído: {result}")

if __name__ == "__main__":
    import sys
    
    # Modo rápido - processar apenas alguns modelos
    if len(sys.argv) > 1 and sys.argv[1] == "--fast":
        safe_log("⚡ Modo rápido ativado")
        # Testar apenas um modelo
        test_single_model()
    else:
        # Modo normal
        compatible = quick_compatibility_check()
        
        if len(compatible) > 0:
            safe_log("🚀 Iniciando análise dos modelos compatíveis...")
            run_all_large_models()
        else:
            safe_log("❌ Nenhum modelo compatível encontrado!")

def quick_compatibility_check():
    """
    Função simples para verificar compatibilidade dos modelos sem carregar dados
    """
    models_dir = os.path.join(os.getcwd(), "models", "Large")
    if not os.path.exists(models_dir):
        safe_log(f"❌ Diretório não encontrado: {models_dir}")
        return []

    safe_log("🔍 Verificação rápida de compatibilidade...")
    
    compatible_models = []
    
    for fname in os.listdir(models_dir):
        if fname.startswith("mlp_qm9_") and fname.endswith(".pth"):
            try:
                # Parse básico do nome
                parts = fname.split("_")
                if len(parts) < 4:
                    continue
                    
                descriptor_type = parts[2]
                att_part = parts[-1].replace("att", "").replace(".pth", "")
                att_index = int(att_part)
                
                # Verificar se arquivo existe e pode ser carregado
                model_path = os.path.join(models_dir, fname)
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Verificar estrutura básica
                if 'layers.0.weight' in state_dict:
                    input_dim = state_dict['layers.0.weight'].shape[1]
                    compatible_models.append((fname, descriptor_type, att_index, input_dim))
                    safe_log(f"✅ {fname}: {input_dim} features")
                else:
                    safe_log(f"❌ {fname}: Estrutura inválida")
                    
            except Exception as e:
                safe_log(f"❌ {fname}: Erro - {str(e)}")
    
    safe_log(f"📊 Modelos compatíveis: {len(compatible_models)}")
    return compatible_models

def create_experiment_directory():
    """
    Cria diretório único para o experimento
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(os.getcwd(), "experiments", f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    safe_log(f"📁 Experimento criado: {experiment_dir}")
    return experiment_dir


# Função para testar um modelo específico  
def test_single_model():
    """Função de teste para um modelo específico"""
    result = run_single_model_safe(0, 'Physicochemical')
    safe_log(f"Teste concluído: {result}")

if __name__ == "__main__":
    # Teste rápido de compatibilidade primeiro
    compatible = quick_compatibility_check()
    
    if len(compatible) > 0:
        safe_log("🚀 Iniciando análise dos modelos compatíveis...")
        # Para executar todos os modelos: run_all_large_models()
        run_all_large_models()
    else:
        safe_log("❌ Nenhum modelo compatível encontrado!")
        safe_log("💡 Verifique se os arquivos .pth estão no diretório models/Large/")