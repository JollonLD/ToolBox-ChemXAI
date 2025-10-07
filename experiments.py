# Experimento: Avaliar fidelidade dos modelos por cluster
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para evitar problemas de memória
import matplotlib.pyplot as plt
# Configurar matplotlib para evitar warnings de muitas figuras
plt.rcParams['figure.max_open_warning'] = 0  # Desabilitar warning
import json
import time
import datetime
import sys
from torch.utils.data import TensorDataset, DataLoader

from chemxai.data import qm9_tabular
from chemxai.models import MLP
from chemxai.explainers import Shap, LIME
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

def run_cluster_analysis(experiment_dir, descriptor_type='AtomPair', att_index=10):
    """
    Função para análise por cluster das explicações
    """
    # Configurar dispositivo
    device = optimize_for_gpu()
    
    # Criar diretório para salvar resultados dos clusters
    cluster_dir = os.path.join(experiment_dir, f"cluster_analysis_{descriptor_type}_att{att_index}")
    os.makedirs(cluster_dir, exist_ok=True)
    
    safe_log(f"📁 Análise de clusters será salva em: {cluster_dir}")

    # 1. Obter os dados (features e targets) com otimização para descriptors lentos
    try:
        safe_log(f"💾 Carregando dados {descriptor_type}...")
        
        # Para descriptors que sabemos que são lentos, usar att_index menor se disponível
        safe_att_index = att_index
        if descriptor_type in ['Physicochemical', '3D', 'Autocorr'] and att_index >= 10:
            safe_att_index = 0  # Usar att_index 0 que tem cache mais rápido
            safe_log(f"⚡ Otimização: usando att_index {safe_att_index} para {descriptor_type}")
        
        X_all, Y_all = data_cache.get_data(descriptor_type, safe_att_index)
        safe_log(f"✅ Dados carregados: {X_all.shape[0]} amostras, {X_all.shape[1]} features")
        
    except Exception as e:
        safe_log(f"❌ Erro ao carregar dados {descriptor_type}: {e}")
        return False
    
    # 2. Criar um Dataset e DataLoader
    dataset = TensorDataset(torch.from_numpy(X_all).float(), torch.from_numpy(Y_all).float())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # 3. Instanciar o Cluster
    cluster_manager = Cluster(dataloader)
    
    # 4. Criar clusters automáticos com otimização
    safe_log("🔄 Criando clusters com KMeans...")
    clusters = cluster_manager.create_clusters_kmeans(n_clusters=5, use_features=False)
    
    # Otimização: limitar amostras por cluster para acelerar SHAP
    max_samples_per_cluster = 500  # Limite para evitar SHAP muito lento
    for cluster_id in clusters.keys():
        if clusters[cluster_id]['size'] > max_samples_per_cluster:
            safe_log(f"⚡ Limitando cluster {cluster_id} de {clusters[cluster_id]['size']} para {max_samples_per_cluster} amostras")
            # Manter apenas as primeiras amostras
            clusters[cluster_id]['features'] = clusters[cluster_id]['features'][:max_samples_per_cluster]
            clusters[cluster_id]['targets'] = clusters[cluster_id]['targets'][:max_samples_per_cluster]
            clusters[cluster_id]['indices'] = clusters[cluster_id]['indices'][:max_samples_per_cluster]
            clusters[cluster_id]['size'] = max_samples_per_cluster
    
    # 5. Imprimir resumo detalhado dos clusters
    cluster_manager.print_cluster_info()
    
    # 6. Carregar todos os modelos de models/Large compatíveis com o descriptor_type
    safe_log("🔍 Procurando modelos compatíveis...")
    models_dir = os.path.join(os.getcwd(), "models", "Large")
    compatible_models = []
    
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.startswith(f"mlp_qm9_{descriptor_type}_") and fname.endswith(".pth"):
                try:
                    # Parse do nome do arquivo
                    parts = fname.split("_")
                    if len(parts) >= 4:
                        model_att_part = parts[-1].replace("att", "").replace(".pth", "")
                        model_att_index = int(model_att_part)
                        
                        # Verificar se o modelo pode ser carregado
                        model_path = os.path.join(models_dir, fname)
                        state_dict = torch.load(model_path, map_location='cpu')
                        
                        if 'layers.0.weight' in state_dict:
                            input_dim = state_dict['layers.0.weight'].shape[1]
                            compatible_models.append((fname, model_att_index, input_dim, model_path))
                            safe_log(f"✅ Modelo encontrado: {fname}")
                        
                except Exception as e:
                    safe_log(f"❌ Erro ao verificar {fname}: {e}")
    
    if not compatible_models:
        safe_log(f"❌ Nenhum modelo compatível encontrado para {descriptor_type}")
        return
    
    safe_log(f"📊 {len(compatible_models)} modelos compatíveis encontrados")
    
    # Selecionar o primeiro modelo compatível para análise
    selected_model = compatible_models[0]
    model_fname, model_att_index, model_input_dim, model_path = selected_model
    safe_log(f"🎯 Usando modelo: {model_fname}")
    
    # 7. Fazer explicação com SHAP para cada cluster do modelo selecionado
    try:
        # Carregar o modelo
        safe_log("📦 Carregando modelo para análise de clusters...")
        model = MLP(model_input_dim, 1, layers=[128, 64, 32], device=device)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        if device.type == 'cuda':
            model = model.to(device)
        
        # Obter nomes das features
        try:
            qm9 = data_cache.get_qm9()
            feature_names = qm9.get_descriptor_names(descriptor_type)
        except:
            feature_names = [f"feature_{i}" for i in range(model_input_dim)]
        
        # Analisar cada cluster
        cluster_explanations = {}
        
        for cluster_id in clusters.keys():
            safe_log(f"🧠 Analisando cluster {cluster_id}...")
            
            cluster_data = clusters[cluster_id]
            if cluster_data['size'] == 0:
                safe_log(f"⚠️ Cluster {cluster_id} vazio, pulando...")
                continue
            
            # Preparar dados do cluster
            cluster_features = np.array(cluster_data['features'])
            cluster_targets = np.array(cluster_data['targets'])
            
            # Verificar compatibilidade dimensional
            if cluster_features.shape[1] != model_input_dim:
                if descriptor_type == 'CM':
                    # Para CM, ajustar dimensões
                    cluster_features = cluster_features[:, :model_input_dim]
                else:
                    safe_log(f"❌ Incompatibilidade dimensional no cluster {cluster_id}: {cluster_features.shape[1]} vs {model_input_dim}")
                    continue
            
            # Limitar amostras para SHAP (performance)
            max_samples = min(100, cluster_features.shape[0])
            cluster_features_sample = cluster_features[:max_samples]
            
            # Converter para tensors
            X_cluster = torch.from_numpy(cluster_features_sample).float().to(device)
            
            try:
                # Executar SHAP no cluster
                safe_log(f"🔍 Executando SHAP no cluster {cluster_id} ({max_samples} amostras)...")
                
                # Usar uma pequena amostra como background
                background_size = min(50, max_samples // 2)
                X_background = X_cluster[:background_size]
                X_test_cluster = X_cluster[:min(25, max_samples)]
                
                explainer = Shap(model, X_background, X_test_cluster, device)
                shap_global_cluster = explainer.explain_global()
                
                # Salvar explicações do cluster
                cluster_explanations[cluster_id] = {
                    'shap_global': shap_global_cluster.tolist() if isinstance(shap_global_cluster, np.ndarray) else shap_global_cluster,
                    'cluster_info': {
                        'size': cluster_data['size'],
                        'min_target': cluster_data.get('min_target'),
                        'max_target': cluster_data.get('max_target'),
                        'mean_target': cluster_data.get('mean_target')
                    },
                    'n_samples_analyzed': max_samples
                }
                
                safe_log(f"✅ SHAP concluído para cluster {cluster_id}")
                
            except Exception as e:
                safe_log(f"❌ Erro SHAP no cluster {cluster_id}: {e}")
                continue
            
            # Limpeza de memória
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 8. Fazer plots das explicações globais por cluster e salvar
        safe_log("📊 Gerando plots por cluster...")
        
        # Salvar explicações em JSON
        explanations_file = os.path.join(cluster_dir, 'cluster_explanations.json')
        with open(explanations_file, 'w') as f:
            json.dump({
                'model_info': {
                    'model_file': model_fname,
                    'descriptor_type': descriptor_type,
                    'att_index': model_att_index,
                    'input_dim': model_input_dim
                },
                'cluster_explanations': cluster_explanations,
                'feature_names': feature_names,
                'timestamp': datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        safe_log(f"💾 Explicações salvas em: {explanations_file}")
        
        # Gerar plots comparativos
        for cluster_id, explanation_data in cluster_explanations.items():
            try:
                shap_values = np.array(explanation_data['shap_global'])
                cluster_info = explanation_data['cluster_info']
                
                # Top 15 features mais importantes
                top_indices = np.argsort(np.abs(shap_values))[-15:]
                top_values = shap_values[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                # Plot individual do cluster
                cluster_plot_dir = os.path.join(cluster_dir, f"cluster_{cluster_id}")
                os.makedirs(cluster_plot_dir, exist_ok=True)
                
                # Bar plot usando horizontal_bar_plot
                try:
                    horizontal_bar_plot(
                        values=top_values,
                        feature_names=top_names,
                        title=f'Cluster {cluster_id} - SHAP Horizontal Top 15',
                        save_path=os.path.join(cluster_plot_dir),
                        filename=f'cluster_{cluster_id}_shap.png'
                    )
                except Exception as bar_e:
                    safe_log(f"⚠️ Erro no horizontal bar plot cluster {cluster_id}: {bar_e}")
                
                # Radar plot
                try:
                    plt.figure(figsize=(10, 10))
                    radar_plot(
                        top_values, 
                        top_names, 
                        title=f"Cluster {cluster_id} - SHAP Radar (Size: {cluster_info['size']})"
                    )
                    plt.savefig(os.path.join(cluster_plot_dir, f'cluster_{cluster_id}_radar.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()  # Fechar figura explicitamente
                except Exception as radar_e:
                    safe_log(f"⚠️ Erro no radar plot cluster {cluster_id}: {radar_e}")
                finally:
                    # Garantir que todas as figuras sejam fechadas
                    plt.close('all')
                
                safe_log(f"📈 Plots gerados para cluster {cluster_id}")
                
            except Exception as e:
                safe_log(f"❌ Erro ao gerar plots do cluster {cluster_id}: {e}")
        
        # Plot comparativo entre clusters
        try:
            safe_log("📊 Gerando plot comparativo entre clusters...")
            
            fig, axes = plt.subplots(len(cluster_explanations), 1, 
                                   figsize=(15, 5 * len(cluster_explanations)))
            
            if len(cluster_explanations) == 1:
                axes = [axes]
            
            for i, (cluster_id, explanation_data) in enumerate(cluster_explanations.items()):
                shap_values = np.array(explanation_data['shap_global'])
                cluster_info = explanation_data['cluster_info']
                
                # Top 10 para comparação
                top_indices = np.argsort(np.abs(shap_values))[-10:]
                top_values = shap_values[top_indices]
                top_names = [feature_names[j] for j in top_indices]
                
                colors = ['red' if v < 0 else 'blue' for v in top_values]
                axes[i].barh(range(len(top_values)), top_values, color=colors)
                axes[i].set_yticks(range(len(top_values)))
                axes[i].set_yticklabels(top_names)
                axes[i].set_xlabel('SHAP Global Importance')
                axes[i].set_title(f'Cluster {cluster_id} (Size: {cluster_info["size"]})')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(cluster_dir, 'clusters_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()  # Fechar figura explicitamente
            
            safe_log("📈 Plot comparativo salvo")
            
        except Exception as e:
            safe_log(f"❌ Erro ao gerar plot comparativo: {e}")
        finally:
            # Garantir limpeza completa de figuras
            plt.close('all')
        
        # Salvar resumo final
        summary = {
            'analysis_summary': {
                'descriptor_type': descriptor_type,
                'att_index': att_index,
                'model_used': model_fname,
                'total_clusters': len(clusters),
                'clusters_analyzed': len(cluster_explanations),
                'total_samples': sum(c['size'] for c in clusters.values()),
                'analysis_timestamp': datetime.datetime.now().isoformat()
            },
            'cluster_summary': {
                cluster_id: {
                    'size': clusters[cluster_id]['size'],
                    'target_stats': {
                        'min': clusters[cluster_id].get('min_target'),
                        'max': clusters[cluster_id].get('max_target'),
                        'mean': clusters[cluster_id].get('mean_target')
                    },
                    'shap_analyzed': cluster_id in cluster_explanations
                }
                for cluster_id in clusters.keys()
            }
        }
        
        summary_file = os.path.join(cluster_dir, 'analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        safe_log(f"📄 Resumo da análise salvo em: {summary_file}")
        safe_log(f"🎉 Análise de clusters concluída! Resultados em: {cluster_dir}")
        
    except Exception as e:
        safe_log(f"❌ Erro na análise de clusters: {e}")
        return False
    
    finally:
        # Limpeza final
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Fechar todas as figuras do matplotlib
        plt.close('all')
        
        # Limpeza adicional de memória
        import gc
        gc.collect()
    
    return True

def run_all_cluster_analysis():
    """
    Executa análise de clusters para múltiplos descriptor types
    """
    safe_log("🚀 Iniciando análise de clusters para múltiplos descriptors...")
    
    # Lista de descriptor types para análise
    descriptor_types = ['AtomPair', 'Morgan', 'Physicochemical', 'MACCS', 'Topological']
    att_index = 10  # Usar att_index padrão
    
    # Criar diretório principal do experimento
    experiment_dir = create_experiment_directory()
    
    successful = 0
    failed = 0
    execution_log = []
    
    for i, descriptor_type in enumerate(descriptor_types):
        safe_log(f"📊 Progresso: {i+1}/{len(descriptor_types)} - Processando {descriptor_type}")
        
        start_time = time.time()
        
        try:
            # Executar análise de cluster para este descriptor
            result = run_cluster_analysis(experiment_dir=experiment_dir, descriptor_type=descriptor_type, att_index=att_index)
            
            duration = round((time.time() - start_time) / 60, 2)
            
            if result:
                successful += 1
                execution_log.append({
                    'descriptor_type': descriptor_type,
                    'att_index': att_index,
                    'status': 'success',
                    'duration_minutes': duration
                })
                safe_log(f"✅ Sucesso - {descriptor_type} ({duration} min)")
            else:
                failed += 1
                execution_log.append({
                    'descriptor_type': descriptor_type,
                    'att_index': att_index,
                    'status': 'failed',
                    'duration_minutes': duration,
                    'error': 'Função retornou False'
                })
                safe_log(f"❌ Falha - {descriptor_type}")
                
        except KeyboardInterrupt:
            safe_log("🛑 Interrompido pelo usuário")
            execution_log.append({
                'descriptor_type': descriptor_type,
                'att_index': att_index,
                'status': 'interrupted',
                'duration_minutes': round((time.time() - start_time) / 60, 2)
            })
            break
            
        except Exception as e:
            duration = round((time.time() - start_time) / 60, 2)
            failed += 1
            execution_log.append({
                'descriptor_type': descriptor_type,
                'att_index': att_index,
                'status': 'error',
                'duration_minutes': duration,
                'error': str(e)
            })
            safe_log(f"❌ Erro fatal em {descriptor_type}: {str(e)}")
        
        # Limpeza periódica e agressiva
        if i % 1 == 0:  # Limpeza após cada análise
            data_cache.clear_cache()
            
            # Limpeza completa de matplotlib
            plt.close('all')
            
            # Garbage collection agressivo
            import gc
            gc.collect()
            
            safe_log("🧹 Limpeza completa realizada")
    
    # Salvar log de execução da análise completa
    final_report = {
        'experiment_info': {
            'experiment_type': 'cluster_analysis_multiple_descriptors',
            'timestamp': datetime.datetime.now().isoformat(),
            'total_descriptors': len(descriptor_types),
            'successful': successful,
            'failed': failed,
            'experiment_directory': experiment_dir,
            'att_index_used': att_index
        },
        'descriptor_types_analyzed': descriptor_types,
        'execution_log': execution_log,
        'summary': {
            'total_time_minutes': sum(log.get('duration_minutes', 0) for log in execution_log),
            'success_rate': f"{(successful / len(descriptor_types) * 100):.1f}%" if descriptor_types else "0%"
        }
    }
    
    report_file = os.path.join(experiment_dir, 'cluster_analysis_report.json')
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    safe_log(f"🏁 Análise completa finalizada: {successful} sucessos, {failed} falhas")
    safe_log(f"📄 Relatório completo salvo em: {report_file}")
    
    # Mostrar resumo dos resultados
    safe_log("\n" + "="*60)
    safe_log("RESUMO DA ANÁLISE DE CLUSTERS")
    safe_log("="*60)
    for log in execution_log:
        status_emoji = "✅" if log['status'] == 'success' else "❌"
        safe_log(f"{status_emoji} {log['descriptor_type']}: {log['status']} ({log['duration_minutes']} min)")
    safe_log("="*60)
    
    return successful, failed

if __name__ == "__main__":
    import sys
    
    # Experimentos Explicações
    # if len(sys.argv) > 1 and sys.argv[1] == "--fast":
    #     safe_log("⚡ Modo rápido ativado")
    #     # Testar apenas um modelo
    #     test_single_model()
    # else:
    #     compatible = quick_compatibility_check()        
    #     if len(compatible) > 0:
    #         safe_log("🚀 Iniciando análise dos modelos compatíveis...")
    #         run_all_large_models()
    #     else:
    #         safe_log("❌ Nenhum modelo compatível encontrado!")

    # Experimentos Clusters
    if len(sys.argv) > 1 and sys.argv[1] == "--all-clusters":
        safe_log("🔬 Modo análise completa de clusters ativado")
        run_all_cluster_analysis()
    else:
        # Análise de cluster individual (padrão)
        run_cluster_analysis()