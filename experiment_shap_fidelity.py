# Experimento: Avaliar fidelidade dos modelos SHAP por descriptor
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from chemxai.data import qm9_tabular
from chemxai.models import MLP
from chemxai.explainers import Shap, LIME
from chemxai.plots import radar_plot, horizontal_bar_plot


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

def optimize_for_gpu():
    """
    Configura otimizações para GPU e controle de memória
    """
    if torch.cuda.is_available():
        # Configurações avançadas de GPU para SHAP
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Configurações de memória para SHAP
        torch.cuda.empty_cache()
        
        # Configurar para permitir crescimento de memória
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)  # Usar 90% da memória disponível
        except:
            pass
        
        # Informações da GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_memory = torch.cuda.memory_reserved(0) / 1e9
        safe_log(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB total, {free_memory:.1f}GB livre)")
        
        # Configurar threads para otimização
        torch.set_num_threads(min(8, torch.get_num_threads()))
        
        return torch.device('cuda')
    else:
        safe_log("💻 Usando CPU (GPU não disponível)")
        # Otimizações para CPU
        torch.set_num_threads(min(8, torch.get_num_threads()))
        return torch.device('cpu')

def create_experiment_directory():
    """
    Cria diretório único para o experimento
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(os.getcwd(), "experiments", f"shap_fidelity_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    safe_log(f"📁 Experimento SHAP Fidelity criado: {experiment_dir}")
    return experiment_dir

def parse_model_filename(filename):
    """
    Função auxiliar para fazer parse do nome do modelo
    
    Formatos suportados:
    - Novo: mlp_qm9_{descriptor_type}_{att_index}_large.pth
    - Antigo com large: mlp_qm9_{descriptor_type}_att{att_index}_large.pth
    - Antigo: mlp_qm9_{descriptor_type}_att{att_index}.pth (para compatibilidade)
    """
    try:
        if filename.endswith("_large.pth"):
            # Remover prefixo e sufixo
            name_part = filename.replace("mlp_qm9_", "").replace("_large.pth", "")
            
            # Verificar se é formato antigo com "att" (ex: CM_att12_large.pth)
            if "_att" in name_part:
                # Formato: {descriptor_type}_att{att_index}_large.pth
                parts = name_part.split("_att")
                if len(parts) == 2:
                    descriptor_type = parts[0]
                    att_index = int(parts[1])
                    return descriptor_type, att_index, "legacy_large"
            else:
                # Novo formato: mlp_qm9_{descriptor_type}_{att_index}_large.pth
                parts = name_part.rsplit("_", 1)
                if len(parts) == 2:
                    descriptor_type = parts[0]
                    att_index = int(parts[1])
                    return descriptor_type, att_index, "large"
        
        elif filename.startswith("mlp_qm9_") and filename.endswith(".pth"):
            # Formato antigo: mlp_qm9_{descriptor_type}_att{att_index}.pth
            parts = filename.split("_")
            if len(parts) >= 4:
                descriptor_type = parts[2]
                att_part = parts[-1].replace("att", "").replace(".pth", "")
                att_index = int(att_part)
                return descriptor_type, att_index, "legacy"
    
    except Exception:
        pass
    
    return None, None, None

def load_model_optimized(descriptor_type, att_index, device):
    """
    Carrega modelo de forma otimizada com cache e verificações
    """
    # Usar formato real dos modelos existentes
    model_path = os.path.join(os.getcwd(), 'models', f'mlp_qm9_{descriptor_type}_att{att_index}_large.pth')
    
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
    
    # Mover para GPU de forma otimizada
    if device.type == 'cuda':
        model = model.to(device)
        try:
            model = torch.jit.script(model)
            safe_log(f"🚀 Modelo carregado na GPU com JIT compilation")
        except:
            safe_log(f"🚀 Modelo carregado na GPU (JIT não aplicado)")
        torch.cuda.empty_cache()
    else:
        safe_log(f"📱 Modelo carregado na CPU")
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )
    
    # Limitar amostras para não sobrecarregar memória
    max_samples = 1000 if device.type == 'cuda' else 500
    X_train = X_train[:max_samples]
    X_test = X_test[:min(200, len(X_test))]
    y_train = y_train[:max_samples]
    y_test = y_test[:min(200, len(y_test))]
    
    # Converter para tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    
    # Mover para device de forma otimizada
    if device.type == 'cuda':
        # Verificar memória disponível antes de mover
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        tensor_memory = (X_train_tensor.numel() + X_test_tensor.numel()) * 4  # 4 bytes por float
        
        if tensor_memory > available_memory * 0.7:  # Usar no máximo 70% da memória
            safe_log("⚠️  Memória GPU limitada, reduzindo dados")
            reduction_factor = max(1, int(tensor_memory / (available_memory * 0.7)))
            X_train_tensor = X_train_tensor[:len(X_train_tensor)//reduction_factor]
            X_test_tensor = X_test_tensor[:len(X_test_tensor)//reduction_factor]
            y_train_tensor = y_train_tensor[:len(y_train_tensor)//reduction_factor]
            y_test_tensor = y_test_tensor[:len(y_test_tensor)//reduction_factor]
            safe_log(f"📉 Dados reduzidos por fator {reduction_factor}")
        
        # Mover para GPU com pin_memory para melhor performance
        X_train_tensor = X_train_tensor.pin_memory().to(device, non_blocking=True)
        X_test_tensor = X_test_tensor.pin_memory().to(device, non_blocking=True)
        y_train_tensor = y_train_tensor.pin_memory().to(device, non_blocking=True)
        y_test_tensor = y_test_tensor.pin_memory().to(device, non_blocking=True)
        safe_log(f"🚀 Dados movidos para GPU com otimizações")
    else:
        X_train_tensor = X_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        safe_log(f"📱 Dados processados na CPU")
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def calculate_shap_fidelity(model, X_test, y_test, shap_values, device, top_k=None):
    """
    Calcula a fidelidade das explicações SHAP
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Targets verdadeiros
        shap_values: Valores SHAP calculados
        device: Dispositivo (CPU/GPU)
        top_k: Número de features mais importantes para análise (None = todas)
    
    Returns:
        dict: Métricas de fidelidade
    """
    safe_log("🔍 Calculando fidelidade SHAP...")
    
    # Predições originais
    with torch.no_grad():
        original_preds = model(X_test).cpu().numpy().flatten()
    
    # Se top_k não especificado, usar todas as features
    if top_k is None:
        top_k = X_test.shape[1]
    
    # Identificar top_k features mais importantes
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:]
    
    # Teste de remoção de features (Faithfulness)
    X_test_masked = X_test.clone()
    X_test_masked[:, top_indices] = 0  # Mascarar features importantes
    
    with torch.no_grad():
        masked_preds = model(X_test_masked).cpu().numpy().flatten()
    
    # Teste de permutação de features
    X_test_permuted = X_test.clone()
    for idx in top_indices:
        X_test_permuted[:, idx] = X_test_permuted[torch.randperm(X_test.shape[0]), idx]
    
    with torch.no_grad():
        permuted_preds = model(X_test_permuted).cpu().numpy().flatten()
    
    # Calcular métricas de fidelidade
    fidelity_metrics = {
        'removal_fidelity': {
            'mse_change': float(mean_squared_error(original_preds, masked_preds)),
            'mae_change': float(mean_absolute_error(original_preds, masked_preds)),
            'prediction_correlation': float(np.corrcoef(original_preds, masked_preds)[0, 1])
        },
        'permutation_fidelity': {
            'mse_change': float(mean_squared_error(original_preds, permuted_preds)),
            'mae_change': float(mean_absolute_error(original_preds, permuted_preds)),
            'prediction_correlation': float(np.corrcoef(original_preds, permuted_preds)[0, 1])
        },
        'explanation_quality': {
            'top_k_features': int(top_k),
            'mean_abs_shap': float(np.mean(mean_abs_shap[top_indices])),
            'std_abs_shap': float(np.std(mean_abs_shap[top_indices])),
            'shap_sparsity': float(np.sum(mean_abs_shap > 0.001) / len(mean_abs_shap))
        },
        'model_performance': {
            'original_mse': float(mean_squared_error(y_test.cpu().numpy(), original_preds)),
            'original_mae': float(mean_absolute_error(y_test.cpu().numpy(), original_preds)),
            'original_r2': float(r2_score(y_test.cpu().numpy(), original_preds))
        }
    }
    
    safe_log(f"✅ Fidelidade calculada - Remoção MSE: {fidelity_metrics['removal_fidelity']['mse_change']:.4f}")
    return fidelity_metrics

def run_shap_fidelity_analysis(descriptor_type, att_index, experiment_dir):
    """
    Executa análise completa de fidelidade SHAP para um modelo específico
    """
    device = optimize_for_gpu()
    
    safe_log(f"🚀 Iniciando análise de fidelidade SHAP para {descriptor_type}_att{att_index}")
    
    # Criar subdiretório para este modelo
    model_dir = os.path.join(experiment_dir, f"fidelity_{descriptor_type}_att{att_index}")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # 1. Carregar modelo
        safe_log("📦 Carregando modelo...")
        model, input_dim = load_model_optimized(descriptor_type, att_index, device)
        
        # 2. Preparar dados
        safe_log("📊 Preparando dados...")
        X_train, X_test, y_train, y_test = prepare_data_optimized(descriptor_type, att_index, input_dim, device)
        
        # 3. Calcular SHAP
        safe_log("🧠 Calculando SHAP...")
        explainer = Shap(model, X_train[:100], X_test[:50], device)
        shap_global = explainer.explain_global()
        shap_local = explainer.explain_local(index=0)
        
        # Para fidelidade, precisamos de valores SHAP para múltiplas amostras
        shap_values_batch = []
        batch_size = 10
        n_samples = min(50, len(X_test))
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_shap = explainer.explain_local_batch(indices=list(range(i, end_idx)))
            shap_values_batch.extend(batch_shap)
        
        shap_values_array = np.array(shap_values_batch)
        
        # 4. Calcular fidelidade para diferentes valores de top_k
        fidelity_results = {}
        top_k_values = [5, 10, 15, 20, input_dim//4, input_dim//2]
        
        for top_k in top_k_values:
            if top_k <= input_dim:
                safe_log(f"📈 Calculando fidelidade para top_{top_k} features...")
                fidelity_metrics = calculate_shap_fidelity(
                    model, X_test[:n_samples], y_test[:n_samples], 
                    shap_values_array, device, top_k
                )
                fidelity_results[f'top_{top_k}'] = fidelity_metrics
        
        # 5. Obter nomes das features
        try:
            qm9 = data_cache.get_qm9()
            feature_names = qm9.get_descriptor_names(descriptor_type)
        except:
            feature_names = [f"feature_{i}" for i in range(input_dim)]
        
        # 6. Salvar resultados
        complete_results = {
            'model_info': {
                'descriptor_type': descriptor_type,
                'att_index': att_index,
                'input_dim': input_dim,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_fidelity_samples': n_samples
            },
            'shap_explanations': {
                'global_importance': shap_global.tolist() if isinstance(shap_global, np.ndarray) else shap_global,
                'local_explanation_sample': shap_local.tolist() if isinstance(shap_local, np.ndarray) else shap_local,
                'batch_explanations_shape': shap_values_array.shape,
                'feature_names': feature_names
            },
            'fidelity_analysis': fidelity_results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Salvar JSON
        results_file = os.path.join(model_dir, 'shap_fidelity_results.json')
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        # 7. Gerar plots de fidelidade
        generate_fidelity_plots(fidelity_results, model_dir, descriptor_type, att_index)
        
        # 8. Gerar plots SHAP (top 15 features)
        generate_shap_plots(shap_global, feature_names, model_dir, descriptor_type, att_index)
        
        safe_log(f"✅ Análise de fidelidade concluída e salva em: {model_dir}")
        return True
        
    except Exception as e:
        safe_log(f"❌ Erro na análise de fidelidade: {str(e)}")
        return False
    
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def generate_fidelity_plots(fidelity_results, model_dir, descriptor_type, att_index):
    """
    Gera plots para visualizar métricas de fidelidade
    """
    try:
        safe_log("📊 Gerando plots de fidelidade...")
        
        # Extrair dados para plots
        top_k_values = []
        removal_mse = []
        removal_corr = []
        permutation_mse = []
        permutation_corr = []
        
        for key, metrics in fidelity_results.items():
            if key.startswith('top_'):
                top_k = int(key.replace('top_', ''))
                top_k_values.append(top_k)
                removal_mse.append(metrics['removal_fidelity']['mse_change'])
                removal_corr.append(metrics['removal_fidelity']['prediction_correlation'])
                permutation_mse.append(metrics['permutation_fidelity']['mse_change'])
                permutation_corr.append(metrics['permutation_fidelity']['prediction_correlation'])
        
        # Plot 1: MSE Change vs Top-K Features
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(top_k_values, removal_mse, 'bo-', label='Removal')
        plt.plot(top_k_values, permutation_mse, 'ro-', label='Permutation')
        plt.xlabel('Top-K Features')
        plt.ylabel('MSE Change')
        plt.title('Fidelity: MSE Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Correlation vs Top-K Features
        plt.subplot(2, 2, 2)
        plt.plot(top_k_values, removal_corr, 'bo-', label='Removal')
        plt.plot(top_k_values, permutation_corr, 'ro-', label='Permutation')
        plt.xlabel('Top-K Features')
        plt.ylabel('Prediction Correlation')
        plt.title('Fidelity: Prediction Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Bar plot comparativo para top_10
        if 'top_10' in fidelity_results:
            plt.subplot(2, 2, 3)
            metrics_10 = fidelity_results['top_10']
            categories = ['MSE Change', 'MAE Change', 'Correlation']
            removal_vals = [
                metrics_10['removal_fidelity']['mse_change'],
                metrics_10['removal_fidelity']['mae_change'],
                metrics_10['removal_fidelity']['prediction_correlation']
            ]
            permutation_vals = [
                metrics_10['permutation_fidelity']['mse_change'],
                metrics_10['permutation_fidelity']['mae_change'],
                metrics_10['permutation_fidelity']['prediction_correlation']
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, removal_vals, width, label='Removal', alpha=0.8)
            plt.bar(x + width/2, permutation_vals, width, label='Permutation', alpha=0.8)
            plt.xlabel('Metrics')
            plt.ylabel('Value')
            plt.title('Fidelity Metrics Comparison (Top-10)')
            plt.xticks(x, categories)
            plt.legend()
        
        # Plot 4: SHAP quality metrics
        if 'top_10' in fidelity_results:
            plt.subplot(2, 2, 4)
            quality_metrics = fidelity_results['top_10']['explanation_quality']
            quality_names = ['Mean Abs SHAP', 'Std Abs SHAP', 'Sparsity']
            quality_values = [
                quality_metrics['mean_abs_shap'],
                quality_metrics['std_abs_shap'],
                quality_metrics['shap_sparsity']
            ]
            
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            plt.bar(quality_names, quality_values, color=colors, alpha=0.8)
            plt.xlabel('Quality Metrics')
            plt.ylabel('Value')
            plt.title('SHAP Explanation Quality')
            plt.xticks(rotation=45)
        
        plt.suptitle(f'SHAP Fidelity Analysis - {descriptor_type} att{att_index}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'fidelity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        safe_log("✅ Plots de fidelidade gerados")
        
    except Exception as e:
        safe_log(f"⚠️ Erro ao gerar plots de fidelidade: {e}")

def generate_shap_plots(shap_global, feature_names, model_dir, descriptor_type, att_index):
    """
    Gera plots SHAP tradicionais
    """
    try:
        safe_log("📈 Gerando plots SHAP...")
        
        # Top 15 features
        shap_array = np.array(shap_global)
        top_indices = np.argsort(np.abs(shap_array))[-15:]
        top_values = shap_array[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Bar plot horizontal
        horizontal_bar_plot(
            top_values, 
            top_names, 
            title=f"SHAP Global Top 15 - {descriptor_type} att{att_index}",
            save_path=model_dir,
            filename='shap_global_top15.png'
        )
        
        # Radar plot
        plt.figure(figsize=(10, 10))
        radar_plot(
            top_values, 
            top_names, 
            title=f"SHAP Radar Top 15 - {descriptor_type} att{att_index}"
        )
        plt.savefig(os.path.join(model_dir, 'shap_radar_top15.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        safe_log("✅ Plots SHAP gerados")
        
    except Exception as e:
        safe_log(f"⚠️ Erro ao gerar plots SHAP: {e}")

def run_all_shap_fidelity_analysis():
    """
    Executa análise de fidelidade SHAP para todos os modelos compatíveis
    """
    device = optimize_for_gpu()
    
    # Criar diretório do experimento
    experiment_dir = create_experiment_directory()
    
    # Procurar modelos compatíveis
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        safe_log(f"❌ Diretório não encontrado: {models_dir}")
        return

    safe_log("🔍 Procurando modelos para análise de fidelidade...")
    
    compatible_models = []
    for fname in os.listdir(models_dir):
        if fname.startswith("mlp_qm9_") and fname.endswith(".pth"):
            try:
                descriptor_type, att_index, model_type = parse_model_filename(fname)
                
                if descriptor_type is not None and att_index is not None:
                    # Verificar se arquivo existe e pode ser carregado
                    model_path = os.path.join(models_dir, fname)
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    if 'layers.0.weight' in state_dict:
                        input_dim = state_dict['layers.0.weight'].shape[1]
                        compatible_models.append((fname, descriptor_type, att_index, input_dim))
                        safe_log(f"✅ {fname}: {input_dim} features")
                        
            except Exception as e:
                safe_log(f"❌ {fname}: Erro - {str(e)}")
    
    if not compatible_models:
        safe_log("❌ Nenhum modelo compatível encontrado")
        return
    
    safe_log(f"📊 {len(compatible_models)} modelos compatíveis encontrados")
    
    successful = 0
    failed = 0
    execution_log = []
    
    for i, (fname, descriptor_type, att_index, input_dim) in enumerate(compatible_models):
        safe_log(f"📊 Progresso: {i+1}/{len(compatible_models)} - {descriptor_type}_att{att_index}")
        
        start_time = time.time()
        
        try:
            result = run_shap_fidelity_analysis(descriptor_type, att_index, experiment_dir)
            duration = round((time.time() - start_time) / 60, 2)
            
            if result:
                successful += 1
                execution_log.append({
                    'model': f"{descriptor_type}_att{att_index}",
                    'status': 'success',
                    'duration_minutes': duration
                })
                safe_log(f"✅ Sucesso - {descriptor_type}_att{att_index} ({duration} min)")
            else:
                failed += 1
                execution_log.append({
                    'model': f"{descriptor_type}_att{att_index}",
                    'status': 'failed',
                    'duration_minutes': duration
                })
                safe_log(f"❌ Falha - {descriptor_type}_att{att_index}")
                
        except KeyboardInterrupt:
            safe_log("🛑 Interrompido pelo usuário")
            break
        except Exception as e:
            duration = round((time.time() - start_time) / 60, 2)
            failed += 1
            execution_log.append({
                'model': f"{descriptor_type}_att{att_index}",
                'status': 'error',
                'duration_minutes': duration,
                'error': str(e)
            })
            safe_log(f"❌ Erro fatal em {descriptor_type}_att{att_index}: {str(e)}")
        
        # Limpeza periódica
        if i % 3 == 0:
            data_cache.clear_cache()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Salvar relatório final
    final_report = {
        'experiment_info': {
            'experiment_type': 'shap_fidelity_analysis',
            'timestamp': datetime.datetime.now().isoformat(),
            'total_models': len(compatible_models),
            'successful': successful,
            'failed': failed,
            'experiment_directory': experiment_dir
        },
        'execution_log': execution_log
    }
    
    report_file = os.path.join(experiment_dir, 'shap_fidelity_report.json')
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    safe_log(f"🏁 Análise de fidelidade finalizada: {successful} sucessos, {failed} falhas")
    safe_log(f"📄 Relatório salvo: {report_file}")
    
    return successful, failed

# Função de teste para um modelo específico
def test_shap_fidelity():
    """
    Testa análise de fidelidade para um modelo específico
    """
    experiment_dir = create_experiment_directory()
    result = run_shap_fidelity_analysis('Physicochemical', 0, experiment_dir)
    safe_log(f"Teste de fidelidade concluído: {result}")

if __name__ == "__main__":
    # safe_log("🚀 Iniciando análise completa de fidelidade SHAP...")
    # run_all_shap_fidelity_analysis()

    test_shap_fidelity()