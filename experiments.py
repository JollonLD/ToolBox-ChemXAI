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

def run_specific_model_with_explanations(att_index = 0, descriptor_type = 'Physicochemical'):
    
    """
    Run SHAP and LIME explanations for a specific model: mlp_qm9_Physicochemical_att0.pth
    This function is optimized for nohup execution with real-time progress monitoring.
    """
    # Create timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define model path
    model_name = f'Large/mlp_qm9_{descriptor_type}_att{att_index}'
    
    # Setup logging
    print(f"[{timestamp}] Starting analysis for model: {model_name}")
    
    # 1. Create experiment folder
    print(f"[{timestamp}] Creating experiment directory...")
    base_dir = "experiments"
    dirname = os.getcwd()
    exp_dir = os.path.join(dirname, base_dir)
    os.makedirs(exp_dir, exist_ok=True)
    
    existing = [int(d.split("_")[-1]) for d in os.listdir(exp_dir) if d.startswith("experiment_") and d.split("_")[-1].isdigit()]
    next_id = max(existing) + 1 if existing else 1
    experiment_id = f"{next_id:03d}"
    experiment_dir = os.path.join(exp_dir, f"experiment_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)

    # NOVO: criar subpasta do modelo dentro do experimento
    model_subdir_name = f"{descriptor_type}_att{att_index}"
    model_dir = os.path.join(experiment_dir, model_subdir_name)
    os.makedirs(model_dir, exist_ok=True)

    log_file = os.path.join(model_dir, f"specific_model_analysis.txt")
    json_log_file = os.path.join(model_dir, f"specific_model_analysis.json")
    print(f"[{timestamp}] Experiment ID: {experiment_id}")
    print(f"[{timestamp}] Experiment directory: {experiment_dir}")
    print(f"[{timestamp}] Model directory: {model_dir}")

    # Initialize JSON log dictionary
    json_log = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "model": model_name,
        "parameters": {
            "att_index": att_index,
            "descriptor_type": descriptor_type
        },
        "progress": [],
        "results": {
            "shap": {},
            "lime": {}
        }
    }
    
    # Update JSON log and save
    def update_json_log(progress_msg, progress_pct=None, result_section=None, result_data=None):
        # Update progress
        progress_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "message": progress_msg
        }
        if progress_pct is not None:
            progress_entry["progress_percent"] = progress_pct
            
        json_log["progress"].append(progress_entry)
        
        # Update results if provided
        if result_section and result_data:
            json_log["results"][result_section] = result_data
            
        # Save to file
        with open(json_log_file, 'w') as f:
            json.dump(json_log, f, indent=2)
            
        # Print to console for real-time monitoring
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {progress_msg}")
        sys.stdout.flush()  # Ensure output is flushed for real-time monitoring with nohup

    update_json_log("Experiment started")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    update_json_log(f"Using device: {device}")
    
    log_lines = []
    log_lines.append(f"Device: {device}\n")
    log_lines.append(f"Analysis for model: {model_name}\n")
    log_lines.append(f"Experiment directory: {experiment_dir}\n")

    # 2. Loading Data
    update_json_log("Loading data...", 5)
    try:
        qm9 = qm9_tabular()
        update_json_log("QM9 tabular instance created", 10)
        
        loaders = qm9.get_paired_dataloaders(
            att_index=att_index,
            batch_size=32,  # Default batch size
            descriptor_type=descriptor_type,
            n_noise=0,
            add_noise=False
        )
        update_json_log("Dataloaders created", 15)
        train_loader, val_loader, test_loader = loaders
        update_json_log(f"Data loaded - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}", 20)
        
        log_lines.append("Data loaded.\n")
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        update_json_log(error_msg)
        raise

    # 3. Loading the specific model
    update_json_log("Loading specific model...", 30)
    try:
        # CORREÇÃO: Primeiro carregar os dados para obter as dimensões corretas
        X_sample, _, _ = qm9.compute_descriptors(
            descriptor_type=descriptor_type, 
            att_index=att_index, 
            list_mols=[]
        )
        
        # Usar as dimensões reais dos dados
        input_dim = X_sample.shape[1]
        output_dim = 1
        update_json_log(f"Model dimensions from data: input={input_dim}, output={output_dim}", 35)
        
        # Criar o modelo com as dimensões corretas
        model = MLP(input_dim, output_dim, layers=[128, 64, 32], device=device)
        update_json_log("MLP model instance created", 40)
        
        # Load the specific model
        model_path = os.path.join(os.getcwd(), 'models', f'{model_name}.pth')
        update_json_log(f"Loading model from: {model_path}", 45)
        
        # Verificar se o arquivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carregar o estado do modelo com verificação de compatibilidade
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                # Modelo incompatível - tentar reconstruir com dimensões corretas do checkpoint
                update_json_log(f"Incompatibilidade detectada: {str(e)}")
                
                # Extrair dimensões do checkpoint
                first_layer_weight = state_dict['layers.0.weight']
                checkpoint_input_dim = first_layer_weight.shape[1]
                
                update_json_log(f"Checkpoint input dim: {checkpoint_input_dim}, Data input dim: {input_dim}")
                
                if checkpoint_input_dim != input_dim:
                    error_msg = (f"Incompatibilidade irreconciliável: "
                               f"modelo foi treinado com {checkpoint_input_dim} features, "
                               f"mas dados atuais têm {input_dim} features. "
                               f"Verifique se o descriptor_type está correto.")
                    update_json_log(error_msg)
                    raise ValueError(error_msg)
            else:
                raise e
        
        model.eval()
        model.to(device) 
        update_json_log(f"Model loaded and moved to {device}", 50)
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        update_json_log(error_msg)
        raise

    # 4. Prepare data for explanation (usar os mesmos dados já carregados)
    update_json_log("Preparing data for explanation...", 55)
    try:
        # Carregar dados completos para explicação
        X_all, Y_all, _ = qm9.compute_descriptors(
            descriptor_type=descriptor_type, 
            att_index=att_index, 
            list_mols=[]
        )
        
        # Dividir em train/test usando a mesma proporção do treinamento
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, Y_all, test_size=0.2, random_state=42
        )
        
        # Limitar amostras para explicação
        X_train = torch.from_numpy(X_train[:500]).float()
        y_train = torch.from_numpy(y_train[:500]).float()
        X_test = torch.from_numpy(X_test[:100]).float()
        y_test = torch.from_numpy(y_test[:100]).float()
        
        update_json_log(f"Prepared data - Train: {X_train.shape}, Test: {X_test.shape}", 60)

        update_json_log("Making predictions...", 70)
        with torch.no_grad():
            y_pred = model(X_test.to(device)).cpu().numpy()
        update_json_log(f"Predictions made - shape: {y_pred.shape}", 75)

        log_lines.append(f"Test set: {X_test.shape[0]} samples\n")
    except Exception as e:
        error_msg = f"Error preparing data: {str(e)}"
        update_json_log(error_msg)
        raise

    # 5. SHAP Explanation
    update_json_log("Starting SHAP explanations...", 80)
    try:
        update_json_log("Creating SHAP explainer...")
        shap_explainer = Shap(model, X_train, X_test, device)
        update_json_log("SHAP explainer created", 82)
        
        update_json_log("Generating global SHAP explanation...")
        shap_explanation = shap_explainer.explain_global()
        update_json_log(f"SHAP explanation generated - shape: {np.array(shap_explanation).shape}", 85)
        
        # Generate local explanations for first few samples
        local_shap_explanations = {}
        for i in tqdm(range(min(5, X_test.shape[0])), desc="Generating SHAP local explanations"):
            local_explanation = shap_explainer.explain_local(index=i)
            local_shap_explanations[f"sample_{i}"] = local_explanation
            update_json_log(f"Generated SHAP local explanation for sample {i}")
        
        # Save SHAP explanations to JSON
        json_log["results"]["shap"]["global_explanation"] = shap_explanation
        json_log["results"]["shap"]["local_explanations"] = local_shap_explanations
        
        # Save to separate JSON files for easy access
        shap_global_path = os.path.join(model_dir, "shap_global_explanation.json")
        with open(shap_global_path, 'w') as f:
            json.dump({"explanation": shap_explanation}, f, indent=2)
            
        shap_local_path = os.path.join(model_dir, "shap_local_explanations.json")
        with open(shap_local_path, 'w') as f:
            json.dump({"explanations": local_shap_explanations}, f, indent=2)
            
        update_json_log(f"SHAP explanations saved to {model_dir}", 88)
        log_lines.append("SHAP explanation generated.\n")
    except Exception as e:
        error_msg = f"Error during SHAP explanation: {str(e)}"
        update_json_log(error_msg)
        json_log["results"]["shap"]["error"] = str(e)

    # 6. LIME Explanation
    update_json_log("Starting LIME explanations...", 90)
    try:
        update_json_log("Creating LIME explainer...")
        lime_explainer = LIME(model, X_train, X_test, device)
        update_json_log("LIME explainer created", 92)
        
        update_json_log("Generating local LIME explanations...")
        
        # Generate multiple local explanations and store them
        lime_explanations = {}
        for i in tqdm(range(min(5, X_test.shape[0])), desc="Generating LIME explanations"):
            lime_explanation = lime_explainer.explain_local(index=i)
            lime_explanations[f"sample_{i}"] = lime_explanation
            update_json_log(f"Generated LIME explanation for sample {i}")
        
        # For backward compatibility, keep the first explanation as the main one
        lime_explanation = lime_explainer.explain_local(index=0)
        update_json_log(f"LIME explanations generated - shape: {np.array(lime_explanation).shape}", 95)
        
        # Save LIME explanations to JSON
        json_log["results"]["lime"]["local_explanations"] = lime_explanations
        
        # Save to a separate JSON file for easy access
        lime_json_path = os.path.join(model_dir, "lime_explanation.json")
        with open(lime_json_path, 'w') as f:
            json.dump({"explanations": lime_explanations}, f, indent=2)
            
        update_json_log(f"LIME explanations saved to {lime_json_path}", 97)
        log_lines.append("LIME explanation generated.\n")
    except Exception as e:
        error_msg = f"Error during LIME explanation: {str(e)}"
        update_json_log(error_msg)
        json_log["results"]["lime"]["error"] = str(e)

    # 7. Calculate metrics
    update_json_log("Collecting metrics...", 98)

    # SHAP Metrics
    try:
        update_json_log("Calculating SHAP metrics...")
        analyzer_shap = TabularAnalyzer(
            model=model,
            explainer=shap_explainer,
            explanation=shap_explanation,
            data=X_test,
            y_true=y_test.numpy(),
            y_pred=y_pred,
            device=device
        )
        
        fidelity_shap = analyzer_shap.get_metrics()
        update_json_log(f"SHAP metrics calculated: {fidelity_shap}")
        
        # Save SHAP metrics to JSON
        json_log["results"]["shap"]["metrics"] = {
            "fidelity_positive": float(fidelity_shap[0]),
            "fidelity_negative": float(fidelity_shap[1])
        }
        
        log_lines.append(f"SHAP Positive Fidelity: {fidelity_shap[0]}\n")
        log_lines.append(f"SHAP Negative Fidelity: {fidelity_shap[1]}\n")
    except Exception as e:
        error_msg = f"Error calculating SHAP metrics: {str(e)}"
        update_json_log(error_msg)
        json_log["results"]["shap"]["metrics_error"] = str(e)

    # LIME Metrics
    try:
        update_json_log("Calculating LIME metrics...")
        analyzer_lime = TabularAnalyzer(
            model=model,
            explainer=lime_explainer,
            explanation=lime_explanation,
            data=X_test,
            y_true=y_test.numpy(),
            y_pred=y_pred,
            device=device
        )
        
        fidelity_lime = analyzer_lime.get_metrics()
        update_json_log(f"LIME metrics calculated: {fidelity_lime}")
        
        # Save LIME metrics to JSON
        json_log["results"]["lime"]["metrics"] = {
            "fidelity_positive": float(fidelity_lime[0]),
            "fidelity_negative": float(fidelity_lime[1])
        }
        
        log_lines.append(f"LIME Positive Fidelity: {fidelity_lime[0]}\n")
        log_lines.append(f"LIME Negative Fidelity: {fidelity_lime[1]}\n")
    except Exception as e:
        error_msg = f"Error calculating LIME metrics: {str(e)}"
        update_json_log(error_msg)
        json_log["results"]["lime"]["metrics_error"] = str(e)

    # 8. Generate visualizations
    update_json_log("Generating plots...", 99)

    # SHAP Plots
    try:
        update_json_log("Generating SHAP radar plot...")
        feature_names = qm9.get_descriptor_names(descriptor_type)
        radar_plot(np.array(shap_explanation), feature_names=feature_names, title=f"SHAP - Radar Plot - {model_name}")
        radar_path = os.path.join(model_dir, "shap_radar.png")
        plt.savefig(radar_path)
        plt.close()
        update_json_log(f"SHAP radar plot saved to {radar_path}")

        update_json_log("Generating SHAP bar plot...")
        horizontal_bar_plot(np.array(shap_explanation), feature_names=feature_names, title=f"SHAP - Feature Importance - {model_name}",
                            save_path=model_dir, filename="shap_bar.png")

        json_log["results"]["shap"]["plots"] = {
            "radar_plot": radar_path,
            "bar_plot": os.path.join(model_dir, "shap_bar.png")
        }
    except Exception as e:
        error_msg = f"Error generating SHAP plots: {str(e)}"
        update_json_log(error_msg)
        json_log["results"]["shap"]["plot_error"] = str(e)

    # LIME Plots
    try:
        update_json_log("Generating LIME radar plot...")
        feature_names = qm9.get_descriptor_names(descriptor_type)
        radar_plot(np.array(lime_explanation), feature_names=feature_names, title=f"LIME - Radar Plot - {model_name}")
        radar_path = os.path.join(model_dir, "lime_radar.png")
        plt.savefig(radar_path)
        plt.close()
        update_json_log(f"LIME radar plot saved to {radar_path}")

        update_json_log("Generating LIME bar plot...")
        horizontal_bar_plot(np.array(lime_explanation), feature_names=feature_names, title=f"LIME - Feature Importance - {model_name}",
                            save_path=model_dir, filename="lime_bar.png")

        json_log["results"]["lime"]["plots"] = {
            "radar_plot": radar_path,
            "bar_plot": os.path.join(model_dir, "lime_bar.png")
        }
    except Exception as e:
        error_msg = f"Error generating LIME plots: {str(e)}"
        update_json_log(error_msg)
        json_log["results"]["lime"]["plot_error"] = str(e)

    # 9. Save logs and finish
    try:
        update_json_log("Saving log files...", 100)
        with open(log_file, 'w') as f:
            for line in log_lines:
                f.write(line)
        update_json_log(f"Log saved to {log_file}")
        
        # Final JSON log update
        json_log["completion_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_log["status"] = "completed"
        with open(json_log_file, 'w') as f:
            json.dump(json_log, f, indent=2)
            
        update_json_log(f"JSON log saved to {json_log_file}")
        update_json_log(f"Results and plots saved to {model_dir}")
        update_json_log("Analysis completed!")
    except Exception as e:
        error_msg = f"Error saving logs: {str(e)}"
        update_json_log(error_msg)

def run_all_large_models():
    """
    Run explanations for all models in models/Large matching mlp_qm9_{descriptor_type}_{att_index}.pth
    """
    models_dir = os.path.join(os.getcwd(), "models", "Large")
    if not os.path.exists(models_dir):
        print(f"Directory not found: {models_dir}")
        return

    # Primeiro, verificar compatibilidade de todos os modelos
    qm9 = qm9_tabular()
    compatibility_report = {}
    
    print("Verificando compatibilidade dos modelos...")
    
    for fname in os.listdir(models_dir):
        if fname.startswith("mlp_qm9_") and fname.endswith(".pth"):
            # Example fname: mlp_qm9_Physicochemical_att0.pth
            parts = fname.split("_")
            if len(parts) < 4 or not parts[-1].startswith("att"):
                continue
                
            descriptor_type = parts[2]
            att_index = parts[-1].replace("att", "").replace(".pth", "")
            
            try:
                att_index = int(att_index)
            except ValueError:
                continue
                
            # Obter dimensões esperadas dos dados
            try:
                X_sample, _, _ = qm9.compute_descriptors(
                    descriptor_type=descriptor_type, 
                    att_index=att_index, 
                    list_mols=list(range(10))  # Apenas algumas moléculas para teste
                )
                expected_dim = X_sample.shape[1]
                
                model_path = os.path.join(models_dir, fname)
                compatibility = check_model_compatibility(model_path, expected_dim, descriptor_type)
                compatibility_report[fname] = compatibility
                
                status = "✓" if compatibility['compatible'] else "✗"
                print(f"{status} {fname}: {compatibility.get('error', 'Compatible')}")
                
            except Exception as e:
                compatibility_report[fname] = {
                    'compatible': False,
                    'error': f"Erro ao verificar dados: {str(e)}"
                }
                print(f"✗ {fname}: Erro ao verificar dados: {str(e)}")
    
    # Executar apenas modelos compatíveis
    compatible_models = [fname for fname, report in compatibility_report.items() 
                        if report['compatible']]
    
    print(f"\nModelos compatíveis: {len(compatible_models)}/{len(compatibility_report)}")
    print("Iniciando análise dos modelos compatíveis...")
    
    for fname in compatible_models:
        parts = fname.split("_")
        descriptor_type = parts[2]
        att_index = int(parts[-1].replace("att", "").replace(".pth", ""))
        
        print(f"Running for model: {fname} (descriptor_type={descriptor_type}, att_index={att_index})")
        try:
            run_specific_model_with_explanations(att_index=att_index, descriptor_type=descriptor_type)
        except Exception as e:
            print(f"Erro ao processar {fname}: {str(e)}")
            continue
    
    # Salvar relatório de compatibilidade
    report_path = os.path.join(os.getcwd(), "experiments", "model_compatibility_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(compatibility_report, f, indent=2)
    print(f"Relatório de compatibilidade salvo em: {report_path}")


def check_model_compatibility(model_path, expected_input_dim, descriptor_type):
    """
    Verifica se o modelo salvo é compatível com as dimensões dos dados atuais.
    
    Parameters:
    -----------
    model_path : str
        Caminho para o arquivo do modelo
    expected_input_dim : int
        Dimensão de entrada esperada
    descriptor_type : str
        Tipo de descritor usado
        
    Returns:
    --------
    dict: Informações sobre compatibilidade
    """
    if not os.path.exists(model_path):
        return {
            'compatible': False,
            'error': f"Arquivo não encontrado: {model_path}"
        }
    
    try:
        # Carregar apenas o estado do modelo para verificação
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Extrair dimensões da primeira camada
        first_layer_weight = state_dict['layers.0.weight']
        model_input_dim = first_layer_weight.shape[1]
        model_output_dim = list(state_dict.keys())[-1]  # Última camada
        
        compatible = (model_input_dim == expected_input_dim)
        
        return {
            'compatible': compatible,
            'model_input_dim': model_input_dim,
            'expected_input_dim': expected_input_dim,
            'descriptor_type': descriptor_type,
            'model_layers': len([k for k in state_dict.keys() if 'weight' in k]),
            'error': None if compatible else f"Dimensão incompatível: modelo={model_input_dim}, dados={expected_input_dim}"
        }
        
    except Exception as e:
        return {
            'compatible': False,
            'error': f"Erro ao verificar modelo: {str(e)}"
        }

if __name__ == "__main__":
    run_all_large_models()
    run_cluster_fidelity_experiment()