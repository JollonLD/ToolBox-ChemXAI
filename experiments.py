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
        input_dim = next(iter(train_loader))[0].shape[1]
        output_dim = 1
        update_json_log(f"Model dimensions: input={input_dim}, output={output_dim}", 35)
        
        # For the specific model, we don't need to specify layers as they'll be loaded from the file
        model = MLP(input_dim, output_dim, layers=[128, 64, 32], device=device)  # Default layers, will be overwritten
        update_json_log("MLP model instance created", 40)
        
        # Load the specific model
        model_path = os.path.join(os.getcwd(), 'models', f'{model_name}.pth')
        update_json_log(f"Loading model from: {model_path}", 45)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device) 
        update_json_log(f"Model loaded and moved to {device}", 50)
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        update_json_log(error_msg)
        raise

    # 4. Prepare data for explanation
    update_json_log("Preparing data for explanation...", 55)
    try:
        X_train = []
        y_train = []
        for i, (xb, yb) in enumerate(tqdm(train_loader, desc="Processing train data")):
            X_train.append(xb)
            y_train.append(yb)
            if i == 0:
                update_json_log(f"First train batch - X shape: {xb.shape}, Y shape: {yb.shape}")
        
        X_train = torch.cat(X_train, dim=0)[:500]  # Limit to 500 samples for explanation
        y_train = torch.cat(y_train, dim=0)[:500]
        update_json_log(f"Processed train data - X shape: {X_train.shape}, Y shape: {y_train.shape}", 60)

        X_test = []
        y_test = []
        for i, (xb, yb) in enumerate(tqdm(test_loader, desc="Processing test data")):
            X_test.append(xb)
            y_test.append(yb)
            if i == 0:
                update_json_log(f"First test batch - X shape: {xb.shape}, Y shape: {yb.shape}")
        
        X_test = torch.cat(X_test, dim=0)[:100]  # Limit to 100 samples for explanation
        y_test = torch.cat(y_test, dim=0)[:100]
        update_json_log(f"Processed test data - X shape: {X_test.shape}, Y shape: {y_test.shape}", 65)

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
            print(f"Running for model: {fname} (descriptor_type={descriptor_type}, att_index={att_index})")
            run_specific_model_with_explanations(att_index=att_index, descriptor_type=descriptor_type)

if __name__ == "__main__":
    run_all_large_models()
    # run_cluster_fidelity_experiment()