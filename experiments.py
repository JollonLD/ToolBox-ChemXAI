import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from chemxai.data import qm9_tabular
from chemxai.models import MLP
from chemxai.train import train_mlp_qm9
from chemxai.explainers import Shap, LIME
from chemxai.evaluate import TabularAnalyzer
from chemxai.plots import radar_plot, horizontal_bar_plot

def experiment_MLP_SHAP_LIME_predict_Feature10(
    att_index=10,
    epochs=10,
    layers=[64, 32],
    learning_rate=1e-3,
    batch_size=32,
    n_noise=0,
    descriptor_type='Physicochemical',
    experiment_id=None
):
    print("Iniciando experimento...")
    
    # 1. Criar pasta do experimento
    print("Criando diretório do experimento...")
    base_dir = "experiments"
    dirname = os.getcwd()
    exp_dir = os.path.join(dirname, base_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f'Diretório criado: {exp_dir}')

    existing = [int(d.split("_")[-1]) for d in os.listdir(exp_dir) if d.startswith("experiment_") and d.split("_")[-1].isdigit()]
    next_id = max(existing) + 1 if existing else 1
    experiment_id = f"{next_id:03d}"
    experiment_dir = os.path.join(exp_dir, f"experiment_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, f"experiment_log_MLP_SHAP_LIME.txt")
    print(f"Experimento ID: {experiment_id}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_lines = []
    log_lines.append(f"Dispositivo: {device}\n")
    print(f'Device: {device}')
    log_lines.append(f"Experimento MLP QM9 - SHAP & LIME\n")
    log_lines.append(f"Parâmetros: att_index={att_index}, epochs={epochs}, layers={layers}, batch_size={batch_size}, n_noise={n_noise}, descriptor_type={descriptor_type}\n")
    log_lines.append(f"Pasta do experimento: {experiment_dir}\n")

    # 2. Dados
    print('Carregando Dados...')
    qm9 = qm9_tabular()
    print('Instância qm9_tabular criada')
    
    loaders = qm9.get_paired_dataloaders(
        att_index=att_index,
        batch_size=batch_size,
        descriptor_type=descriptor_type,
        n_noise=n_noise,
        add_noise=False
    )
    print('Dataloaders criados')
    train_loader, val_loader, test_loader = loaders
    print(f'Train loader batches: {len(train_loader)}')
    print(f'Test loader batches: {len(test_loader)}')

    log_lines.append("Dados carregados.\n")
    log_lines.append("Iniciando Treinamento.\n")

    # 3. Treinamento
    print('Iniciando Treinamento...')
    history = train_mlp_qm9(
        att_index=att_index,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=n_noise,
        descriptor_type=descriptor_type,
    )
    print('Treinamento concluído')
    log_lines.append("Histórico de treinamento (época, treino_loss, val_loss):\n")
    for epoch, train_loss, val_loss in history:
        log_lines.append(f"Época {epoch}: Loss Treino={train_loss:.4f} | Loss Validação={val_loss:.4f}\n")
    log_lines.append("Treinamento finalizado.\n")
    log_lines.append("Iniciando Explicações.\n")

    # 4. Carregar modelo treinado
    print('Carregando Modelo...')
    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = 1
    print(f'Input dim: {input_dim}, Output dim: {output_dim}')
    
    model = MLP(input_dim, output_dim, layers, device, lr=learning_rate)
    print('Modelo MLP criado')
    
    model_path = os.path.join(os.getcwd(), 'models', f'mlp_qm9_{descriptor_type}.pth')
    print(f'Caminho do modelo: {model_path}')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print(f'Modelo carregado e movido para {device}')

    # 5. Preparar dados para explicação
    print('Preparando dados para explicação...')
    X_train = []
    y_train = []
    for i, (xb, yb) in enumerate(train_loader):
        X_train.append(xb)
        y_train.append(yb)
        if i == 0:
            print(f'Primeiro batch train - X shape: {xb.shape}, Y shape: {yb.shape}')
            print(f'Primeiro batch train - X device: {xb.device}, Y device: {yb.device}')
    
    X_train = torch.cat(X_train, dim=0)[:500]
    y_train = torch.cat(y_train, dim=0)[:500]
    print(f'X_train final shape: {X_train.shape}, device: {X_train.device}')
    print(f'y_train final shape: {y_train.shape}, device: {y_train.device}')

    X_test = []
    y_test = []
    for i, (xb, yb) in enumerate(test_loader):
        X_test.append(xb)
        y_test.append(yb)
        if i == 0:
            print(f'Primeiro batch test - X shape: {xb.shape}, Y shape: {yb.shape}')
            print(f'Primeiro batch test - X device: {xb.device}, Y device: {yb.device}')
    
    X_test = torch.cat(X_test, dim=0)[:100]
    y_test = torch.cat(y_test, dim=0)[:100]
    print(f'X_test final shape: {X_test.shape}, device: {X_test.device}')
    print(f'y_test final shape: {y_test.shape}, device: {y_test.device}')

    print('Fazendo predições...')
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()
    print(f'y_pred shape: {y_pred.shape}, type: {type(y_pred)}')

    log_lines.append(f"Test set: {X_test.shape[0]} amostras\n")

    print('Iniciando Explicações...')

    # 6. SHAP
    print('Criando explicador SHAP...')
    shap_explainer = Shap(model, X_train, X_test, device)
    print('SHAP explainer criado')
    
    print('Gerando explicação SHAP global...')
    shap_explanation = shap_explainer.explain_global()
    print(f'SHAP explanation shape: {np.array(shap_explanation).shape}')
    log_lines.append("Explicação SHAP gerada.\n")

    # 7. LIME
    print('Criando explicador LIME...')
    lime_explainer = LIME(model, X_train, X_test, device)
    print('LIME explainer criado')
    
    print('Gerando explicação LIME local...')
    lime_explanation = lime_explainer.explain_local(index=0)
    print(f'LIME explanation shape: {np.array(lime_explanation).shape}')
    log_lines.append("Explicação LIME gerada.\n")

    print('Coletando Métricas...')

    # 8. Métricas SHAP
    print('Criando TabularAnalyzer para SHAP...')
    print(f'Parâmetros - model device: {next(model.parameters()).device}')
    print(f'Parâmetros - X_test device: {X_test.device}')
    print(f'Parâmetros - y_test type: {type(y_test.numpy())}, shape: {y_test.numpy().shape}')
    print(f'Parâmetros - y_pred type: {type(y_pred)}, shape: {y_pred.shape}')
    
    analyzer_shap = TabularAnalyzer(
        model=model,
        explainer=shap_explainer,
        explanation=shap_explanation,
        data=X_test,
        y_true=y_test.numpy(),
        y_pred=y_pred,
        device=device
    )
    print('TabularAnalyzer SHAP criado')
    
    print('Calculando métricas SHAP...')
    fidelity_shap = analyzer_shap.get_metrics()
    print(f'Métricas SHAP calculadas: {fidelity_shap}')
    
    log_lines.append(f"Fidelidade Positiva SHAP: {fidelity_shap[0]}\n")
    log_lines.append(f"Fidelidade Negativa SHAP: {fidelity_shap[1]}\n")

    # 9. Métricas LIME
    print('Criando TabularAnalyzer para LIME...')
    analyzer_lime = TabularAnalyzer(
        model=model,
        explainer=lime_explainer,
        explanation=lime_explanation,
        data=X_test,
        y_true=y_test.numpy(),
        y_pred=y_pred,
        device=device
    )
    print('TabularAnalyzer LIME criado')
    
    print('Calculando métricas LIME...')
    fidelity_lime = analyzer_lime.get_metrics()
    print(f'Métricas LIME calculadas: {fidelity_lime}')
    
    log_lines.append(f"Fidelidade Positiva LIME: {fidelity_lime[0]}\n")
    log_lines.append(f"Fidelidade Negativa LIME: {fidelity_lime[1]}\n")

    # 10. Plots
    print('Gerando Gráficos...')

    try:
        print('Gerando radar plot SHAP...')
        radar_plot(np.array(shap_explanation), title="SHAP - Radar Plot")
        radar_path = os.path.join(experiment_dir, "shap_radar.png")
        plt.savefig(radar_path)
        plt.close()
        log_lines.append(f"Radar plot SHAP salvo em: {radar_path}\n")
        print(f'Radar plot SHAP salvo: {radar_path}')
    except Exception as e:
        log_lines.append(f"Erro ao gerar radar plot SHAP: {e}\n")
        print(f'Erro radar plot SHAP: {e}')

    try:
        print('Gerando bar plot SHAP...')
        horizontal_bar_plot(np.array(shap_explanation), title="SHAP - Feature Importance",
                            save_path=experiment_dir, filename="shap_bar.png")
        log_lines.append(f"Bar plot SHAP salvo em: {os.path.join(experiment_dir, 'shap_bar.png')}\n")
        print(f'Bar plot SHAP salvo')
    except Exception as e:
        log_lines.append(f"Erro ao gerar bar plot SHAP: {e}\n")
        print(f'Erro bar plot SHAP: {e}')

    # LIME - Radar e Bar
    try:
        print('Gerando radar plot LIME...')
        radar_plot(np.array(lime_explanation), title="LIME - Radar Plot")
        radar_path = os.path.join(experiment_dir, "lime_radar.png")
        plt.savefig(radar_path)
        plt.close()
        log_lines.append(f"Radar plot LIME salvo em: {radar_path}\n")
        print(f'Radar plot LIME salvo: {radar_path}')
    except Exception as e:
        log_lines.append(f"Erro ao gerar radar plot LIME: {e}\n")
        print(f'Erro radar plot LIME: {e}')

    try:
        print('Gerando bar plot LIME...')
        horizontal_bar_plot(np.array(lime_explanation), title="LIME - Feature Importance",
                            save_path=experiment_dir, filename="lime_bar.png")
        log_lines.append(f"Bar plot LIME salvo em: {os.path.join(experiment_dir, 'lime_bar.png')}\n")
        print(f'Bar plot LIME salvo')
    except Exception as e:
        log_lines.append(f"Erro ao gerar bar plot LIME: {e}\n")
        print(f'Erro bar plot LIME: {e}')

    # 11. Salvar log
    print('Salvando log...')
    with open(log_file, 'w') as f:
        for line in log_lines:
            f.write(line)
    print(f"Log salvo em {log_file}")
    print(f"Resultados e gráficos salvos em {experiment_dir}")
    print('Experimento finalizado!')

if __name__ == "__main__":
    experiment_MLP_SHAP_LIME_predict_Feature10()