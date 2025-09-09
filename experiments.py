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

def experiment_MLP_SHAP_LIME(
    att_index=10,
    epochs=10,
    layers=[64, 32],
    learning_rate=1e-3,
    batch_size=32,
    n_noise=0,
    descriptor_type='Physicochemical',
    experiment_id=None
):
    # 1. Criar pasta do experimento
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_lines = []
    log_lines.append(f"Dispositivo: {device}\n")
    print(f'Device: {device}')
    log_lines.append(f"Experimento MLP QM9 - SHAP & LIME\n")
    log_lines.append(f"Parâmetros: att_index={att_index}, epochs={epochs}, layers={layers}, batch_size={batch_size}, n_noise={n_noise}, descriptor_type={descriptor_type}\n")
    log_lines.append(f"Pasta do experimento: {experiment_dir}\n")

    # 2. Dados
    print('Carregando Dados...\n')
    qm9 = qm9_tabular()
    loaders = qm9.get_paired_dataloaders(
        att_index=att_index,
        batch_size=batch_size,
        descriptor_type=descriptor_type,
        n_noise=n_noise,
        add_noise=False
    )
    train_loader, val_loader, test_loader = loaders

    log_lines.append("Dados carregados.\n")
    log_lines.append("Iniciando Treinamento.\n")

    # # 3. Treinamento
    # print('Iniciando Treinamento...\n')
    # history = train_mlp_qm9(
    #     att_index=att_index,
    #     epochs=epochs,
    #     layers=layers,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     n_noise=n_noise,
    #     descriptor_type=descriptor_type,
    # )
    # log_lines.append("Histórico de treinamento (época, treino_loss, val_loss):\n")
    # for epoch, train_loss, val_loss in history:
    #     log_lines.append(f"Época {epoch}: Loss Treino={train_loss:.4f} | Loss Validação={val_loss:.4f}\n")
    # log_lines.append("Treinamento finalizado.\n")
    log_lines.append("Iniciando Explicações.\n")

    # 4. Carregar modelo treinado
    print('Carregando Modelo...')
    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = 1
    model = MLP(input_dim, output_dim, layers, device, lr=learning_rate)
    model_path = os.path.join(os.getcwd(), 'models', f'mlp_qm9_{descriptor_type}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # 5. Preparar dados para explicação
    X_train = []
    y_train = []
    for xb, yb in train_loader:
        X_train.append(xb)
        y_train.append(yb)
    X_train = torch.cat(X_train, dim=0)[:500]
    y_train = torch.cat(y_train, dim=0)[:500]

    X_test = []
    y_test = []
    for xb, yb in test_loader:
        X_test.append(xb)
        y_test.append(yb)
    X_test = torch.cat(X_test, dim=0)[:100]
    y_test = torch.cat(y_test, dim=0)[:100]

    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()

    log_lines.append(f"Test set: {X_test.shape[0]} amostras\n")

    print('Iniciando Explicações...\n')

    # 6. SHAP
    shap_explainer = Shap(model, X_train, X_test, device)
    shap_explanation = shap_explainer.explain_global()
    log_lines.append("Explicação SHAP gerada.\n")

    # 7. LIME
    lime_explainer = LIME(model, X_train, X_test, device)
    lime_explanation = lime_explainer.explain_local(index=0)  # Exemplo: primeira amostra
    log_lines.append("Explicação LIME gerada.\n")

    print('Colentando Métricas...\n')

    # 8. Métricas SHAP
    analyzer_shap = TabularAnalyzer(
        model=model,
        explainer=shap_explainer,
        explanation=shap_explanation,
        data=X_test,
        y_true=y_test.numpy(),
        y_pred=y_pred,
        device=device
    )
    metrics_shap, fidelity_shap = analyzer_shap.get_metrics()
    log_lines.append(f"Métricas SHAP: {metrics_shap}\n")
    log_lines.append(f"Fidelidade SHAP: {fidelity_shap}\n")

    # 9. Métricas LIME
    analyzer_lime = TabularAnalyzer(
        model=model,
        explainer=lime_explainer,
        explanation=lime_explanation,
        data=X_test,
        y_true=y_test.numpy(),
        y_pred=y_pred,
        device=device
    )
    metrics_lime, fidelity_lime = analyzer_lime.get_metrics()
    log_lines.append(f"Métricas LIME: {metrics_lime}\n")
    log_lines.append(f"Fidelidade LIME: {fidelity_lime}\n")

    # 10. Plots
    # SHAP - Radar e Bar
    print('Gerando Gráficos...\n')

    try:
        radar_plot(np.array(shap_explanation), title="SHAP - Radar Plot")
        radar_path = os.path.join(experiment_dir, "shap_radar.png")
        plt.savefig(radar_path)
        plt.close()
        log_lines.append(f"Radar plot SHAP salvo em: {radar_path}\n")
    except Exception as e:
        log_lines.append(f"Erro ao gerar radar plot SHAP: {e}\n")

    try:
        horizontal_bar_plot(np.array(shap_explanation), title="SHAP - Feature Importance",
                            save_path=experiment_dir, filename="shap_bar.png")
        log_lines.append(f"Bar plot SHAP salvo em: {os.path.join(experiment_dir, 'shap_bar.png')}\n")
    except Exception as e:
        log_lines.append(f"Erro ao gerar bar plot SHAP: {e}\n")

    # LIME - Radar e Bar
    try:
        radar_plot(np.array(lime_explanation), title="LIME - Radar Plot")
        radar_path = os.path.join(experiment_dir, "lime_radar.png")
        plt.savefig(radar_path)
        plt.close()
        log_lines.append(f"Radar plot LIME salvo em: {radar_path}\n")
    except Exception as e:
        log_lines.append(f"Erro ao gerar radar plot LIME: {e}\n")

    try:
        horizontal_bar_plot(np.array(lime_explanation), title="LIME - Feature Importance",
                            save_path=experiment_dir, filename="lime_bar.png")
        log_lines.append(f"Bar plot LIME salvo em: {os.path.join(experiment_dir, 'lime_bar.png')}\n")
    except Exception as e:
        log_lines.append(f"Erro ao gerar bar plot LIME: {e}\n")

    # 11. Salvar log
    with open(log_file, 'w') as f:
        for line in log_lines:
            f.write(line)
    print(f"Log salvo em {log_file}")
    print(f"Resultados e gráficos salvos em {experiment_dir}")

if __name__ == "__main__":
    experiment_MLP_SHAP_LIME()