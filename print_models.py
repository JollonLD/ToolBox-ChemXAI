# import torch
# import torch.nn as nn
# from torchviz import make_dot

# from chemxai.models import MLP, GCN

# # Parâmetros conforme seu treinamento
# # Para Physicochemical: input_dim = 9
# # Para Morgan: input_dim = 512
# # Para CM: input_dim = 64
# input_dim = 9  # Altere para 512 (Morgan) ou 64 (CM) se quiser visualizar outros descritores
# output_dim = 1
# layers = [128, 64]
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # --- Visualização do MLP ---
# mlp_model = MLP(input_dim=input_dim, output_dim=output_dim, layers=layers, device=device)
# dummy_input_mlp = torch.randn(1, input_dim)  # (batch_size, input_dim)
# y_mlp = mlp_model(dummy_input_mlp)

# make_dot(y_mlp, params=dict(mlp_model.named_parameters())).render("MLP_architecture", format="pdf")
# print("Diagrama do MLP salvo em MLP_architecture.pdf")

# # --- Visualização do GCN ---
# # Para QM9, normalmente são 11 features por nó
# num_features = 11
# gcn_model = GCN(num_features=num_features)
# num_nodes = 10
# dummy_x_gcn = torch.randn(num_nodes, num_features)
# dummy_edge_index = torch.randint(0, num_nodes, (2, 20))
# y_gcn = gcn_model(dummy_x_gcn, dummy_edge_index, batch=torch.zeros(num_nodes, dtype=torch.long))

# make_dot(y_gcn, params=dict(gcn_model.named_parameters())).render("GCN_architecture", format="pdf")
# print("Diagrama do GCN salvo em GCN_architecture.pdf")

import matplotlib.pyplot as plt

def draw_mlp(layer_sizes, layer_labels=None, filename="MLP_schematic_simple.pdf"):
    fig, ax = plt.subplots(figsize=(8, 4))
    v_spacing = 1.0
    h_spacing = 2.0
    radius = 0.18

    n_layers = len(layer_sizes)
    for i, n_neurons in enumerate(layer_sizes):
        x = i * h_spacing
        y_start = -(n_neurons - 1) * v_spacing / 2
        for j in range(n_neurons):
            y = y_start + j * v_spacing
            circle = plt.Circle((x, y), radius, color="#cce5ff", ec="k", zorder=2)
            ax.add_patch(circle)
            if i > 0:
                prev_x = (i - 1) * h_spacing
                prev_n = layer_sizes[i - 1]
                prev_y_start = -(prev_n - 1) * v_spacing / 2
                for k in range(prev_n):
                    prev_y = prev_y_start + k * v_spacing
                    ax.plot([prev_x + radius, x - radius], [prev_y, y], "k-", lw=0.7, zorder=1)
        # Nome da camada embaixo
        if layer_labels:
            ax.text(x, y_start - 0.6, layer_labels[i], ha="center", fontsize=12, weight="bold")

    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Exemplo: 9 entradas, 2 camadas ocultas (128, 64), 1 saída
layer_sizes = [9, 8, 5, 1]  # Reduza para visualização, aumente se quiser mais neurônios
layer_labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
draw_mlp(layer_sizes, layer_labels)

import matplotlib.pyplot as plt

def draw_gcn_schematic_vertical(filename="GCN_schematic_vertical.pdf"):
    fig, ax = plt.subplots(figsize=(4, 8))
    h_spacing = 1.0
    v_spacing = 2.0
    radius = 0.18

    n_nodes = 5  # Para visualização didática

    # Posições das camadas (vertical, input em cima)
    layers_y = [0, 1, 2, 3, 4]
    layer_labels = [
        "Input\nNode Features",
        "GCN Layer 1\n+ ReLU",
        "GCN Layer 2\n+ ReLU",
        "GCN Layer 3\n+ ReLU",
        "Global Pooling\n+ Linear Output"
    ]

    # Inverter a ordem para input em cima
    for i, y in enumerate(layers_y):
        y_plot = -y * v_spacing  # Inverte o eixo y
        x_start = -(n_nodes - 1) * h_spacing / 2
        for j in range(n_nodes if i < 4 else 1):
            x = x_start + j * h_spacing if i < 4 else 0
            circle = plt.Circle((x, y_plot), radius, color="#cce5ff", ec="k", zorder=2)
            ax.add_patch(circle)
            # Conexões entre camadas
            if i > 0 and i < 4:
                for k in range(n_nodes):
                    prev_x = x_start + k * h_spacing
                    ax.plot(
                        [prev_x, x],
                        [-(y - 1) * v_spacing - radius, y_plot + radius],
                        "k-", lw=0.7, zorder=1, alpha=0.7
                    )
        # Conexão do último GCN para o pooling/output
        if i == 4:
            for k in range(n_nodes):
                prev_x = x_start + k * h_spacing
                ax.plot(
                    [prev_x, 0],
                    [-(y - 1) * v_spacing - radius, y_plot + radius],
                    "k-", lw=0.7, zorder=1, alpha=0.7
                )
        # Nome da camada à esquerda
        ax.text(-n_nodes * h_spacing / 2 - 0.5, y_plot, layer_labels[i], ha="right", va="center", fontsize=12, weight="bold")

    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

draw_gcn_schematic_vertical()