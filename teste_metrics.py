# %% [markdown]
# # Usando MLP com QM9

# %%
from chemxai.models import MLP
from chemxai.data import qm9_tabular
from chemxai.evaluate import Evaluator
from chemxai.train import train_mlp_qm9

import torch
import torch.nn.functional as F

# %%
# qm9 = qm9_tabular()
# learning_rate = 1e-3
# layers = [64, 32]

# # train_mlp_qm9()

# train_loader, val_loader, test_loader, _ = qm9.get_dataloader(batch_size=32)
# input_dim = next(iter(train_loader))[0].shape[1]
# output_dim = 1
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_without_noise = MLP(input_dim, output_dim, layers, device, lr=learning_rate)
# model_without_noise.load_state_dict(torch.load('models/mlp_qm9.pth'))

# print(f'Model without noise:\n{model_without_noise}')

# # %%
# # train_mlp_qm9(n_noise=3)

# train_loader_noise, val_loader_noise, test_loader_noise, _, is_noise = qm9.get_dataloader_with_noise(batch_size=32, n_noise=3)
# input_dim_noise = next(iter(train_loader_noise))[0].shape[1]
# output_dim_noise = 1
# model_with_noise = MLP(input_dim_noise, output_dim_noise, layers, device, lr=learning_rate)
# model_with_noise.load_state_dict(torch.load('models/mlp_qm9_noise.pth'))

# print(f'Model with noise:\n{model_with_noise}')

# # %%
# model_without_noise.to(device)
# model_without_noise.eval()
# test_loss = 0.0

# with torch.no_grad():
#     for batch in test_loader:
#         inputs = batch[0].to(device)
#         targets = batch[1].to(device)
#         preds = model_without_noise(inputs)
#         test_loss += F.mse_loss(preds, targets).item() * inputs.size(0)
        
# test_loss = test_loss / len(test_loader.dataset)

# print(f"\nMSE Sem Noise no teste: {test_loss:.4f}")
# print(f"RMSE Sem Noiseno teste: {test_loss ** 0.5:.4f}")

# # %%
# model_with_noise.to(device)
# model_with_noise.eval()
# test_loss = 0.0

# with torch.no_grad():
#     for batch in test_loader_noise:
#         inputs = batch[0].to(device)
#         targets = batch[1].to(device)
#         preds = model_with_noise(inputs)
#         test_loss += F.mse_loss(preds, targets).item() * inputs.size(0)
        
# test_loss = test_loss / len(test_loader.dataset)

# print(f"\nMSE Com Noise no teste: {test_loss:.4f}")
# print(f"RMSE Com Noise no teste: {test_loss ** 0.5:.4f}")


# # %%

# evaluator = Evaluator(model_without_noise, model_with_noise, train_loader, test_loader, train_loader_noise, test_loader_noise, device)

# similarities, l1_differences, l2_differences, spearman_correlations, fig = evaluator.robustness()

from chemxai.models import GCN
from chemxai.data import graph_datasets
from chemxai.train import train_gcn_qm9, train_gcn_pcqm4

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gd = graph_datasets()

train_loader_normal, val_loader_normal, test_loader_normal, train_loader_noise, val_loader_noise, test_loader_noise = gd.get_paired_dataloaders(
    dataset_name='PCQM4', 
    batch_size=32,
    seed=42,
    noise_type='gaussian',
    noise_scale=1.0
)

data_normal = next(iter(train_loader_normal))
data_noise = next(iter(train_loader_noise))

# train_gcn_qm9()
# train_gcn_pcqm4()
# train_gcn_pcqm4(n_noise=1)
# train_gcn_qm9(n_noise=1)

model_normal = GCN(num_features=data_normal.x.size(1)).to(device)
model_normal.load_state_dict(torch.load('models/gcn_pcqm4.pth', map_location=torch.device(device)))
model_normal = model_normal.to(device)
print(f'Model without noise:\n{model_normal}')

model_noise = GCN(num_features=data_noise.x.size(1)).to(device)
model_noise.load_state_dict(torch.load('models/gcn_pcqm4_noise.pth', map_location=torch.device(device)))
model_noise = model_noise.to(device)
print(f'Model with noise:\n{model_noise}')

evaluator = Evaluator(model_normal, model_noise, train_loader_normal, test_loader_normal, train_loader_noise, test_loader_noise, device, model_type='graph', explainer_type='gnn_explainer', mol_index=0, atom_index=0)

evaluator.robustness()

# # %%
# # Example usage with your data
# # Assuming 'explanation' contains your SHAP explanation
# feature_names = explanation.get('feature_names', None)  # Replace with your actual feature names if available
# lime_values = explanation['feature_importance']

# fig, ax = radar_plot(lime_values, feature_names)

# fig.show()

# # %%



