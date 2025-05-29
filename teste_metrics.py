# %% [markdown]
# # Usando MLP com QM9

# %%
from chemxai.models import MLP
from chemxai.data import qm9_tabular
from chemxai.evaluate import robustness

import torch
import torch.nn.functional as F

# %%
qm9 = qm9_tabular()

train_loader, val_loader, test_loader, _ = qm9.get_dataloader(batch_size=32)
input_dim = next(iter(train_loader))[0].shape[1]
output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-3
layers = [64, 32]

model_without_noise = MLP(input_dim, output_dim, layers, device, lr=learning_rate)
model_without_noise.load_state_dict(torch.load('models/mlp_qm9.pth'))
print(model_without_noise)

# %%
train_loader_noise, val_loader_noise, test_loader_noise, _, is_noise = qm9.get_dataloader_with_noise(batch_size=32)

print(is_noise)

input_dim_noise = next(iter(train_loader_noise))[0].shape[1]
output_dim_noise = 1
model_with_noise = MLP(input_dim_noise, output_dim_noise, layers, device, lr=learning_rate)
model_with_noise.load_state_dict(torch.load('models/mlp_qm9_noise.pth'))
print(model_with_noise)

# %%
model_without_noise.to(device)
model_without_noise.eval()
test_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        preds = model_without_noise(inputs)
        test_loss += F.mse_loss(preds, targets).item() * inputs.size(0)
        
test_loss = test_loss / len(test_loader.dataset)

print(f"\nMSE no teste: {test_loss:.4f}")
print(f"RMSE no teste: {test_loss ** 0.5:.4f}")

# %%
model_with_noise.to(device)
model_with_noise.eval()
test_loss = 0.0

with torch.no_grad():
    for batch in test_loader_noise:
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        preds = model_with_noise(inputs)
        test_loss += F.mse_loss(preds, targets).item() * inputs.size(0)
        
test_loss = test_loss / len(test_loader.dataset)

print(f"\nMSE no teste: {test_loss:.4f}")
print(f"RMSE no teste: {test_loss ** 0.5:.4f}")


# %%
similarities, l1_differences, l2_differences, spearman_correlations, fig = robustness(model_without_noise, model_with_noise, train_loader, test_loader, train_loader_noise, test_loader_noise, device)

# # %% [markdown]
# # # GCN e QM9 com GraphShap

# # %%
# from chemxai.data import prepare_data_graph
# from chemxai.models import GCN
# from chemxai.explainers import GraphShap
# from chemxai.plots import radar_plot

# # %%
# # Teste GCN com QM9 e GraphShap -> Funcional -> Explicação para as features do grafo 0
# data = prepare_data_graph('QM9')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(num_features=data.num_features).to(device)
# model.load_state_dict(torch.load('models/gcn_qm9.pth', map_location=torch.device(device)))
# model = model.to(device)
# explainer = GraphShap(data[0], model, device, gpu=torch.cuda.is_available())
# explanation = explainer.explain()
# print("Shapley Values for Node Features:")
# print(explanation['shap_values'])
# print("\nTop Features (if applicable):")
# print(explanation['top_features'])
# print("\nTop Shapley Values (if applicable):")
# print(explanation['top_values'])

# # %%
# # Example usage with your data
# # Assuming 'explanation' contains your SHAP explanation
# feature_names = explanation.get('feature_names', None)  # Replace with your actual feature names if available
# shap_values = explanation['shap_values']

# fig, ax = radar_plot(shap_values, feature_names)

# fig.show()

# # %% [markdown]
# # ## Avaliações

# # %% [markdown]
# # Teste 1: [ 5 10  1  4  3  0  2  7  8  6  9]\
# # Teste 2: [ 5 10  1  9  7  2  3  4  0  6  8]\
# # Teste 3: [ 5 10  1  6  7  8  4  0  2  9  3]\
# # Teste 4: [ 5 10  1  9  6  7  8  0  4  2  3]\
# # Teste 5: [ 5 10  2  8  9  7  0  1  4  3  6]\
# # Teste 6: [ 5 10  0  6  1  2  9  7  4  8  3]\
# # Teste 7: [ 5 10  9  8  1  4  2  7  3  6  0]\
# # Teste 8: [ 5 10  4  0  3  9  1  6  8  2  7]
# # 

# # %% [markdown]
# # # GCN e QM9 com GraphLIME

# # %%
# from chemxai.explainers import GraphLIME

# # %%
# data = prepare_data_graph('QM9')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(num_features=data.num_features).to(device)
# model.load_state_dict(torch.load('models/gcn_qm9.pth', map_location=torch.device(device)))
# model = model.to(device)
# explainer = GraphLIME(model=model, device=device, rho=0.1)
# explanation = explainer.explain(data[0], num_samples=100)
# print("Features Importance:")
# print(explanation['feature_importance'])
# print("\nTop Features:")
# print(explanation['top_features'])

# # %%
# # Example usage with your data
# # Assuming 'explanation' contains your SHAP explanation
# feature_names = explanation.get('feature_names', None)  # Replace with your actual feature names if available
# lime_values = explanation['feature_importance']

# fig, ax = radar_plot(lime_values, feature_names)

# fig.show()

# # %%



