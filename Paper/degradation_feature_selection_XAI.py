import pandas as pd
import matplotlib.pyplot as plt

import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chemxai.data import qm9_tabular
from chemxai.explainers import Shap


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, device, lr=0.001):
        super(MLP, self).__init__()
        all_layers = []
        prev_dim = input_dim

        # Camadas ocultas
        for layer_dim in layers:
            all_layers.append(nn.Linear(prev_dim, layer_dim))
            all_layers.append(nn.ReLU())
            prev_dim = layer_dim

        # Camada de saída (ativação linear - regressão)
        all_layers.append(nn.Linear(prev_dim, output_dim))

        # Combinando as camadas
        self.layers = nn.Sequential(*all_layers)

        self.criterion = nn.L1Loss() #nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, x):
        return self.layers(x)
    
def load_mordred_descriptors():

    df = pd.read_csv("all_desc_qm9.csv")
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_valid = df.describe()
    df_usable = df[df_valid.columns]
    df_usable.to_csv("desc_mordred_qm9.csv")

    return df_usable

def dataframe_to_numpy(df):
    """
    Converte DataFrame para arrays NumPy
    """
    X = df[df.columns].values
    
    # Remover NaN se existirem
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    
    print(f"Features shape: {X.shape}")
    
    return X

def get_qm9_desc():

    df = pd.read_csv("desc_mordred_qm9.csv")

    # descritores (1057)
    feats = dataframe_to_numpy(df)

    qm9 = qm9_tabular()

    _, prop, _ = qm9.load_qm9_dataset()

    # todas propriedades qm9
    props = np.array(prop)

    return feats, props

def get_dataloaders(X, y, split=[80, 10, 10], batch_size=512):
    """
    Cria DataLoaders para treino, validação e teste
    
    Args:
        X: Features (numpy array)
        y: Targets (numpy array) 
        split: Lista com percentuais [treino, validação, teste]
        batch_size: Tamanho do batch
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Classe Dataset customizada
    class CustomDataset(Dataset):
        def __init__(self, features, targets, normalize=True):
            self.features = features.astype(np.float32)
            self.targets = targets.astype(np.float32)
            
            # Normalizar dados
            if normalize:
                self.scaler_X = StandardScaler()
                self.scaler_y = StandardScaler()
                
                self.features = self.scaler_X.fit_transform(self.features)
                
                # Normalizar targets (reshape se necessário)
                if self.targets.ndim == 1:
                    targets_reshaped = self.targets.reshape(-1, 1)
                    self.targets = self.scaler_y.fit_transform(targets_reshaped).flatten()
                else:
                    self.targets = self.scaler_y.fit_transform(self.targets)
            
            # Converter para tensors PyTorch
            self.features = torch.FloatTensor(self.features)
            self.targets = torch.FloatTensor(self.targets)
            
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.targets[idx]
    
    # Garantir que X e y tenham o mesmo número de amostras
    min_samples = min(len(X), len(y))
    X = X[:min_samples]
    y = y[:min_samples]
    
    # Criar dataset
    dataset = CustomDataset(X, y, normalize=True)
    
    # Calcular tamanhos dos splits (converter percentuais para proporções)
    total_size = len(dataset)
    train_size = int(total_size * split[0] / 100)
    val_size = int(total_size * split[1] / 100)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Dividir dataset aleatoriamente
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Para reprodutibilidade
    )
    
    # Criar DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def train(X, y, epochs=100, batch_size=512, lr=0.001, layers=[512, 256, 128], patience=10):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Criar DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(X, y, batch_size=batch_size)
    
    # Criar modelo
    input_dim = X.shape[1]
    output_dim = 1
    model = MLP(input_dim, output_dim, layers, device, lr)
    model.to(device)
    
    # Variáveis de controle
    history = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        
        # Treinamento
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.criterion(outputs.squeeze(), targets)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
    
        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(inputs)
                loss = model.criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Teste
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(inputs)
                loss = model.criterion(outputs.squeeze(), targets)
                test_loss += loss.item() * inputs.size(0)
                
        avg_test_loss = test_loss / len(test_loader.dataset)

        # Armazenar histórico
        history.append([epoch + 1, avg_train_loss, avg_val_loss, avg_test_loss])

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            break
        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, Test: {avg_test_loss:.4f}')

    return model, history

def select_features(model, X, device):
    # Criar split dos dados para SHAP
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Converter para tensores se necessário
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
    else:
        X_train_tensor = X_train.to(device)
        X_test_tensor = X_test.to(device)
    
    # Criar explicador SHAP
    explainer = Shap(model, X_train_tensor, X_test_tensor, device)
    shap_global = explainer.explain_global()
    
    # Calcular importância absoluta das features
    feature_importance = np.abs(shap_global)
    
    # Calcular quantas features manter (remover 10%)
    n_features = len(feature_importance)
    n_features_to_keep = int(n_features * 0.9)
    
    # Garantir mínimo de 20 features
    n_features_to_keep = max(n_features_to_keep, 20)
    
    # Selecionar índices das features mais importantes
    selected_indices = np.argsort(feature_importance)[::-1][:n_features_to_keep]
    selected_indices = np.sort(selected_indices)  # Manter ordem original
    
    # Filtrar features
    X_new = X[:, selected_indices]
    
    print(f"Features reduzidas de {n_features} para {n_features_to_keep} ({(1-n_features_to_keep/n_features)*100:.1f}% removidas)")
    
    return X_new, selected_indices

def run_train_degradation():
    """
    Executa treinamento com degradação progressiva de features baseada em SHAP
    """
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar dados iniciais
    # load_mordred_descriptors()
    X, y = get_qm9_desc()
    y_selected = y[:, 10]  # mu (momento dipolar)
    
    print(f"Dados iniciais - Features: {X.shape[1]}, Amostras: {X.shape[0]}")
    
    # Lista para armazenar resultados de cada iteração
    degradation_results = []
    
    current_X = X.copy()
    iteration = 0
    
    # Loop de degradação - continua até ter no mínimo 20 features
    while current_X.shape[1] > 20:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"ITERAÇÃO {iteration} - Features: {current_X.shape[1]}")
        print(f"{'='*50}")
        
        # Treinar modelo com features atuais
        print("Treinando modelo...")
        model, history = train(current_X, y_selected, epochs=100)
        
        # Extrair métricas finais
        final_epoch = history[-1]
        train_loss = final_epoch[1]
        val_loss = final_epoch[2] 
        test_loss = final_epoch[3]
        
        print(f"Resultado final - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")
        
        # Armazenar resultados
        degradation_results.append({
            'iteration': iteration,
            'n_features': current_X.shape[1],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'n_epochs': len(history)
        })
        
        # Mostrar progresso do erro de teste
        if iteration == 1:
            print(f"Erro de teste inicial: {test_loss:.4f}")
        else:
            prev_test_loss = degradation_results[-2]['test_loss']
            change = test_loss - prev_test_loss
            change_pct = (change / prev_test_loss) * 100
            direction = "↑" if change > 0 else "↓"
            print(f"Mudança no erro de teste: {direction} {abs(change):.4f} ({change_pct:+.1f}%)")
        
        # Verificar se deve continuar
        if current_X.shape[1] <= 20:
            print(f"\nParando: Atingiu o limite mínimo de 20 features")
            break
        
        # Selecionar features para próxima iteração
        print("Selecionando features com SHAP...")
        try:
            current_X, selected_indices = select_features(model, current_X, device)
        except Exception as e:
            print(f"Erro na seleção de features: {e}")
            break
    
    # Exibir resumo final
    print(f"\n{'='*50}")
    print("RESUMO DA DEGRADAÇÃO")
    print(f"{'='*50}")
    
    for result in degradation_results:
        print(f"Iteração {result['iteration']}: {result['n_features']} features -> "
              f"Train: {result['train_loss']:.4f}, Val: {result['val_loss']:.4f}, Test: {result['test_loss']:.4f}")
    
    return degradation_results


if __name__ == '__main__':
    
    # Executar degradação progressiva de features
    print("Iniciando análise de degradação de features...")
    results = run_train_degradation()
    
    # Plotar resultados (opcional)
    if results:

        iterations = [r['iteration'] for r in results]
        n_features = [r['n_features'] for r in results]
        train_losses = [r['train_loss'] for r in results]
        val_losses = [r['val_loss'] for r in results]
        test_losses = [r['test_loss'] for r in results]
        
        plt.figure(figsize=(18, 6))
        
        # Plot 1: Número de features vs Iteração
        plt.subplot(1, 3, 1)
        plt.plot(iterations, n_features, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Iteração')
        plt.ylabel('Número de Features')
        plt.title('Degradação de Features')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Losses vs Número de features
        plt.subplot(1, 3, 2)
        plt.plot(n_features, train_losses, 'go-', label='Train Loss', linewidth=2, markersize=6)
        plt.plot(n_features, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=6)
        plt.plot(n_features, test_losses, 'bo-', label='Test Loss', linewidth=2, markersize=6)
        plt.xlabel('Número de Features')
        plt.ylabel('Loss (MAE)')
        plt.title('Performance vs Número de Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Test Loss vs Iteração
        plt.subplot(1, 3, 3)
        plt.plot(iterations, test_losses, 'mo-', linewidth=2, markersize=8)
        plt.xlabel('Iteração')
        plt.ylabel('Test Loss (MAE)')
        plt.title('Evolução do Erro de Teste')
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações nos pontos
        for i, (iter_num, test_loss, n_feat) in enumerate(zip(iterations, test_losses, n_features)):
            plt.annotate(f'{n_feat}f', (iter_num, test_loss), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('feature_degradation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Exibir estatísticas adicionais
        print(f"\n{'='*50}")
        print("ESTATÍSTICAS DA DEGRADAÇÃO")
        print(f"{'='*50}")
        
        best_test_idx = np.argmin(test_losses)
        worst_test_idx = np.argmax(test_losses)
        
        print(f"Melhor erro de teste: {test_losses[best_test_idx]:.4f} (Iteração {iterations[best_test_idx]}, {n_features[best_test_idx]} features)")
        print(f"Pior erro de teste: {test_losses[worst_test_idx]:.4f} (Iteração {iterations[worst_test_idx]}, {n_features[worst_test_idx]} features)")
        print(f"Variação total do erro de teste: {max(test_losses) - min(test_losses):.4f}")
        print(f"Features removidas: {n_features[0] - n_features[-1]} ({((n_features[0] - n_features[-1])/n_features[0]*100):.1f}%)")
        
        # Salvar resultados em CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('degradation_results.csv', index=False)
        print(f"Resultados salvos em 'degradation_results.csv'")
        
        print(f"\nGráfico salvo como 'feature_degradation_analysis.png'")
    