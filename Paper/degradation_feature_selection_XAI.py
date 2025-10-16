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

import optuna
from optuna.trial import TrialState

import sys
import os
import gdown
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chemxai.data import qm9_tabular
from chemxai.explainers import Shap
from chemxai.evaluate import TabularAnalyzer

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, device, lr=0.001, optimizer_name='Adam', loss_function='L1Loss', dropout_rate=0.0):
        super(MLP, self).__init__()
        all_layers = []
        prev_dim = input_dim

        # Camadas ocultas
        for layer_dim in layers:
            all_layers.append(nn.Linear(prev_dim, layer_dim))
            all_layers.append(nn.ReLU())
            if dropout_rate > 0:
                all_layers.append(nn.Dropout(dropout_rate))
            prev_dim = layer_dim

        # Camada de saída (ativação linear - regressão)
        all_layers.append(nn.Linear(prev_dim, output_dim))

        # Combinando as camadas
        self.layers = nn.Sequential(*all_layers)

        # Função de perda
        if loss_function == 'L1Loss':
            self.criterion = nn.L1Loss()
        elif loss_function == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_function == 'SmoothL1Loss':
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.L1Loss()

        # Otimizador
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = device

    def forward(self, x):
        return self.layers(x)
    
def load_mordred_descriptors():

    drive_url = "https://drive.google.com/file/d/1U8mCfVmzDcx30f_7OW1CZzytKJLthB0C/view?usp=sharing"
    output_path = "Paper/desc_mordred_qm9.csv"

    print(f"Tentando baixar o arquivo de: {drive_url}")

    try:
        # O gdown fará a solicitação e o download.
        # O 'fuzzy=True' ajuda a resolver problemas de URL ou ID.
        gdown.download(drive_url, output_path, fuzzy=True, quiet=False)
        
        # Verifica se o arquivo foi baixado com sucesso
        if os.path.exists(output_path):
            print(f"\n✅ Download concluído! Arquivo salvo como: {output_path}")
        else:
            print("\n❌ Falha no download. Verifique se o link está correto e se o arquivo é público ou compartilhado com sua conta.")

    except Exception as e:
        print(f"\nOcorreu um erro: {e}")

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

    df = pd.read_csv("Paper/desc_mordred_qm9.csv")
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
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

def train_with_params(X, y, epochs=100, batch_size=512, lr=0.001, layers=[512, 256, 128], 
                     optimizer_name='Adam', loss_function='L1Loss', dropout_rate=0.0, 
                     weight_decay=0.0, patience=10, device=None):
    """
    Treina modelo com hiperparâmetros específicos
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Criar DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(X, y, batch_size=batch_size)
    
    # Criar modelo
    input_dim = X.shape[1]
    output_dim = 1
    model = MLP(input_dim, output_dim, layers, device, lr, optimizer_name, loss_function, dropout_rate)
    model.to(device)
    
    # Ajustar weight decay se aplicável
    if weight_decay > 0 and optimizer_name in ["Adam", "AdamW"]:
        for param_group in model.optimizer.param_groups:
            param_group['weight_decay'] = weight_decay
    
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

    return model, history

def select_features(model, X, device):
    # Criar split dos dados para SHAP
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Converter para tensores se necessário
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.FloatTensor(X_train[:100]).to(device)
        X_test_tensor = torch.FloatTensor(X_test[:50]).to(device)
    else:
        X_train_tensor = X_train[:100].to(device)
        X_test_tensor = X_test[:50].to(device)
    
    print("Iniciando SHAP explanation...")
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
    
    return X_new, selected_indices, shap_global

def get_fidelity(X, y, model, explanation, sample_size=250):
    """
    Calcula a fidelidade das explicações usando TabularAnalyzer
    
    Args:
        X: Features (numpy array ou tensor)
        y: Targets (numpy array ou tensor)
        model: Modelo treinado
        explanation: Explicação SHAP (numpy array)
        sample_size: Tamanho da amostra para calcular fidelidade
    
    Returns:
        pos_fidel: Fidelidade positiva
        neg_fidel: Fidelidade negativa
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Converter para tensors se necessário
    if isinstance(X_sample, np.ndarray):
        X_tensor = torch.FloatTensor(X_sample).to(device)
    else:
        X_tensor = X_sample.to(device)
        
    if isinstance(y_sample, np.ndarray):
        y_tensor = torch.FloatTensor(y_sample).to(device)
    else:
        y_tensor = y_sample.to(device)
    
    # Obter predições do modelo
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
    
    # Criar analisador tabular
    analyzer = TabularAnalyzer(
        model=model,
        explainer=None,  # Não precisamos do explainer para fidelidade
        explanation=explanation,
        data=X_tensor,
        y_true=y_tensor,
        y_pred=y_pred,
        device=device
    )
    
    pos_fidel, neg_fidel = analyzer.get_metrics(classification=False)
    
    return pos_fidel, neg_fidel


def normalize_prop(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    y_reshaped = y.reshape(-1, 1)
    
    scaler = StandardScaler()
    y_normalized = scaler.fit_transform(y_reshaped).flatten()
    
    # Exibir estatísticas
    print(f"Normalização Standard aplicada:")
    print(f"  Original: min={np.min(y):.4f}, max={np.max(y):.4f}, mean={np.mean(y):.4f}, std={np.std(y):.4f}")
    print(f"  Normalizado: min={np.min(y_normalized):.4f}, max={np.max(y_normalized):.4f}, mean={np.mean(y_normalized):.4f}, std={np.std(y_normalized):.4f}")
    
    return y_normalized


def objective(trial, property_idx=10, max_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar dados
    X, y = get_qm9_desc()
    y_selected = y[:, property_idx]  # Propriedade selecionada
    y_selected = normalize_prop(y_selected)
    
    # Hiperparâmetros a serem otimizados
    # Arquitetura da rede
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 64, 1024, step=64)
        layers.append(layer_size)
    
    # Otimizador e taxa de aprendizado
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # Função de perda
    loss_function = trial.suggest_categorical("loss_function", ["L1Loss", "MSELoss", "SmoothL1Loss"])
    
    # Dropout
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    
    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    
    # Weight decay (apenas para Adam e AdamW)
    if optimizer_name in ["Adam", "AdamW"]:
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        weight_decay = 0.0
    
    try:
        # Criar DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(X, y_selected, batch_size=batch_size)
        
        # Criar modelo
        input_dim = X.shape[1]  # 1057 descritores Mordred
        output_dim = 1
        model = MLP(input_dim, output_dim, layers, device, lr, optimizer_name, loss_function, dropout_rate)
        model.to(device)
        
        # Ajustar weight decay se aplicável
        if weight_decay > 0 and optimizer_name in ["Adam", "AdamW"]:
            for param_group in model.optimizer.param_groups:
                param_group['weight_decay'] = weight_decay
        
        # Variáveis de controle
        best_val_loss = float('inf')
        patience = 10
        early_stop_counter = 0
        
        # Treinamento
        for epoch in range(max_epochs):
            # Fase de treinamento
            model.train()
            train_loss = 0.0
            n_train_batches = 0
            
            for batch in train_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                
                model.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.criterion(outputs.squeeze(), targets)
                loss.backward()
                model.optimizer.step()
                
                train_loss += loss.item()
                n_train_batches += 1
                
                # Limitar batches para acelerar otimização
                if n_train_batches >= 50:  # Máximo 50 batches por época
                    break
            
            avg_train_loss = train_loss / n_train_batches
            
            # Fase de validação
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)
                    outputs = model(inputs)
                    loss = model.criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    n_val_batches += 1
                    
                    # Limitar batches para acelerar otimização
                    if n_val_batches >= 20:  # Máximo 20 batches de validação
                        break
            
            avg_val_loss = val_loss / n_val_batches
            
            # Early stopping e pruning
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Reportar resultado intermediário
            trial.report(avg_val_loss, epoch)
            
            # Verificar se deve fazer pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping
            if early_stop_counter >= patience:
                break
        
        return best_val_loss
        
    except Exception as e:
        print(f"Erro durante o trial: {e}")
        return float('inf')


def run_optuna(property_idx=10, n_trials=100, timeout=3600, study_name=None):
    """
    Executa otimização de hiperparâmetros usando Optuna
    
    Args:
        property_idx: Índice da propriedade QM9 (0-14)
        n_trials: Número de trials para otimização
        timeout: Tempo limite em segundos
        study_name: Nome do estudo (opcional)
    
    Returns:
        study: Objeto do estudo Optuna com resultados
    """
    
    # Nomes das propriedades QM9 para referência
    properties = [
        'Rotational constant A: GHz', 'Rotational constant B: GHz', 'Rotational constant C: GHz',
        'Dipole moment (μ): Debye (D)', 'Isotropic polarizability (α): atomic units (a.u.)',
        'Energy of HOMO (ϵHOMO): Hartree (Ha)', 'Energy of LUMO (ϵLUMO): Hartree (Ha)',
        'Gap (ϵgap): Hartree (Ha)', 'Electronic spatial extent: atomic units (a.u.)',
        'Zero point vibrational energy (zpve): Hartree (Ha)', 'Internal energy at 0 K (U0): Hartree (Ha)',
        'Internal energy at 298.15 K (U): Hartree (Ha)', 'Enthalpy at 298.15 K (H): Hartree (Ha)',
        'Free energy at 298.15 K (G): Hartree (Ha)', 'Heat capacity at 298.15 K (Cv): cal/mol·K'
    ]
    
    property_name = properties[property_idx] if property_idx < len(properties) else f"prop_{property_idx}"
    
    if study_name is None:
        study_name = f"mlp_optimization_{property_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}"
    
    print(f"Iniciando otimização para propriedade: {property_name} (índice: {property_idx})")
    print(f"Configuração: {n_trials} trials, timeout: {timeout}s")
    
    # Criar estudo com direção de minimização
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Função objetivo com propriedade fixa
    objective_with_property = lambda trial: objective(trial, property_idx=property_idx)
    
    # Executar otimização
    study.optimize(objective_with_property, n_trials=n_trials, timeout=timeout)
    
    # Obter estatísticas dos trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS DA OTIMIZAÇÃO")
    print(f"Propriedade: {property_name}")
    print(f"{'='*50}")
    print(f"Número total de trials: {len(study.trials)}")
    print(f"Trials completos: {len(complete_trials)}")
    print(f"Trials podados: {len(pruned_trials)}")
    
    if len(complete_trials) > 0:
        print(f"\nMelhor trial:")
        trial = study.best_trial
        print(f"  Valor (loss de validação): {trial.value:.6f}")
        print(f"  Parâmetros:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Salvar resultados
        safe_property_name = property_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace('μ', 'mu').replace('α', 'alpha').replace('ϵ', 'epsilon')
        results_file = f"optuna_results_{safe_property_name}.csv"
        df_results = study.trials_dataframe()
        df_results.to_csv(results_file, index=False)
        print(f"\nResultados salvos em: {results_file}")
        
        # Criar visualizações (se disponível)
        try:
            import plotly
            
            # Gráfico de otimização
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_html(f"optimization_history_{safe_property_name}.html")
            
            # Importância dos parâmetros
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(f"param_importances_{safe_property_name}.html")
            
            print(f"Gráficos salvos como HTML")
            
        except ImportError:
            print("Plotly não disponível - gráficos não foram gerados")
    
    else:
        print("Nenhum trial foi completado com sucesso!")
    
    return study


def run_degradation_with_params(X, y, best_params, property_name, property_idx, device):
    """
    Executa degradação de features usando hiperparâmetros otimizados
    """
    df_desc = pd.read_csv('Paper/desc_mordred_qm9.csv')
    df_desc.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    print(f"Iniciando degradação para {property_name}")
    print(f"Dados iniciais - Features: {X.shape[1]}, Amostras: {X.shape[0]}")
    
    # Lista para armazenar resultados de cada iteração
    degradation_results = []
    explanation_results = []

    current_X = X.copy()
    current_feature_indices = np.arange(X.shape[1])
    iteration = 0
    
    # Loop de degradação - continua até ter no mínimo 20 features
    while current_X.shape[1] > 20:
        iteration += 1
        print(f"Iteração {iteration} - Features: {current_X.shape[1]}")
        
        # Treinar modelo com hiperparâmetros otimizados
        model, history = train_with_params(
            X=current_X, 
            y=y,
            epochs=100,
            batch_size=best_params['batch_size'],
            lr=best_params['lr'],
            layers=[best_params[f'layer_{i}_size'] for i in range(best_params['n_layers'])],
            optimizer_name=best_params['optimizer'],
            loss_function=best_params['loss_function'],
            dropout_rate=best_params['dropout_rate'],
            weight_decay=best_params.get('weight_decay', 0.0),
            patience=10,
            device=device
        )
        
        # Extrair métricas finais
        final_epoch = history[-1]
        train_loss = final_epoch[1]
        val_loss = final_epoch[2] 
        test_loss = final_epoch[3]
        
        print(f"Resultado - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")
        
        # Armazenar resultados
        degradation_results.append({
            'property': property_name,
            'property_idx': property_idx,
            'iteration': iteration,
            'n_features': current_X.shape[1],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'n_epochs': len(history)
        })

        # Verificar se deve continuar
        if current_X.shape[1] <= 20:
            print(f"Atingiu o limite mínimo de 20 features")
            break
        
        # Selecionar features para próxima iteração
        try:
            prev_X = current_X.copy()
            current_X, selected_indices_relative, shap_explanation = select_features(model, current_X, device)
            
            # Calcular fidelidade (opcional)
            pos_fidel, neg_fidel = None, None
            if shap_explanation is not None:
                try:
                    pos_fidel, neg_fidel = get_fidelity(prev_X, y, model, shap_explanation)
                except Exception as e:
                    print(f"Erro ao calcular fidelidade: {e}")
            
            selected_original_indices = current_feature_indices[selected_indices_relative]
            current_feature_indices = selected_original_indices
            
            explanation_results.append({
                'property': property_name,
                'property_idx': property_idx,
                'iteration': iteration,
                'n_features': current_X.shape[1],
                'features_selected': df_desc.columns[selected_original_indices].tolist(),
                'original_indices': selected_original_indices.tolist(),
                'test_loss': test_loss,
                'shap_explanation': shap_explanation if shap_explanation is not None else None,
                'positive_fidelity': float(pos_fidel) if pos_fidel is not None else None,
                'negative_fidelity': float(neg_fidel) if neg_fidel is not None else None
            })
            
        except Exception as e:
            print(f"Erro na seleção de features: {e}")
            break
    
    return degradation_results, explanation_results


def optimize_and_degrade_all_properties(n_trials=50, timeout_per_property=1800):
    """
    Otimiza hiperparâmetros e executa degradação para todas as propriedades do QM9
    """
    properties = [
        'Rotational constant A: GHz', 'Rotational constant B: GHz', 'Rotational constant C: GHz',
        'Dipole moment (μ): Debye (D)', 'Isotropic polarizability (α): atomic units (a.u.)',
        'Energy of HOMO (ϵHOMO): Hartree (Ha)', 'Energy of LUMO (ϵLUMO): Hartree (Ha)',
        'Gap (ϵgap): Hartree (Ha)', 'Electronic spatial extent: atomic units (a.u.)',
        'Zero point vibrational energy (zpve): Hartree (Ha)', 'Internal energy at 0 K (U0): Hartree (Ha)',
        'Internal energy at 298.15 K (U): Hartree (Ha)', 'Enthalpy at 298.15 K (H): Hartree (Ha)',
        'Free energy at 298.15 K (G): Hartree (Ha)', 'Heat capacity at 298.15 K (Cv): cal/mol·K'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carregar dados uma vez
    X, y = get_qm9_desc()
    
    all_degradation_results = []
    all_explanation_results = []
    optimization_summary = {}
    
    print(f"Iniciando processamento de {len(properties)} propriedades do QM9")
    print(f"Configuração: {n_trials} trials por propriedade, {timeout_per_property}s timeout")
    
    for idx, prop_name in enumerate(properties):
        print(f"\n{'='*80}")
        print(f"PROCESSANDO PROPRIEDADE {idx+1}/{len(properties)}: {prop_name}")
        print(f"{'='*80}")
        
        try:
            # Preparar dados para esta propriedade
            y_selected = y[:, idx]
            y_normalized = normalize_prop(y_selected)
            
            # ETAPA 1: Otimização de hiperparâmetros
            print(f"\n--- ETAPA 1: Otimizando hiperparâmetros ---")
            study = run_optuna(
                property_idx=idx,
                n_trials=n_trials,
                timeout=timeout_per_property,
                study_name=f"mlp_qm9_{prop_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}"
            )
            
            if study.best_trial is None:
                print(f"ERRO: Nenhum trial foi completado para {prop_name}")
                optimization_summary[prop_name] = {'error': 'Nenhum trial completado'}
                continue
            
            best_params = study.best_trial.params
            best_value = study.best_trial.value
            
            print(f"Melhor loss de validação: {best_value:.6f}")
            print(f"Melhores parâmetros: {best_params}")
            
            optimization_summary[prop_name] = {
                'best_value': best_value,
                'best_params': best_params,
                'n_trials': len(study.trials)
            }
            
            # ETAPA 2: Degradação usando hiperparâmetros otimizados
            print(f"\n--- ETAPA 2: Executando degradação ---")
            degradation_results, explanation_results = run_degradation_with_params(
                X=X,
                y=y_normalized,
                best_params=best_params,
                property_name=prop_name,
                property_idx=idx,
                device=device
            )
            
            # Armazenar resultados
            all_degradation_results.extend(degradation_results)
            all_explanation_results.extend(explanation_results)
            
            print(f"Degradação concluída: {len(degradation_results)} iterações")
            
        except Exception as e:
            print(f"ERRO ao processar propriedade {prop_name}: {e}")
            optimization_summary[prop_name] = {'error': str(e)}
            continue
    
    # Salvar todos os resultados
    print(f"\n{'='*80}")
    print("SALVANDO RESULTADOS FINAIS")
    print(f"{'='*80}")
    
    # Salvar resultados de degradação
    if all_degradation_results:
        degradation_df = pd.DataFrame(all_degradation_results)
        degradation_df.to_csv('all_properties_degradation_results.csv', index=False, float_format='%.8f')
        print(f"Resultados de degradação salvos: {len(all_degradation_results)} registros")
    
    # Salvar resultados de explicação
    if all_explanation_results:
        explanation_df = pd.DataFrame(all_explanation_results)
        explanation_df.to_csv('all_properties_explanation_results.csv', index=False)
        print(f"Resultados de explicação salvos: {len(all_explanation_results)} registros")
    
    # Salvar resumo de otimização
    with open('optimization_summary_all_properties.txt', 'w') as f:
        f.write("RESUMO DA OTIMIZAÇÃO E DEGRADAÇÃO - TODAS AS PROPRIEDADES QM9\n")
        f.write("="*80 + "\n\n")
        
        for prop_name, result in optimization_summary.items():
            f.write(f"Propriedade: {prop_name}\n")
            if 'error' in result:
                f.write(f"  ERRO: {result['error']}\n")
            else:
                f.write(f"  Melhor loss de validação: {result['best_value']:.6f}\n")
                f.write(f"  Número de trials: {result['n_trials']}\n")
                f.write(f"  Melhores parâmetros:\n")
                for param, value in result['best_params'].items():
                    f.write(f"    {param}: {value}\n")
            f.write("\n")
    
    print("Resumo de otimização salvo em: optimization_summary_all_properties.txt")
    
    # Estatísticas finais
    successful_properties = len([r for r in optimization_summary.values() if 'error' not in r])
    print(f"\nESTATÍSTICAS FINAIS:")
    print(f"Propriedades processadas com sucesso: {successful_properties}/{len(properties)}")
    print(f"Total de iterações de degradação: {len(all_degradation_results)}")
    print(f"Total de explicações geradas: {len(all_explanation_results)}")
    
    return {
        'degradation_results': all_degradation_results,
        'explanation_results': all_explanation_results,
        'optimization_summary': optimization_summary
    }


if __name__ == '__main__':
    # Executar otimização e degradação para todas as propriedades
    print("Iniciando otimização de hiperparâmetros e análise de degradação para todas as propriedades QM9...")
    
    # Configurar parâmetros
    n_trials = 30  # Número de trials para otimização por propriedade
    timeout_per_property = 1200  # 20 minutos por propriedade
    
    # Executar processo completo
    results = optimize_and_degrade_all_properties(
        n_trials=n_trials,
        timeout_per_property=timeout_per_property
    )
    
    print(f"\nProcesso completo finalizado!")
    print(f"Resultados salvos em:")
    print(f"  - all_properties_degradation_results.csv")
    print(f"  - all_properties_explanation_results.csv")
    print(f"  - optimization_summary_all_properties.txt")


if __name__ == "__main__":
    # Executar otimização e degradação para todas as propriedades
    print("Iniciando otimização de hiperparâmetros e análise de degradação para todas as propriedades QM9...")
    
    # Configurar parâmetros
    n_trials = 30  # Número de trials para otimização por propriedade
    timeout_per_property = 1200  # 20 minutos por propriedade
    
    # Executar processo completo
    results = optimize_and_degrade_all_properties(
        n_trials=n_trials,
        timeout_per_property=timeout_per_property
    )
    
    print(f"\nProcesso completo finalizado!")
    print(f"Resultados salvos em:")
    print(f"  - all_properties_degradation_results.csv")
    print(f"  - all_properties_explanation_results.csv")
    print(f"  - optimization_summary_all_properties.txt")
    