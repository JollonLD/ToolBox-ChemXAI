import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import os
import time
import datetime
import json
import logging
from pathlib import Path

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models import GCN, MLP
from data import graph_datasets, qm9_tabular

# Configure logging system
def setup_logging(log_dir="logs"):
    """Configure logging to file and console with timestamps"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def train_mlp_qm9(att_index=10, epochs=10, layers=[64, 32], learning_rate=1e-3, 
                batch_size=32, n_noise=3, descriptor_type='Morgan', cache_descriptors=True,
                num_workers=4, morgan_radius=2, morgan_nBits=512, log_dir="logs"):
    """
    Função para treinar um modelo MLP com dados moleculares usando descritores otimizados.

    Args:
        att_index (int): Índice da propriedade a ser predita (default: 10)
        epochs (int): Número de épocas de treinamento (default: 10)
        layers (list): Lista de dimensões das camadas ocultas (default: [64, 32])
        learning_rate (float): Taxa de aprendizado (default: 1e-3)
        batch_size (int): Tamanho do lote (default: 32)
        n_noise (int): Número de features de ruído a adicionar (default: 3)
        descriptor_type (str): Tipo de descritor molecular. Opções:
            - 'CM': Matriz de Coulomb
            - 'Morgan': Fingerprints circulares de Morgan (ECFP)
            - 'MorganCount': Fingerprints de Morgan com contagem
            - 'Physicochemical': Descritores físico-químicos 2D
            - '3D': Descritores baseados na geometria 3D
            - 'MACCS': Keys MACCS (166 bits)
            - 'Topological': Fingerprints topológicos (tipo Daylight)
            - 'AtomPair': Fingerprints baseados em pares de átomos
            - 'EState': Fingerprints baseados em índices de estado eletrotopológico
            - 'Pattern': Fingerprints baseados em padrões SMARTS
            - 'Avalon': Fingerprints Avalon para triagem de subestruturas
            - 'Autocorr': Descritores de autocorrelação 2D
        cache_descriptors (bool): Se deve usar cache para descritores (default: True)
        num_workers (int): Número de workers para carregamento de dados (default: 4)
        morgan_radius (int): Raio para fingerprints Morgan (default: 2)
        morgan_nBits (int): Número de bits para fingerprints (default: 512)
        log_dir (str): Diretório para salvar logs detalhados (default: "logs")

    Returns:
        list: Uma lista de tuplas (epoch, train_loss, val_loss) para cada época.
    """
    # Configurar logging específico para este modelo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mlp_att{att_index}_{descriptor_type}"
    model_log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    
    os.makedirs(log_dir, exist_ok=True)
    model_logger = logging.getLogger(f"model_{model_name}")
    model_logger.setLevel(logging.INFO)
    
    # Verificar se o logger já tem handlers para evitar duplicatas
    if not model_logger.handlers:
        file_handler = logging.FileHandler(model_log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        model_logger.addHandler(file_handler)
    
    # Log de início do treinamento
    model_logger.info(f"Iniciando treinamento do modelo {model_name}")
    model_logger.info(f"Configuração: att_index={att_index}, epochs={epochs}, layers={layers}, "
                      f"lr={learning_rate}, batch_size={batch_size}, descriptor_type={descriptor_type}")

    dirname = os.getcwd()
    models_dir = os.path.join(dirname, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_logger.info(f'Diretório para modelos: {models_dir}')
    
    if n_noise > 0:
        path = os.path.join(models_dir, f'mlp_qm9_noise_{descriptor_type}_att{att_index}.pth')
    else:
        path = os.path.join(models_dir, f'mlp_qm9_{descriptor_type}_att{att_index}.pth')

    # Carregar os dados usando a função otimizada
    model_logger.info("Carregando dados...")
    start_time_data = time.time()
    qm9 = qm9_tabular()
    
    try:
        if n_noise > 0:
            model_logger.info(f"Preparando dataloaders com ruído (n_noise={n_noise})...")
            train_loader, val_loader, test_loader, train_loader_noise, val_loader_noise, test_loader_noise, is_noise = qm9.get_paired_dataloaders_tabular(
                att_index=att_index,
                batch_size=batch_size,
                descriptor_type=descriptor_type,
                list_mols=[],
                n_noise=n_noise,
                morgan_radius=morgan_radius, 
                morgan_nBits=morgan_nBits,
                cache_descriptors=cache_descriptors
            )
        else:    
            model_logger.info(f"Preparando dataloaders sem ruído...")
            train_loader, val_loader, test_loader, *_ = qm9.get_paired_dataloaders_tabular(
                att_index=att_index,
                batch_size=batch_size,
                descriptor_type=descriptor_type,
                list_mols=[],
                n_noise=n_noise,
                morgan_radius=morgan_radius, 
                morgan_nBits=morgan_nBits,
                cache_descriptors=cache_descriptors
            )
        
        data_load_time = time.time() - start_time_data
        model_logger.info(f"Dados carregados em {data_load_time:.2f} segundos")
        model_logger.info(f"Tamanhos dos conjuntos - Treino: {len(train_loader.dataset)}, " 
                          f"Validação: {len(val_loader.dataset)}, Teste: {len(test_loader.dataset)}")
    
    except Exception as e:
        model_logger.error(f"Erro ao carregar dados: {str(e)}")
        raise e

    # Definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_logger.info(f'Usando dispositivo: {device}')

    # Obter a dimensão da entrada (tamanho do descritor)
    try:
        batch_data = next(iter(train_loader))
        input_dim = batch_data[0].shape[1]
        output_dim = 1  # Previsão de uma única propriedade
        model_logger.info(f"Dimensão de entrada: {input_dim}, Dimensão de saída: {output_dim}")
    except Exception as e:
        model_logger.error(f"Erro ao determinar dimensões do modelo: {str(e)}")
        raise e

    # Definir a arquitetura da MLP
    try:
        model_logger.info(f"Criando modelo MLP com camadas: {layers}")
        model = MLP(input_dim, output_dim, layers, device, lr=learning_rate)
        model_logger.info(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    except Exception as e:
        model_logger.error(f"Erro ao criar modelo: {str(e)}")
        raise e

    history = []
    model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    patience = 10  # Número de épocas para esperar melhoria na val_loss

    # Início do tempo de treinamento
    train_start_time = time.time()
    model_logger.info(f"Iniciando treinamento com {epochs} épocas...")

    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Treinamento
        model.train()
        train_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.criterion(outputs, targets)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            batch_count += 1
            
            # Log a cada 50 batches
            if batch_count % 50 == 0:
                model_logger.info(f"Época {epoch+1}/{epochs}, Batch {batch_count}/{len(train_loader)}, "
                                f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader.dataset)
    
        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)

                outputs = model(inputs)
                loss = model.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        # Calcular tempo da época
        epoch_time = time.time() - epoch_start
        
        # Log de progresso
        model_logger.info(f"[{epoch+1}/{epochs}] "
                         f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                         f"Tempo: {epoch_time:.2f}s")

        # Armazenar as médias das losses
        history.append((epoch + 1, avg_train_loss, avg_val_loss))

        # Verificar se é o melhor modelo até agora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, path)
            model_logger.info(f"Melhor modelo salvo em: {path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= patience:
            model_logger.info(f"Early stopping ativado após {patience} épocas sem melhoria")
            break

    # Tempo total de treinamento
    total_train_time = time.time() - train_start_time
    model_logger.info(f"Treinamento concluído em {total_train_time:.2f} segundos "
                     f"({total_train_time/60:.2f} minutos)")

    # Avaliação final
    model_logger.info("Iniciando avaliação no conjunto de teste...")
    test_start_time = time.time()
    try:
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        test_loss = 0.0

        predictions = []
        true_values = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                preds = model(inputs)
                test_loss += F.mse_loss(preds, targets).item() * inputs.size(0)
                
                # Coletar predições e valores verdadeiros para análise
                predictions.extend(preds.cpu().numpy())
                true_values.extend(targets.cpu().numpy())
                
        test_loss = test_loss / len(test_loader.dataset)
        rmse = test_loss ** 0.5
        
        # Calcular métricas adicionais
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
        
        test_time = time.time() - test_start_time
        model_logger.info(f"Avaliação do teste concluída em {test_time:.2f} segundos")
        model_logger.info(f"MSE no teste: {test_loss:.4f}")
        model_logger.info(f"RMSE no teste: {rmse:.4f}")
        model_logger.info(f"MAE no teste: {mae:.4f}")
        
        # Salvar métricas em arquivo JSON
        metrics = {
            "model": model_name,
            "descriptor_type": descriptor_type,
            "att_index": att_index,
            "layers": layers,
            "train_loss": float(history[-1][1]),
            "val_loss": float(history[-1][2]),
            "test_loss": float(test_loss),
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "training_time_seconds": float(total_train_time),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metrics_file = os.path.join(log_dir, f"{model_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        model_logger.info(f"Métricas salvas em {metrics_file}")
        
    except Exception as e:
        model_logger.error(f"Erro durante avaliação final: {str(e)}")
    
    return history

def train_gcn_qm9(target_idx=3, epochs=10, batch_size=64, lr=0.001, weight_decay=1e-4, n_noise=0, log_dir="logs"):
    # Configuração do logger similar ao MLP
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"gcn_target{target_idx}"
    model_log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    
    os.makedirs(log_dir, exist_ok=True)
    model_logger = logging.getLogger(f"model_{model_name}")
    model_logger.setLevel(logging.INFO)
    
    if not model_logger.handlers:
        file_handler = logging.FileHandler(model_log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        model_logger.addHandler(file_handler)
    
    model_logger.info(f"Iniciando treinamento do modelo GCN para target_idx={target_idx}")
    model_logger.info(f"Configuração: epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")
    
    dirname = os.getcwd()
    models_dir = os.path.join(dirname, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_logger.info(f'Diretório para modelos: {models_dir}')
    
    if n_noise > 0:
        path = os.path.join(models_dir, f'gcn_qm9_noise_target{target_idx}.pth')
    else:
        path = os.path.join(models_dir, f'gcn_qm9_target{target_idx}.pth')

    # Detectar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_logger.info(f"Usando dispositivo: {device}")

    # Carregar dados
    start_time_data = time.time()
    model_logger.info("Carregando dados de grafos...")
    
    try:
        gd = graph_datasets()

        # Usar get_paired_dataloaders para obter dados alinhados
        loaders = gd.get_paired_dataloaders(
            dataset_name='QM9', 
            batch_size=batch_size,
            seed=42,
            noise_type='gaussian',
            noise_scale=1.0
        )
        
        # Decidir quais loaders usar com base na presença de ruído
        if n_noise > 0:
            model_logger.info(f"Usando loaders com ruído (n_noise={n_noise})...")
            _, _, _, train_loader, val_loader, test_loader = loaders
        else:
            model_logger.info(f"Usando loaders sem ruído...")
            train_loader, val_loader, test_loader, _, _, _ = loaders

        data_load_time = time.time() - start_time_data
        model_logger.info(f"Dados carregados em {data_load_time:.2f} segundos")
        
    except Exception as e:
        model_logger.error(f"Erro ao carregar dados: {str(e)}")
        raise e

    # Obter exemplo do primeiro batch para inicializar o modelo
    try:
        data_sample = next(iter(train_loader))
        model_logger.info(f"Características do grafo de exemplo - Nós: {data_sample.x.size(0)}, Features: {data_sample.x.size(1)}")
    
        # 4. Instanciar modelo para regressão
        model = GCN(
            num_features=data_sample.x.size(1)
        ).to(device)
        model_logger.info(f"Modelo GCN criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    
    except Exception as e:
        model_logger.error(f"Erro ao criar modelo GCN: {str(e)}")
        raise e

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Início do tempo de treinamento
    train_start_time = time.time()
    model_logger.info(f"Iniciando treinamento com {epochs} épocas...")

    # 5. Treinamento
    best_val_loss = float('inf')
    history = []
    early_stop_counter = 0
    patience = 10  # Número de épocas para esperar melhoria na val_loss

    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        total_loss = 0
        num_samples = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Selecionar apenas a propriedade específica que queremos prever
            target = batch.y[:, target_idx].view(-1, 1)
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Garantir que as dimensões correspondam
            if out.shape[0] != target.shape[0]:
                min_size = min(out.shape[0], target.shape[0])
                out = out[:min_size]
                target = target[:min_size]
            
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * out.size(0)
            num_samples += out.size(0)
            batch_count += 1
            
            # Log a cada 20 batches
            if batch_count % 20 == 0:
                model_logger.info(f"Época {epoch+1}/{epochs}, Batch {batch_count}/{len(train_loader)}, "
                                f"Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / num_samples

        # Validação
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Selecionar a propriedade alvo específica
                target = batch.y[:, target_idx].view(-1, 1)
                
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                
                # Garantir que as dimensões correspondam
                if out.shape[0] != target.shape[0]:
                    min_size = min(out.shape[0], target.shape[0])
                    out = out[:min_size]
                    target = target[:min_size]
                
                loss = criterion(out, target)
                val_loss += loss.item() * out.size(0)
                val_samples += out.size(0)

        avg_val_loss = val_loss / val_samples
        
        # Calcular tempo da época
        epoch_time = time.time() - epoch_start

        # Log de progresso
        model_logger.info(f"[{epoch+1}/{epochs}] "
                         f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                         f"Tempo: {epoch_time:.2f}s")

        # Armazenar as médias das losses
        history.append((epoch + 1, avg_train_loss, avg_val_loss))

        # Verificar se é o melhor modelo até agora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path)
            model_logger.info(f"Melhor modelo salvo em: {path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= patience:
            model_logger.info(f"Early stopping ativado após {patience} épocas sem melhoria")
            break

    # Tempo total de treinamento
    total_train_time = time.time() - train_start_time
    model_logger.info(f"Treinamento concluído em {total_train_time:.2f} segundos "
                     f"({total_train_time/60:.2f} minutos)")

    # 6. Avaliação final
    model_logger.info("Iniciando avaliação no conjunto de teste...")
    test_start_time = time.time()
    
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        test_loss = 0
        test_samples = 0
        
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Selecionar a propriedade alvo específica
                target = batch.y[:, target_idx].view(-1, 1)
                
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                
                # Garantir que as dimensões correspondam
                if out.shape[0] != target.shape[0]:
                    min_size = min(out.shape[0], target.shape[0])
                    out = out[:min_size]
                    target = target[:min_size]
                
                # Coletar predições e valores verdadeiros para análise
                predictions.extend(out.cpu().numpy())
                true_values.extend(target.cpu().numpy())
                
                loss = F.mse_loss(out, target).item()
                test_loss += loss * out.size(0)
                test_samples += out.size(0)

        test_loss /= test_samples
        rmse = test_loss ** 0.5
        
        # Calcular métricas adicionais
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
        
        test_time = time.time() - test_start_time
        model_logger.info(f"Avaliação do teste concluída em {test_time:.2f} segundos")
        model_logger.info(f"MSE no teste: {test_loss:.4f}")
        model_logger.info(f"RMSE no teste: {rmse:.4f}")
        model_logger.info(f"MAE no teste: {mae:.4f}")
        
        # Salvar métricas em arquivo JSON
        metrics = {
            "model": model_name,
            "target_idx": target_idx,
            "train_loss": float(history[-1][1]),
            "val_loss": float(history[-1][2]),
            "test_loss": float(test_loss),
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "training_time_seconds": float(total_train_time),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metrics_file = os.path.join(log_dir, f"{model_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        model_logger.info(f"Métricas salvas em {metrics_file}")
        
    except Exception as e:
        model_logger.error(f"Erro durante avaliação final: {str(e)}")

    return history

def main():
    """
    Treina modelos MLP para cada tipo de descritor disponível em data.py
    """
    # Configuração de logging global
    log_dir = os.path.join(os.getcwd(), "logs")
    log_file = setup_logging(log_dir)
    
    logging.info("="*80)
    logging.info("INICIANDO TREINAMENTO DE MODELOS MLP COM DIFERENTES DESCRITORES")
    logging.info("="*80)
    
    # Criar sumário de treinamento
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(log_dir, f"training_summary_{timestamp}.txt")
    results_file = os.path.join(log_dir, f"training_results_{timestamp}.json")
    
    logging.info(f"Log global: {log_file}")
    logging.info(f"Arquivo de sumário: {summary_file}")
    logging.info(f"Arquivo de resultados: {results_file}")
    
    # Lista de todos os tipos de descritores disponíveis
    all_descriptors = [
        'Morgan', 'CM', 'Physicochemical', '3D', 'MACCS', 
        'Topological', 'AtomPair', 'EState', 'Pattern', 
        'Avalon', 'MorganCount', 'Autocorr'
    ]
    
    # Propriedades de interesse (índices de atributos)
    att_indices = [0, 10]  # Rotational constant A, Internal energy at 0K
    
    # Configurações de camadas para diferentes modelos
    layer_configs = {
        'small': [32, 16],
        'medium': [64, 32, 16],
        'large': [128, 64, 32]
    }
    
    # Parâmetros fixos
    epochs = 30
    n_noise = 0  # Sem ruído
    learning_rate = 1e-3
    batch_size = 32
    num_workers = 4
    
    # Armazenar configurações e resultados
    all_configs = []
    results = {}
    
    # Contadores para o relatório final
    successful_models = 0
    failed_models = 0
    
    logging.info("Parâmetros de treinamento:")
    logging.info(f"  Épocas: {epochs}")
    logging.info(f"  Learning Rate: {learning_rate}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Descritores: {all_descriptors}")
    logging.info(f"  Propriedades: {att_indices}")
    logging.info(f"  Arquiteturas: {layer_configs}")
    
    # Criar todas as combinações de configuração
    for att_index in att_indices:
        for descriptor_type in all_descriptors:
            for layer_name, layers in layer_configs.items():
                config = {
                    'name': f"MLP_att{att_index}_{descriptor_type}_{layer_name}",
                    'att_index': att_index,
                    'descriptor_type': descriptor_type,
                    'layers': layers,
                    'cache_descriptors': True
                }
                all_configs.append(config)
    
    total_models = len(all_configs)
    logging.info(f"Total de {total_models} modelos a serem treinados")
    
    # Treinar cada modelo
    start_time_all = time.time()
    for i, config in enumerate(all_configs, 1):
        model_name = config['name']
        att_index = config['att_index']
        descriptor_type = config['descriptor_type']
        layers = config['layers']
        
        logging.info(f"[{i}/{total_models}] Iniciando treino: {model_name}")
        logging.info(f"  Propriedade: {att_index}")
        logging.info(f"  Descritor: {descriptor_type}")
        logging.info(f"  Arquitetura: {layers}")
        
        # Treinar o modelo com tratamento de exceção
        try:
            model_start_time = time.time()
            history = train_mlp_qm9(
                att_index=att_index,
                descriptor_type=descriptor_type,
                epochs=epochs,
                layers=layers,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_noise=n_noise,
                cache_descriptors=config['cache_descriptors'],
                num_workers=num_workers,
                log_dir=log_dir
            )
            model_time = time.time() - model_start_time
            
            # Armazenar resultados
            results[model_name] = {
                'config': config,
                'history': history,
                'final_train_loss': history[-1][1] if history else None,
                'final_val_loss': history[-1][2] if history else None,
                'status': 'success',
                'time': model_time
            }
            
            logging.info(f"✓ {model_name} treinado com sucesso em {model_time:.2f}s!")
            if history:
                logging.info(f"  Loss final treino: {history[-1][1]:.4f}")
                logging.info(f"  Loss final validação: {history[-1][2]:.4f}")
            
            successful_models += 1
            
        except Exception as e:
            logging.error(f"✗ Erro no treinamento de {model_name}: {str(e)}")
            results[model_name] = {
                'config': config,
                'error': str(e),
                'status': 'failed'
            }
            failed_models += 1
        
        # Progresso atual
        progress = (i / total_models) * 100
        elapsed_time = time.time() - start_time_all
        est_total_time = elapsed_time * (total_models / i)
        est_remaining = est_total_time - elapsed_time
        
        logging.info(f"Progresso: {progress:.1f}% | Tempo decorrido: {elapsed_time/3600:.1f}h | "
                    f"Estimado restante: {est_remaining/3600:.1f}h")
        logging.info("-" * 80)
    
    # Tempo total de execução
    total_time = time.time() - start_time_all
    logging.info(f"Treinamento de todos os modelos concluído em {total_time/3600:.2f} horas")
    
    # Salvar resultados completos em formato JSON para análise posterior
    serializable_results = {}
    for name, result in results.items():
        serializable_results[name] = result.copy()
        if 'history' in result and result['history']:
            # Converter history para formato serializável
            serializable_results[name]['history'] = [
                (int(epoch), float(train_loss), float(val_loss)) 
                for epoch, train_loss, val_loss in result['history']
            ]
        if 'config' in result:
            # Garantir que a configuração seja serializável
            serializable_results[name]['config'] = {
                k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                for k, v in result['config'].items()
            }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Gerar relatório final
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RESUMO DO TREINAMENTO\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Data de execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de modelos: {total_models}\n")
        f.write(f"Sucessos: {successful_models}\n")
        f.write(f"Falhas: {failed_models}\n")
        f.write(f"Taxa de sucesso: {successful_models/total_models*100:.1f}%\n")
        f.write(f"Tempo total: {total_time/3600:.2f} horas\n\n")
        
        f.write("MELHORES MODELOS POR PROPRIEDADE E TIPO DE DESCRITOR:\n")
        f.write("-" * 80 + "\n")
        
        # Encontrar os melhores modelos por propriedade
        for att_index in att_indices:
            f.write(f"\nPROPRIEDADE {att_index}:\n")
            
            # Agrupar por tipo de descritor
            descriptor_best = {}
            for descriptor_type in all_descriptors:
                models = {name: result for name, result in results.items() 
                          if result['status'] == 'success' and 
                          result['config']['att_index'] == att_index and
                          result['config']['descriptor_type'] == descriptor_type}
                
                if models:
                    best = min(models.items(), key=lambda x: x[1]['final_val_loss'])
                    descriptor_best[descriptor_type] = best
            
            # Mostrar os melhores por descritor
            for desc_type, (model_name, model_info) in sorted(descriptor_best.items()):
                f.write(f"  {desc_type}: {model_name}\n")
                f.write(f"    Val Loss: {model_info['final_val_loss']:.4f}\n")
                f.write(f"    Arquitetura: {model_info['config']['layers']}\n")
            
            # Melhor modelo geral para esta propriedade
            if descriptor_best:
                overall_best = min(descriptor_best.values(), key=lambda x: x[1]['final_val_loss'])
                f.write(f"\n  Melhor geral: {overall_best[0]}\n")
                f.write(f"    Val Loss: {overall_best[1]['final_val_loss']:.4f}\n")
                f.write(f"    Descritor: {overall_best[1]['config']['descriptor_type']}\n")
                f.write(f"    Arquitetura: {overall_best[1]['config']['layers']}\n")
        
        f.write("\n\nMODELOS FALHOS:\n")
        for name, result in results.items():
            if result['status'] == 'failed':
                f.write(f"  - {name}: {result['error']}\n")
    
    logging.info(f"Relatório de resumo salvo em: {summary_file}")
    logging.info(f"Resultados detalhados salvos em: {results_file}")
    
    return results, summary_file

if __name__ == '__main__':
    # Executar o treinamento de todos os modelos com descritores
    results, summary_file = main()
    
    # Exibir caminho para o relatório de resumo para referência rápida
    print(f"\nTreinamento concluído! Relatório de resumo: {summary_file}")