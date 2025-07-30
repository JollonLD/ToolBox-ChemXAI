#!/usr/bin/env python3
# filepath: /home/jonas/Documents/ToolBox-ChemXAI/Testing/run_experiments.py
import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

from chemxai.train import train_mlp_qm9, train_gcn_qm9, train_gcn_pcqm4
from chemxai.data import qm9_tabular, graph_datasets
from chemxai.models import MLP, GCN
from chemxai.explainers import Shap, LIME, GNNExplain, GraphShap, GraphLIME, NodeGraphShap
from chemxai.evaluate import Evaluator
from chemxai.plots import radar_plot, horizontal_bar_plot

class ExperimentRunner:
    def __init__(self, config_file):
        """
        Inicializa o executor de experimentos com um arquivo de configuração
        
        Parameters:
        -----------
        config_file : str
            Caminho para o arquivo de configuração JSON
        """
        self.config = self.load_config(config_file)
        self.results_dir = self.setup_experiment_dir()
        self.models_dir = os.path.join(os.getcwd(), "models")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        
        # Criar diretórios necessários
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Arquivo de log principal
        self.log_file = os.path.join(self.logs_dir, "experiment_log.txt")
        
        # Registrar metadados do experimento
        self.log(f"Experimentos iniciados em: {datetime.now()}")
        self.log(f"Arquivo de configuração: {config_file}")
        self.log(f"Resultados salvos em: {self.results_dir}")
        self.log("-" * 80)
        
        # Dispositivo para execução (CPU/GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log(f"Dispositivo utilizado: {self.device}")
        
        # Dicionário para armazenar modelos treinados
        self.trained_models = {}

    def load_config(self, config_file):
        """Carrega arquivo de configuração JSON"""
        with open(config_file, 'r') as f:
            return json.load(f)

    def setup_experiment_dir(self):
        """Cria diretório para resultados do experimento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join("results", f"run_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def log(self, message, also_print=True):
        """
        Registra uma mensagem no arquivo de log
        
        Parameters:
        -----------
        message : str
            Mensagem a ser registrada
        also_print : bool
            Se True, também imprime a mensagem no console
        """
        if also_print:
            print(message)
            
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")

    def run_training_experiments(self):
        """Executa todos os experimentos de treinamento"""
        self.log("\n" + "=" * 80)
        self.log("INICIANDO EXPERIMENTOS DE TREINAMENTO")
        self.log("=" * 80)
        
        for exp in self.config.get('training_experiments', []):
            exp_id = exp['id']
            exp_type = exp['type']
            exp_desc = exp['description']
            params = exp['params']
            
            # Registrar início do experimento
            self.log("\n" + "-" * 80)
            self.log(f"Experimento ID: {exp_id}")
            self.log(f"Tipo: {exp_type}")
            self.log(f"Descrição: {exp_desc}")
            self.log(f"Parâmetros: {json.dumps(params, indent=2)}")
            self.log("-" * 80 + "\n")
            
            # Criar arquivo de log específico para este experimento
            exp_log_file = os.path.join(self.logs_dir, f"{exp_id}.txt")
            
            # Redirecionar saída para capturar métricas de treino/validação
            with open(exp_log_file, 'w') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    try:
                        # Medir tempo de execução
                        start_time = time.time()
                        
                        # Executar experimento com base no tipo
                        if exp_type == 'train_mlp_qm9':
                            model_path = os.path.join(self.models_dir, f"{exp_id}.pt")
                            params['model_save_path'] = model_path
                            history = train_mlp_qm9(**params)
                            self.trained_models[exp_id] = model_path
                            
                            # Plotar curvas de aprendizado
                            self.plot_training_history(history, exp_id)
                            
                        elif exp_type == 'train_gcn_qm9':
                            model_path = os.path.join(self.models_dir, f"{exp_id}.pt")
                            params['model_save_path'] = model_path
                            history = train_gcn_qm9(**params)
                            self.trained_models[exp_id] = model_path
                            
                            # Plotar curvas de aprendizado
                            self.plot_training_history(history, exp_id)
                            
                        elif exp_type == 'train_gcn_pcqm4':
                            model_path = os.path.join(self.models_dir, f"{exp_id}.pt")
                            params['model_save_path'] = model_path
                            test_loader, history = train_gcn_pcqm4(**params)
                            self.trained_models[exp_id] = model_path
                            
                            # Plotar curvas de aprendizado
                            self.plot_training_history(history, exp_id)
                            
                        else:
                            raise ValueError(f"Tipo de experimento desconhecido: {exp_type}")
                        
                        # Registrar tempo de execução
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                    except Exception as e:
                        # Registrar erro
                        self.log(f"Erro ao executar experimento {exp_id}: {str(e)}")
                        import traceback
                        self.log(traceback.format_exc())
                        continue
            
            # Ler o arquivo de log específico do experimento para exibir métricas
            with open(exp_log_file, 'r') as f:
                exp_log = f.read()
                
            # Extrair métricas de treino e validação para log principal
            metrics_summary = self.extract_metrics(exp_log, exp_type)
            self.log(metrics_summary)
            self.log(f"Tempo de execução: {elapsed_time:.2f} segundos")
            self.log(f"Modelo salvo em: {model_path}")
            self.log(f"Log completo em: {exp_log_file}")

    def extract_metrics(self, log_text, exp_type):
        """
        Extrai métricas de treino e validação do texto de log
        
        Parameters:
        -----------
        log_text : str
            Texto de log completo
        exp_type : str
            Tipo de experimento
            
        Returns:
        --------
        str : Resumo das métricas
        """
        lines = log_text.split('\n')
        metrics_lines = []
        
        if 'mlp' in exp_type or 'gcn' in exp_type:
            # Extrair última linha de treino/validação
            for line in lines:
                if '] Train Loss:' in line:
                    metrics_lines.append(line)
                    
            # Extrair métricas finais de teste
            for line in lines:
                if 'MSE no teste:' in line or 'RMSE no teste:' in line:
                    metrics_lines.append(line)
        
        return "\n".join(metrics_lines[-5:])  # Últimas 5 métricas relevantes

    def plot_training_history(self, history, exp_id):
        """
        Plota e salva curvas de aprendizado para um experimento
        
        Parameters:
        -----------
        history : list
            Lista de tuplas (epoch, train_loss, val_loss)
        exp_id : str
            ID do experimento
        """
        if not history:
            return  # Sair se não há histórico
        
        # Extrair dados do histórico
        epochs = [x[0] for x in history]
        train_losses = [x[1] for x in history]
        val_losses = [x[2] for x in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Treino')
        plt.plot(epochs, val_losses, 'r-', label='Validação')
        plt.title(f'Curva de Aprendizado - {exp_id}')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Salvar figura
        curves_dir = os.path.join(self.results_dir, 'learning_curves')
        os.makedirs(curves_dir, exist_ok=True)
        fig_path = os.path.join(curves_dir, f"{exp_id}_learning_curve.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Curva de aprendizado salva em: {fig_path}")

    def run_explanation_experiments(self):
        """Executa todos os experimentos de explicação"""
        self.log("\n" + "=" * 80)
        self.log("INICIANDO EXPERIMENTOS DE EXPLICAÇÃO")
        self.log("=" * 80)
        
        for exp in self.config.get('explanation_experiments', []):
            exp_id = exp['id']
            exp_type = exp['type']
            exp_desc = exp['description']
            params = exp['params']
            
            # Verificar se o modelo associado foi treinado
            model_name = params.get('model_name')
            if model_name is None:
                self.log(f"AVISO: Parâmetro 'model_name' não passado para explicação {exp_id}. Pulando.")
                continue

            # Registrar início do experimento
            self.log("\n" + "-" * 80)
            self.log(f"Explicação ID: {exp_id}")
            self.log(f"Tipo: {exp_type}")
            self.log(f"Descrição: {exp_desc}")
            self.log(f"Parâmetros: {json.dumps(params, indent=2)}")
            self.log("-" * 80 + "\n")
            
            # Criar arquivo de log específico para esta explicação
            exp_log_file = os.path.join(self.logs_dir, f"{exp_id}.txt")
            
            # Redirecionar saída para capturar informações da explicação
            with open(exp_log_file, 'w') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    try:
                        # Medir tempo de execução
                        start_time = time.time()
                        
                        # Carregar modelo treinado
                        model_path = None
                        if model_path in self.trained_models:
                            model_path = self.trained_models[model_name]
                        else:
                            model_path = os.path.join(self.models_dir, f"{model_name}.pth")

                        model = self.load_model(params, model_name, model_path)
                        
                        # Executar explicação com base no tipo
                        if exp_type == 'shap':
                            explanation = self.run_shap_explainer(model, params)
                            
                        elif exp_type == 'lime':
                            explanation = self.run_lime_explainer(model, params)
                            
                        elif exp_type == 'gnnexplain':
                            explanation = self.run_gnn_explainer(model, params)
                            
                        elif exp_type == 'graphshap':
                            explanation = self.run_graphshap_explainer(model, params)
                            
                        elif exp_type == 'graphlime':
                            explanation = self.run_graphlime_explainer(model, params)
                            
                        else:
                            raise ValueError(f"Tipo de explicação desconhecido: {exp_type}")
                        
                        # Salvar resultados da explicação
                        explanation_dir = os.path.join(self.results_dir, 'explanations')
                        os.makedirs(explanation_dir, exist_ok=True)
                        explanation_path = os.path.join(explanation_dir, f"{exp_id}.json")
                        
                        with open(explanation_path, 'w') as ef:
                            json.dump(explanation, ef, indent=2)
                        
                        # Visualizar explicação
                        if 'feature_names' in params:
                            self.visualize_explanation(
                                explanation,
                                exp_id,
                                exp_type,
                                params.get('feature_names')
                            )
                        
                        # Registrar tempo de execução
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                    except Exception as e:
                        # Registrar erro
                        self.log(f"Erro ao executar explicação {exp_id}: {str(e)}")
                        import traceback
                        self.log(traceback.format_exc())
                        continue
            
            # Ler o arquivo de log específico da explicação para exibir informações
            with open(exp_log_file, 'r') as f:
                exp_log = f.read()
                
            # Extrair informações relevantes para log principal
            info_summary = self.extract_explanation_info(exp_log, exp_type)
            self.log(info_summary)
            self.log(f"Tempo de execução: {elapsed_time:.2f} segundos")
            self.log(f"Explicação salva em: {explanation_path}")
            self.log(f"Log completo em: {exp_log_file}")

    def load_model(self, params, model_name, model_path):
        """
        Carrega um modelo treinado a partir do caminho salvo
        
        Parameters:
        -----------
        model_name : str
            Nome do modelo a ser carregado
        model_path : str
            Caminho para o arquivo do modelo
            
        Returns:
        --------
        model : Modelo PyTorch carregado
        """
        # Determinar tipo de modelo com base no nome
        model_type = None
        if 'mlp' in model_name.lower():
            model_type = 'mlp'
        elif 'gcn' in model_name.lower():
            model_type = 'gcn'
        else:
            raise ValueError(f"Tipo de modelo não reconhecido para {model_name}")
            
        # Carregar o modelo
        if model_type == 'mlp':
            # Obter dimensões do primeiro batch de dados
            qm9 = qm9_tabular()
            if 'noise' in model_name.lower():
                train_loader, _, _, _, _, _, _ = qm9.get_paired_dataloaders(
                    descriptor_type=params.get('descriptor_type', 'Physicochemical'),
                    n_noise=2  # Valor padrão para exemplos com ruído
                )
            else:
                train_loader, _, _ = qm9.get_paired_dataloaders(
                    descriptor_type=params.get('descriptor_type', 'Physicochemical'),
                    n_noise=0
                )
            
            input_dim = next(iter(train_loader))[0].shape[1]
            output_dim = 1
            
            # Instanciar e carregar modelo
            model = MLP(input_dim=input_dim, output_dim=output_dim, layers=[128, 64], device=self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
        elif model_type == 'gcn':
            # Obter dados para determinar dimensões do modelo
            gd = graph_datasets()
            if 'pcqm4' in model_name.lower():
                data = gd.prepare_data_graph('PCQM4')
            else:
                data = gd.prepare_data_graph('QM9')
            
            # Obter dimensões do primeiro grafo
            first_graph = data[0]
            num_features = first_graph.x.size(1)
            
            # Instanciar e carregar modelo
            model = GCN(num_features=num_features)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
        return model

    def run_shap_explainer(self, model, params):
        """
        Executa o explainer SHAP
        
        Parameters:
        -----------
        model : torch.nn.Module
            Modelo treinado
        params : dict
            Parâmetros para o explainer
            
        Returns:
        --------
        dict : Resultados da explicação
        """
        # Obter dados para explicação
        qm9 = qm9_tabular()
        
        if 'noise' in params.get('model_name', '').lower():
            train_loader, _, test_loader, _, _, _, is_noise = qm9.get_paired_dataloaders(
                descriptor_type=params.get('descriptor_type', 'Physicochemical'),
                n_noise=2,
                batch_size=params.get('batch_size', 32)
            )
        else:
            train_loader, _, test_loader = qm9.get_paired_dataloaders(
                descriptor_type=params.get('descriptor_type', 'Physicochemical'),
                n_noise=0,
                batch_size=params.get('batch_size', 32)
            )
            is_noise = None
            
        # Obter amostras para explicação
        background = next(iter(train_loader))[0]
        test_tensor = next(iter(test_loader))[0]
        
        # Criar explainer SHAP
        explainer = Shap(model=model, background_tensor=background, 
                         test_tensor=test_tensor, device=self.device)
        
        # Gerar explicação
        explanation_type = params.get('explanation_type', 'global')
        
        if explanation_type == 'global':
            shap_values = explainer.explain_global()
            result = {
                'shap_values': shap_values,
                'explanation_type': 'global'
            }
        else:  # local
            index = params.get('index', 0)
            shap_values = explainer.explain_local(index=index)
            result = {
                'shap_values': shap_values,
                'explanation_type': 'local',
                'index': index
            }
        
        # Adicionar informação sobre ruído se disponível
        if is_noise is not None:
            result['is_noise'] = is_noise.tolist()
            
        return result

    def run_lime_explainer(self, model, params):
        """Executa o explainer LIME"""
        # Obter dados para explicação
        qm9 = qm9_tabular()
        
        if 'noise' in params.get('model_name', '').lower():
            train_loader, _, test_loader, _, _, _, is_noise = qm9.get_paired_dataloaders(
                descriptor_type=params.get('descriptor_type', 'Physicochemical'),
                n_noise=2,
                batch_size=params.get('batch_size', 32)
            )
        else:
            train_loader, _, test_loader = qm9.get_paired_dataloaders(
                descriptor_type=params.get('descriptor_type', 'Physicochemical'),
                n_noise=0,
                batch_size=params.get('batch_size', 32)
            )
            is_noise = None
            
        # Obter amostras para explicação
        background = next(iter(train_loader))[0]
        test_tensor = next(iter(test_loader))[0]
        
        # Criar explainer LIME
        explainer = LIME(model=model, background_tensor=background, 
                         test_tensor=test_tensor, device=self.device)
        
        # Gerar explicação
        index = params.get('index', 0)
        num_features = params.get('num_features', None)
        
        lime_values = explainer.explain_local(index=index, num_features=num_features)
        
        result = {
            'lime_values': lime_values,
            'index': index
        }
        
        # Adicionar informação sobre ruído se disponível
        if is_noise is not None:
            result['is_noise'] = is_noise.tolist()
            
        return result
    

# acrescentar parametro para indicar qual dado é (QM9 ou PCQM4)
    def run_gnn_explainer(self, model, params):
        """Executa o explainer GNNExplainer"""
        # Obter dados para explicação
        gd = graph_datasets()
        
        if 'pcqm4' in params.get('model_name', '').lower():
            data = gd.prepare_data_graph('PCQM4')
        else:
            data = gd.prepare_data_graph('QM9')
            
        # Obter amostra para explicação
        index = params.get('index', 0)
        graph_sample = data[index]
        
        # Criar explainer
        explainer = GNNExplain(
            model=model, 
            device=self.device,
            data=graph_sample,
            epochs=params.get('epochs', 100),
            mode='regression',
            task_level='graph',
            return_type='raw'
        )
        
        # Gerar explicação
        node_mask, edge_mask, _ = explainer.explain(index=None)  # None para grafo inteiro
        
        result = {
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'graph_index': index
        }
        
        return result

    def run_graphshap_explainer(self, model, params):
        """Executa o explainer GraphShap"""
        # Obter dados para explicação
        gd = graph_datasets()
        
        if 'pcqm4' in params.get('model_name', '').lower():
            data = gd.prepare_data_graph('PCQM4')
        else:
            data = gd.prepare_data_graph('QM9')
            
        # Obter amostra para explicação
        index = params.get('index', 0)
        graph_sample = data[index]
        
        # Criar explainer
        explainer = GraphShap(
            data=graph_sample,
            model=model,
            device=self.device
        )
        
        # Gerar explicação
        num_samples = params.get('num_samples', 30)
        feature_importance = explainer.explain(num_samples=num_samples)
        
        result = {
            'feature_importance': feature_importance,
            'graph_index': index
        }
        
        return result

    def run_graphlime_explainer(self, model, params):
        """Executa o explainer GraphLIME"""
        # Obter dados para explicação
        gd = graph_datasets()
        
        if 'pcqm4' in params.get('model_name', '').lower():
            data = gd.prepare_data_graph('PCQM4')
        else:
            data = gd.prepare_data_graph('QM9')
            
        # Obter amostra para explicação
        index = params.get('index', 0)
        graph_sample = data[index]
        
        # Criar explainer
        explainer = GraphLIME(
            model=model,
            device=self.device
        )
        
        # Gerar explicação
        num_samples = params.get('num_samples', 100)
        feature_importance = explainer.explain(data=graph_sample, num_samples=num_samples)
        
        result = {
            'feature_importance': feature_importance,
            'graph_index': index
        }
        
        return result

    def extract_explanation_info(self, log_text, exp_type):
        """
        Extrai informações relevantes do log da explicação
        
        Parameters:
        -----------
        log_text : str
            Texto de log completo
        exp_type : str
            Tipo de explicação
            
        Returns:
        --------
        str : Resumo das informações
        """
        lines = log_text.split('\n')
        info_lines = []
        
        # Extrair informações relevantes com base no tipo de explicação
        if exp_type == 'shap':
            for line in lines:
                if 'Background shape:' in line or 'Test data shape:' in line:
                    info_lines.append(line)
        elif exp_type == 'lime':
            for line in lines:
                if 'Background shape:' in line or 'Test data shape:' in line:
                    info_lines.append(line)
        elif exp_type in ['gnnexplain', 'graphshap', 'graphlime']:
            # Extrair informações sobre o grafo
            for line in lines:
                if 'nodes' in line.lower() or 'edge' in line.lower() or 'feature' in line.lower():
                    info_lines.append(line)
        
        if not info_lines:
            # Se nenhuma informação específica foi encontrada, pegar algumas linhas gerais
            info_lines = lines[:min(5, len(lines))]
            
        return "\n".join(info_lines)

    def visualize_explanation(self, explanation, exp_id, exp_type, feature_names=None):
        """
        Visualiza e salva os resultados da explicação
        
        Parameters:
        -----------
        explanation : dict
            Resultados da explicação
        exp_id : str
            ID do experimento
        exp_type : str
            Tipo de explicação
        feature_names : list
            Nomes das features (opcional)
        """
        vis_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        if exp_type == 'shap':
            if 'shap_values' not in explanation:
                return
                
            shap_values = np.array(explanation['shap_values'])
            
            # Gráfico de barras horizontais
            fig, ax = horizontal_bar_plot(
                values=shap_values,
                feature_names=feature_names,
                title=f"SHAP Feature Importance - {exp_id}",
                sort=True,
                max_features=20,
                save_path=vis_dir,
                filename=f"{exp_id}_shap_bars.png"
            )
            
            # Gráfico de radar para valores absolutos
            abs_values = np.abs(shap_values)
            if len(abs_values) <= 10:  # Radar plot é bom para poucas features
                fig, ax = radar_plot(
                    values=abs_values,
                    feature_names=feature_names,
                    title=f"SHAP Feature Importance - {exp_id}"
                )
                plt.savefig(os.path.join(vis_dir, f"{exp_id}_shap_radar.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
        elif exp_type == 'lime':
            if 'lime_values' not in explanation:
                return
                
            lime_values = np.array(explanation['lime_values'])
            
            # Gráfico de barras horizontais
            fig, ax = horizontal_bar_plot(
                values=lime_values,
                feature_names=feature_names,
                title=f"LIME Feature Importance - {exp_id}",
                sort=True,
                max_features=20,
                save_path=vis_dir,
                filename=f"{exp_id}_lime_bars.png"
            )
            
        elif exp_type == 'gnnexplain':
            if 'node_mask' not in explanation:
                return
                
            node_mask = np.array(explanation['node_mask'])
            
            # Gráfico de barras horizontais para importância dos nós
            fig, ax = horizontal_bar_plot(
                values=node_mask,
                feature_names=[f"Node {i}" for i in range(len(node_mask))],
                title=f"GNNExplain Node Importance - {exp_id}",
                sort=True,
                max_features=20,
                save_path=vis_dir,
                filename=f"{exp_id}_gnnexplain_nodes.png"
            )
            
            if 'edge_mask' in explanation:
                edge_mask = np.array(explanation['edge_mask'])
                
                # Gráfico de barras horizontais para importância das arestas
                fig, ax = horizontal_bar_plot(
                    values=edge_mask,
                    feature_names=[f"Edge {i}" for i in range(len(edge_mask))],
                    title=f"GNNExplain Edge Importance - {exp_id}",
                    sort=True,
                    max_features=20,
                    save_path=vis_dir,
                    filename=f"{exp_id}_gnnexplain_edges.png"
                )
            
        elif exp_type == 'graphshap' or exp_type == 'graphlime':
            key = 'feature_importance'
            if key not in explanation:
                return
                
            importance = np.array(explanation[key])
            
            # Gráfico de barras horizontais
            fig, ax = horizontal_bar_plot(
                values=importance,
                feature_names=[f"Feature {i}" for i in range(len(importance))],
                title=f"{exp_type.capitalize()} Feature Importance - {exp_id}",
                sort=True,
                max_features=20,
                save_path=vis_dir,
                filename=f"{exp_id}_{exp_type}_importance.png"
            )

    def run_evaluation_experiments(self):
        """Executa todos os experimentos de avaliação"""
        self.log("\n" + "=" * 80)
        self.log("INICIANDO EXPERIMENTOS DE AVALIAÇÃO")
        self.log("=" * 80)
        
        for exp in self.config.get('evaluation_experiments', []):
            exp_id = exp['id']
            exp_type = exp['type']
            exp_desc = exp['description']
            params = exp['params']
            
            # Registrar início do experimento
            self.log("\n" + "-" * 80)
            self.log(f"Avaliação ID: {exp_id}")
            self.log(f"Tipo: {exp_type}")
            self.log(f"Descrição: {exp_desc}")
            self.log(f"Parâmetros: {json.dumps(params, indent=2)}")
            self.log("-" * 80 + "\n")
            
            # Verificar se os modelos associados foram treinados
            model_normal_name = params.get('model_normal')
            model_noise_name = params.get('model_noise')
            
            if model_normal_name not in self.trained_models:
                self.log(f"AVISO: Modelo normal {model_normal_name} não encontrado. Pulando avaliação.")
                continue
                
            if model_noise_name not in self.trained_models:
                self.log(f"AVISO: Modelo com ruído {model_noise_name} não encontrado. Pulando avaliação.")
                continue
            
            # Criar arquivo de log específico para esta avaliação
            eval_log_file = os.path.join(self.logs_dir, f"{exp_id}.txt")
            
            # Redirecionar saída para capturar informações da avaliação
            with open(eval_log_file, 'w') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    try:
                        # Medir tempo de execução
                        start_time = time.time()
                        
                        # Carregar modelos treinados
                        model_normal_path = self.trained_models[model_normal_name]
                        model_normal = self.load_model(params, model_normal_name, model_normal_path)
                        
                        model_noise_path = self.trained_models[model_noise_name]
                        model_noise = self.load_model(params, model_noise_name, model_noise_path)
                        
                        # Executar avaliação com base no tipo
                        if exp_type == 'robustness':
                            metrics, figures = self.evaluate_robustness(
                                model_normal, 
                                model_noise, 
                                params
                            )
                            
                            # Salvar figuras geradas
                            eval_dir = os.path.join(self.results_dir, 'evaluations', exp_id)
                            os.makedirs(eval_dir, exist_ok=True)
                            
                            # Salvar métricas
                            metrics_path = os.path.join(eval_dir, f"{exp_id}_metrics.json")
                            with open(metrics_path, 'w') as mf:
                                json.dump(metrics, mf, indent=2)
                            
                        else:
                            raise ValueError(f"Tipo de avaliação desconhecido: {exp_type}")
                        
                        # Registrar tempo de execução
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                    except Exception as e:
                        # Registrar erro
                        self.log(f"Erro ao executar avaliação {exp_id}: {str(e)}")
                        import traceback
                        self.log(traceback.format_exc())
                        continue
            
            # Ler o arquivo de log específico da avaliação para exibir informações
            with open(eval_log_file, 'r') as f:
                eval_log = f.read()
                
            # Extrair informações relevantes para log principal
            metrics_summary = self.extract_evaluation_metrics(eval_log, metrics)
            self.log(metrics_summary)
            self.log(f"Tempo de execução: {elapsed_time:.2f} segundos")
            self.log(f"Métricas salvas em: {metrics_path}")
            self.log(f"Log completo em: {eval_log_file}")

    def evaluate_robustness(self, model_normal, model_noise, params):
        """
        Avalia a robustez de explainers
        
        Parameters:
        -----------
        model_normal : torch.nn.Module
            Modelo treinado sem ruído
        model_noise : torch.nn.Module
            Modelo treinado com ruído
        params : dict
            Parâmetros para a avaliação
            
        Returns:
        --------
        tuple : (métricas, figuras)
        """
        # Configurar parâmetros da avaliação
        model_type = params.get('model_type', 'tabular')
        explainer_type = params.get('explainer_type', 'shap_local')
        
        # Obter dados para avaliação
        if model_type == 'tabular':
            qm9 = qm9_tabular()
            
            train_loader_normal, test_loader_normal, _ = qm9.get_paired_dataloaders(
                descriptor_type='Physicochemical',
                n_noise=0,
                batch_size=params.get('batch_size', 32)
            )
            
            _, _, _, train_loader_noise, test_loader_noise, _, _ = qm9.get_paired_dataloaders(
                descriptor_type='Physicochemical',
                n_noise=params.get('n_noise', 2),
                batch_size=params.get('batch_size', 32)
            )
            
        elif model_type == 'graph':
            gd = graph_datasets()
            
            train_loader_normal, _, test_loader_normal, train_loader_noise, _, test_loader_noise = gd.get_paired_dataloaders(
                dataset_name=params.get('dataset_name', 'QM9'),
                batch_size=params.get('batch_size', 32),
                noise_type=params.get('noise_type', 'gaussian'),
                noise_scale=params.get('noise_scale', 1.0)
            )
            
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
            
        # Criar avaliador
        evaluator = Evaluator(
            model_normal=model_normal,
            model_noise=model_noise,
            train_loader_normal=train_loader_normal,
            test_loader_normal=test_loader_normal,
            train_loader_noise=train_loader_noise,
            test_loader_noise=test_loader_noise,
            device=self.device,
            model_type=model_type,
            explainer_type=explainer_type,
            mol_index=params.get('mol_index', 0),
            atom_index=params.get('atom_index', 0)
        )
        
        # Executar avaliação de robustez
        similarities, l1_differences, l2_differences, spearman_correlations, figures = evaluator.robustness()
        
        # Converter para tipos serializáveis
        metrics = {
            'similarities': [float(s) for s in similarities],
            'l1_differences': [float(d) for d in l1_differences],
            'l2_differences': [float(d) for d in l2_differences],
            'spearman_correlations': [float(c) for c in spearman_correlations],
            'mean_similarity': float(np.mean(similarities)),
            'mean_l1_diff': float(np.mean(l1_differences)),
            'mean_l2_diff': float(np.mean(l2_differences)),
            'mean_spearman': float(np.mean(spearman_correlations))
        }
        
        return metrics, figures

    def extract_evaluation_metrics(self, log_text, metrics):
        """
        Extrai métricas relevantes do log da avaliação
        
        Parameters:
        -----------
        log_text : str
            Texto de log completo
        metrics : dict
            Métricas calculadas
            
        Returns:
        --------
        str : Resumo das métricas
        """
        summary = []
        
        # Adicionar métricas gerais
        if 'mean_similarity' in metrics:
            summary.append(f"Similaridade média: {metrics['mean_similarity']:.4f}")
            
        if 'mean_l1_diff' in metrics:
            summary.append(f"Diferença L1 média: {metrics['mean_l1_diff']:.4f}")
            
        if 'mean_l2_diff' in metrics:
            summary.append(f"Diferença L2 média: {metrics['mean_l2_diff']:.4f}")
            
        if 'mean_spearman' in metrics:
            summary.append(f"Correlação Spearman média: {metrics['mean_spearman']:.4f}")
            
        return "\n".join(summary)

    def run_all(self):
        """Executa todos os experimentos em sequência"""
        self.run_training_experiments()
        self.run_explanation_experiments()
        self.run_evaluation_experiments()
        self.log("\nTodos os experimentos foram concluídos com sucesso!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Executa experimentos de ChemXAI")
    parser.add_argument('--config', type=str, default='experiments.json', help='Arquivo de configuração JSON')
    parser.add_argument('--training', action='store_true', help='Executar apenas experimentos de treinamento')
    parser.add_argument('--explanation', action='store_true', help='Executar apenas experimentos de explicação')
    parser.add_argument('--evaluation', action='store_true', help='Executar apenas experimentos de avaliação')
    
    args = parser.parse_args()
    
    # Inicializar executor de experimentos
    runner = ExperimentRunner(args.config)
    
    # Executar experimentos com base nos argumentos
    if args.training:
        runner.run_training_experiments()
    elif args.explanation:
        runner.run_explanation_experiments()
    elif args.evaluation:
        runner.run_evaluation_experiments()
    else:
        runner.run_all()


if __name__ == '__main__':
    main()