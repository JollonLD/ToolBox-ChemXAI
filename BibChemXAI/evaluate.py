import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import contextlib
import io
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from contextlib import redirect_stdout
from IPython.display import display, HTML

from chemxai.explainers import Shap, LIME, GNNExplain, GraphShap, GraphLIME, NodeGraphShap, NodeGrapLIME
from chemxai.data import qm9_tabular


class Evaluator:
    def __init__(self, model_normal, model_noise, train_loader_normal, test_loader_normal, train_loader_noise, 
                 test_loader_noise, device, model_type='graph', explainer_type='shap_local', mol_index=0, atom_index=0):
        self.model_normal = model_normal
        self.model_noise = model_noise
        self.train_loader_normal = train_loader_normal
        self.test_loader_normal = test_loader_normal
        self.train_loader_noise = train_loader_noise
        self.test_loader_noise = test_loader_noise
        self.device = device
        self.model_type = model_type
        self.explainer_type = explainer_type

    def robustness(self):
        dirname = os.getcwd()
        graphs_dir = os.path.join(dirname, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        print(f'Diretório criado: {graphs_dir}')

        similarities = []
        l1_differences = []
        l2_differences = []
        spearman_correlations = []

        for batch_idx, (batch_train, batch_test, batch_train_noise, batch_test_noise) in enumerate(
            zip(self.train_loader_normal, self.test_loader_normal, 
                self.train_loader_noise, self.test_loader_noise)):
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                if self.model_type == 'tabular':
                    background = batch_train[0]
                    test_tensor = batch_test[0]
                    background_noise = batch_train_noise[0]
                    test_tensor_noise = batch_test_noise[0]
                    
                    # Para análise local, iterar sobre todas as moléculas do batch
                    if self.explainer_type in ['shap_local', 'lime']:
                        batch_size = len(test_tensor)
                        
                        if self.explainer_type == 'shap_local' or self.explainer_type == 'shap_global':

                            explainer = Shap(model=self.model_normal, background_tensor=background, 
                                            test_tensor=test_tensor, device=self.device)
                            explainer_noise = Shap(model=self.model_noise, background_tensor=background_noise, 
                                                test_tensor=test_tensor_noise, device=self.device)
                        
                        # Analisar cada molécula do batch
                        for idx in range(batch_size):
                            if self.explainer_type == 'shap_local':
                                explanation_without_noise = explainer.explain_local(index=idx)
                                explanation_with_noise = explainer_noise.explain_local(index=idx)
                            elif self.explainer_type == 'lime':
                                explainer_lime = LIME(model=self.model_normal, background_tensor=background, 
                                                test_tensor=test_tensor, device=self.device)
                                explainer_lime_noise = LIME(model=self.model_noise, background_tensor=background_noise, 
                                                        test_tensor=test_tensor_noise, device=self.device)
                                explanation_without_noise = explainer_lime.explain_local(index=idx)
                                explanation_with_noise = explainer_lime_noise.explain_local(index=idx)
                            
                            # Calcular métricas para esta molécula individual
                            self._calculate_metrics(explanation_without_noise, explanation_with_noise, 
                                                similarities, l1_differences, l2_differences, spearman_correlations)
                    
                    # Para análise global, uma explicação por batch
                    elif self.explainer_type == 'shap_global':
                        explainer = Shap(model=self.model_normal, background_tensor=background, 
                                        test_tensor=test_tensor, device=self.device)
                        explainer_noise = Shap(model=self.model_noise, background_tensor=background_noise, 
                                            test_tensor=test_tensor_noise, device=self.device)
                        explanation_without_noise = explainer.explain_global()
                        explanation_with_noise = explainer_noise.explain_global()
                        
                        # Calcular métricas para o batch inteiro
                        self._calculate_metrics(explanation_without_noise, explanation_with_noise, 
                                            similarities, l1_differences, l2_differences, spearman_correlations)
                
                elif self.model_type == 'graph':
                    # Para grafos, analisar cada molécula individualmente
                    if self.explainer_type == 'gnn_explainer':
                        # Calcular para cada molécula do batch
                        batch_size = len(batch_test)
                        for idx in range(batch_size):
                            data_normal = batch_test[idx]
                            data_noise = batch_test_noise[idx]
                            
                            # Criar explainers uma vez por molécula para explicar a molécula inteira
                            explainer = GNNExplain(model=self.model_normal, data=data_normal, 
                                                device=self.device, epochs=100, mode='regression', 
                                                task_level='graph', return_type='raw')
                            explainer_noise = GNNExplain(model=self.model_noise, data=data_noise, 
                                                        device=self.device, epochs=100, mode='regression', 
                                                        task_level='graph', return_type='raw')
                            
                            # Explicar a molécula inteira
                            explanation_without_noise, _ = explainer.explain()
                            explanation_with_noise, _ = explainer_noise.explain()
                            
                            # Calcular métricas para esta molécula
                            self._calculate_metrics(explanation_without_noise, explanation_with_noise,
                                                similarities, l1_differences, l2_differences, 
                                                spearman_correlations)
                    
                    elif self.explainer_type == 'graph_shap':
                        # GraphShap para explainer de grafo inteiro
                        batch_size = len(batch_test)
                        for idx in range(batch_size):
                            data_normal = batch_test[idx]
                            data_noise = batch_test_noise[idx]
                            
                            # Criar explainers GraphShap
                            explainer = GraphShap(data=data_normal, model=self.model_normal, 
                                                device=self.device)
                            explainer_noise = GraphShap(data=data_noise, model=self.model_noise, 
                                                    device=self.device)
                            
                            # Obter explicações
                            explanation_without_noise = explainer.explain(num_samples=30)
                            explanation_with_noise = explainer_noise.explain(num_samples=30)
                            
                            # Calcular métricas para esta molécula
                            self._calculate_metrics(explanation_without_noise, explanation_with_noise,
                                                similarities, l1_differences, l2_differences, 
                                                spearman_correlations)
                    
                    elif self.explainer_type == 'graph_lime':
                        # GraphLIME para explainer de grafo inteiro
                        batch_size = len(batch_test)
                        for idx in range(batch_size):
                            data_normal = batch_test[idx]
                            data_noise = batch_test_noise[idx]
                            
                            # Criar explainers GraphLIME
                            explainer = GraphLIME(model=self.model_normal, device=self.device)
                            explainer_noise = GraphLIME(model=self.model_noise, device=self.device)
                            
                            # Obter explicações
                            explanation_without_noise = explainer.explain(data=data_normal)
                            explanation_with_noise = explainer_noise.explain(data=data_noise)
                            
                            # Calcular métricas para esta molécula
                            self._calculate_metrics(explanation_without_noise, explanation_with_noise,
                                                similarities, l1_differences, l2_differences, 
                                                spearman_correlations)
                    
                    elif self.explainer_type == 'node_graph_shap':
                        # NodeGraphShap para explicar nós específicos
                        batch_size = len(batch_test)
                        for idx in range(batch_size):
                            data_normal = batch_test[idx]
                            data_noise = batch_test_noise[idx]
                            
                            # Criar explainers NodeGraphShap
                            explainer = NodeGraphShap(data=data_normal, model=self.model_normal, 
                                                    device=self.device)
                            explainer_noise = NodeGraphShap(data=data_noise, model=self.model_noise, 
                                                        device=self.device)
                            
                            # Explicar nó específico (se self.atom_index estiver definido, caso contrário usa 0)
                            node_idx = getattr(self, 'atom_index', 0)
                            
                            # Obter explicações
                            explanation_without_noise = explainer.explain(node_index=node_idx, hops=2, num_samples=30)
                            explanation_with_noise = explainer_noise.explain(node_index=node_idx, hops=2, num_samples=30)
                            
                            # Calcular métricas para esta molécula
                            self._calculate_metrics(explanation_without_noise, explanation_with_noise,
                                                similarities, l1_differences, l2_differences, 
                                                spearman_correlations)
                    
                    elif self.explainer_type == 'node_graph_lime':
                        # NodeGraphLIME para explicar nós específicos
                        batch_size = len(batch_test)
                        for idx in range(batch_size):
                            data_normal = batch_test[idx]
                            data_noise = batch_test_noise[idx]
                            
                            # Criar explainers NodeGraphLIME
                            explainer = NodeGrapLIME(data=data_normal, model=self.model_normal, 
                                                    device=self.device)
                            explainer_noise = NodeGrapLIME(data=data_noise, model=self.model_noise, 
                                                        device=self.device)
                            
                            # Explicar nó específico (se self.atom_index estiver definido, caso contrário usa 0)
                            node_idx = getattr(self, 'atom_index', 0)
                            
                            # Obter explicações
                            explanation_without_noise = explainer.explain(node_index=node_idx, hops=2, num_samples=30)
                            explanation_with_noise = explainer_noise.explain(node_index=node_idx, hops=2, num_samples=30)
                            
                            # Calcular métricas para esta molécula
                            self._calculate_metrics(explanation_without_noise, explanation_with_noise,
                                                similarities, l1_differences, l2_differences, 
                                                spearman_correlations)
                            
        

        fig_cos, ax = plt.subplots(figsize=(6, 4))
        ax.hist(similarities, bins=40, edgecolor='black', color='lightcoral')
        ax.set_title('Distribuição da Cosine Similarity')
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Frequency')
        fig_cos.savefig(graphs_dir + '/similarity_cosine.png')

        fig_l1, ax = plt.subplots(figsize=(6, 4))
        ax.hist(l1_differences, bins=40, edgecolor='black', color='lightcoral')
        ax.set_title('Distribuição das Diferenças')
        ax.set_xlabel('Difference')
        ax.set_ylabel('Frequency')
        fig_l1.savefig(graphs_dir + '/l1_differences.png')
        
        fig_spearman, ax = plt.subplots(figsize=(6, 4))
        ax.hist(spearman_correlations, bins=40, edgecolor='black', color='lightcoral')
        ax.set_title('Distribuição das Correlações Spearman')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Frequency')
        fig_spearman.savefig(graphs_dir + '/spearman_corr.png')
        
        figs = [fig_cos, fig_l1, fig_spearman]

        return similarities, l1_differences, l2_differences, spearman_correlations, figs

    def _calculate_metrics(self, explanation_without_noise, explanation_with_noise, 
                      similarities, l1_differences, l2_differences, spearman_correlations):
        """
        Calcula métricas comparativas entre explicações de modelos com e sem ruído.
        
        Esta função compara as explicações geradas por explainers para modelos com e sem ruído, 
        calculando várias métricas de similaridade. A função lida com diferentes formatos de 
        explicações, incluindo tuplas, arrays 1D/2D e explicações com diferentes dimensões.
        
        Parameters:
        -----------
        explanation_without_noise : list, ndarray, tuple
            Explicação gerada pelo modelo sem ruído
        explanation_with_noise : list, ndarray, tuple
            Explicação gerada pelo modelo com ruído
        similarities : list
            Lista onde as similaridades de cosseno serão armazenadas
        l1_differences : list
            Lista onde as diferenças L1 serão armazenadas
        l2_differences : list
            Lista onde as diferenças L2 serão armazenadas
        spearman_correlations : list
            Lista onde as correlações de Spearman serão armazenadas
        """
        # 1. Verificações preliminares e tratamento de tipos
        try:
            # Verificar se as explicações são tuplas (caso do GNNExplainer)
            if isinstance(explanation_without_noise, tuple):
                # GNNExplainer retorna (node_mask, edge_mask)
                explanation_without_noise = explanation_without_noise[0]
                
            if isinstance(explanation_with_noise, tuple):
                explanation_with_noise = explanation_with_noise[0]
                
            # Garantir que temos arrays ou listas
            if not isinstance(explanation_without_noise, (list, np.ndarray)):
                explanation_without_noise = [explanation_without_noise]
                
            if not isinstance(explanation_with_noise, (list, np.ndarray)):
                explanation_with_noise = [explanation_with_noise]
                
            # Verificar dimensões das explicações
            is_2d_without_noise = any(isinstance(item, (list, np.ndarray)) 
                                    for item in explanation_without_noise)
            is_2d_with_noise = any(isinstance(item, (list, np.ndarray))
                                for item in explanation_with_noise)
            
            # 2. Processamento específico baseado na estrutura dos dados
            try:
                # Caso as explicações tenham estruturas diferentes
                if is_2d_without_noise != is_2d_with_noise:
                    print("Aviso: Estruturas de explicação inconsistentes. Tentando conversão segura.")
                
                if is_2d_without_noise or is_2d_with_noise:
                    # Tentativa de achatar arrays 2D de forma segura
                    try:
                        if is_2d_without_noise:
                            flattened_without_noise = []
                            for item in explanation_without_noise:
                                if isinstance(item, (list, np.ndarray)):
                                    flattened_without_noise.extend(item)
                                else:
                                    flattened_without_noise.append(item)
                            explanation_without_noise_flat = np.array(flattened_without_noise, dtype=float)
                        else:
                            explanation_without_noise_flat = np.array(explanation_without_noise, dtype=float)
                            
                        if is_2d_with_noise:
                            flattened_with_noise = []
                            for item in explanation_with_noise:
                                if isinstance(item, (list, np.ndarray)):
                                    flattened_with_noise.extend(item)
                                else:
                                    flattened_with_noise.append(item)
                            explanation_with_noise_flat = np.array(flattened_with_noise, dtype=float)
                        else:
                            explanation_with_noise_flat = np.array(explanation_with_noise, dtype=float)
                            
                    except Exception as e:
                        # Fallback para caso de erro na conversão
                        print(f"Erro ao achatar arrays: {e}. Usando método alternativo.")
                        explanation_without_noise_flat = np.array([float(x) for x in np.array(explanation_without_noise).flatten() 
                                                                if str(x).replace('.','',1).isdigit()])
                        explanation_with_noise_flat = np.array([float(x) for x in np.array(explanation_with_noise).flatten() 
                                                            if str(x).replace('.','',1).isdigit()])
                else:
                    # Caso simples: explicações 1D
                    explanation_without_noise_flat = np.array(explanation_without_noise, dtype=float)
                    explanation_with_noise_flat = np.array(explanation_with_noise, dtype=float)
                    
                # 3. Verificar se temos dados válidos após conversão
                if len(explanation_without_noise_flat) == 0 or len(explanation_with_noise_flat) == 0:
                    print("Aviso: Explicação vazia após conversão. Pulando métricas.")
                    similarities.append(0.5)  # valor neutro
                    l1_differences.append(0)
                    l2_differences.append(0)
                    spearman_correlations.append(0)
                    return
                    
                # 4. Criar vetores de importância alinhados
                max_feature = max(len(explanation_without_noise_flat), len(explanation_with_noise_flat))
                
                # Converter para dicionários para fazer alinhamento quando dimensões são diferentes
                importance_dict = dict(zip(range(len(explanation_without_noise_flat)), explanation_without_noise_flat))
                importance_dict_noise = dict(zip(range(len(explanation_with_noise_flat)), explanation_with_noise_flat))
                
                # Criar listas alinhadas (preenchendo com zeros quando necessário)
                importance_list = [importance_dict.get(i, 0) for i in range(max_feature)]
                importance_list_noise = [importance_dict_noise.get(i, 0) for i in range(max_feature)]
                
                # 5. Calcular métricas
                imp = np.array(importance_list)
                imp_noise = np.array(importance_list_noise)
                
                # Verificar dados inválidos antes do cálculo de métricas
                if np.isnan(imp).any() or np.isnan(imp_noise).any():
                    print("Aviso: Detectados valores NaN nas explicações.")
                    imp = np.nan_to_num(imp)
                    imp_noise = np.nan_to_num(imp_noise)
                    
                # Verificar se os vetores estão com valores iguais (evitar divisão por zero)
                if np.array_equal(imp, imp_noise) or (np.sum(np.abs(imp)) < 1e-10 and np.sum(np.abs(imp_noise)) < 1e-10):
                    similarities.append(1.0)  # Vetores idênticos ou próximos de zero
                    l1_differences.append(0.0)
                    l2_differences.append(0.0)
                    spearman_correlations.append(1.0)
                    return
                    
                # Calcular similaridade de cosseno
                try:
                    sim = 1 - cosine(imp, imp_noise)
                    if np.isnan(sim):
                        sim = 0.5  # Valor neutro se cosseno não puder ser calculado
                except Exception:
                    sim = 0.5
                similarities.append(sim)
                
                # Calcular diferença L1 (soma dos valores absolutos das diferenças)
                l1_diff = np.sum(np.abs(imp - imp_noise))
                l1_differences.append(l1_diff)
                
                # Calcular diferença L2 (norma euclidiana da diferença)
                l2_diff = np.linalg.norm(imp - imp_noise)
                l2_differences.append(l2_diff)
                
                # Calcular correlação de Spearman
                try:
                    rho, _ = spearmanr(imp, imp_noise)
                    if np.isnan(rho):
                        rho = 0  # Valor neutro se correlação não puder ser calculada
                except Exception:
                    rho = 0
                spearman_correlations.append(rho)
                
            except Exception as e:
                print(f"Erro ao processar explicações: {e}")
                # Valores neutros em caso de erro
                similarities.append(0.5)
                l1_differences.append(0)
                l2_differences.append(0)
                spearman_correlations.append(0)
                
        except Exception as global_e:
            print(f"Erro global ao calcular métricas: {global_e}")
            # Valores neutros em caso de erro
            similarities.append(0.5)
            l1_differences.append(0)
            l2_differences.append(0)
            spearman_correlations.append(0)


class FingerprintAnalyzer:
    """
    Classe para analisar os fingerprints e explicações SHAP de moléculas.
    
    Esta classe organiza o processo de análise em métodos menores e mais focados,
    facilitando a compreensão e manutenção do código.
    """
    
    def __init__(self, explanation, batch_idx, mol_idx, dataset_type='test', device='cpu'):
        """
        Inicializa o analisador de fingerprints.
        
        Args:
            explanation: Explicação SHAP pré-calculada.
            batch_idx: Índice do batch a ser analisado.
            mol_idx: Índice da molécula dentro do batch.
            dataset_type: Tipo do conjunto ('train', 'test', 'val').
            device: Dispositivo para processamento (CPU/GPU).
        """
        import io
        from chemxai.data import qm9_tabular
        
        self.explanation = explanation
        self.batch_idx = batch_idx
        self.mol_idx = mol_idx
        self.dataset_type = dataset_type
        self.device = device
        
        # Inicializa o qm9 e cria os dataloaders internamente
        self.qm9 = qm9_tabular()
        self.train_loader, self.val_loader, self.test_loader, _, _, _, _ = self.qm9.get_paired_dataloaders_tabular(
            batch_size=32, n_noise=3, descriptor_type='Morgan', morgan_radius=3, morgan_nBits=512
        )
        
        # Atributos que serão inicializados posteriormente
        self.mol = None
        self.test_smiles = None
        self.morgan_fp = None
        self.bit_info = {}
        self.important_bits = []
        self.importance_values = []
        self.fp_array = None
        self.active_bits = None
        self.top_bits = []
        self.top_values = []
        self.active_important_bits = []
        
        # Captura de saída para retornar texto
        self.output = io.StringIO()
    
    def load_molecule(self):
        """
        Carrega a molécula do batch e índice especificados.
        
        Returns:
            bool: True se a molécula foi carregada com sucesso, False caso contrário.
        """
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Seleciona o dataloader correto
        if self.dataset_type == 'train':
            data_loader = self.train_loader
        elif self.dataset_type == 'test':
            data_loader = self.test_loader
        elif self.dataset_type == 'val':
            data_loader = self.val_loader
        else:
            raise ValueError("dataset_type deve ser 'train', 'test' ou 'val'")
        
        # Calcula o índice global baseado no batch_idx e mol_idx
        global_idx = self.batch_idx * data_loader.batch_size + self.mol_idx
        
        try:
            # Obter o SMILES da molécula usando o qm9 interno
            self.test_smiles = self.qm9.get_smiles_by_dataloader_idx(
                idx=global_idx, dataset_type=self.dataset_type
            )
            self.mol = Chem.MolFromSmiles(self.test_smiles)
            
            if self.mol is None:
                print(f"Erro: Não foi possível converter SMILES para molécula: {self.test_smiles}")
                return False
            
            # Obter fingerprint Morgan e informações dos bits
            self.bit_info = {}
            self.morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                self.mol, radius=3, nBits=512, bitInfo=self.bit_info
            )
            
            # Converter fingerprint para array numpy
            self.fp_array = np.array(list(self.morgan_fp))
            self.active_bits = np.where(self.fp_array == 1)[0]
            
            return True
        
        except Exception as e:
            print(f"Erro ao carregar molécula: {e}")
            return False
    
    def process_explanation(self):
        """
        Processa a explicação SHAP para extrair bits importantes.
        """
        # Filtra valores SHAP positivos
        explanation_ranking = {'idx': [], 'values': []}
        for i, exp in enumerate(self.explanation):
            if exp > 0.0:
                explanation_ranking['idx'].append(i)
                explanation_ranking['values'].append(exp)
        
        # Prepara dados para análise
        self.important_bits = explanation_ranking['idx']
        self.importance_values = explanation_ranking['values']
        
        # Ordenar bits por importância
        sorted_indices = sorted(range(len(self.importance_values)), 
                               key=lambda k: self.importance_values[k], reverse=True)
        self.top_bits = [self.important_bits[i] for i in sorted_indices]
        self.top_values = [self.importance_values[i] for i in sorted_indices]
        
        # Identificar bits ativos importantes
        self.active_important_bits = [bit for bit in self.top_bits if bit in self.active_bits]
        
        with redirect_stdout(self.output):
            print(f"Molécula: {self.test_smiles}")
            print(f"Total de bits importantes: {len(self.important_bits)}")
            print(f"Total de bits ativos no fingerprint: {np.sum(self.fp_array)}")
    
    def get_environment_atoms(self, atom_idx, radius):
        """
        Obtém todos os átomos dentro de um raio específico de um átomo central.
        
        Args:
            atom_idx: Índice do átomo central.
            radius: Raio para buscar átomos vizinhos.
            
        Returns:
            set: Conjunto de índices dos átomos no ambiente.
        """
        env_atoms = set([atom_idx])
        atoms_to_process = set([atom_idx])
        
        for r in range(radius):
            new_atoms = set()
            for idx in atoms_to_process:
                atom = self.mol.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx not in env_atoms:
                        new_atoms.add(n_idx)
                        env_atoms.add(n_idx)
            atoms_to_process = new_atoms
            if not atoms_to_process:
                break
        
        return env_atoms
    
    def show_original_molecule(self):
        """
        Exibe a visualização da molécula original.
        """
        from IPython.display import HTML, display
        import matplotlib.pyplot as plt
        from rdkit.Chem import Draw
        
        print("## Molécula Original")
        mol_img = Draw.MolToImage(self.mol, size=(600, 350))
        display(mol_img)
    
    def show_bits_summary(self):
        """
        Exibe o resumo dos bits importantes da Explicação.
        """
        display(HTML("<h2 style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>Resumo dos bits importantes da Explicação</h2>"))
        
        with redirect_stdout(self.output):
            print("Os bits destacados em VERMELHO estão ATIVOS (presentes) na molécula.")
            print("Os bits destacados em AZUL estão INATIVOS (ausentes) na molécula.")
            
            # Resumo dos top N bits importantes
            top_n = min(10, len(self.top_bits))
            print("\nResumo dos 10 bits mais importantes:")
            for i, (bit, value) in enumerate(zip(self.top_bits[:top_n], self.top_values[:top_n])):
                status = "ATIVO" if bit in self.active_bits else "INATIVO"
                color = "🔴" if bit in self.active_bits else "🔵"
                print(f"{i+1}. {color} Bit {bit}: {value:.4f} ({status})")
    
    def show_fragments_header(self):
        """
        Exibe o cabeçalho para a seção de fragmentos importantes.
        """
        display(HTML("<h2 style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>Fragmentos Moleculares Importantes</h2>"))
        
        with redirect_stdout(self.output):
            if not self.active_important_bits:
                print("Nenhum bit ativo importante encontrado no fingerprint desta molécula.")
            else:
                print(f"Encontrados {len(self.active_important_bits)} fragmentos ativos importantes:")
    
    def visualize_fragment(self, fragment_idx, bit):
        """
        Visualiza um fragmento molecular específico.
        
        Args:
            fragment_idx: Índice do fragmento para exibição.
            bit: Número do bit correspondente ao fragmento.
        """
        if bit not in self.bit_info:
            return
        
        # Obter o índice original para pegar o valor SHAP
        orig_idx = self.important_bits.index(bit)
        value = self.importance_values[orig_idx]
        
        # Cabeçalho para cada fragmento
        display(HTML(f"<h3 style='background-color:#ffeeee; padding:5px; border-radius:5px;'>Fragmento {fragment_idx+1}: Bit {bit} (Value: {value:.4f})</h3>"))
        
        # Coletar todos os átomos envolvidos neste bit
        atoms_to_highlight = []
        bit_radius = 0
        for atom_info in self.bit_info[bit]:
            atom_idx = atom_info[0]  # Índice do átomo central
            radius = atom_info[1]    # Raio do ambiente
            bit_radius = max(bit_radius, radius)
            atoms_to_highlight.append(atom_idx)
        
        # Criar visualização lado a lado: molécula completa com átomos destacados
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        # Molécula completa com átomo central destacado
        colors = {atom_idx: (1,0,0) for atom_idx in atoms_to_highlight}  # Vermelho para átomo central
        
        # Imagem da molécula completa com destaque
        img_whole = Draw.MolToImage(self.mol, highlightAtoms=atoms_to_highlight, 
                                   highlightAtomColors=colors, size=(400, 250))
        axs[0].imshow(img_whole)
        axs[0].set_title('Molécula Completa')
        axs[0].axis('off')
        
        # Mostrar o ambiente químico completo
        env_atoms = set()
        for atom_idx in atoms_to_highlight:
            # Usar o raio específico do bit para mostrar o ambiente completo
            env_atoms.update(self.get_environment_atoms(atom_idx, bit_radius))
        
        # Colorir os átomos do ambiente
        env_colors = {}
        for atom_idx in env_atoms:
            if atom_idx in atoms_to_highlight:
                env_colors[atom_idx] = (1,0,0)  # Vermelho para átomo central
            else:
                env_colors[atom_idx] = (1,0.7,0.4)  # Laranja para vizinhos
        
        # Imagem do fragmento
        img_fragment = Draw.MolToImage(self.mol, highlightAtoms=list(env_atoms), 
                                      highlightAtomColors=env_colors, size=(400, 250))
        axs[1].imshow(img_fragment)
        axs[1].set_title(f'Fragmento (Raio {bit_radius})')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        with redirect_stdout(self.output):
            print(f"  - Átomo(s) central(is): {atoms_to_highlight}")
            print(f"  - Ambiente químico: {len(env_atoms)} átomos no raio {bit_radius}")
        
        # Se possível, mostrar a subestrutura isolada
        self.extract_substructure(atoms_to_highlight, env_atoms, env_colors)
        
        with redirect_stdout(self.output):
            print("\n" + "-"*50 + "\n")  # Separador entre fragmentos
        
        # # Use plt.figure() e plt.show() em vez de HTML
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_whole)
        # plt.title('Molécula Completa')
        # plt.axis('off')
        
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_fragment)
        # plt.title(f'Fragmento (Raio {bit_radius})')
        # plt.axis('off')
        
        # plt.tight_layout()
        # plt.show()
    
    def extract_substructure(self, atoms_to_highlight, env_atoms, env_colors):
        """
        Extrai e exibe a subestrutura correspondente ao fragmento.
        
        Args:
            atoms_to_highlight: Lista de índices dos átomos centrais.
            env_atoms: Conjunto de índices de todos os átomos no ambiente.
            env_colors: Dicionário de cores para cada átomo.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        from IPython.display import display
        
        if len(atoms_to_highlight) != 1 or len(env_atoms) <= 1:
            return
        
        try:
            # Criar uma subestrutura correspondente ao ambiente
            env_list = sorted(list(env_atoms))
            
            # Tentar extrair a subestrutura como SMILES
            submol = Chem.MolFragmentToSmiles(self.mol, env_list, isomericSmiles=True)
            fragment_mol = Chem.MolFromSmiles(submol)
            
            if fragment_mol is not None and fragment_mol.GetNumAtoms() > 0:
                print("  - Subestrutura isolada:")
                # Melhorar a visualização do fragmento
                AllChem.Compute2DCoords(fragment_mol)
                submol_img = Draw.MolToImage(fragment_mol, size=(300, 200))
                display(submol_img)
            else:
                self.show_alternative_view(env_atoms, env_colors)
        except Exception as e:
            print(f"  - Erro ao extrair subestrutura: {str(e)}")
            self.show_alternative_view(env_atoms, env_colors)
    
    def show_alternative_view(self, env_atoms, env_colors):
        """
        Mostra uma visualização alternativa quando a extração da subestrutura falha.
        
        Args:
            env_atoms: Conjunto de índices dos átomos no ambiente.
            env_colors: Dicionário de cores para cada átomo.
        """
        with redirect_stdout(self.output):
            print("  - Não foi possível isolar o fragmento como molécula válida")
            print("  - Visualização alternativa do fragmento:")
        
        cropped_img = Draw.MolToImage(self.mol, highlightAtoms=list(env_atoms), 
                                      highlightAtomColors=env_colors,
                                      highlightBonds=[], 
                                      size=(300, 200))
        display(cropped_img)
    
    def analyze(self):
        """
        Executa a análise completa da molécula e seus fingerprints importantes.
        
        Returns:
            str: Texto resumindo a análise dos fingerprints importantes.
        """
        self.load_molecule()
        self.process_explanation()
        self.show_original_molecule()
        self.show_bits_summary()
        self.show_fragments_header()
        
        # Visualizar cada fragmento importante
        for i, bit in enumerate(self.active_important_bits):
            self.visualize_fragment(i, bit)
        
        # Retornar o texto compilado
        return self.output.getvalue()

# Pensar na Corretividade fazendo a incerteza do modelo, usar mais modelos diferentes e fazer a variancia do erro obtido por
# esses modelos diferentes


if __name__ == '__main__':
    pass
    # # Exemplo de uso da classe
    # # Calculando explicação SHAP para um batch específico
    # batch_idx = 0
    # mol_idx = 0

    # # Obtendo dados do batch
    # batch_data = next(iter(test_loader))

    # # Calculando explicação SHAP
    # explainer = Shap(model=model_without_noise, background_tensor=background, 
    #                 test_tensor=batch_data[0], device=device)
    # explanation = explainer.explain_local(index=mol_idx)

    # # Criando o analisador e executando a análise
    # analyzer = FingerprintAnalyzer(
    #     explanation=explanation,
    #     batch_idx=batch_idx,
    #     mol_idx=mol_idx,
    #     dataset_type='test',
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device
    # )

    # # Executar a análise e obter o texto resumo
    # resultado = analyzer.analyze()
    # print(resultado)