import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import contextlib
import io

from chemxai.explainers import Shap, LIME, GNNExplain

class Evaluator:
    def __init__(self, model_normal, model_noise, train_loader_normal, test_loader_normal, train_loader_noise, test_loader_noise, device, model_type='graph', explainer_type='shap_local', mol_index=0, atom_index=0):
        self.model_normal = model_normal
        self.model_noise = model_noise
        self.train_loader_normal = train_loader_normal
        self.test_loader_normal = test_loader_normal
        self.train_loader_noise = train_loader_noise
        self.test_loader_noise = test_loader_noise
        self.device = device
        self.model_type = model_type
        self.explainer_type = explainer_type
        self.mol_index = mol_index
        self.atom_index = atom_index

    def robustness (self):
        
        dirname = os.getcwd()
        graphs_dir = os.path.join(dirname, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        print(f'Diretório criado: {graphs_dir}')


        similarities = []
        l1_differences = []
        l2_differences = []
        spearman_correlations = []

        for batch_train, batch_test, batch_train_noise, batch_test_noise in zip(self.train_loader_normal, self.test_loader_normal, self.train_loader_noise, self.test_loader_noise):

        
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                if self.model_type == 'tabular':

                    background = batch_train[0]
                    test_tensor = batch_test[0]
                    background_noise = batch_train_noise[0]
                    test_tensor_noise = batch_test_noise[0]

                    if self.explainer_type == 'shap_global':
                        explainer = Shap(model=self.model_normal, background_tensor=background, test_tensor=test_tensor, device=self.device)
                        explainer_noise = Shap(model=self.model_noise, background_tensor=background_noise, test_tensor=test_tensor_noise, device=self.device)
                        explanation_without_noise = explainer.explain_global()
                        explanation_with_noise = explainer_noise.explain_global()

                    if self.explainer_type == 'shap_local':
                        explainer = Shap(model=self.model_normal, background_tensor=background, test_tensor=test_tensor, device=self.device)
                        explainer_noise = Shap(model=self.model_noise, background_tensor=background_noise, test_tensor=test_tensor_noise, device=self.device)
                        explanation_without_noise = explainer.explain_local(index=self.mol_index)
                        explanation_with_noise = explainer_noise.explain_local(index=self.mol_index)

                    if self.explainer_type == 'lime':
                        explainer = LIME(model=self.model_normal, background_tensor=background, test_tensor=test_tensor, device=self.device)
                        explainer_noise = LIME(model=self.model_noise, background_tensor=background_noise, test_tensor=test_tensor_noise, device=self.device)
                        explanation_without_noise = explainer.explain_local(index=self.mol_index)
                        explanation_with_noise = explainer_noise.explain_local(index=self.mol_index)
                
                if self.model_type == 'graph':
            
                    data_normal = batch_test[self.mol_index]
                    data_noise = batch_test_noise[self.mol_index]

                    if self.explainer_type == 'gnn_explainer':
                        explainer = GNNExplain(model=self.model_normal, data=data_normal, device=self.device, epochs=20, mode='regression', task_level='graph', return_type='raw')
                        explanation_with_noise = explainer.explain(index=self.atom_index)
                        explainer_noise = GNNExplain(model=self.model_noise, data=data_noise, device=self.device, epochs=20, mode='regression', task_level='graph', return_type='raw')
                        explanation_without_noise = explainer_noise.explain(index=self.atom_index)


                    # if self.explainer_type == 'graphnode_shap':
                        

                    # if self.explainer_type == 'graph_shap':

                    
                    # if self.explainer_type == 'graphnode_lime':
                        

                    # if self.explainer_type == 'graphlime':


                
            # Descobre o maior índice entre as duas explicações
            max_feature = max(len(explanation_without_noise), len(explanation_with_noise))

            is_2d = isinstance(explanation_without_noise[0], (list, np.ndarray))
            
            if is_2d:
                # Achatar as explicações para comparação
                explanation_without_noise_flat = np.array([item for sublist in explanation_without_noise for item in sublist])
                explanation_with_noise_flat = np.array([item for sublist in explanation_with_noise for item in sublist])
                
                # Usar as versões achatadas
                max_feature = max(len(explanation_without_noise_flat), len(explanation_with_noise_flat))
                
                # Cria dicionários de importâncias
                importance_dict = dict(zip(range(len(explanation_without_noise_flat)), explanation_without_noise_flat))
                importance_dict_noise = dict(zip(range(len(explanation_with_noise_flat)), explanation_with_noise_flat))
            else:
                # Código original para explicações 1D
                max_feature = max(len(explanation_without_noise), len(explanation_with_noise))
                
                # Cria dicionários de importâncias
                importance_dict = dict(zip(range(len(explanation_without_noise)), explanation_without_noise))
                importance_dict_noise = dict(zip(range(len(explanation_with_noise)), explanation_with_noise))
            
            # Continua com o resto do código...
            # Transforma em listas de tamanho igual
            importance_list = [importance_dict.get(i, 0) for i in range(max_feature)]
            importance_list_noise = [importance_dict_noise.get(i, 0) for i in range(max_feature)]

            # Transforma em arrays
            imp = np.array(importance_list)
            imp_noise = np.array(importance_list_noise)

            # Cosine similarity (1 - distance)
            sim = 1 - cosine(imp, imp_noise)
            similarities.append(sim)

            # L1 norm
            l1_diff = np.sum(np.abs(imp - imp_noise))
            l1_differences.append(l1_diff)

            # L2 norm
            l2_diff = np.linalg.norm(imp - imp_noise)
            l2_differences.append(l2_diff)

            # Spearman rank correlation
            rho, _ = spearmanr(imp, imp_noise)
            spearman_correlations.append(rho)

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



if __name__ == '__main__':
    pass