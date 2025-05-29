import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import contextlib
import io

from chemxai.explainers import Shap, LIME

def robustness (model_normal, model_noise, train_loader_normal, test_loader_normal, train_loader_noise, test_loader_noise, device, model_type='tubular', explainer_type='shap'):
    base_dir = os.getcwd()

    similarities = []
    l1_differences = []
    l2_differences = []
    spearman_correlations = []

    for batch_train, batch_test, batch_train_noise, batch_test_noise in zip(train_loader_normal, test_loader_normal, train_loader_noise, test_loader_noise):
        
        background = batch_train[0]
        test_tensor = batch_test[0]

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            explainer = Shap(model=model_normal, background_tensor=background, test_tensor=test_tensor, device=device)
            explanation_without_noise = explainer.explain_global()
        
        background_noise = batch_train_noise[0]
        test_tensor_noise = batch_test_noise[0]

        with contextlib.redirect_stdout(f):
            explainer_noise = Shap(model=model_noise, background_tensor=background_noise, test_tensor=test_tensor_noise, device=device)
            explanation_with_noise = explainer_noise.explain_global()

        # Converte 'Feature' para int
        explanation_without_noise['Feature'] = explanation_without_noise['Feature'].astype(int)
        explanation_with_noise['Feature'] = explanation_with_noise['Feature'].astype(int)

        # Descobre o maior índice entre as duas explicações
        max_feature = max(explanation_without_noise['Feature'].max(), explanation_with_noise['Feature'].max())

        # Cria dicionários de importâncias
        importance_dict = dict(zip(explanation_without_noise['Feature'], explanation_without_noise['Importance']))
        importance_dict_noise = dict(zip(explanation_with_noise['Feature'], explanation_with_noise['Importance']))

        # Transforma em listas de tamanho igual
        importance_list = [importance_dict.get(i, 0) for i in range(max_feature + 1)]
        importance_list_noise = [importance_dict_noise.get(i, 0) for i in range(max_feature + 1)]

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

    # Após todos os batches:
    # print(f"Cosine Similarity: média={np.mean(similarities):.4f}, std={np.std(similarities):.4f}")
    # print(f"L1 diferença média: {np.mean(l1_differences):.4f}")
    # print(f"L2 diferença média: {np.mean(l2_differences):.4f}")
    # print(f"Spearman correlação média: {np.mean(spearman_correlations):.4f}")

    fig_cos, ax = plt.subplots(figsize=(6, 4))
    ax.hist(similarities, bins=20)
    ax.set_title('Distribuição da Cosine Similarity')
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Frequency')
    fig_cos.savefig('similarity_cosine.png')

    fig_l1, ax = plt.subplots(figsize=(6, 4))
    ax.hist(l1_differences, bins=20)
    ax.set_title('Distribuição das Diferenças')
    ax.set_xlabel('Difference')
    ax.set_ylabel('Frequency')
    fig_l1.savefig('l1_differences.png')
    
    fig_spearman, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spearman_correlations, bins=20)
    ax.set_title('Distribuição das Correlações Spearman')
    ax.set_xlabel('Difference')
    ax.set_ylabel('Frequency')
    fig_spearman.savefig('spearman_corr.png')
    
    figs = [fig_cos, fig_l1, fig_spearman]

    return similarities, l1_differences, l2_differences, spearman_correlations, figs



if __name__ == '__main__':
    pass