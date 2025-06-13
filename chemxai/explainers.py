# Packages
import shap
from lime import lime_tabular
import numpy as np
import pandas as pd
import scipy.special
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.nn import MessagePassing
from copy import deepcopy
import torch.nn.functional as F
import contextlib
import io


from .plots import k_hop_subgraph

#================================================================#
# Tabular Explainers
#================================================================#

class Shap:
    def __init__(self, model, background_tensor, test_tensor, device):
        """
        Initializes the Shap class with the model and tensor data.
        
        Parameters:
        - model: model to be explained.
        - background_tensor: tensor containing background data for the explainer
        - test_tensor: tensor containing test data to be explained
        - device: device (CPU or GPU) for operations.
        """
        self.model = model
        self.device = device

        # Convert tensors to numpy arrays for KernelExplainer
        background = background_tensor.cpu().numpy()
        print("Background shape:", background.shape)
        
        # Define prediction function for KernelExplainer compatibility with PyTorch model
        def predict_fn(data):
            self.model.eval()
            with torch.no_grad():
                data_tensor = torch.from_numpy(data).float().to(self.device)
                return self.model(data_tensor).cpu().numpy()
        
        # Initialize KernelExplainer with model and background data
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.explainer = shap.KernelExplainer(predict_fn, background)
            
        
        # Use test tensor directly
        self.test_data = test_tensor.cpu().numpy()
        print("Test data shape:", self.test_data.shape)
        
        # Compute shap_values for test data
        with contextlib.redirect_stdout(f):
            self.shap_values = self.explainer.shap_values(self.test_data)

    def explain_local(self, index):
        """
        Generates a local explanation for a specific instance and displays a DataFrame
        with feature indices and SHAP values for the chosen instance.
        
        Parameters:
        - index: index of the instance in the test set to be explained.
        """
        # Select SHAP values for the instance and flatten the extra dimension
        local_shap_values = self.shap_values[index].flatten()
        
        return local_shap_values.tolist()

    def explain_global(self):
        """
        Generates global explanations by calculating the overall mean importance across all instances.
        
        Returns:
        - feature_importance: DataFrame with average feature importance across all instances.
        """
        # Squeeze shap_values to remove any extra dimension if present
        shap_values_2d = np.squeeze(self.shap_values)  # Converts to Matrix
        
        # Compute global importance as the mean of absolute SHAP values across all instances
        mean_absolute_shap_values = np.mean(np.abs(shap_values_2d), axis=0)
        
        return mean_absolute_shap_values.tolist()

class LIME:
    def __init__(self, model, background_tensor, test_tensor, device, mode='regression'):
        """
        Initializes the LIME class with the model and tensor data.
        
        Parameters:
        - model: model to be explained.
        - background_tensor: tensor containing background data for the explainer
        - test_tensor: tensor containing test data to be explained
        - device: device (CPU or GPU) for operations.
        - mode: select whether it is a regression or classification model.
        """
        self.model = model
        self.device = device
        self.mode = mode
        
        # Convert tensors to numpy arrays for LIME
        self.x_train = background_tensor.cpu().numpy()
        print("Background shape:", self.x_train.shape)
        
        # Test data for explanation
        self.x_test = test_tensor.cpu().numpy()
        print("Test data shape:", self.x_test.shape)
        
        # Detect the number of features from training data
        self.num_features = self.x_train.shape[1]
        
        # Configure LimeTabularExplainer with training data
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.explainer_lime = lime_tabular.LimeTabularExplainer(
                training_data=self.x_train,
                mode=self.mode,  # Use "classification" if the model is a classifier
                feature_names=[f"Feature {i}" for i in range(self.num_features)],
                discretize_continuous=True,
                verbose=True
            )
        
    def predict_fn(self, data):
        """Prediction function to adapt the PyTorch model for LIME."""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().to(self.device)
            return self.model(data_tensor).cpu().numpy().flatten()
            
    def explain_local(self, index, num_features=None):
        """
        Gera explicação local para uma instância específica.
        """
        # Define the number of features for explanation if not specified
        if num_features is None:
            num_features = self.num_features
            
        # Select the instance from the test set for explanation
        instance_to_explain = self.x_test[index]
        
        # Generate explanation with LIME
        exp = self.explainer_lime.explain_instance(
            data_row=instance_to_explain,
            predict_fn=self.predict_fn,
            num_features=num_features
        )
        
        # Extract the explanation as a list of tuples
        explanation_list = exp.as_list()
        
        # Inicializar array de zeros com o tamanho total de features
        lime_values = np.zeros(self.num_features)
        
        # Preencher o array com os valores de importância nas posições corretas
        import re
        for feature_name, importance in explanation_list:
            # Extrair o índice usando expressão regular 
            # Procura por "Feature X" onde X é um número
            match = re.search(r'Feature\s+(\d+)', feature_name)
            if match:
                feature_idx = int(match.group(1))
                lime_values[feature_idx] = importance
            else:
                print(f"Aviso: Não foi possível extrair o índice da feature '{feature_name}'")
        
        return lime_values.tolist()

#================================================================#
# Graph Based Explainers
#================================================================#

class GNNExplain:
    """
    GNNExplainer (Model Explainability for Graph Neural Networks)

    This class implements an explainer for Graph Neural Networks (GNNs). It is designed to explain the predictions of a GNN model by identifying the important nodes and edges that contribute to the prediction of a specific node. The method uses a masking strategy, where it iteratively removes or perturbs nodes and edges in the graph to measure their impact on the model's output.

    Parameters
    ----------
    model : torch.nn.Module
        The trained GNN model to be explained.
    device : torch.device
        The device to run the model (CPU or GPU).
    data : torch_geometric.data.Data
        The graph data containing node features and edge indices.
    epochs : int
        Number of epochs for training the explainer.
    mode : str, optional (default='regression')
        The type of prediction task ('regression' or 'classification').
    task_level : str, optional (default='node')
        Whether the explanation is at the node level or graph level.
    return_type : str, optional (default='raw')
        The format of the returned explanation ('raw' or 'probabilities').
    """

    def __init__(self, model, device, data, epochs, mode='regression', task_level='graph', return_type='raw'):
        self.model = model
        self.data = data.to(device)
        self.device = device
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode=mode,
                task_level=task_level,
                return_type=return_type,
            ),
        )
    def explain(self, index=None):
        """
        Explains the prediction of a graph neural network (GNN) model for a specific node by calculating
        a mask of important features using the GNNExplainer method.

        This function computes the feature importance mask for a given node in the graph, indicating which
        features (nodes/edges) are most relevant to the model's prediction.

        Args:
            index (int): The index of the node for which the explanation is generated.

        Returns:
            tuple: A tuple containing:
                - node_mask (list): A list of feature importances for each node in the graph.
                - prediction (float): The predicted value for the specified node.

        Example:
            >>> explainer = GNNExplainer(model)
            >>> node_mask, prediction = explainer.explain(node_index)
        """

        # Para explicação do grafo inteiro
        batch = torch.zeros(self.data.x.size(0), dtype=torch.long, device=self.device)
        
        # Gerar explicação para o grafo inteiro
        explanation = self.explainer(
            self.data.x, 
            self.data.edge_index,
            batch=batch,  # Importante para indicar que todos os nós pertencem ao mesmo grafo
            index=0 if index is None else index  # Índice 0 no batch
        )

        return explanation.node_mask.squeeze().tolist()

class NodeGrapLIME: # Extracted from https://github.com/AlexDuvalinho/GraphSVX.git
    """
    NodeGrapLIME (Graph LIME for Node Features)

    This class adapts the LIME (Local Interpretable Model-agnostic Explanations) technique to graph neural networks. It explains the predictions of a GNN by perturbing node features and observing the effect on the prediction. The method uses a kernel-based similarity function to identify which features of the node and its neighbors are most influential for the model's prediction.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data containing node features and edge indices.
    model : torch.nn.Module
        The trained GNN model to be explained.
    device : torch.device
        The device to run the model (CPU or GPU).
    gpu : bool, optional (default=False)
        Whether to use GPU acceleration for model inference.
    hop : int, optional (default=2)
        The number of hops for extracting the subgraph around the node of interest.
    rho : float, optional (default=0.1)
        Regularization parameter for Lasso regression.
    cached : bool, optional (default=True)
        Whether to cache the model's predictions for efficiency.
    """


    def __init__(self, data, model, device, gpu=False, hop=2, rho=0.1, cached=True):
        self.data = data
        self.model = model
        self.hop = hop
        self.rho = rho
        self.cached = cached
        self.cached_result = None
        self.M = self.data.num_features
        self.F = self.data.num_features
        self.model = model
        self.gpu = gpu
        if self.gpu:
            self.data = data.to(device)
        else:
            self.data = data

        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        # Grafico da biblioteca GraphSVX
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # Get the initial prediction.
            with torch.no_grad():
                if self.gpu:
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    self.model = self.model.to(device)
                    log_logits = self.model(
                        x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
                else:
                    log_logits = self.model(
                        x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result

    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape

        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        std = np.sqrt(d)

        # (n, n, 1) or (n, n, d)
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))

        return K

    def __compute_gram_matrix__(self, x):
        # unstable implementation due to matrix product (HxH)
        # n = x.shape[0]
        # H = np.eye(n, dtype=np.float) - 1.0 / n * np.ones(n, dtype=np.float)
        # G = np.dot(np.dot(H, x), H)

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, *unused, **kwargs):
        """
        Generates a LIME (Local Interpretable Model-agnostic Explanations) explanation for a specific node's 
        prediction in a graph-based model by approximating the decision boundary of the model using a simpler 
        model trained on perturbed versions of the graph.

        This function computes an explanation that shows the importance of various nodes and features in 
        predicting the class of a given target node.

        Args:
            node_index (int): The index of the target node whose prediction is being explained.
            hops (int): The number of hops to consider when creating the subgraph around the target node.
            num_samples (int): The number of perturbed graphs to generate for the LIME explanation.
            info (bool, optional): Whether to return additional information about the explanation.
            multiclass (bool, optional): Whether the model is a multiclass classifier.

        Returns:
            dict: A dictionary containing:
                - "coef_pca" (np.ndarray): The PCA-reduced coefficients for the perturbed graph.
                - "coef_original" (np.ndarray): The original feature coefficients before PCA reduction.
                - "top_features" (dict): A dictionary mapping each class to the top 5 features influencing the prediction.

        Example:
            >>> explainer = NodeGraphLIME(model)
            >>> explanation = explainer.explain(node_index=0, hops=2, num_samples=100)
        """

        # Preparar dados do grafo
        x = self.data.x
        edge_index = self.data.edge_index

        # Predição do modelo
        probas = self.__init_predict__(x, edge_index, **kwargs)

        # Extrai subgrafo centrado no nó-alvo
        x_sub, y_sub, edge_index_sub, mapping, edge_mask, _ = self.__subgraph__(
        node_index, x, self.data.y, edge_index, **kwargs
        )

        # Calcula a predição no subgrafo
        with torch.no_grad():
            logits = self.model(x_sub, edge_index_sub)
            probas = F.softmax(logits, dim=-1)

        # Preparar para regressão (converter tensores)
        x = x.cpu().detach().numpy()  # (n, d)
        y = probas.cpu().detach().numpy()  # (n, classes)

        n, d = x.shape

        # Kernel de similaridade
        K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)

        if multiclass:
            L = self.__compute_kernel__(y, reduce=False)  # (n, n, 1)
        else:
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

        # Centralizar e normalizar gram matrix
        K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
        L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

        # Flatten para usar no Lasso
        K_bar_flat = K_bar.reshape(n**2, d)

        if multiclass:
            L_bar_flat = L_bar.reshape(n**2, self.data.num_classes)
        else:
            L_bar_flat = L_bar.reshape(n**2,)

        # Redução de dimensionalidade com PCA
        n_components = min(50, K_bar_flat.shape[0], K_bar_flat.shape[1])
        pca = PCA(n_components=n_components)
        K_bar_reduced = pca.fit_transform(K_bar_flat)

        # Normalização dos targets
        if multiclass:
            scaler_Y = StandardScaler()
            L_bar_scaled = scaler_Y.fit_transform(L_bar_flat * n)
        else:
            L_bar_scaled = (L_bar_flat * n - np.mean(L_bar_flat * n)) / (np.std(L_bar_flat * n) + 1e-10)

        # Regressão com LassoLars
        solver = LassoLars(self.rho, fit_intercept=False, positive=True)
        solver.fit(K_bar_reduced, L_bar_scaled)

        coef_pca = solver.coef_.T if multiclass else solver.coef_  # shape: (n_components, n_classes)

        # Mapear de volta para as features originais
        coef_original = np.dot(pca.components_.T, coef_pca)  # shape: (n_features_original, n_classes)

        # Top features por classe
        # top_features = {}
        # for c in range(coef_original.shape[1]):
        #     indices = np.argsort(coef_original[:, c])[::-1][:5]
        #     top_features[c] = indices

        return coef_original.tolist()

class NodeGraphShap: # Extracted from https://github.com/AlexDuvalinho/GraphSVX.git
    """
    NodeGraphShap (KernelSHAP for Node Features in Graphs)

    This class implements the KernelSHAP algorithm for explaining node-level predictions in graph neural networks. The method computes Shapley values by sampling perturbations of node features and using weighted least squares to approximate the Shapley values. It helps to identify which features of a node and its neighbors are most important for a given prediction.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data containing node features and edge indices.
    model : torch.nn.Module
        The trained GNN model to be explained.
    device : torch.device
        The device to run the model (CPU or GPU).
    gpu : bool, optional (default=False)
        Whether to use GPU acceleration for model inference.
    """

    def __init__(self, data, model, device, gpu=False):
        self.model = model
        self.gpu = gpu
        self.data = data
        if self.gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.data = self.data.to(device)
        # number of nonzero features - for each node index
        self.M = self.data.num_features
        self.neighbours = None
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=10, info=True, multiclass=False, *unused):
        """
        Generates a SHAP (SHapley Additive exPlanations) explanation for the prediction of a specific node 
        in a graph-based model by calculating the Shapley values, which measure the contribution of each 
        feature (node/edge) to the prediction of the target node.

        This function approximates the Shapley values for a node's prediction by sampling different feature 
        combinations and using weighted linear regression.

        Args:
            node_index (int, optional): The index of the target node to explain. Default is 0.
            hops (int, optional): The number of hops to consider for the subgraph around the target node.
            num_samples (int, optional): The number of samples to generate for the Shapley explanation.
            info (bool, optional): Whether to return additional information about the explanation.
            multiclass (bool, optional): Whether the model is a multiclass classifier.

        Returns:
            dict: A dictionary containing:
                - "shap_values" (np.ndarray): The Shapley values for each feature in the graph.
                - "top_features" (dict): A dictionary of the top features contributing to the prediction for each class.
                - "top_values" (dict): The corresponding Shapley values for the top features.

        Example:
            >>> explainer = NodeGraphShap(model)
            >>> explanation = explainer.explain(node_index=0, num_samples=50)
        """

        # Compute true prediction of model, for original instance
        with torch.no_grad():
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                true_conf, true_pred = self.model(
                    x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[node_index].max(dim=0)

            else:
                true_conf, true_pred = self.model(
                    x=self.data.x, edge_index=self.data.edge_index).exp()[node_index].max(dim=0)

        # Determine z => features whose importance is investigated
        # Decrease number of samples because nodes are not considered
        num_samples = num_samples//3

        # Consider all features (+ use expectation like below)
        # feat_idx = torch.unsqueeze(torch.arange(self.F), 1)

        # Sample z - binary vector of dimension (num_samples, M)
        z_ = torch.empty(num_samples, self.M).random_(2)
        # Compute |z| for each sample z
        s = (z_ != 0).sum(dim=1)

        # Define weights associated with each sample using shapley kernel formula
        weights = self.shapley_kernel(s)

        # Create dataset (z, f(z')), stored as (z_, fz)
        # Retrive z' from z and x_v, then compute f(z')
        fz = self.compute_pred(node_index, num_samples,
                            z_, multiclass, true_pred)

        # OLS estimator for weighted linear regression
        phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)

        # Calcular top-k features por classe
        phi = np.array(phi)  # garantir que é NumPy

        top_features = {}
        top_values = {}
        top_k = 10  # número de features mais relevantes por classe

        num_classes = phi.shape[1] if multiclass else 1
        for c in range(num_classes):
            class_phi = np.abs(phi[:, c]) if multiclass else np.abs(phi)
            top_idx = np.argsort(class_phi)[::-1][:top_k]
            top_features[c] = top_idx
            top_values[c] = phi[top_idx, c] if multiclass else phi[top_idx]

        return phi.tolist()
           

    def shapley_kernel(self, s):
        """
        :param s: dimension of z' (number of features + neighbours included)
        :return: [scalar] value of shapley value 
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1/self.M)
            else:
                shap_kernel.append(
                    (self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, node_index, num_samples, z_, multiclass, true_pred):
        """
        Variables are exactly as defined in explainer function, where compute_pred is used
        This function aims to construct z' (from z and x_v) and then to compute f(z'), 
        meaning the prediction of the new instances with our original model. 
        In fact, it builds the dataset (z, f(z')), required to train the weighted linear model.

        :return fz: probability of belonging to each target classes, for all samples z'
        fz is of dimension N*C where N is num_samples and C num_classses. 
        """
        # This implies retrieving z from z' - wrt sampled neighbours and node features
        # We start this process here by storing new node features for v and neigbours to
        # isolate
        X_v = torch.zeros([num_samples, self.F])

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Do it for each sample
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Features where z_j == 1 are kept, others are set to 0
            for j in range(self.F):
                if z_[i, j].item() == 1:
                    X_v[i, j] = 1

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)
            X[node_index, :] = X_v[i, :]

            # Apply model on (X,A) as input.
            with torch.no_grad():
                if self.gpu:
                    proba = self.model(x=X.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                        node_index]
                else:
                    proba = self.model(x=X, edge_index=self.data.edge_index).exp()[
                        node_index]
            # Multiclass
            if not multiclass:
                fz[i] = proba[true_pred]
            else:
                fz[i] = proba

        return fz

    def OLS(self, z_, weights, fz):
        """
        :param z_: z - binary vector  
        :param weights: shapley kernel weights for z
        :param fz: f(z') where z is a new instance - formed from z and x
        :return: estimated coefficients of our weighted linear regression - on (z, f(z'))
        phi is of dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(0.00001 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.cpu().detach().numpy()))

        # Test accuracy
        # y_pred=z_.detach().numpy() @ phi
        #	print('r2: ', r2_score(fz, y_pred))
        #	print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]
    
class GraphLIME: # Adapted from https://github.com/AlexDuvalinho/GraphSVX.git
    """
    GraphLIME (LIME for Graph-Level Prediction)

    This class adapts the LIME method for graph-level predictions. It explains how the aggregated features of the nodes in a graph influence the graph's prediction. The method perturbs node features and calculates the impact on the prediction, using a regression model to identify the importance of each node feature.

    Parameters
    ----------
    model : torch.nn.Module
        The trained GNN model to be explained.
    device : torch.device, optional (default='cpu')
        The device to run the model (CPU or GPU).
    rho : float, optional (default=0.1)
        Regularization parameter for Lasso regression.
    """

    def __init__(self, model, device='cpu', rho=0.1):
        self.model = model.to(device)
        self.device = device
        self.rho = rho
        self.model.eval()

    def explain(self, data, num_samples=100):
        """
        Generates a LIME (Local Interpretable Model-agnostic Explanations) explanation for the prediction of 
        a graph-based model by approximating the decision boundary using a simpler model trained on perturbed 
        versions of the entire graph.

        This function computes the explanation by generating perturbed versions of the graph, using them to 
        predict the output, and then applying Lasso regression to determine the most important features for 
        the graph's prediction.

        Args:
            data (torch_geometric.data.Data): The graph data containing node features and edge indices.
            num_samples (int, optional): The number of perturbed graphs to generate for the LIME explanation.

        Returns:
            dict: A dictionary containing:
                - "feature_importance" (np.ndarray): The importance of each feature (node/edge) for the graph.
                - "top_features" (np.ndarray): The indices of the most important features.
                - "coef_matrix" (np.ndarray): The Lasso regression coefficients indicating feature importance.

        Example:
            >>> explainer = GraphLIME(model)
            >>> explanation = explainer.explain(graph_data, num_samples=100)
        """

        x = data.x.clone().to(self.device)  # (n_nodes, n_features)
        edge_index = data.edge_index.to(self.device)

        with torch.no_grad():
            base_pred = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            base_pred = base_pred.item()  # valor escalar

        n_nodes, n_feats = x.shape
        X_perturbed = []
        y_preds = []

        # Gerar perturbações
        for _ in range(num_samples):
            mask = torch.bernoulli(torch.full((n_nodes, n_feats), 0.8)).to(self.device)  # 80% dos valores mantidos
            x_new = x * mask
            with torch.no_grad():
                pred = self.model(x_new, edge_index)
                y_preds.append(pred.item())
                X_perturbed.append(mask.cpu().numpy().flatten())  # flatten para usar no Lasso

        X_perturbed = np.array(X_perturbed)  # shape (samples, n_nodes * n_feats)
        y_preds = np.array(y_preds)  # shape (samples, )

        # Padronização
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_perturbed)
        y_scaled = scaler_y.fit_transform(y_preds.reshape(-1, 1)).flatten()

        # Regressão LassoLars
        reg = LassoLars(alpha=self.rho, positive=True)
        reg.fit(X_scaled, y_scaled)

        # Coeficientes reshape
        coef = reg.coef_.reshape(n_nodes, n_feats)

        # Agregar importância por feature
        feature_importance = coef.sum(axis=0)  # shape: (n_feats,)

        top_features = np.argsort(feature_importance)[::-1]

        return feature_importance.tolist()
    
class GraphShap: # Adapted from https://github.com/AlexDuvalinho/GraphSVX.git
    """
    GraphShap (KernelSHAP for Graph-Level Prediction)

    This class implements the KernelSHAP algorithm for explaining graph-level predictions in Graph Neural Networks. It explains how the aggregated features of the nodes influence the graph's prediction. The method perturbs node features and computes Shapley values based on the impact of these perturbations on the overall graph prediction.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data containing node features and edge indices.
    model : torch.nn.Module
        The trained GNN model to be explained.
    device : torch.device
        The device to run the model (CPU or GPU).
    gpu : bool, optional (default=False)
        Whether to use GPU acceleration for model inference.
    """

    def __init__(self, data, model, device=None, gpu=False):
        self.model = model
        self.gpu = gpu
        self.device = device or torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.M = self.data.num_features  # número de features por nó
        self.F = self.M

    def explain(self, num_samples=30, info=True):
        """
        Generates a SHAP (SHapley Additive exPlanations) explanation for the prediction of the entire graph 
        by calculating the Shapley values for the graph features (nodes and edges) and identifying which 
        components of the graph contribute most to the model's prediction.

        This function approximates the Shapley values for the graph's prediction by using binary feature 
        masks and weighted linear regression.

        Args:
            num_samples (int, optional): The number of binary samples to generate for the SHAP explanation.
            info (bool, optional): Whether to return additional information about the explanation.

        Returns:
            dict: A dictionary containing:
                - "shap_values" (np.ndarray): The SHAP values for each feature (node/edge) in the graph.
                - "top_features" (np.ndarray): The indices of the most important features for the prediction.
                - "top_values" (np.ndarray): The SHAP values for the top features.
                - "true_prediction" (float): The true prediction value for the entire graph.
                - "base_value" (float): The base value for the prediction before considering feature importance.

        Example:
            >>> explainer = GraphShap(model)
            >>> explanation = explainer.explain(num_samples=50)
        """

        # Output real do modelo (para o grafo inteiro)
        with torch.no_grad():
            batch = torch.zeros(self.data.x.size(0), dtype=torch.long).to(self.device)
            true_output = self.model(
                x=self.data.x,
                edge_index=self.data.edge_index,
                batch=batch
            )
            true_output = true_output.view(-1)  # Flatten caso seja [1,1]

        # Amostras binárias para presença/ausência de features
        z_ = torch.empty(num_samples, self.M).random_(2)
        s = (z_ != 0).sum(dim=1)

        weights = self.shapley_kernel(s)
        fz = self.compute_pred(z_, true_output)

        phi, base_value = self.OLS(z_, weights, fz)

        # Seleção das top features
        phi = np.array(phi)
        top_idx = np.argsort(np.abs(phi))[::-1]

        return phi.tolist()

    def shapley_kernel(self, s):
        shap_kernel = []
        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1 / self.M)
            else:
                shap_kernel.append((self.M - 1) / (scipy.special.binom(self.M, a) * a * (self.M - a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, z_, true_output):
        """
        Aplica z_ às features dos nós e coleta as predições do grafo.
        """
        fz = torch.zeros(z_.size(0)).to(self.device)

        for i in range(z_.size(0)):
            X_masked = self.data.x.clone()

            for j in range(self.F):
                if z_[i, j].item() == 0:
                    X_masked[:, j] = 0  # zera a feature j em todos os nós

            batch = torch.zeros(X_masked.size(0), dtype=torch.long).to(self.device)

            with torch.no_grad():
                out = self.model(x=X_masked.to(self.device),
                                 edge_index=self.data.edge_index.to(self.device),
                                 batch=batch)
                fz[i] = out.view(-1)

        return fz

    def OLS(self, z_, weights, fz):
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)  # intercepto
        z_np = z_.cpu().detach().numpy()
        w_np = weights.cpu().detach().numpy()
        fz_np = fz.cpu().detach().numpy()

        try:
            tmp = np.linalg.inv(z_np.T @ np.diag(w_np) @ z_np)
        except np.linalg.LinAlgError:
            tmp = z_np.T @ np.diag(w_np) @ z_np
            tmp += np.diag(1e-5 * np.random.randn(tmp.shape[1]))
            tmp = np.linalg.inv(tmp)

        phi = tmp @ (z_np.T @ np.diag(w_np) @ fz_np)

        return phi[:-1], phi[-1]
    
