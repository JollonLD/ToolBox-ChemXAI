# Packages
import shap
from lime import lime_tabular
import numpy as np
import pandas as pd
import scipy.special
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.linear_model import LassoLars
from torch_geometric.nn import MessagePassing
from copy import deepcopy

from src.plots import k_hop_subgraph

#================================================================#
# Tubular Explainers
#================================================================#

class Shap:
    def __init__(self, model, train_loader, test_loader, device):
        """
        Initializes the Shap class with the model, training, and test data.
        
        Parameters:
        - model: model to be explained.
        - train_loader: training DataLoader to obtain the background data.
        - test_loader: test DataLoader for explanations.
        - device: device (CPU or GPU) for operations.
        """
        self.model = model
        self.device = device

        # Get a batch from the training DataLoader as background data
        background = next(iter(train_loader))[0].cpu().numpy()  # Convert to numpy for KernelExplainer
        print("Background shape:", background.shape)  # Check background shape
        
        # Define prediction function for KernelExplainer compatibility with PyTorch model
        def predict_fn(data):
            self.model.eval()
            with torch.no_grad():
                data_tensor = torch.from_numpy(data).float().to(self.device)
                return self.model(data_tensor).cpu().numpy()
        
        # Initialize KernelExplainer with model and background data
        self.explainer = shap.KernelExplainer(predict_fn, background)
        
        # Load test data for explanations
        self.test_data, _ = next(iter(test_loader))
        self.test_data = self.test_data.cpu().numpy()  # Convert to numpy for KernelExplainer
        print("Test data shape:", self.test_data.shape)  # Check test data shape
        
        # Compute shap_values for test data
        self.shap_values = self.explainer.shap_values(self.test_data)

    def local_explanation(self, index):
        """
        Generates a local explanation for a specific instance and displays a DataFrame
        with feature indices and SHAP values for the chosen instance.
        
        Parameters:
        - index: index of the instance in the test set to be explained.
        """
        # Select SHAP values for the instance and flatten the extra dimension
        local_shap_values = self.shap_values[index].flatten()
        
        # Create a DataFrame with feature indices and SHAP values
        feature_importance = pd.DataFrame({
            'Feature Index': range(len(local_shap_values)),
            'SHAP Value': local_shap_values
        }).sort_values(by='SHAP Value', ascending=False).reset_index(drop=True)
        
        return feature_importance

    def global_explanation(self):
        """
        Generates global explanations by calculating the average feature importance for each instance
        in the test data and the overall mean importance across all instances.
        
        Returns:
        - all_local_importances: DataFrame where each row represents an instance and each column represents a feature.
        - global_feature_importance: DataFrame with average feature importance across all instances.
        """
        # Squeeze shap_values to remove any extra dimension if present
        shap_values_2d = np.squeeze(self.shap_values)  # Converts to Matrix
        
        # Compute local explanations for each instance and collect them in a DataFrame
        all_local_importances = pd.DataFrame(shap_values_2d)
        all_local_importances.columns = [f'Feature {i}' for i in range(shap_values_2d.shape[1])]
        
        # Compute global importance as the mean of absolute SHAP values across all instances
        mean_absolute_shap_values = np.mean(np.abs(shap_values_2d), axis=0)
        
        # Additional check on the shape of the mean SHAP values
        print("Global SHAP values shape:", mean_absolute_shap_values.shape)
        
        # DataFrame with mean absolute feature importance across all instances
        global_feature_importance = pd.DataFrame({
            'Feature': [f'{i}' for i in range(len(mean_absolute_shap_values))],
            'Importance': mean_absolute_shap_values
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return all_local_importances, global_feature_importance

class LIME:
    def __init__(self, model, train_loader, test_loader, device, mode='regression'):
        """
        Initializes the LIME class with the model and DataLoaders.
        
        Parameters:
        - model: model to be explained.
        - train_loader: DataLoader for the training set.
        - test_loader: DataLoader for the test set.
        - device: device (CPU or GPU) for operations.
        - mode: select whether it is a regression or classification model.
        """
        self.model = model
        self.device = device
        self.mode = mode
        
        # Get a batch from the training DataLoader and convert it to numpy
        self.x_train = next(iter(train_loader))[0].cpu().numpy()
        self.x_test = next(iter(test_loader))[0].cpu().numpy()  # Test data for explanation
        
        # Detect the number of features from x_train
        self.num_features = self.x_train.shape[1]
        
        # Configure LimeTabularExplainer with training data
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

    def local_explanation(self, index, num_features=None):
        """
        Generates a local explanation for a specific instance and displays a DataFrame
        with feature indices and LIME values for the chosen instance.
        
        Parameters:
        - index: index of the instance in the test set to be explained.
        - num_features: number of features to display in the explanation. If None, use all features.
        """
        # Define the number of features for explanation if not specified
        if num_features is None:
            num_features = self.num_features  # Use all features if `num_features` is not provided

        # Select the instance from the test set for explanation
        instance_to_explain = self.x_test[index]

        # Generate explanation with LIME
        exp = self.explainer_lime.explain_instance(
            data_row=instance_to_explain,
            predict_fn=self.predict_fn,
            num_features=num_features
        )
        
        # Extract the explanation as a list of tuples and convert to DataFrame
        explanation_list = exp.as_list()
        lime_df = pd.DataFrame(explanation_list, columns=["Feature", "LIME Value"]).sort_values(by="LIME Value", ascending=False)
        
        return lime_df

#================================================================#
# Graph Based Explainers
#================================================================#

class GNNEx:
    """

    """
    def __init__(self, model, data, epochs, mode, task_level):
        self.model = model
        self.data = data

    
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode=mode,
                task_level=task_level,
                return_type='log_probs',  # Model returns log probabilities.
            ),
        )
    def explanation(self, index):
        # Generate explanation for the node at index
        explanation = self.explainer(self.data.x, self.data.edge_index, index=index)
        print(explanation.edge_mask)
        print(explanation.node_mask)

        explanation.visualize_feature_importance(top_k=10)

        explanation.visualize_graph()

class GraphLIME: # Extracted from https://github.com/AlexDuvalinho/GraphSVX.git
    """ GraphLIME explainer - code taken from original repository
    Explains only node features

    """

    def __init__(self, data, model, gpu=False, hop=2, rho=0.1, cached=True):
        self.data = data
        self.model = model
        self.hop = hop
        self.rho = rho
        self.cached = cached
        self.cached_result = None
        self.M = self.data.num_features
        self.F = self.data.num_features
        self.gpu = gpu

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
        # hops, num_samples, info are useless: just to copy graphshap pipeline
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_index, x, probas, edge_index, **kwargs)

        x = x.cpu().detach().numpy()  # (n, d)
        y = probas.cpu().detach().numpy()  # (n, classes)

        n, d = x.shape

        if multiclass:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=False)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2, self.data.num_classes)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False,
                               normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_.T

        else:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False,
                               normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_

class GraphShap: # Extracted from https://github.com/AlexDuvalinho/GraphSVX.git
    """ KernelSHAP explainer - adapted to GNNs
    Explains only node features

    """
    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        # number of nonzero features - for each node index
        self.M = self.data.num_features
        self.neighbours = None
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=10, info=True, multiclass=False, *unused):
        """
        :param node_index: index of the node of interest
        :param hops: number k of k-hop neighbours to consider in the subgraph around node_index
        :param num_samples: number of samples we want to form GraphSVX's new dataset 

        :return: shapley values for features that influence node v's pred
        """
        # Compute true prediction of model, for original instance
        with torch.no_grad():
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                true_conf, true_pred = self.model(
                    x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[node_index][0].max(dim=0)
            else:
                true_conf, true_pred = self.model(
                    x=self.data.x, edge_index=self.data.edge_index).exp()[node_index][0].max(dim=0)

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

        return phi

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
    

