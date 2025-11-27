# ToolBox-ChemXAI

A comprehensive toolkit for explaining AI models based on graphs and neural networks for chemical applications. See ToolBox Documentation [here](https://jollonld.github.io/ToolBox-ChemXAI/)

## 📋 Description

This repository contains a collection of tools and utilities for explainability of artificial intelligence models applied to chemical data, especially focused on molecular graph-based models and neural networks. The toolkit includes implementations of explainability methods such as SHAP, LIME, GNNExplainer, and robustness analysis.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/JollonLD/ToolBox-ChemXAI.git
cd ToolBox-ChemXAI

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

- `torch` - Deep learning framework
- `torch-geometric` - PyTorch extension for geometric data
- `rdkit` - Cheminformatics toolkit
- `dscribe` - Descriptors for machine learning in materials science
- `mordred` - Molecular descriptor calculator
- `shap` - SHAP library for explainability
- `lime` - LIME library for local explanations
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning
- `scipy` - Scientific computing
- `optuna` - Hyperparameter optimization framework

## 📁 Project Structure

```
ToolBox-ChemXAI/
├── chemxai/
│   ├── __init__.py
│   ├── data.py            # Data loading and preprocessing utilities
│   ├── models.py          # Neural network model architectures
│   ├── train.py           # Training pipelines and utilities
│   ├── explainers.py      # Explainability method implementations
│   ├── evaluate.py        # Classes for evaluation and robustness analysis
│   └── plots.py           # Visualization functions
├── README.md
├── requirements.txt       # Project dependencies
```

## 🔧 Python Modules

### `models.py`

Contains neural network architectures for both tabular and graph-based molecular data.

#### Tabular Models

##### `MLP`
Multi-Layer Perceptron for regression tasks with molecular descriptors.

**Parameters:**
- `input_dim`: number of input features
- `output_dim`: number of output targets (typically 1 for regression)
- `layers`: list of hidden layer dimensions (e.g., `[256, 128, 64]`)
- `device`: computation device (CPU or GPU)
- `lr`: learning rate (default: 0.001)

**Methods:**
- `forward(x)`: forward pass through the network

**Features:**
- Flexible architecture with configurable hidden layers
- ReLU activation functions
- L1Loss (MAE) criterion for training
- Adam optimizer

#### Graph-Based Models

##### `GCN`
Graph Convolutional Network for molecular property prediction from molecular graphs.

**Parameters:**
- `num_features`: number of node features
- `hidden_dim`: dimension of hidden layers (default: 256)

**Methods:**
- `forward(x, edge_index, batch=None)`: forward pass on graph data

**Architecture:**
- 3 GCN convolutional layers with batch normalization
- Global pooling for graph-level predictions
- Dropout (0.3) for regularization
- 2 fully connected layers for final prediction

### `train.py`

Provides comprehensive training pipelines for both MLP and GCN models with extensive logging and monitoring.

#### Main Functions

##### `train_mlp_qm9(att_index, epochs, layers, learning_rate, ...)`
Trains an MLP model on QM9 dataset with molecular descriptors.

**Parameters:**
- `att_index`: index of QM9 property to predict (0-14)
- `epochs`: number of training epochs (default: 10)
- `layers`: list of hidden layer dimensions (default: `[64, 32]`)
- `learning_rate`: learning rate (default: 1e-3)
- `batch_size`: batch size (default: 32)
- `n_noise`: number of noise features to add (default: 3)
- `descriptor_type`: type of molecular descriptor. Options:
  - `'CM'`: Coulomb Matrix
  - `'Morgan'`: Morgan fingerprints (ECFP)
  - `'MorganCount'`: Morgan fingerprints with counts
  - `'Physicochemical'`: 2D physicochemical descriptors
  - `'3D'`: 3D geometry-based descriptors
  - `'MACCS'`: MACCS keys (166 bits)
  - `'Topological'`: Topological fingerprints
  - `'AtomPair'`: Atom pair fingerprints
  - `'EState'`: E-state fingerprints
  - `'Pattern'`: SMARTS pattern fingerprints
  - `'Avalon'`: Avalon fingerprints
  - `'Autocorr'`: 2D autocorrelation descriptors
- `cache_descriptors`: whether to cache descriptors (default: True)
- `morgan_radius`: radius for Morgan fingerprints (default: 2)
- `morgan_nBits`: number of bits for fingerprints (default: 512)
- `log_dir`: directory for detailed logs (default: "logs")
- `layer_name`: identifier for model size (default: 'medium')

**Returns:**
- List of tuples `(epoch, train_loss, val_loss)` for each epoch

**Features:**
- Comprehensive logging with timestamps
- Automatic model checkpointing
- Early stopping with patience
- Training/validation/test split
- Support for noise injection experiments
- Detailed metrics tracking (MAE, MSE, R²)
- JSON export of training history

##### `train_gcn_qm9(target_idx, epochs, batch_size, lr, ...)`
Trains a GCN model on QM9 molecular graphs.

**Parameters:**
- `target_idx`: index of target property (default: 3)
- `epochs`: number of training epochs (default: 10)
- `batch_size`: batch size (default: 64)
- `lr`: learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 1e-4)
- `n_noise`: number of noise features (default: 0)
- `log_dir`: logging directory (default: "logs")

**Returns:**
- Trained model and training history

**Features:**
- Graph-level property prediction
- Support for noisy feature experiments
- Automatic best model saving
- Comprehensive logging and metrics
- Training curves visualization

##### `setup_logging(log_dir)`
Configures logging system with file and console output.

**Parameters:**
- `log_dir`: directory for log files (default: "logs")

**Returns:**
- Path to the created log file

**Features:**
- Timestamped log files
- Dual output (file + console)
- Structured logging format

### `data.py`

Provides comprehensive data loading and preprocessing utilities for molecular datasets, with caching and parallel processing for efficiency.

**Key Classes:**
- `graph_datasets`: Handler for graph-based molecular datasets (PCQM4Mv2, QM9)
- `qm9_tabular`: Handler for QM9 dataset with tabular descriptor computation

**Key Features:**
- Multiple molecular descriptor types support
- Automatic caching for faster reloading
- Parallel processing for descriptor computation
- Noise injection capabilities for robustness testing
- Paired dataloaders for clean/noisy model comparison

### `explainers.py`

Contains implementations of explainability methods for chemical models, divided into tabular and graph-based explainers.

#### Tabular Explainers

##### `Shap`
SHAP KernelExplainer implementation for tabular data (e.g., molecular fingerprints).

**Parameters:**
- `model`: model to be explained
- `background_tensor`: tensor containing background data for the explainer
- `test_tensor`: tensor containing test data to be explained
- `device`: device (CPU or GPU)

**Methods:**
- `explain_local(index)`: generates local explanation for a specific instance
- `explain_global()`: generates global explanation with average feature importance

##### `LIME`
LIME (Local Interpretable Model-agnostic Explanations) implementation for tabular data.

**Parameters:**
- `model`: model to be explained
- `background_tensor`: tensor with training data
- `test_tensor`: tensor with test data
- `device`: device (CPU or GPU)
- `mode`: 'regression' or 'classification'

**Methods:**
- `explain_local(index, num_features=None)`: explains a specific instance

#### Graph-Based Explainers

##### `GNNExplain`
GNNExplainer implementation for explaining GNN predictions.

**Parameters:**
- `model`: trained GNN model
- `device`: device
- `data`: graph data
- `epochs`: number of epochs to train the explainer
- `mode`: 'regression' or 'classification'
- `task_level`: 'node' or 'graph'
- `return_type`: 'raw' or 'probabilities'

**Methods:**
- `explain(index=None)`: returns important node and edge masks

##### `NodeGrapLIME`
LIME adaptation for explaining specific node predictions in GNNs.

**Parameters:**
- `data`: graph data
- `model`: GNN model
- `device`: device
- `hop`: number of hops to extract subgraph
- `rho`: Lasso regularization parameter

**Methods:**
- `explain(node_index, hops, num_samples)`: explains prediction of a specific node

##### `NodeGraphShap`
KernelSHAP implementation for explaining node predictions in GNNs.

**Parameters:**
- `data`: graph data
- `model`: GNN model
- `device`: device

**Methods:**
- `explain(node_index=0, hops=2, num_samples=10)`: computes SHAP values for a node

##### `GraphLIME`
LIME adapted for explaining graph-level predictions.

**Parameters:**
- `model`: GNN model
- `device`: device
- `rho`: regularization parameter

**Methods:**
- `explain(data, num_samples=100)`: explains entire graph prediction

##### `GraphShap`
KernelSHAP for explaining graph-level predictions.

**Parameters:**
- `data`: graph data
- `model`: GNN model
- `device`: device

**Methods:**
- `explain(num_samples=30)`: computes SHAP values for the graph

### `evaluate.py`

Contains classes for robustness evaluation and molecular fingerprint analysis.

#### `Evaluator`
Class for evaluating explainer robustness by comparing explanations from models with and without noise.

**Parameters:**
- `model_normal`: model without noise
- `model_noise`: model with noise
- `train_loader_normal`: normal training dataloader
- `test_loader_normal`: normal test dataloader
- `train_loader_noise`: noisy training dataloader
- `test_loader_noise`: noisy test dataloader
- `device`: device
- `model_type`: 'graph' or 'tabular'
- `explainer_type`: type of explainer to use

**Methods:**
- `robustness()`: evaluates robustness by computing similarity metrics between explanations

#### `FingerprintAnalyzer`
Class for analyzing molecular fingerprints and SHAP explanations.

**Parameters:**
- `explanation`: pre-computed SHAP explanation
- `batch_idx`: batch index
- `mol_idx`: molecule index within batch
- `dataset_type`: 'train', 'test', or 'val'
- `device`: device

**Methods:**
- `analyze()`: performs complete analysis of molecule and its important fingerprints
- `load_molecule()`: loads molecule from dataset
- `visualize_fragment(fragment_idx, bit)`: visualizes specific molecular fragment

### `plots.py`

Contains utility functions for visualization.

#### Main Functions

##### `k_hop_subgraph(node_idx, num_hops, edge_index, ...)`
Computes k-hop subgraph around a node.

**Parameters:**
- `node_idx`: central node
- `num_hops`: number of hops
- `edge_index`: edge indices
- `relabel_nodes`: whether to reindex nodes
- `flow`: flow direction

##### `radar_plot(values, feature_names=None, title="...")`
Creates radar plot of SHAP values.

**Parameters:**
- `values`: explanation values
- `feature_names`: feature names
- `title`: plot title

##### `horizontal_bar_plot(values, feature_names=None, ...)`
Creates horizontal bar plot for feature importance.

**Parameters:**
- `values`: importance values
- `feature_names`: feature names
- `sort`: whether to sort by importance
- `max_features`: maximum number of features to show

## 💻 Usage Examples

### Example 1: Training an MLP Model

```python
from chemxai.train import train_mlp_qm9

# Train MLP with Morgan fingerprints for property prediction
history = train_mlp_qm9(
    att_index=10,  # Internal energy at 0K (U0)
    epochs=100,
    layers=[512, 256, 128],
    learning_rate=1e-3,
    batch_size=64,
    descriptor_type='Morgan',
    morgan_radius=3,
    morgan_nBits=2048,
    layer_name='large'
)

# Model is automatically saved to models/mlp_qm9_Morgan_att10_large.pth
print(f"Training completed. Final validation loss: {history[-1][2]:.4f}")
```

### Example 2: Training a GCN Model

```python
from chemxai.train import train_gcn_qm9

# Train GCN on molecular graphs
model, history = train_gcn_qm9(
    target_idx=3,  # Dipole moment
    epochs=50,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4
)

print(f"Best validation MAE: {min([h[2] for h in history]):.4f}")
```

### Example 3: Computing Molecular Descriptors

```python
from chemxai.data import qm9_tabular

# Initialize QM9 handler
qm9 = qm9_tabular()

# Compute different types of descriptors
X_morgan, Y, props = qm9.compute_descriptors(
    descriptor_type='Morgan',
    morgan_radius=3,
    morgan_nBits=512,
    att_index=10
)

X_coulomb, Y, props = qm9.compute_descriptors(
    descriptor_type='CM',
    att_index=10
)

print(f"Morgan fingerprints shape: {X_morgan.shape}")
print(f"Coulomb matrix shape: {X_coulomb.shape}")
```

### Example 4: Tabular SHAP Explanation

```python
import torch
from explainers import Shap

# Load model and data
model = your_trained_model
background_data = torch.tensor(background_features)
test_data = torch.tensor(test_features)

# Initialize SHAP explainer
explainer = Shap(
    model=model,
    background_tensor=background_data,
    test_tensor=test_data,
    device='cpu'
)

# Generate local explanation for first instance
explanation = explainer.explain_local(index=0)
print(f"SHAP values: {explanation}")

# Generate global explanation
global_explanation = explainer.explain_global()
print(f"Global importance: {global_explanation}")
```

### Example 5: GNNExplainer for Graphs

```python
from explainers import GNNExplain
import torch_geometric.data as data

# Prepare graph data
graph_data = data.Data(x=node_features, edge_index=edge_indices)

# Initialize GNNExplainer
explainer = GNNExplain(
    model=gnn_model,
    device='cpu',
    data=graph_data,
    epochs=100,
    mode='regression',
    task_level='graph'
)

# Generate explanation
node_mask, edge_mask, explanation = explainer.explain()
print(f"Node importance: {node_mask}")
print(f"Edge importance: {edge_mask}")
```

### Example 6: Robustness Evaluation

```python
from evaluate import Evaluator

# Initialize evaluator
evaluator = Evaluator(
    model_normal=model_without_noise,
    model_noise=model_with_noise,
    train_loader_normal=train_loader,
    test_loader_normal=test_loader,
    train_loader_noise=train_loader_noise,
    test_loader_noise=test_loader_noise,
    device='cpu',
    model_type='tabular',
    explainer_type='shap_local'
)

# Evaluate robustness
similarities, l1_diffs, l2_diffs, spearman_corrs, figs = evaluator.robustness()

print(f"Average similarity: {np.mean(similarities)}")
print(f"Average L1 difference: {np.mean(l1_diffs)}")
```

### Example 7: Molecular Fingerprint Analysis

```python
from evaluate import FingerprintAnalyzer

# Calculate SHAP explanation first
explainer = Shap(model, background, test_data, device)
explanation = explainer.explain_local(index=0)

# Analyze fingerprints
analyzer = FingerprintAnalyzer(
    explanation=explanation,
    batch_idx=0,
    mol_idx=0,
    dataset_type='test',
    device='cpu'
)

# Execute complete analysis
result = analyzer.analyze()
print(result)
```

### Example 8: Explanation Visualization

```python
from plots import horizontal_bar_plot, radar_plot

# Example data
shap_values = [0.1, -0.3, 0.5, -0.2, 0.4]
feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']

# Horizontal bar plot
fig, ax = horizontal_bar_plot(
    values=shap_values,
    feature_names=feature_names,
    title="Feature Importance - SHAP",
    sort=True,
    max_features=10
)

# Radar plot
fig_radar, ax_radar = radar_plot(
    values=shap_values,
    feature_names=feature_names,
    title="SHAP Analysis - Radar Plot"
)
```

### Example 9: Graph-Level SHAP Analysis

```python
from explainers import GraphShap
import torch_geometric.data as data

# Prepare graph data
graph_data = data.Data(x=node_features, edge_index=edge_indices)

# Initialize GraphShap explainer
explainer = GraphShap(
    data=graph_data,
    model=gnn_model,
    device='cpu'
)

# Generate graph-level explanation
shap_values = explainer.explain(num_samples=50)
print(f"Graph SHAP values: {shap_values}")
```

### Example 10: Node-Level Analysis with NodeGraphLIME

```python
from explainers import NodeGrapLIME

# Initialize NodeGraphLIME
explainer = NodeGrapLIME(
    data=graph_data,
    model=gnn_model,
    device='cpu',
    hop=2,
    rho=0.1
)

# Explain specific node
node_explanation = explainer.explain(
    node_index=5,
    hops=2,
    num_samples=100
)
print(f"Node explanation: {node_explanation}")
```

## 🔍 Supported Explainability Methods

### Tabular
- **SHAP (KernelExplainer)**: SHAP values for tabular data
- **LIME**: Local interpretable explanations

### Graph-Based
- **GNNExplainer**: Native explainer for GNNs
- **GraphSHAP**: SHAP adapted for entire graphs
- **GraphLIME**: LIME adapted for entire graphs  
- **NodeGraphSHAP**: SHAP for specific nodes
- **NodeGraphLIME**: LIME for specific nodes

## 📊 Evaluation Metrics

The `evaluate.py` module implements several metrics to assess explanation quality and robustness:

- **Cosine Similarity**: Measures similarity between explanations
- **L1 Difference**: Sum of absolute differences
- **L2 Difference**: Euclidean norm of differences
- **Spearman Correlation**: Rank correlation between explanations

## 🎯 Use Cases

- **Drug Discovery**: Identify important molecular fragments
- **Chemical Property Prediction**: Explain predictions of solubility, toxicity, etc.
- **Safety Analysis**: Assess model robustness for critical applications
- **Chemical Research**: Understand structure-activity relationships

## 🔬 Robustness Analysis

The toolkit provides comprehensive robustness evaluation through the `Evaluator` class:

```python
# Robustness evaluation workflow
evaluator = Evaluator(
    model_normal=clean_model,
    model_noise=noisy_model,
    train_loader_normal=train_loader,
    test_loader_normal=test_loader,
    train_loader_noise=noisy_train_loader,
    test_loader_noise=noisy_test_loader,
    device='cpu',
    model_type='tabular',  # or 'graph'
    explainer_type='shap_local'  # or other explainer types
)

# Get robustness metrics
similarities, l1_diffs, l2_diffs, correlations, figures = evaluator.robustness()
```

The evaluation generates:
- Distribution plots of similarity metrics
- Statistical analysis of explanation stability
- Visual comparisons between clean and noisy model explanations

## 🧪 Molecular Fragment Analysis

The `FingerprintAnalyzer` provides detailed molecular fragment analysis:

```python
# Analyze molecular fragments contributing to predictions
analyzer = FingerprintAnalyzer(
    explanation=shap_explanation,
    batch_idx=0,
    mol_idx=0,
    dataset_type='test'
)

# Get detailed fragment analysis
analysis_report = analyzer.analyze()
```

Features include:
- Automatic SMILES to molecular structure conversion
- Morgan fingerprint bit analysis
- Visual highlighting of important molecular fragments
- Interactive molecular visualizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support, please open an issue on the GitHub repository.

## 🏷️ Keywords

`explainable-ai` `chemical-informatics` `graph-neural-networks` `shap` `lime` `molecular-analysis` `drug-discovery` `pytorch` `rdkit`

---

**Note**: This toolkit was developed for research in chemical AI explainability. For production use, additional validation of the methods is recommended.
