# ChemXAI - Chemical Explainable AI Toolbox

## Overview

ChemXAI is a comprehensive Python library for explainable AI in chemical machine learning. It provides unified interfaces for explaining predictions from both tabular and graph-based molecular models using various state-of-the-art explanation methods.

## Features

### Data Handling
- **Graph Datasets**: Support for PCQM4Mv2 and QM9 molecular graph datasets
- **Tabular Datasets**: QM9 with 12+ molecular descriptor types (Morgan, MACCS, Physicochemical, etc.)
- **Noise Features**: Built-in support for robustness testing with controlled noise injection

### Models
- **MLP**: Multi-Layer Perceptron for tabular molecular descriptors
- **GCN**: Graph Convolutional Network for molecular graphs

### Explainers

#### Tabular Explainers
- **SHAP**: SHapley Additive exPlanations with KernelExplainer
- **LIME**: Local Interpretable Model-agnostic Explanations

#### Graph Explainers
- **GNNExplain**: Native GNN explainer using masking
- **GraphShap**: SHAP for graph-level predictions
- **GraphLIME**: LIME for graph-level predictions
- **NodeGraphShap**: SHAP for node-level predictions
- **NodeGrapLIME**: LIME for node-level predictions

### Evaluation Tools
- **RobustnessEvaluator**: Compare explanations across models with multiple metrics
- **MolecularAnalyzer**: Analyze and visualize molecular fingerprints
- **TabularAnalyzer**: Fidelity metrics for tabular explanations
- **GraphAnalyzer**: Metrics for graph explanations

### Visualization
- **radar_plot**: Circular feature importance plots
- **horizontal_bar_plot**: Bar charts with automatic sorting and color coding
- **k_hop_subgraph**: Extract graph neighborhoods for visualization

## Installation

```bash
pip install chemxai
```

Or install from source:

```bash
git clone https://github.com/your-repo/chemxai.git
cd chemxai
pip install -e .
```

## Quick Start

### Tabular Model Explanation

```python
import torch
from chemxai.data import qm9_tabular
from chemxai.models import MLP
from chemxai.explainers import Shap
from chemxai.plots import horizontal_bar_plot

# Load data
qm9 = qm9_tabular()
train_loader, val_loader, test_loader, *_ = qm9.get_paired_dataloaders_tabular(
    att_index=7,  # Energy gap
    batch_size=64,
    descriptor_type='Morgan',
    morgan_nBits=2048
)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dim=2048, output_dim=1, layers=[256, 128, 64], device=device)
# ... training code ...

# Generate explanations
train_batch = next(iter(train_loader))
test_batch = next(iter(test_loader))

explainer = Shap(
    model=model,
    background_tensor=train_batch[0],
    test_tensor=test_batch[0],
    device=device
)

# Local explanation
shap_values = explainer.explain_local(index=0)

# Visualize
fig, ax = horizontal_bar_plot(shap_values, title="Feature Importance", max_features=20)
```

### Graph Model Explanation

```python
from chemxai.data import graph_datasets
from chemxai.models import GCN
from chemxai.explainers import GNNExplain

# Load graph data
gd = graph_datasets()
dataset = gd.prepare_data_graph('QM9')

# Create model
model = GCN(num_features=dataset[0].x.shape[1], hidden_dim=256)
# ... training code ...

# Explain
explainer = GNNExplain(
    model=model,
    device=device,
    data=dataset[0],
    epochs=100,
    mode='regression',
    task_level='graph'
)

node_mask, edge_mask, explanation = explainer.explain()
```

### Robustness Evaluation

```python
from chemxai.evaluate import RobustnessEvaluator

# Get paired data (clean and noisy)
result = qm9.get_paired_dataloaders_tabular(
    att_index=7,
    n_noise=3,
    batch_size=32
)
train, val, test, train_n, val_n, test_n, is_noise = result

# Train two models...
# model_clean = ... (on clean data)
# model_noise = ... (on noisy data)

# Evaluate robustness
evaluator = RobustnessEvaluator(
    first_model=model_clean,
    second_model=model_noise,
    x1_train=train,
    x1_test=test,
    x2_train=train_n,
    x2_test=test_n,
    device=device,
    model_type='tabular',
    explainer_type='shap_local'
)

similarities, l1_diffs, l2_diffs, spearman, figs = evaluator.get_metrics()
```

## Documentation

Full documentation is available at: [https://JollonLD.github.io/ToolBox-ChemXAI/](https://JollonLD.github.io/ToolBox-ChemXAI/)

### Main Documentation Files
- **[Complete API Reference](complete_api_documentation.html)**: Comprehensive single-page API documentation
- **[Data Module](data_module.html)**: Dataset loading and preprocessing
- **[Models Module](models_module.html)**: Neural network architectures
- **[Explainers](explainers/)**: All explanation methods
- **[Evaluators](evaluator/)**: Robustness and fidelity evaluation
- **[Plots](plots/)**: Visualization tools

## Supported Molecular Descriptors

- Morgan Fingerprints (ECFP)
- Morgan with Counts
- Coulomb Matrix
- Physicochemical Descriptors (2D)
- 3D Geometric Descriptors
- MACCS Keys
- Topological Fingerprints
- Atom Pair Fingerprints
- EState Indices
- Pattern Fingerprints
- Avalon Fingerprints
- 2D Autocorrelation

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0
- RDKit >= 2021.09.1
- scikit-learn >= 1.0.0
- SHAP >= 0.40.0
- LIME >= 0.2.0
- matplotlib >= 3.3.0
- numpy >= 1.20.0

## Citation

If you use ChemXAI in your research, please cite:

```bibtex
@software{chemxai2025,
  title={ChemXAI: A Comprehensive Toolbox for Explainable AI in Chemical Machine Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/JollonLD/ToolBox-ChemXAI}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/JollonLD/ToolBox-ChemXAI/issues)
- Documentation: [Full documentation](https://JollonLD.github.io/ToolBox-ChemXAI/)

## Acknowledgments

This library builds upon and integrates several excellent open-source projects:
- SHAP by Scott Lundberg
- LIME by Marco Tulio Ribeiro
- PyTorch Geometric
- RDKit
- GraphSVX implementations
