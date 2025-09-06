import torch
import numpy as np
import matplotlib.pyplot as plt

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
            node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
                    node(s).
            num_hops: (int): The number of hops :math:`k`.
            edge_index (LongTensor): The edge indices.
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                    :obj:`edge_index` will be relabeled to hold consecutive indices
                    starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                    :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
            flow (string, optional): The flow direction of :math:`k`-hop
                    aggregation (:obj:`"source_to_target"` or
                    :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
                     :class:`BoolTensor`)
    """

    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def radar_plot(values, feature_names=None, title="Feature Importance Radar Plot"):
    """
    Create a radar plot from SHAP values
    
    Parameters:
    -----------
    values : numpy.ndarray
        The values given by the explanations methods
    feature_names : list, optional
        Names of the features. If None, will use generic names
    title : str
        Title of the plot
    """
    # If values is for multiple instances, take the mean absolute value
    if len(values.shape) > 1:
        values = np.abs(values).mean(axis=0)
    else:
        values = np.abs(values)
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(values))]
    
    # Number of features
    N = len(values)
    
    # Angle for each feature
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular by appending the first value to the end
    values = np.append(values, values[0])
    angles = np.append(angles, angles[0])
    feature_names = np.append(feature_names, feature_names[0])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot the values
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names[:-1])
    
    # Add title
    plt.title(title, size=15, y=1.1)
    
    # Add grid and make it pretty
    ax.grid(True)
    
    return fig, ax

def horizontal_bar_plot(values, feature_names=None, title="Feature Importance", sort=True, 
                       max_features=None, color_positive='blue', color_negative='red',
                       figsize=(12, 8), save_path='graphs', filename='feature_importance.png'):
    """
    Creates a horizontal bar plot to visualize feature importance.
    
    Parameters:
    -----------
    values : array-like
        Importance values for each feature
    feature_names : list, optional
        Names of the features. If None, will use generic names
    title : str
        Title of the plot
    sort : bool
        Whether to sort features by absolute importance value
    max_features : int, optional
        Maximum number of features to display. If None, shows all features
    color_positive : str
        Color for positive values
    color_negative : str
        Color for negative values
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Directory to save the plot (default: 'graphs')
    filename : str
        Filename for the saved plot (default: 'feature_importance.png')
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    values = np.asarray(values)
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(values))]
        
    # Ensure we have enough names
    while len(feature_names) < len(values):
        feature_names.append(f"Feature {len(feature_names)}")
    
    # Sort features by absolute importance if requested
    if sort:
        # Get indices sorted by absolute value
        indices = np.argsort(np.abs(values))
        
        # Limit number of features if specified
        if max_features is not None and max_features < len(values):
            indices = indices[-max_features:]
            
        # Reorder values and names
        sorted_values = [values[i] for i in indices]
        sorted_names = [feature_names[i] for i in indices]
    else:
        # Use original order
        sorted_values = values
        sorted_names = feature_names
        
        # Limit number of features if specified
        if max_features is not None and max_features < len(values):
            sorted_values = sorted_values[-max_features:]
            sorted_names = sorted_names[-max_features:]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set colors based on value sign
    colors = [color_positive if v >= 0 else color_negative for v in sorted_values]
    
    # Create horizontal bar plot
    ax.barh(range(len(sorted_values)), sorted_values, color=colors)
    
    # Set y-ticks to feature names
    ax.set_yticks(range(len(sorted_values)))
    ax.set_yticklabels(sorted_names)
    
    # Set labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance Value", fontsize=12)
    
    # Add grid lines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adjusted layout to ensure everything fits
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Save the figure
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    
    return fig, ax

