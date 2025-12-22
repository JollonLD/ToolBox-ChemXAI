#!/usr/bin/env python3
import re

# Ler o arquivo evaluate como referência de estrutura
with open('evaluate/index.html', 'r') as f:
    evaluate_content = f.read()

# Extrair apenas a parte do conteúdo principal do evaluate
evaluate_main = evaluate_content.split('<div class="section" itemprop="articleBody">')[1].split('</div>\n          </div><footer>')[0]

# Definir conteúdo correto para cada módulo
modules_content_map = {
    'data': (evaluate_main, '''<h1 id="data">Data Module<a class="headerlink" href="#data" title="Permanent link">&para;</a></h1>
<h2 id="overview">Overview<a class="headerlink" href="#overview" title="Permanent link">&para;</a></h2>
<p>The Data Module provides utilities for loading and preprocessing molecular datasets for both tabular and graph-based models. It includes support for various molecular descriptors and data transformations.</p>

<h2 id="classes">Classes<a class="headerlink" href="#classes" title="Permanent link">&para;</a></h2>

<h3 id="graph-datasets">graph_datasets<a class="headerlink" href="#graph-datasets" title="Permanent link">&para;</a></h3>
<p>Handles graph-based molecular datasets from PyTorch Geometric.</p>

<h4 id="methods">Methods<a class="headerlink" href="#methods" title="Permanent link">&para;</a></h4>

<h5 id="prepare-data-graph">prepare_data_graph()<a class="headerlink" href="#prepare-data-graph" title="Permanent link">&para;</a></h5>
<pre><code class="language-python">prepare_data_graph(dataset_name='PCQM4')
</code></pre>
<p>Load and prepare graph-based molecular datasets.</p>
<p><strong>Parameters:</strong></p>
<ul>
<li><code>dataset_name</code> (str): Name of the dataset. Options: 'PCQM4', 'QM9'. Default: 'PCQM4'</li>
</ul>
<p><strong>Returns:</strong></p>
<ul>
<li><code>torch_geometric.data.InMemoryDataset</code>: Dataset with normalized features</li>
</ul>

<h5 id="prepare-data-graph-noise">prepare_data_graph_noise()<a class="headerlink" href="#prepare-data-graph-noise" title="Permanent link">&para;</a></h5>
<pre><code class="language-python">prepare_data_graph_noise(dataset_name='PCQM4', noise_type='gaussian', 
                         noise_scale=1.0, seed=42)
</code></pre>
<p>Prepare graph data with additional noise features for robustness testing.</p>
<p><strong>Parameters:</strong></p>
<ul>
<li><code>dataset_name</code> (str): Name of the dataset ('PCQM4' or 'QM9')</li>
<li><code>noise_type</code> (str): Type of noise: 'gaussian', 'uniform', 'binary'</li>
<li><code>noise_scale</code> (float): Scale of the noise</li>
<li><code>seed</code> (int): Random seed for reproducibility</li>
</ul>

<h3 id="qm9-tabular">qm9_tabular<a class="headerlink" href="#qm9-tabular" title="Permanent link">&para;</a></h3>
<p>Handles tabular representations of molecular data using various descriptor types.</p>

<h4 id="supported-descriptors">Supported Descriptor Types<a class="headerlink" href="#supported-descriptors" title="Permanent link">&para;</a></h4>
<ul>
<li><strong>CM</strong>: Coulomb Matrix - quantum mechanical descriptors</li>
<li><strong>Morgan</strong>: Morgan Fingerprints (ECFP) - circular fingerprints</li>
<li><strong>MorganCount</strong>: Morgan Fingerprints with count information</li>
<li><strong>Physicochemical</strong>: 2D physicochemical descriptors (200 features)</li>
<li><strong>3D</strong>: Geometry-based 3D descriptors</li>
<li><strong>MACCS</strong>: MACCS Keys (166 bits)</li>
<li><strong>Topological</strong>: Daylight-like topological fingerprints</li>
<li><strong>AtomPair</strong>: Atom pair fingerprints</li>
<li><strong>EState</strong>: Electrotopological state indices</li>
<li><strong>Pattern</strong>: SMARTS pattern-based fingerprints</li>
<li><strong>Avalon</strong>: Avalon fingerprints for substructure screening</li>
<li><strong>Autocorr</strong>: 2D autocorrelation descriptors</li>
<li><strong>RDKit2D</strong>: Comprehensive RDKit 2D descriptors</li>
<li><strong>Mordred</strong>: Complete Mordred descriptor set (1800+ features)</li>
</ul>

<h2 id="usage-example">Usage Example<a class="headerlink" href="#usage-example" title="Permanent link">&para;</a></h2>
<pre><code class="language-python">from chemxai.data import graph_datasets, qm9_tabular

# Load graph dataset
graph_data = graph_datasets()
dataset = graph_data.prepare_data_graph(dataset_name='QM9')

# Load tabular dataset with Morgan fingerprints
tabular_data = qm9_tabular(
    att_index=7,  # Target property index
    descriptor_type='Morgan',
    cache_descriptors=True,
    morgan_radius=2,
    morgan_nBits=512
)
</code></pre>'''),
}

# Processar cada módulo
for module_name, (old_content, new_content) in modules_content_map.items():
    file_path = f'{module_name}/index.html'
    print(f"Updating {file_path}...")
    
    with open(file_path, 'r') as f:
        html = f.read()
    
    # Substituir o conteúdo
    html = html.replace(old_content, new_content)
    
    with open(file_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Updated {file_path}")

print("\nAll modules updated successfully!")
