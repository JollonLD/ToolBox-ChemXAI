"""data.py

    Import datasets used in the ToolBox
"""

# Packages
from torch_geometric.datasets import PCQM4Mv2, QM9
import torch_geometric.transforms as T
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader, random_split
from ase import Atoms
from dscribe.descriptors import CoulombMatrix
import os as os
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#================================================================#
# Graph Based Datasets
#================================================================#
class CastXToFloat:
    def __call__(self, data):
        data.x = data.x.float()
        return data

def prepare_data_graph(dataset_name='PCQM4'):
    """Get Data to be used

    Parameter:
            dataset_name (str): name of the dataset (default: QM9)

    Return:
            [torch_geometric.Data]: dataset with the correct format to be used in the explanations
    """
    # Get the project path
    dirname = os.getcwd()
    # Download the dataset

    if dataset_name == 'PCQM4':
        path = os.path.join(dirname, 'data', 'PCQM4')
        print(path)
        transform = T.Compose([
            CastXToFloat(),
            T.NormalizeFeatures()
        ])
        data = PCQM4Mv2(root=path, transform=transform)

    elif dataset_name == 'QM9':
        path = os.path.join(dirname, 'data', 'QM9')
        print(path)
        data = QM9(root=path, transform=T.NormalizeFeatures())

    return data

def prepare_data_graph_noise(dataset_name='PCQM4', noise_type='gaussian', noise_scale=1.0, seed=42):
    """
    Prepare graph data with an additional noise feature for each node.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('PCQM4' or 'QM9')
    noise_type : str
        Type of noise: 'gaussian', 'uniform', 'binary'
    noise_scale : float
        Scale of the noise
    seed : int
        Random seed
        
    Returns:
    --------
    tuple : (data, is_noise)
        - data: PyG dataset with added noise feature
        - is_noise: Boolean mask indicating which node features are noise
    """
    # First get the original data
    data = prepare_data_graph(dataset_name)
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a modified dataset with noise feature
    modified_dataset = []
    
    # Create a boolean mask to track which feature is noise
    # Assuming the first graph has the same node feature dimensions as all others
    first_graph = data[0]
    n_features = first_graph.x.shape[1]
    is_noise = torch.zeros(n_features + 1, dtype=bool)
    is_noise[-1] = True  # Last feature is noise
    
    for graph in data:
        # Get number of nodes in this graph
        n_nodes = graph.x.shape[0]
        
        # Generate noise for each node based on specified type
        if noise_type == 'gaussian':
            noise = torch.randn(n_nodes, 1) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand(n_nodes, 1) * 2 - 1) * noise_scale
        elif noise_type == 'binary':
            noise = torch.randint(0, 2, (n_nodes, 1)).float()
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Add noise feature to node features
        new_x = torch.cat([graph.x, noise], dim=1)
        
        # Create new graph with noise feature
        new_graph = graph.clone()
        new_graph.x = new_x
        
        modified_dataset.append(new_graph)
    
    # Return modified dataset
    return modified_dataset, is_noise

def prepare_data_graph_edge_noise(dataset_name='PCQM4', noise_type='gaussian', noise_scale=1.0, seed=42):
    """
    Adiciona uma feature de ruído às arestas do grafo.

    Parameters:
    -----------
    dataset_name : str
        Nome do dataset ('PCQM4' ou 'QM9')
    noise_type : str
        Tipo de ruído: 'gaussian', 'uniform', 'binary'
    noise_scale : float
        Escala do ruído
    seed : int
        Semente aleatória
        
    Returns:
    --------
    tuple : (data, is_noise)
        - data: Dataset PyG com feature de ruído nas arestas
        - is_noise: Máscara booleana indicando qual feature das arestas é ruído
    """
    # Obter dados originais
    data = prepare_data_graph(dataset_name)

    # Configurar semente
    np.random.seed(seed)
    torch.manual_seed(seed)

    modified_dataset = []

    # Máscara para indicar qual feature é ruído
    first_graph = data[0]
    n_edge_features = first_graph.edge_attr.shape[1]
    is_noise = torch.zeros(n_edge_features + 1, dtype=bool)
    is_noise[-1] = True  # Última feature é ruído

    for graph in data:
        # Número de arestas
        n_edges = graph.edge_index.shape[1]
        
        # Gerar ruído para cada aresta
        if noise_type == 'gaussian':
            noise = torch.randn(n_edges, 1) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand(n_edges, 1) * 2 - 1) * noise_scale
        elif noise_type == 'binary':
            noise = torch.randint(0, 2, (n_edges, 1)).float()
        else:
            raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")
        
        # Adicionar feature de ruído às arestas
        new_edge_attr = torch.cat([graph.edge_attr, noise], dim=1)
        
        # Criar novo grafo
        new_graph = graph.clone()
        new_graph.edge_attr = new_edge_attr
        
        modified_dataset.append(new_graph)

    return modified_dataset, is_noise

def prepare_data_subgraph_noise(dataset_name='PCQM4', subgraph_ratio=0.2, 
                            noise_type='gaussian', noise_scale=1.0, seed=42):
    """
    Adiciona ruído apenas a um subgrafo (subconjunto de nós) em cada grafo.
    
    Parameters:
    -----------
    dataset_name : str
        Nome do dataset ('PCQM4' ou 'QM9')
    subgraph_ratio : float
        Fração de nós a serem incluídos no subgrafo (0-1)
    noise_type : str
        Tipo de ruído: 'gaussian', 'uniform', 'binary'
    noise_scale : float
        Escala do ruído
    seed : int
        Semente aleatória
        
    Returns:
    --------
    tuple : (data, is_noisy_node)
        - data: Dataset PyG com feature de ruído em subgrafo
        - is_noisy_node: Lista de máscaras binárias indicando quais nós têm ruído
    """
    # Obter dados originais
    data = prepare_data_graph(dataset_name)
    
    # Configurar semente
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    modified_dataset = []
    is_noisy_node_list = []
    
    for graph in data:
        # Número de nós
        n_nodes = graph.x.shape[0]
        n_features = graph.x.shape[1]
        
        # Selecionar aleatoriamente um subconjunto de nós
        num_noisy_nodes = max(1, int(n_nodes * subgraph_ratio))
        noisy_node_indices = np.random.choice(n_nodes, num_noisy_nodes, replace=False)
        
        # Criar máscara para nós com ruído
        is_noisy_node = torch.zeros(n_nodes, dtype=bool)
        is_noisy_node[noisy_node_indices] = True
        
        # Adicionar feature de ruído apenas para os nós selecionados
        new_x = graph.x.clone()
        
        if noise_type == 'gaussian':
            noise = torch.randn(num_noisy_nodes, n_features) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand(num_noisy_nodes, n_features) * 2 - 1) * noise_scale
        elif noise_type == 'binary':
            noise = torch.randint(0, 2, (num_noisy_nodes, n_features)).float()
        else:
            raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")
        
        # Aplicar ruído nos nós selecionados
        new_x[noisy_node_indices] += noise
        
        # Criar novo grafo
        new_graph = graph.clone()
        new_graph.x = new_x
        
        modified_dataset.append(new_graph)
        is_noisy_node_list.append(is_noisy_node)
    
    return modified_dataset, is_noisy_node_list

def prepare_data_graph_structural_noise(dataset_name='PCQM4', edge_ratio=0.1, seed=42):
        """
        Adiciona ruído estrutural adicionando arestas aleatórias ao grafo.
        
        Parameters:
        -----------
        dataset_name : str
            Nome do dataset ('PCQM4' ou 'QM9')
        edge_ratio : float
            Fração de arestas aleatórias a adicionar em relação ao número original
        seed : int
            Semente aleatória
            
        Returns:
        --------
        tuple : (data, is_noisy_edge)
            - data: Dataset PyG com arestas de ruído
            - is_noisy_edge: Lista de máscaras binárias indicando quais arestas são ruído
        """
        # Obter dados originais
        data = prepare_data_graph(dataset_name)
        
        # Configurar semente
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        modified_dataset = []
        is_noisy_edge_list = []
        
        for graph in data:
            # Número de nós e arestas
            n_nodes = graph.x.shape[0]
            n_edges = graph.edge_index.shape[1]
            n_edge_features = graph.edge_attr.shape[1] if hasattr(graph, 'edge_attr') else 0
            
            # Número de arestas aleatórias a adicionar
            num_random_edges = max(1, int(n_edges * edge_ratio))
            
            # Gerar pares de nós aleatórios para novas arestas
            random_src = torch.randint(0, n_nodes, (num_random_edges,))
            random_dst = torch.randint(0, n_nodes, (num_random_edges,))
            
            # Criar novos índices de arestas
            new_edges = torch.stack([random_src, random_dst], dim=0)
            new_edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
            
            # Criar máscara para arestas de ruído
            is_noisy_edge = torch.zeros(n_edges + num_random_edges, dtype=bool)
            is_noisy_edge[n_edges:] = True
            
            # Criar novo grafo
            new_graph = graph.clone()
            new_graph.edge_index = new_edge_index
            
            # Se o grafo tiver atributos de aresta, também precisamos adicionar para as novas arestas
            if hasattr(graph, 'edge_attr') and n_edge_features > 0:
                # Criar atributos aleatórios para as novas arestas
                random_edge_attr = torch.randn(num_random_edges, n_edge_features)
                new_edge_attr = torch.cat([graph.edge_attr, random_edge_attr], dim=0)
                new_graph.edge_attr = new_edge_attr
            
            modified_dataset.append(new_graph)
            is_noisy_edge_list.append(is_noisy_edge)
        
        return modified_dataset, is_noisy_edge_list

#================================================================#
# Tubular Datasets
#================================================================#

class qm9_tubular:
    def __init__(self):
        # Caminho absoluto baseado na localização do script atual
        base_dir = os.getcwd()
        
        self.directory_path = os.path.join(base_dir, "data", "QM9_tubular_Data")
        self.zip_path = os.path.join(base_dir, "data", "QM9.zip")
        self.qm9_folder = self.directory_path

        self.properties = [
            'Rotational constant A: GHz',
            'Rotational constant B: GHz',
            'Rotational constant C: GHz',
            'Dipole moment (μ): Debye (D)',
            'Isotropic polarizability (α): atomic units (a.u.)',
            'Energy of HOMO (ϵHOMO): Hartree (Ha)',
            'Energy of LUMO (ϵLUMO): Hartree (Ha)',
            'Gap (ϵgap): Hartree (Ha)',
            'Electronic spatial extent: atomic units (a.u.)',
            'Zero point vibrational energy (zpve): Hartree (Ha)',
            'Internal energy at 0 K (U0): Hartree (Ha)',
            'Internal energy at 298.15 K (U): Hartree (Ha)',
            'Enthalpy at 298.15 K (H): Hartree (Ha)',
            'Free energy at 298.15 K (G): Hartree (Ha)',
            'Heat capacity at 298.15 K (Cv): cal/mol·K'
        ]

        self.url = "https://www.dropbox.com/scl/fi/lbs52lc0av3eqi9zws3wp/QM9.zip?rlkey=925vtuebvf7kf9ifq9143d6az&dl=1"

        # Baixar o arquivo se ele não existir localmente
        if not os.path.exists(self.zip_path):
            print("Baixando o arquivo QM9.zip...")
            response = requests.get(self.url, stream=True)
            with open(self.zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download concluído.")

        # Criar a pasta de extração, se necessário
        if not os.path.exists(self.qm9_folder):
            os.makedirs(self.qm9_folder)

        # Extrair o conteúdo do arquivo zip
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.qm9_folder)

        print(f"Dados salvos em: {self.qm9_folder}")

    def load_qm9_xyz(self, file_path):
        """Load a single QM9.xyz file."""
        
        with open(file_path, 'r') as f:
            # Number of atoms
            natoms = int(f.readline())
            # Properties are in the second line
            properties = list(map(float, f.readline().split()[2:]))
            # Read atomic coordinates and types
            atoms = []
            coordinates = []
            for num_line, line in enumerate(f):
                if num_line >= 0 and num_line < natoms:
                    info = line.replace("*^","e").split()
                    atoms.append(info[0])
                    coordinates.append(list(map(float, info[1:-1])))

        return {
            "natoms": natoms,
            "atoms": atoms,
            "coordinates": np.array(coordinates),
            "properties": properties
        }

    def load_qm9_dataset(self, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files."""
        
        coords = []
        prop = []
        natoms = []
        i = 0
        for file_name in os.listdir(self.directory_path):
            if i == 10:
                break
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                molecule_data = self.load_qm9_xyz(file_path)
                if molecule_data['natoms'] in list_mols or len(list_mols) == 0:
                    coords.append([molecule_data['atoms'], molecule_data['coordinates']])
                    prop.append(molecule_data['properties'])
                    natoms.append(molecule_data['natoms'])
            i += 1
        return coords, prop, natoms
    
    def dataset_to_numpy(dataset):  
        """Convert a dataset to NumPy arrays."""
        
        all_data = []
        all_targets = []

        for data, target in dataset:
            all_data.append(data.numpy())
            all_targets.append(target.numpy())
            
        data_numpy = np.array(all_data)
        targets_numpy = np.array(all_targets)
        
        return data_numpy, targets_numpy

    def get_smiles(self, file_path):
        """Get the SMILES representation of a molecule."""
        
        smiles = []
        
        with open(file_path, 'r') as f:
            # Read number of atoms
            natoms = int(f.readline())
            # Skip the second line
            f.readline()
            for i in range(natoms+1):  # Skip to the SMILES line
                f.readline()
            smiles_tuple = tuple(f.readline().strip().split('\t'))
            smiles.append(smiles_tuple)

        return smiles
        
    def load_smiles(self):
        """Load all SMILES representations in the QM9 dataset."""    
    
        list_smiles = []
        i = 0
        for file_name in os.listdir(self.directory_path):
            if i == 100:
                break
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                list_smiles.append(self.get_smiles(file_path))
            i += 1
        
        return list_smiles    
    
    def df_props(self):
        """Create a DataFrame with the properties of the QM9 dataset."""
        
        coords, props, natoms = self.load_qm9_dataset()
        
        # Create the DataFrame with properties
        df = pd.DataFrame(props)
        
        # Reset the DataFrame indices
        df.reset_index(drop=True, inplace=True)
        
        # Rename the DataFrame columns
        df.columns = self.properties
        
        return df  
    
    class Data(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.from_numpy(self.data[idx]).float(), torch.from_numpy(self.targets[idx]).float()

    def get_dataloader(self, att_index=10, batch_size=256, descriptor_type='CM', list_mols=[]):
        coords, props, natoms = self.load_qm9_dataset(list_mols=list_mols)
        props = np.array(props)

        # Convertendo para formato do ASE
        mols = [Atoms(positions=xyz, symbols=symbols) for symbols, xyz in coords]
        n_atoms_max = max(natoms)

        if descriptor_type == 'CM':
            cm = CoulombMatrix(n_atoms_max=n_atoms_max, permutation="eigenspectrum")
            X = cm.create(mols)
        
        # Shuffle e normalização
        X, props = shuffle(X, props, random_state=0)
        Ys = props[:, att_index].reshape(-1, 1)

        scaler = StandardScaler()
        Xn = scaler.fit_transform(X)

        dataset = self.Data(Xn, Ys)
        
        # Split
        train_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        
        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, X

    def get_dataloader_with_noise(self, att_index=10, batch_size=256, descriptor_type='CM', 
                             list_mols=[], noise_type='gaussian', noise_scale=1.0, seed=42):
        """
        Versão modificada de get_dataloader que adiciona uma coluna de ruído
        à matriz de Coulomb antes do processamento.
        
        Parameters:
        -----------
        att_index : int
            Índice da propriedade a ser predita
        batch_size : int
            Tamanho do batch
        descriptor_type : str
            Tipo de descritor (atualmente só suporta 'CM')
        list_mols : list
            Lista de moléculas a considerar
        noise_type : str
            Tipo de ruído: 'gaussian', 'uniform', 'binary'
        noise_scale : float
            Escala do ruído
        seed : int
            Semente aleatória
            
        Returns:
        --------
        tuple : (train_loader, val_loader, test_loader, X, is_noise)
            is_noise é uma máscara booleana indicando qual coluna é ruído
        """
        # Carregar dados normalmente até a matriz de Coulomb
        coords, props, natoms = self.load_qm9_dataset(list_mols=list_mols)
        props = np.array(props)

        # Convertendo para formato do ASE
        mols = [Atoms(positions=xyz, symbols=symbols) for symbols, xyz in coords]
        n_atoms_max = max(natoms)

        if descriptor_type == 'CM':
            cm = CoulombMatrix(n_atoms_max=n_atoms_max, permutation="eigenspectrum")
            X = cm.create(mols)
        
        # Configurar semente aleatória
        np.random.seed(seed)
        
        # Criar coluna de ruído
        n_samples = X.shape[0]
        if noise_type == 'gaussian':
            noise_feature = np.random.normal(0, noise_scale, size=(n_samples, 1))
        elif noise_type == 'uniform':
            noise_feature = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, 1))
        elif noise_type == 'binary':
            noise_feature = np.random.choice([0, 1], size=(n_samples, 1))
        else:
            raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")
        
        # Adicionar coluna de ruído à matriz de Coulomb
        X_with_noise = np.hstack((X, noise_feature))
        
        # Criar máscara para indicar qual coluna é ruído
        is_noise = np.zeros(X_with_noise.shape[1], dtype=bool)
        is_noise[-1] = True  # Última coluna é ruído
        
        # Continuar o processamento normal
        X_with_noise, props = shuffle(X_with_noise, props, random_state=0)
        Ys = props[:, att_index].reshape(-1, 1)

        scaler = StandardScaler()
        Xn = scaler.fit_transform(X_with_noise)

        dataset = self.Data(Xn, Ys)
        
        # Split
        train_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        
        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, X_with_noise, is_noise


if __name__ == '__main__':

    # qm9 = qm9_tubular()

    # # 1. Carregar os dados com DataLoaders (para usar com PyTorch)
    # # O argumento att_index escolhe qual propriedade química você quer prever
    # # 10 = 'Internal energy at 0 K (U0)'
    # train_loader, val_loader, test_loader, X_original = qm9.get_dataloader(
    #     att_index=10,           # Índice da propriedade a ser prevista
    #     batch_size=256,         # Tamanho do lote
    #     descriptor_type='CM',   # Usar Coulomb Matrix como descritor
    #     list_mols=[]            # Lista vazia = todas as moléculas (ou especifique uma lista)
    # )

    # train_loader_noise, val_loader_noise, test_loader_noise, X_noise, is_noise = qm9.get_dataloader_with_noise(
    #     att_index=10,           # Índice da propriedade a ser prevista
    #     batch_size=256,         # Tamanho do lote
    #     descriptor_type='CM',   # Usar Coulomb Matrix como descritor
    #     list_mols=[]            # Lista vazia = todas as moléculas (ou especifique uma lista)
    # )

    # print(train_loader.dataset[0])
    # print(train_loader_noise.dataset[0])

    data_qm9 = prepare_data_graph('QM9')

    print(data_qm9.data.x.shape)
    print(data_qm9.data.edge_attr.shape)

    data_pcqm = prepare_data_graph('PCQM4')

    print(data_pcqm[0].x.shape)
    print(data_pcqm[0].edge_attr.shape)
