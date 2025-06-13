"""data.py

    Import datasets used in the ToolBox
"""

# Packages
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import PCQM4Mv2, QM9
from torch_geometric.loader import DataLoader as GraphDataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ase import Atoms
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
RDLogger.DisableLog('rdApp.warning')
from dscribe.descriptors import CoulombMatrix

import os
import zipfile
import requests



#================================================================#
# Graph Based Datasets
#================================================================#

class CastXToFloat:
    def __call__(self, data):
        data.x = data.x.float()
        return data

class graph_datasets:

    def __init__(self):
        pass

    def prepare_data_graph(self, dataset_name='PCQM4'):
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

    def prepare_data_graph_noise(self, dataset_name='PCQM4', noise_type='gaussian', noise_scale=1.0, seed=42):
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
        torch_geometric.data.InMemoryDataset: Dataset with added noise feature
        """
        # First get the original data
        original_data = self.prepare_data_graph(dataset_name)
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create a modified dataset with noise feature
        modified_graphs = []
        
        # Create a boolean mask to track which feature is noise
        # Assuming the first graph has the same node feature dimensions as all others
        first_graph = original_data[0]
        n_features = first_graph.x.shape[1]
        is_noise = torch.zeros(n_features + 1, dtype=bool)
        is_noise[-1] = True  # Last feature is noise
        
        for graph in original_data:
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
            
            modified_graphs.append(new_graph)
        
        # Create a custom dataset with the modified graphs
        class NoisyGraphDataset(InMemoryDataset):
            def __init__(self, data_list):
                super().__init__(None)
                self.data, self.slices = self.collate(data_list)
                self.is_noise = is_noise
                
            @property
            def num_features(self):
                return self.data.x.size(1)
                
        # Return a dataset with the same interface as the original
        return NoisyGraphDataset(modified_graphs)

    def prepare_data_graph_edge_noise(self, dataset_name='PCQM4', noise_type='gaussian', noise_scale=1.0, seed=42):
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
        data = self.prepare_data_graph(dataset_name)

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

    def prepare_data_subgraph_noise(self, dataset_name='PCQM4', subgraph_ratio=0.2, 
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
        data = self.prepare_data_graph(dataset_name)
        
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

    def prepare_data_graph_structural_noise(self, dataset_name='PCQM4', edge_ratio=0.1, seed=42):
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
            data = self.prepare_data_graph(dataset_name)
            
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

    def get_paired_dataloaders(self, dataset_name='QM9', batch_size=32, 
                      split_ratio=[0.8, 0.1, 0.1], shuffle=True, 
                      seed=42, noise_type='gaussian', noise_scale=1.0, n_noise=1):
        """
        Prepara dois datasets (normal e com ruído) e garante que os mesmos índices 
        sejam usados para divisão de treino/validação/teste.
        """
        # Configurar semente para reprodutibilidade
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Preparar ambos os datasets
        dataset_normal = self.prepare_data_graph(dataset_name)
        dataset_noise = self.prepare_data_graph_noise(dataset_name, noise_type, noise_scale, seed)
        
        # Calcular tamanhos de cada conjunto
        num_samples = len(dataset_normal)
        train_size = int(num_samples * split_ratio[0])
        val_size = int(num_samples * split_ratio[1])
        
        # Criar uma única divisão aleatória
        indices = torch.randperm(num_samples)
        
        # Converter tensores para listas Python (aqui está a correção)
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size+val_size].tolist()
        test_indices = indices[train_size+val_size:].tolist()
        
        # Criar subconjuntos com os mesmos índices para ambos os datasets
        train_dataset_normal = torch.utils.data.Subset(dataset_normal, train_indices)
        val_dataset_normal = torch.utils.data.Subset(dataset_normal, val_indices)
        test_dataset_normal = torch.utils.data.Subset(dataset_normal, test_indices)
        
        train_dataset_noise = torch.utils.data.Subset(dataset_noise, train_indices)
        val_dataset_noise = torch.utils.data.Subset(dataset_noise, val_indices)
        test_dataset_noise = torch.utils.data.Subset(dataset_noise, test_indices)
        
        # Criar DataLoaders
        train_loader_normal = GraphDataLoader(train_dataset_normal, batch_size=batch_size, shuffle=shuffle)
        val_loader_normal = GraphDataLoader(val_dataset_normal, batch_size=batch_size, shuffle=False)
        test_loader_normal = GraphDataLoader(test_dataset_normal, batch_size=batch_size, shuffle=False)
        
        train_loader_noise = GraphDataLoader(train_dataset_noise, batch_size=batch_size, shuffle=shuffle)
        val_loader_noise = GraphDataLoader(val_dataset_noise, batch_size=batch_size, shuffle=False)
        test_loader_noise = GraphDataLoader(test_dataset_noise, batch_size=batch_size, shuffle=False)
        
        return (train_loader_normal, val_loader_normal, test_loader_normal,
                train_loader_noise, val_loader_noise, test_loader_noise)


#================================================================#
# Tabular Datasets
#================================================================#

class qm9_tabular:
    def __init__(self):
        # Caminho absoluto baseado na localização do script atual
        base_dir = os.getcwd()
        
        self.directory_path = os.path.join(base_dir, "data", "QM9_tabular_Data")
        self.zip_path = os.path.join(base_dir, "data", "QM9.zip")
        self.qm9_folder = self.directory_path

        self.properties = [
            'Rotational constant A: GHz', 'Rotational constant B: GHz', 'Rotational constant C: GHz',
            'Dipole moment (μ): Debye (D)', 'Isotropic polarizability (α): atomic units (a.u.)',
            'Energy of HOMO (ϵHOMO): Hartree (Ha)', 'Energy of LUMO (ϵLUMO): Hartree (Ha)',
            'Gap (ϵgap): Hartree (Ha)', 'Electronic spatial extent: atomic units (a.u.)',
            'Zero point vibrational energy (zpve): Hartree (Ha)', 'Internal energy at 0 K (U0): Hartree (Ha)',
            'Internal energy at 298.15 K (U): Hartree (Ha)', 'Enthalpy at 298.15 K (H): Hartree (Ha)',
            'Free energy at 298.15 K (G): Hartree (Ha)', 'Heat capacity at 298.15 K (Cv): cal/mol·K'
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
        if not os.listdir(self.qm9_folder):
            print(f"Extraindo dados para: {self.qm9_folder}")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.qm9_folder)
            print("Extração concluída.")
        else:
            print(f"Dados já existem em: {self.qm9_folder}")

    def inverse_transform_features(self, normalized_features, is_noise=False):
        """Desnormaliza features usando o scaler armazenado."""
        if is_noise:
            if hasattr(self, 'scaler_noise'):
                return self.scaler_noise.inverse_transform(normalized_features)
            else:
                raise AttributeError("O scaler de ruído não foi armazenado. Execute get_paired_dataloaders primeiro.")
        else: 
            if hasattr(self, 'scaler_normal'):
                return self.scaler_normal.inverse_transform(normalized_features)
            else:
                raise AttributeError("O scaler normal não foi armazenado. Execute get_paired_dataloaders primeiro.")

    def load_qm9_xyz(self, file_path):
        """Load a single QM9.xyz file."""
        with open(file_path, 'r') as f:
            natoms = int(f.readline())
            properties = list(map(float, f.readline().split()[2:]))
            atoms = []
            coordinates = []
            for num_line, line in enumerate(f):
                if num_line >= 0 and num_line < natoms:
                    info = line.replace("*^","e").split()
                    atoms.append(info[0])
                    coordinates.append(list(map(float, info[1:-1])))
        return {"natoms": natoms, "atoms": atoms, "coordinates": np.array(coordinates), "properties": properties}

    def load_qm9_dataset(self, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files."""
        coords, prop, natoms = [], [], []
        for file_name in sorted(os.listdir(self.directory_path)):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                molecule_data = self.load_qm9_xyz(file_path)
                if not list_mols or molecule_data['natoms'] in list_mols:
                    coords.append((molecule_data['atoms'], molecule_data['coordinates']))
                    prop.append(molecule_data['properties'])
                    natoms.append(molecule_data['natoms'])
        return coords, prop, natoms
    
    def dataset_to_numpy(dataset):
        """Convert a dataset to NumPy arrays."""
        all_data, all_targets = [], []
        for data, target in dataset:
            all_data.append(data.numpy())
            all_targets.append(target.numpy())
        return np.array(all_data), np.array(all_targets)

    def get_smiles(self):
        """Get the SMILES representation of a molecule."""
        smiles = []
        for file_name in sorted(os.listdir(self.directory_path)):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                with open(file_path, 'r') as f:
                    natoms = int(f.readline())
                    f.readline()
                    for i in range(natoms+1): f.readline()
                    smiles.append(tuple(f.readline().strip().split('\t'))[0])
        return smiles
        
    def load_smiles(self):
        """Load all SMILES representations in the QM9 dataset."""    
        list_smiles = []
        i = 0
        for file_name in sorted(os.listdir(self.directory_path)):
            if i == 100: break
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                list_smiles.append(self.get_smiles(file_path))
            i += 1
        return list_smiles    
    
    def df_props(self):
        """Create a DataFrame with the properties of the QM9 dataset."""
        _, props, _ = self.load_qm9_dataset()
        df = pd.DataFrame(props)
        df.reset_index(drop=True, inplace=True)
        df.columns = self.properties
        return df  
    
    def get_physicochemical_descriptors(self):
        """Calcula um conjunto de descritores 2D e retorna features e índices válidos."""
        smiles_list = self.get_smiles()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        
        descriptors = []
        valid_indices = []
        descriptor_names = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                            "NumRotatableBonds", "NumAromaticRings", "BalabanJ", "qed"]
        
        funcs = {name: getattr(Descriptors, name) for name in descriptor_names}

        for i, mol in enumerate(mols):
            if mol:
                desc_values = {name: func(mol) for name, func in funcs.items()}
                descriptors.append(desc_values)
                valid_indices.append(i)
        
        return pd.DataFrame(descriptors), valid_indices

    def get_3d_descriptors(self):
        """Calcula descritores 3D e retorna features e índices válidos."""
        smiles_list = self.get_smiles()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        
        descriptors_3d = []
        valid_indices = []
        descriptor_names = ["NPR1", "NPR2", "RadiusOfGyration", "Asphericity", "SpherocityIndex"]
        funcs = {name: getattr(Descriptors3D, name) for name in descriptor_names}
        
        for i, mol in enumerate(mols):
            if not mol: continue
            
            mol_with_hs = Chem.AddHs(mol)
            conf_id = AllChem.EmbedMolecule(mol_with_hs, randomSeed=42)
            
            if conf_id >= 0:
                AllChem.UFFOptimizeMolecule(mol_with_hs, confId=conf_id)
                desc_values = {name: func(mol_with_hs, confId=conf_id) for name, func in funcs.items()}
                descriptors_3d.append(desc_values)
                valid_indices.append(i)

        return pd.DataFrame(descriptors_3d), valid_indices

    def get_morgan_fingerprints(self, radius=3, n_bits=512):
        """Calcula os Morgan Fingerprints (ECFP) e retorna features e índices válidos."""
        smiles_list = self.get_smiles()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]

        fps = []
        valid_indices = []
        for i, mol in enumerate(mols):
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(np.array(list(fp)))
                valid_indices.append(i)
        return np.array(fps), valid_indices
        
    def get_coulomb_matrix(self, n_atoms_max=29, list_mols=[]):
        """Calcula a Matriz de Coulomb para as moléculas."""
        coords, _, natoms = self.load_qm9_dataset(list_mols=list_mols)
        mols_ase = [Atoms(positions=xyz, symbols=symbols) for symbols, xyz in coords]
        cm_generator = CoulombMatrix(n_atoms_max=n_atoms_max, permutation="eigenspectrum")
        X = cm_generator.create(mols_ase)
        return X
    
    class Data(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return torch.from_numpy(self.data[idx]).float(), torch.from_numpy(self.targets[idx]).float()

    def compute_descriptors(self, descriptor_type='CM', morgan_radius=3, morgan_nBits=512, att_index=10, list_mols=[]):
        """
        Calcula os descritores para as moléculas e retorna as features (X) e os alvos (Y) alinhados.
        
        Parameters:
        -----------
        descriptor_type : str
            Tipo de descritor ('CM', 'Morgan', 'Physicochemical', '3D')
        morgan_radius : int
            Raio para o cálculo dos fingerprints Morgan
        morgan_nBits : int
            Número de bits para os fingerprints Morgan
        att_index : int
            Índice da propriedade a ser predita
        list_mols : list
            Lista de moléculas a considerar
            
        Returns:
        --------
        tuple : (X, Y, props)
            - X: Array NumPy com os descritores
            - Y: Array NumPy com os valores alvo para a propriedade selecionada
            - props: Array NumPy com todas as propriedades (alinhado com X e Y)
        """
        # Carregar dados 
        coords, props_list, natoms = self.load_qm9_dataset(list_mols=list_mols)
        props = np.array(props_list)
        
        if descriptor_type == 'CM':
            X = self.get_coulomb_matrix(n_atoms_max=max(natoms) if natoms else 29, list_mols=list_mols)
            # Assume que get_coulomb_matrix retorna features para todas as moléculas carregadas
        
        elif descriptor_type == 'Morgan':
            X_morgan, valid_indices = self.get_morgan_fingerprints(radius=morgan_radius, n_bits=morgan_nBits)
            X = X_morgan
            props = props[valid_indices]

        elif descriptor_type == 'Physicochemical':
            X_df, valid_indices = self.get_physicochemical_descriptors()
            X = X_df.values
            props = props[valid_indices]
            
        elif descriptor_type == '3D':
            X_df, valid_indices = self.get_3d_descriptors()
            X = X_df.values
            props = props[valid_indices]

        else:
            raise ValueError(f"Tipo de descritor desconhecido: {descriptor_type}")
        
        # Garante que X e props tenham o mesmo número de amostras
        if X.shape[0] != props.shape[0]:
            raise ValueError(f"Inconsistência no número de amostras entre X ({X.shape[0]}) e props ({props.shape[0]})")
            
        # Extrair o valor alvo selecionado
        Y = props[:, att_index].reshape(-1, 1)
        
        return X, Y, props

    def normalize_data(self, X_normal, X_with_noise, Ys):
        """Normaliza os dados usando StandardScaler."""
        scaler_normal = StandardScaler()
        self.scaler_normal = scaler_normal
        X_normal_scaled = scaler_normal.fit_transform(X_normal)
        
        scaler_noise = StandardScaler()
        self.scaler_noise = scaler_noise
        X_noise_scaled = scaler_noise.fit_transform(X_with_noise)
        
        target_scaler = StandardScaler()
        self.target_scaler = target_scaler
        Ys_scaled = target_scaler.fit_transform(Ys)
        
        return X_normal_scaled, X_noise_scaled, Ys_scaled

    def create_dataloaders(self, X_normal_scaled, X_noise_scaled, Ys_scaled, 
                        split_ratio=[0.8, 0.1, 0.1], batch_size=256):
        """Cria os dataloaders a partir dos dados normalizados."""
        dataset_normal = self.Data(X_normal_scaled, Ys_scaled)
        dataset_noise = self.Data(X_noise_scaled, Ys_scaled)
        
        dataset_size = len(dataset_normal)
        train_size = int(dataset_size * split_ratio[0])
        val_size = int(dataset_size * split_ratio[1])
        
        all_indices = list(range(dataset_size))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size+val_size]
        test_indices = all_indices[train_size+val_size:]
        
        train_dataset_normal = torch.utils.data.Subset(dataset_normal, train_indices)
        val_dataset_normal = torch.utils.data.Subset(dataset_normal, val_indices)
        test_dataset_normal = torch.utils.data.Subset(dataset_normal, test_indices)
        
        train_dataset_noise = torch.utils.data.Subset(dataset_noise, train_indices)
        val_dataset_noise = torch.utils.data.Subset(dataset_noise, val_indices)
        test_dataset_noise = torch.utils.data.Subset(dataset_noise, test_indices)
        
        train_loader_normal = DataLoader(train_dataset_normal, batch_size=batch_size, shuffle=False)
        val_loader_normal = DataLoader(val_dataset_normal, batch_size=batch_size, shuffle=False)
        test_loader_normal = DataLoader(test_dataset_normal, batch_size=batch_size, shuffle=False)
        
        train_loader_noise = DataLoader(train_dataset_noise, batch_size=batch_size, shuffle=False)
        val_loader_noise = DataLoader(val_dataset_noise, batch_size=batch_size, shuffle=False)
        test_loader_noise = DataLoader(test_dataset_noise, batch_size=batch_size, shuffle=False)
        
        return (train_loader_normal, val_loader_normal, test_loader_normal,
                train_loader_noise, val_loader_noise, test_loader_noise)

    def get_paired_dataloaders(self, att_index=10, batch_size=256, descriptor_type='CM', 
                    list_mols=[], split_ratio=[0.8, 0.1, 0.1], seed=42, noise_type='gaussian', 
                    noise_scale=1.0, n_noise=1, morgan_radius=3, morgan_nBits=512):
        """Prepara dois conjuntos de DataLoaders (normal e com ruído) usando os descritores calculados."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_normal, Ys, props = self.compute_descriptors(
            descriptor_type=descriptor_type, morgan_radius=morgan_radius,
            morgan_nBits=morgan_nBits, att_index=att_index, list_mols=list_mols
        )
        
        X_with_noise = X_normal.copy()
        n_samples = X_with_noise.shape[0]
        
        if noise_type == 'gaussian':
            noise_features = np.random.normal(0, noise_scale, size=(n_samples, n_noise))
        elif noise_type == 'uniform':
            noise_features = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_noise))
        elif noise_type == 'binary':
            noise_features = np.random.choice([0, 1], size=(n_samples, n_noise))
        else:
            raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")
        
        X_with_noise = np.hstack((X_with_noise, noise_features))
        is_noise = np.zeros(X_with_noise.shape[1], dtype=bool)
        is_noise[-n_noise:] = True
        
        indices = np.arange(len(props))
        np.random.shuffle(indices)
        self.shuffled_indices = indices

        X_normal = X_normal[indices]
        X_with_noise = X_with_noise[indices]
        Ys = Ys[indices]
        
        X_normal_scaled, X_noise_scaled, Ys_scaled = self.normalize_data(X_normal, X_with_noise, Ys)
        
        dataloaders = self.create_dataloaders(
            X_normal_scaled, X_noise_scaled, Ys_scaled, 
            split_ratio=split_ratio, batch_size=batch_size)
        
        if n_noise > 0:
            return (*dataloaders, is_noise)
        else:
            return dataloaders[:3]

    def get_smiles_by_dataloader_idx(self, idx, dataset_type='test'):
        """Recupera o SMILES correspondente a um índice em um dataloader específico."""
        if not hasattr(self, 'shuffled_indices'):
            raise AttributeError("Índices embaralhados não disponíveis. Execute get_paired_dataloaders primeiro.")
        
        dataset_size = len(self.shuffled_indices)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)
        
        if dataset_type == 'train':
            if idx >= train_size: raise IndexError(f"Índice {idx} fora do alcance do conjunto de treino")
            original_idx = self.shuffled_indices[idx]
        elif dataset_type == 'val':
            if idx >= val_size: raise IndexError(f"Índice {idx} fora do alcance do conjunto de validação")
            original_idx = self.shuffled_indices[train_size + idx]
        elif dataset_type == 'test':
            test_size = dataset_size - train_size - val_size
            if idx >= test_size: raise IndexError(f"Índice {idx} fora do alcance do conjunto de teste")
            original_idx = self.shuffled_indices[train_size + val_size + idx]
        else:
            raise ValueError("dataset_type deve ser 'train', 'val' ou 'test'")
        
        all_smiles = self.get_smiles()
        return all_smiles[original_idx]


if __name__ == '__main__':
    qm9 = qm9_tabular()
    # 10 = 'Internal energy at 0 K (U0)'
    train_loader, val_loader, test_loader, train_loader_noise, val_loader_noise, test_loader_noise, is_noise = qm9.get_paired_dataloaders(descriptor_type='Morgan')
    print(train_loader.dataset[0])
    print(train_loader_noise.dataset[0])
