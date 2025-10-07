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
from sklearn.cluster import KMeans
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
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any, Union



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
        
        # Criar pasta data se não existir
        os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

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

    # Add the missing Data class that's used in create_dataloaders
    class Data(Dataset):
        """
        PyTorch Dataset class for handling tabular molecular data.
        
        Parameters:
        -----------
        x : numpy.ndarray or torch.Tensor
            Feature data (descriptors)
        y : numpy.ndarray or torch.Tensor
            Target values (properties)
        """
        def __init__(self, x, y):
            if isinstance(x, np.ndarray):
                self.x = torch.from_numpy(x).float()
            else:
                self.x = x.float()
                
            if isinstance(y, np.ndarray):
                self.y = torch.from_numpy(y).float()
            else:
                self.y = y.float()
            
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
    
    def inverse_transform_features(self, normalized_features, is_noise=False):
        """
        Desnormaliza features usando o scaler armazenado.
        
        Parameters:
        -----------
        normalized_features : np.ndarray
            Features normalizadas a serem convertidas de volta
        is_noise : bool
            Se True, usa o scaler para dados com ruído, senão usa o scaler normal
            
        Returns:
        --------
        np.ndarray: Features desnormalizadas
        
        Raises:
        -------
        AttributeError: Se o scaler requisitado não estiver disponível
        """
        scaler = getattr(self, f'scaler_{"noise" if is_noise else "normal"}', None)
        if scaler is not None:
            return scaler.inverse_transform(normalized_features)
        else:
            raise AttributeError(f"O scaler {'de ruído' if is_noise else 'normal'} não foi armazenado. "
                                 "Execute get_paired_dataloaders_tabular primeiro.")

    @lru_cache(maxsize=64)
    def load_qm9_xyz(self, file_path):
        """
        Carrega um arquivo QM9.xyz e extrai suas informações com cache para melhorar a performance.
        
        Parameters:
        -----------
        file_path : str
            Caminho para o arquivo .xyz
            
        Returns:
        --------
        dict: Dicionário contendo informações da molécula
        """
        try:
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
        except Exception as e:
            print(f"Erro ao carregar o arquivo {file_path}: {e}")
            return {"natoms": 0, "atoms": [], "coordinates": np.array([]), "properties": []}

    def load_qm9_dataset(self, list_mols=None):
        """
        Carrega o dataset QM9 completo de arquivos .xyz com otimização de desempenho.
        
        Parameters:
        -----------
        list_mols : list, optional
            Lista de moléculas (por número de átomos) a considerar. Se vazio, carrega todas.
            
        Returns:
        --------
        tuple: (coordenadas, propriedades, número de átomos)
        """
        if list_mols is None:
            list_mols = []
            
        # Usar cache de disco se disponível
        cache_path = os.path.join(os.getcwd(), "data", "qm9_dataset_cache.pkl")
        if os.path.exists(cache_path) and not list_mols:
            try:
                with open(cache_path, 'rb') as f:
                    print(f"Carregando QM9 do cache: {cache_path}")
                    return pickle.load(f)
            except Exception as e:
                print(f"Erro ao carregar cache: {e}. Carregando dados brutos.")
        
        coords, prop, natoms = [], [], []
        
        # Listar todos os arquivos .xyz
        xyz_files = sorted([f for f in os.listdir(self.directory_path) if f.endswith(".xyz")])
        total_files = len(xyz_files)
        
        print(f"Carregando {total_files} arquivos .xyz...")
        
        # Função para processar um arquivo
        def process_file(file_name):
            file_path = os.path.join(self.directory_path, file_name)
            molecule_data = self.load_qm9_xyz(file_path)
            
            if not list_mols or molecule_data['natoms'] in list_mols:
                return (
                    (molecule_data['atoms'], molecule_data['coordinates']), 
                    molecule_data['properties'],
                    molecule_data['natoms']
                )
            return None
            
        # Processamento paralelo para arquivos grandes
        if total_files > 1000:
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                results = list(tqdm(
                    executor.map(process_file, xyz_files),
                    total=total_files,
                    desc="Carregando moléculas"
                ))
                
            # Filtrar resultados None e separar dados
            results = [r for r in results if r is not None]
            coords = [r[0] for r in results]
            prop = [r[1] for r in results]
            natoms = [r[2] for r in results]
        else:
            # Processamento sequencial para conjuntos menores
            for file_name in tqdm(xyz_files, desc="Carregando moléculas"):
                result = process_file(file_name)
                if result:
                    coords.append(result[0])
                    prop.append(result[1])
                    natoms.append(result[2])
        
        # Salvar cache apenas se carregarmos todo o conjunto
        if not list_mols:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump((coords, prop, natoms), f)
                print(f"Cache do dataset salvo em: {cache_path}")
            except Exception as e:
                print(f"Aviso: Não foi possível salvar o cache: {e}")
            
        return coords, prop, natoms
    
    @staticmethod
    def dataset_to_numpy(dataset):
        """
        Converte um dataset PyTorch para arrays NumPy.
        
        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            Dataset a ser convertido
            
        Returns:
        --------
        tuple: (dados_numpy, alvos_numpy)
        """
        all_data, all_targets = [], []
        
        # Usar loader para processar em batches para datasets grandes
        if len(dataset) > 10000:
            loader = DataLoader(dataset, batch_size=1000, num_workers=4)
            for data, target in loader:
                all_data.append(data.numpy())
                all_targets.append(target.numpy())
            return np.vstack(all_data), np.vstack(all_targets)
        else:
            # Método direto para datasets menores
            for data, target in dataset:
                all_data.append(data.numpy())
                all_targets.append(target.numpy())
            return np.array(all_data), np.array(all_targets)

    def get_smiles(self, max_mols=None):
        """
        Obtém as representações SMILES para todas as moléculas com melhor desempenho.
        
        Parameters:
        -----------
        max_mols : int, optional
            Número máximo de moléculas a processar (None para todas)
            
        Returns:
        --------
        list: Lista de strings SMILES
        """
        # Tentar carregar do cache
        cache_path = os.path.join(os.getcwd(), "data", "qm9_smiles_cache.pkl")
        if os.path.exists(cache_path) and max_mols is None:
            try:
                with open(cache_path, 'rb') as f:
                    print(f"Carregando SMILES do cache: {cache_path}")
                    return pickle.load(f)
            except Exception as e:
                print(f"Erro ao carregar cache de SMILES: {e}")
        
        smiles = []
        file_list = sorted(os.listdir(self.directory_path))
        xyz_files = [f for f in file_list if f.endswith(".xyz")]
        
        if max_mols is not None:
            xyz_files = xyz_files[:max_mols]
        
        # Função para extrair SMILES de um arquivo
        def extract_smiles(file_path):
            try:
                with open(file_path, 'r') as f:
                    natoms = int(f.readline())
                    f.readline()  # Propriedades
                    # Pular linhas de coordenadas
                    for _ in range(natoms+1):
                        f.readline()
                    # Ler SMILES
                    smiles_line = f.readline().strip().split('\t')
                    return smiles_line[0] if smiles_line else ""
            except Exception:
                return ""
        
        # Processamento paralelo para conjuntos grandes
        if len(xyz_files) > 1000:
            file_paths = [os.path.join(self.directory_path, f) for f in xyz_files]
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                smiles = list(tqdm(
                    executor.map(extract_smiles, file_paths), 
                    total=len(file_paths),
                    desc="Extraindo SMILES"
                ))
        else:
            # Processamento sequencial para conjuntos pequenos
            for file_name in tqdm(xyz_files, desc="Extraindo SMILES"):
                file_path = os.path.join(self.directory_path, file_name)
                smiles.append(extract_smiles(file_path))
        
        # Salvar cache apenas se processarmos todo o conjunto
        if max_mols is None:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(smiles, f)
                print(f"Cache de SMILES salvo em: {cache_path}")
            except Exception as e:
                print(f"Aviso: Não foi possível salvar o cache de SMILES: {e}")
        
        return smiles
    
    def df_props(self, include_smiles=False):
        """
        Cria um DataFrame com as propriedades do dataset QM9 com opção de incluir SMILES.
        
        Parameters:
        -----------
        include_smiles : bool, optional
            Se True, inclui uma coluna com os SMILES
            
        Returns:
        --------
        pd.DataFrame: DataFrame com propriedades moleculares
        """
        _, props, _ = self.load_qm9_dataset()
        df = pd.DataFrame(props, columns=self.properties)
        
        if include_smiles:
            df['SMILES'] = self.get_smiles()
            
        df.reset_index(drop=True, inplace=True)
        return df
    
    def get_coulomb_matrix(self, n_atoms_max=29, list_mols=[], n_jobs=4):
        """
        Calcula a matriz de Coulomb para as moléculas usando DScribe com processamento paralelo.
        
        Parameters:
        -----------
        n_atoms_max : int
            Número máximo de átomos para normalizar o tamanho da matriz
        list_mols : list
            Lista de moléculas por número de átomos (vazio para todas)
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        np.ndarray: Matrizes de Coulomb achatadas
        """
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"coulomb_matrix_n{n_atoms_max}_cache.npz")
        if os.path.exists(cache_path) and not list_mols:
            try:
                data = np.load(cache_path)
                print(f"Carregando matrizes de Coulomb do cache: {cache_path}")
                return data['cm_matrices']
            except Exception as e:
                print(f"Erro ao carregar cache de matrizes de Coulomb: {e}")
        
        # Carregar dados moleculares
        coords, props, natoms = self.load_qm9_dataset(list_mols=list_mols)
        
        # Configurar descritor de matriz de Coulomb
        cm_descriptor = CoulombMatrix(n_atoms_max=n_atoms_max, permutation="sorted_l2")
        
        # Função para processar uma molécula
        def process_molecule(mol_idx):
            try:
                atoms_list, coordinates = coords[mol_idx]
                
                # Criar objeto Atoms do ASE
                molecule = Atoms(symbols=atoms_list, positions=coordinates)
                
                # Calcular matriz de Coulomb
                cm_matrix = cm_descriptor.create(molecule)
                
                return mol_idx, cm_matrix
            except Exception as e:
                print(f"Erro ao processar molécula {mol_idx}: {e}")
                return None
        
        # Processamento paralelo
        cm_matrices = []
        valid_indices = []
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(coords))]
            
            for future in tqdm(futures, desc="Calculando matrizes de Coulomb", total=len(futures)):
                result = future.result()
                if result:
                    mol_idx, cm_matrix = result
                    cm_matrices.append(cm_matrix)
                    valid_indices.append(mol_idx)
        
        cm_matrices = np.array(cm_matrices)
        
        # Salvar cache apenas se processarmos todo o conjunto
        if not list_mols:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez_compressed(cache_path, cm_matrices=cm_matrices, valid_indices=valid_indices)
                print(f"Cache de matrizes de Coulomb salvo em: {cache_path}")
            except Exception as e:
                print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        print(f"Matrizes de Coulomb calculadas para {len(cm_matrices)} moléculas")
        
        return cm_matrices

    def get_physicochemical_descriptors(self, n_jobs=4):
        """
        Calcula descritores físico-químicos 2D usando RDKit com processamento paralelo.
        
        Parameters:
        -----------
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (pd.DataFrame de descritores, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        
        descriptor_names = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                          "NumRotatableBonds", "NumAromaticRings", "BalabanJ", "qed"]
        
        funcs = {name: getattr(Descriptors, name) for name in descriptor_names}
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            mol = mols[mol_idx]
            if mol:
                try:
                    desc_values = {name: func(mol) for name, func in funcs.items()}
                    return mol_idx, desc_values
                except Exception:
                    return None
            return None
        
        # Processar em paralelo
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_molecule, range(len(mols))))
        
        # Filtrar resultados None
        results = [r for r in results if r is not None]
        valid_indices = [r[0] for r in results]
        descriptors = [r[1] for r in results]
        
        return pd.DataFrame(descriptors), valid_indices
    
    def get_3d_descriptors(self, n_jobs=4, random_seed=42):
        """
        Calcula descritores 3D usando RDKit com melhor tratamento de erros e paralelização.
        
        Parameters:
        -----------
        n_jobs : int
            Número de threads para processamento paralelo
        random_seed : int
            Semente para geração de conformações 3D
            
        Returns:
        --------
        tuple: (pd.DataFrame de descritores 3D, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        
        descriptor_names = ["NPR1", "NPR2", "RadiusOfGyration", "Asphericity", "SpherocityIndex"]
        funcs = {name: getattr(Descriptors3D, name) for name in descriptor_names}
        
        # Processar uma molécula por vez com tratamento de erros
        def process_molecule(mol_idx):
            mol = mols[mol_idx]
            if not mol:
                return None
            
            try:
                mol_with_hs = Chem.AddHs(mol)
                conf_id = AllChem.EmbedMolecule(mol_with_hs, randomSeed=random_seed)
                
                if conf_id < 0:
                    return None
                    
                # Otimização da conformação
                AllChem.UFFOptimizeMolecule(mol_with_hs, confId=conf_id)
                
                # Calcular descritores 3D
                desc_values = {name: func(mol_with_hs, confId=conf_id) for name, func in funcs.items()}
                
                return mol_idx, desc_values
            except Exception:
                return None
        
        # Processar em paralelo com barra de progresso
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(mols))]
            
            for future in tqdm(futures, desc="Calculando descritores 3D", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Filtrar resultados válidos
        valid_indices = [r[0] for r in results]
        descriptors = [r[1] for r in results]
        
        print(f"Descritores 3D calculados para {len(valid_indices)} moléculas de {len(mols)}")
        return pd.DataFrame(descriptors), valid_indices
    
    def get_morgan_fingerprints(self, radius=3, n_bits=512, n_jobs=4):
        """
        Calcula fingerprints Morgan (ECFP) com processamento paralelo otimizado.
        
        Parameters:
        -----------
        radius : int
            Raio para o cálculo do fingerprint
        n_bits : int
            Número de bits para o fingerprint
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"morgan_fp_r{radius}_n{n_bits}_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints Morgan do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    return mol_idx, np.array(list(fp))
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints Morgan", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints Morgan salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
        
    def get_maccs_keys(self, n_jobs=4):
        """
        Calcula os fingerprints MACCS Keys (166 bits) para as moléculas.
        
        Parameters:
        -----------
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", "maccs_keys_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando MACCS Keys do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de MACCS Keys: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = Chem.MACCSkeys.GenMACCSKeys(mol)
                    return mol_idx, np.array(list(fp))
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando MACCS Keys", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de MACCS Keys salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
    
    def get_topological_fingerprints(self, n_bits=2048, n_jobs=4):
        """
        Calcula fingerprints topológicos (tipo Daylight) para as moléculas.
        
        Parameters:
        -----------
        n_bits : int
            Tamanho do fingerprint em bits
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"topological_fp_n{n_bits}_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints topológicos do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints topológicos: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                    return mol_idx, np.array(list(fp))
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints topológicos", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints topológicos salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
        
    def get_atom_pair_fingerprints(self, n_bits=2048, n_jobs=4):
        """
        Calcula fingerprints de pares de átomos para as moléculas.
        
        Parameters:
        -----------
        n_bits : int
            Tamanho do fingerprint em bits
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        from rdkit.Chem import AllChem
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"atom_pair_fp_n{n_bits}_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints de pares de átomos do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints de pares de átomos: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
                    return mol_idx, np.array(list(fp))
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints de pares de átomos", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints de pares de átomos salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
    
    def get_estate_fingerprints(self, n_jobs=4):
        """
        Calcula fingerprints EState para as moléculas.
        
        Parameters:
        -----------
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        from rdkit.Chem import EState
        from rdkit.Chem.EState import Fingerprinter
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", "estate_fp_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints EState do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints EState: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = Fingerprinter.FingerprintMol(mol)[0]
                    # Converter para array binário
                    binary_fp = [1 if x else 0 for x in fp]
                    return mol_idx, np.array(binary_fp)
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints EState", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints EState salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
    
    def get_pattern_fingerprints(self, n_bits=2048, n_jobs=4):
        """
        Calcula fingerprints de padrões SMARTS para as moléculas.
        
        Parameters:
        -----------
        n_bits : int
            Tamanho do fingerprint em bits
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"pattern_fp_n{n_bits}_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints de padrões do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints de padrões: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
                    return mol_idx, np.array(list(fp))
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints de padrões", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints de padrões salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
        
    def get_avalon_fingerprints(self, n_bits=1024, n_jobs=4):
        """
        Calcula fingerprints Avalon para as moléculas.
        
        Parameters:
        -----------
        n_bits : int
            Tamanho do fingerprint em bits
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        try:
            from rdkit.Avalon import pyAvalonTools
        except ImportError:
            print("Aviso: O módulo Avalon não está disponível. Instale com 'conda install -c rdkit avalon-toolkit'")
            return np.array([]), []
            
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"avalon_fp_n{n_bits}_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints Avalon do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints Avalon: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
                    return mol_idx, np.array(list(fp))
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints Avalon", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints Avalon salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
        
    def get_morgan_count_fingerprints(self, radius=3, n_bits=1024, n_jobs=4):
        """
        Calcula fingerprints Morgan com contagem (ECFP com contagem de ocorrências) para as moléculas.
        
        Parameters:
        -----------
        radius : int
            Raio para o cálculo do fingerprint
        n_bits : int
            Número de bits para o fingerprint
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de fingerprints, lista de índices válidos)
        """
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", f"morgan_count_fp_r{radius}_n{n_bits}_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando fingerprints Morgan com contagem do cache: {cache_path}")
                return data['fps'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de fingerprints Morgan com contagem: {e}")
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Usar o GetHashedMorganFingerprint em vez do GetMorganFingerprintAsBitVect
                    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
                    # Converter para array
                    arr = np.zeros((n_bits,), dtype=np.int32)
                    for idx, count in fp.GetNonzeroElements().items():
                        arr[idx % n_bits] += int(count)
                    return mol_idx, arr
            except:
                pass
            return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando fingerprints Morgan com contagem", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        fps = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, fps=fps, valid_indices=valid_indices)
            print(f"Cache de fingerprints Morgan com contagem salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return fps, valid_indices
        
    def get_autocorr_descriptors(self, n_jobs=4):
        """
        Calcula descritores de autocorrelação 2D para as moléculas.
        Esses descritores capturam a distribuição de propriedades atômicas ao longo da molécula.
        
        Parameters:
        -----------
        n_jobs : int
            Número de threads para processamento paralelo
            
        Returns:
        --------
        tuple: (np.ndarray de descritores, lista de índices válidos)
        """
        from rdkit.Chem import Descriptors, Lipinski
        from rdkit.Chem.AtomPairs import Utils
        import rdkit.Chem.EState.EState as EState
        
        smiles_list = self.get_smiles()
        
        # Verificar cache
        cache_path = os.path.join(os.getcwd(), "data", "autocorr_desc_cache.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"Carregando descritores de autocorrelação do cache: {cache_path}")
                return data['descs'], data['valid_indices']
            except Exception as e:
                print(f"Erro ao carregar cache de descritores de autocorrelação: {e}")
        
        # Propriedades atômicas a serem usadas
        properties = {
            'AtomicNumber': lambda a: a.GetAtomicNum(),
            'Mass': lambda a: a.GetMass(),
            'Valence': lambda a: a.GetTotalValence(),
            'Charge': lambda a: a.GetFormalCharge(),
            'HCount': lambda a: a.GetTotalNumHs(),
            'Aromatic': lambda a: 1 if a.GetIsAromatic() else 0,
            'EState': lambda a: EState.AtomTypes.EStateIndices(Chem.GetPeriodicTable())[a.GetAtomicNum()-1],
        }
        
        # Processar uma molécula por vez
        def process_molecule(mol_idx):
            smiles = smiles_list[mol_idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    return None
                
                # Calcular distâncias topológicas entre todos os pares de átomos
                dm = Chem.GetDistanceMatrix(mol)
                n_atoms = mol.GetNumAtoms()
                
                # Calcular autocorrelações para cada propriedade e distância
                autocorr = []
                max_dist = 5  # Máxima distância topológica a considerar
                
                for prop_name, prop_func in properties.items():
                    # Calcular propriedade para cada átomo
                    atom_props = [prop_func(mol.GetAtomWithIdx(i)) for i in range(n_atoms)]
                    
                    for dist in range(max_dist + 1):
                        # Autocorrelação para distância dist
                        ac_value = 0
                        count = 0
                        for i in range(n_atoms):
                            for j in range(i+1, n_atoms):
                                if dm[i][j] == dist:
                                    ac_value += atom_props[i] * atom_props[j]
                                    count += 1
                        
                        if count > 0:
                            ac_value /= count
                        
                        autocorr.append(ac_value)
                
                return mol_idx, np.array(autocorr)
            except Exception as e:
                print(f"Erro ao processar molécula {mol_idx}: {e}")
                return None
        
        # Processamento paralelo
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_molecule, i) for i in range(len(smiles_list))]
            
            for future in tqdm(futures, desc="Calculando descritores de autocorrelação", total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        # Organizar resultados
        valid_indices = [r[0] for r in results]
        descs = np.array([r[1] for r in results])
        
        # Salvar cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, descs=descs, valid_indices=valid_indices)
            print(f"Cache de descritores de autocorrelação salvo em: {cache_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o cache: {e}")
        
        return descs, valid_indices
    
    def compute_descriptors(self, descriptor_type='CM', morgan_radius=3, morgan_nBits=512, att_index=10, list_mols=[]):
        """
        Calcula os descritores para as moléculas e retorna as features (X) e os alvos (Y) alinhados.
        
        Parameters:
        -----------
        descriptor_type : str
            Tipo de descritor ('CM', 'Morgan', 'Physicochemical', '3D', 'MACCS', 'Topological', 
            'AtomPair', 'EState', 'Pattern', 'Avalon', 'MorganCount', 'Autocorr')
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
        
        # Novos tipos de descritores
        elif descriptor_type == 'MACCS':
            X_maccs, valid_indices = self.get_maccs_keys()
            X = X_maccs
            props = props[valid_indices]
            
        elif descriptor_type == 'Topological':
            X_topo, valid_indices = self.get_topological_fingerprints(n_bits=morgan_nBits)
            X = X_topo
            props = props[valid_indices]
            
        elif descriptor_type == 'AtomPair':
            X_ap, valid_indices = self.get_atom_pair_fingerprints(n_bits=morgan_nBits)
            X = X_ap
            props = props[valid_indices]
            
        elif descriptor_type == 'EState':
            X_estate, valid_indices = self.get_estate_fingerprints()
            X = X_estate
            props = props[valid_indices]
            
        elif descriptor_type == 'Pattern':
            X_pattern, valid_indices = self.get_pattern_fingerprints(n_bits=morgan_nBits)
            X = X_pattern
            props = props[valid_indices]
            
        elif descriptor_type == 'Avalon':
            X_avalon, valid_indices = self.get_avalon_fingerprints(n_bits=morgan_nBits)
            X = X_avalon
            props = props[valid_indices]
            
        elif descriptor_type == 'MorganCount':
            X_morgan_count, valid_indices = self.get_morgan_count_fingerprints(radius=morgan_radius, n_bits=morgan_nBits)
            X = X_morgan_count
            props = props[valid_indices]
            
        elif descriptor_type == 'Autocorr':
            X_autocorr, valid_indices = self.get_autocorr_descriptors()
            X = X_autocorr
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
        """
        Normaliza os dados usando StandardScaler com tratamento de erros melhorado.
        
        Parameters:
        -----------
        X_normal : np.ndarray
            Matriz de features sem ruído
        X_with_noise : np.ndarray
            Matriz de features com ruído
        Ys : np.ndarray
            Valores alvo
            
        Returns:
        --------
        tuple: (X_normal_scaled, X_noise_scaled, Ys_scaled)
        """
        # Verificar entradas
        for name, data in [("X_normal", X_normal), ("X_with_noise", X_with_noise), ("Ys", Ys)]:
            if data is None or (hasattr(data, 'size') and data.size == 0):
                raise ValueError(f"Dados de entrada {name} vazios ou inválidos")
        
        try:
            # Normalizar features normais
            scaler_normal = StandardScaler()
            X_normal_scaled = scaler_normal.fit_transform(X_normal)
            self.scaler_normal = scaler_normal
            
            # Normalizar features com ruído
            scaler_noise = StandardScaler()
            X_noise_scaled = scaler_noise.fit_transform(X_with_noise)
            self.scaler_noise = scaler_noise
            
            # Normalizar alvos
            target_scaler = StandardScaler()
            Ys_scaled = target_scaler.fit_transform(Ys)
            self.target_scaler = target_scaler
            
            # Verificar dados normalizados
            for name, data in [("X_normal_scaled", X_normal_scaled), 
                              ("X_noise_scaled", X_noise_scaled), 
                              ("Ys_scaled", Ys_scaled)]:
                if np.isnan(data).any() or np.isinf(data).any():
                    print(f"Aviso: Valores NaN ou Inf detectados em {name} após normalização")
                    # Substituir valores problemáticos
                    data = np.nan_to_num(data)
            
            return X_normal_scaled, X_noise_scaled, Ys_scaled
            
        except Exception as e:
            print(f"Erro durante normalização: {e}")
            raise
    
    def create_dataloaders(self, X_normal_scaled, X_noise_scaled, Ys_scaled, 
                         split_ratio=[0.8, 0.1, 0.1], batch_size=256, n_noise=0,
                         shuffle_train=True, num_workers=4):
        """
        Cria DataLoaders a partir de dados normalizados com opções melhoradas.
        
        Parameters:
        -----------
        X_normal_scaled : np.ndarray
            Matriz de features normalizadas sem ruído
        X_noise_scaled : np.ndarray
            Matriz de features normalizadas com ruído
        Ys_scaled : np.ndarray
            Valores alvo normalizados
        split_ratio : list
            Proporções para divisão dos dados [treino, validação, teste]
        batch_size : int
            Tamanho do lote
        n_noise : int
            Número de features de ruído (0 para não usar dados com ruído)
        shuffle_train : bool
            Se True, embaralha os dados de treino
        num_workers : int
            Número de workers para DataLoader
            
        Returns:
        --------
        tuple: DataLoaders para treino, validação e teste (com/sem ruído)
        """
        # Verificar entradas
        if sum(split_ratio) != 1.0:
            print(f"Aviso: As proporções de divisão ({split_ratio}) não somam 1.0. Normalizando.")
            total = sum(split_ratio)
            split_ratio = [r/total for r in split_ratio]
            
        # Criar dataset para dados normais
        dataset_normal = self.Data(X_normal_scaled, Ys_scaled)
        
        # Calcular tamanhos das divisões
        dataset_size = len(dataset_normal)
        train_size = int(dataset_size * split_ratio[0])
        val_size = int(dataset_size * split_ratio[1])
        test_size = dataset_size - train_size - val_size  # Para garantir que soma = total
        
        # Verificar divisões
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(f"Divisão inválida: train={train_size}, val={val_size}, test={test_size}")
        
        # Gerar índices de divisão
        all_indices = list(range(dataset_size))
        
        # Criar subsets
        train_dataset_normal = torch.utils.data.Subset(dataset_normal, all_indices[:train_size])
        val_dataset_normal = torch.utils.data.Subset(dataset_normal, all_indices[train_size:train_size+val_size])
        test_dataset_normal = torch.utils.data.Subset(dataset_normal, all_indices[train_size+val_size:])
        
        # Criar DataLoaders para dados normais
        train_loader_normal = DataLoader(
            train_dataset_normal, 
            batch_size=batch_size, 
            shuffle=shuffle_train, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader_normal = DataLoader(
            val_dataset_normal, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader_normal = DataLoader(
            test_dataset_normal, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Se não houver ruído, retorna apenas loaders normais
        if n_noise <= 0:
            return train_loader_normal, val_loader_normal, test_loader_normal
        
        # Criar dataset e loaders para dados com ruído
        dataset_noise = self.Data(X_noise_scaled, Ys_scaled)
        
        train_dataset_noise = torch.utils.data.Subset(dataset_noise, all_indices[:train_size])
        val_dataset_noise = torch.utils.data.Subset(dataset_noise, all_indices[train_size:train_size+val_size])
        test_dataset_noise = torch.utils.data.Subset(dataset_noise, all_indices[train_size+val_size:])
        
        train_loader_noise = DataLoader(
            train_dataset_noise, 
            batch_size=batch_size, 
            shuffle=shuffle_train, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader_noise = DataLoader(
            val_dataset_noise, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader_noise = DataLoader(
            test_dataset_noise, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return (
            train_loader_normal, val_loader_normal, test_loader_normal,
            train_loader_noise, val_loader_noise, test_loader_noise
        )

    def get_paired_dataloaders(self, att_index=10, batch_size=256, descriptor_type='CM', 
                    list_mols=[], split_ratio=[0.8, 0.1, 0.1], seed=42, noise_type='gaussian', 
                    noise_scale=1.0, n_noise=1, morgan_radius=3, morgan_nBits=512, add_noise=True):
        """Prepara dois conjuntos de DataLoaders (normal e com ruído) usando os descritores calculados."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_normal, Ys, props = self.compute_descriptors(
            descriptor_type=descriptor_type, morgan_radius=morgan_radius,
            morgan_nBits=morgan_nBits, att_index=att_index, list_mols=list_mols
        )
        
        if not add_noise or n_noise == 0:
            # Apenas dataloaders normais, sem ruído
            X_normal_scaled, _, Ys_scaled = self.normalize_data(X_normal, X_normal, Ys)
            dataloaders = self.create_dataloaders(
                X_normal_scaled, X_normal_scaled, Ys_scaled, 
                split_ratio=split_ratio, batch_size=batch_size, n_noise=0)
            return dataloaders

        # Caso queira adicionar ruído
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
            split_ratio=split_ratio, batch_size=batch_size, n_noise=n_noise)
        
        return dataloaders

    def get_paired_dataloaders_tabular(self, att_index=10, batch_size=256, descriptor_type='CM', 
                    list_mols=[], split_ratio=[0.8, 0.1, 0.1], seed=42, noise_type='gaussian', 
                    noise_scale=1.0, n_noise=1, morgan_radius=3, morgan_nBits=512, add_noise=True,
                    cache_descriptors=True):
        """
        Prepara dois conjuntos de DataLoaders (normal e com ruído) usando os descritores calculados,
        com otimizações para melhor performance e uso de memória.
        
        Parameters:
        -----------
        att_index : int
            Índice da propriedade a ser predita (default: 10)
        batch_size : int
            Tamanho do lote (default: 256)
        descriptor_type : str
            Tipo de descritor ('CM', 'Morgan', 'Physicochemical', '3D')
        list_mols : list
            Lista de moléculas a considerar (default: [])
        split_ratio : list
            Razões para divisão dos dados [treino, validação, teste]
        seed : int
            Semente para reprodutibilidade
        noise_type : str
            Tipo de ruído ('gaussian', 'uniform', 'binary')
        noise_scale : float
            Escala do ruído a ser adicionado
        n_noise : int
            Número de features de ruído a adicionar
        morgan_radius : int
            Raio para descritores Morgan
        morgan_nBits : int
            Número de bits para descritores Morgan
        add_noise : bool
            Se deve adicionar ruído aos dados
        cache_descriptors : bool
            Se deve armazenar em cache os descritores calculados
            
        Returns:
        --------
        tuple : Conjunto de DataLoaders organizados em tuplas:
            (train_loader_normal, val_loader_normal, test_loader_normal,
             train_loader_noise, val_loader_noise, test_loader_noise, is_noise_mask)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Caminho para cache de descritores
        cache_dir = os.path.join(os.getcwd(), "data", "descriptor_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, 
            f"{descriptor_type}_r{morgan_radius}_b{morgan_nBits}_a{att_index}_cache.npz"
        )
        
        # Tentar carregar do cache se habilitado
        if cache_descriptors and os.path.exists(cache_file):
            print(f"Carregando descritores do cache: {cache_file}")
            cached_data = np.load(cache_file, allow_pickle=True)
            X_normal = cached_data['X_normal']
            Ys = cached_data['Ys']
            props = cached
        else:
            # Calcular descritores
            X_normal, Ys, props = self.compute_descriptors(
                descriptor_type=descriptor_type, 
                morgan_radius=morgan_radius,
                morgan_nBits=morgan_nBits, 
                att_index=att_index, 
                list_mols=list_mols
            )
            
            # Salvar no cache se habilitado
            if cache_descriptors:
                print(f"Salvando descritores em cache: {cache_file}")
                np.savez_compressed(
                    cache_file,
                    X_normal=X_normal,
                    Ys=Ys,
                    props=props
                )
        
        if not add_noise or n_noise == 0:
            # Apenas dataloaders normais, sem ruído
            X_normal_scaled, _, Ys_scaled = self.normalize_data(X_normal, X_normal, Ys)
            dataloaders = self.create_dataloaders(
                X_normal_scaled, X_normal_scaled, Ys_scaled, 
                split_ratio=split_ratio, batch_size=batch_size, n_noise=0)
            
            # Retornar formato completo para compatibilidade
            return (*dataloaders, None)  # Adiciona None como is_noise_mask

        # Caso queira adicionar ruído
        X_with_noise = X_normal.copy()
        n_samples, n_features = X_with_noise.shape
        
        # Geração de ruído otimizada
        if noise_type == 'gaussian':
            noise_features = np.random.normal(0, noise_scale, size=(n_samples, n_noise))
        elif noise_type == 'uniform':
            noise_features = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_noise))
        elif noise_type == 'binary':
            noise_features = np.random.choice([0, 1], size=(n_samples, n_noise))
        else:
            raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")
        
        X_with_noise = np.hstack((X_with_noise, noise_features))
        
        # Máscara para identificar features de ruído
        is_noise = np.zeros(X_with_noise.shape[1], dtype=bool)
        is_noise[-n_noise:] = True
        
        # Embaralhar os dados de maneira consistente
        indices = np.arange(len(props))
        np.random.shuffle(indices)
        self.shuffled_indices = indices

        X_normal = X_normal[indices]
        X_with_noise = X_with_noise[indices]
        Ys = Ys[indices]
        
        # Normalizar os dados
        X_normal_scaled, X_noise_scaled, Ys_scaled = self.normalize_data(X_normal, X_with_noise, Ys)
        
        # Criar dataloaders
        dataloaders = self.create_dataloaders(
            X_normal_scaled, X_noise_scaled, Ys_scaled, 
            split_ratio=split_ratio, batch_size=batch_size, n_noise=n_noise)
        
        return (*dataloaders, is_noise)

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

    def get_descriptor_names(self, descriptor_type='Morgan', morgan_radius=2, morgan_nBits=512):
        """
        Retorna os nomes das features para um tipo específico de descritor.
        
        Parameters:
        -----------
        descriptor_type : str
            Tipo de descritor ('CM', 'Morgan', 'Physicochemical', '3D', 'MACCS', 
            'Topological', 'AtomPair', 'EState', 'Pattern', 'Avalon', 'MorganCount', 'Autocorr')
        morgan_radius : int
            Raio para os descritores Morgan (usado apenas para 'Morgan' e 'MorganCount')
        morgan_nBits : int
            Número de bits para fingerprints (usado para vários tipos de fingerprints)
            
        Returns:
        --------
        list: Lista com os nomes das features para o descritor especificado
        """
        if descriptor_type == 'Morgan' or descriptor_type == 'MorganCount':
            return [f"Morgan{morgan_radius}_Bit_{i}" for i in range(morgan_nBits)]
            
        elif descriptor_type == 'Physicochemical':
            return ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                    "NumRotatableBonds", "NumAromaticRings", "BalabanJ", "QED"]
                    
        elif descriptor_type == '3D':
            return ["NPR1", "NPR2", "RadiusOfGyration", "Asphericity", "SpherocityIndex"]
            
        elif descriptor_type == 'CM':
            # Para matriz de Coulomb, geramos nomes para cada elemento da matriz
            n_atoms_max = 29  # valor padrão para QM9
            names = []
            for i in range(n_atoms_max):
                for j in range(n_atoms_max):
                    names.append(f"CM_{i}_{j}")
            return names
            
        elif descriptor_type == 'MACCS':
            # MACCS keys têm significados específicos na RDKit, mas aqui usaremos índices simples
            return [f"MACCS_{i}" for i in range(166)]
            
        elif descriptor_type == 'Topological':
            return [f"Topo_Bit_{i}" for i in range(morgan_nBits)]
            
        elif descriptor_type == 'AtomPair':
            return [f"AtomPair_Bit_{i}" for i in range(morgan_nBits)]
            
        elif descriptor_type == 'EState':
            # Estado E tem 79 bits padrão
            estate_types = [
                "EState_01", "EState_02", "EState_03", "EState_04", "EState_05",
                "EState_06", "EState_07", "EState_08", "EState_09", "EState_10",
                # Continuar com nomenclatura padrão até 79
            ]
            # Se não tivermos todos os nomes específicos, use nomes genéricos
            return [f"EState_{i}" for i in range(79)]
            
        elif descriptor_type == 'Pattern':
            return [f"Pattern_Bit_{i}" for i in range(morgan_nBits)]
            
        elif descriptor_type == 'Avalon':
            return [f"Avalon_Bit_{i}" for i in range(morgan_nBits)]
            
        elif descriptor_type == 'Autocorr':
            # Descritores de autocorrelação são compostos de propriedades atômicas e distâncias
            properties = ["AtomicNumber", "Mass", "Valence", "Charge", "HCount", "Aromatic", "EState"]
            max_dist = 5
            names = []
            for prop in properties:
                for dist in range(max_dist + 1):
                    names.append(f"Autocorr_{prop}_Dist{dist}")
            return names
            
        else:
            raise ValueError(f"Tipo de descritor desconhecido: {descriptor_type}")
    
    def get_descriptor_details(self, descriptor_type='Morgan'):
        """
        Retorna uma descrição detalhada de cada tipo de descritor molecular.
        
        Parameters:
        -----------
        descriptor_type : str
            Tipo de descritor ('CM', 'Morgan', etc.)
            
        Returns:
        --------
        dict: Dicionário com detalhes sobre o descritor
        """
        details = {
            'Morgan': {
                'description': 'Fingerprints circulares de Morgan (ECFP), que capturam ambientes químicos em torno de cada átomo',
                'dimensionality': 'Configurável (padrão: 512 bits)',
                'interpretability': 'Baixa (bits específicos podem ser rastreados a subestruturas)',
                'strengths': 'Bom para similaridade molecular e modelagem de propriedades',
                'weaknesses': 'Não captura diretamente propriedades 3D'
            },
            'MorganCount': {
                'description': 'Fingerprints circulares de Morgan com contagem (variante de ECFP com contagem de ocorrências)',
                'dimensionality': 'Configurável (padrão: 512 bits)',
                'interpretability': 'Baixa a média (contagens podem indicar prevalência de fragmentos)',
                'strengths': 'Captura frequência de subestruturas',
                'weaknesses': 'Maior sensibilidade a ruído'
            },
            'CM': {
                'description': 'Matriz de Coulomb, representando interações entre pares de átomos baseadas em cargas e distâncias',
                'dimensionality': 'n_atoms² (para QM9, geralmente 29²)',
                'interpretability': 'Média (representa interações atômicas)',
                'strengths': 'Invariante a rotação e translação, bom para propriedades energéticas',
                'weaknesses': 'Alta dimensionalidade, sensível ao tamanho molecular'
            },
            'Physicochemical': {
                'description': 'Conjunto de descritores físico-químicos 2D calculados a partir da estrutura',
                'dimensionality': '9 features padrão',
                'interpretability': 'Alta (cada descritor tem significado físico ou químico específico)',
                'strengths': 'Altamente interpretável, útil para propriedades ADME',
                'weaknesses': 'Número limitado de features, menos específico para identificação molecular'
            },
            '3D': {
                'description': 'Descritores baseados na geometria 3D da molécula',
                'dimensionality': '5 features padrão',
                'interpretability': 'Média a alta (descritores têm significado geométrico)',
                'strengths': 'Captura informação conformacional importante',
                'weaknesses': 'Depende de geometria 3D acurada, computacionalmente intensivo'
            },
            'MACCS': {
                'description': 'Keys MACCS - 166 bits representando fragments estruturais específicos',
                'dimensionality': '166 bits',
                'interpretability': 'Alta (cada bit corresponde a uma subestrutura específica)',
                'strengths': 'Padrão da indústria, boa interpretabilidade',
                'weaknesses': 'Número limitado de features, pode não capturar propriedades complexas'
            },
            'Topological': {
                'description': 'Fingerprints topológicos, similares ao Daylight, baseados em caminhos na estrutura 2D',
                'dimensionality': 'Configurável (padrão: 2048 bits)',
                'interpretability': 'Baixa',
                'strengths': 'Bom para similaridade estrutural',
                'weaknesses': 'Menos específico que Morgan para propriedades'
            },
            'AtomPair': {
                'description': 'Fingerprints baseados em pares de átomos e suas distâncias topológicas',
                'dimensionality': 'Configurável (padrão: 2048 bits)',
                'interpretability': 'Média (representa pares de átomos)',
                'strengths': 'Captura relações entre átomos distantes',
                'weaknesses': 'Pode ter colisões de hash'
            },
            'EState': {
                'description': 'Fingerprints baseados em índices de estado eletrotopológico',
                'dimensionality': '79 bits',
                'interpretability': 'Média',
                'strengths': 'Combina informação eletrônica e topológica',
                'weaknesses': 'Mais complexo de interpretar'
            },
            'Pattern': {
                'description': 'Fingerprints baseados em padrões SMARTS pré-definidos',
                'dimensionality': 'Configurável',
                'interpretability': 'Média (baseado em padrões conhecidos)',
                'strengths': 'Pode ser personalizado para tipos específicos de grupos funcionais',
                'weaknesses': 'Dependente do conjunto de padrões'
            },
            'Avalon': {
                'description': 'Fingerprints Avalon, otimizados para triagem de subestruturas',
                'dimensionality': 'Configurável (padrão: 1024 bits)',
                'interpretability': 'Baixa',
                'strengths': 'Bom para busca de subestruturas e similaridade',
                'weaknesses': 'Requer biblioteca adicional, menos interpretável'
            },
            'Autocorr': {
                'description': 'Descritores de autocorrelação 2D baseados em propriedades atômicas',
                'dimensionality': 'n_props × max_dist (geralmente 7 × 6 = 42)',
                'interpretability': 'Média',
                'strengths': 'Captura distribuição de propriedades ao longo da molécula',
                'weaknesses': 'Mais abstrato, requer pós-processamento para interpretação'
            }
        }
        
        if descriptor_type in details:
            return details[descriptor_type]
        else:
            return {'description': 'Descritor não reconhecido ou sem detalhes disponíveis.'}
        

#================================================================#
# Extra Functions
#================================================================#


import pickle
import json
from collections import defaultdict

# Criar Clusters para separar os dados com propriedades de valores parecidos
class Cluster:
    def __init__(self, data):
        """
        Inicializa o gerenciador de clusters.
        
        Args:
            data: DataLoader com os dados para clusterização
        """
        self.data = data
        self.clusters = {}
        self.cluster_stats = {}
        self.target_values = []
        self.features_data = []

    def _extract_data_from_loader(self):
        """
        Extrai features e targets do DataLoader.
        
        Returns:
            tuple: (features_list, targets_list)
        """
        features_list = []
        targets_list = []
        
        print("Extraindo dados do DataLoader...")
        
        try:
            for batch_idx, batch in enumerate(self.data):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Formato padrão: (features, targets)
                    features, targets = batch[0], batch[1]
                    
                    # Converter tensors para numpy se necessário
                    if hasattr(features, 'cpu'):
                        features = features.cpu().numpy()
                    if hasattr(targets, 'cpu'):
                        targets = targets.cpu().numpy()
                    
                    # Garantir que temos arrays numpy
                    features = np.array(features) if not isinstance(features, np.ndarray) else features
                    targets = np.array(targets) if not isinstance(targets, np.ndarray) else targets
                    
                    # Achatar targets se necessário (para targets 2D)
                    if targets.ndim > 1:
                        targets = targets.flatten()
                    
                    features_list.extend(features)
                    targets_list.extend(targets)
                    
                else:
                    print(f"Formato de batch não reconhecido no índice {batch_idx}")
                    continue
                    
        except Exception as e:
            print(f"Erro ao extrair dados: {e}")
            return [], []
        
        print(f"Dados extraídos: {len(features_list)} amostras")
        return features_list, targets_list

    def create_clusters(self, num_clusters=5, size_cluster=100, custom_intervals=None):
        """
        Cria clusters baseados nos valores dos targets.
        
        Args:
            num_clusters (int): Número de clusters a criar
            size_cluster (int): Tamanho máximo de cada cluster
            custom_intervals (list, optional): Lista de tuplas (min, max) para intervalos customizados
                                             Se None, intervalos são calculados automaticamente
        
        Returns:
            dict: Dicionário com os clusters criados
        """
        print(f"Iniciando criação de {num_clusters} clusters com tamanho máximo {size_cluster}")
        
        # Extrair dados do DataLoader
        self.features_data, self.target_values = self._extract_data_from_loader()
        
        if not self.target_values:
            print("Nenhum dado foi extraído. Verificar formato do DataLoader.")
            return {}
        
        # Converter para arrays numpy para facilitar manipulação
        self.target_values = np.array(self.target_values)
        self.features_data = np.array(self.features_data)
        
        # Determinar intervalos
        if custom_intervals:
            intervals = custom_intervals
            print(f"Usando intervalos customizados: {intervals}")
        else:
            intervals = self._calculate_intervals(num_clusters)
            print(f"Intervalos calculados automaticamente: {intervals}")
        
        # Inicializar clusters
        self.clusters = {}
        self.cluster_stats = {}
        
        for i, (min_val, max_val) in enumerate(intervals):
            self.clusters[i] = {
                'features': [],
                'targets': [],
                'indices': [],
                'interval': (min_val, max_val),
                'size': 0
            }
            self.cluster_stats[i] = {
                'min_target': float('inf'),
                'max_target': float('-inf'),
                'mean_target': 0,
                'count': 0,
                'interval': (min_val, max_val)
            }
        
        # Distribuir dados nos clusters
        self._distribute_data_to_clusters(intervals, size_cluster)
        
        # Calcular estatísticas finais
        self._calculate_cluster_statistics()
        
        print(f"Clusters criados com sucesso!")
        return self.clusters

    def create_clusters_kmeans(self, n_clusters=5, use_features=True, random_state=42):
        """
        Cria clusters usando KMeans, com base nas features, target ou ambos.
        Args:
            n_clusters (int): Número de clusters a criar
            use_features (bool): Se True, usa as features para clusterizar. Se False, usa apenas o target.
            random_state (int): Semente para reprodutibilidade
        Returns:
            dict: Dicionário com os clusters criados
        """
        
        print(f"Iniciando criação de {n_clusters} clusters usando KMeans")
        
        # Extrair dados do DataLoader
        self.features_data, self.target_values = self._extract_data_from_loader()
        
        if not self.target_values:
            print("Nenhum dado foi extraído. Verificar formato do DataLoader.")
            return {}
            
        # Converter para arrays numpy para facilitar manipulação
        self.target_values = np.array(self.target_values)
        self.features_data = np.array(self.features_data)
            
        X = np.array(self.features_data)
        y = np.array(self.target_values).reshape(-1, 1)

        # Escolher dados para clusterização
        if use_features and X.shape[1] > 1:
            data_for_clustering = X
            print(f"Usando features para clustering ({X.shape[1]} dimensões)")
        else:
            data_for_clustering = y
            print("Usando apenas targets para clustering")

        # Rodar KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data_for_clustering)

        # Inicializar clusters e cluster_stats
        self.clusters = {}
        self.cluster_stats = {}
        
        for i in range(n_clusters):
            self.clusters[i] = {
                'features': [],
                'targets': [],
                'indices': [],
                'size': 0
            }
            self.cluster_stats[i] = {
                'min_target': float('inf'),
                'max_target': float('-inf'),
                'mean_target': 0,
                'count': 0,
                'interval': (None, None)  # Para KMeans não há intervalos pré-definidos
            }

        # Organizar dados nos clusters
        for idx, label in enumerate(cluster_labels):
            self.clusters[label]['features'].append(X[idx])
            self.clusters[label]['targets'].append(float(y[idx]))
            self.clusters[label]['indices'].append(idx)
            self.clusters[label]['size'] += 1

        # Calcular estatísticas finais para cada cluster
        for i in range(n_clusters):
            targets = self.clusters[i]['targets']
            if targets:
                targets_array = np.array(targets)
                self.clusters[i]['min_target'] = float(np.min(targets_array))
                self.clusters[i]['max_target'] = float(np.max(targets_array))
                self.clusters[i]['mean_target'] = float(np.mean(targets_array))
                
                # Atualizar cluster_stats
                self.cluster_stats[i].update({
                    'min_target': float(np.min(targets_array)),
                    'max_target': float(np.max(targets_array)),
                    'mean_target': float(np.mean(targets_array)),
                    'std_target': float(np.std(targets_array)),
                    'count': len(targets),
                    'interval': (float(np.min(targets_array)), float(np.max(targets_array)))
                })
                
                print(f"Cluster {i}: {len(targets)} amostras "
                      f"(targets: {np.min(targets_array):.4f} - {np.max(targets_array):.4f})")
            else:
                self.clusters[i]['min_target'] = None
                self.clusters[i]['max_target'] = None
                self.clusters[i]['mean_target'] = None
                
                self.cluster_stats[i].update({
                    'min_target': None,
                    'max_target': None,
                    'mean_target': None,
                    'std_target': None,
                    'count': 0,
                    'interval': (None, None)
                })

        print(f"Clusters KMeans criados com sucesso!")
        return self.clusters

    def _calculate_intervals(self, num_clusters):
        """
        Calcula intervalos automáticos baseados na distribuição dos targets.
        
        Args:
            num_clusters (int): Número de clusters desejado
            
        Returns:
            list: Lista de tuplas (min, max) para cada intervalo
        """
        min_val = float(np.min(self.target_values))
        max_val = float(np.max(self.target_values))
        
        print(f"Valores target: min={min_val:.4f}, max={max_val:.4f}")
        
        # Criar intervalos igualmente espaçados
        interval_size = (max_val - min_val) / num_clusters
        intervals = []
        
        for i in range(num_clusters):
            start = min_val + i * interval_size
            end = min_val + (i + 1) * interval_size
            
            # Ajustar o último intervalo para incluir o valor máximo
            if i == num_clusters - 1:
                end = max_val
            
            intervals.append((start, end))
        
        return intervals

    def _distribute_data_to_clusters(self, intervals, size_cluster):
        """
        Distribui os dados nos clusters baseado nos intervalos.
        
        Args:
            intervals (list): Lista de tuplas (min, max) para cada cluster
            size_cluster (int): Tamanho máximo de cada cluster
        """
        print("Distribuindo dados nos clusters...")
        
        for idx, (feature, target) in enumerate(zip(self.features_data, self.target_values)):
            # Encontrar o cluster apropriado para este target
            cluster_id = self._find_cluster_for_target(target, intervals)
            
            if cluster_id is not None and self.clusters[cluster_id]['size'] < size_cluster:
                # Adicionar dados ao cluster
                self.clusters[cluster_id]['features'].append(feature)
                self.clusters[cluster_id]['targets'].append(float(target))
                self.clusters[cluster_id]['indices'].append(idx)
                self.clusters[cluster_id]['size'] += 1
        
        # Mostrar estatísticas de distribuição
        for cluster_id, cluster_data in self.clusters.items():
            print(f"Cluster {cluster_id}: {cluster_data['size']} amostras "
                  f"(intervalo: {cluster_data['interval'][0]:.4f} - {cluster_data['interval'][1]:.4f})")

    def _find_cluster_for_target(self, target, intervals):
        """
        Encontra o cluster apropriado para um valor de target.
        
        Args:
            target (float): Valor do target
            intervals (list): Lista de intervalos
            
        Returns:
            int or None: ID do cluster ou None se não encontrar
        """
        for cluster_id, (min_val, max_val) in enumerate(intervals):
            # Para o último cluster, incluir o valor máximo
            if cluster_id == len(intervals) - 1:
                if min_val <= target <= max_val:
                    return cluster_id
            else:
                if min_val <= target < max_val:
                    return cluster_id
        return None

    def _calculate_cluster_statistics(self):
        """
        Calcula estatísticas para cada cluster.
        """
        for cluster_id, cluster_data in self.clusters.items():
            if cluster_data['targets']:
                targets = np.array(cluster_data['targets'])
                self.cluster_stats[cluster_id].update({
                    'min_target': float(np.min(targets)),
                    'max_target': float(np.max(targets)),
                    'mean_target': float(np.mean(targets)),
                    'std_target': float(np.std(targets)),
                    'count': len(targets)
                })

    def get_cluster_summary(self):
        """
        Retorna um resumo dos clusters criados.
        
        Returns:
            dict: Resumo com estatísticas de cada cluster
        """
        return {
            'total_clusters': len(self.clusters),
            'total_samples': sum(cluster['size'] for cluster in self.clusters.values()),
            'cluster_details': self.cluster_stats
        }

    def print_cluster_info(self):
        """
        Imprime informações detalhadas dos clusters.
        """
        print("\n" + "="*60)
        print("RESUMO DOS CLUSTERS")
        print("="*60)
        
        summary = self.get_cluster_summary()
        print(f"Total de clusters: {summary['total_clusters']}")
        print(f"Total de amostras: {summary['total_samples']}")
        print("-"*60)
        
        for cluster_id, stats in summary['cluster_details'].items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Intervalo: [{stats['interval'][0]:.4f}, {stats['interval'][1]:.4f}]")
            print(f"  Amostras: {stats['count']}")
            if stats['count'] > 0:
                print(f"  Target - Min: {stats['min_target']:.4f}")
                print(f"  Target - Max: {stats['max_target']:.4f}")
                print(f"  Target - Média: {stats['mean_target']:.4f}")
                if 'std_target' in stats:
                    print(f"  Target - Desvio Padrão: {stats['std_target']:.4f}")

    def get_cluster_by_id(self, cluster_id):
        """
        Retorna os dados de um cluster específico.
        
        Args:
            cluster_id (int): ID do cluster
            
        Returns:
            dict or None: Dados do cluster ou None se não existir
        """
        return self.clusters.get(cluster_id)

    def save_clusters(self, filepath):
        """
        Salva os clusters em arquivo.
        
        Args:
            filepath (str): Caminho para salvar o arquivo
        """
        data_to_save = {
            'clusters': {k: {
                'features': [f.tolist() if isinstance(f, np.ndarray) else f 
                           for f in v['features']],
                'targets': v['targets'],
                'indices': v['indices'],
                'interval': v['interval'],
                'size': v['size']
            } for k, v in self.clusters.items()},
            'cluster_stats': self.cluster_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Clusters salvos em: {filepath}")

    def load_clusters(self, filepath):
        """
        Carrega clusters de arquivo.
        
        Args:
            filepath (str): Caminho do arquivo a carregar
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.clusters = {k: {
            'features': [np.array(f) for f in v['features']],
            'targets': v['targets'],
            'indices': v['indices'],
            'interval': v['interval'],
            'size': v['size']
        } for k, v in data['clusters'].items()}
        
        self.cluster_stats = data['cluster_stats']
        print(f"Clusters carregados de: {filepath}")


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class FakeDataset(Dataset):
    """Dataset sintético para testar clustering."""
    
    def __init__(self, n_samples=1000, n_features=50, noise_level=0.1, seed=42):
        """
        Cria um dataset sintético com padrões bem definidos.
        
        Args:
            n_samples: Número total de amostras
            n_features: Número de features
            noise_level: Nível de ruído nos dados
            seed: Semente aleatória
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Criar diferentes grupos de dados com características distintas
        samples_per_group = n_samples // 4
        
        # Grupo 1: Targets baixos (0.0 - 0.3)
        group1_features = np.random.normal(0, 1, (samples_per_group, n_features))
        group1_targets = np.random.uniform(0.0, 0.3, samples_per_group)
        
        # Grupo 2: Targets médio-baixos (0.3 - 0.6)
        group2_features = np.random.normal(2, 1.5, (samples_per_group, n_features))
        group2_targets = np.random.uniform(0.3, 0.6, samples_per_group)
        
        # Grupo 3: Targets médio-altos (0.6 - 0.85)
        group3_features = np.random.normal(-1, 1.2, (samples_per_group, n_features))
        group3_targets = np.random.uniform(0.6, 0.85, samples_per_group)
        
        # Grupo 4: Targets altos (0.85 - 1.0)
        group4_features = np.random.normal(1, 2, (samples_per_group, n_features))
        group4_targets = np.random.uniform(0.85, 1.0, samples_per_group)
        
        # Combinar todos os grupos
        self.features = np.vstack([group1_features, group2_features, 
                                  group3_features, group4_features])
        self.targets = np.concatenate([group1_targets, group2_targets, 
                                     group3_targets, group4_targets])
        
        # Adicionar ruído
        self.features += np.random.normal(0, noise_level, self.features.shape)
        self.targets += np.random.normal(0, noise_level * 0.1, self.targets.shape)
        
        # Garantir que targets ficam no intervalo [0, 1]
        self.targets = np.clip(self.targets, 0, 1)
        
        # Embaralhar dados
        indices = np.random.permutation(len(self.features))
        self.features = self.features[indices]
        self.targets = self.targets[indices]
        
        print(f"Dataset criado com {len(self.features)} amostras")
        print(f"Target range: {self.targets.min():.3f} - {self.targets.max():.3f}")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.features[idx]), 
                torch.FloatTensor([self.targets[idx]]))

def create_fake_dataloader(n_samples=1000, batch_size=64, n_features=50):
    """Cria um DataLoader com dados sintéticos."""
    dataset = FakeDataset(n_samples=n_samples, n_features=n_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataset

if __name__ == '__main__':
    test_loader, dataset = create_fake_dataloader()

    cluster_manager = Cluster(test_loader)
    
    # Criar clusters automáticos
    clusters = cluster_manager.create_clusters(
        num_clusters=5,
        size_cluster=200
    )
    
    cluster_manager.print_cluster_info()