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

def prepare_data_graph(dataset_name='QM9'):
    """Get Data to be used

    Parameter:
            dataset_name (str): name of the dataset (default: QM9)

    Return:
            [torch_geometric.Data]: dataset with the correct format to be used in the explanations
    """
    # Get the project path
    dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Download the dataset
    if dataset_name == 'PCQM4Mv2':
        path = os.path.join(dirname, 'data')
        data = PCQM4Mv2(root=path, split='train', transform=T.NormalizeFeatures())

    elif dataset_name == 'QM9':
        path = os.path.join(dirname, 'data', 'QM9')
        data = QM9(root=path, transform=T.NormalizeFeatures())

    return data

#================================================================#
# Tubular Datasets
#================================================================#

class qm9_tubular:
    def __init__(self):
        # Caminho absoluto baseado na localização do script atual
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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

class iris_dataset:
    def __init__(self):
        self.data = load_iris()
    