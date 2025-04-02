import numpy as np
import pandas as pd
import os
from torch_geometric.datasets import Planetoid
from sklearn.datasets import load_iris

#================================================================#
# Tubular Datasets
#================================================================#

class qm9_tubular:
    def __init__(self):
        self.directory_path = r"C:\IC\toolboxXAI\DataSets\QM9"
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


class iris_dataset:
    def __init__(self):
        self.data = load_iris()
    
#================================================================#
# Graph Based Datasets
#================================================================#

