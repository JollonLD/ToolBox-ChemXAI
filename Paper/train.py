import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, BatchNorm1d, Dropout



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, device, lr=0.001):
        super(MLP, self).__init__()
        all_layers = []
        prev_dim = input_dim

        # Camadas ocultas
        for layer_dim in layers:
            all_layers.append(nn.Linear(prev_dim, layer_dim))
            all_layers.append(nn.ReLU())
            prev_dim = layer_dim

        # Camada de saída (ativação linear - regressão)
        all_layers.append(nn.Linear(prev_dim, output_dim))

        # Combinando as camadas
        self.layers = nn.Sequential(*all_layers)

        self.criterion = nn.L1Loss() #nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, x):
        return self.layers(x)
    


def load_mordred_descriptors():

    df = pd.read_csv("all_desc_qm9.csv")
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_valid = df.describe()
    df_usable = df[df_valid.columns]
    df_usable.to_csv("desc_mordred_qm9.csv")

    return df_usable



def train():

    

    return history


if __name__ == '__main__':

    load_mordred_descriptors()