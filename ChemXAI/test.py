from src.data import prepare_data_graph
from src.explainers import GNNEx, GraphLIME, GraphShap

def main():

    data_QM9 = prepare_data_graph('QM9')

    exp_GNN = GNNEx()

if __name__ == '__main__':
    main()