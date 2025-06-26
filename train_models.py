from chemxai.train import train_mlp_qm9, train_gcn_qm9
import time

def run_train():
    print("Iniciando treinamento de modelos...")

    # Parâmetros comuns
    epochs = 50
    layers = [128, 64]
    learning_rate = 1e-3
    batch_size = 32

    # MLP com descritor Physicochemical com noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor Physicochemical sem noise")
    print("="*50)
    train_mlp_qm9(
        att_index=0,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=0,
        descriptor_type='Physicochemical'
    )

    # MLP com descritor Physicochemical com noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor Physicochemical com noise")
    print("="*50)
    train_mlp_qm9(
        att_index=0,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=2,
        descriptor_type='Physicochemical'
    )

    # GCN com QM9 e sem noise
    print("\n" + "="*50)
    print("Treinando GCN sem noise")
    print("="*50)
    train_gcn_qm9(target_idx=0)

    # GCN com QM9 e sem noise
    print("\n" + "="*50)
    print("Treinando GCN sem noise")
    print("="*50)
    train_gcn_qm9(target_idx=0, n_noise=2)

    print("\n" + "="*50)
    print("Treinamento de todos os modelos concluído!")
    print("="*50)

if __name__ == '__main__':

    run_train()