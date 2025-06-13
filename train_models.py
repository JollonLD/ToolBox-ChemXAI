from chemxai.train import train_mlp_qm9
import time

def run_train():
    print("Iniciando treinamento de modelos...")

    # Parâmetros comuns
    epochs = 50
    layers = [128, 64]
    learning_rate = 1e-3
    batch_size = 32

    # MLP com descritor CM sem noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor CM sem noise")
    print("="*50)
    train_mlp_qm9(
        att_index=10,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=0,
        descriptor_type='CM'
    )
    time.sleep(2)  # Pequena pausa para organizar logs

    # MLP com descritor CM com noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor CM com noise")
    print("="*50)
    train_mlp_qm9(
        att_index=10,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=3,
        descriptor_type='CM'
    )
    time.sleep(2)  # Pequena pausa para organizar logs

    # MLP com descritor Morgan sem noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor Morgan sem noise")
    print("="*50)
    train_mlp_qm9(
        att_index=10,
        epochs=epochs,
        layers=layers, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=0,
        descriptor_type='Morgan'
    )
    time.sleep(2)  # Pequena pausa para organizar logs

    # MLP com descritor Morgan com noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor Morgan com noise")
    print("="*50)
    train_mlp_qm9(
        att_index=10,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=3,
        descriptor_type='Morgan'
    )
    time.sleep(2)  # Pequena pausa para organizar logs

    # MLP com descritor Physicochemical sem noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor Physicochemical sem noise")
    print("="*50)
    train_mlp_qm9(
        att_index=10,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=0,
        descriptor_type='Physicochemical'
    )
    time.sleep(2)  # Pequena pausa para organizar logs

    # MLP com descritor Physicochemical com noise
    print("\n" + "="*50)
    print("Treinando MLP com descritor Physicochemical com noise")
    print("="*50)
    train_mlp_qm9(
        att_index=10,
        epochs=epochs,
        layers=layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_noise=3,
        descriptor_type='Physicochemical'
    )

    print("\n" + "="*50)
    print("Treinamento de todos os modelos concluído!")
    print("="*50)

if __name__ == '__main__':

    run_train()