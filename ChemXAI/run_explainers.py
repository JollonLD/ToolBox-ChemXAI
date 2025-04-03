from src.data import prepare_data

def main():
    # Load the dataset
    data = prepare_data('QM9')

    print(data)
    

if __name__ == '__main__':
    main()
