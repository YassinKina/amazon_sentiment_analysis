from src.data_utils import get_final_ds


def main():
    # Get dataset
    dataset = get_final_ds()
    print(dataset)
    # Split dataset
    
    # print(dataset["train"][0])
    # print(dataset["validation"][0])
    # print(dataset["test"][0])
    
    
if __name__ == "__main__":
    main()