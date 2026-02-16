from src.download_data import download_data


def main():
    dataset = download_data()
    print(dataset)
    
if __name__ == "__main__":
    main()