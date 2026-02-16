from datasets import load_dataset, DatasetDict
import torch
import os
import json
from .constants import DATA_DIR, DATASET_PATH, DATA_URL, RANDOM_SEED

def get_final_ds():
    # Combine all functionality itnno
    dataset = download_data()
    split_ds = split_dataset(dataset)
    return split_ds
    

def download_data(num_samples:int = 12500):
    
    # If data is downloaded locally, load the dataset
    if os.path.exists(DATASET_PATH):
        print(f"Data exists at path: {DATASET_PATH}, loading data...")
        dataset = load_dataset("json",data_files=DATASET_PATH, split="train")
        return dataset
    
    print(f"Data does not exist at path {DATASET_PATH}, downloading data...")
    # Create data dir if it doesnt exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset_stream = load_dataset("json", data_files=DATA_URL, split="train", streaming=True)
    data_slice = list(dataset_stream.take(num_samples))
    # Create file with num_samples samples from the dataset
    with open(DATASET_PATH, "w") as f:
        for entry in data_slice:
            f.write(json.dumps(entry) + "\n")
        
    print(f"Data saved at path {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    return dataset
    
def split_dataset(dataset:DatasetDict):
    split_train_test = dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    split_val_test = split_train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_SEED)

    split_ds = DatasetDict({
        "train": split_train_test["train"],
        "validation": split_val_test["train"],
        "test": split_val_test["test"]
    })
    return split_ds
    
    
    
    
    

    
 
    
    