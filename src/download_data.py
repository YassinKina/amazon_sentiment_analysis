import datasets
from datasets import load_dataset
import torch
import os
import json
from .paths import DATA_DIR, DATASET_PATH, FILE_URL

def download_data(num_samples:int = 12500):
    
    # If data is downloaded locally, load the dataset
    if os.path.exists(DATASET_PATH):
        print(f"Data exists at path: {DATASET_PATH}, loading data...")
        dataset = load_dataset("json",data_files=DATASET_PATH, split="train")
        return dataset
    
    print(f"Data does not exist at path {DATASET_PATH}, downloading data...")
    # Create data dir if it doesnt exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset_stream = load_dataset("json", data_files=FILE_URL, split="train", streaming=True)
    data_slice = list(dataset_stream.take(num_samples))
    # Create file with num_samples samples from the dataset
    with open(DATASET_PATH, "w") as f:
        json.dump(data_slice, f)
        
    print(f"Data saved at path {DATASET_PATH}")
    dataset = load_dataset("json",data_files=DATASET_PATH, split="train")
    return dataset
    
    
    

    
 
    
    