from datasets import load_dataset, DatasetDict
import torch
import os
import json
from .constants import DATA_DIR, DATASET_PATH, DATA_URL, RANDOM_SEED

def get_final_ds():
    """
    Orchestrates the full data pipeline from acquisition to final splitting.

    This function serves as the primary entry point for the data module, 
    triggering the download (or local load) and applying the required 
    Train/Validation/Test splits.

    Returns:
        DatasetDict: A dictionary containing the 'train', 'validation', 
            and 'test' splits.
    """
    dataset = download_data()
    split_ds = split_dataset(dataset)
    return split_ds
    

def download_data(num_samples:int = 12500):
    """
    Downloads a subset of the dataset or loads it if already present on disk.

    If the file exists at DATASET_PATH, it is loaded directly. Otherwise, it 
    streams the dataset from DATA_URL, slices the specified number of samples, 
    and saves them locally in JSON Lines (JSONL) format.

    Args:
        num_samples (int): The number of samples to download if the local 
            file is missing. Defaults to 12500.

    Returns:
        Dataset: The loaded Hugging Face Dataset object for the 'train' split.
    """
    if os.path.exists(DATASET_PATH):
        print(f"Data exists at path: {DATASET_PATH}, loading data...")
        dataset = load_dataset("json",data_files=DATASET_PATH, split="train")
        return dataset
    
    print(f"Data does not exist at path {DATASET_PATH}, downloading data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset_stream = load_dataset("json", data_files=DATA_URL, split="train", streaming=True)
    data_slice = list(dataset_stream.take(num_samples))

    with open(DATASET_PATH, "w") as f:
        for entry in data_slice:
            f.write(json.dumps(entry) + "\n")
        
    print(f"Data saved at path {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    return dataset
    
def split_dataset(dataset:DatasetDict):
    """
    Splits a single dataset into training, validation, and testing sets.

    The division follows an 80/10/10 ratio:
    1. Initial split creates 80% Train and 20% Temporary Test.
    2. The Temporary Test is split 50/50 to create 10% Validation and 10% Test.

    Args:
        dataset (Dataset): The source dataset to be split.

    Returns:
        DatasetDict: A Hugging Face DatasetDict object containing 
            'train', 'validation', and 'test' keys.
    """
    split_train_test = dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    split_val_test = split_train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_SEED)

    split_ds = DatasetDict({
        "train": split_train_test["train"],
        "validation": split_val_test["train"],
        "test": split_val_test["test"]
    })
    
    return split_ds