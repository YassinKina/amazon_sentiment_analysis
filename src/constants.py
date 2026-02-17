import os
#________ Constants and paths used throughout the project _______

# Ensure that data is downloaded in current directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "datasets")
DATA_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models")
RANDOM_SEED = 20
NUM_CLASSES = 5
ROBERTA_BASE = "roberta-base"
       