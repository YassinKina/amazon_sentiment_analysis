import torch
from transformers import AutoModelForSequenceClassification
from src.constants import ROBERTA_BASE, NUM_CLASSES
from src.data_utils import get_final_ds
from src.tokenize import tokenize
from src.engine import start_fine_tuning

def main():
    dataset = get_final_ds()
    tokenized_dataset, tokenizer = tokenize(dataset)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_BASE, num_labels=NUM_CLASSES)
    start_fine_tuning(model, tokenizer, tokenized_dataset)
    
   
    
if __name__ == "__main__":
    main()
    
