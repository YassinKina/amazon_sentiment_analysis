from transformers import AutoModelForSequenceClassification
from src.constants import ROBERTA_BASE
from src.data_utils import get_final_ds
from src.tokenize import tokenize


def main():
    # Get final split dataset
    dataset = get_final_ds()
    tokenized_dataset = tokenize(dataset)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_BASE)
    fine_tune_model(model, tokenized_dataset)
 
   
    
  
    
if __name__ == "__main__":
    main()