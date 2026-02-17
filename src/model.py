from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict
import evaluate
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding


def fine_tune_model(model: AutoModelForSequenceClassification, tokenized_dataset: DatasetDict):
    
    


