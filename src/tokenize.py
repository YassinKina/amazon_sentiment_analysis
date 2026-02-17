from transformers.tokenization_utils_base import BatchEncoding
from datasets import DatasetDict
from .constants import ROBERTA_BASE
from transformers import AutoTokenizer

def tokenize(dataset: DatasetDict) -> DatasetDict:
    """This function takes in a dataset of customer reviews and returns a tokenized 
    version of that dataset.

    Args:
        dataset (DatasetDict): A dataset of customer reviews to be tokeknized.

    Returns:
        DatasetDict: The tokenized customer reviews dataset.
    """
    print("Tokenzing dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
    tokenized_data = dataset.map(process_data, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_data

def process_data(examples: DatasetDict, tokenizer: AutoTokenizer) -> BatchEncoding:
    """
    Tokenizes text inputs and maps star ratings to 0-indexed labels for the model.

    Args:
        examples (DatasetDict): A batch of raw data containing 'text' and 'rating' columns.
    Returns:
        BatchEncoding: A Hugging Face object containing 'input_ids', 'attention_mask',
                       and transformed 'labels'.
    """
    # We will add padding later with DataCollatorWithPadding
    tokenized_data = tokenizer(examples["text"], truncation=True)
    # Map 1-5 star ratings to 0-4 
    tokenized_data["labels"] = [int(rating) - 1 for rating in examples["rating"]]
    return tokenized_data