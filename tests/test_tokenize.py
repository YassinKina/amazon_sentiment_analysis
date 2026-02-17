import pytest
import numpy as np
from datasets import Dataset, DatasetDict
from unittest.mock import patch, MagicMock
from src.tokenize import tokenize, process_data
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Mock dataset fixture ---
@pytest.fixture
def sample_dataset():
    data = {
        "text": ["I love this product!", "Terrible service."],
        "rating": [5, 1]
    }
    ds = Dataset.from_dict(data)
    dataset_dict = DatasetDict({
        "train": ds,
        "validation": ds
    })
    return dataset_dict

# --- Fake tokenizer: pure mock object ---
class MockTokenizer:
    def __call__(self, texts, truncation=True):
        batch_size = len(texts)
        seq_len = 5  # fixed length for simplicity
        return {
            "input_ids": [[1]*seq_len for _ in range(batch_size)],
            "attention_mask": [[1]*seq_len for _ in range(batch_size)]
        }

# --- Test process_data ---
def test_process_data_shapes():
    tokenizer = MockTokenizer()
    examples = {
        "text": ["Amazing!", "Bad!"],
        "rating": [5, 2]
    }
    tokenized = process_data(examples, tokenizer)

    # Check keys exist
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert "labels" in tokenized

    # Check lengths match
    assert len(tokenized["input_ids"]) == len(examples["text"])
    assert len(tokenized["attention_mask"]) == len(examples["text"])
    assert len(tokenized["labels"]) == len(examples["text"])

    # Check labels are 0-indexed
    assert tokenized["labels"] == [4, 1]

# --- Test tokenize ---
def test_tokenize_returns_dataset_and_tokenizer(sample_dataset):
    fake_tokenizer = MockTokenizer()

    # Patch AutoTokenizer.from_pretrained to return our fake tokenizer
    with patch("src.tokenize.AutoTokenizer.from_pretrained", return_value=fake_tokenizer):
        tokenized_dataset, tokenizer = tokenize(sample_dataset)

    # Check tokenized_dataset type
    from datasets import DatasetDict
    assert isinstance(tokenized_dataset, DatasetDict)

    # Check labels
   # Convert all labels to Python int
    train_labels = [int(l) for l in tokenized_dataset["train"]["labels"]]
    val_labels = [int(l) for l in tokenized_dataset["validation"]["labels"]]

    # Check labels are 0-indexed and in range
    assert all(0 <= l <= 4 for l in train_labels)
    assert all(0 <= l <= 4 for l in val_labels)

    # Check expected columns
    for split in ["train", "validation"]:
        for col in ["input_ids", "attention_mask", "labels"]:
            assert col in tokenized_dataset[split].column_names
