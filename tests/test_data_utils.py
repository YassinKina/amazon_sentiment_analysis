import pytest
from unittest.mock import patch, MagicMock, mock_open, ANY
from datasets import Dataset, DatasetDict
from src.data_utils import download_data, split_dataset

@patch("src.data_utils.os.path.exists")
@patch("src.data_utils.load_dataset")
def test_download_data_already_exists(mock_load, mock_exists):
    """Test case where the data file already exists locally."""
    # Setup mocks
    mock_exists.return_value = True
    mock_dataset = MagicMock()
    mock_load.return_value = mock_dataset
    
    # Execute
    result = download_data(num_samples=10)
    
    # Assertions
    # We use ANY because the exact path is defined in your constants
    mock_load.assert_called_once_with("json", data_files=ANY, split="train")
    assert result == mock_dataset
    

@patch("src.data_utils.os.path.exists")
@patch("src.data_utils.os.makedirs")
@patch("src.data_utils.load_dataset")
@patch("src.data_utils.open", new_callable=mock_open)
@patch("src.data_utils.json.dumps")
def test_download_data_downloads_new(mock_json_dumps, mock_file, mock_load, mock_makedirs, mock_exists):
    """Test case where data is missing, downloaded, and saved as JSONL."""
    mock_exists.return_value = False
    
    # Mock data to simulate the streaming result
    sample_data = [{"text": "sample1"}, {"text": "sample2"}]
    mock_stream = MagicMock()
    mock_stream.take.return_value = sample_data
    
    # Mock final loaded dataset
    mock_final_ds = MagicMock()
    mock_load.side_effect = [mock_stream, mock_final_ds]
    
    # Define what json.dumps returns for each item in the loop
    mock_json_dumps.side_effect = ['{"text": "sample1"}', '{"text": "sample2"}']

    result = download_data(num_samples=2)
    
    # Check that directory was created
    mock_makedirs.assert_called_once()
    
    # Check that we actually took the correct number of samples from the stream
    mock_stream.take.assert_called_with(2)
    
    # Check that json.dumps was called for EACH item in our sample_data
    assert mock_json_dumps.call_count == 2
    
    # Check that we wrote to the file twice
    handle = mock_file()
    assert handle.write.call_count == 2
    
    # Verify the final return value
    assert result == mock_final_ds
  

@patch("src.data_utils.os.path.exists")
@patch("src.data_utils.os.makedirs") 
@patch("src.data_utils.load_dataset")
@patch("src.data_utils.open", new_callable=mock_open) # ensure data remains unchanged in data/dataset
@patch("src.data_utils.json.dumps")
def test_download_data_default_samples(mock_json, mock_file, mock_load, mock_makedirs, mock_exists):
    """Verify the default num_samples logic without wiping real files."""
    mock_exists.return_value = False
    mock_stream = MagicMock()
    # Return a dummy list so the loop doesn't crash
    mock_stream.take.return_value = [{"data": 1}] 
    mock_load.side_effect = [mock_stream, MagicMock()]
    
    download_data() 
    
    mock_stream.take.assert_called_with(12500)
    
def test_split_dataset_proportions():
    """Verify that the dataset is split into 80/10/10 proportions."""
    # 1. Create a dummy dataset with 100 samples
    data = {"text": [f"sample {i}" for i in range(100)], "label": [i % 2 for i in range(100)]}
    dummy_ds = Dataset.from_dict(data)
    
    # 2. Run the split
    # Note: We assume RANDOM_SEED is imported in data_utils.py
    result = split_dataset(dummy_ds)
    
    # 3. Assertions
    assert isinstance(result, DatasetDict)
    assert set(result.keys()) == {"train", "validation", "test"}
    
    # Check lengths (80% / 10% / 10%)
    assert len(result["train"]) == 80
    assert len(result["validation"]) == 10
    assert len(result["test"]) == 10
    
    # Verify data integrity (ensure no data was lost)
    total_samples = len(result["train"]) + len(result["validation"]) + len(result["test"])
    assert total_samples == 100

def test_split_dataset_reproducibility():
    """Verify that the seed ensures the split is reproducible."""
    data = {"text": [f"sample {i}" for i in range(20)], "label": [0] * 20}
    dummy_ds = Dataset.from_dict(data)
    
    result1 = split_dataset(dummy_ds)
    result2 = split_dataset(dummy_ds)
    
    # The first item in the training set should be the same both times due to RANDOM_SEED
    assert result1["train"][0] == result2["train"][0]