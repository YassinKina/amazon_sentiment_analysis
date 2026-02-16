import pytest
from unittest.mock import patch, MagicMock, mock_open, ANY
from src.download_data import download_data 

@patch("src.download_data.os.path.exists")
@patch("src.download_data.load_dataset")
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
    

@patch("src.download_data.os.path.exists")
@patch("src.download_data.os.makedirs")
@patch("src.download_data.load_dataset")
@patch("src.download_data.open", new_callable=mock_open)
@patch("src.download_data.json.dump")
def test_download_data_downloads_new(mock_json_dump, mock_file, mock_load, mock_makedirs, mock_exists):
    """Test case where the data does not exist and needs to be downloaded."""
    # Setup mocks
    mock_exists.return_value = False
    
    # Mock the streaming dataset
    mock_stream = MagicMock()
    mock_stream.take.return_value = [{"text": "sample1"}, {"text": "sample2"}]
    
    # First call returns the stream, second call returns the final loaded dataset
    mock_final_dataset = MagicMock()
    mock_load.side_effect = [mock_stream, mock_final_dataset]
    
    # Execute
    result = download_data(num_samples=2)
    
    # Assertions
    mock_makedirs.assert_called_once()
    mock_stream.take.assert_called_with(2)
    mock_json_dump.assert_called_once()
    assert result == mock_final_dataset
  

@patch("src.download_data.os.path.exists")
@patch("src.download_data.load_dataset")
def test_download_data_default_samples(mock_load, mock_exists):
    """Verify the default num_samples logic."""
    mock_exists.return_value = False
    mock_stream = MagicMock()
    mock_load.side_effect = [mock_stream, MagicMock()]
    
    download_data() 
    
    mock_stream.take.assert_called_with(12500)