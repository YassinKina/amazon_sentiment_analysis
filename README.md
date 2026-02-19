# Customer Review Sentiment Analysis with RoBERTa

![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![WandB](https://img.shields.io/badge/Experiment_Tracking-WandB-gold)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Tests](https://img.shields.io/badge/Tests-Pytest-green)

## Project Overview

This project implements an end-to-end Machine Learning pipeline to fine-tune a **RoBERTa** transformer model for multi-class sentiment analysis on customer reviews. There is also an interactive **Streamlit** dashboard for model inference.

This project focuses on production-grade engineering practices, including handling **class imbalance** via custom loss functions, integrating **MLOps** tools for experiment tracking, and ensuring code reliability through **unit testing**.

#### Live Demo

<a href="https://customersentimentanalysis.streamlit.app/" target=_blank rel="noopener noreferrer">Check out the interactive web app here!</a>

### Key Features

- **Advanced Fine-Tuning**: Utilizes `RoBERTa-base` for sequence classification, mapping 1-5 star ratings to specific sentiment classes.
- **Interactive Web App**: Features a Streamlit dashboard for real-time single-text analysis and bulk CSV processing with interactive visualizations.
- **Handling Class Imbalance**: Implements a custom `WeightedTrainer` that overrides the standard Hugging Face Trainer loss function. It dynamically computes class weights based on the training distribution to penalize the model more for misclassifying rare classes.
- **Robust Data Pipeline**: Features a streaming data loader that caches datasets locally (`jsonl`), splits data reproducibly (Train/Val/Test), and handles tokenization efficiently.
- **Experiment Tracking**: Fully integrated with **Weights & Biases (WandB)** to log metrics (F1-score, Accuracy), hyperparameters, and training loss curves.
- **Software Engineering Standards**: Includes type hinting, modular architecture, and a suite of unit tests using `pytest`.

## Technical Architecture

The project is structured as a modular Python package:

```text
src/
├── engine.py           # Training loop orchestration, WandB init, and evaluation
├── data_utils.py       # ETL pipeline: Download, stream, cache, and split data
├── weighted_trainer.py # Custom Trainer subclass for weighted CrossEntropyLoss
├── tokenize.py         # Tokenization logic using AutoTokenizer
├── progress_bar.py     # Custom nested progress bar for training visibility
└── constants.py        # Configuration constants (Paths, Hyperparams)
tests/
├── test_data_utils.py  # Mocks and tests for data ingestion
└── test_tokenize.py    # Tests for tensor shapes and label alignment
```

## Methodology

### 1. Data Processing

The pipeline streams data from a remote JSON source to avoid memory overhead. It performs an 80/10/10 split (Train/Validation/Test) using a fixed random seed to ensure reproducibility across runs.

### 2. Modeling & Class Imbalance

Real-world review data is often skewed (e.g., mostly 5-star reviews). To prevent the model from biasing towards the majority class, I calculated class weights using `sklearn.utils.class_weight`:

```python
# Logic from src/engine.py
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)
# These weights are passed to the CrossEntropyLoss function in the custom Trainer.
```

### 3. Training & Evaluation

- **Optimizer**: AdamW with linear learning rate decay.
- **Callbacks**: Early Stopping (patience=3) to prevent overfitting.
- **Metrics**: Weighted F1-Score (crucial for imbalanced datasets) and Accuracy.

## Installation and Usage

### Prerequisites

- Python 3.9+
- CUDA or MPS (Apple Silicon) capable GPU recommended.

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/review_sentiment_analysis.git
   cd review_sentiment_analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run tests to ensure environment integrity:
   ```bash
   pytest tests/
   ```

### Running the Training Pipeline

To start the fine-tuning process:

```bash
python fine_tune.py
```

## Future Improvements

- **Model Distillation**: Compress the fine-tuned RoBERTa model into a smaller version (DistilRoBERTa) for lower latency inference.
- **ONNX Export**: Convert the model to ONNX format for optimized deployment on CPU-bound environments.
- **API Deployment**: Wrap the inference logic in a FastAPI container.
