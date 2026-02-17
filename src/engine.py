from datasets import DatasetDict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import os
import yaml
from .constants import CONFIG_PATH, MODEL_OUTPUT_PATH, NUM_CLASSES
from .weighted_trainer import WeightedTrainer
from .progress_bar import NestedProgressBar
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments, 
                          DataCollatorWithPadding, 
                          AutoTokenizer, 
                          EarlyStoppingCallback)


def start_fine_tuning(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, tokenized_dataset: DatasetDict):
    # Set model to best option available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training model on device: {device}")
    model.to(device)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    class_weights = compute_class_weights(tokenized_dataset["train"], NUM_CLASSES)
    # Get training args
    training_args = get_training_arguments()
    # Get vars for nested progress bar
    num_epochs = training_args.num_train_epochs
    num_batches = len(tokenized_dataset["train"]) // training_args.per_device_train_batch_size

    nested_bar = NestedProgressBar(
        total_epochs=num_epochs,
        total_batches=num_batches,
        mode="train"
    )
    
    # Init trainer params
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights, # predicting rare classes matters more
        nested_bar=nested_bar
)
    
    # Fine tune model
    trainer.nested_bar = nested_bar
    trainer.train()
    nested_bar.close("Training finished.")
    
    trainer.save_model(MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    
    metrics = trainer.evaluate()
    print("Final Evaluation:", metrics)
    print(metrics)

def get_training_arguments():
    # Get training hyperparams from config.yaml
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)["training_args"]
    # Create model output if it doesn't exist 
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    #All hyperparmas come from config.yaml
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        overwrite_output_dir=cfg["overwrite_output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],

        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",

        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],

        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        bf16=False,
        fp16=False,
        dataloader_num_workers=0,

        seed=42,
    )
    return training_args


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """
    Computes the accuracy of the model's predictions during evaluation.

    Args:
        eval_pred (tuple): A tuple containing:
            - logits (np.ndarray): The raw prediction scores from the model.
              Shape: (batch_size, num_labels)
            - labels (np.ndarray): The actual correct integers (0-4) for the inputs.

    Returns:
        dict: A dictionary containing the accuracy score, e.g., {'accuracy': 0.85}
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def compute_class_weights(train_dataset, num_labels):
    
    labels = np.array(train_dataset["labels"]).astype(np.int64)
    classes = np.arange(num_labels).astype(np.int64)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    return torch.tensor(weights, dtype=torch.float)
    

