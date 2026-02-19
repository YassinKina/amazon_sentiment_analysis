import streamlit as st
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .constants import HF_MODEL_NAME



# --- INFERENCE FUNCTION ---
def analyze_review(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the winner
    predicted_class = torch.argmax(probabilities).item()

    # Map 0-4 index to 1-5 Stars
    star_rating = predicted_class + 1
    confidence = probabilities[0][predicted_class].item()

    # Return everything for the UI
    return star_rating, confidence, probabilities[0].tolist()


@st.cache_resource
def load_model() ->tuple[AutoTokenizer,AutoModelForSequenceClassification]:
    """
    Loads the tokenizer and model
    Returns:
        tuple: (AutoTokenizer, AutoModelForSequenceClassification)

    """

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
        return tokenizer, model
    except OSError:
        return None, None

def analyze_csv(uploaded_file, model, tokenizer):
    """
    Reads a CSV, predicts sentiment for all reviews, and returns the DataFrame with results.
    """
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Check for Text Column
    possible_columns = ["text", "review", "review_body", "content", "comment"]
    text_col = next((col for col in df.columns if col.lower() in possible_columns), None)

    if not text_col:
        st.error(f"Could not find a review column. Please name your column one of: {possible_columns}")
        return None

    # Batch Inference 
    results = []
    progress_bar = st.progress(0)
    total_rows = len(df)

    st.write(f"🔄 Analyzing {total_rows} reviews...")

    for i, row in df.iterrows():
        text = str(row[text_col])  
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item() + 1  # Convert 0-4 to 1-5
            results.append(prediction)

        # Update progress bar every 10%
        if i % (total_rows // 10 + 1) == 0:
            progress_bar.progress((i + 1) / total_rows)

    progress_bar.progress(1.0) 
    df["Predicted_Sentiment"] = results
    return df