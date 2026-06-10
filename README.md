# Emotion Analyzer Streamlit App

This project deploys a production-ready emotion analysis application built from the uploaded notebook. It uses a tuned Bidirectional GRU model trained on the Kaggle `emotions-dataset-for-nlp` dataset.

## Project Structure

- `app.py` - Main Streamlit application.
- `requirements.txt` - Python dependencies.
- `README.md` - Project documentation.
- `models/` - Saved model weights used for inference.
- `src/`
  - `preprocessing.py` - Text cleaning, tokenizer creation, and dataset loading.
  - `predictor.py` - Model definition, weight loading, and prediction functions.
  - `visualization.py` - Chart helpers for probability distribution and dashboard analytics.
  - `utils.py` - Shared configuration, file paths, and model metadata.

## How It Works

- Loads the training dataset and reuses the notebook preprocessing pipeline:
  - emoji conversion
  - emoticon normalization
  - lowercasing
  - punctuation removal
  - stopword removal
  - lemmatization
- Builds the Keras Bidirectional GRU architecture matching the notebook.
- Loads saved weights from `models/best_bidirectional_gru_weights.h5`.
- Supports single-text prediction, CSV batch prediction, session dashboard, and model analytics.

## Run Locally

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment Notes

- This application is designed for Streamlit Cloud deployment with minimal modification.
- The model weights are stored in `project/models` and are loaded at runtime using a cached resource.
- The dataset files are sourced from the existing `data/` folder in the workspace.
