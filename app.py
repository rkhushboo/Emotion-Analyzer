from pathlib import Path
import streamlit as st
import pandas as pd
import pickle
from preprocessing import download_nltk_resources, clean_text, build_tokenizer, prepare_sequence, load_dataframes
from predictor import (
    build_label_encoder,
    build_text_model,
    load_saved_model,
    predict_emotion,
    batch_predict,
    evaluate_model_metrics,
)
from visualization import plot_probability_chart, plot_distribution_chart, plot_history_table
from utils import MODEL_INFO, MODEL_CONFIG, MODEL_WEIGHT_PATH

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Emotion Analyzer", page_icon="😊", layout="wide"
)

@st.cache_data(show_spinner=False)
def initialize_data():
    download_nltk_resources()
    train_df, test_df, _ = load_dataframes()
    vectorizer_path = BASE_DIR / 'vectorizer.pkl'
    label_encoder_path = BASE_DIR / 'label_encoder.pkl'

    if vectorizer_path.exists():
        with open(vectorizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = build_tokenizer(train_df['clean_text'].tolist(), MODEL_CONFIG['num_words'], MODEL_CONFIG['oov_token'])

    if label_encoder_path.exists():
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        label_encoder = build_label_encoder(train_df['emotion'].tolist())
    model = build_text_model(
        vocab_size=MODEL_CONFIG['num_words'],
        input_length=MODEL_CONFIG['max_len'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        units=MODEL_CONFIG['units'],
        dropout_rate=MODEL_CONFIG['dropout'],
        num_classes=len(label_encoder.classes_),
    )
    load_saved_model(model, MODEL_WEIGHT_PATH)
    test_metrics = evaluate_model_metrics(
        model,
        tokenizer,
        label_encoder,
        test_texts=test_df['clean_text'].tolist(),
        test_labels=test_df['emotion'].tolist(),
    )
    return {
        'train_df': train_df,
        'test_df': test_df,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder,
        'model': model,
        'test_metrics': test_metrics,
    }

@st.cache_resource
def load_app_model():
    data = initialize_data()
    return data

if 'history' not in st.session_state:
    st.session_state.history = []

model_data = load_app_model()
train_df = model_data['train_df']
test_metrics = model_data['test_metrics']
model = model_data['model']
tokenizer = model_data['tokenizer']
label_encoder = model_data['label_encoder']

st.title("Emotion Analyzer")
st.write(
    "A production-ready emotion detection app powered by the tuned Bidirectional GRU model trained on the Kaggle Emotions Dataset."
)

with st.sidebar:
    st.header("Navigation")
    nav_choice = st.radio(
        "Go to",
        ["Prediction", "Batch Upload", "Dashboard", "About Model"],
        index=0,
    )
    st.markdown("---")
    st.subheader("Model Summary")
    st.metric("Model", MODEL_INFO['name'])
    st.metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
    st.metric("Weighted F1", f"{test_metrics['f1_score']:.2%}")
    st.caption("Dataset: Kaggle/emotions-dataset-for-nlp")

if nav_choice == "Prediction":
    st.header("Single Text Emotion Prediction")
    st.write("Enter a sentence to predict the underlying emotion and review the confidence spread.")
    user_text = st.text_area("Type a message here...", height=160)
    submitted = st.button("Predict Emotion")

    if submitted:
        if not user_text or not user_text.strip():
            st.warning("Please enter some text before predicting.")
        else:
            cleaned = clean_text(user_text)
            prediction, probability, confidence = predict_emotion(
                cleaned, model, tokenizer, label_encoder, MODEL_CONFIG['max_len']
            )
            st.success(f"Predicted Emotion: {prediction}")
            st.metric("Confidence", f"{confidence:.2%}")
            st.markdown("**Prediction Summary**")
            st.write(
                f"The model interprets the text as **{prediction}** with a confidence score of **{confidence:.2%}**. "
                "Probability values reflect the emotion distribution across the six target classes."
            )
            st.plotly_chart(
                plot_probability_chart(label_encoder.classes_.tolist(), probability),
                use_container_width=True,
            )
            st.session_state.history.insert(
                0,
                {
                    'text': user_text,
                    'clean_text': cleaned,
                    'prediction': prediction,
                    'confidence': confidence,
                },
            )

elif nav_choice == "Batch Upload":
    st.header("Bulk Emotion Prediction")
    st.write(
        "Upload a CSV file with a `text` column and receive emotion predictions for every sentence."
    )
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if 'text' not in batch_df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                batch_df['clean_text'] = batch_df['text'].astype(str).apply(clean_text)
                results_df = batch_predict(
                    batch_df,
                    model,
                    tokenizer,
                    label_encoder,
                    max_len=MODEL_CONFIG['max_len'],
                )
                st.success("Batch prediction complete.")
                st.dataframe(results_df[['text', 'Predicted Emotion', 'Confidence']].head(15))
                csv_download = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions",
                    data=csv_download,
                    file_name='emotion_predictions.csv',
                    mime='text/csv',
                )
                st.session_state.history = [
                    {
                        'text': row['text'],
                        'prediction': row['Predicted Emotion'],
                        'confidence': row['Confidence'],
                    }
                    for _, row in results_df.iterrows()
                ]
        except Exception as exc:
            st.error(f"Unable to process file: {exc}")

elif nav_choice == "Dashboard":
    st.header("Session Dashboard")
    st.write("Monitor prediction history and emotion distribution during the session.")
    history_table = plot_history_table(st.session_state.history)
    if history_table.empty:
        st.info("No predictions yet. Run a single prediction or upload a batch to populate the dashboard.")
    else:
        st.dataframe(history_table)
        fig = plot_distribution_chart(history_table)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Session Analytics")
        st.metric("Total Predictions", len(history_table))
        emotion_counts = history_table['prediction'].value_counts().to_dict()
        st.write(emotion_counts)

else:
    st.header("About the Model")
    st.markdown(
        "This application deploys a tuned Bidirectional GRU model trained on the Kaggle Emotions Dataset. "
        "The preprocessing pipeline includes emoji conversion, emoticon normalization, lowercasing, punctuation removal, stopword filtering, and lemmatization."
    )
    st.markdown("### Model Details")
    st.write(MODEL_INFO)
    st.markdown("### Test Performance")
    st.write(test_metrics)
    st.markdown("### Dataset Information")
    st.write(
        {
            'Training samples': len(train_df),
            'Emotion classes': list(label_encoder.classes_),
            'Source': 'Kaggle/emotions-dataset-for-nlp',
        }
    )
