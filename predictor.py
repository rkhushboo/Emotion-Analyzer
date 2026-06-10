from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dropout, Dense
from preprocessing import prepare_sequence


def build_label_encoder(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder


def build_text_model(vocab_size, input_length, embed_dim, units, dropout_rate, num_classes):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=input_length))
    model.add(Bidirectional(GRU(units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def load_saved_model(model, weight_path: Path):
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
    model.load_weights(str(weight_path))
    return model


def predict_emotion(cleaned_text, model, tokenizer, label_encoder, max_len):
    sequence = prepare_sequence(tokenizer, [cleaned_text], max_len)
    probabilities = model.predict(sequence, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])
    return predicted_label, probabilities.tolist(), confidence


def batch_predict(batch_df, model, tokenizer, label_encoder, max_len):
    cleaned_texts = batch_df['clean_text'].astype(str).tolist()
    sequences = prepare_sequence(tokenizer, cleaned_texts, max_len)
    predictions = model.predict(sequences, verbose=0)
    predicted_indices = np.argmax(predictions, axis=-1)
    batch_df['Predicted Emotion'] = label_encoder.inverse_transform(predicted_indices)
    batch_df['Confidence'] = [float(predictions[i, idx]) for i, idx in enumerate(predicted_indices)]
    return batch_df


def evaluate_model_metrics(model, tokenizer, label_encoder, test_texts, test_labels, max_len=100):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    cleaned_embeddings = prepare_sequence(tokenizer, test_texts, max_len)
    predictions = model.predict(cleaned_embeddings, verbose=0)
    predicted_indices = np.argmax(predictions, axis=-1)
    true_indices = label_encoder.transform(test_labels)

    return {
        'accuracy': float(accuracy_score(true_indices, predicted_indices)),
        'precision': float(precision_score(true_indices, predicted_indices, average='weighted', zero_division=0)),
        'recall': float(recall_score(true_indices, predicted_indices, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(true_indices, predicted_indices, average='weighted', zero_division=0)),
    }
