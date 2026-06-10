from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_WEIGHT_PATH = BASE_DIR / 'best_bidirectional_gru_weights.h5'

MODEL_CONFIG = {
    'num_words': 10000,
    'oov_token': '<OOV>',
    'max_len': 100,
    'embed_dim': 64,
    'units': 128,
    'dropout': 0.2,
}

MODEL_INFO = {
    'name': 'Tuned Bidirectional GRU',
    'description': 'A Keras Bidirectional GRU text classifier trained on the Kaggle Emotions Dataset.',
    'dataset': 'Kaggle/emotions-dataset-for-nlp',
    'architecture': 'Embedding + Bidirectional GRU + Dropout + Dense(softmax)',
    'best_tuner_trial': 'trial_2',
    'best_accuracy': 0.9368749856948853,
}
