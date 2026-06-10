from pathlib import Path
import sys, pickle
proj = Path(__file__).resolve().parent
sys.path.insert(0, str(proj))

# Test 1: Verify pickle files exist
print('✓ Pickle files:')
for pkl in ['vectorizer.pkl', 'label_encoder.pkl']:
    path = proj / pkl
    print(f'  - {pkl}: {path.exists()}')

# Test 2: Load pickles
print('\n✓ Loading pickles...')
with open(proj / 'vectorizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print(f'  Tokenizer loaded: {type(tokenizer).__name__}')

with open(proj / 'label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
print(f'  LabelEncoder loaded: {type(le).__name__}')
print(f'  Emotion classes: {le.classes_.tolist()}')

# Test 3: Import app modules
print('\n✓ Importing modules...')
from preprocessing import clean_text, build_tokenizer, load_dataframes
from predictor import build_text_model, load_saved_model
from visualization import plot_probability_chart
from utils import MODEL_CONFIG, MODEL_WEIGHT_PATH
print('  All imports successful')

# Test 4: Load data
print('\n✓ Loading data...')
train_df, test_df, _ = load_dataframes()
print(f'  Train samples: {len(train_df)}, Test samples: {len(test_df)}')

# Test 5: Build and load model
print('\n✓ Building model...')
model = build_text_model(
    vocab_size=MODEL_CONFIG['num_words'],
    input_length=MODEL_CONFIG['max_len'],
    embed_dim=MODEL_CONFIG['embed_dim'],
    units=MODEL_CONFIG['units'],
    dropout_rate=MODEL_CONFIG['dropout'],
    num_classes=len(le.classes_)
)
load_saved_model(model, MODEL_WEIGHT_PATH)
print(f'  Model loaded from {MODEL_WEIGHT_PATH.name}')

# Test 6: Quick prediction
print('\n✓ Testing prediction...')
test_text = "I love this amazing thing"
cleaned = clean_text(test_text)
from predictor import predict_emotion
emotion, probs, confidence = predict_emotion(cleaned, model, tokenizer, le, MODEL_CONFIG['max_len'])
print(f'  Input: "{test_text}"')
print(f'  Cleaned: "{cleaned}"')
print(f'  Predicted emotion: {emotion} (confidence: {confidence:.2%})')

print('\n✅ All startup tests passed!')
