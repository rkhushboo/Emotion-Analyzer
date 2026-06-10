from pathlib import Path
import pickle, sys
proj = Path(__file__).resolve().parent
sys.path.insert(0, str(proj))
from preprocessing import load_dataframes, build_tokenizer
from predictor import build_label_encoder
try:
    train_df, _, _ = load_dataframes()

    # build tokenizer with same defaults used in the app
    tokenizer = build_tokenizer(train_df['clean_text'].tolist(), num_words=10000, oov_token='<OOV>')
    label_encoder = build_label_encoder(train_df['emotion'].tolist())

    with open(proj / 'vectorizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(proj / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print('Saved:', proj / 'vectorizer.pkl', proj / 'label_encoder.pkl')
except Exception as exc:
    import traceback
    with open(proj / 'artifact_error.txt', 'w', encoding='utf-8') as ef:
        ef.write(str(exc) + "\n")
        traceback.print_exc(file=ef)
    print('Error during artifact creation. See artifact_error.txt in project folder.')
