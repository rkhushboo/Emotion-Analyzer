import re
import string
from pathlib import Path
try:
    import emoji
except Exception:
    emoji = None
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Robust data directory discovery: look up to 3 parents for a 'data' folder
def find_data_dir(start_path: Path, max_levels=3):
    cur = start_path
    for _ in range(max_levels + 1):
        candidate = cur / 'data'
        if candidate.exists() and candidate.is_dir():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return start_path / 'data'

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = find_data_dir(BASE_DIR)

EMOTICONS_DICT = {
    ":)": "smile",
    ":-)": "smile",
    ";)": "wink",
    ":(": "sad",
    ":-(": "sad",
    ":D": "laugh",
    "XD": "laugh",
    "<3": "love",
    "B)": "cool",
}

lemmatizer = WordNetLemmatizer()


def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


def convert_emoticons(text: str) -> str:
    for emoticon, word in EMOTICONS_DICT.items():
        text = text.replace(emoticon, word)
    return text


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    if emoji is not None:
        text = emoji.demojize(text, delimiters=(" ", " "))
    else:
        text = text
    text = convert_emoticons(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def build_tokenizer(texts, num_words=10000, oov_token='<OOV>'):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def prepare_sequence(tokenizer, texts, max_len=100):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')


def load_dataframes():
    train_path = DATA_DIR / 'train.txt'
    test_path = DATA_DIR / 'test.txt'
    val_path = DATA_DIR / 'val.txt'
    train_df = pd.read_csv(train_path, sep=';', names=['text', 'emotion'])
    test_df = pd.read_csv(test_path, sep=';', names=['text', 'emotion'])
    val_df = pd.read_csv(val_path, sep=';', names=['text', 'emotion'])
    for df in (train_df, test_df, val_df):
        df['clean_text'] = df['text'].astype(str).apply(clean_text)
    return train_df, test_df, val_df
