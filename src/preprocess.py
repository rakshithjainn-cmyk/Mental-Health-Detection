import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Lowercase, clean symbols, remove stopwords, and lemmatize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and preprocess text/labels."""
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
    return df
