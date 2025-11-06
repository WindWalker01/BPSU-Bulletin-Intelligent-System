import os

import re, string
import ssl

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from models import TrainingData  # ORM model
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

try:
    STOPWORDS = set(stopwords.words('english'))
    LEMMA = WordNetLemmatizer()
except LookupError:
    STOPWORDS = set()
    LEMMA = type('MockLemmatizer', (object,), {'lemmatize': lambda self, w: w})()


def text_preprocessor(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = [LEMMA.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)


def get_db_session():
    """Connect to production database using pg8000 driver."""
    load_dotenv()
    db_url = os.getenv("IS_DATABASE_URL")

    print(db_url)

    if not db_url:
        raise EnvironmentError(
            "❌ DATABASE_URL environment variable not set. Example:\n"
            "export DATABASE_URL='postgresql+pg8000://user:pass@host:5432/dbname'"
        )

    # Create SQLAlchemy engine with pg8000
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()

def train_classifier():
    """
    Train moderation models using local exported data if available,
    otherwise fall back to production PostgreSQL database.
    """

    print("🚀 Starting model retraining...")

    load_dotenv()
    local_csv_path = "database.csv"
    df = None

    # --- 1️⃣ Try using local CSV if it exists ---
    if os.path.exists(local_csv_path):
        print(f"📂 Found local dataset: {local_csv_path}")
        df = pd.read_csv(local_csv_path)
        print(f"📊 Loaded {len(df):,} rows from CSV.")
    else:
        # --- 2️⃣ Otherwise, connect to live DB ---
        print("🌐 No CSV found — connecting to database instead...")
        db_url = os.getenv("IS_DATABASE_URL")
        if not db_url:
            raise EnvironmentError("❌ IS_DATABASE_URL not set in .env file.")

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        engine = create_engine(db_url, connect_args={"ssl_context": ssl_context})

        query = text("SELECT text, toxic, spam FROM training_data;")
        df_iter = pd.read_sql(query, engine, chunksize=5000)
        df = pd.concat(df_iter, ignore_index=True)
        print(f"📊 Retrieved {len(df):,} rows from database.")

    if df.empty:
        raise ValueError("❌ No training data available.")

    # --- Preprocess text (with progress bar) ---
    print("🧹 Cleaning text data...")
    tqdm.pandas(desc="Cleaning Text")
    df["text_clean"] = df["text"].astype(str).progress_apply(text_preprocessor)

    # --- Vectorize ---
    print("⚙️ Vectorizing text...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.7,
        sublinear_tf=True,
        lowercase=True,
    )
    X_vec = vectorizer.fit_transform(df["text_clean"])

    # --- Train models ---
    print("🧠 Training models...")
    models = {}
    for label in tqdm(["toxic", "spam"], desc="Training Models"):
        y_train = df[label].astype(int)
        if y_train.nunique() < 2:
            print(f"⚠️ Skipping '{label}' — only one class present.")
            continue
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=4.0,
            max_iter=500,
            n_jobs=-1,
        )
        model.fit(X_vec, y_train)
        models[label] = model

    # --- Save trained artifacts ---
    print("💾 Saving model and vectorizer...")
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("model.pkl", "wb") as f:
        pickle.dump(models, f)

    print("🎉 Training complete — saved 'vectorizer.pkl' and 'model.pkl'.")

def classify_text(text, vectorizer, models):
    cleaned_text = text_preprocessor(text)
    text_vec = vectorizer.transform([cleaned_text])

    preds = {"toxic": 0.0, "spam": 0.0}
    if "toxic" in models:
        preds["toxic"] = round(models["toxic"].predict_proba(text_vec)[0][1], 4)
    if "spam" in models:
        preds["spam"] = round(models["spam"].predict_proba(text_vec)[0][1], 4)

    if preds["toxic"] >= 0.55:
        result = "HARMFUL"
    elif preds["spam"] >= 0.55:
        result = "SPAM"
    else:
        result = "SAFE"

    return {"content": text, "prediction": result, "probability": preds}
