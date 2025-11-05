import pandas as pd
import re, string, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from models import TrainingData

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


def train_classifier(session):
    print("🔄 Training model using database data...")
    data = session.query(TrainingData).all()
    if not data:
        raise ValueError("No training data found.")

    df = pd.DataFrame(
        [{"text": d.text, "toxic": int(d.toxic), "spam": int(d.spam)} for d in data]
    )
    df["text_clean"] = df["text"].apply(text_preprocessor)

    X_train = df["text_clean"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.7)
    X_vec = vectorizer.fit_transform(X_train)

    models = {}
    for label in ["toxic", "spam"]:
        y_train = df[label]
        if y_train.nunique() < 2:
            print(f"⚠️ Skipping {label} (only one class).")
            continue
        model = LogisticRegression(solver="liblinear", class_weight="balanced", C=4.0)
        model.fit(X_vec, y_train)
        models[label] = model
        print(f"✅ Trained model for: {label}")

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("model.pkl", "wb") as f:
        pickle.dump(models, f)

    print("🎉 Training complete. Models saved.")


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
