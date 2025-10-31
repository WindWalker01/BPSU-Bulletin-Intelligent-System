import pandas as pd
import re
import string
import os  # Added for file operations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TRAIN_FILE = "train.csv"
LABELED_FILE = "labeled_data.csv"
SPAM_FILE = "spam.csv"
MANUAL_TRAINING_FILE = "manual_training_data.csv"


try:
    # Attempt to use full NLTK resources
    STOPWORDS = set(stopwords.words('english'))
    LEMMA = WordNetLemmatizer()
except LookupError:
    print("Warning: NLTK data missing. Using simplified preprocessing.")
    STOPWORDS = set()
    LEMMA = type('MockLemmatizer', (object,), {'lemmatize': lambda self, word: word})()


def text_preprocessor(text):
    """
    Cleans, normalizes, and tokenizes a single string of text.
    """
    text = str(text).lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Remove non-alphanumeric characters (KEEPING digits for spam detection)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 6. Tokenize, Stop Word Removal, and Lemmatize
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [LEMMA.lemmatize(word) for word in tokens]

    return " ".join(tokens)



def train_classifier():
    """
    Loads data from all CSV files (original and manual additions), standardizes the labels,
    combines the datasets, preprocesses the text, trains the TfidfVectorizer,
    and trains separate Logistic Regression models for 'toxic' (harmful) and 'spam'.
    Returns the vectorizer and the dictionary of trained models.
    """
    print("--- Starting Model Training with All Available Data ---")

    combined_data = []

    # --- Load and process train.csv (Jigsaw Toxic) ---
    print(f"1/4 Loading and processing {TRAIN_FILE}...")
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        # Create a unified 'toxic' label: 1 if ANY of the toxic sub-labels are 1
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        existing_toxic_cols = [col for col in toxic_cols if col in df_train.columns]
        if existing_toxic_cols:
            df_train['toxic'] = (df_train[existing_toxic_cols].sum(axis=1) > 0).astype(int)
        else:
            df_train['toxic'] = 0
        df_train['spam'] = 0
        df_train = df_train.rename(columns={'comment_text': 'text'})[['text', 'toxic', 'spam']]
        combined_data.append(df_train)
    except Exception as e:
        print(f"Error loading {TRAIN_FILE}: {e}")

    # --- Load and process labeled_data.csv (Hate Speech) ---
    print(f"2/4 Loading and processing {LABELED_FILE}...")
    try:
        df_labeled = pd.read_csv(LABELED_FILE)
        # The 'class' column is 0=hate, 1=offensive, 2=neither.
        # We classify 0 and 1 as toxic/harmful.
        df_labeled['toxic'] = (df_labeled['class'] != 2).astype(int)
        df_labeled['spam'] = 0
        df_labeled = df_labeled.rename(columns={'tweet': 'text'})[['text', 'toxic', 'spam']]
        combined_data.append(df_labeled)
    except Exception as e:
        print(f"Error loading {LABELED_FILE}: {e}")

    # --- Load and process spam.csv (SMS Spam) ---
    print(f"3/4 Loading and processing {SPAM_FILE}...")
    try:
        df_spam = pd.read_csv(SPAM_FILE, header=None, names=['v1', 'v2', 'v3', 'v4', 'v5'])
        df_spam['spam'] = (df_spam['v1'] == 'spam').astype(int)
        df_spam['toxic'] = 0
        df_spam = df_spam.rename(columns={'v2': 'text'})[['text', 'toxic', 'spam']]
        combined_data.append(df_spam)
    except Exception as e:
        print(f"Error loading {SPAM_FILE}: {e}")

    # --- Load and process manual_training_data.csv (User Additions) ---
    print(f"4/4 Checking for {MANUAL_TRAINING_FILE}...")
    try:
        if os.path.exists(MANUAL_TRAINING_FILE):
            df_manual = pd.read_csv(MANUAL_TRAINING_FILE)
            # Ensure the manual file has the expected columns, if not, skip it
            if all(col in df_manual.columns for col in ['text', 'toxic', 'spam']):
                combined_data.append(df_manual)
                print(f"Loaded {len(df_manual)} user-added samples.")
            else:
                print(f"Warning: {MANUAL_TRAINING_FILE} exists but has incorrect columns. Skipping.")
    except Exception as e:
        print(f"Error loading {MANUAL_TRAINING_FILE}: {e}")

    # Combine all datasets
    if not combined_data:
        raise ValueError("No data frames were loaded successfully. Cannot train model.")

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.drop_duplicates(subset=['text'], inplace=True)

    if 'toxic' not in combined_df.columns or 'spam' not in combined_df.columns:
        raise ValueError("Required 'toxic' or 'spam' label column missing after merging.")

    print(f"Combined Data Size: {len(combined_df)} rows.")

    # Apply Text Preprocessing
    combined_df['text_clean'] = combined_df['text'].fillna('').apply(text_preprocessor)
    print("Text cleaning complete.")

    X_train = combined_df['text_clean']

    # 1. Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7
    )

    # Fit the vectorizer
    X_vec = vectorizer.fit_transform(X_train)

    # Train separate models for each label
    models = {}
    labels = ['toxic', 'spam']

    for label in labels:
        y_train = combined_df[label]

        # FIX: Check for single-class data before fitting
        if y_train.nunique() < 2:
            print(
                f"Warning: Skipping training for '{label.upper()}' as it contains only one class ({y_train.iloc[0]}) after data aggregation/filtering.")
            continue

        # Use a balanced class weight to help with the class imbalance common in these tasks
        model = LogisticRegression(solver='liblinear', class_weight='balanced', C=4.0)
        model.fit(X_vec, y_train)
        models[label] = model
        print(f"Trained model for: {label.upper()}")

    if not models:
        raise ValueError("No models were trained successfully. Please check your data classes.")

    print("--- Training Complete. Models are ready. ---")
    return vectorizer, models



def classify_text(text: str, vectorizer: TfidfVectorizer, models: dict):
    """
    Takes a raw string, cleans it, vectorizes it, and predicts its category
    based on the trained models.

    Args:
        text: The raw input string to classify.
        vectorizer: The fitted TfidfVectorizer.
        models: A dictionary containing the trained LogisticRegression models
                for each category ('toxic', 'spam').

    Returns:
        The predicted classification: 'Harmful (Toxic)', 'Spam', or 'Safe'.
    """
    if not text or not isinstance(text, str):
        return "Safe (Empty or Invalid Input)"

    # 1. Preprocess the input text
    cleaned_text = text_preprocessor(text)

    # 2. Vectorize the cleaned text
    text_vec = vectorizer.transform([cleaned_text])

    # 3. Predict probabilities for each label
    predictions = {'toxic': 0.0, 'spam': 0.0}

    # Predict TOXICITY (Harmful)
    if 'toxic' in models:
        toxic_proba = models['toxic'].predict_proba(text_vec)[0][1]
        predictions['toxic'] = toxic_proba
    else:
        toxic_proba = 0.0

    # Predict SPAM
    if 'spam' in models:
        spam_proba = models['spam'].predict_proba(text_vec)[0][1]
        predictions['spam'] = spam_proba
    else:
        spam_proba = 0.0

    # Determine the final classification based on thresholds
    TOXIC_THRESHOLD = 0.55
    SPAM_THRESHOLD = 0.55

    # Priority 1: Check for Harmful (Toxic) content
    if predictions['toxic'] >= TOXIC_THRESHOLD:
        return {"content": text, "prediction": "HARMFUL", "probability": predictions}

    # Priority 2: Check for Spam
    elif predictions['spam'] >= SPAM_THRESHOLD:
        return {"content": text, "prediction": "SPAM", "probability": predictions}

    # Priority 3: Default to Safe
    else:
        return {"content": text, "prediction": "SAFE", "probability": predictions}



def add_training_data(text: str, is_toxic: bool, is_spam: bool):
    """
    Adds a new sample to the persistent manual training file.
    The model must be re-trained by calling train_classifier() after adding data
    for the change to take effect.

    Args:
        text: The raw text of the comment/message.
        is_toxic: Boolean (True/False) indicating if the text is harmful/toxic.
        is_spam: Boolean (True/False) indicating if the text is spam.
    """
    # Convert booleans to 1/0 integers
    toxic_val = 1 if is_toxic else 0
    spam_val = 1 if is_spam else 0

    new_data = pd.DataFrame([{
        'text': text,
        'toxic': toxic_val,
        'spam': spam_val
    }])

    # Check if the file already exists
    if os.path.exists(MANUAL_TRAINING_FILE):
        # Append without header
        new_data.to_csv(MANUAL_TRAINING_FILE, mode='a', header=False, index=False)
        print(f"\nSuccessfully appended 1 new sample to {MANUAL_TRAINING_FILE}.")
    else:
        # Create a new file with header
        new_data.to_csv(MANUAL_TRAINING_FILE, mode='w', header=True, index=False)
        print(f"\nSuccessfully created and saved 1 new sample to {MANUAL_TRAINING_FILE}.")

    print("REMINDER: Call train_classifier() to update the model with this new data!")