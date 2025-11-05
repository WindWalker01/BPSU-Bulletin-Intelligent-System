import pandas as pd
import os
from app import app
from models import db, TrainingData

TRAIN_FILE = "train.csv"
LABELED_FILE = "labeled_data.csv"
SPAM_FILE = "spam.csv"
MANUAL_TRAINING_FILE = "manual_training_data.csv"


def import_csv_to_db():
    combined_data = []

    # --- 1️⃣ train.csv ---
    if os.path.exists(TRAIN_FILE):
        df_train = pd.read_csv(TRAIN_FILE)
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        existing_cols = [c for c in toxic_cols if c in df_train.columns]
        if existing_cols:
            df_train['toxic'] = (df_train[existing_cols].sum(axis=1) > 0).astype(int)
        else:
            df_train['toxic'] = 0
        df_train['spam'] = 0
        df_train = df_train.rename(columns={'comment_text': 'text'})[['text', 'toxic', 'spam']]
        combined_data.append(df_train)
        print(f"Loaded {len(df_train)} rows from {TRAIN_FILE}")

    # --- 2️⃣ labeled_data.csv ---
    if os.path.exists(LABELED_FILE):
        df_label = pd.read_csv(LABELED_FILE)
        df_label['toxic'] = (df_label['class'] != 2).astype(int)
        df_label['spam'] = 0
        df_label = df_label.rename(columns={'tweet': 'text'})[['text', 'toxic', 'spam']]
        combined_data.append(df_label)
        print(f"Loaded {len(df_label)} rows from {LABELED_FILE}")

    # --- 3️⃣ spam.csv ---
    if os.path.exists(SPAM_FILE):
        try:
            df_spam = pd.read_csv(SPAM_FILE, header=None, names=['v1', 'v2', 'v3', 'v4', 'v5'], encoding='latin1')
        except UnicodeDecodeError:
            df_spam = pd.read_csv(SPAM_FILE, header=None, names=['v1', 'v2', 'v3', 'v4', 'v5'],
                                  encoding='windows-1252', errors='ignore')
        df_spam['spam'] = (df_spam['v1'] == 'spam').astype(int)
        df_spam['toxic'] = 0
        df_spam = df_spam.rename(columns={'v2': 'text'})[['text', 'toxic', 'spam']]
        combined_data.append(df_spam)
        print(f"Loaded {len(df_spam)} rows from {SPAM_FILE}")

    # --- 4️⃣ manual_training_data.csv ---
    if os.path.exists(MANUAL_TRAINING_FILE):
        df_manual = pd.read_csv(MANUAL_TRAINING_FILE)
        if all(col in df_manual.columns for col in ['text', 'toxic', 'spam']):
            combined_data.append(df_manual)
            print(f"Loaded {len(df_manual)} rows from {MANUAL_TRAINING_FILE}")
        else:
            print(f"⚠️ {MANUAL_TRAINING_FILE} has unexpected columns, skipping.")

    # --- Combine all datasets ---
    if not combined_data:
        print("❌ No CSV files found. Nothing to import.")
        return

    df = pd.concat(combined_data, ignore_index=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Total combined dataset: {len(df)} rows.")

    # --- Check existing texts in DB ---
    print("🔍 Checking for duplicates already in database...")
    existing_texts = {t[0] for t in db.session.query(TrainingData.text).all()}

    # Filter new entries
    df_new = df[~df['text'].isin(existing_texts)]
    new_count = len(df_new)
    print(f"🆕 New samples to insert: {new_count}")

    if new_count == 0:
        print("✅ No new entries to add — database already up to date.")
        return

    # --- Insert new data ---
    for _, row in df_new.iterrows():
        entry = TrainingData(
            text=row['text'],
            toxic=bool(row['toxic']),
            spam=bool(row['spam'])
        )
        db.session.add(entry)

    db.session.commit()
    print(f"✅ Successfully inserted {new_count} new rows into the database!")


if __name__ == "__main__":
    with app.app_context():
        import_csv_to_db()
