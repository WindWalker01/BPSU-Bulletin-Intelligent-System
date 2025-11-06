import pickle
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from intelligent_system import classify_text

from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import ssl, os

app = Flask(__name__)
CORS(app)

# Get the database URL
uri = os.getenv("DATABASE_URL", "sqlite:///local.db")

# Render provides DATABASE_URL starting with "postgresql://"
# SQLAlchemy expects "+pg8000" to use the correct driver
if uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql+pg8000://", 1)
elif uri.startswith("postgresql://"):
    uri = uri.replace("postgresql://", "postgresql+pg8000://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

with app.app_context():
    db.create_all()

# Load trained model
try:
    with open("vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        prediction_models = pickle.load(f)
except Exception as e:
    print("⚠️ Warning: model files not found. Classification endpoints may fail.")
    tfidf_vectorizer = None
    prediction_models = None


@app.route("/api/classify", methods=["POST"])
def classify():
    content = request.form.get("content")
    if not content:
        return jsonify({"error": "No input provided"}), 400

    if not tfidf_vectorizer or not prediction_models:
        return jsonify({"error": "Model not loaded"}), 500

    classification = classify_text(content, tfidf_vectorizer, prediction_models)
    return jsonify({"classification": classification}), 200


@app.route("/api/add", methods=["POST"])
def add():
    # --- Load environment ---
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        return jsonify({"error": "Database URL not configured"}), 500

    # --- Secure SSL connection for Render PostgreSQL ---
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # --- Create DB engine ---
    engine = create_engine(db_url, connect_args={"ssl_context": ssl_context})

    # --- Parse incoming form data ---
    content = request.form.get("content")
    toxic = request.form.get("is-toxic")
    spam = request.form.get("is-spam")

    if not content or toxic is None or spam is None:
        return jsonify({"error": "Missing input"}), 400

    # --- Prepare SQL INSERT ---
    query = text("""
        INSERT INTO training_data (text, toxic, spam)
        VALUES (:text, :toxic, :spam)
    """)

    params = {
        "text": content.strip(),
        "toxic": True if toxic == "1" else False,
        "spam": True if spam == "1" else False
    }

    # --- Execute safely ---
    try:
        with engine.begin() as conn:
            conn.execute(query, params)
        print(f"✅ Added training sample: {params}")
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"❌ Database insert failed: {e}")
        return jsonify({"error": "Database insert failed"}), 500


@app.route("/")
def index():
    return "✅ Flask ML API running successfully!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
