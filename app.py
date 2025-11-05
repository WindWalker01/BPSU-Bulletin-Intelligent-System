import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, TrainingData
from intelligent_system import classify_text

app = Flask(__name__)
CORS(app)

# MySQL connection (Render compatible)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    "DATABASE_URL", "mysql+pymysql://root@localhost/intelligent_system"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

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
    content = request.form.get("content")
    toxic = request.form.get("is-toxic")
    spam = request.form.get("is-spam")

    if not content or toxic is None or spam is None:
        return jsonify({"error": "Missing input"}), 400

    new_data = TrainingData(
        text=content,
        toxic=(toxic == "1"),
        spam=(spam == "1"),
    )
    db.session.add(new_data)
    db.session.commit()

    return jsonify({"success": True}), 200


@app.route("/")
def index():
    return "✅ Flask ML API running successfully!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
