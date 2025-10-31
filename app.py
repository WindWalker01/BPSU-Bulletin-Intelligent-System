import os
from flask import Flask, request, jsonify
from intelligent_system import train_classifier, classify_text, add_training_data

import pickle

with open("vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    prediction_models = pickle.load(f)

app = Flask(__name__)


@app.route('/api/classify', methods=['POST'])
def classify():

    content = request.form.get('content')


    if not content:
        return jsonify({"error": "No input provided"}), 400


    classification = classify_text(content, tfidf_vectorizer, prediction_models)


    return jsonify({"classification" : classification}), 200


@app.route('/api/add', methods=['POST'])
def add():

    content = request.form.get('content')
    toxic = request.form.get('is-toxic')
    spam = request.form.get('is-spam')

    if not content or not toxic or not spam:
        return jsonify({"error": "No input provided"}), 400

    add_training_data(content, (toxic == "1"), (spam == "1"))

    return jsonify({"success": True}), 200


@app.route('/api/train', methods=['PATCH'])
def train():

    train_classifier()
    return jsonify({"success": True}), 200





@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
