from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class TrainingData(db.Model):
    __tablename__ = "training_data"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    toxic = db.Column(db.Boolean, default=False)
    spam = db.Column(db.Boolean, default=False)
