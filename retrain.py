from app import db
from intelligent_system import train_classifier

if __name__ == "__main__":
    with db.engine.connect() as conn:
        train_classifier(db.session)
