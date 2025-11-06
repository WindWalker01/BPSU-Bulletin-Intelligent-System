from app import app, db
from intelligent_system import train_classifier

if __name__ == "__main__":
    # Create Flask application context
    with app.app_context():
        print("🔄 Starting retraining process...")
        train_classifier()
        print("✅ Model retraining complete.")
