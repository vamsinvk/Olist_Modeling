import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "modeling")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def train_model():
    print(f"🧠 ML STEP 2: Training Random Forest")
    print("=" * 60)

    # 1. Load Data
    print("   • Loading training data...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # 2. Initialize Model
    # n_estimators=100 -> Build 100 trees
    # random_state=42 -> Ensures we get same results every time
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 3. TRAIN (Fit)
    print("   • Training model (this may take a moment)...")
    rf.fit(X_train, y_train)

    # 4. PREDICT
    print("   • Testing on 22,003 unseen orders...")
    y_pred = rf.predict(X_test)

    # 5. EVALUATE
    print("\n📊 MODEL RESULTS:")
    print("-" * 30)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.2%}")
    print("\n   Detailed Report:")
    print(classification_report(y_test, y_pred))

    print("\n   Confusion Matrix (Truth vs Prediction):")
    print(confusion_matrix(y_test, y_pred))

    # 6. SAVE
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "rf_model_v1.pkl")
    joblib.dump(rf, model_path)
    print(f"\n✅ Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()