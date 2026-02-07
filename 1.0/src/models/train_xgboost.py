import pandas as pd
import xgboost as xgb
import os
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "modeling")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def train_xgboost():
    print(f"🚀 UPGRADE: Training XGBoost (The Industry Standard)")
    print("=" * 60)

    # 1. Load Data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # 2. Initialize XGBoost
    # we use 'scale_pos_weight' because we know our classes are imbalanced (only 23% angry)
    # This tells the model: "Pay 3x more attention to the Angry class!"
    ratio = float(len(y_train[y_train == 0])) / len(y_train[y_train == 1])

    model = xgb.XGBClassifier(
        n_estimators=200,  # More trees
        learning_rate=0.05,  # Learn slower but deeper
        max_depth=6,  # Deeper trees
        scale_pos_weight=ratio,  # Handle the imbalance automatically!
        random_state=42,
        n_jobs=-1
    )

    # 3. Train
    print("   • Training... (This finds patterns RF might miss)")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    # 4. Evaluate
    y_pred = model.predict(X_test)

    print("\n📊 XGBOOST RESULTS:")
    print(f"   • Time Taken: {elapsed:.2f}s")
    print(f"   • Accuracy:   {accuracy_score(y_test, y_pred):.2%}")
    print(f"   • Precision:  {precision_score(y_test, y_pred):.2%}")
    print(f"   • Recall:     {recall_score(y_test, y_pred):.2%}  <-- Watch this number!")
    print(f"   • F1-Score:   {f1_score(y_test, y_pred):.4f}")

    # 5. Save
    joblib.dump(model, os.path.join(MODEL_DIR, "best_model_xgb.pkl"))
    print("\n✅ Saved as 'best_model_xgb.pkl'")


if __name__ == "__main__":
    train_xgboost()