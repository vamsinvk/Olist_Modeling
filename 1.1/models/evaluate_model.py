import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.1"
RF_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model_v1.1.pkl")
CB_PATH = os.path.join(PROJECT_ROOT, "models", "cb_model_v1.1.pkl")
X_TEST_PATH = os.path.join(PROJECT_ROOT, "models", "X_test_balanced.csv")
Y_TEST_PATH = os.path.join(PROJECT_ROOT, "models", "y_test_balanced.csv")


def evaluate():
    print("📊 EVALUATION: Balanced Hybrid Ensemble...")

    if not os.path.exists(RF_PATH): return print("❌ Models not found.")

    rf_model = joblib.load(RF_PATH)
    cb_model = joblib.load(CB_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    # Soft Voting
    prob_rf = rf_model.predict_proba(X_test)[:, 1]
    prob_cb = cb_model.predict_proba(X_test)[:, 1]
    avg_prob = (prob_rf + prob_cb) / 2

    # Prediction (Threshold 0.5)
    y_pred = (avg_prob >= 0.5).astype(int)

    # Metrics
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 40)
    print(f"🔥 FINAL BALANCED F1-SCORE: {f1:.4f}")
    print("=" * 40)
    print("\nDetailed Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(f"True Negatives (Good -> Good): {cm[0][0]}")
    print(f"False Positives (Good -> Bad): {cm[0][1]}  <-- We accept more of these")
    print(f"False Negatives (Bad -> Good): {cm[1][0]}  <-- We want this LOWER")
    print(f"True Positives (Bad -> Bad):   {cm[1][1]}  <-- We want this HIGHER")

    # Feature Importance
    importances = rf_model.feature_importances_
    feature_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})
    print("\n🌟 TOP 10 REAL PREDICTORS (No Cheating):")
    print(feature_df.sort_values(by='Importance', ascending=False).head(10))


if __name__ == "__main__":
    evaluate()