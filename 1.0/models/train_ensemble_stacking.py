import pandas as pd
import os
import joblib
import numpy as np
import catboost as cb_lib  # Import with alias to avoid local naming conflicts
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "final_table.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_hybrid")


def train_hybrid_final():
    print(f"👑 THE FINAL BOSS: Dual-Engine Hybrid (RF + CatBoost)")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load and Prepare Target
    df = pd.read_csv(INPUT_FILE)
    df['is_bad_review'] = df['review_score'].apply(lambda x: 1 if x <= 3 else 0)

    # 2. Feature Engineering for Hybrid
    cat_cols = ['product_category_name_english', 'customer_state']
    num_cols = ['actual_delivery_days', 'is_late', 'price', 'freight_value',
                'product_weight_g', 'sentiment_score', 'review_length']

    # We must encode for RF and Ensemble unity
    le_cat = LabelEncoder()
    le_state = LabelEncoder()
    df['cat_encoded'] = le_cat.fit_transform(df['product_category_name_english'].fillna('unknown').astype(str))
    df['state_encoded'] = le_state.fit_transform(df['customer_state'].fillna('unknown').astype(str))

    X = df[['cat_encoded', 'state_encoded'] + num_cols].fillna(0)
    y = df['is_bad_review']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Model A: Random Forest (The Conservative Expert)
    print("   • Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 4. Train Model B: CatBoost (The Aggressive Specialist)
    print("   • Training CatBoost...")
    cb = cb_lib.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0,
                                   auto_class_weights='Balanced')
    cb.fit(X_train, y_train)

    # 5. THE HYBRID STEP: Weighted Soft Voting
    # We give slightly more weight to RF for Precision and CatBoost for Recall
    print("   • Blending predictions...")

    # Get probabilities for "Bad Review" (Class 1)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    cb_probs = cb.predict_proba(X_test)[:, 1]

    # 50/50 Blend
    hybrid_probs = (rf_probs + cb_probs) / 2

    # Final Classification
    y_pred_hybrid = (hybrid_probs > 0.5).astype(int)

    # 6. Final Evaluation
    f1 = f1_score(y_test, y_pred_hybrid)
    acc = accuracy_score(y_test, y_pred_hybrid)

    print("\n🏆 HYBRID MODEL RESULTS:")
    print("-" * 30)
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Accuracy: {acc:.2%}")
    print("\n" + classification_report(y_test, y_pred_hybrid))

    # 7. Save Everything
    joblib.dump(rf, os.path.join(MODEL_DIR, "hybrid_rf.pkl"))
    cb.save_model(os.path.join(MODEL_DIR, "hybrid_cb.bin"))
    joblib.dump(le_cat, os.path.join(MODEL_DIR, "le_cat.pkl"))
    joblib.dump(le_state, os.path.join(MODEL_DIR, "le_state.pkl"))
    print(f"   ✅ All hybrid components saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_hybrid_final()