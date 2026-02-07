import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.1"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "final_dataset_basic.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_balanced():
    print("🚀 TRAINING: Balanced Dual-Engine (No Leaks)...")

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)

    # Target: 1 = Bad Experience (<= 3 stars), 0 = Good.
    df['target'] = np.where(df['review_score'] <= 3, 1, 0)

    # 2. FEATURE SELECTION (CRITICAL CHANGE)
    # We DROP the features that happen *after* the review is written.
    # We want to predict satisfaction based on Logistics, Physics, and History only.
    leak_cols = ['comment_length', 'has_comment', 'response_time_hours']
    meta_cols = ['review_score', 'target', 'cust_lat', 'cust_lng', 'sell_lat', 'sell_lng']

    drop_cols = leak_cols + meta_cols
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['target']

    print(f"   • Dropped Leaks: {leak_cols}")

    # 3. Encode Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   • Split: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}")

    # 5. MODEL 1: RANDOM FOREST (With Class Weights)
    print("   • Training Random Forest (Balanced)...")
    rf = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,  # Deeper to find subtle patterns
        class_weight='balanced',  # <--- THE MAGIC FIX
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model_v1.1.pkl"))

    # 6. MODEL 2: CATBOOST (With Class Weights)
    print("   • Training CatBoost (Balanced)...")
    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=7,
        auto_class_weights='Balanced',  # <--- THE MAGIC FIX
        verbose=0,
        random_seed=42
    )
    cb.fit(X_train, y_train)
    joblib.dump(cb, os.path.join(MODEL_DIR, "cb_model_v1.1.pkl"))

    # 7. Save Test Data (Overwrite previous)
    X_test.to_csv(os.path.join(MODEL_DIR, "X_test_balanced.csv"), index=False)
    y_test.to_csv(os.path.join(MODEL_DIR, "y_test_balanced.csv"), index=False)

    print("✅ SUCCESS: Balanced Models Trained.")


if __name__ == "__main__":
    train_balanced()