import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "final_table.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "final")


def train_final():
    print(f"🚀 TRAINING: The final Model (Context + NLP + Reputation)")
    print("=" * 60)

    # Setup Dirs
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)

    # 2. Prepare Target
    df['is_bad_review'] = df['review_score'].apply(lambda x: 1 if x <= 3 else 0)

    # 3. ENCODING: Categorical Columns
    # For 'customer_state', we use simple Label Encoding
    le = LabelEncoder()
    df['customer_state_code'] = le.fit_transform(df['customer_state'].astype(str))

    # 4. SPLIT (Crucial: Split BEFORE Target Encoding to avoid leakage)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 5. ADVANCED FEATURE ENGINEERING: Target Encoding (Risk Scores)
    print("   • Calculating Risk Scores (Seller & Category)...")

    # A. Product Risk
    # Calculate average bad review rate per category on TRAIN data
    cat_risk = train_df.groupby('product_category_name_english')['is_bad_review'].mean()
    global_avg = train_df['is_bad_review'].mean()

    # Map to Train and Test (Fill unknown categories with global average)
    train_df['category_risk'] = train_df['product_category_name_english'].map(cat_risk).fillna(global_avg)
    test_df['category_risk'] = test_df['product_category_name_english'].map(cat_risk).fillna(global_avg)

    # B. Seller Risk (Reputation)
    seller_risk = train_df.groupby('seller_id')['is_bad_review'].mean()
    train_df['seller_risk'] = train_df['seller_id'].map(seller_risk).fillna(global_avg)
    test_df['seller_risk'] = test_df['seller_id'].map(seller_risk).fillna(global_avg)

    # 6. SELECT FEATURES
    features = [
        # Logistics
        'actual_delivery_days', 'is_late', 'price', 'freight_value', 'product_weight_g',
        # Physics / Location
        'customer_state_code', 'is_same_state',
        # NLP
        'sentiment_score', 'review_length',
        # Context / Reputation
        'category_risk', 'seller_risk'
    ]

    X_train = train_df[features].fillna(0)
    y_train = train_df['is_bad_review']
    X_test = test_df[features].fillna(0)
    y_test = test_df['is_bad_review']

    print(f"   • Training Features: {features}")

    # 7. TRAIN
    print("\n   • Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 8. EVALUATE
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n🏆 FINAL final RESULTS:")
    print("-" * 30)
    print(f"   Accuracy: {acc:.2%}")
    print(f"   F1-Score: {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred))

    # 9. FEATURE IMPORTANCE
    importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('The Final Verdict: What drives Customer Happiness?')
    plt.savefig(os.path.join(FIGURE_DIR, "final_importance.png"))
    print(f"   ✅ Feature Importance saved to {FIGURE_DIR}")

    # 10. SAVE
    joblib.dump(rf, os.path.join(MODEL_DIR, "best_final_model.pkl"))
    # Save the Risk Maps too (needed for Dashboard)
    joblib.dump(cat_risk, os.path.join(MODEL_DIR, "category_risk_map.pkl"))
    joblib.dump(seller_risk, os.path.join(MODEL_DIR, "seller_risk_map.pkl"))
    print(f"   ✅ Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_final()