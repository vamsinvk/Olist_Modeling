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
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "modeling", "nlp_ready_data.csv")

# 📂 OUTPUT DIRECTORIES
DATA_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "modeling", "nlp")
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models", "nlp")
FIGURE_OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "nlp")


def train_nlp_model():
    print(f"🤖 NLP PIPELINE STEP 3: Training with Sentiment Features")
    print("=" * 60)

    # 1. Setup Directories
    for folder in [DATA_OUT_DIR, MODEL_OUT_DIR, FIGURE_OUT_DIR]:
        os.makedirs(folder, exist_ok=True)

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: nlp_ready_data.csv not found.")
        return

    # 3. Create Target
    # 1-3 Stars = Bad (1), 4-5 Stars = Good (0)
    df['is_bad_review'] = df['review_score'].apply(lambda x: 1 if x <= 3 else 0)

    # Drop columns we don't need (review_score causes leakage)
    X = df.drop(columns=['review_score', 'is_bad_review'])
    y = df['is_bad_review']

    # --- 3.5 FIX: ENCODE CATEGORICAL DATA ---
    print("   • Encoding 'customer_state' (Text -> Numbers)...")
    le = LabelEncoder()
    # Forces column to string type just in case mixed types exist
    X['customer_state'] = le.fit_transform(X['customer_state'].astype(str))

    # Fill any remaining NaNs with 0 to prevent crashes
    X = X.fillna(0)

    print(f"   • Training Data Shape: {X.shape}")
    print(f"   • Features: {list(X.columns)}")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save Split Data
    X_train.to_csv(os.path.join(DATA_OUT_DIR, "X_train_nlp.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_OUT_DIR, "X_test_nlp.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_OUT_DIR, "y_train_nlp.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_OUT_DIR, "y_test_nlp.csv"), index=False)

    # 5. Train Random Forest
    print("\n   • Training Random Forest (w/ NLP)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 RESULTS (NLP VERSION):")
    print("-" * 30)
    print(f"   Accuracy: {acc:.2%}")
    print(f"   F1-Score: {f1:.4f} (Previous Best: 0.5435)")
    print("\n   Detailed Report:")
    print(classification_report(y_test, y_pred))

    # 7. Feature Importance Plot
    importances = rf.feature_importances_
    feature_names = X.columns

    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='magma')
    plt.title('What Drives Reviews? (Now with NLP)')

    fig_path = os.path.join(FIGURE_OUT_DIR, "nlp_feature_importance.png")
    plt.savefig(fig_path)
    print(f"\n   ✅ Chart saved to: {fig_path}")

    # 8. Save Model
    model_path = os.path.join(MODEL_OUT_DIR, "rf_model_nlp.pkl")
    joblib.dump(rf, model_path)
    print(f"   ✅ Model saved to: {model_path}")


if __name__ == "__main__":
    train_nlp_model()