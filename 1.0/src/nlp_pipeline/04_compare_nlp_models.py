import pandas as pd
import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "modeling", "nlp_ready_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "nlp")


def compare_nlp_models():
    print(f"⚔️ THE NLP ARENA: Comparing Models on Sentiment Data")
    print("=" * 60)

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: nlp_ready_data.csv not found.")
        return

    # 2. Preprocessing (Same as before)
    # Create Target
    df['is_bad_review'] = df['review_score'].apply(lambda x: 1 if x <= 3 else 0)
    X = df.drop(columns=['review_score', 'is_bad_review'])
    y = df['is_bad_review']

    # Encode State
    le = LabelEncoder()
    X['customer_state'] = le.fit_transform(X['customer_state'].astype(str))
    X = X.fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define Contenders
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = []

    # 4. Fight!
    for name, model in models.items():
        print(f"   • Training {name}...")
        start_time = time.time()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Score
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        elapsed = time.time() - start_time

        results.append({
            "Model": name,
            "F1-Score": f1,
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "Time (s)": round(elapsed, 2)
        })

    # 5. Leaderboard
    df_results = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

    print("\n🏆 NLP LEADERBOARD (Ranked by F1-Score)")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # 6. Save the Winner
    winner_name = df_results.iloc[0]['Model']
    winner_model = models[winner_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(winner_model, os.path.join(MODEL_DIR, "best_nlp_model.pkl"))
    print(f"\n✅ The Winner is [{winner_name}]!")
    print(f"   Saved to: models/nlp/best_nlp_model.pkl")


if __name__ == "__main__":
    compare_nlp_models()