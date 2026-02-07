import pandas as pd
import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "modeling")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def compare_models():
    print(f"⚔️ THE ARENA: Model Comparison")
    print("=" * 60)

    # 1. Load Data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # 2. Define the Contenders
    # We use a dictionary so we can loop through them
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = []

    # 3. Fight!
    for name, model in models.items():
        print(f"   • Training {name}...")
        start_time = time.time()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        elapsed = time.time() - start_time

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Time (s)": round(elapsed, 2)
        })

    # 4. Create Leaderboard
    df_results = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

    print("\n🏆 LEADERBOARD (Ranked by F1-Score)")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # 5. Save the Winner
    winner_name = df_results.iloc[0]['Model']
    winner_model = models[winner_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(winner_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    print(f"\n✅ The Winner is [{winner_name}]!")
    print(f"   Saved to: models/best_model.pkl")


if __name__ == "__main__":
    compare_models()