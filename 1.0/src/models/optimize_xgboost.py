import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "modeling")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def optimize_xgboost():
    print(f"🧪 LAB: Hyperparameter Optimization (Grid Search)")
    print("=" * 60)

    # 1. Load Data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

    # 2. Define the "Grid" of options to try
    # The computer will try every combination of these numbers
    param_grid = {
        'n_estimators': [100, 200],  # Trees
        'max_depth': [4, 6, 8],  # Depth (Complexity)
        'learning_rate': [0.05, 0.1],  # Speed
        'scale_pos_weight': [1, 3]  # Balance (1=Normal, 3=Aggressive)
    }

    # 3. Setup the Search
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

    # StratifiedKFold ensures we test fairly on the imbalance
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1',  # We want the best balance (F1)
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    # 4. Run the Experiment
    print("   • Testing 24 different model combinations...")
    grid_search.fit(X_train, y_train)

    # 5. The Winner
    print("\n🏆 OPTIMIZATION COMPLETE")
    print("-" * 30)
    print(f"   Best F1 Score: {grid_search.best_score_:.4f}")
    print(f"   Best Parameters: {grid_search.best_params_}")

    # 6. Save the Absolute Best Model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model_optimized.pkl"))
    print(f"\n✅ Saved Optimized Model to: models/best_model_optimized.pkl")


if __name__ == "__main__":
    optimize_xgboost()