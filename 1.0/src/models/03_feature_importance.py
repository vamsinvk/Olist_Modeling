import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model_v1.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Feature names must match the Training Columns exactly
FEATURES = [
    'actual_delivery_days', 'is_late',
    'price', 'freight_value', 'total_payment_value', 'max_installments',
    'product_weight_g', 'product_length_cm',
    'customer_state'
]


def plot_feature_importance():
    print(f"🧠 ML STEP 3: Visualizing the Model's Brain")
    print("=" * 60)

    # 1. Load Model
    try:
        rf = joblib.load(MODEL_PATH)
        print("   ✅ Model loaded successfully.")
    except FileNotFoundError:
        print("   ❌ Model not found. Run training script first.")
        return

    # 2. Extract Importances
    importances = rf.feature_importances_

    # 3. Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n   --- TOP PREDICTORS ---")
    print(df_imp.head(10))

    # 4. Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title('What drives a Bad Review? (Feature Importance)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)

    # 5. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, "feature_importance_rf.png")
    plt.savefig(file_path)
    print(f"\n   ✅ Chart saved to: {file_path}")


if __name__ == "__main__":
    plot_feature_importance()