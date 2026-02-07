import joblib
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model_v1.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Features (Must match training order)
FEATURES = [
    'actual_delivery_days', 'is_late',
    'price', 'freight_value', 'total_payment_value', 'max_installments',
    'product_weight_g', 'product_length_cm',
    'customer_state'
]


def visualize_tree():
    print(f"🌲 VISUALIZING: One Decision Tree from the Forest")

    # 1. Load the "Pickled" Brain
    model = joblib.load(MODEL_PATH)

    # 2. Extract just ONE tree (The model has 100, we'll look at index 0)
    single_tree = model.estimators_[0]

    # 3. Plot it
    plt.figure(figsize=(20, 10))
    plot_tree(single_tree,
              feature_names=FEATURES,
              class_names=['Good', 'Bad'],
              filled=True,
              rounded=True,
              max_depth=3,  # Only show top 3 levels so it's readable
              fontsize=10)

    plt.title("How the Model Thinks (Top 3 Decisions)")

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, "tree_visualization.png")
    plt.savefig(file_path)
    print(f"✅ Saved visualization to: {file_path}")


if __name__ == "__main__":
    visualize_tree()