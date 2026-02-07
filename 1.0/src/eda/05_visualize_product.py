import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def plot_product_impact():
    print(f"🎨 PLOTTING: Product Weight vs. Happiness")
    print("=" * 50)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    df = df[df['review_score'] != -1]
    df = df[df['product_weight_g'] > 0]  # Remove bad data

    # 2. FEATURE ENGINEERING: Weight Classes
    # Convert grams to kg for readability
    df['weight_kg'] = df['product_weight_g'] / 1000

    # Define classes: Light (<2kg), Medium (2-10kg), Heavy (>10kg)
    conditions = [
        (df['weight_kg'] < 2),
        (df['weight_kg'] >= 2) & (df['weight_kg'] < 10),
        (df['weight_kg'] >= 10)
    ]
    choices = ['Light (<2kg)', 'Medium (2-10kg)', 'Heavy (>10kg)']

    # FIX: Added default='Unknown' to prevent Type Error
    df['weight_class'] = np.select(conditions, choices, default='Unknown')

    print(f"   • Data loaded: {len(df):,} items.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CHART 1: Weight Class vs. Review Score (Bar Chart)
    plt.figure(figsize=(10, 6))
    # We filter out 'Unknown' for the plot order
    order = ['Light (<2kg)', 'Medium (2-10kg)', 'Heavy (>10kg)']
    sns.barplot(x='weight_class', y='review_score', data=df[df['weight_class'] != 'Unknown'], order=order,
                palette='magma')
    plt.ylim(3.5, 4.5)  # Zoom in to see the difference
    plt.title('Do Heavy Items Get Worse Reviews?')
    plt.ylabel('Average Star Rating')

    file_1 = os.path.join(OUTPUT_DIR, "bar_weight_class.png")
    plt.savefig(file_1)
    print(f"   ✅ Saved Chart 1: {file_1}")
    plt.close()

    # CHART 2: The "Broken" Correlation (Scatter/Reg plot)
    # Does weight correlate with "Product Quality" complaints?
    quality_issues = df[df['issue_category'] == 'Product Quality']

    plt.figure(figsize=(10, 6))
    sns.histplot(data=quality_issues, x='weight_kg', bins=30, kde=True, color='red')
    plt.title('Distribution of Broken Items by Weight')
    plt.xlabel('Weight (kg)')
    plt.xlim(0, 30)  # Focus on common items

    file_2 = os.path.join(OUTPUT_DIR, "hist_broken_weight.png")
    plt.savefig(file_2)
    print(f"   ✅ Saved Chart 2: {file_2}")
    plt.close()


if __name__ == "__main__":
    plot_product_impact()