import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def plot_money_impact():
    print(f"🎨 PLOTTING: Money vs. Happiness")
    print("=" * 50)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)

    # Filter: Remove missing reviews and negative prices (just in case)
    df = df[df['review_score'] != -1]
    df = df[df['price'] > 0]

    # 2. FEATURE ENGINEERING (On the fly for plotting)
    # Calculate "Freight Ratio" -> What % of the total cost is shipping?
    df['freight_ratio'] = df['freight_value'] / (df['price'] + df['freight_value'])

    print(f"   • Data loaded: {len(df):,} items.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CHART 1: Freight Ratio vs. Stars (Boxplot)
    # Do 1-star reviews have higher shipping %?
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='review_score', y='freight_ratio', data=df, palette='viridis')
    plt.title('Does High Shipping Cost Cause Bad Reviews?')
    plt.xlabel('Review Score')
    plt.ylabel('Freight Ratio (Shipping / Total Cost)')
    # Limit y-axis because some ratios are outliers (e.g., free item, 100% shipping)
    plt.ylim(0, 0.6)
    plt.grid(True, alpha=0.3)

    file_1 = os.path.join(OUTPUT_DIR, "boxplot_freight_ratio.png")
    plt.savefig(file_1)
    print(f"   ✅ Saved Chart 1: {file_1}")
    plt.close()

    # CHART 2: Correlation Heatmap
    # Let's check raw correlations between money numbers and the score
    cols_to_corr = ['review_score', 'price', 'freight_value', 'product_weight_g', 'total_payment_value']
    corr_matrix = df[cols_to_corr].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix: Money & Weight vs Score')

    file_2 = os.path.join(OUTPUT_DIR, "heatmap_money_corr.png")
    plt.savefig(file_2)
    print(f"   ✅ Saved Chart 2: {file_2}")
    plt.close()


if __name__ == "__main__":
    plot_money_impact()