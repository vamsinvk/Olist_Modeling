import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def plot_time_impact():
    print(f"🎨 PLOTTING: Delivery Time vs. Happiness")
    print("=" * 50)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)

    # 2. Filter Data
    # Remove missing dates and missing reviews (-1)
    df = df.dropna(subset=['actual_delivery_days', 'review_score'])
    df = df[df['review_score'] != -1]

    # Remove outliers for the chart (Orders taking > 60 days are rare and squash the chart)
    df_clean = df[df['actual_delivery_days'] <= 60]

    print(f"   • Data loaded: {len(df_clean):,} orders (filtered outliers > 60 days)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CHART 1: Boxplot (The Spread)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='review_score', y='actual_delivery_days', data=df_clean, palette='coolwarm')
    plt.title('Delivery Time vs. Review Score')
    plt.xlabel('Stars')
    plt.ylabel('Days to Deliver')
    plt.grid(True, alpha=0.3)

    file_1 = os.path.join(OUTPUT_DIR, "boxplot_time_vs_stars.png")
    plt.savefig(file_1)
    print(f"   ✅ Saved Chart 1: {file_1}")
    plt.close()

    # CHART 2: The "Patience Curve" (Line Chart)
    # We group by "Day" and see the average score
    daily_avg = df_clean.groupby(df_clean['actual_delivery_days'].astype(int))['review_score'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='actual_delivery_days', y='review_score', data=daily_avg, color='red', linewidth=2.5)
    plt.axhline(y=3, color='grey', linestyle='--', label='Neutral (3.0)')
    plt.title('The "Patience Curve": Avg Score by Delivery Speed')
    plt.xlabel('Days Passed')
    plt.ylabel('Average Star Rating')
    plt.legend()
    plt.grid(True)

    file_2 = os.path.join(OUTPUT_DIR, "line_patience_curve.png")
    plt.savefig(file_2)
    print(f"   ✅ Saved Chart 2: {file_2}")
    plt.close()


if __name__ == "__main__":
    plot_time_impact()