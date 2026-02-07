import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def plot_geography():
    print(f"🎨 PLOTTING: Geography vs. Happiness")
    print("=" * 50)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    df = df[df['review_score'] != -1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CHART 1: The "State of Hate" (Bar Chart)
    # Calculate average review score per state
    state_avg = df.groupby('customer_state')['review_score'].mean().sort_values()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=state_avg.index, y=state_avg.values, palette='RdYlGn')
    plt.axhline(y=df['review_score'].mean(), color='black', linestyle='--', label='National Avg')
    plt.title('Average Review Score by State (Worst to Best)')
    plt.ylim(3, 5)  # Zoom in to see differences
    plt.legend()

    file_1 = os.path.join(OUTPUT_DIR, "bar_state_scores.png")
    plt.savefig(file_1)
    print(f"   ✅ Saved Chart 1: {file_1}")
    plt.close()

    # CHART 2: The Customer Map (Scatter)
    # We use Lat/Lng to plot the shape of Brazil
    # We color code by Review Score (Red=1, Green=5)

    # Filter for valid lat/lng
    geo_df = df.dropna(subset=['geolocation_lat', 'geolocation_lng'])
    # Downsample for speed (plot 10k points instead of 100k)
    geo_sample = geo_df.sample(n=min(10000, len(geo_df)), random_state=42)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x='geolocation_lng', y='geolocation_lat',
        hue='review_score', palette='RdYlGn',
        data=geo_sample, alpha=0.6, s=15
    )
    plt.title('Customer Map: Where are the angry people?')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    file_2 = os.path.join(OUTPUT_DIR, "scatter_map_brazil.png")
    plt.savefig(file_2)
    print(f"   ✅ Saved Chart 2: {file_2}")
    plt.close()


if __name__ == "__main__":
    plot_geography()