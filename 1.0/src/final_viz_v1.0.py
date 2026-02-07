import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_final_result():
    print("📊 GENERATING: Final v1.0 Impact Chart...")

    # 1. THE EXACT DATA FROM YOUR LOGS
    data = {
        'Model': ['Baseline (Non-NLP)', 'NLP Model (Text)'],
        'F1-Score': [0.5436, 0.7184],
        'Recall': [0.4137, 0.6459],
        'Precision': [0.7924, 0.8090]
    }

    df = pd.DataFrame(data)

    # 2. PLOT SETUP
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(df['Model']))

    # Colors: Slate Blue (Math) vs. Deep Orange (Text)
    rects1 = ax.bar(index, df['F1-Score'], bar_width, label='F1-Score (Overall Quality)', color='#5D6D7E')
    rects2 = ax.bar(index + bar_width, df['Recall'], bar_width, label='Recall (Catch Rate)', color='#E67E22')

    # 3. LABELS & STYLING
    ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of NLP: Reading Reviews vs. Metadata Only', fontsize=14, pad=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df['Model'])
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 4. ADD VALUES ON BARS
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    # 5. INSIGHT BOX
    plt.figtext(0.5, 0.02,
                "KEY INSIGHT: The NLP Model catches 65% of churn risks.\n"
                "The Non-NLP model only catches 41%.\n"
                "Text analysis is critical for identifying defects.",
                ha="center", fontsize=10,
                bbox={"facecolor": "#f9f9f9", "edgecolor": "gray", "boxstyle": "round,pad=0.5"})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    save_path = os.path.join(OUTPUT_DIR, "v1.0_final_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Chart saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_final_result()