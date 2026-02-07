import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_final_showdown():
    print("📊 GENERATING: Final Battle - Non-NLP vs NLP...")

    # 1. THE HARD DATA (From your logs)
    data = {
        'Model': ['Non-NLP Model (v1.0)', 'NLP Model (v1.0)'],
        'F1-Score': [0.5436, 0.7184],  # Real numbers from your logs
        'Recall': [0.4137, 0.6459],  # Catch Rate
        'Precision': [0.7924, 0.8090],  # Trust Factor
        'Type': ['Operational Data Only', 'Review Text Data']
    }

    df = pd.DataFrame(data)

    # 2. SETUP PLOT
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar Widths
    bar_width = 0.35
    index = np.arange(len(df['Model']))

    # Plotting
    # Colors: Steel Blue for Non-NLP, Deep Orange for NLP
    rects1 = ax.bar(index, df['F1-Score'], bar_width, label='F1-Score (Balance)', color='#4682B4', alpha=0.9)
    rects2 = ax.bar(index + bar_width, df['Recall'], bar_width, label='Recall (Catch Rate)', color='#FF8C00', alpha=0.9)

    # 3. STYLING
    ax.set_xlabel('Model Approach', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('The Value of Language: Non-NLP vs NLP Performance', fontsize=14, pad=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df['Model'])
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 4. ADD LABELS (The Numbers)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=11)

    autolabel(rects1)
    autolabel(rects2)

    # 5. INSIGHT BOX
    improvement = ((0.7184 - 0.5436) / 0.5436) * 100
    plt.figtext(0.5, 0.01,
                f"CONCLUSION: Adding Text (NLP) improved performance by +{improvement:.1f}%.\n"
                "Without text, we miss 60% of angry customers (Recall 0.41).\n"
                "With text, we catch nearly 65% (Recall 0.65).",
                ha="center", fontsize=10, style='italic',
                bbox={"facecolor": "#f0f0f0", "edgecolor": "gray", "boxstyle": "round,pad=0.5"})

    # 6. SAVE
    save_path = os.path.join(OUTPUT_DIR, "final_impact_chart.png")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for text
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✅ SUCCESS: Chart saved to {save_path}")


if __name__ == "__main__":
    plot_final_showdown()