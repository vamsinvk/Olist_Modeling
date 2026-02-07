import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_reviews_dataset.csv")


def inspect_review_breakdown():
    print(f"🕵️ INSPECTING: Review Comments by Star Rating")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(FILE_PATH)

    # 2. Group by Score and Count Nulls
    # We create a custom summary for each score (1-5)
    breakdown = df.groupby('review_score')['review_comment_message'].apply(lambda x: pd.Series({
        'Total Reviews': len(x),
        'With Comment': x.notnull().sum(),
        'No Comment (Null)': x.isnull().sum(),
        '% Missing Text': (x.isnull().sum() / len(x)) * 100
    })).unstack()

    # Formatting for cleaner output
    pd.options.display.float_format = '{:.1f}'.format
    print(breakdown)

    # 3. Insight Generator
    print("\n--- INSIGHTS ---")
    worst_score_missing = breakdown['% Missing Text'].idxmax()
    best_score_missing = breakdown['% Missing Text'].idxmin()

    print(f"• The 'Laziest' Customers give {worst_score_missing} Stars (Highest % missing text).")
    print(f"• The 'Most Vocal' Customers give {best_score_missing} Stars (Lowest % missing text).")


if __name__ == "__main__":
    inspect_review_breakdown()