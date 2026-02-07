import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")


def inspect_gold():
    print(f"📊 EDA STEP 1: Inspecting the Master Table")
    print("=" * 60)

    try:
        df = pd.read_csv(GOLD_FILE)
    except FileNotFoundError:
        print("❌ Error: master_table.csv not found.")
        return

    # 1. Quick Stats
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 2. Check for Missing Data (Crucial for ML)
    print(f"\n--- MISSING VALUES (Top 5) ---")
    missing = df.isnull().sum().sort_values(ascending=False).head(5)
    print(missing)

    # 3. Check the Target Variable (Review Score)
    print(f"\n--- TARGET VARIABLE: Review Scores ---")
    print(df['review_score'].value_counts().sort_index())

    # 4. Check the New Categories
    print(f"\n--- ISSUE CATEGORIES (Why people are mad) ---")
    print(df['issue_category'].value_counts())

    # 5. List all columns for our reference
    print(f"\n--- AVAILABLE FEATURES ---")
    print(list(df.columns))


if __name__ == "__main__":
    inspect_gold()