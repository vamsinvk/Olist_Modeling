import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_customers_dataset.csv")


def inspect_customers():
    print(f"🕵️ INSPECTING: Customers Dataset")
    print("=" * 50)

    df = pd.read_csv(FILE_PATH)
    print(f"ROWS: {len(df):,}")

    # 1. Unique vs Transactional
    # customer_id = ID for ONE specific order
    # customer_unique_id = ID for the PERSON
    print(f"\n--- ID ANALYSIS ---")
    print(f"Total Transaction IDs (customer_id): {df['customer_id'].nunique():,}")
    print(f"Total Unique People (customer_unique_id): {df['customer_unique_id'].nunique():,}")

    # 2. Where are they from?
    print(f"\n--- TOP LOCATIONS ---")
    print(df['customer_state'].value_counts().head(5))

    # 3. Completeness
    print(f"\n--- MISSING VALUES ---")
    print(df.isnull().sum())


if __name__ == "__main__":
    inspect_customers()