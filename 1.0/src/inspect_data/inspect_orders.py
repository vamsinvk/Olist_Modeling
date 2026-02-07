import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_orders_dataset.csv")


def inspect_orders():
    print(f"🕵️ INSPECTING: {os.path.basename(FILE_PATH)}\n" + "=" * 50)

    # 1. Load Data
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {FILE_PATH}")
        return

    # 2. The Basics
    print(f"ROWS:    {df.shape[0]:,}")
    print(f"COLUMNS: {df.shape[1]}")

    # 3. Status Distribution (The most important business metric)
    print("\n--- 1. ORDER STATUS COUNTS ---")
    print(df['order_status'].value_counts())

    # 4. Integrity Check: Delivered but missing a date?
    # Logic: If status is 'delivered', 'order_delivered_customer_date' MUST exist.
    problem_orders = df[
        (df['order_status'] == 'delivered') &
        (df['order_delivered_customer_date'].isnull())
        ]
    print(f"\n--- 2. CRITICAL INTEGRITY CHECK ---")
    print(f"Orders marked 'delivered' but missing delivery date: {len(problem_orders)}")

    # 5. Missing Data Breakdown
    print("\n--- 3. MISSING VALUES PER COLUMN ---")
    print(df.isnull().sum())


if __name__ == "__main__":
    inspect_orders()