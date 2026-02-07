import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_payments_dataset.csv")


def inspect_payments():
    print(f"🕵️ INSPECTING: Payments Dataset")
    print("=" * 50)

    df = pd.read_csv(FILE_PATH)
    print(f"ROWS: {len(df):,}")

    # 1. Payment Methods
    print(f"\n--- TOP PAYMENT METHODS ---")
    print(df['payment_type'].value_counts())

    # 2. Installments (The "Debt" factor)
    print(f"\n--- INSTALLMENT BEHAVIOR ---")
    print(f"Max Installments found: {df['payment_installments'].max()}")
    print(f"Avg Installments: {df['payment_installments'].mean():.1f}")

    # Check High Installments (>12 months)
    high_debt = len(df[df['payment_installments'] > 12])
    print(f"Orders with >1 year of installments: {high_debt:,}")

    # 3. Value Check
    print(f"\n--- VALUE STATS (R$) ---")
    print(df['payment_value'].describe().round(2))


if __name__ == "__main__":
    inspect_payments()