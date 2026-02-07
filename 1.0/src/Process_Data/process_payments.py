import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_payments_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "payments_data.csv")


def process_payments():
    print(f"🔨 PROCESSING: Payments Dataset")
    print("=" * 50)

    df = pd.read_csv(RAW_FILE)

    # Aggregation: 1 Row per Order
    agg_df = df.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'max',  # Did they take long-term debt?
        'payment_sequential': 'count',  # How many cards did they use?
        'payment_type': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
    }).reset_index()

    # Rename
    agg_df.rename(columns={
        'payment_value': 'total_payment_value',
        'payment_installments': 'max_installments',
        'payment_sequential': 'payment_method_count',
        'payment_type': 'primary_payment_method'
    }, inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    agg_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Saved {len(agg_df):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_payments()