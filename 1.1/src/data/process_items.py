import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
# "1.1" logic implies we keep using the original raw data source
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_items_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "items_data.csv")

def process_items():
    print(f"🔨 PROCESSING: Items Dataset (Financial Logic)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. FEATURE: Total Item Cost
    # The actual amount leaving the customer's wallet for this specific item
    df['total_item_value'] = df['price'] + df['freight_value']

    # 3. FEATURE: Freight Ratio (The "Rip-off" Factor)
    # Formula: Freight / (Price + Freight).
    # Result: 0.10 means 10% of cost is shipping. 0.50 means 50% is shipping.
    # Logic: High ratio (>0.3) usually triggers "Buyer's Remorse" if delivery is even 1 day late.
    df['freight_ratio'] = df['freight_value'] / df['total_item_value']

    # 4. FEATURE: Is High Freight? (Binary Risk Signal)
    # We flag items where shipping is more than 40% of the total cost.
    df['is_high_freight'] = (df['freight_ratio'] > 0.4).astype(int)

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} items.")
    print(f"   Avg Freight Ratio: {df['freight_ratio'].mean():.2%}")
    print(f"   High Freight Items: {df['is_high_freight'].sum():,} ({df['is_high_freight'].mean():.1%})")

if __name__ == "__main__":
    process_items()