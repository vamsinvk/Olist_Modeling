import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_items_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "items_data.csv")


def process_items_granular():
    print(f"🔨 PROCESSING: Items Dataset (Granular Mode)")
    print("=" * 50)

    # 1. Load Raw Data
    try:
        df = pd.read_csv(RAW_FILE)
        print(f"   • Raw Rows: {len(df):,}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {RAW_FILE}")
        return

    # 2. SELECT COLUMNS (No Grouping!)
    # We want to keep every single item row so we know which product it is.
    cols_to_keep = [
        'order_id',
        'order_item_id',
        'product_id',  # <--- CRITICAL: This was likely missing before
        'seller_id',
        'shipping_limit_date',
        'price',
        'freight_value'
    ]

    df_silver = df[cols_to_keep].copy()

    # 3. DATE CONVERSION
    df_silver['shipping_limit_date'] = pd.to_datetime(df_silver['shipping_limit_date'])

    # 4. SAVE
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_silver.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Saved {len(df_silver):,} rows to {OUTPUT_FILE}")
    print(f"   • Columns: {list(df_silver.columns)}")
    print("   • You can now run the Master Merge script.")


if __name__ == "__main__":
    process_items_granular()