import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_items_dataset.csv")


def inspect_items():
    print(f"🕵️ INSPECTING: Items Dataset")
    df = pd.read_csv(FILE_PATH)

    print(f"ROWS: {len(df):,}")

    # 1. Price Check (Are there free items?)
    print(f"\n--- PRICE STATISTICS ---")
    print(df[['price', 'freight_value']].describe())

    # 2. Zero Price Check
    zero_price = df[df['price'] <= 0]
    print(f"\n--- SUSPICIOUS DATA ---")
    print(f"Items with Price <= 0: {len(zero_price)}")

    # 3. Order Multiplicity (How many items per order usually?)
    item_counts = df.groupby('order_id').size()
    print(f"\n--- BASKET SIZE ---")
    print(f"Max items in one order: {item_counts.max()}")
    print(f"Avg items per order: {item_counts.mean():.2f}")


if __name__ == "__main__":
    inspect_items()