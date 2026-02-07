import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_customers_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "customers_data.csv")


def process_customers():
    print(f"🔨 PROCESSING: Customers Dataset")
    print("=" * 50)

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. FEATURE: Calculate Order Count per Person
    # We count how many times 'customer_unique_id' appears
    print("   • Calculating Repeat Buyers...")
    counts = df['customer_unique_id'].value_counts().reset_index()
    counts.columns = ['customer_unique_id', 'order_count']

    # Merge it back to the main dataframe
    df = df.merge(counts, on='customer_unique_id', how='left')

    # 3. SEGMENTATION: Create a simple flag
    # "New" = 1 order, "Repeat" = >1 order
    df['customer_type'] = df['order_count'].apply(lambda x: 'Repeat' if x > 1 else 'New')

    # 4. SAVE
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Saved {len(df):,} customers to {OUTPUT_FILE}")
    print(f"   • Repeat Customers Identified: {len(df[df['order_count'] > 1]):,}")
    print(f"   • Breakdown:\n{df['customer_type'].value_counts()}")


if __name__ == "__main__":
    process_customers()