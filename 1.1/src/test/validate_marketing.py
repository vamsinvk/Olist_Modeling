import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "marketing_data.csv")


def validate():
    print("🧪 TEST: Validating Marketing Logic...")
    if not os.path.exists(PROCESSED_FILE): return print("❌ File not found.")

    df = pd.read_csv(PROCESSED_FILE)

    # TEST 1: Unique Sellers
    # We must have 1 row per seller_id to join safely later.
    if df['seller_id'].is_unique:
        print("   ✅ PASS: Unique Seller IDs.")
    else:
        print(f"   ❌ FAIL: Found duplicate seller_ids (Count: {df['seller_id'].duplicated().sum()}).")

    # TEST 2: Data Integrity
    # Check if we have valuable data or just empty rows
    unknowns = df[df['origin'] == 'Unknown']
    print(
        f"   ℹ️  Info: {len(unknowns)} sellers have 'Unknown' origin (Expected, as not all sellers are in marketing DB).")

    # TEST 3: Check Segment Logic
    if 'Home_Goods' in df['seller_segment_group'].unique():
        print("   ✅ PASS: Segment simplification worked.")
    else:
        print("   ❌ FAIL: Segment simplification failed.")


if __name__ == "__main__":
    validate()