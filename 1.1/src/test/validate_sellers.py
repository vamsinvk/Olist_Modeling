import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "sellers_data.csv")


def validate():
    print("🧪 TEST: Validating Sellers...")
    if not os.path.exists(PROCESSED_FILE): return print("❌ File not found.")

    df = pd.read_csv(PROCESSED_FILE)

    # Check ID uniqueness
    if df['seller_id'].is_unique:
        print("   ✅ PASS: Unique Seller IDs.")
    else:
        print("   ❌ FAIL: Duplicate Sellers found.")

    # Check Region
    if 'Other' not in df['seller_region'].unique():
        print("   ✅ PASS: All sellers mapped to valid regions.")
    else:
        print("   ⚠️ WARN: Some sellers have unknown regions.")


if __name__ == "__main__":
    validate()