import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT, "1.1","data", "processed", "items_data.csv")


def validate():
    print("🧪 TEST: Validating Items Logic...")

    if not os.path.exists(PROCESSED_FILE):
        print("❌ ERROR: Processed file not found. Run process_items.py first.")
        return

    df = pd.read_csv(PROCESSED_FILE)

    # TEST 1: Check Total Value Logic
    # Price + Freight MUST equal Total
    # We use a tiny epsilon for float comparison safety
    diff = (df['price'] + df['freight_value']) - df['total_item_value']
    errors = diff[diff.abs() > 0.01]

    if len(errors) == 0:
        print("   ✅ PASS: Total Item Value math is perfect.")
    else:
        print(f"   ❌ FAIL: Found {len(errors)} math errors in total_item_value.")

    # TEST 2: Check Freight Ratio Bounds
    # Ratio should be between 0 and 1 (inclusive)
    # Note: If price is 0 (gift/promo), ratio could be 1.0.
    out_of_bounds = df[(df['freight_ratio'] < 0) | (df['freight_ratio'] > 1.0)]

    if len(out_of_bounds) == 0:
        print("   ✅ PASS: Freight Ratios are strictly between 0.0 and 1.0.")
    else:
        print(f"   ❌ FAIL: Found {len(out_of_bounds)} Impossible Freight Ratios.")

    # TEST 3: Check for Negative Prices
    negatives = df[(df['price'] < 0) | (df['freight_value'] < 0)]
    if len(negatives) == 0:
        print("   ✅ PASS: No negative prices found.")
    else:
        print(f"   ❌ FAIL: Found {len(negatives)} negative prices.")

    print("-" * 30)
    print("📊 SAMPLE INSIGHTS:")
    print(f"   Avg Item Price: R$ {df['price'].mean():.2f}")
    print(f"   Avg Shipping: R$ {df['freight_value'].mean():.2f}")


if __name__ == "__main__":
    validate()