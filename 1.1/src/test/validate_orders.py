import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1" "data", "processed", "orders_data.csv")


def validate():
    print("🧪 TEST: Validating Orders Logic...")

    if not os.path.exists(PROCESSED_FILE):
        print("❌ ERROR: Processed file not found. Run process_orders.py first.")
        return

    df = pd.read_csv(PROCESSED_FILE)

    # TEST 1: Check Expectation Gap Logic
    # If gap is negative, is_late MUST be 1
    late_check = df[df['expectation_gap'] < 0]
    errors = late_check[late_check['is_late'] == 0]

    if len(errors) == 0:
        print("   ✅ PASS: Negative Expectation Gap correctly flagged as Late.")
    else:
        print(f"   ❌ FAIL: Found {len(errors)} rows where Gap is Negative but is_late is 0.")

    # TEST 2: Check Seller Process Days
    # It shouldn't be negative (Seller can't ship before approval... usually)
    # Note: In real data, there might be tiny errors, but we want to see if it's huge.
    negative_process = df[df['seller_process_days'] < 0]
    if len(negative_process) < 100:  # Allowing small noise
        print(f"   ✅ PASS: Seller Process logic is sound (Only {len(negative_process)} anomalies).")
    else:
        print(f"   ⚠️ WARN: Found {len(negative_process)} orders shipped before approval. Check data.")

    # TEST 3: Weekend Check
    # Verify values are only 0 or 1
    unique_vals = df['is_weekend_order'].unique()
    if set(unique_vals).issubset({0, 1}):
        print("   ✅ PASS: Weekend flag is strictly binary.")
    else:
        print(f"   ❌ FAIL: Weekend flag has weird values: {unique_vals}")

    print("-" * 30)
    print("📊 SAMPLE INSIGHTS:")
    print(f"   Avg Process Delay: {df['seller_process_days'].median():.1f} days")
    print(f"   Avg Expectation Gap: {df['expectation_gap'].median():.1f} days (Positive = Early)")


if __name__ == "__main__":
    validate()