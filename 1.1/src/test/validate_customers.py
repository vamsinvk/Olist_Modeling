import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "customers_data.csv")


def validate():
    print("🧪 TEST: Validating Customers Logic...")

    if not os.path.exists(PROCESSED_FILE):
        print("❌ ERROR: Processed file not found. Run process_customers.py first.")
        return

    df = pd.read_csv(PROCESSED_FILE)

    # TEST 1: Check Region Integrity
    # We shouldn't have any 'Other' unless the raw data had a fake state.
    # There are 27 states in Brazil.
    unknowns = df[df['customer_region'] == 'Other']

    if len(unknowns) == 0:
        print("   ✅ PASS: All customer states mapped to valid Macro-Regions.")
    else:
        print(f"   ❌ FAIL: Found {len(unknowns)} customers with unknown states/regions.")
        print(f"          States: {unknowns['customer_state'].unique()}")

    # TEST 2: Check Hub Logic
    # 'is_hub_customer' should be 1 ONLY if state is SP
    # We check for contradictions
    errors = df[((df['customer_state'] == 'SP') & (df['is_hub_customer'] == 0)) |
                ((df['customer_state'] != 'SP') & (df['is_hub_customer'] == 1))]

    if len(errors) == 0:
        print("   ✅ PASS: Hub Logic (SP = 1) is perfect.")
    else:
        print(f"   ❌ FAIL: Hub logic mismatch in {len(errors)} rows.")

    # TEST 3: Check Unique IDs
    # Olist has 'customer_id' (per order) and 'customer_unique_id' (per person).
    # Unique ID count should be <= Customer ID count.
    n_orders = df['customer_id'].nunique()
    n_humans = df['customer_unique_id'].nunique()

    print("-" * 30)
    print("📊 SAMPLE INSIGHTS:")
    print(f"   Total Orders (IDs): {n_orders:,}")
    print(f"   Unique Humans: {n_humans:,}")
    print(f"   Repeat Ratio: {n_orders / n_humans:.2f} orders per human (avg)")


if __name__ == "__main__":
    validate()