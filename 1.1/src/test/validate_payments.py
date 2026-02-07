import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "payments_data.csv")


def validate():
    print("🧪 TEST: Validating Payments Logic...")

    if not os.path.exists(PROCESSED_FILE):
        print("❌ ERROR: Processed file not found. Run process_payments.py first.")
        return

    df = pd.read_csv(PROCESSED_FILE)

    # TEST 1: Check Uniqueness
    # We must have exactly 1 row per order_id now.
    if df['order_id'].is_unique:
        print("   ✅ PASS: Aggregation successful (One row per order).")
    else:
        print(f"   ❌ FAIL: Duplicate order_ids found. Aggregation failed.")

    # TEST 2: Check Monthly Burden Math
    # If installments > 1, Monthly * Installments should approx equal Value
    sample = df[df['payment_installments'] > 1].iloc[0]
    calc_total = sample['avg_monthly_payment'] * sample['payment_installments']

    if abs(calc_total - sample['payment_value']) < 1.0:  # allow rounding noise
        print("   ✅ PASS: Monthly payment calculation is mathematically sound.")
    else:
        print(f"   ❌ FAIL: Math mismatch. {calc_total:.2f} vs {sample['payment_value']:.2f}")

    # TEST 3: Boleto Logic
    # If payment_type is 'boleto', is_boleto must be 1
    boleto_errors = df[(df['payment_type'] == 'boleto') & (df['is_boleto'] == 0)]
    if len(boleto_errors) == 0:
        print("   ✅ PASS: Boleto flag logic is correct.")
    else:
        print(f"   ❌ FAIL: Found {len(boleto_errors)} boleto flagging errors.")

    print("-" * 30)
    print("📊 SAMPLE INSIGHTS:")
    print(f"   Avg Installments: {df['payment_installments'].mean():.1f}")
    print(f"   Complex Payments (Split): {df['is_complex_payment'].mean():.1%}")


if __name__ == "__main__":
    validate()