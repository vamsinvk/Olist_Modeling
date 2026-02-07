import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_payments_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "payments_data.csv")


def process_payments():
    print(f"🔨 PROCESSING: Payments Dataset (Financial Anxiety Logic)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. AGGREGATE PER ORDER
    # The raw file has one row per *transaction* (e.g., part credit card, part voucher).
    # We need one row per *order_id*.

    # Logic:
    # - Sum of installments (max might be better, but sum captures total 'swipes')
    # - Sum of payment value
    # - Count of payment methods used (Sequential)

    pay_agg = df.groupby('order_id').agg({
        'payment_installments': 'max',  # Max installments represents the timeline commitment
        'payment_value': 'sum',  # Total amount paid
        'payment_sequential': 'max',  # If this is 3, they tried/used 3 methods
        'payment_type': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]  # Main method
    }).reset_index()

    # 3. FEATURE: FINANCIAL STRAIN
    # Formula: Installments / Total Value.
    # WAIT: That doesn't make sense. A $1000 item in 10 installments is $100/mo.
    # Better Metric: "Installment Count" itself is the best proxy for "Can't pay cash now."
    # We keep 'payment_installments' as is.

    # 4. FEATURE: AVERAGE MONTHLY BURDEN
    # How much are they paying per month?
    # Logic: If installments = 0 or 1, burden is the full price.
    pay_agg['avg_monthly_payment'] = np.where(
        pay_agg['payment_installments'] <= 1,
        pay_agg['payment_value'],
        pay_agg['payment_value'] / pay_agg['payment_installments']
    )

    # 5. FEATURE: IS HIGH COMPLEXITY?
    # Did they use multiple cards or split payment?
    pay_agg['is_complex_payment'] = (pay_agg['payment_sequential'] > 1).astype(int)

    # 6. FEATURE: IS VOUCHER (Boleto)?
    # Boleto causes delays because bank clearance takes 2-3 days.
    pay_agg['is_boleto'] = (pay_agg['payment_type'] == 'boleto').astype(int)

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pay_agg.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed payments for {len(pay_agg):,} orders.")
    print(f"   High Installment Orders (>10): {len(pay_agg[pay_agg['payment_installments'] > 10]):,}")
    print(f"   Boleto (Slow) Payments: {pay_agg['is_boleto'].mean():.1%}")


if __name__ == "__main__":
    process_payments()