import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_orders_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "orders_data.csv")


def process_orders():
    print(f"🔨 PROCESSING: Orders Dataset (Psychology & Temporal Logic) [FIXED]")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. Convert Dates
    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                 'order_estimated_delivery_date']

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. BASIC STATUS FLAGS
    # Logic Update: If it has a delivery date, we consider it delivered, regardless of status string
    df['is_delivered'] = np.where(df['order_delivered_customer_date'].notnull(), 1, 0)

    # 4. TEMPORAL FEATURES
    df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days

    # 5. EXPECTATION GAP
    # Formula: Estimated - Actual.
    df['expectation_gap'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days

    # 6. PROCESS DELAY (Seller Latency) - FIXED
    # Logic: Clip negative values to 0. (Warehouse cannot ship before it exists).
    df['seller_process_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days
    df['seller_process_days'] = df['seller_process_days'].clip(lower=0)

    # 7. WEEKEND ORDER
    df['is_weekend_order'] = df['order_purchase_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

    # 8. IS LATE - FIXED
    # Logic: Trust the DATE, not the Status. If Gap is Negative, it is Late.
    df['is_late'] = 0
    mask_late = (df['expectation_gap'] < 0)
    df.loc[mask_late, 'is_late'] = 1

    # 9. HUMAN READABLE STATUS
    df['delivery_performance'] = 'Processing/Other'

    # Hierarchy: Late > Early > On Time
    mask_early = (df['is_delivered'] == 1) & (df['expectation_gap'] > 0)
    df.loc[mask_early, 'delivery_performance'] = 'Early (Delight)'

    mask_ontime = (df['is_delivered'] == 1) & (df['expectation_gap'] == 0)
    df.loc[mask_ontime, 'delivery_performance'] = 'On Time'

    # Late overrides everything
    df.loc[mask_late, 'delivery_performance'] = 'Late (Risk)'

    # Handle "Ghosts" (Status delivered but no date) - rare edge case
    ghost_mask = (df['order_status'] == 'delivered') & (df['order_delivered_customer_date'].isnull())
    df.loc[ghost_mask, 'delivery_performance'] = 'Data Error (Missing Date)'

    # 10. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} orders.")
    print(f"   Late Rate: {df['is_late'].mean():.2%}")
    print(f"   Weekend Orders: {df['is_weekend_order'].mean():.2%}")


if __name__ == "__main__":
    process_orders()