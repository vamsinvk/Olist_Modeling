import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_orders_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "orders_data.csv")


def process_orders():
    print(f"🔨 PROCESSING: Orders Dataset (Success vs Failure Logic)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. Convert Dates
    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                 'order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. FEATURE: IS DELIVERED (The User Request)
    # Logic: 1 if the customer actually got it (even if late). 0 if canceled/unavailable/processing.
    df['is_delivered'] = np.where(df['order_status'] == 'delivered', 1, 0)

    # 4. FEATURE: IS LATE (Speed Metric)
    # Logic: 1 if Delivered AND Late. 0 if On Time OR Not Delivered.
    df['is_late'] = 0
    mask_late = (df['order_status'] == 'delivered') & (
                df['order_delivered_customer_date'] > df['order_estimated_delivery_date'])
    df.loc[mask_late, 'is_late'] = 1

    # 5. FEATURE: Days to Deliver
    df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    # 6. FEATURE: Human Readable Status
    df['delivery_performance'] = df['order_status']

    # Update labels based on your rules
    df.loc[mask_late, 'delivery_performance'] = 'Delayed'  # Late but delivered

    mask_ontime = (df['order_status'] == 'delivered') & (
                df['order_delivered_customer_date'] <= df['order_estimated_delivery_date'])
    df.loc[mask_ontime, 'delivery_performance'] = 'On Time'

    # Identify Ghosts
    ghost_mask = (df['order_status'] == 'delivered') & (df['order_delivered_customer_date'].isnull())
    df.loc[ghost_mask, 'delivery_performance'] = 'Delivery Error (Missing Date)'

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} rows.")
    print(f"   Success Rate (is_delivered=1): {df['is_delivered'].mean():.2%}")
    print(f"   Breakdown:\n{df['delivery_performance'].value_counts().head()}")


if __name__ == "__main__":
    process_orders()