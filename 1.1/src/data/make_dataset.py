import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_DIR = os.path.join(PROJECT_ROOT,"1.1", "data", "processed")
OUTPUT_BASIC = os.path.join(PROCESSED_DIR, "final_dataset_basic.csv")
OUTPUT_NLP = os.path.join(PROCESSED_DIR, "final_dataset_nlp.csv")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in KM between two lat/lon points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def make_dataset():
    print("🏗️ BUILD: Assembling Final Dataset...")

    # ==========================================
    # 1. LOAD ALL PROCESSED FILES
    # ==========================================
    print("   • Loading 8 processed files...")
    orders = pd.read_csv(os.path.join(PROCESSED_DIR, "orders_data.csv"))
    items = pd.read_csv(os.path.join(PROCESSED_DIR, "items_data.csv"))
    products = pd.read_csv(os.path.join(PROCESSED_DIR, "products_data.csv"))
    payments = pd.read_csv(os.path.join(PROCESSED_DIR, "payments_data.csv"))
    reviews = pd.read_csv(os.path.join(PROCESSED_DIR, "reviews_data.csv"), keep_default_na=False)  # Important for text
    customers = pd.read_csv(os.path.join(PROCESSED_DIR, "customers_data.csv"))
    sellers = pd.read_csv(os.path.join(PROCESSED_DIR, "sellers_data.csv"))
    geo = pd.read_csv(os.path.join(PROCESSED_DIR, "geolocation_data.csv"))
    marketing = pd.read_csv(os.path.join(PROCESSED_DIR, "marketing_data.csv"))

    # ==========================================
    # 2. AGGREGATE ITEMS (The "Flattening")
    # ==========================================
    # An order can have 5 items. We need 1 row per order.
    # We take the SUM of prices/freight, and the FIRST product/seller (Main Item logic)
    items_agg = items.groupby('order_id').agg({
        'product_id': 'first',
        'seller_id': 'first',
        'price': 'sum',
        'freight_value': 'sum',
        'is_high_freight': 'max'  # If any item has high freight, the order is high freight
    }).reset_index()

    # ==========================================
    # 3. THE GRAND JOIN (Left Joins to preserve Orders)
    # ==========================================
    print("   • Merging Tables...")

    # A. Orders + Reviews (Target Variable)
    # Note: We filter for delivered orders ONLY because we want to predict satisfaction, not cancellations.
    df = orders[orders['is_delivered'] == 1].copy()
    df = df.merge(reviews, on='order_id', how='left')

    # B. + Customers
    df = df.merge(customers, on='customer_id', how='left')

    # C. + Payments
    df = df.merge(payments, on='order_id', how='left')

    # D. + Items (Aggregated)
    df = df.merge(items_agg, on='order_id', how='left')

    # E. + Products (Via Product ID from Items)
    df = df.merge(products, on='product_id', how='left')

    # F. + Sellers (Via Seller ID from Items)
    df = df.merge(sellers, on='seller_id', how='left')

    # G. + Marketing (Via Seller ID)
    df = df.merge(marketing[['seller_id', 'seller_segment_group', 'origin']], on='seller_id', how='left')
    df['seller_segment_group'] = df['seller_segment_group'].fillna('Unknown')
    df['origin'] = df['origin'].fillna('Unknown')

    # ==========================================
    # 4. GEOSPATIAL ENGINE (The Physics)
    # ==========================================
    print("   • Calculating Geophysics...")

    # Merge Customer Coordinates
    df = df.merge(geo, left_on='customer_zip_code_prefix', right_on='zip_code_prefix', how='left')
    df.rename(columns={'lat': 'cust_lat', 'lng': 'cust_lng'}, inplace=True)
    df.drop(columns=['zip_code_prefix', 'geo_city', 'geo_state'], inplace=True)

    # Merge Seller Coordinates
    df = df.merge(geo, left_on='seller_zip_code_prefix', right_on='zip_code_prefix', how='left')
    df.rename(columns={'lat': 'sell_lat', 'lng': 'sell_lng'}, inplace=True)
    df.drop(columns=['zip_code_prefix', 'geo_city', 'geo_state'], inplace=True)

    # Calculate Haversine Distance
    df['distance_km'] = haversine_distance(df['cust_lat'], df['cust_lng'], df['sell_lat'], df['sell_lng'])

    # Fill NaN distances (rare edge cases) with 0 or median
    df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())

    # ==========================================
    # 5. SPLIT & SAVE
    # ==========================================

    # --- DATASET A: BASIC (No NLP) ---
    # We drop the raw text columns.
    cols_to_drop = ['review_comment_title', 'review_comment_message', 'customer_id', 'order_id',
                    'product_id', 'seller_id', 'review_id', 'review_creation_date', 'review_answer_timestamp',
                    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                    'order_delivered_customer_date', 'order_estimated_delivery_date']

    # Keep 'review_score' as the target!

    df_basic = df.drop(columns=cols_to_drop, errors='ignore')

    # One last cleanup: Drop rows where Target is missing (rare)
    df_basic = df_basic.dropna(subset=['review_score'])

    df_basic.to_csv(OUTPUT_BASIC, index=False)
    print(f"   ✅ SAVED BASIC: {OUTPUT_BASIC} ({len(df_basic):,} rows)")

    # --- DATASET B: NLP (Full) ---
    # We keep the text columns for NLP work
    # We only drop IDs that are truly useless
    nlp_cols_drop = ['customer_id', 'order_id', 'product_id', 'seller_id', 'review_id']
    df_nlp = df.drop(columns=nlp_cols_drop, errors='ignore')
    df_nlp = df_nlp.dropna(subset=['review_score'])

    df_nlp.to_csv(OUTPUT_NLP, index=False)
    print(f"   ✅ SAVED NLP:   {OUTPUT_NLP} ({len(df_nlp):,} rows)")


if __name__ == "__main__":
    make_dataset()