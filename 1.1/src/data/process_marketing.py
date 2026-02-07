import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
MQL_FILE = os.path.join(PROJECT_ROOT, "olist", "marketing", "archive", "olist_marketing_qualified_leads_dataset.csv")
DEALS_FILE = os.path.join(PROJECT_ROOT, "olist", "marketing", "archive", "olist_closed_deals_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "1.1","data", "processed", "marketing_data.csv")


def process_marketing():
    print(f"🔨 PROCESSING: Marketing Data (Seller DNA)")

    # 1. Load Data
    mql = pd.read_csv(MQL_FILE)
    deals = pd.read_csv(DEALS_FILE)

    # 2. MERGE MQL + DEALS
    # We want to attach the "Origin" info from MQL to the "Seller ID" in Deals.
    # Join key: 'mql_id'
    merged = deals.merge(mql, on='mql_id', how='left')

    # 3. SELECT & RENAME COLUMNS
    # We only care about the Seller ID and the new features.
    # Note: Not all sellers are in this file! Only the ones tracked by marketing.
    # This will be a subset of total sellers.

    marketing_features = merged[['seller_id', 'business_segment', 'lead_type', 'origin']]

    # 4. CLEANING / IMPUTATION
    # Fill NaNs with 'Unknown' so the model treats them as a specific category
    marketing_features = marketing_features.fillna('Unknown')

    # 5. FEATURE ENGINEERING: BUSINESS TYPE
    # Simplify segments. 'home_decor', 'household_utilities' -> 'Home/House'
    # This reduces noise (High Cardinality).

    def simplify_segment(seg):
        seg = str(seg).lower()
        if 'home' in seg or 'house' in seg or 'decor' in seg: return 'Home_Goods'
        if 'health' in seg or 'beauty' in seg: return 'Health_Beauty'
        if 'tech' in seg or 'computer' in seg or 'phone' in seg: return 'Tech'
        if 'car' in seg or 'auto' in seg: return 'Auto'
        return 'Other'

    marketing_features['seller_segment_group'] = marketing_features['business_segment'].apply(simplify_segment)

    # 6. FEATURE: IS MANUFACTURER?
    # Manufacturers usually have better inventory control than resellers.
    # We look for "manufacturer" in the lead type or segment (if exists)
    # The 'lead_type' column often has 'industry' or 'online_medium'.
    # We'll create a flag for 'industry' which implies making things.
    marketing_features['is_manufacturer'] = marketing_features['lead_type'].apply(
        lambda x: 1 if 'industry' in str(x).lower() else 0
    )

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    marketing_features.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed marketing info for {len(marketing_features):,} sellers.")
    print(f"   Segments:\n{marketing_features['seller_segment_group'].value_counts().head()}")


if __name__ == "__main__":
    process_marketing()