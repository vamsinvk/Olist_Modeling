import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_sellers_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "sellers_data.csv")


def process_sellers():
    print(f"🔨 PROCESSING: Sellers Dataset (Regional Logic)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. FEATURE: MACRO-REGIONS
    # We apply the same economic zone logic as Customers.
    regions = {
        'SE': ['SP', 'RJ', 'ES', 'MG'],  # Southeast (Hub)
        'S': ['PR', 'SC', 'RS'],  # South
        'NE': ['BA', 'PE', 'CE', 'RN', 'PB', 'MA', 'AL', 'SE', 'PI'],  # Northeast
        'CO': ['DF', 'GO', 'MT', 'MS'],  # Center-West
        'N': ['AM', 'PA', 'RO', 'TO', 'AC', 'AP', 'RR']  # North
    }
    state_to_region = {state: region for region, states in regions.items() for state in states}

    df['seller_region'] = df['seller_state'].map(state_to_region).fillna('Other')

    # 3. FEATURE: IS HUB SELLER?
    # Logic: Sellers in SP have access to better logistics infrastructure.
    df['is_hub_seller'] = (df['seller_state'] == 'SP').astype(int)

    # 4. CLEANING
    df['seller_city'] = df['seller_city'].str.title()

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} sellers.")
    print(f"   Hub Sellers (SP): {df['is_hub_seller'].mean():.1%}")


if __name__ == "__main__":
    process_sellers()