import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_customers_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "customers_data.csv")


def process_customers():
    print(f"🔨 PROCESSING: Customers Dataset (Regional Logic)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. FEATURE: MACRO-REGIONS
    # Grouping 27 states into 5 economic zones.
    # This helps the model generalize (e.g., "North is hard to ship to")
    # instead of memorizing specific states.
    regions = {
        'SE': ['SP', 'RJ', 'ES', 'MG'],  # Southeast (Hub)
        'S': ['PR', 'SC', 'RS'],  # South (Wealthy)
        'NE': ['BA', 'PE', 'CE', 'RN', 'PB', 'MA', 'AL', 'SE', 'PI'],  # Northeast
        'CO': ['DF', 'GO', 'MT', 'MS'],  # Center-West
        'N': ['AM', 'PA', 'RO', 'TO', 'AC', 'AP', 'RR']  # North (Amazon/Remote)
    }

    # Invert the dictionary for mapping
    state_to_region = {state: region for region, states in regions.items() for state in states}

    df['customer_region'] = df['customer_state'].map(state_to_region).fillna('Other')

    # 3. FEATURE: IS HUB?
    # Logic: SP (São Paulo) is the main logistics hub of Brazil.
    # Customers in SP usually get 1-day delivery. Everyone else waits.
    df['is_hub_customer'] = (df['customer_state'] == 'SP').astype(int)

    # 4. CLEANING: Standardization
    # Title case for cities to make them look professional
    df['customer_city'] = df['customer_city'].str.title()

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} customers.")
    print(f"   Region Breakdown:\n{df['customer_region'].value_counts().head()}")
    print(f"   Hub Customers (SP): {df['is_hub_customer'].mean():.1%}")


if __name__ == "__main__":
    process_customers()