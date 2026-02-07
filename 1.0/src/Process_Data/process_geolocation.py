import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_geolocation_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "geolocation_data.csv")


def process_geolocation():
    print(f"🔨 PROCESSING: Geolocation Dataset")
    print("=" * 50)

    # Load only necessary columns to save memory
    df = pd.read_csv(RAW_FILE, usecols=['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng'])

    # Group by Zip and take Median
    geo_data = df.groupby('geolocation_zip_code_prefix').median().reset_index()

    # Rename to match Customers file
    geo_data.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix'}, inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    geo_data.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Compressed to {len(geo_data):,} unique locations.")


if __name__ == "__main__":
    process_geolocation()