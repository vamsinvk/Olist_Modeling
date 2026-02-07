import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_geolocation_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "geolocation_data.csv")


def process_geolocation():
    print(f"🔨 PROCESSING: Geolocation Dataset (Compression & Centroids)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. COMPRESS: One Row Per Zip Code
    # The raw data has multiple points for one zip code (tracking trucks, etc).
    # We need the "Center" (Median) of the zip code.

    geo_lookup = df.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'median',
        'geolocation_lng': 'median',
        'geolocation_city': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        'geolocation_state': 'first'
    }).reset_index()

    # 3. RENAME
    # Rename columns to match the output schema we want
    geo_lookup.rename(columns={
        'geolocation_zip_code_prefix': 'zip_code_prefix',
        'geolocation_lat': 'lat',
        'geolocation_lng': 'lng',
        'geolocation_city': 'geo_city',
        'geolocation_state': 'geo_state'
    }, inplace=True)

    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    geo_lookup.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Compressed {len(df):,} raw rows into {len(geo_lookup):,} unique Zip Codes.")


if __name__ == "__main__":
    process_geolocation()