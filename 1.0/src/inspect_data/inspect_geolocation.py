import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_geolocation_dataset.csv")


def inspect_geolocation():
    print(f"🕵️ INSPECTING: Geolocation Dataset")
    print("=" * 50)

    df = pd.read_csv(FILE_PATH)
    print(f"ROWS: {len(df):,}")

    # 1. Duplicates
    unique_zips = df['geolocation_zip_code_prefix'].nunique()
    print(f"\n--- DUPLICATION CHECK ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Unique Zip Codes: {unique_zips:,}")
    print(f"Duplication Factor: {len(df) / unique_zips:.1f}x (Avg rows per zip)")

    # 2. State Distribution
    print(f"\n--- TOP STATES (GPS Pings) ---")
    print(df['geolocation_state'].value_counts().head())


if __name__ == "__main__":
    inspect_geolocation()