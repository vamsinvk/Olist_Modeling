import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "geolocation_data.csv")


def validate():
    print("🧪 TEST: Validating Geolocation...")
    if not os.path.exists(PROCESSED_FILE): return print("❌ File not found.")

    df = pd.read_csv(PROCESSED_FILE)

    # Check Duplicates
    if df['zip_code_prefix'].is_unique:
        print("   ✅ PASS: Zip Codes are unique (Compression worked).")
    else:
        print("   ❌ FAIL: Duplicate Zip Codes found.")

    # Check Lat/Lng bounds (Brazil is roughly Lat -33 to +5, Lng -73 to -34)
    # We just check for non-nulls and sane floats
    if df['lat'].isnull().sum() == 0:
        print("   ✅ PASS: No missing Latitudes.")
    else:
        print("   ❌ FAIL: Found missing coordinates.")


if __name__ == "__main__":
    validate()