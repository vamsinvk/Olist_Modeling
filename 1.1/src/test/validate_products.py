import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "products_data.csv")


def validate():
    print("🧪 TEST: Validating Products Logic...")

    if not os.path.exists(PROCESSED_FILE):
        print("❌ ERROR: Processed file not found. Run process_products.py first.")
        return

    df = pd.read_csv(PROCESSED_FILE)

    # TEST 1: Check for Remaining NaNs
    # We imputed everything, so there should be 0 NaNs in dimensions
    dims = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    nan_count = df[dims].isnull().sum().sum()

    if nan_count == 0:
        print("   ✅ PASS: All physical dimensions imputed successfully.")
    else:
        print(f"   ❌ FAIL: Found {nan_count} remaining NaNs in dimensions.")

    # TEST 2: Check Volumetric Math
    # Manual check: Is vol_weight approx (L*W*H)/6000?
    # We check the first row as a spot check
    sample = df.iloc[0]
    calc_vol = (sample['product_length_cm'] * sample['product_width_cm'] * sample['product_height_cm']) / 6000

    if np.isclose(sample['volumetric_weight_kg'], calc_vol, atol=0.01):
        print("   ✅ PASS: Volumetric Weight formula applied correctly.")
    else:
        print(f"   ❌ FAIL: Math mismatch. Expected {calc_vol:.3f}, got {sample['volumetric_weight_kg']:.3f}")

    # TEST 3: Check Density Ratio sanity
    # It shouldn't be negative
    negatives = df[df['density_ratio'] < 0]
    if len(negatives) == 0:
        print("   ✅ PASS: No negative density ratios.")
    else:
        print(f"   ❌ FAIL: Found {len(negatives)} negative density ratios.")

    print("-" * 30)
    print("📊 SAMPLE INSIGHTS:")
    print(f"   Heaviest Item: {df['product_weight_kg'].max():.1f} kg")
    print(f"   Bulkiest Item (Volumetric): {df['volumetric_weight_kg'].max():.1f} kg")
    print(f"   Products with 0 Photos: {df[df['product_photos_qty'] == 0].shape[0]}")


if __name__ == "__main__":
    validate()