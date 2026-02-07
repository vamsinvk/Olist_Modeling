import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_products_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "products_data.csv")


def process_products():
    print(f"🔨 PROCESSING: Products Dataset (Physics Logic)")

    # 1. Load Data
    df = pd.read_csv(RAW_FILE)

    # 2. IMPUTE MISSING DIMENSIONS (Hierarchical Logic)
    # Strategy: If a 'watch' is missing weight, use the median weight of all other 'watches'.
    # If the category is unique or missing, fall back to the global median.

    dims = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']

    for col in dims:
        # Step A: Category Context
        df[col] = df[col].fillna(df.groupby('product_category_name')[col].transform('median'))

        # Step B: Global Fallback (Safety Net)
        df[col] = df[col].fillna(df[col].median())

    # 3. FEATURE: VOLUMETRIC WEIGHT (The "Bulky" Factor)
    # Formula: (L x W x H) / 6000. Result is in kg.
    # We use a standard logistics divisor of 6000.
    df['volumetric_weight_kg'] = (
            (df['product_length_cm'] * df['product_width_cm'] * df['product_height_cm']) / 6000
    ).round(3)

    # 4. FEATURE: ACTUAL WEIGHT (KG)
    df['product_weight_kg'] = df['product_weight_g'] / 1000

    # 5. FEATURE: DENSITY RATIO
    # Formula: Actual Weight / Volumetric Weight
    # Logic:
    #   > 1.0 = Dense (Heavy for its size, e.g., Dumbbells).
    #   < 1.0 = Volumetric (Light for its size, e.g., Pillows).
    # We replace infinity (div by zero) with 0.
    df['density_ratio'] = (df['product_weight_kg'] / df['volumetric_weight_kg']).replace([np.inf, -np.inf], 0).fillna(0)

    # 6. FEATURE: MISSING INFO RISK
    # If photos or description are 0/NaN, the product is "Ghostly" (High Risk of dissatisfaction)
    df['product_photos_qty'] = df['product_photos_qty'].fillna(0)
    df['is_missing_info'] = np.where(
        (df['product_photos_qty'] == 0) | (df['product_description_lenght'].fillna(0) == 0), 1, 0)

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} products.")
    print(f"   Avg Density Ratio: {df['density_ratio'].mean():.2f}")
    print(f"   Missing Info Products: {df['is_missing_info'].sum():,} ({df['is_missing_info'].mean():.1%})")


if __name__ == "__main__":
    process_products()