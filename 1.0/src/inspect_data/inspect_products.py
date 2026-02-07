import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
FILE_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_products_dataset.csv")


def inspect_products():
    print(f"🕵️ INSPECTING: Products Dataset")
    print(f"   File: {os.path.basename(FILE_PATH)}")
    print("=" * 50)

    # 1. Load Data
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {FILE_PATH}")
        return

    print(f"ROWS:    {len(df):,}")
    print(f"COLUMNS: {len(df.columns)}")

    # 2. Missing Categories (The "Business Problem")
    # If a product has no category, we can't analyze "Sales by Category"
    missing_cat = df['product_category_name'].isnull().sum()
    print(f"\n--- MISSING CATEGORIES ---")
    print(f"Products with NO Category: {missing_cat:,} ({missing_cat / len(df):.2%})")

    # 3. Missing Physical Specs (The "Logistics Problem")
    # If weight or dimensions are missing, we can't estimate shipping costs
    cols_specs = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    missing_specs = df[cols_specs].isnull().any(axis=1).sum()
    print(f"\n--- MISSING SPECS ---")
    print(f"Products with missing dimensions/weight: {missing_specs:,}")

    # 4. Top Categories (What are we actually selling?)
    print(f"\n--- TOP 5 CATEGORIES (Raw Portuguese) ---")
    print(df['product_category_name'].value_counts().head(5))


if __name__ == "__main__":
    inspect_products()