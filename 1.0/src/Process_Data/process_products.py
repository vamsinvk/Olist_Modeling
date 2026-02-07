import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "products_data.csv")


def process_products():
    print(f"🔨 PROCESSING: Products Dataset")
    print("=" * 50)

    # 1. Load Data
    products = pd.read_csv(os.path.join(RAW_PATH, "olist_products_dataset.csv"))

    # Try to load translations. If missing, we'll warn the user.
    translation_path = os.path.join(RAW_PATH, "product_category_name_translation.csv")
    try:
        translations = pd.read_csv(translation_path)
        print(f"   Translations loaded: {len(translations)} categories.")
    except FileNotFoundError:
        print("⚠️ WARNING: Translation file not found. Keeping Portuguese names.")
        translations = None

    # 2. MERGE Translations (if available)
    if translations is not None:
        # Merge on the Portuguese name
        products = products.merge(translations, on='product_category_name', how='left')

        # Determine final category name: Use English if available, else Portuguese
        products['category_name'] = products['product_category_name_english'].fillna(products['product_category_name'])
    else:
        products['category_name'] = products['product_category_name']

    # 3. IMPUTATION (Filling Holes)
    # Fill missing categories with 'unknown'
    missing_count = products['category_name'].isnull().sum()
    products['category_name'] = products['category_name'].fillna('unknown')
    print(f"   - Filled {missing_count} missing categories with 'unknown'.")

    # Fill missing weight/dimensions with the Median (safer than Mean for outliers)
    # We only have 2 missing, but good practice for production code
    cols_to_fix = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    for col in cols_to_fix:
        median_val = products[col].median()
        print(f"medianvalue is:{median_val}")
        products[col] = products[col].fillna(median_val)

    # 4. CLEANUP
    # Select only columns we need for analysis
    final_cols = [
        'product_id',
        'category_name',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ]
    products_data = products[final_cols]

    # 5. SAVE
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    products_data.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Saved {len(products_data):,} products to {OUTPUT_FILE}")
    print(f"   Top Categories (English):\n{products_data['category_name'].value_counts().head()}")


if __name__ == "__main__":
    process_products()