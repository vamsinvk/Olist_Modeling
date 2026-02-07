import pandas as pd
import os
import unidecode

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
# Inputs
NLP_DATA = os.path.join(PROJECT_ROOT, "data", "gold", "nlp_master_table.csv")
ITEMS_DATA = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_items_dataset.csv")
PRODUCTS_DATA = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_products_dataset.csv")
TRANS_DATA = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive",
                          "product_category_name_translation.csv")
SELLERS_DATA = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_sellers_dataset.csv")
# Output
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "final_table.csv")

# 🇧🇷 SENTIMENT ENGINE
PT_LEXICON = {
    'otimo': 1, 'bom': 1, 'excelente': 1, 'boa': 1, 'adorei': 1, 'gostei': 1,
    'recomendo': 1, 'rapido': 1, 'perfeito': 1, 'lindo': 1, 'parabens': 1,
    'chegou': 0.5, 'antes': 0.5, 'certo': 0.5, 'bem': 0.5,
    'ruim': -1, 'pessimo': -1, 'horrivel': -1, 'demora': -1, 'atraso': -1,
    'atrasou': -1, 'nunca': -1, 'nao': -0.5, 'defeito': -1, 'quebrado': -1,
    'errado': -1, 'triste': -1, 'aguardando': -0.5, 'diferente': -0.5
}


def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == "": return 0.0
    try:
        text = unidecode.unidecode(str(text)).lower()
    except:
        text = str(text).lower()
    score = sum([PT_LEXICON.get(w, 0) for w in text.split()])
    count = sum([1 for w in text.split() if w in PT_LEXICON])
    return max(min(score / count, 1.0), -1.0) if count > 0 else 0.0


def build_final():
    print(f"💎 ENGINEERING: Building Final final Dataset")
    print("=" * 60)

    # 1. Load Data
    try:
        df_main = pd.read_csv(NLP_DATA)
        df_items = pd.read_csv(ITEMS_DATA)
        df_products = pd.read_csv(PRODUCTS_DATA)
        df_trans = pd.read_csv(TRANS_DATA)
        df_sellers = pd.read_csv(SELLERS_DATA)
        print(f"   • Loaded Base Data: {len(df_main):,} rows")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Run previous NLP steps first.")
        return

    # 2. NLP: Ensure Sentiment Score is present
    print("   • Calculating Sentiment Scores...")
    df_main['full_text'] = df_main['review_comment_title'].fillna("") + " " + df_main['review_comment_message'].fillna(
        "")
    df_main['sentiment_score'] = df_main['full_text'].apply(get_sentiment)
    df_main['review_length'] = df_main['full_text'].str.len()

    # 3. CONTEXT: Add Product Categories (English)
    print("   • Merging Product Data...")

    # A. Prepare Category Translations
    df_prod = pd.merge(df_products, df_trans, on='product_category_name', how='left')
    df_prod = df_prod[['product_id', 'product_category_name_english']]
    df_prod['product_category_name_english'] = df_prod['product_category_name_english'].fillna('unknown')

    # B. Prepare Items (Order -> Product)
    df_items_unique = df_items.groupby('order_id').first().reset_index()

    # --- CRITICAL FIX: CLEAN COLUMNS BEFORE MERGE ---
    # We want to bring 'product_id' and 'seller_id' from items.
    # If df_main ALREADY has them, drop them first to avoid "product_id_x" duplicates.
    cols_to_drop = [c for c in ['product_id', 'seller_id'] if c in df_main.columns]
    if cols_to_drop:
        print(f"   • Dropping existing columns to prevent duplicates: {cols_to_drop}")
        df_main = df_main.drop(columns=cols_to_drop)
    # ------------------------------------------------

    # C. Merge Orders + Items (to get product_id)
    df_merged = pd.merge(df_main, df_items_unique[['order_id', 'product_id', 'seller_id']], on='order_id', how='left')

    # D. Merge + Products (to get category name)
    # Now 'product_id' is guaranteed to exist and be unique
    df_merged = pd.merge(df_merged, df_prod, on='product_id', how='left')

    # 4. REPUTATION & PHYSICS: Add Seller Info
    print("   • Merging Seller Data...")
    df_merged = pd.merge(df_merged, df_sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

    # Create "Same State" Feature
    df_merged['is_same_state'] = (df_merged['customer_state'] == df_merged['seller_state']).astype(int)

    # 5. Save
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("=" * 60)
    print(f"✅ SUCCESS: final Data saved to {OUTPUT_FILE}")
    print(f"   • Columns Added: product_category, seller_id, is_same_state, sentiment_score")


if __name__ == "__main__":
    build_final()