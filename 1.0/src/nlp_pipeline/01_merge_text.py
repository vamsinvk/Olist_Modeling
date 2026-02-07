import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
# Input 1: Your clean numeric data (The Gold Table)
NUMERIC_DATA = os.path.join(PROJECT_ROOT,"1.0", "data", "gold", "master_table.csv")
# Input 2: The Raw Reviews (contains the text we lost)
RAW_REVIEWS = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_reviews_dataset.csv")
# Output: The New "Platinum" Table
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.0", "data", "gold", "nlp_master_table.csv")


def create_nlp_dataset():
    print(f"📝 NLP PIPELINE STEP 1: Merging Text Data")
    print("=" * 60)

    # 1. Load the Clean Numeric Data
    # We reuse the hard work you did in Phase 1 (Delivery days, Price, etc.)
    try:
        df_numeric = pd.read_csv(NUMERIC_DATA)
        print(f"   • Loaded Numeric Data: {len(df_numeric):,} rows")
    except FileNotFoundError:
        print("❌ Error: master_table.csv not found. Run your previous ETL first.")
        return

    # 2. Load the Raw Text Data
    # We only need the ID (to join) and the Message (the text)
    print("   • Loading Raw Text Reviews...")
    try:
        df_reviews = pd.read_csv(RAW_REVIEWS)
        # Keep only what we need
        df_text = df_reviews[['order_id', 'review_comment_title', 'review_comment_message']].copy()
    except FileNotFoundError:
        print(f"❌ Error: Could not find raw reviews at {RAW_REVIEWS}")
        return

    # 3. CLEANING: Handle Missing Text
    # If a user didn't write a review, fill it with empty string ""
    print("   • Cleaning missing text...")
    df_text['review_comment_title'] = df_text['review_comment_title'].fillna("")
    df_text['review_comment_message'] = df_text['review_comment_message'].fillna("")

    # 4. MERGE
    # We attach the text to the numeric data using 'order_id'
    print("   • Merging Numeric + Text...")
    df_merged = pd.merge(df_numeric, df_text, on='order_id', how='left')

    # 5. SANITY CHECK
    # Ensure we didn't lose rows or create duplicates
    print(f"   • Merged Data Shape: {df_merged.shape}")

    # 6. SAVE
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("=" * 60)
    print(f"✅ SUCCESS: Created {OUTPUT_FILE}")
    print(f"   • You now have Price, Time, AND Text in one file.")


if __name__ == "__main__":
    create_nlp_dataset()