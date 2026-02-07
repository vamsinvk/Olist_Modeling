import pandas as pd
import os
import numpy as np

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_FILE = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive", "olist_order_reviews_dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "reviews_data.csv")

def process_reviews():
    print(f"🔨 PROCESSING: Reviews Dataset (NLP & Interaction Logic) [FIXED]")

    # 1. Load Data (Force text columns to be strings to avoid DtypeWarnings)
    df = pd.read_csv(RAW_FILE, dtype={
        'review_comment_title': str,
        'review_comment_message': str
    })

    # 2. Convert Dates
    df['review_creation_date'] = pd.to_datetime(df['review_creation_date'], errors='coerce')
    df['review_answer_timestamp'] = pd.to_datetime(df['review_answer_timestamp'], errors='coerce')

    # 3. CLEANING: Text Handling
    # Replace actual NaNs with empty strings.
    # We also replace the string 'nan' which sometimes appears in bad CSVs.
    df['review_comment_title'] = df['review_comment_title'].fillna('').replace('nan', '')
    df['review_comment_message'] = df['review_comment_message'].fillna('').replace('nan', '')

    # 4. FEATURE: COMMENT VOLUME
    df['comment_length'] = df['review_comment_message'].str.len()

    # 5. FEATURE: HAS COMMENT?
    df['has_comment'] = np.where(df['comment_length'] > 0, 1, 0)

    # 6. FEATURE: SELLER RESPONSE TIME
    df['response_time_hours'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.total_seconds() / 3600
    df['response_time_hours'] = df['response_time_hours'].clip(lower=0)

    # 7. DEDUPLICATION
    # Sort by answer time to keep the most relevant one
    df = df.sort_values(by='review_answer_timestamp', ascending=False)
    df_dedup = df.drop_duplicates(subset=['order_id'], keep='first')

    # 8. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_dedup.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df_dedup):,} unique reviews.")
    print(f"   Reviews with Comments: {df_dedup['has_comment'].mean():.1%}")

if __name__ == "__main__":
    process_reviews()