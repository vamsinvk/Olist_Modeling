import pandas as pd
import os
from textblob import TextBlob
import time

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table_nlp.csv")


def calculate_sentiment(text):
    """
    Returns a score from -1 (Negative) to +1 (Positive).
    If no text is present, returns 0 (Neutral).
    """
    if pd.isna(text) or str(text).strip() == "":
        return 0.0

    # TextBlob does the heavy lifting
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity


def add_nlp_features():
    print(f"🧠 ENGINEERING: Extracting Sentiment from Text")
    print("=" * 60)

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   • Loaded {len(df):,} rows.")
    except FileNotFoundError:
        print("❌ Error: master_table.csv not found.")
        return

    # 2. Check if we have the text column
    if 'review_comment_message' not in df.columns:
        print("❌ Error: 'review_comment_message' column is missing!")
        return

    # 3. Apply NLP (This might take 10-20 seconds)
    print("   • Analyzing customer emotions (this takes a moment)...")
    start_time = time.time()

    # The .apply() function runs our sentiment tool on every single row
    df['sentiment_score'] = df['review_comment_message'].apply(calculate_sentiment)

    # Let's also count how long the review is (Long rant vs Short 'Good')
    df['review_length'] = df['review_comment_message'].astype(str).apply(len)

    elapsed = time.time() - start_time
    print(f"   • Done in {elapsed:.2f} seconds.")

    # 4. Preview the Magic
    print("\n   --- SENTIMENT EXAMPLES ---")
    examples = df[df['review_comment_message'].notna()].sample(5)
    for _, row in examples.iterrows():
        print(f"   Score: {row['sentiment_score']:.2f} | Text: {str(row['review_comment_message'])[:50]}...")

    # 5. Save the Upgrade
    df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 60)
    print(f"✅ SUCCESS: Saved enhanced data to: {OUTPUT_FILE}")
    print(f"   • New Feature: sentiment_score (-1 to +1)")
    print(f"   • New Feature: review_length (number of characters)")


if __name__ == "__main__":
    add_nlp_features()