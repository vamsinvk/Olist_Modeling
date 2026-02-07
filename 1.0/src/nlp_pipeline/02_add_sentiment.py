import pandas as pd
import os
import unidecode  # You might need to install this: pip install unidecode
import time

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "nlp_master_table.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "modeling", "nlp_ready_data.csv")

# 🇧🇷 PORTUGUESE SENTIMENT DICTIONARY
# We map common words to scores.
# Positive words = +1, Negative words = -1
PT_LEXICON = {
    # Positive
    'otimo': 1, 'bom': 1, 'excelente': 1, 'boa': 1, 'adorei': 1, 'gostei': 1,
    'recomendo': 1, 'rapido': 1, 'perfeito': 1, 'lindo': 1, 'parabens': 1,
    'chegou': 0.5, 'antes': 0.5, 'certo': 0.5, 'bem': 0.5,
    # Negative
    'ruim': -1, 'pessimo': -1, 'horrivel': -1, 'demora': -1, 'atraso': -1,
    'atrasou': -1, 'nunca': -1, 'nao': -0.5, 'defeito': -1, 'quebrado': -1,
    'errado': -1, 'triste': -1, 'aguardando': -0.5, 'diferente': -0.5
}


def get_portuguese_sentiment(text):
    """
    Scans the text for Portuguese keywords and calculates a score.
    Returns: Float between -1.0 (Negative) and +1.0 (Positive)
    """
    if pd.isna(text) or str(text).strip() == "":
        return 0.0

    # 1. Normalize: Lowercase and remove accents (ótimo -> otimo)
    # We use unidecode to handle accents easily
    try:
        clean_text = unidecode.unidecode(str(text)).lower()
    except:
        clean_text = str(text).lower()

    words = clean_text.split()
    score = 0

    # 2. Score the words
    found_words = 0
    for word in words:
        if word in PT_LEXICON:
            score += PT_LEXICON[word]
            found_words += 1

    # 3. Normalize score to be between -1 and 1
    if found_words > 0:
        final_score = score / found_words
        # Cap at -1 and 1
        return max(min(final_score, 1.0), -1.0)
    else:
        return 0.0  # Neutral


def feature_engineering_nlp():
    print(f"🇧🇷 NLP PIPELINE STEP 2: Portuguese Sentiment Analysis")
    print("=" * 60)

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   • Loaded {len(df):,} rows.")
    except FileNotFoundError:
        print("❌ Error: Input file not found. Did you run Step 1 (Merge)?")
        return

    # 2. APPLY NLP
    print("   • Combining Title + Message...")
    df['full_text'] = df['review_comment_title'].fillna("") + " " + df['review_comment_message'].fillna("")

    print("   • Scanning for Portuguese keywords (Fast)...")
    start_time = time.time()

    # Apply our custom function
    df['sentiment_score'] = df['full_text'].apply(get_portuguese_sentiment)

    # Track Length
    df['review_length'] = df['full_text'].str.len()

    elapsed = time.time() - start_time
    print(f"   • Done in {elapsed:.2f} seconds.")

    # 3. PREVIEW
    print("\n   --- SAMPLE RESULTS ---")
    # Show some non-neutral examples
    subset = df[df['sentiment_score'] != 0][['review_score', 'sentiment_score', 'full_text']].sample(5)
    for _, row in subset.iterrows():
        print(
            f"   Stars: {row['review_score']} | Score: {row['sentiment_score']:.2f} | Text: {row['full_text'][:50]}...")

    # 4. SAVE
    columns_to_keep = [
        'review_score', 'actual_delivery_days', 'is_late', 'price',
        'freight_value', 'product_weight_g', 'customer_state',
        'sentiment_score', 'review_length'
    ]

    # Filter only finished orders
    df_final = df[df['review_score'] != -1][columns_to_keep].copy()
    df_final = df_final.fillna(0)  # Safety fill

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"✅ READY FOR ML: Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    feature_engineering_nlp()