import pandas as pd
import os
import numpy as np
import unicodedata

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
RAW_PATH = os.path.join(PROJECT_ROOT, "olist", "Brazilian dataset", "archive")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_data.csv")


def normalize_text(text):
    """
    Removes accents and converts to lowercase.
    Example: "Não chegou!" -> "nao chegou!"
    Helps catch misspelled words like 'nao' vs 'não'.
    """
    if pd.isna(text): return ""
    # Normalize unicode characters (remove accents)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()


def process_reviews():
    print(f"🔨 PROCESSING: Reviews (Final Platinum Edition)")
    print("=" * 50)

    # 1. Load Data
    reviews = pd.read_csv(os.path.join(RAW_PATH, "olist_order_reviews_dataset.csv"))
    orders = pd.read_csv(os.path.join(RAW_PATH, "olist_orders_dataset.csv"))

    # 2. Date Setup
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

    # 3. Merge for Time Context
    df = reviews.merge(orders[['order_id', 'order_delivered_customer_date']], on='order_id', how='left')

    # 4. Advanced Text Cleaning (Normalization)
    print("   • Normalizing Portuguese text (removing accents)...")
    df['raw_text'] = (df['review_comment_title'].fillna("") + " " + df['review_comment_message'].fillna(""))
    df['clean_text'] = df['raw_text'].apply(normalize_text)
    df['has_comment'] = np.where(df['clean_text'] != "", 1, 0)

    # 5. Sentiment Label
    conditions = [
        (df['review_score'] <= 2),
        (df['review_score'] == 3),
        (df['review_score'] >= 4)
    ]
    df['sentiment_label'] = np.select(conditions, ['Negative', 'Neutral', 'Positive'], default='Unknown')

    # 6. EXPANDED KEYWORD ENGINE

    # A. Service / Communication Issues
    keywords_service = [
        'atendimento', 'responde', 'resposta', 'ignorou', 'contato', 'telefone',
        'email', 'sac', 'loja', 'vendedor', 'chat', 'ninguem', 'descaso'
    ]

    # B. Cancellation / Money Issues
    keywords_cancel = [
        'cancelar', 'cancelamento', 'dinheiro', 'estorno', 'devolver', 'reembolso',
        'paguei', 'valor', 'cartao', 'fatura'
    ]

    # C. Product Quality
    keywords_quality = [
        'defeito', 'quebrado', 'riscado', 'amassado', 'velho', 'usado', 'diferente',
        'errado', 'cor', 'tamanho', 'peca', 'funcionou', 'ruim', 'pessima', 'falso',
        'veio apenas', 'so veio', 'só veio'
    ]

    # D. Logistics Late (Time words)
    keywords_late = [
        'atraso', 'atrasado', 'demora', 'demorou', 'prazo', 'esperando', 'agora',
        'dia', 'semana', 'mes', 'tempo', 'tarde'
    ]

    # E. Not Received (Ghost words)
    keywords_lost = [
        'nao recebi', 'nao chegou', 'nunca chegou', 'cade', 'entregue', 'correios',
        'extraviado', 'nada', 'sumiu'
    ]

    # Helper to check keywords
    def check(pattern_list):
        pattern = '|'.join(pattern_list)
        return df['clean_text'].str.contains(pattern, regex=True).astype(int)

    df['k_service'] = check(keywords_service)
    df['k_cancel'] = check(keywords_cancel)
    df['k_quality'] = check(keywords_quality)
    df['k_late'] = check(keywords_late)
    df['k_lost'] = check(keywords_lost)

    # 7. HIERARCHICAL CLASSIFICATION
    df['issue_category'] = 'Unclassified'

    # Only classify Negative/Neutral reviews
    mask_bad = df['sentiment_label'] != 'Positive'

    # Tier 1: Money & Cancellation (High Priority)
    df.loc[mask_bad & (df['k_cancel'] == 1), 'issue_category'] = 'Cancellation/Refund'

    # Tier 2: Product Defects
    df.loc[mask_bad & (df['issue_category'] == 'Unclassified') & (
                df['k_quality'] == 1), 'issue_category'] = 'Product Quality'

    # Tier 3: Service/Rude
    df.loc[mask_bad & (df['issue_category'] == 'Unclassified') & (
                df['k_service'] == 1), 'issue_category'] = 'Service/Communication'

    # Tier 4: Not Received (Explicit text mentions)
    df.loc[mask_bad & (df['issue_category'] == 'Unclassified') & (
                df['k_lost'] == 1), 'issue_category'] = 'Logistics (Not Received)'

    # Tier 5: Late (Explicit text mentions)
    df.loc[mask_bad & (df['issue_category'] == 'Unclassified') & (
                df['k_late'] == 1), 'issue_category'] = 'Logistics (Late)'

    # Tier 6: The "Ghost Rule" (Pre-Delivery Fallback)
    df['days_diff'] = (df['review_creation_date'] - df['order_delivered_customer_date']).dt.days
    df['is_pre_delivery'] = np.where((df['days_diff'] < 0) | (df['order_delivered_customer_date'].isnull()), 1, 0)

    mask_ghost = (mask_bad) & (df['issue_category'] == 'Unclassified') & (df['is_pre_delivery'] == 1)
    df.loc[mask_ghost, 'issue_category'] = 'Logistics (Undelivered)'

    # Tier 7: Vague Dissatisfaction (Has text, but no keywords match)
    mask_vague = (mask_bad) & (df['issue_category'] == 'Unclassified') & (df['has_comment'] == 1)
    df.loc[mask_vague, 'issue_category'] = 'Vague Dissatisfaction'

    # Clean up positives
    df.loc[df['sentiment_label'] == 'Positive', 'issue_category'] = 'No Issue'

    # 8. SAVE
    cols_final = ['review_id', 'order_id', 'review_score', 'has_comment', 'sentiment_label', 'issue_category']
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df[cols_final].to_csv(OUTPUT_FILE, index=False)

    print(f"✅ SUCCESS: Processed {len(df):,} reviews.")
    print(f"--- FINAL CATEGORY BREAKDOWN ---")
    print(df['issue_category'].value_counts())


if __name__ == "__main__":
    process_reviews()