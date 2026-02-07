import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
BASIC_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "final_dataset_basic.csv")
NLP_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "final_dataset_nlp.csv")


def validate():
    print("🧪 TEST: Validating Final Datasets...")

    if not os.path.exists(BASIC_FILE): return print("❌ Basic File missing.")

    df = pd.read_csv(BASIC_FILE)

    # TEST 1: Check Target Variable
    if 'review_score' in df.columns:
        print("   ✅ PASS: Target 'review_score' is present.")
    else:
        print("   ❌ FAIL: Target variable missing!")

    # TEST 2: Check Distance Calculation
    # Distance should be positive and max around 4000-5000km (Brazil is big)
    if df['distance_km'].min() >= 0 and df['distance_km'].max() < 10000:
        print(f"   ✅ PASS: Distances look physical (Avg: {df['distance_km'].mean():.1f} km).")
    else:
        print(f"   ❌ FAIL: Distances are weird (Min: {df['distance_km'].min()}, Max: {df['distance_km'].max()}).")

    # TEST 3: Check Marketing Join
    # 'seller_segment_group' should exist and have values like 'Unknown'
    if 'seller_segment_group' in df.columns:
        print("   ✅ PASS: Marketing data merged successfully.")
    else:
        print("   ❌ FAIL: Marketing columns missing.")

    # TEST 4: Compare Basic vs NLP
    # NLP file should have 'review_comment_message'
    df_nlp = pd.read_csv(NLP_FILE, keep_default_na=False)
    if 'review_comment_message' in df_nlp.columns:
        print("   ✅ PASS: NLP Dataset contains text column.")
    else:
        print("   ❌ FAIL: NLP Dataset missing text.")


if __name__ == "__main__":
    validate()