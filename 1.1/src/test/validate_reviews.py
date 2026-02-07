import pandas as pd
import os

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml"
PROCESSED_FILE = os.path.join(PROJECT_ROOT,"1.1", "data", "processed", "reviews_data.csv")


def validate():
    print("🧪 TEST: Validating Reviews Logic... [FIXED]")

    if not os.path.exists(PROCESSED_FILE):
        print("❌ ERROR: Processed file not found.")
        return

    # CRITICAL FIX: keep_default_na=False ensures empty strings "" are read as "" and not NaN
    df = pd.read_csv(PROCESSED_FILE, keep_default_na=False)

    # TEST 1: Check Duplicates
    if df['order_id'].is_unique:
        print("   ✅ PASS: Deduplication successful.")
    else:
        print(f"   ❌ FAIL: Found duplicates.")

    # TEST 2: Check Text Cleaning
    # Now that we turned off auto-NaN, we check if any 'nan' strings slipped through
    # or if purely empty strings are present (which is GOOD/Expected).

    # We want to ensure no *Null* objects exist. Since we used keep_default_na=False,
    # real nulls would be empty strings.
    # We check if the column type is object/string.

    # Let's check if the string "nan" exists (bad)
    bad_nans = df[df['review_comment_message'] == 'nan']

    if len(bad_nans) == 0:
        print("   ✅ PASS: Text columns sanitized (Empty strings preserved correctly).")
    else:
        print(f"   ❌ FAIL: Found literal 'nan' strings in message.")

    # TEST 3: Response Time
    # Since we used keep_default_na=False, numeric columns might be read as strings if they have empty values.
    # We force convert response_time to numeric for the test.
    df['response_time_hours'] = pd.to_numeric(df['response_time_hours'], errors='coerce').fillna(0)

    neg_response = df[df['response_time_hours'] < 0]
    if len(neg_response) == 0:
        print("   ✅ PASS: Response times are valid.")
    else:
        print(f"   ❌ FAIL: Found negative response times.")

    print("-" * 30)
    print("📊 SAMPLE INSIGHTS:")
    print(f"   Avg Comment Length: {df['comment_length'].mean():.0f} characters")


if __name__ == "__main__":
    validate()