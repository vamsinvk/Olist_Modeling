import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "master_table.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "modeling")


def preprocess_for_ml():
    print(f"🤖 ML STEP 1: Preprocessing Data")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)

    # 2. Filter: We only want orders that are completed (delivered) and have a score
    df = df[df['review_score'] != -1]
    df = df[df['order_status'] == 'delivered']

    # 3. CREATE TARGET VARIABLE
    # We want to catch the "Bad" experiences.
    # Definition: Bad = 1, 2, or 3 stars. Good = 4 or 5 stars.
    # (We include 3 in 'Bad' because in business, Neutral is usually a failure to delight)
    df['is_bad_review'] = np.where(df['review_score'] <= 3, 1, 0)

    print(f"   • Class Balance:")
    print(df['is_bad_review'].value_counts(normalize=True))
    # Expect ~20-25% to be '1' (Bad)

    # 4. FEATURE SELECTION
    # We pick the columns we proved matter in EDA
    features = [
        # Time
        'actual_delivery_days', 'is_late',
        # Money
        'price', 'freight_value', 'total_payment_value', 'max_installments',
        # Product
        'product_weight_g', 'product_length_cm',
        # Location (High Cardinality - we will encode)
        'customer_state'
    ]

    X = df[features].copy()
    y = df['is_bad_review'].copy()

    # 5. HANDLING CATEGORICAL DATA
    # "customer_state" is text (SP, RJ). We need numbers.
    # We use Label Encoding (0=AC, 1=AL, ... 26=SP)
    le = LabelEncoder()
    X['customer_state'] = le.fit_transform(X['customer_state'])

    # 6. FILL MISSING VALUES
    # ML models crash on NaNs.
    X = X.fillna(0)

    # 7. TRAIN / TEST SPLIT
    # We hide 20% of data to test the model later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 8. SAVE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    print(f"\n✅ SUCCESS: Data Split & Saved!")
    print(f"   • Training Data: {len(X_train):,} rows")
    print(f"   • Testing Data:  {len(X_test):,} rows")


if __name__ == "__main__":
    preprocess_for_ml()