import pandas as pd
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score

# CONFIGURATION
PROJECT_ROOT = r"C:\Users\vamsi\PycharmProjects\ml\1.0"
# We use the final data (Metadata + NLP + Context)
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "gold", "final_table.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "deep_learning")


def train_neural_network():
    print(f"🧠 ADVANCED: Training Deep Neural Network (TensorFlow)")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)

    # 2. Prepare Target
    df['is_bad_review'] = df['review_score'].apply(lambda x: 1 if x <= 3 else 0)

    # 3. ENCODING & SCALING (Crucial for Neural Nets!)
    # Neural Nets fail if numbers are big (like Price=5000). We must scale them to 0-1 range.

    # A. Encode State
    le = LabelEncoder()
    df['customer_state_code'] = le.fit_transform(df['customer_state'].astype(str))

    # B. Target Encoding for Risks (Same logic as before)
    # Note: For simplicity in this demo, we calculate global risk.
    # In prod, you calculate this strictly on train set to avoid leakage.
    cat_risk = df.groupby('product_category_name_english')['is_bad_review'].mean()
    df['category_risk'] = df['product_category_name_english'].map(cat_risk)

    seller_risk = df.groupby('seller_id')['is_bad_review'].mean()
    df['seller_risk'] = df['seller_id'].map(seller_risk)

    # 4. Select Features
    features = [
        'actual_delivery_days', 'is_late', 'price', 'freight_value', 'product_weight_g',
        'customer_state_code', 'is_same_state',
        'sentiment_score', 'review_length',
        'category_risk', 'seller_risk'
    ]

    X = df[features].fillna(0)
    y = df['is_bad_review']

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. SCALING (The "Advanced" Step)
    # Trees don't care if Price is 100 or 0.1. Neural Nets DO care.
    print("   • Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler so we can use it in the dashboard later
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 7. BUILD THE BRAIN (Architecture)
    print("   • Building 3-Layer Neural Network...")
    model = keras.Sequential([
        # Layer 1: 64 Neurons, ReLU activation (The standard)
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),  # Forget 30% of what you learn (prevents overfitting)

        # Layer 2: 32 Neurons
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        # Layer 3: 16 Neurons
        layers.Dense(16, activation='relu'),

        # Output Layer: 1 Neuron (0 to 1 Probability)
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Standard for Yes/No problems
        metrics=['accuracy']
    )

    # 8. TRAIN
    print("\n   • Training for 10 Epochs...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=10,
        verbose=1
    )

    # 9. EVALUATE
    print("\n   • Predicting...")
    # Neural net gives probability (0.75). We round it to 0 or 1.
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    acc = history.history['val_accuracy'][-1]
    f1 = f1_score(y_test, y_pred)

    print("\n🏆 DEEP LEARNING RESULTS:")
    print("-" * 30)
    print(f"   Accuracy: {acc:.2%}")
    print(f"   F1-Score: {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred))

    # 10. SAVE
    model.save(os.path.join(MODEL_DIR, "dnn_model.h5"))
    print(f"   ✅ Brain saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_neural_network()