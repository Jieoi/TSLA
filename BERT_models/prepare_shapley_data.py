"""
Prepare Data for Shapley Value Analysis
========================================
Extracts test data from trained models for SHAP analysis
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


def prepare_data_for_model(model_name: str, daily_data_csv: str, model_path: str):
    """Prepare test data for a specific model"""
    
    if not os.path.exists(daily_data_csv):
        print(f"[WARN] Data file not found: {daily_data_csv}")
        return None
    
    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found: {model_path}")
        return None
    
    # Load daily dataset
    df = pd.read_csv(daily_data_csv)
    df['market_date'] = pd.to_datetime(df['market_date']).dt.date
    
    # Time splits
    from shared_bert_library import TIME_SPLITS
    TRAIN_START = TIME_SPLITS["TRAIN_START"]
    TRAIN_END = TIME_SPLITS["TRAIN_END"]
    TEST_START = TIME_SPLITS["TEST_START"]
    TEST_END = TIME_SPLITS["TEST_END"]
    
    # Filter to test set
    dates = pd.to_datetime(df["market_date"]).dt.date
    test_idx = (dates >= TEST_START) & (dates <= TEST_END)
    df_test = df[test_idx].copy()
    
    if len(df_test) == 0:
        print(f"[WARN] No test data found for {model_name}")
        return None
    
    # Get feature columns (exclude metadata columns)
    feat_cols = [c for c in df_test.columns if c not in 
                 ["market_date", "prev_close", "residual_close", "direction"]]
    
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test["direction"].values.astype(int)
    
    # Standardize using training data stats
    train_idx = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    X_train = df[train_idx][feat_cols].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save to npz file
    output_path = os.path.join(OUTPUT_DIR, f"shapley_data_{model_name}.npz")
    np.savez_compressed(
        output_path,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=np.array(feat_cols)
    )
    
    print(f"[OK] Prepared data for {model_name}: {len(X_test_scaled)} test samples")
    return output_path


def main():
    """Prepare data for all models"""
    
    print("=" * 80)
    print("Preparing Data for Shapley Value Analysis")
    print("=" * 80)
    
    models_config = [
        {
            'name': 'tweet',
            'daily_data_csv': os.path.join(OUTPUT_DIR, 'daily_tweet_direction_dataset.csv'),
            'model_path': os.path.join(OUTPUT_DIR, 'tweet_direction_model.pt')
        },
        {
            'name': 'tesla_article',
            'daily_data_csv': os.path.join(OUTPUT_DIR, 'daily_tesla_article_direction_dataset.csv'),
            'model_path': os.path.join(OUTPUT_DIR, 'tesla_article_direction_model.pt')
        },
        {
            'name': 'market_news',
            'daily_data_csv': os.path.join(OUTPUT_DIR, 'daily_market_article_direction_dataset.csv'),
            'model_path': os.path.join(OUTPUT_DIR, 'market_article_direction_model.pt')
        }
    ]
    
    for config in models_config:
        print(f"\nProcessing {config['name']}...")
        prepare_data_for_model(**config)
    
    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

