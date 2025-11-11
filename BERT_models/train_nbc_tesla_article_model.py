"""
BERT Training Script: NBC Tesla Article Model
==============================================
Predicts next-day TSLA price direction from NBC Tesla news articles
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

# Add parent directory to path for shared library
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_bert_library import (
    get_device, safe_read_csv, build_or_load_embeds,
    DirectionClassificationHead, to_loader, eval_classification,
    print_evaluation_results, resolve_first_existing, TIME_SPLITS,
    MODEL_NAME, BATCH_EMB, FP16, HID, DROPOUT, LR, WEIGHT_DECAY
)

# Configuration
DEVICE = get_device()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLE_PATH_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "tesla_articles_with_content_labeled.csv"),
    os.path.join(SCRIPT_DIR, "nbc_news_labeled.csv"),
    os.path.join(SCRIPT_DIR, "..", "BERT", "tesla_articles_with_content_labeled.csv"),
    os.path.join(SCRIPT_DIR, "..", "data", "BERT", "tesla_articles_with_content_labeled.csv"),
]
ARTICLES_RAW = resolve_first_existing(ARTICLE_PATH_CANDIDATES, description="Tesla article dataset CSV")
PRICE_PATH_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "TLSA_price_labelled_processed.csv"),
    os.path.join(SCRIPT_DIR, "..", "..", "TLSA_price_labelled_processed.csv"),
    os.path.join(SCRIPT_DIR, "..", "TLSA_price_labelled_processed.csv"),
]
PRICES_RAW = resolve_first_existing(PRICE_PATH_CANDIDATES, description="price dataset CSV")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_CACHE_NPY = os.path.join(OUTPUT_DIR, "tesla_article_embeds.npy")
DAILY_DATA_CSV = os.path.join(OUTPUT_DIR, "daily_tesla_article_direction_dataset.csv")
PRED_VAL_CSV = os.path.join(OUTPUT_DIR, "tesla_article_direction_preds_val.csv")
PRED_TEST_CSV = os.path.join(OUTPUT_DIR, "tesla_article_direction_preds_test.csv")
PRED_COMBINED_CSV = os.path.join(OUTPUT_DIR, "tesla_article_predictions.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "tesla_article_direction_model.pt")
MODEL_KERAS_PATH = os.path.join(OUTPUT_DIR, "tesla_article_price_classifier_model.keras")
SHAPLEY_DATA_PATH = os.path.join(OUTPUT_DIR, "tesla_article_shapley_data.npz")

# Time splits
TRAIN_START = TIME_SPLITS["TRAIN_START"]
TRAIN_END = TIME_SPLITS["TRAIN_END"]
VAL_START = TIME_SPLITS["VAL_START"]
VAL_END = TIME_SPLITS["VAL_END"]
TEST_START = TIME_SPLITS["TEST_START"]
TEST_END = TIME_SPLITS["TEST_END"]

# Article-specific configuration
MAX_LEN = 256  # Longer for articles
ARTICLES_BATCH_EMB = 32 if torch.cuda.is_available() else 16


def main():
    print("=" * 80)
    print("BERT Training: NBC Tesla Article Model")
    print("Model: DeBERTa-v3-base")
    print("Task: Binary classification - Predict UP/DOWN from Tesla articles")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    
    articles = safe_read_csv(ARTICLES_RAW)
    prices = safe_read_csv(PRICES_RAW)
    
    print(f"[OK] Loaded {len(articles):,} articles")
    print(f"[OK] Loaded {len(prices):,} price records")
    
    # Identify content column
    content_col = next((c for c in ["content", "text", "body", "article"] if c in articles.columns), None)
    if content_col is None:
        raise ValueError("No content column found in articles CSV")
    
    # Parse dates
    articles["market_date"] = pd.to_datetime(articles["market_date"], errors="coerce").dt.date
    articles = articles.dropna(subset=["market_date", content_col])
    
    # Parse price dates
    price_date_col = next((c for c in ["market_date", "timestamp", "date"] if c in prices.columns), None)
    if price_date_col is None:
        raise ValueError("No date column found in prices CSV")
    
    prices["market_date"] = pd.to_datetime(prices[price_date_col], errors="coerce").dt.date
    prices = prices.dropna(subset=["market_date"])
    
    # Get close price column
    close_col = next((c for c in ["close", "Close", "adj_close", "adjclose"] if c in prices.columns), None)
    if close_col is None:
        raise ValueError("No close price column found")
    
    prices[close_col] = pd.to_numeric(prices[close_col], errors="coerce")
    prices = prices.sort_values("market_date").reset_index(drop=True)
    prices["prev_close"] = prices[close_col].shift(1)
    prices["residual_close"] = prices[close_col] - prices["prev_close"]
    prices = prices.dropna(subset=["residual_close", "prev_close"])
    
    # Filter articles to valid price dates
    valid_dates = set(prices["market_date"])
    articles = articles[articles["market_date"].isin(valid_dates)].copy()
    
    print(f"[OK] Filtered to {len(articles):,} articles with valid trading dates")
    
    # ------------------------------------------------------------------------
    # 2. Create BERT embeddings (one per article, NOT aggregated by day)
    # ------------------------------------------------------------------------
    print("\n[2/6] Creating BERT embeddings for articles...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder.to(DEVICE).eval()
    
    if torch.cuda.is_available() and hasattr(encoder, 'gradient_checkpointing_enable'):
        encoder.gradient_checkpointing_enable()
        print("[OK] Gradient checkpointing enabled")
    
    article_embeds = build_or_load_embeds(
        articles[content_col], tokenizer, encoder, EMB_CACHE_NPY,
        DEVICE, batch_size=ARTICLES_BATCH_EMB, max_len=MAX_LEN
    )
    
    # Free encoder from memory
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------------
    # 3. Merge articles with prices (keep each article as separate row)
    # ------------------------------------------------------------------------
    print("\n[3/6] Merging articles with prices...")
    
    # Create embeddings DataFrame
    emb_df = pd.DataFrame(article_embeds, index=articles.index)
    emb_df["market_date"] = articles["market_date"].values
    
    # Merge with prices
    Xy = emb_df.merge(
        prices[["market_date", "prev_close", "residual_close"]],
        on="market_date", how="inner"
    ).dropna(subset=["residual_close", "prev_close"]).copy()
    
    print(f"[OK] Merged Xy rows: {len(Xy)}")
    print(f"[OK] Date range: {Xy['market_date'].min()} -> {Xy['market_date'].max()}")
    
    # Create binary direction labels (1=up, 0=down)
    Xy["direction"] = (Xy["residual_close"] > 0).astype(int)
    print(f"[OK] Class distribution: Up={Xy['direction'].sum()}, Down={(~Xy['direction'].astype(bool)).sum()}")
    
    # ------------------------------------------------------------------------
    # 4. Split data
    # ------------------------------------------------------------------------
    print("\n[4/6] Creating train/val/test splits...")
    
    dates = pd.to_datetime(Xy["market_date"]).dt.date
    
    train_idx = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_idx = (dates >= VAL_START) & (dates <= VAL_END)
    test_idx = (dates >= TEST_START) & (dates <= TEST_END)
    
    feat_cols = [c for c in Xy.columns if c not in ["market_date", "prev_close", "residual_close", "direction"]]
    X = Xy[feat_cols].to_numpy(dtype=np.float32)
    y = Xy["direction"].astype(int).to_numpy()
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[test_idx] = scaler.transform(X[test_idx])
    
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    
    print(f"[OK] Train: {X_train.shape[0]} articles")
    print(f"[OK] Val:   {X_val.shape[0]} articles")
    print(f"[OK] Test:  {X_test.shape[0]} articles")
    print(f"[OK] Train class distribution: Up={y_train.sum()} ({100*y_train.mean():.1f}%), "
          f"Down={len(y_train)-y_train.sum()} ({100*(1-y_train.mean()):.1f}%)")
    
    # Save daily dataset
    Xy.to_csv(DAILY_DATA_CSV, index=False)
    print(f"[OK] Saved dataset -> {DAILY_DATA_CSV}")
    
    # ------------------------------------------------------------------------
    # 5. Train model
    # ------------------------------------------------------------------------
    print("\n[5/6] Training BERT Classification Model...")
    print("-" * 80)
    
    model = DirectionClassificationHead(X_train.shape[1], hidden_dim=HID, dropout=DROPOUT).to(DEVICE)
    
    # Calculate pos_weight for class imbalance
    n_neg = len(y_train) - y_train.sum()
    n_pos = y_train.sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"[OK] Class balance: {n_pos} up, {n_neg} down -> pos_weight={pos_weight.item():.3f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler_grad = torch.amp.GradScaler("cuda", enabled=FP16 and DEVICE == "cuda")
    
    train_loader = to_loader(X_train, y_train, bs=32, shuffle=True)
    val_loader = to_loader(X_val, y_val, bs=64, shuffle=False)
    test_loader = to_loader(X_test, y_test, bs=64, shuffle=False)
    
    # Training loop with early stopping
    BEST_AUC = 0.0
    PATIENCE = 10
    pat = 0
    
    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=FP16 and DEVICE == "cuda"):
                pred = model(Xb)
                loss = criterion(pred, yb)
            
            scaler_grad.scale(loss).backward()
            scaler_grad.step(optimizer)
            scaler_grad.update()
            
            running_loss += loss.item() * Xb.size(0)
        
        val_metrics, _, _ = eval_classification(model, val_loader, DEVICE)
        avg_loss = running_loss / max(1, len(train_loader.dataset))
        
        # Print metrics with emphasis on F1 and ACC
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
              f"VAL ACC={val_metrics['accuracy']:.4f} F1={val_metrics['f1']:.4f} | "
              f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f} AUC={val_metrics['auc']:.4f}")
        
        val_auc = val_metrics["auc"] if not math.isnan(val_metrics["auc"]) else 0.0
        if val_auc > BEST_AUC:
            BEST_AUC = val_auc
            pat = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> New best AUC: {BEST_AUC:.4f} (saved model)")
        else:
            pat += 1
            if pat >= PATIENCE:
                print("Early stopping.")
                break
    
    # ------------------------------------------------------------------------
    # 6. Final evaluation
    # ------------------------------------------------------------------------
    print("\n[6/6] Final Evaluation...")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    val_metrics, val_preds, val_probs = eval_classification(model, val_loader, DEVICE)
    test_metrics, test_preds, test_probs = eval_classification(model, test_loader, DEVICE)
    
    print_evaluation_results(val_metrics, test_metrics)
    
    # Save predictions
    def save_preds(mask, preds, probs, true_labels, out_path):
        out = Xy.loc[mask, ["market_date"]].copy().reset_index(drop=True)
        out["y_true"] = true_labels
        out["y_pred"] = preds
        out["prob_up"] = probs
        out.to_csv(out_path, index=False)
        print(f"[OK] Saved predictions -> {out_path} ({len(out)} rows)")
    
    save_preds(val_idx, val_preds, val_probs, y_val, PRED_VAL_CSV)
    save_preds(test_idx, test_preds, test_probs, y_test, PRED_TEST_CSV)

    # Combined predictions CSV (validation + test)
    val_dates = pd.to_datetime(Xy.loc[val_idx, "market_date"]).reset_index(drop=True)
    test_dates = pd.to_datetime(Xy.loc[test_idx, "market_date"]).reset_index(drop=True)
    combined_predictions = pd.DataFrame(
        {
            "market_date": pd.concat([val_dates, test_dates], ignore_index=True),
            "split": ["val"] * len(val_probs) + ["test"] * len(test_probs),
            "prob_up": np.concatenate([val_probs, test_probs]),
            "pred_direction": np.concatenate([val_preds, test_preds]),
            "actual_direction": np.concatenate([y_val, y_test]),
        }
    )
    combined_predictions.to_csv(PRED_COMBINED_CSV, index=False)
    print(f"[OK] Saved combined predictions -> {PRED_COMBINED_CSV} ({len(combined_predictions)} rows)")

    # Shapley-ready dataset
    np.savez_compressed(
        SHAPLEY_DATA_PATH,
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(int),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(int),
        val_probs=val_probs.astype(np.float32),
        test_probs=test_probs.astype(np.float32),
        val_dates=np.array(val_dates.dt.date),
        test_dates=np.array(test_dates.dt.date),
    )
    print(f"[OK] Saved Shapley dataset -> {SHAPLEY_DATA_PATH}")

    # Convert PyTorch head to Keras for interoperability
    model_cpu = model.to("cpu").eval()
    hidden_dim = model_cpu.net[0].out_features
    input_dim = model_cpu.net[0].in_features

    keras_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    with torch.no_grad():
        dense1_weight = model_cpu.net[0].weight.detach().cpu().numpy().T
        dense1_bias = model_cpu.net[0].bias.detach().cpu().numpy()
        dense2_weight = model_cpu.net[3].weight.detach().cpu().numpy().T
        dense2_bias = model_cpu.net[3].bias.detach().cpu().numpy()

    keras_model.layers[0].set_weights([dense1_weight, dense1_bias])
    keras_model.layers[2].set_weights([dense2_weight, dense2_bias])

    keras_model.save(MODEL_KERAS_PATH)
    print(f"[OK] Saved Keras-compatible model -> {MODEL_KERAS_PATH}")
    
    print("\n[OK] Training Complete!")
    print(f"[OK] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    import math
    main()

