"""
Shared BERT Library for Training
=================================
Common functions for BERT-based classification models
"""

import os
import re
import math
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Time splits
TIME_SPLITS = {
    "TRAIN_START": pd.to_datetime("2011-12-01").date(),
    "TRAIN_END": pd.to_datetime("2023-12-31").date(),
    "VAL_START": pd.to_datetime("2024-01-01").date(),
    "VAL_END": pd.to_datetime("2024-04-14").date(),
    "TEST_START": pd.to_datetime("2024-04-15").date(),
    "TEST_END": pd.to_datetime("2025-04-14").date(),
}

# Model configuration
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 128  # Default for tweets, can be overridden for articles
BATCH_EMB = 64
FP16 = True

# Classification head configuration
HID = 256
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 0.01


def get_device():
    """Get compute device (CUDA or CPU)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[OK] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA Version: {torch.version.cuda}")
        print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[OK] TF32 enabled for faster training")
    else:
        print("[WARN] No GPU detected - training will be SLOW on CPU")
    return device


def repair_csv_unclosed_quotes(in_path: str, out_path: str, encoding="utf-8"):
    """Repair CSV files with unclosed quotes"""
    def quotes_balanced(s: str) -> bool:
        return (s.replace('""', '').count('"') % 2) == 0
    
    rows_written = 0
    with open(in_path, "r", encoding=encoding, errors="replace", newline="") as fin, \
         open(out_path, "w", encoding=encoding, newline="") as fout:
        header = fin.readline()
        if not header:
            raise ValueError("Empty CSV.")
        fout.write(header)
        buf = ""
        for line in fin:
            buf = (buf + line) if buf else line
            if quotes_balanced(buf):
                fout.write(buf)
                rows_written += 1
                buf = ""
    print(f"[OK] Repaired {out_path} (~{rows_written:,} rows)")


def safe_read_csv(path: str, fallback_path: str = None, **kwargs) -> pd.DataFrame:
    """Safely read CSV file, repairing if necessary"""
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)
    except Exception as e:
        if not fallback_path:
            fallback_path = os.path.splitext(path)[0] + "_clean.csv"
        print(f"Direct read failed -> repairing CSV: {e}")
        repair_csv_unclosed_quotes(path, fallback_path)
        return pd.read_csv(fallback_path, dtype=str, low_memory=False, **kwargs)


def clean_text(s: str) -> str:
    """Clean text for BERT processing"""
    if not isinstance(s, str):
        return ""
    _url = re.compile(r"https?://\S+")
    _mention = re.compile(r"@\w+")
    _whitespace = re.compile(r"\s+")
    
    s = _url.sub("<URL>", s)
    s = _mention.sub("<USER>", s)
    s = _whitespace.sub(" ", s).strip()
    return s


def mean_pooling(last_hidden_state, attention_mask):
    """Mean pooling for sentence embeddings"""
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-6)
    return summed / counts


class TextDataset(Dataset):
    """Dataset for text data"""
    def __init__(self, texts):
        self.texts = list(texts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        return self.texts[i]


def collate_fn_factory(tokenizer, max_len):
    """Create collate function for DataLoader"""
    def collate(batch_texts):
        return tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return collate


def build_or_load_embeds(texts_df, tokenizer, encoder, cache_path, device, batch_size=BATCH_EMB, max_len=MAX_LEN):
    """Build or load cached embeddings"""
    n_expected = len(texts_df)
    
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)
            if arr.shape[0] == n_expected:
                print(f"[OK] Loaded cached embeddings: {arr.shape}")
                return arr
            else:
                print(f"[WARN] Cache mismatch ({arr.shape[0]} vs {n_expected}) -> rebuilding.")
        except Exception as e:
            print(f"[WARN] Cache load failed ({e}) -> rebuilding.")
    
    print(f"Building embeddings for {n_expected:,} texts...")
    ds = TextDataset(texts_df.fillna(""))
    collate_fn = collate_fn_factory(tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    
    chunks = []
    encoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=FP16 and device == "cuda"):
                out = encoder(**batch)
                pooled = mean_pooling(out.last_hidden_state, batch["attention_mask"])
            chunks.append(pooled.detach().float().cpu())
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {(i + 1) * batch_size:,}/{n_expected:,} texts...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    arr = torch.vstack(chunks).numpy()
    np.save(cache_path, arr)
    print(f"[OK] Saved embeddings cache: {arr.shape}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return arr


class DirectionClassificationHead(nn.Module):
    """Classification head for price direction prediction (UP/DOWN)"""
    def __init__(self, in_dim, hidden_dim=HID, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)  # Returns logits


def to_loader(X, y, bs=64, shuffle=True):
    """Create DataLoader from numpy arrays"""
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=bs, shuffle=shuffle)


def eval_classification(model, loader, device):
    """Evaluate classification model - returns metrics including F1 and ACC"""
    model.eval()
    logits_list, labels_list = [], []
    
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", enabled=FP16 and device == "cuda"):
                out = model(Xb)
            logits_list.append(out.detach().cpu().numpy())
            labels_list.append(yb.detach().cpu().numpy())
    
    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list).astype(int)
    
    # Clip logits to prevent overflow
    logits = np.clip(logits, -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    # Calculate metrics - F1 and ACC are key
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Handle AUC calculation
    if len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }, preds, probs


def print_evaluation_results(val_metrics, test_metrics):
    """Print evaluation results with emphasis on F1 and ACC"""
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"VALIDATION SET:")
    print(f"  Accuracy (ACC): {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score (F1):  {val_metrics['f1']:.4f}")
    print(f"  Precision:      {val_metrics['precision']:.4f}")
    print(f"  Recall:         {val_metrics['recall']:.4f}")
    print(f"  ROC-AUC:        {val_metrics['auc']:.4f}")
    print()
    print(f"TEST SET:")
    print(f"  Accuracy (ACC): {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score (F1):  {test_metrics['f1']:.4f}")
    print(f"  Precision:      {test_metrics['precision']:.4f}")
    print(f"  Recall:         {test_metrics['recall']:.4f}")
    print(f"  ROC-AUC:        {test_metrics['auc']:.4f}")
    print("=" * 80)

