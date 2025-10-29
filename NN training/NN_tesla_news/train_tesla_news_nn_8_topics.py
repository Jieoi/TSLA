#!/usr/bin/env python3
"""Standardized training pipeline for Tesla-focused news neural network classification.

This script reproduces the workflow from `tesla_news_neural_net.ipynb`.

Developer: Khoi
Edited by: Xinjie
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.optimizers import AdamW

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_PATH_CANDIDATES = [
    "data/daily_tesla_topic_sentiment_price_8.csv",
   
]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_START = pd.to_datetime("2018-01-01").date()
TRAIN_END = pd.to_datetime("2022-12-31").date()
VAL_START = pd.to_datetime("2023-01-01").date()
VAL_END = pd.to_datetime("2023-12-31").date()
TEST_START = pd.to_datetime("2024-01-01").date()
TEST_END = pd.to_datetime("2025-04-14").date()

FEATURES = [
    "topic_0",
    "topic_1",
    "topic_2",
    "topic_3",
    "topic_4",
    "topic_5",
    "topic_6",
    "topic_7",
    "min_sentiment",
    "max_sentiment",
    "avg_sentiment",
]
TARGET_COLUMN = "change_vs_T_minus_1"

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "tesla_news_price_classifier_model_8.keras")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "tesla_news_predictions_8.csv")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected column '{TARGET_COLUMN}' not found in {path}")

    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d")
        else:
            raise ValueError("Neither 'date' nor 'datetime' column present for splitting")
    else:
        df["date"] = pd.to_datetime(df["date"])

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return df


def assign_split(dt: pd.Timestamp) -> str:
    d = dt.date()
    if TRAIN_START <= d <= TRAIN_END:
        return "train"
    if VAL_START <= d <= VAL_END:
        return "val"
    if TEST_START <= d <= TEST_END:
        return "test"
    return "exclude"


def prepare_splits(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    df = df.copy()
    df["split"] = df["date"].apply(lambda x: assign_split(pd.to_datetime(x)))
    df = df[df["split"] != "exclude"].reset_index(drop=True)

    X = df[FEATURES].fillna(0.0).to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    mask_train = df["split"] == "train"
    mask_val = df["split"] == "val"
    mask_test = df["split"] == "test"

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, df


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Sequential:
    model = build_model(X_train.shape[1])

    optimizer = AdamW(learning_rate=5e-4, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    class_weights_arr = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(np.unique(y_train), class_weights_arr)}

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=60,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    return model


def evaluate_split(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    results = {
        "split": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }

    print("=" * 80)
    print(f"{name.upper()} RESULTS")
    print("=" * 80)
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-Score : {results['f1']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")
    print("\nClassification Report:\n" + classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return results


def resolve_data_path() -> str:
    """Return first dataset path that exists among predefined candidates."""
    for candidate in DATA_PATH_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate dataset. Checked: {DATA_PATH_CANDIDATES}")


def main() -> None:
    data_path = resolve_data_path()
    print("=" * 80)
    print("TESLA NEWS PRICE DIRECTION CLASSIFIER")
    print(f"Dataset: {data_path}")
    print("=" * 80)

    df = load_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, df_with_splits = prepare_splits(df)

    print(f"Train shape: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    model = train_model(X_train, y_train, X_val, y_val)

    val_probs = model.predict(X_val, verbose=0).flatten()
    test_probs = model.predict(X_test, verbose=0).flatten()

    evaluate_split("Validation", y_val, val_probs)
    evaluate_split("Test", y_test, test_probs)

    predictions_df = pd.DataFrame(
        {
            "split": ["val"] * len(y_val) + ["test"] * len(y_test),
            "prob_up": np.concatenate([val_probs, test_probs]),
            "pred_direction": np.concatenate([(val_probs >= 0.5).astype(int), (test_probs >= 0.5).astype(int)]),
            "actual_direction": np.concatenate([y_val, y_test]),
        }
    )
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[OK] Saved predictions -> {PREDICTIONS_CSV}")

    model.save(MODEL_SAVE_PATH)
    print(f"[OK] Saved model -> {MODEL_SAVE_PATH}")

    shapley_data_path = os.path.join(OUTPUT_DIR, "tesla_news_shapley_data_8.npz")
    # Get test dates for alignment
    test_dates = df_with_splits[df_with_splits["split"] == "test"]["date"].values
    
    np.savez(
        shapley_data_path,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        val_probs=val_probs,
        test_probs=test_probs,
        test_dates=test_dates,
    )
    print(f"[OK] Saved Shapley dataset -> {shapley_data_path}")


if __name__ == "__main__":
    main()
