"""
Compute Shapley Values for Neural Network Models
================================================
Unified script for computing SHAP values for all NN models.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NN_DIR = ROOT_DIR / "NN training"
OUTPUT_DIR = NN_DIR / "outputs"
SHAPLEY_OUTPUT_DIR = OUTPUT_DIR / "shapley"

os.makedirs(SHAPLEY_OUTPUT_DIR, exist_ok=True)

# Ensure the NN utilities are importable if needed elsewhere
sys.path.insert(0, str(NN_DIR))


def load_model_and_data(model_path: Path, data_path: Path) -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """Load a trained TensorFlow model and the associated dataset."""
    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(int)

    model = tf.keras.models.load_model(model_path)
    return model, X_test, y_test


def compute_shap_values(
    model: tf.keras.Model, background: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Compute feature-level SHAP values using DeepExplainer."""
    explainer = shap.DeepExplainer(model, background)
    shap_values: Any = explainer.shap_values(target, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = np.squeeze(shap_values, axis=-1)

    return shap_values


def group_shap_by_source(shap_values: np.ndarray, feature_groups: dict[str, list[int]]) -> np.ndarray:
    """Aggregate feature-level SHAP values into broader groups."""
    n_samples, _ = shap_values.shape
    grouped = np.zeros((n_samples, len(feature_groups)), dtype=np.float32)

    for idx, feature_indices in enumerate(feature_groups.values()):
        grouped[:, idx] = shap_values[:, feature_indices].sum(axis=1)

    return grouped


def compute_aggregated_metrics(group_shap: np.ndarray, source_names: list[str]) -> pd.DataFrame:
    """Compute summary statistics for each SHAP group."""
    results: list[dict[str, Any]] = []
    abs_group_shap = np.abs(group_shap)
    row_sums = abs_group_shap.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Prevent divide-by-zero

    for idx, source in enumerate(source_names):
        phi_s = group_shap[:, idx]
        abs_phi_s = np.abs(phi_s)
        normalized = abs_phi_s / row_sums

        results.append(
            {
                "source": source,
                "importance_median": np.median(abs_phi_s),
                "importance_iqr": np.percentile(abs_phi_s, 75) - np.percentile(abs_phi_s, 25),
                "direction_median": np.median(phi_s),
                "p_plus": np.mean(phi_s > 0),
                "p_minus": np.mean(phi_s < 0),
                "share_median": np.median(normalized),
                "share_iqr": np.percentile(normalized, 75) - np.percentile(normalized, 25),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    """Compute SHAP values for all configured NN models."""
    def resolve_first_existing(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    models_config = [
        {
            "name": "market_news",
            "model_candidates": [
                OUTPUT_DIR / "market_topic_direction_model.h5",
                OUTPUT_DIR / "market_news_price_classifier_model.keras",
            ],
            "data_candidates": [
                OUTPUT_DIR / "shapley_data_market_news.npz",
                OUTPUT_DIR / "market_news_shapley_data.npz",
            ],
            "prediction_candidates": [
                OUTPUT_DIR / "market_news_predictions.csv",
                OUTPUT_DIR / "market_ouputs" / "market_news_predictions_9.csv",
                OUTPUT_DIR / "market_ouputs" / "market_news_predictions_10.csv",
                OUTPUT_DIR / "market_ouputs" / "market_news_predictions_8.csv",
            ],
            "output_path": SHAPLEY_OUTPUT_DIR / "shapley_market_news_nn.pkl",
        },
        {
            "name": "tesla_news",
            "model_candidates": [
                OUTPUT_DIR / "tesla_topic_direction_model.h5",
                OUTPUT_DIR / "tesla_news_price_classifier_model.keras",
            ],
            "data_candidates": [
                OUTPUT_DIR / "shapley_data_tesla_news.npz",
                OUTPUT_DIR / "tesla_news_shapley_data.npz",
            ],
            "prediction_candidates": [
                OUTPUT_DIR / "tesla_news_predictions.csv",
                OUTPUT_DIR / "tesla_outputs" / "tesla_news_predictions_9.csv",
                OUTPUT_DIR / "tesla_outputs" / "tesla_news_predictions_8.csv",
                OUTPUT_DIR / "tesla_outputs" / "tesla_news_predictions_7.csv",
            ],
            "output_path": SHAPLEY_OUTPUT_DIR / "shapley_tesla_news_nn.pkl",
        },
        {
            "name": "tweet",
            "model_candidates": [
                OUTPUT_DIR / "tweet_topic_direction_model.h5",
                OUTPUT_DIR / "tweet_price_classifier_model.keras",
            ],
            "data_candidates": [
                OUTPUT_DIR / "shapley_data_tweet.npz",
                OUTPUT_DIR / "tweet_shapley_data.npz",
            ],
            "prediction_candidates": [
                OUTPUT_DIR / "tweet_predictions.csv",
            ],
            "output_path": SHAPLEY_OUTPUT_DIR / "shapley_tweet_nn.pkl",
        },
    ]

    print("=" * 80)
    print("Computing Shapley Values for Neural Network Models")
    print("=" * 80)

    for config in models_config:
        name = config["name"]
        model_path = resolve_first_existing(config["model_candidates"])
        data_path = resolve_first_existing(config["data_candidates"])
        prediction_path = resolve_first_existing(config["prediction_candidates"])
        output_path: Path = config["output_path"]

        print(f"\nProcessing {name} model...")

        if model_path is None:
            print(f"  [WARN] No model found. Checked: {config['model_candidates']}")
            continue

        if data_path is None:
            print(f"  [WARN] No Shapley dataset found. Checked: {config['data_candidates']}")
            print("  [INFO] Run prepare_shapley_data_nn.py first.")
            continue
        if prediction_path is None:
            print(f"  [WARN] No predictions CSV found. Checked: {config['prediction_candidates']}")

        try:
            model, X_test, y_test = load_model_and_data(model_path, data_path)

            n_background = min(100, len(X_test))
            n_target = min(200, len(X_test))
            background = X_test[:n_background]
            target = X_test[:n_target]

            print("  [OK] Loaded model and data")
            print(f"  [INFO] Background: {n_background}, Target: {n_target}")

            print("  [1/5] Computing feature-level SHAP values...")
            shap_values = compute_shap_values(model, background, target)
            print(f"  [OK] SHAP values shape: {shap_values.shape}")

            n_features = shap_values.shape[1]
            feature_groups = {"topics": list(range(n_features))}
            source_names = ["topics"]

            print("  [2/5] Grouping SHAP values...")
            group_shap = group_shap_by_source(shap_values, feature_groups)

            print("  [3-5/5] Computing aggregated metrics...")
            metrics_df = compute_aggregated_metrics(group_shap, source_names)

            results = {
                "shap_values": shap_values,
                "group_shap": group_shap,
                "metrics": metrics_df,
                "target": target[:n_target],
                "y_test": y_test[:n_target],
                "model_path": str(model_path),
                "data_path": str(data_path),
                "prediction_path": str(prediction_path) if prediction_path else None,
            }

            if prediction_path is not None:
                try:
                    predictions_df = pd.read_csv(prediction_path)
                    results["predictions"] = predictions_df
                    print(f"  [INFO] Loaded predictions CSV: {prediction_path}")
                except Exception as pred_exc:  # pylint: disable=broad-except
                    print(f"  [WARN] Failed to load predictions CSV ({prediction_path}): {pred_exc}")

            with output_path.open("wb") as handle:
                pickle.dump(results, handle)

            print(f"  [OK] Saved results to {output_path}")
            print("\n  Metrics Summary:")
            print(metrics_df.to_string(index=False))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [ERROR] Failed to process {name}: {exc}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Shapley value computation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


