"""
Compute Shapley Values for BERT Models
=======================================
Consolidated script for computing SHAP values for all BERT-based models.
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
import torch
import torch.nn as nn

# Resolve important paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BERT_DIR = ROOT_DIR / "BERT_models"
OUTPUT_DIR = BERT_DIR / "outputs"
SHAPLEY_OUTPUT_DIR = OUTPUT_DIR / "shapley"

os.makedirs(SHAPLEY_OUTPUT_DIR, exist_ok=True)

# Ensure we can import from BERT_models
sys.path.insert(0, str(BERT_DIR))
from shared_bert_library import DirectionClassificationHead, TIME_SPLITS, get_device  # type: ignore

DEVICE = get_device()


class SigmoidModel(nn.Module):
    """Wrapper that applies sigmoid activation for SHAP compatibility."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base(x)).unsqueeze(-1)


def load_model_and_data(model_path: Path, data_path: Path) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Load a trained model along with its evaluation dataset."""
    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(int)

    model = DirectionClassificationHead(X_test.shape[1])
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    return model, X_test, y_test


def compute_shap_values(
    model: nn.Module, background: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Compute feature-level SHAP values using DeepExplainer."""
    model = model.to("cpu").eval()
    wrapped_model = SigmoidModel(model)

    background_tensor = torch.from_numpy(background.astype(np.float32))
    target_tensor = torch.from_numpy(target.astype(np.float32))

    explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    shap_values: Any = explainer.shap_values(target_tensor, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.asarray(shap_values)


def group_shap_by_source(shap_values: np.ndarray, feature_groups: dict[str, list[int]]) -> np.ndarray:
    """Aggregate feature-level SHAP values into group-level contributions."""
    n_samples, _ = shap_values.shape
    n_sources = len(feature_groups)
    grouped = np.zeros((n_samples, n_sources), dtype=np.float32)

    for source_idx, feature_indices in enumerate(feature_groups.values()):
        grouped[:, source_idx] = shap_values[:, feature_indices].sum(axis=1)

    return grouped


def compute_aggregated_metrics(group_shap: np.ndarray, source_names: list[str]) -> pd.DataFrame:
    """Compute summary metrics (importance, direction, share) for each source."""
    results: list[dict[str, Any]] = []
    abs_group_shap = np.abs(group_shap)
    row_sums = abs_group_shap.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

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
    """Compute SHAP values for all configured BERT models."""
    models_config = [
        {
            "name": "tweet",
            "model_path": OUTPUT_DIR / "tweet_direction_model.pt",
            "data_path": OUTPUT_DIR / "shapley_data_tweet.npz",
            "output_path": SHAPLEY_OUTPUT_DIR / "shapley_tweet.pkl",
        },
        {
            "name": "tesla_article",
            "model_path": OUTPUT_DIR / "tesla_article_direction_model.pt",
            "data_path": OUTPUT_DIR / "shapley_data_tesla_article.npz",
            "output_path": SHAPLEY_OUTPUT_DIR / "shapley_tesla_article.pkl",
        },
        {
            "name": "market_news",
            "model_path": OUTPUT_DIR / "market_article_direction_model.pt",
            "data_path": OUTPUT_DIR / "shapley_data_market_news.npz",
            "output_path": SHAPLEY_OUTPUT_DIR / "shapley_market_news.pkl",
        },
    ]

    print("=" * 80)
    print("Computing Shapley Values for BERT Models")
    print("=" * 80)

    for config in models_config:
        name = config["name"]
        model_path: Path = config["model_path"]
        data_path: Path = config["data_path"]
        output_path: Path = config["output_path"]

        print(f"\nProcessing {name} model...")

        if not model_path.exists():
            print(f"  [WARN] Model not found: {model_path}")
            continue

        if not data_path.exists():
            print(f"  [WARN] Data file not found: {data_path}")
            print("  [INFO] Run prepare_shapley_data.py first to create data files.")
            continue

        try:
            model, X_test, y_test = load_model_and_data(model_path, data_path)
            n_background = min(100, len(X_test))
            n_target = min(200, len(X_test))

            background = X_test[:n_background]
            target = X_test[:n_target]

            print(f"  [OK] Loaded model and data")
            print(f"  [INFO] Background samples: {n_background}, Target samples: {n_target}")

            print("  [1/5] Computing feature-level SHAP values...")
            shap_values = compute_shap_values(model, background, target)
            print(f"  [OK] SHAP values shape: {shap_values.shape}")

            n_features = shap_values.shape[1]
            feature_groups = {"bert_embeddings": list(range(n_features))}
            source_names = ["bert_embeddings"]

            print("  [2/5] Grouping SHAP values by source...")
            group_shap = group_shap_by_source(shap_values, feature_groups)

            print("  [3-5/5] Computing aggregated metrics...")
            metrics_df = compute_aggregated_metrics(group_shap, source_names)

            results = {
                "shap_values": shap_values,
                "group_shap": group_shap,
                "metrics": metrics_df,
                "target": target[:n_target],
                "y_test": y_test[:n_target],
            }

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


