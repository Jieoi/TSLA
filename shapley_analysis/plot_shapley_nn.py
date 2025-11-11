"""
Plot Shapley Value Analysis Results for NN Models
=================================================
Generates consolidated plots for NN-based SHAP outputs.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NN_DIR = ROOT_DIR / "NN training"
OUTPUT_DIR = NN_DIR / "outputs"
SHAPLEY_OUTPUT_DIR = OUTPUT_DIR / "shapley"

os.makedirs(SHAPLEY_OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def load_shapley_results(model_name: str):
    """Load SHAP results for a given NN model."""
    results_path = SHAPLEY_OUTPUT_DIR / f"shapley_{model_name}_nn.pkl"
    if not results_path.exists():
        print(f"[WARN] Results not found: {results_path}")
        return None

    with results_path.open("rb") as handle:
        return pickle.load(handle)


def plot_comparison() -> None:
    """Plot comparisons across available NN models."""
    model_names = ["market_news", "tesla_news"]
    model_labels = ["Market News", "Tesla News"]
    all_metrics = []

    for model_name, model_label in zip(model_names, model_labels):
        results = load_shapley_results(model_name)
        if results is None:
            continue

        metrics = results["metrics"].copy()
        metrics["model"] = model_label
        all_metrics.append(metrics)

    if not all_metrics:
        print("[ERROR] No results found to plot")
        return

    df = pd.concat(all_metrics, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NN Models: Shapley Value Analysis Comparison", fontsize=16, fontweight="bold")

    # Importance
    ax1 = axes[0, 0]
    sns.barplot(data=df, x="model", y="importance_median", ax=ax1, palette="husl")
    ax1.set_title("Importance: Median |φ_s|", fontweight="bold")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Importance")
    for i, value in enumerate(df["importance_median"]):
        ax1.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontweight="bold")

    # Direction
    ax2 = axes[0, 1]
    sns.barplot(data=df, x="model", y="direction_median", ax=ax2, palette="husl")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Direction: Median φ_s", fontweight="bold")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Direction")
    for i, value in enumerate(df["direction_median"]):
        ax2.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontweight="bold",
        )

    # Share
    ax3 = axes[1, 0]
    sns.barplot(data=df, x="model", y="share_median", ax=ax3, palette="husl")
    ax3.set_title("Share: Median Normalized |φ_s|", fontweight="bold")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Share")
    for i, value in enumerate(df["share_median"]):
        ax3.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontweight="bold")

    # Polarity
    ax4 = axes[1, 1]
    df_polarity = pd.melt(df, id_vars=["model"], value_vars=["p_plus", "p_minus"], var_name="polarity", value_name="probability")
    df_polarity["polarity"] = df_polarity["polarity"].map({"p_plus": "Positive (P+)", "p_minus": "Negative (P-)"})
    sns.barplot(data=df_polarity, x="model", y="probability", hue="polarity", ax=ax4, palette=["green", "red"])
    ax4.set_title("Polarity: P+ vs P-", fontweight="bold")
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Probability")
    ax4.legend(title="Polarity")

    plt.tight_layout()
    output_path = SHAPLEY_OUTPUT_DIR / "shapley_nn_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved comparison plot: {output_path}")
    plt.close()


def main() -> None:
    """Generate consolidated SHAP plots for NN models."""
    print("=" * 80)
    print("Generating Shapley Value Plots for NN Models")
    print("=" * 80)

    plot_comparison()

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print(f"Plots saved to: {SHAPLEY_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()


