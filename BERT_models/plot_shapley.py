"""
Plot Shapley Value Analysis Results
====================================
Creates visualizations of Shapley value contributions by source
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
SHAPLEY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "shapley")
os.makedirs(SHAPLEY_OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_shapley_results(model_name: str):
    """Load Shapley results for a model"""
    results_path = os.path.join(SHAPLEY_OUTPUT_DIR, f'shapley_{model_name}.pkl')
    
    if not os.path.exists(results_path):
        print(f"[WARN] Results not found: {results_path}")
        return None
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def plot_importance_comparison():
    """Plot importance comparison across all three models"""
    
    model_names = ['tweet', 'tesla_article', 'market_news']
    model_labels = ['Elon Musk Tweets', 'NBC Tesla Articles', 'NBC Market News']
    
    all_metrics = []
    
    for model_name, model_label in zip(model_names, model_labels):
        results = load_shapley_results(model_name)
        if results is None:
            continue
        
        metrics = results['metrics'].copy()
        metrics['model'] = model_label
        all_metrics.append(metrics)
    
    if not all_metrics:
        print("[ERROR] No results found to plot")
        return
    
    df = pd.concat(all_metrics, ignore_index=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Shapley Value Analysis: Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Importance (Median Absolute Group SHAP)
    ax1 = axes[0, 0]
    sns.barplot(data=df, x='model', y='importance_median', ax=ax1, palette='husl')
    ax1.set_title('Importance: Median |φ_s|', fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df['importance_median']):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Direction (Median Group SHAP)
    ax2 = axes[0, 1]
    sns.barplot(data=df, x='model', y='direction_median', ax=ax2, palette='husl')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Direction: Median φ_s', fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Direction')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df['direction_median']):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    # 3. Share (Normalized Median Share)
    ax3 = axes[1, 0]
    sns.barplot(data=df, x='model', y='share_median', ax=ax3, palette='husl')
    ax3.set_title('Share: Median Normalized |φ_s|', fontweight='bold')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Share')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df['share_median']):
        ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Polarity (P+ vs P-)
    ax4 = axes[1, 1]
    df_polarity = pd.melt(df, id_vars=['model'], value_vars=['p_plus', 'p_minus'],
                          var_name='polarity', value_name='probability')
    df_polarity['polarity'] = df_polarity['polarity'].map({'p_plus': 'Positive (P+)', 'p_minus': 'Negative (P-)'})
    sns.barplot(data=df_polarity, x='model', y='probability', hue='polarity', ax=ax4, palette=['green', 'red'])
    ax4.set_title('Polarity: P+ vs P-', fontweight='bold')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Probability')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Polarity')
    
    plt.tight_layout()
    
    output_path = os.path.join(SHAPLEY_OUTPUT_DIR, 'shapley_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparison plot: {output_path}")
    
    plt.close()


def plot_individual_model_shapley(model_name: str, model_label: str):
    """Plot detailed Shapley analysis for a single model"""
    
    results = load_shapley_results(model_name)
    if results is None:
        return
    
    group_shap = results['group_shap']
    metrics = results['metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Shapley Value Analysis: {model_label}', fontsize=16, fontweight='bold')
    
    # 1. Distribution of group SHAP values
    ax1 = axes[0, 0]
    ax1.hist(group_shap.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.set_title('Distribution of Group SHAP Values', fontweight='bold')
    ax1.set_xlabel('Group SHAP Value (φ_s)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Importance metrics
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(metrics)), metrics['importance_median'], 
                   yerr=metrics['importance_iqr'], capsize=5, alpha=0.7, edgecolor='black')
    ax2.set_title('Importance: Median |φ_s| with IQR', fontweight='bold')
    ax2.set_xlabel('Source')
    ax2.set_ylabel('Importance')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metrics['source'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metrics['importance_median'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + metrics['importance_iqr'].iloc[i],
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Direction metrics
    ax3 = axes[1, 0]
    colors = ['green' if d > 0 else 'red' for d in metrics['direction_median']]
    bars = ax3.bar(range(len(metrics)), metrics['direction_median'], 
                   color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Direction: Median φ_s', fontweight='bold')
    ax3.set_xlabel('Source')
    ax3.set_ylabel('Direction')
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(metrics['source'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metrics['direction_median'])):
        ax3.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    # 4. Share metrics
    ax4 = axes[1, 1]
    bars = ax4.bar(range(len(metrics)), metrics['share_median'],
                   yerr=metrics['share_iqr'], capsize=5, alpha=0.7, edgecolor='black')
    ax4.set_title('Share: Median Normalized |φ_s| with IQR', fontweight='bold')
    ax4.set_xlabel('Source')
    ax4.set_ylabel('Share')
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(metrics['source'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metrics['share_median'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + metrics['share_iqr'].iloc[i],
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(SHAPLEY_OUTPUT_DIR, f'shapley_{model_name}_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved detailed plot for {model_name}: {output_path}")
    
    plt.close()


def main():
    """Generate all Shapley value plots"""
    
    print("=" * 80)
    print("Generating Shapley Value Plots")
    print("=" * 80)
    
    # Plot individual models
    models = [
        ('tweet', 'Elon Musk Tweets'),
        ('tesla_article', 'NBC Tesla Articles'),
        ('market_news', 'NBC Market News')
    ]
    
    for model_name, model_label in models:
        print(f"\nPlotting {model_label}...")
        plot_individual_model_shapley(model_name, model_label)
    
    # Plot comparison
    print("\nPlotting comparison across models...")
    plot_importance_comparison()
    
    print("\n" + "=" * 80)
    print("Plotting complete!")
    print(f"All plots saved to: {SHAPLEY_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

