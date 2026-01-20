#!/usr/bin/env python3
"""
SHAP Analysis Script for Pan-Cancer Model

Generates interpretable SHAP visualizations with Gene Symbol mapping.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Error: SHAP not available. Install with: pip install shap")
    sys.exit(1)

from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


def load_gene_mapping(model_dir: Path) -> dict:
    """Load Ensembl to Gene Symbol mapping."""
    mapping_path = model_dir / "ensembl_to_gene_symbol.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            return json.load(f)
    return {}


def run_shap_analysis(model_dir: str = "models/rnaseq/pancancer", n_samples: int = 200):
    """
    Run comprehensive SHAP analysis.

    Args:
        model_dir: Path to model directory
        n_samples: Number of samples for SHAP
    """
    model_path = Path(model_dir)
    output_dir = model_path / "evaluation" / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHAP Analysis for Pan-Cancer Classifier")
    print("=" * 60)

    # Load classifier
    print("\n[1] Loading model...")
    classifier = PanCancerClassifier(model_dir=model_dir)
    classifier.load()

    # Load gene mapping
    print("[2] Loading gene mappings...")
    ensembl_to_symbol = load_gene_mapping(model_path)
    print(f"  Loaded {len(ensembl_to_symbol)} gene mappings")

    # Load training results for class names
    with open(model_path / "training_results.json") as f:
        results = json.load(f)
    class_names = results.get("class_names", [])
    print(f"  Classes: {class_names}")

    # Get preprocessor feature names
    feature_names = list(classifier.preprocessor.selected_genes)
    print(f"  Features: {len(feature_names)} genes")

    # Map to gene symbols
    symbol_names = []
    for feat in feature_names:
        # Handle versioned Ensembl IDs
        base_id = feat.split('.')[0]
        symbol = ensembl_to_symbol.get(feat, ensembl_to_symbol.get(base_id, feat))
        symbol_names.append(symbol)

    # Count how many were mapped
    mapped_count = sum(1 for s in symbol_names if not s.startswith('ENSG'))
    print(f"  Mapped to symbols: {mapped_count}/{len(symbol_names)}")

    # Create sample data for SHAP
    print(f"\n[3] Generating {n_samples} samples for SHAP analysis...")

    # Generate random samples from preprocessor stats
    np.random.seed(42)
    n_features = len(feature_names)

    # Use preprocessor scaler stats (StandardScaler has mean_ and scale_)
    scaler = classifier.preprocessor.scaler
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        mean = scaler.mean_
        std = scaler.scale_
    else:
        # Fallback: use zeros and ones if scaler not fitted
        mean = np.zeros(n_features)
        std = np.ones(n_features)

    # Generate samples around the training distribution
    X_sample = np.random.randn(n_samples, n_features) * std + mean

    # Get CatBoost model
    model = classifier.ensemble.models['catboost']

    print("\n[4] Computing SHAP values...")
    print("  This may take a few minutes...")

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For multi-class, shap_values is a list
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        print(f"  Computed SHAP values for {n_classes} classes")
        print(f"  SHAP values shape per class: {shap_values[0].shape}")

        # Average absolute SHAP across all classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        print(f"  SHAP values shape: {shap_values.shape}")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Flatten if needed (handle edge cases)
    mean_abs_shap = np.array(mean_abs_shap).flatten()
    print(f"  Mean abs SHAP shape: {mean_abs_shap.shape}")

    # Get top features
    sorted_indices = np.argsort(mean_abs_shap)[::-1][:30]
    top_idx = sorted_indices.flatten().tolist()

    print("\n[5] Top 30 Important Genes (SHAP):")
    print("-" * 50)

    top_genes = []
    for i, idx in enumerate(top_idx):
        gene_id = feature_names[idx]
        gene_symbol = symbol_names[idx]
        importance = float(mean_abs_shap[idx])
        top_genes.append({
            'rank': i + 1,
            'ensembl_id': gene_id,
            'gene_symbol': gene_symbol,
            'shap_importance': importance
        })
        print(f"  {i+1:2}. {gene_symbol:<15} ({gene_id:<20}) - {importance:.4f}")

    # Save top genes
    with open(output_dir / "top_shap_genes.json", 'w') as f:
        json.dump(top_genes, f, indent=2)
    print(f"\n  Saved: {output_dir / 'top_shap_genes.json'}")

    # Create visualizations
    print("\n[6] Generating visualizations...")

    # 1. Summary bar plot with gene symbols
    fig, ax = plt.subplots(figsize=(12, 10))
    top20_idx = top_idx[:20]  # Already a list from .tolist()
    y_pos = np.arange(20)

    bars = ax.barh(y_pos, [mean_abs_shap[i] for i in top20_idx][::-1], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([symbol_names[i] for i in top20_idx][::-1])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top 20 Important Genes (SHAP Analysis)')

    # Add values on bars
    for i, (bar, idx) in enumerate(zip(bars, top20_idx[::-1])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean_abs_shap[idx]:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance_genes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_importance_genes.png")

    # 2. Per-class SHAP importance
    if isinstance(shap_values, list):
        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        axes = axes.flatten()

        for class_idx, (class_name, ax) in enumerate(zip(class_names, axes)):
            if class_idx >= len(shap_values):
                break

            class_importance = np.abs(shap_values[class_idx]).mean(axis=0)
            class_top_idx = np.argsort(class_importance)[::-1][:10].tolist()

            ax.barh(range(10), [class_importance[i] for i in class_top_idx][::-1], color=f'C{class_idx % 10}')
            ax.set_yticks(range(10))
            ax.set_yticklabels([symbol_names[i] for i in class_top_idx][::-1], fontsize=8)
            ax.set_title(class_name, fontsize=10)
            ax.set_xlabel('Mean |SHAP|', fontsize=8)

        # Hide extra subplots
        for idx in range(len(class_names), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Per-Cancer Type SHAP Feature Importance', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_per_class.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: shap_per_class.png")

    # 3. SHAP beeswarm plot
    plt.figure(figsize=(12, 10))

    # Use first class for beeswarm or average
    if isinstance(shap_values, list):
        # Create combined summary
        shap_df = pd.DataFrame(
            shap_values[0],  # Use first class
            columns=symbol_names
        )

        shap.summary_plot(
            shap_values[0], X_sample,
            feature_names=symbol_names,
            max_display=20,
            show=False
        )
    else:
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=symbol_names,
            max_display=20,
            show=False
        )

    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_summary_beeswarm.png")

    # 4. Feature correlation heatmap (top genes)
    top10_idx = top_idx[:10]  # Already a list
    top10_data = X_sample[:, top10_idx]
    top10_names = [symbol_names[i] for i in top10_idx]

    corr_matrix = np.corrcoef(top10_data.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(top10_names, rotation=45, ha='right')
    ax.set_yticklabels(top10_names)

    # Add correlation values
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8)

    plt.colorbar(im, label='Correlation')
    plt.title('Correlation Matrix of Top 10 SHAP Genes')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_genes_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: top_genes_correlation.png")

    # Generate summary report
    print("\n[7] Generating SHAP summary report...")

    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': len(class_names),
        'class_names': class_names,
        'top_30_genes': top_genes,
        'output_files': [
            'shap_importance_genes.png',
            'shap_per_class.png',
            'shap_summary_beeswarm.png',
            'top_genes_correlation.png',
            'top_shap_genes.json'
        ]
    }

    with open(output_dir / 'shap_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n" + "=" * 60)
    print("SHAP Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SHAP Analysis for Pan-Cancer Model')
    parser.add_argument('--model-dir', default='models/rnaseq/pancancer',
                       help='Path to model directory')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of samples for SHAP')

    args = parser.parse_args()
    run_shap_analysis(args.model_dir, args.n_samples)
