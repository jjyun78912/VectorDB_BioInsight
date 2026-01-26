#!/usr/bin/env python3
"""
Update Pan-Cancer Model with Reference Distribution for Batch Correction

This script updates the existing preprocessor with TCGA reference distribution
to enable batch correction for external validation datasets.

Usage:
    python scripts/update_model_reference_distribution.py

Author: BioInsight AI
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_preprocessor_with_reference(model_dir: str = "models/rnaseq/pancancer"):
    """
    Update preprocessor with reference distribution from TCGA training data.

    If TCGA data is not available, use a synthetic reference based on model features.
    """
    model_dir = Path(model_dir)
    preprocessor_path = model_dir / "preprocessor.joblib"

    if not preprocessor_path.exists():
        logger.error(f"Preprocessor not found: {preprocessor_path}")
        return False

    # Load preprocessor
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    save_dict = joblib.load(preprocessor_path)

    # Check if already has reference distribution
    if save_dict.get('reference_means') is not None:
        logger.info("Preprocessor already has reference distribution. Skipping update.")
        return True

    selected_genes = save_dict['selected_genes']
    logger.info(f"Selected genes: {len(selected_genes)}")

    # Try to load TCGA training data
    tcga_data_path = Path("data/tcga_pancancer")
    if tcga_data_path.exists():
        logger.info("Loading TCGA training data for reference distribution...")
        try:
            counts = pd.read_csv(tcga_data_path / "pancancer_counts.csv", index_col=0)

            # Normalize and log transform (same as training)
            lib_sizes = counts.sum(axis=0)
            normalized = counts * 1e6 / lib_sizes
            transformed = np.log2(normalized + 1)

            # Filter to selected genes
            common_genes = [g for g in selected_genes if g in transformed.index]
            selected_data = transformed.loc[common_genes]

            # Calculate reference statistics
            reference_means = selected_data.mean(axis=1)
            reference_stds = selected_data.std(axis=1)

            # Quantile distribution
            all_values = selected_data.values.flatten()
            all_values = all_values[~np.isnan(all_values)]
            reference_quantiles = np.sort(all_values)

            # Sample 1000 quantiles for efficiency
            if len(reference_quantiles) > 1000:
                indices = np.linspace(0, len(reference_quantiles) - 1, 1000).astype(int)
                reference_quantiles = reference_quantiles[indices]

            logger.info(f"Reference distribution: {len(common_genes)} genes, {len(reference_quantiles)} quantiles")

        except Exception as e:
            logger.warning(f"Failed to load TCGA data: {e}")
            logger.info("Using synthetic reference distribution...")
            reference_means, reference_stds, reference_quantiles = _create_synthetic_reference(selected_genes, save_dict['scaler'])
    else:
        logger.info("TCGA data not found. Using synthetic reference distribution...")
        reference_means, reference_stds, reference_quantiles = _create_synthetic_reference(selected_genes, save_dict['scaler'])

    # Update save_dict
    save_dict['reference_means'] = reference_means
    save_dict['reference_stds'] = reference_stds
    save_dict['reference_quantiles'] = reference_quantiles

    # Save updated preprocessor
    joblib.dump(save_dict, preprocessor_path)
    logger.info(f"Updated preprocessor saved to {preprocessor_path}")

    return True


def _create_synthetic_reference(selected_genes, scaler):
    """
    Create synthetic reference distribution based on StandardScaler parameters.

    The scaler was fitted on TCGA data, so we can reverse-engineer approximate
    reference statistics from its mean_ and scale_ attributes.
    """
    # StandardScaler stores: mean_ and scale_ (= std_)
    # These are per-feature (gene) statistics from the training data

    if scaler is None:
        logger.warning("No scaler found, using default reference")
        n_genes = len(selected_genes)
        reference_means = pd.Series(np.zeros(n_genes), index=selected_genes)
        reference_stds = pd.Series(np.ones(n_genes), index=selected_genes)
        reference_quantiles = np.linspace(-3, 15, 1000)  # Typical log2 CPM range
        return reference_means, reference_stds, reference_quantiles

    # Recover pre-standardization statistics
    # StandardScaler: z = (x - mean) / scale
    # So original mean = scaler.mean_, original std = scaler.scale_

    original_means = scaler.mean_
    original_stds = scaler.scale_

    reference_means = pd.Series(original_means, index=selected_genes)
    reference_stds = pd.Series(original_stds, index=selected_genes)

    # Create synthetic quantile distribution
    # Assume approximately normal distribution with some skewness typical of log-transformed expression
    mean_of_means = np.mean(original_means)
    std_of_means = np.std(original_means)
    combined_std = np.sqrt(np.mean(original_stds**2) + std_of_means**2)

    # Generate quantiles spanning typical expression range
    reference_quantiles = np.linspace(
        mean_of_means - 3 * combined_std,
        mean_of_means + 3 * combined_std,
        1000
    )

    logger.info(f"Synthetic reference: mean range [{original_means.min():.2f}, {original_means.max():.2f}], "
                f"std range [{original_stds.min():.2f}, {original_stds.max():.2f}]")

    return reference_means, reference_stds, reference_quantiles


def test_batch_correction(model_dir: str = "models/rnaseq/pancancer"):
    """Test batch correction with a sample external dataset."""
    from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier

    model_dir = Path(model_dir)

    # Load test data
    test_data_path = Path("data/GSE81089_lung_cancer_input/count_matrix.csv")
    if not test_data_path.exists():
        logger.warning(f"Test data not found: {test_data_path}")
        return

    logger.info(f"\nTesting batch correction with {test_data_path}")

    counts = pd.read_csv(test_data_path, index_col=0)
    metadata = pd.read_csv(test_data_path.parent / "metadata.csv")

    # Filter tumor samples
    tumor_samples = metadata[metadata['condition'] == 'tumor']['sample_id'].tolist()
    tumor_counts = counts[[c for c in counts.columns if c in tumor_samples]]

    logger.info(f"Tumor samples: {tumor_counts.shape[1]}")

    # Test with and without batch correction
    classifier = PanCancerClassifier(str(model_dir))

    # Without batch correction
    logger.info("\n--- Without Batch Correction ---")
    results_no_bc = classifier.predict(
        tumor_counts,
        apply_batch_correction=False,
        use_secondary_validation=False
    )

    # Count predictions
    pred_counts_no_bc = {}
    for r in results_no_bc:
        cancer = r.predicted_cancer
        pred_counts_no_bc[cancer] = pred_counts_no_bc.get(cancer, 0) + 1

    logger.info(f"Predictions: {pred_counts_no_bc}")
    luad_count_no_bc = pred_counts_no_bc.get('LUAD', 0)
    lusc_count_no_bc = pred_counts_no_bc.get('LUSC', 0)
    lung_total_no_bc = luad_count_no_bc + lusc_count_no_bc
    logger.info(f"Lung cancer (LUAD+LUSC): {lung_total_no_bc}/{len(results_no_bc)} ({lung_total_no_bc/len(results_no_bc)*100:.1f}%)")

    # With batch correction
    logger.info("\n--- With Batch Correction ---")
    results_bc = classifier.predict(
        tumor_counts,
        apply_batch_correction=True,
        use_secondary_validation=False
    )

    pred_counts_bc = {}
    for r in results_bc:
        cancer = r.predicted_cancer
        pred_counts_bc[cancer] = pred_counts_bc.get(cancer, 0) + 1

    logger.info(f"Predictions: {pred_counts_bc}")
    luad_count_bc = pred_counts_bc.get('LUAD', 0)
    lusc_count_bc = pred_counts_bc.get('LUSC', 0)
    lung_total_bc = luad_count_bc + lusc_count_bc
    logger.info(f"Lung cancer (LUAD+LUSC): {lung_total_bc}/{len(results_bc)} ({lung_total_bc/len(results_bc)*100:.1f}%)")

    # With quantile normalization
    logger.info("\n--- With Quantile Normalization ---")
    results_qn = classifier.predict(
        tumor_counts,
        apply_batch_correction=True,
        use_quantile_norm=True,
        use_secondary_validation=False
    )

    pred_counts_qn = {}
    for r in results_qn:
        cancer = r.predicted_cancer
        pred_counts_qn[cancer] = pred_counts_qn.get(cancer, 0) + 1

    logger.info(f"Predictions: {pred_counts_qn}")
    luad_count_qn = pred_counts_qn.get('LUAD', 0)
    lusc_count_qn = pred_counts_qn.get('LUSC', 0)
    lung_total_qn = luad_count_qn + lusc_count_qn
    logger.info(f"Lung cancer (LUAD+LUSC): {lung_total_qn}/{len(results_qn)} ({lung_total_qn/len(results_qn)*100:.1f}%)")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Summary: Lung Cancer (LUAD+LUSC) Prediction Accuracy")
    logger.info(f"{'='*60}")
    logger.info(f"Without batch correction:    {lung_total_no_bc}/{len(results_no_bc)} ({lung_total_no_bc/len(results_no_bc)*100:.1f}%)")
    logger.info(f"With batch correction:       {lung_total_bc}/{len(results_bc)} ({lung_total_bc/len(results_bc)*100:.1f}%)")
    logger.info(f"With quantile normalization: {lung_total_qn}/{len(results_qn)} ({lung_total_qn/len(results_qn)*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update Pan-Cancer model with reference distribution")
    parser.add_argument('--model-dir', type=str, default='models/rnaseq/pancancer',
                       help='Model directory')
    parser.add_argument('--test', action='store_true',
                       help='Run batch correction test after update')

    args = parser.parse_args()

    # Update preprocessor
    success = update_preprocessor_with_reference(args.model_dir)

    if success and args.test:
        test_batch_correction(args.model_dir)
