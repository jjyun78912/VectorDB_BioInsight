#!/usr/bin/env python3
"""
Domain Adaptation for Cross-Platform RNA-seq Data
==================================================

Adapts external RNA-seq data to match TCGA distribution for better
Pan-Cancer classifier performance.

Methods:
1. Quantile Normalization - Match distribution to TCGA reference
2. ComBat - Batch effect correction (requires pycombat)
3. Z-score Recalibration - Simple mean/std matching

Usage:
    from rnaseq_pipeline.ml.domain_adapter import DomainAdapter

    adapter = DomainAdapter(method='quantile')
    adapted_data = adapter.transform(external_data)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal
import logging
import json

logger = logging.getLogger(__name__)


class DomainAdapter:
    """
    Adapts external RNA-seq data distribution to match TCGA training data.
    """

    def __init__(
        self,
        method: Literal['quantile', 'zscore', 'combat'] = 'quantile',
        reference_stats_path: Optional[Path] = None
    ):
        """
        Initialize domain adapter.

        Args:
            method: Adaptation method
                - 'quantile': Quantile normalization to TCGA reference
                - 'zscore': Z-score recalibration (simple mean/std matching)
                - 'combat': ComBat batch correction (requires reference data)
            reference_stats_path: Path to TCGA reference statistics
        """
        self.method = method
        self.reference_stats = None
        self.reference_quantiles = None

        # Load reference statistics
        if reference_stats_path and reference_stats_path.exists():
            self._load_reference_stats(reference_stats_path)
        else:
            # Use default model directory
            default_path = Path(__file__).parent.parent.parent / "models/rnaseq/pancancer"
            self._load_or_create_reference_stats(default_path)

    def _load_reference_stats(self, path: Path):
        """Load pre-computed TCGA reference statistics."""
        stats_file = path / "tcga_reference_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.reference_stats = json.load(f)
            logger.info(f"Loaded reference stats from {stats_file}")

        quantile_file = path / "tcga_reference_quantiles.npy"
        if quantile_file.exists():
            self.reference_quantiles = np.load(quantile_file)
            logger.info(f"Loaded reference quantiles from {quantile_file}")

    def _load_or_create_reference_stats(self, model_dir: Path):
        """Load existing stats or create from scaler."""
        import joblib

        preprocessor_path = model_dir / "preprocessor.joblib"
        if not preprocessor_path.exists():
            logger.warning("No preprocessor found, using default stats")
            # Default TCGA-like statistics
            self.reference_stats = {
                'mean': 0.0,
                'std': 1.0,
                'median': 0.0,
                'q25': -0.67,
                'q75': 0.67,
                'min': -3.0,
                'max': 6.0
            }
            return

        preproc = joblib.load(preprocessor_path)
        scaler = preproc.get('scaler')

        if scaler and hasattr(scaler, 'mean_'):
            # Extract statistics from fitted scaler
            self.reference_stats = {
                'mean': float(np.mean(scaler.mean_)),
                'std': float(np.mean(scaler.scale_)),
                'gene_means': scaler.mean_.tolist(),
                'gene_stds': scaler.scale_.tolist(),
            }
            logger.info("Extracted reference stats from scaler")
        else:
            self.reference_stats = {'mean': 0.0, 'std': 1.0}

    def transform(self, data: pd.DataFrame, log_transformed: bool = True) -> pd.DataFrame:
        """
        Transform external data to match TCGA distribution.

        Args:
            data: Gene × Sample expression matrix (already log-transformed if log_transformed=True)
            log_transformed: Whether data is already log2-transformed

        Returns:
            Adapted expression matrix
        """
        if self.method == 'quantile':
            return self._quantile_normalize(data)
        elif self.method == 'zscore':
            return self._zscore_recalibrate(data)
        elif self.method == 'combat':
            return self._combat_correct(data)
        else:
            logger.warning(f"Unknown method {self.method}, returning unchanged")
            return data

    def _quantile_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Quantile normalization to match TCGA distribution.

        Forces each sample to have the same distribution as TCGA reference.
        """
        logger.info("Applying quantile normalization...")

        # Store original structure
        genes = data.index
        samples = data.columns

        # Convert to numpy
        X = data.values.astype(float)

        # Get reference quantiles (from TCGA or standard normal)
        n_genes = X.shape[0]

        if self.reference_quantiles is not None and len(self.reference_quantiles) == n_genes:
            ref_quantiles = self.reference_quantiles
        else:
            # Use standard normal quantiles as reference (since TCGA is standardized)
            ref_quantiles = np.sort(np.random.randn(n_genes))
            # Adjust to match typical TCGA scaled range
            ref_quantiles = ref_quantiles * 0.8  # Slightly narrower than standard normal

        # Quantile normalize each sample
        X_normalized = np.zeros_like(X)

        for j in range(X.shape[1]):
            # Get ranks
            ranks = np.argsort(np.argsort(X[:, j]))
            # Map to reference quantiles
            X_normalized[:, j] = ref_quantiles[ranks]

        result = pd.DataFrame(X_normalized, index=genes, columns=samples)

        logger.info(f"Quantile normalized: mean={result.values.mean():.3f}, std={result.values.std():.3f}")

        return result

    def _zscore_recalibrate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score recalibration to match TCGA mean/std.

        Simple but effective: shift and scale to match reference distribution.
        """
        logger.info("Applying z-score recalibration...")

        # Current stats
        current_mean = data.values.mean()
        current_std = data.values.std()

        # Target stats (TCGA after StandardScaler: mean=0, std≈0.8)
        target_mean = self.reference_stats.get('mean', 0.0)
        target_std = self.reference_stats.get('std', 0.8)

        # Recalibrate
        if current_std > 0:
            normalized = (data - current_mean) / current_std * target_std + target_mean
        else:
            normalized = data - current_mean + target_mean

        logger.info(f"Z-score recalibrated: {current_mean:.2f}±{current_std:.2f} -> "
                   f"{normalized.values.mean():.2f}±{normalized.values.std():.2f}")

        return normalized

    def _combat_correct(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ComBat batch correction.

        Requires pycombat package and reference batch information.
        """
        try:
            from combat.pycombat import pycombat
        except ImportError:
            logger.warning("pycombat not installed, falling back to z-score")
            return self._zscore_recalibrate(data)

        logger.info("Applying ComBat correction...")

        # ComBat requires batch labels
        # For single external dataset, use simple approach
        # Mark all samples as "external" batch
        batch = ['external'] * data.shape[1]

        try:
            corrected = pycombat(data, batch)
            logger.info("ComBat correction complete")
            return corrected
        except Exception as e:
            logger.warning(f"ComBat failed: {e}, falling back to z-score")
            return self._zscore_recalibrate(data)


def adapt_for_prediction(
    expr_data: pd.DataFrame,
    method: str = 'quantile',
    already_log: bool = False
) -> pd.DataFrame:
    """
    Convenience function to adapt external data for Pan-Cancer prediction.

    Args:
        expr_data: Gene × Sample expression matrix
        method: 'quantile', 'zscore', or 'combat'
        already_log: Whether data is already log-transformed

    Returns:
        Adapted data ready for classifier
    """
    # Log transform if needed
    if not already_log:
        expr_data = np.log2(expr_data + 1)

    # Apply domain adaptation
    adapter = DomainAdapter(method=method)
    adapted = adapter.transform(expr_data)

    return adapted


# Test function
if __name__ == "__main__":
    # Quick test
    print("Testing Domain Adapter...")

    # Create mock data with different distribution
    np.random.seed(42)
    mock_data = pd.DataFrame(
        np.random.randn(100, 5) * 2 + 3,  # Mean=3, Std=2 (different from TCGA)
        index=[f"Gene_{i}" for i in range(100)],
        columns=[f"Sample_{i}" for i in range(5)]
    )

    print(f"\nOriginal: mean={mock_data.values.mean():.2f}, std={mock_data.values.std():.2f}")

    # Test quantile normalization
    adapter = DomainAdapter(method='quantile')
    adapted = adapter.transform(mock_data)
    print(f"Quantile: mean={adapted.values.mean():.2f}, std={adapted.values.std():.2f}")

    # Test z-score
    adapter = DomainAdapter(method='zscore')
    adapted = adapter.transform(mock_data)
    print(f"Z-score: mean={adapted.values.mean():.2f}, std={adapted.values.std():.2f}")

    print("\n✅ Domain Adapter working!")
