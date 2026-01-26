#!/usr/bin/env python3
"""
Test Single-Cell ML Prediction with Skip Options
=================================================

Tests the updated agent5_cnv_ml.py that now properly handles
raw counts vs normalized data with appropriate skip options.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import scanpy as sc
from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


def test_singlecell_prediction():
    """Test single-cell cancer prediction with proper skip options."""
    print("=" * 70)
    print("Single-Cell Cancer Prediction Test (with Skip Options)")
    print("=" * 70)

    # Load data
    data_path = Path("/Users/admin/VectorDB_BioInsight/rnaseq_test_results/singlecell_real_cancer/input/breast_cancer_real.h5ad")
    print(f"\nLoading: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Check available layers
    print(f"\n  Available layers: {list(adata.layers.keys()) if adata.layers else 'None'}")
    print(f"  adata.X type: {type(adata.X)}")
    print(f"  adata.X shape: {adata.X.shape}")

    # Load classifier
    model_dir = Path("/Users/admin/VectorDB_BioInsight/models/rnaseq/pancancer")
    classifier = PanCancerClassifier(model_dir=model_dir)

    # Test 1: Using adata.X (normalized data) with skip options
    print("\n" + "=" * 60)
    print("Test 1: Using adata.X (normalized data) with SKIP options")
    print("=" * 60)

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Create pseudobulk (mean)
    pseudobulk_norm = np.mean(X, axis=0)
    pseudobulk_df_norm = pd.DataFrame(
        pseudobulk_norm,
        index=adata.var_names,
        columns=["pseudobulk"]
    )

    print(f"\n  Pseudobulk stats (normalized): min={pseudobulk_norm.min():.3f}, max={pseudobulk_norm.max():.3f}, mean={pseudobulk_norm.mean():.3f}")

    # Predict with skip options (for normalized data)
    try:
        results = classifier.predict(
            pseudobulk_df_norm,
            sample_ids=["pseudobulk"],
            skip_normalization=True,
            skip_log=True
        )

        if results:
            result = results[0]
            print(f"\n  Predicted: {result.predicted_cancer} (confidence: {result.confidence*100:.1f}%)")
            print(f"  Is Unknown: {result.is_unknown}")
            print(f"\n  Top 5 predictions:")
            for i, pred in enumerate(result.top_k_predictions[:5]):
                mark = "★" if pred.cancer_type == "BRCA" else ""
                print(f"    {i+1}. {pred.cancer_type}: {pred.probability*100:.1f}% {mark}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 2: Using raw counts if available
    print("\n" + "=" * 60)
    print("Test 2: Using raw counts (if available) WITHOUT skip options")
    print("=" * 60)

    if 'counts' in adata.layers:
        X_raw = adata.layers['counts']
        if hasattr(X_raw, 'toarray'):
            X_raw = X_raw.toarray()

        # Create pseudobulk (sum for raw counts)
        pseudobulk_raw = np.sum(X_raw, axis=0)
        pseudobulk_df_raw = pd.DataFrame(
            pseudobulk_raw,
            index=adata.var_names,
            columns=["pseudobulk"]
        )

        print(f"\n  Pseudobulk stats (raw counts): min={pseudobulk_raw.min():.0f}, max={pseudobulk_raw.max():.0f}, mean={pseudobulk_raw.mean():.1f}")

        # Predict without skip options (model will apply CPM + log2)
        try:
            results_raw = classifier.predict(
                pseudobulk_df_raw,
                sample_ids=["pseudobulk"],
                skip_normalization=False,
                skip_log=False
            )

            if results_raw:
                result_raw = results_raw[0]
                print(f"\n  Predicted: {result_raw.predicted_cancer} (confidence: {result_raw.confidence*100:.1f}%)")
                print(f"  Is Unknown: {result_raw.is_unknown}")
                print(f"\n  Top 5 predictions:")
                for i, pred in enumerate(result_raw.top_k_predictions[:5]):
                    mark = "★" if pred.cancer_type == "BRCA" else ""
                    print(f"    {i+1}. {pred.cancer_type}: {pred.probability*100:.1f}% {mark}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  No raw counts layer available. Skipping Test 2.")

        # Try to store raw counts for future tests
        print("\n  Attempting to create raw counts from adata.raw...")
        if hasattr(adata, 'raw') and adata.raw is not None:
            X_raw = adata.raw.X
            if hasattr(X_raw, 'toarray'):
                X_raw = X_raw.toarray()

            # Check if raw looks like counts (integers or close to integers)
            is_counts = np.allclose(X_raw, np.round(X_raw))
            if is_counts:
                print(f"  Found raw counts in adata.raw!")
                pseudobulk_raw = np.sum(X_raw, axis=0)
                pseudobulk_df_raw = pd.DataFrame(
                    pseudobulk_raw,
                    index=adata.raw.var_names,
                    columns=["pseudobulk"]
                )
                print(f"  Pseudobulk stats (from raw): min={pseudobulk_raw.min():.0f}, max={pseudobulk_raw.max():.0f}")

                try:
                    results_raw = classifier.predict(
                        pseudobulk_df_raw,
                        sample_ids=["pseudobulk"],
                        skip_normalization=False,
                        skip_log=False
                    )

                    if results_raw:
                        result_raw = results_raw[0]
                        print(f"\n  Predicted: {result_raw.predicted_cancer} (confidence: {result_raw.confidence*100:.1f}%)")
                        print(f"\n  Top 5 predictions:")
                        for i, pred in enumerate(result_raw.top_k_predictions[:5]):
                            mark = "★" if pred.cancer_type == "BRCA" else ""
                            print(f"    {i+1}. {pred.cancer_type}: {pred.probability*100:.1f}% {mark}")
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print(f"  adata.raw doesn't appear to contain integer counts")
        else:
            print("  No adata.raw available either")

    # Test 3: Compare with NO skip options (old behavior)
    print("\n" + "=" * 60)
    print("Test 3: Using adata.X WITHOUT skip options (OLD behavior)")
    print("=" * 60)

    try:
        results_old = classifier.predict(
            pseudobulk_df_norm,
            sample_ids=["pseudobulk"],
            skip_normalization=False,
            skip_log=False
        )

        if results_old:
            result_old = results_old[0]
            print(f"\n  Predicted: {result_old.predicted_cancer} (confidence: {result_old.confidence*100:.1f}%)")
            print(f"  Is Unknown: {result_old.is_unknown}")
            print(f"\n  Top 5 predictions:")
            for i, pred in enumerate(result_old.top_k_predictions[:5]):
                mark = "★" if pred.cancer_type == "BRCA" else ""
                print(f"    {i+1}. {pred.cancer_type}: {pred.probability*100:.1f}% {mark}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Expected behavior:
- Test 1 (normalized + skip): Should work well if data is already normalized
- Test 2 (raw counts): Should work well with proper CPM + log2 normalization
- Test 3 (normalized + no skip): Old behavior - double normalization issue

For scRNA-seq breast cancer data:
- Ground truth: BRCA
- Good result: BRCA in top 3 with reasonable probability
- Poor result: UNKNOWN or completely wrong cancer type
""")


if __name__ == "__main__":
    test_singlecell_prediction()
