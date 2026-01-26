#!/usr/bin/env python3
"""
External Validation with GEO Datasets
======================================

Downloads and validates Pan-Cancer model with independent GEO datasets:
- GSE31210: LUAD (Lung Adenocarcinoma) - 226 samples
- GSE14520: LIHC (Liver Hepatocellular Carcinoma) - 445 samples

This provides true external validation separate from TCGA training data.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import GEOparse
import requests
from io import StringIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


def download_gse31210():
    """
    Download GSE31210 - LUAD dataset (226 tumor samples).
    Platform: GPL570 (Affymetrix Human Genome U133 Plus 2.0 Array)
    """
    print("=" * 70)
    print("Downloading GSE31210 (LUAD - Lung Adenocarcinoma)")
    print("=" * 70)

    output_dir = Path("data/external_validation/GSE31210")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    count_file = output_dir / "count_matrix.csv"
    if count_file.exists():
        print(f"  ✅ Already downloaded: {count_file}")
        return pd.read_csv(count_file, index_col=0)

    print("  📥 Downloading from GEO (this may take a few minutes)...")

    try:
        gse = GEOparse.get_GEO(geo="GSE31210", destdir=str(output_dir), silent=True)

        # Get expression data
        print("  📊 Processing expression data...")

        # GSE31210 has expression data in the GSM samples
        gsms = list(gse.gsms.values())

        # Build expression matrix
        sample_data = {}
        sample_metadata = []

        for gsm in gsms:
            # Get sample characteristics
            chars = gsm.metadata.get('characteristics_ch1', [])

            # Parse characteristics
            sample_info = {'sample_id': gsm.name}
            for char in chars:
                if ':' in char:
                    key, val = char.split(':', 1)
                    sample_info[key.strip().lower()] = val.strip()

            # Only include tumor samples (not normal)
            # GSE31210 contains both tumor and normal
            tissue_type = sample_info.get('histological type', '').lower()
            if 'adenocarcinoma' in tissue_type or 'tumor' in tissue_type.lower():
                # Get expression values
                if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
                    expr = gsm.table.set_index('ID_REF')['VALUE']
                    sample_data[gsm.name] = expr
                    sample_metadata.append(sample_info)

        if not sample_data:
            print("  ⚠️ No tumor samples found in standard format, trying alternative...")
            # Try to get from series matrix
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
                    expr = gsm.table.set_index('ID_REF')['VALUE']
                    sample_data[gsm_name] = expr
                    sample_metadata.append({'sample_id': gsm_name})

        if sample_data:
            expr_matrix = pd.DataFrame(sample_data)
            print(f"  ✅ Expression matrix: {expr_matrix.shape[0]} probes × {expr_matrix.shape[1]} samples")

            # Map probe IDs to gene symbols using platform annotation
            print("  🔄 Mapping probe IDs to gene symbols...")
            expr_matrix = map_probes_to_genes(expr_matrix, gse, 'GPL570')

            # Save
            expr_matrix.to_csv(count_file)
            print(f"  💾 Saved to {count_file}")

            # Save metadata
            meta_df = pd.DataFrame(sample_metadata)
            meta_df.to_csv(output_dir / "metadata.csv", index=False)

            return expr_matrix
        else:
            print("  ❌ Could not extract expression data")
            return None

    except Exception as e:
        print(f"  ❌ Error downloading GSE31210: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_gse14520():
    """
    Download GSE14520 - LIHC dataset (Liver Cancer).
    Platform: GPL3921 (Affymetrix HT Human Genome U133A Array)
    Contains ~445 samples (tumor + adjacent normal).
    """
    print("\n" + "=" * 70)
    print("Downloading GSE14520 (LIHC - Liver Hepatocellular Carcinoma)")
    print("=" * 70)

    output_dir = Path("data/external_validation/GSE14520")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    count_file = output_dir / "count_matrix.csv"
    if count_file.exists():
        print(f"  ✅ Already downloaded: {count_file}")
        return pd.read_csv(count_file, index_col=0)

    print("  📥 Downloading from GEO (this may take a few minutes)...")

    try:
        gse = GEOparse.get_GEO(geo="GSE14520", destdir=str(output_dir), silent=True)

        print("  📊 Processing expression data...")

        gsms = list(gse.gsms.values())
        sample_data = {}
        sample_metadata = []

        for gsm in gsms:
            chars = gsm.metadata.get('characteristics_ch1', [])

            sample_info = {'sample_id': gsm.name}
            for char in chars:
                if ':' in char:
                    key, val = char.split(':', 1)
                    sample_info[key.strip().lower()] = val.strip()

            # Get tissue type - filter for tumor samples
            tissue = sample_info.get('tissue', '').lower()

            # Include HCC tumor samples
            if 'tumor' in tissue or 'hcc' in tissue or 'hepatocellular' in tissue:
                if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
                    expr = gsm.table.set_index('ID_REF')['VALUE']
                    sample_data[gsm.name] = expr
                    sample_metadata.append(sample_info)

        if not sample_data:
            print("  ⚠️ No tumor samples found, including all samples...")
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
                    expr = gsm.table.set_index('ID_REF')['VALUE']
                    sample_data[gsm_name] = expr
                    sample_metadata.append({'sample_id': gsm_name})

        if sample_data:
            expr_matrix = pd.DataFrame(sample_data)
            print(f"  ✅ Expression matrix: {expr_matrix.shape[0]} probes × {expr_matrix.shape[1]} samples")

            # Map probes to genes
            print("  🔄 Mapping probe IDs to gene symbols...")
            platform = list(gse.gpls.keys())[0] if gse.gpls else 'GPL3921'
            expr_matrix = map_probes_to_genes(expr_matrix, gse, platform)

            # Save
            expr_matrix.to_csv(count_file)
            print(f"  💾 Saved to {count_file}")

            meta_df = pd.DataFrame(sample_metadata)
            meta_df.to_csv(output_dir / "metadata.csv", index=False)

            return expr_matrix
        else:
            print("  ❌ Could not extract expression data")
            return None

    except Exception as e:
        print(f"  ❌ Error downloading GSE14520: {e}")
        import traceback
        traceback.print_exc()
        return None


def map_probes_to_genes(expr_matrix, gse, platform_id):
    """Map Affymetrix probe IDs to gene symbols."""

    # Try to get platform annotation
    if platform_id in gse.gpls:
        gpl = gse.gpls[platform_id]
        if hasattr(gpl, 'table') and gpl.table is not None:
            annot = gpl.table

            # Find gene symbol column
            gene_col = None
            for col in ['Gene Symbol', 'gene_symbol', 'GENE_SYMBOL', 'Gene symbol', 'Symbol']:
                if col in annot.columns:
                    gene_col = col
                    break

            if gene_col:
                # Create probe to gene mapping
                probe_to_gene = annot.set_index('ID')[gene_col].to_dict()

                # Map and aggregate by gene
                expr_matrix['gene'] = expr_matrix.index.map(lambda x: probe_to_gene.get(x, ''))
                expr_matrix = expr_matrix[expr_matrix['gene'].notna() & (expr_matrix['gene'] != '')]

                # Handle multiple genes per probe (take first)
                expr_matrix['gene'] = expr_matrix['gene'].str.split(' /// ').str[0]
                expr_matrix['gene'] = expr_matrix['gene'].str.split(' // ').str[0]

                # Aggregate by gene (mean of multiple probes)
                gene_expr = expr_matrix.groupby('gene').mean()

                print(f"  ✅ Mapped to {len(gene_expr)} unique genes")
                return gene_expr

    print("  ⚠️ Could not map probes to genes, using probe IDs")
    return expr_matrix


def detect_data_scale(expr_data):
    """
    Detect if data is log-transformed or linear scale.
    Returns: 'log2', 'linear', 'unknown'
    """
    vals = expr_data.values.flatten()
    vals = vals[~np.isnan(vals)]

    max_val = np.max(vals)
    median_val = np.median(vals)

    # Log2 transformed data typically has max < 20 and median 4-8
    if max_val < 20 and median_val > 2 and median_val < 15:
        return 'log2'
    # Linear/count data has much higher range
    elif max_val > 100:
        return 'linear'
    else:
        return 'unknown'


def normalize_to_tpm_like(expr_data, is_log2=False):
    """
    Normalize expression data to be comparable with TPM-trained model.

    For microarray data (already normalized/log2):
    - If log2: Convert back to linear, then scale like TPM
    - If linear: Just scale like TPM

    The model expects log2(TPM+1) values approximately in range 0-15.
    """
    data = expr_data.copy()

    # Convert to linear if log2
    if is_log2:
        # Data is already log2, use as-is but ensure proper range
        # Shift to have similar distribution as log2(TPM+1)
        pass
    else:
        # Linear scale - apply log2(x+1) transformation
        data = np.log2(data + 1)

    return data


def convert_symbols_to_ensembl(expr_data, model_dir, normalize=True):
    """Convert gene symbols to ENSEMBL IDs to match model."""
    symbol_map_file = model_dir / "symbol_to_model_ensembl.json"

    if not symbol_map_file.exists():
        print("  ⚠️ No symbol_to_model_ensembl.json found")
        return expr_data, 0

    with open(symbol_map_file) as f:
        symbol_to_ensembl = json.load(f)

    # Detect data scale and normalize if needed
    if normalize:
        scale = detect_data_scale(expr_data)
        print(f"  📊 Detected data scale: {scale}")

        if scale == 'log2':
            # Already log2 - use as-is, this is what the model expects
            print("  ✅ Data already log2-transformed, using directly")
        elif scale == 'linear':
            # Apply log2(x+1) transformation
            print("  🔄 Applying log2(x+1) transformation...")
            expr_data = np.log2(expr_data + 1)
        else:
            print("  ⚠️ Unknown scale, proceeding without transformation")

    # Map symbols to ENSEMBL IDs
    new_index = []
    valid_rows = []

    for gene in expr_data.index:
        if gene in symbol_to_ensembl:
            new_index.append(symbol_to_ensembl[gene])
            valid_rows.append(gene)

    if not valid_rows:
        return expr_data, 0

    mapped_data = expr_data.loc[valid_rows].copy()
    mapped_data.index = new_index

    # Handle duplicates (take mean)
    if mapped_data.index.duplicated().any():
        mapped_data = mapped_data.groupby(mapped_data.index).mean()

    return mapped_data, len(valid_rows)


def validate_with_pancancer_model(expr_data, expected_cancer, dataset_name):
    """
    Validate expression data with Pan-Cancer classifier.
    """
    print(f"\n{'=' * 70}")
    print(f"Validating {dataset_name} (Expected: {expected_cancer})")
    print("=" * 70)

    if expr_data is None:
        print("  ❌ No expression data available")
        return None

    # Load classifier
    print("  📊 Loading Pan-Cancer classifier...")
    model_dir = Path("models/rnaseq/pancancer")
    classifier = PanCancerClassifier(model_dir)
    classifier.load()

    # Convert gene symbols to ENSEMBL IDs
    print("  🔄 Converting gene symbols to ENSEMBL IDs...")
    mapped_data, n_mapped = convert_symbols_to_ensembl(expr_data, model_dir)
    print(f"  ✅ Mapped {n_mapped} genes to ENSEMBL IDs")

    # Check gene overlap - use preprocessor.selected_genes
    model_genes = set(classifier.preprocessor.selected_genes)
    data_genes = set(mapped_data.index)
    overlap = model_genes & data_genes

    print(f"  🔍 Gene overlap: {len(overlap)}/{len(model_genes)} model genes ({len(overlap)/len(model_genes)*100:.1f}%)")

    if len(overlap) < 1000:
        print(f"  ⚠️ Low gene overlap ({len(overlap)} < 1000), prediction may be unreliable")

    if len(overlap) < 100:
        print(f"  ❌ Too few genes overlap ({len(overlap)} < 100), skipping prediction")
        return None

    # Use the mapped data for prediction
    expr_data = mapped_data

    # Predict samples (subset for speed if many samples)
    n_samples = min(50, expr_data.shape[1])  # Test up to 50 samples
    test_samples = expr_data.iloc[:, :n_samples]

    print(f"  🧬 Predicting {n_samples} samples...")

    try:
        results = classifier.predict(test_samples)

        # Analyze results
        predictions = {}
        correct = 0

        for r in results:
            pred = r.predicted_cancer
            predictions[pred] = predictions.get(pred, 0) + 1
            if pred == expected_cancer:
                correct += 1

        accuracy = correct / len(results) * 100

        print(f"\n  📊 Results:")
        print(f"  {'Predicted':<10} {'Count':>6} {'%':>8}")
        print(f"  {'-' * 26}")

        for cancer, count in sorted(predictions.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            marker = "✅" if cancer == expected_cancer else ""
            print(f"  {cancer:<10} {count:>6} {pct:>7.1f}% {marker}")

        print(f"\n  🎯 Accuracy for {expected_cancer}: {accuracy:.1f}% ({correct}/{len(results)})")

        # Get confidence distribution
        confidences = [r.confidence for r in results]
        print(f"  📈 Mean confidence: {np.mean(confidences)*100:.1f}% ± {np.std(confidences)*100:.1f}%")

        return {
            'dataset': dataset_name,
            'expected': expected_cancer,
            'n_samples': len(results),
            'accuracy': accuracy,
            'predictions': predictions,
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'gene_overlap': len(overlap),
            'gene_overlap_pct': len(overlap) / len(model_genes) * 100
        }

    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run external validation pipeline."""
    print("=" * 70)
    print("🧬 Pan-Cancer Model External Validation")
    print("   Using Independent GEO Datasets")
    print("=" * 70)

    results = []

    # 1. GSE31210 - LUAD
    luad_data = download_gse31210()
    luad_result = validate_with_pancancer_model(luad_data, 'LUAD', 'GSE31210')
    if luad_result:
        results.append(luad_result)

    # 2. GSE14520 - LIHC
    lihc_data = download_gse14520()
    lihc_result = validate_with_pancancer_model(lihc_data, 'LIHC', 'GSE14520')
    if lihc_result:
        results.append(lihc_result)

    # Summary
    print("\n" + "=" * 70)
    print("📊 External Validation Summary")
    print("=" * 70)

    if results:
        print(f"\n{'Dataset':<12} {'Cancer':<8} {'Accuracy':>10} {'Confidence':>12} {'Gene Overlap':>14}")
        print("-" * 60)

        for r in results:
            print(f"{r['dataset']:<12} {r['expected']:<8} {r['accuracy']:>9.1f}% "
                  f"{r['mean_confidence']*100:>10.1f}% {r['gene_overlap_pct']:>13.1f}%")

        # Overall assessment
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        print(f"\n🎯 Average External Validation Accuracy: {avg_accuracy:.1f}%")

        if avg_accuracy >= 90:
            print("✅ Excellent external generalization")
        elif avg_accuracy >= 80:
            print("✅ Good external generalization")
        elif avg_accuracy >= 70:
            print("⚠️ Moderate external generalization - may need calibration")
        else:
            print("❌ Poor external generalization - model may be TCGA-specific")

        # Save results
        output_file = Path("data/external_validation/validation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    else:
        print("❌ No validation results generated")

    return results


if __name__ == "__main__":
    main()
