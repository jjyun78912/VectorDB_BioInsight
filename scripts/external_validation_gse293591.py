#!/usr/bin/env python3
"""
External Validation with GSE293591 RNA-seq Dataset
===================================================

This script validates the Pan-Cancer classifier with an independent
RNA-seq dataset (GSE293591) containing 365 samples from multiple cancer types.

Dataset: GSE293591 (TPM format, Illumina NovaSeq 6000)
- Breast cancer: 201 samples -> BRCA
- Colorectal Adenocarcinoma: 27 samples -> COAD
- Lung Adenocarcinoma: 20 samples -> LUAD
- Gastric Adenocarcinoma: 17 samples -> STAD
- Squamous Cell Carcinoma of Lung: 14 samples -> LUSC
- Pancreatic Adenocarcinoma: 10 samples -> PAAD
- And more...
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


# Mapping from GSE293591 diagnosis to TCGA cancer codes
DIAGNOSIS_TO_TCGA = {
    'Breast cancer': 'BRCA',
    'Colorectal Adenocarcinoma': 'COAD',
    'Lung Adenocarcinoma': 'LUAD',
    'Gastric Adenocarcinoma': 'STAD',
    'Squamous Cell Carcinoma of Lung': 'LUSC',
    'Pancreatic Adenocarcinoma': 'PAAD',
    'Liver carcinoma and Cholangiocarcinoma': 'LIHC',
    'Prostate adenocarcinoma': 'PRAD',
    'Renal cell carcinoma': 'KIRC',
    'Esophageal Adenocarcinoma': 'STAD',  # Similar to gastric
    'Urothelial_cancer': 'BLCA',
    'High grade serous carcinoma': 'OV',
    'Endometrial carcinoma': 'UCEC',
    'Adrenocortical carcinoma': 'UNKNOWN',  # Not in our 17 types
    'Neuroendocrine neoplasm': 'UNKNOWN',  # Not in our 17 types
    'Squamous Cell Carcinoma (other than Lung)': 'HNSC',  # Could be HNSC
    'Salivary Adenocarcinoma': 'UNKNOWN',  # Not in our 17 types
}


def load_gse293591_data():
    """Load GSE293591 expression data and metadata."""
    # Use absolute path
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data/external_validation/GSE293591"

    # Load TPM data
    tpm_file = data_dir / "GSE293591_TPM_all_samples.tsv"
    print(f"Loading TPM data from {tpm_file}...")
    expr_data = pd.read_csv(tpm_file, sep='\t', index_col=0)
    print(f"  Expression matrix: {expr_data.shape[0]} genes × {expr_data.shape[1]} samples")

    # Load metadata
    meta_file = data_dir / "metadata.csv"
    metadata = pd.read_csv(meta_file)
    print(f"  Metadata: {len(metadata)} samples")

    return expr_data, metadata


def map_symbols_to_ensembl(expr_data, model_dir):
    """Map gene symbols to ENSEMBL IDs."""
    symbol_map_file = model_dir / "symbol_to_model_ensembl.json"

    with open(symbol_map_file) as f:
        symbol_to_ensembl = json.load(f)

    # Map symbols
    mapped_genes = []
    mapped_indices = []

    for gene in expr_data.index:
        if gene in symbol_to_ensembl:
            mapped_genes.append(symbol_to_ensembl[gene])
            mapped_indices.append(gene)

    mapped_data = expr_data.loc[mapped_indices].copy()
    mapped_data.index = mapped_genes

    # Handle duplicates
    if mapped_data.index.duplicated().any():
        mapped_data = mapped_data.groupby(mapped_data.index).mean()

    return mapped_data, len(mapped_indices)


def run_external_validation():
    """Run external validation with GSE293591."""
    print("=" * 80)
    print("External Validation: GSE293591 RNA-seq Dataset")
    print("=" * 80)

    # Load data
    expr_data, metadata = load_gse293591_data()

    # Load classifier
    print("\nLoading Pan-Cancer classifier...")
    script_dir = Path(__file__).parent.parent
    model_dir = script_dir / "models/rnaseq/pancancer"
    classifier = PanCancerClassifier(model_dir)
    classifier.load()

    # Map gene symbols to ENSEMBL
    print("\nMapping gene symbols to ENSEMBL IDs...")
    mapped_data, n_mapped = map_symbols_to_ensembl(expr_data, model_dir)
    print(f"  Mapped {n_mapped} genes")

    # Check overlap
    model_genes = set(classifier.preprocessor.selected_genes)
    overlap = model_genes & set(mapped_data.index)
    print(f"  Gene overlap: {len(overlap)}/{len(model_genes)} ({len(overlap)/len(model_genes)*100:.1f}%)")

    # Predict by cancer type
    print("\n" + "=" * 80)
    print("Running Predictions by Cancer Type")
    print("=" * 80)

    results_by_cancer = defaultdict(list)
    all_results = []

    # Create sample to TCGA code mapping
    sample_to_tcga = {}
    for _, row in metadata.iterrows():
        sample_id = row['sample_id']
        diagnosis = row['diagnosis']
        tcga_code = DIAGNOSIS_TO_TCGA.get(diagnosis, 'UNKNOWN')
        sample_to_tcga[sample_id] = (tcga_code, diagnosis)

    # Match sample IDs (Sample_001, Sample_002, etc.)
    sample_ids = []
    for col in mapped_data.columns:
        # Column names are Sample_001, Sample_002, etc.
        gsm_id = f"GSM{col.replace('Sample_', '')}"  # Try GSM format
        if gsm_id not in sample_to_tcga:
            # Try direct match
            gsm_id = col
        sample_ids.append(col)

    # Get unique sample IDs from metadata
    meta_sample_ids = set(metadata['sample_id'].values)
    print(f"\nMetadata sample IDs: {list(meta_sample_ids)[:5]}...")
    print(f"Expression columns: {list(mapped_data.columns)[:5]}...")

    # Create mapping from expression column to metadata
    # Columns are Sample_001, Sample_002, etc.
    # Metadata sample_id might be GSM... or Sample_...
    col_to_meta = {}
    for col in mapped_data.columns:
        # Try direct match
        if col in meta_sample_ids:
            col_to_meta[col] = col
        else:
            # Try to find matching sample number
            for meta_id in meta_sample_ids:
                if col.replace('Sample_', '') in meta_id or meta_id.replace('GSM', '') in col:
                    col_to_meta[col] = meta_id
                    break

    # If mapping failed, assume order matches
    if not col_to_meta:
        print("  Creating mapping based on order...")
        for i, col in enumerate(mapped_data.columns):
            if i < len(metadata):
                col_to_meta[col] = metadata.iloc[i]['sample_id']

    # Get predictions for each cancer type
    cancer_samples = defaultdict(list)

    for col in mapped_data.columns:
        meta_id = col_to_meta.get(col, col)
        meta_row = metadata[metadata['sample_id'] == meta_id]

        if len(meta_row) > 0:
            diagnosis = meta_row.iloc[0]['diagnosis']
            tcga_code = DIAGNOSIS_TO_TCGA.get(diagnosis, 'UNKNOWN')
            cancer_samples[tcga_code].append((col, diagnosis))

    print(f"\nSamples by cancer type:")
    for cancer, samples in sorted(cancer_samples.items(), key=lambda x: -len(x[1])):
        print(f"  {cancer}: {len(samples)} samples")

    # Run predictions
    for tcga_code, samples in sorted(cancer_samples.items()):
        if tcga_code == 'UNKNOWN':
            continue

        sample_cols = [s[0] for s in samples]
        sample_data = mapped_data[sample_cols]

        print(f"\n{'='*60}")
        print(f"Predicting {tcga_code} ({len(samples)} samples)")
        print("=" * 60)

        try:
            # Apply log2(TPM+1) transformation (model expects this)
            sample_data_log = np.log2(sample_data + 1)

            predictions = classifier.predict(sample_data_log)

            correct = 0
            pred_counts = defaultdict(int)

            for pred in predictions:
                pred_cancer = pred.predicted_cancer
                pred_counts[pred_cancer] += 1

                is_correct = pred_cancer == tcga_code
                if is_correct:
                    correct += 1

                all_results.append({
                    'true_cancer': tcga_code,
                    'predicted': pred_cancer,
                    'confidence': pred.confidence,
                    'correct': is_correct
                })

            accuracy = correct / len(predictions) * 100

            print(f"\nPrediction distribution:")
            for pred, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
                marker = "✅" if pred == tcga_code else ""
                print(f"  {pred}: {count} ({count/len(predictions)*100:.1f}%) {marker}")

            print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(predictions)})")

            results_by_cancer[tcga_code] = {
                'n_samples': len(predictions),
                'accuracy': accuracy,
                'predictions': dict(pred_counts),
                'correct': correct
            }

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("EXTERNAL VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Cancer':<10} {'Samples':>8} {'Accuracy':>10} {'Top Prediction':<15}")
    print("-" * 50)

    total_correct = 0
    total_samples = 0

    for cancer, result in sorted(results_by_cancer.items()):
        n = result['n_samples']
        acc = result['accuracy']
        top_pred = max(result['predictions'].items(), key=lambda x: x[1])[0]

        marker = "✅" if acc >= 70 else "❌" if acc < 30 else "⚠️"
        print(f"{cancer:<10} {n:>8} {acc:>9.1f}% {top_pred:<15} {marker}")

        total_correct += result['correct']
        total_samples += n

    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0

    print("-" * 50)
    print(f"{'OVERALL':<10} {total_samples:>8} {overall_accuracy:>9.1f}%")

    # Assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if overall_accuracy >= 80:
        print("✅ EXCELLENT: Model generalizes well to external RNA-seq data")
    elif overall_accuracy >= 60:
        print("✅ GOOD: Model shows reasonable external generalization")
    elif overall_accuracy >= 40:
        print("⚠️ MODERATE: Model may need calibration for external data")
    else:
        print("❌ POOR: Model may be TCGA-specific, consider domain adaptation")

    # Save results
    script_dir = Path(__file__).parent.parent
    output_file = script_dir / "data/external_validation/GSE293591/validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'by_cancer': results_by_cancer,
            'gene_overlap_pct': len(overlap) / len(model_genes) * 100
        }, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

    return results_by_cancer, overall_accuracy


if __name__ == "__main__":
    run_external_validation()
