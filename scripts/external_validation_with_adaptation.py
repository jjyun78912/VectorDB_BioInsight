#!/usr/bin/env python3
"""
External Validation with Domain Adaptation
===========================================

Tests Pan-Cancer classifier on GSE293591 with domain adaptation methods.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier
from rnaseq_pipeline.ml.domain_adapter import DomainAdapter, adapt_for_prediction


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
    'Esophageal Adenocarcinoma': 'STAD',
    'Urothelial_cancer': 'BLCA',
    'High grade serous carcinoma': 'OV',
    'Endometrial carcinoma': 'UCEC',
    'Squamous Cell Carcinoma (other than Lung)': 'HNSC',
}


def load_gse293591():
    """Load GSE293591 data and metadata."""
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data/external_validation/GSE293591"

    expr = pd.read_csv(data_dir / "GSE293591_TPM_all_samples.tsv", sep='\t', index_col=0)
    meta = pd.read_csv(data_dir / "metadata.csv")

    return expr, meta


def map_to_ensembl(expr_data, model_dir):
    """Map gene symbols to ENSEMBL IDs."""
    with open(model_dir / "symbol_to_model_ensembl.json") as f:
        symbol_map = json.load(f)

    mapped = []
    indices = []
    for gene in expr_data.index:
        if gene in symbol_map:
            mapped.append(symbol_map[gene])
            indices.append(gene)

    result = expr_data.loc[indices].copy()
    result.index = mapped

    if result.index.duplicated().any():
        result = result.groupby(result.index).mean()

    return result


def run_validation(method='none'):
    """Run validation with specified adaptation method."""
    print(f"\n{'='*70}")
    print(f"External Validation: {method.upper()} adaptation")
    print('='*70)

    # Load data
    expr, meta = load_gse293591()
    print(f"Loaded {expr.shape[1]} samples, {expr.shape[0]} genes")

    # Load classifier
    script_dir = Path(__file__).parent.parent
    model_dir = script_dir / "models/rnaseq/pancancer"
    classifier = PanCancerClassifier(model_dir)
    classifier.load()

    # Map to ENSEMBL
    mapped = map_to_ensembl(expr, model_dir)
    print(f"Mapped to {mapped.shape[0]} ENSEMBL genes")

    # Log transform (TPM → log2(TPM+1))
    log_data = np.log2(mapped + 1)

    # Apply domain adaptation
    if method == 'quantile':
        adapter = DomainAdapter(method='quantile')
        adapted = adapter.transform(log_data)
    elif method == 'zscore':
        adapter = DomainAdapter(method='zscore')
        adapted = adapter.transform(log_data)
    else:
        adapted = log_data  # No adaptation

    # Check distribution
    print(f"Data distribution: mean={adapted.values.mean():.2f}, std={adapted.values.std():.2f}")

    # Create sample mapping
    sample_to_cancer = {}
    for i, row in meta.iterrows():
        diagnosis = row['diagnosis']
        tcga = DIAGNOSIS_TO_TCGA.get(diagnosis, 'UNKNOWN')
        # Match by position (Sample_001 = first metadata row)
        col_name = f"Sample_{i+1:03d}"
        if col_name in adapted.columns:
            sample_to_cancer[col_name] = tcga

    # Group samples by cancer type
    cancer_samples = defaultdict(list)
    for col in adapted.columns:
        tcga = sample_to_cancer.get(col, 'UNKNOWN')
        if tcga != 'UNKNOWN':
            cancer_samples[tcga].append(col)

    # Run predictions
    results = {}
    total_correct = 0
    total_samples = 0

    for cancer, samples in sorted(cancer_samples.items()):
        if len(samples) < 2:
            continue

        sample_data = adapted[samples]

        try:
            preds = classifier.predict(sample_data)

            correct = 0
            pred_counts = defaultdict(int)

            for pred in preds:
                predicted = pred.predicted_cancer
                pred_counts[predicted] += 1
                if predicted == cancer:
                    correct += 1

            accuracy = correct / len(preds) * 100
            total_correct += correct
            total_samples += len(preds)

            top_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
            marker = "✅" if accuracy >= 50 else "❌"

            print(f"  {cancer}: {accuracy:.0f}% ({correct}/{len(preds)}) → {top_pred} {marker}")

            results[cancer] = {
                'accuracy': accuracy,
                'n_samples': len(preds),
                'top_prediction': top_pred
            }

        except Exception as e:
            print(f"  {cancer}: Error - {e}")

    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n  OVERALL: {overall:.1f}% ({total_correct}/{total_samples})")

    return overall, results


def main():
    print("=" * 70)
    print("Pan-Cancer External Validation with Domain Adaptation")
    print("Dataset: GSE293591 (365 samples, RNA-seq TPM)")
    print("=" * 70)

    results = {}

    # Test without adaptation
    results['none'], _ = run_validation('none')

    # Test with z-score recalibration
    results['zscore'], _ = run_validation('zscore')

    # Test with quantile normalization
    results['quantile'], _ = run_validation('quantile')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Domain Adaptation Comparison")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Accuracy':>10}")
    print("-" * 30)
    for method, acc in results.items():
        marker = "⭐" if acc == max(results.values()) else ""
        print(f"{method:<15} {acc:>9.1f}% {marker}")

    best = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Best method: {best[0]} ({best[1]:.1f}%)")

    # Save results
    script_dir = Path(__file__).parent.parent
    output_file = script_dir / "data/external_validation/GSE293591/adaptation_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
