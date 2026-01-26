#!/usr/bin/env python3
"""
LUAD/LUSC Classification Full Validation
========================================

Tests secondary validation effectiveness using full test set.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


def main():
    """Run full LUAD/LUSC validation."""
    print("=" * 70)
    print("LUAD/LUSC Classification Full Validation")
    print("=" * 70)

    model_dir = Path("models/rnaseq/pancancer")
    data_dir = Path("data/tcga/pancancer_test")

    # Load classifier
    print("\n1. Loading Pan-Cancer Classifier...")
    classifier = PanCancerClassifier(model_dir)
    classifier.load()
    print("   ✅ Model loaded")

    # Load test data
    print("\n2. Loading test data...")
    counts = pd.read_csv(data_dir / "test_counts.csv", index_col=0)
    metadata = pd.read_csv(data_dir / "test_metadata.csv")

    # Filter for tumor samples only
    tumor_metadata = metadata[metadata['is_tumor'] == 1]
    print(f"   Total tumor samples: {len(tumor_metadata)}")

    # Get LUAD and LUSC samples
    luad_barcodes = tumor_metadata[tumor_metadata['cancer_type'] == 'LUAD']['barcode'].tolist()
    lusc_barcodes = tumor_metadata[tumor_metadata['cancer_type'] == 'LUSC']['barcode'].tolist()

    print(f"   LUAD samples: {len(luad_barcodes)}")
    print(f"   LUSC samples: {len(lusc_barcodes)}")

    # Filter count matrix
    all_barcodes = luad_barcodes + lusc_barcodes
    available_barcodes = [b for b in all_barcodes if b in counts.columns]
    print(f"   Available in count matrix: {len(available_barcodes)}")

    if len(available_barcodes) == 0:
        print("   ❌ No matching samples found")
        # Try alternative column matching
        print("\n   Trying partial barcode matching...")
        sample_cols = counts.columns.tolist()

        # Create mapping from partial barcodes
        partial_map = {}
        for col in sample_cols:
            # Extract TCGA barcode part (first 15 chars typically)
            for barcode in all_barcodes:
                if barcode[:12] in col or col[:12] in barcode:
                    partial_map[barcode] = col
                    break

        if partial_map:
            print(f"   Found {len(partial_map)} partial matches")
            available_barcodes = list(partial_map.keys())
        else:
            print("   ❌ No partial matches either")
            return

    # Create true labels
    true_labels = {}
    for barcode in available_barcodes:
        if barcode in luad_barcodes:
            true_labels[barcode] = 'LUAD'
        else:
            true_labels[barcode] = 'LUSC'

    # Run predictions
    print("\n3. Running predictions...")
    results = []

    for i, barcode in enumerate(available_barcodes):
        # Get count column (use partial match if needed)
        col = partial_map.get(barcode, barcode) if 'partial_map' in dir() else barcode
        if col not in counts.columns:
            continue

        sample_df = counts[[col]]

        try:
            pred_results = classifier.predict(sample_df, sample_ids=[barcode])

            if pred_results and len(pred_results) > 0:
                result = pred_results[0]
                results.append({
                    "sample_id": barcode,
                    "true_label": true_labels[barcode],
                    "predicted": result.predicted_cancer,
                    "confidence": result.confidence,
                    "secondary_validation": result.secondary_validation,
                    "warnings": result.warnings,
                })

                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(available_barcodes)} samples")

        except Exception as e:
            print(f"   Error predicting {barcode}: {e}")

    # Analyze results
    print("\n4. Analyzing results...")

    # Group by true label
    by_true = defaultdict(list)
    for r in results:
        by_true[r["true_label"]].append(r)

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for label in ["LUAD", "LUSC"]:
        label_results = by_true.get(label, [])
        if not label_results:
            continue

        total = len(label_results)
        correct = sum(1 for r in label_results if r["predicted"] == label)
        misclassified = [r for r in label_results if r["predicted"] != label]

        print(f"\n{label}:")
        print(f"  Total: {total}")
        print(f"  Correct: {correct} ({correct/total:.1%})")
        print(f"  Misclassified: {len(misclassified)}")

        if misclassified:
            # Count misclassifications by predicted label
            mis_counts = defaultdict(int)
            for r in misclassified:
                mis_counts[r["predicted"]] += 1
            print(f"  Misclassified as:")
            for pred_label, count in sorted(mis_counts.items(), key=lambda x: -x[1]):
                print(f"    - {pred_label}: {count}")

            # Check secondary validation corrections
            corrected = [r for r in misclassified
                        if r.get("secondary_validation") and
                        r["secondary_validation"].get("corrected_prediction") == label]
            if corrected:
                print(f"  ✅ Would be corrected by secondary validation: {len(corrected)}")

    # Show detailed misclassifications
    print("\n" + "=" * 70)
    print("Detailed Misclassifications")
    print("=" * 70)

    for r in results:
        if r["predicted"] != r["true_label"]:
            print(f"\n{r['sample_id']}:")
            print(f"  True: {r['true_label']}")
            print(f"  Predicted: {r['predicted']} ({r['confidence']:.1%})")

            sv = r.get("secondary_validation", {})
            if sv:
                print(f"  Is confusable pair: {sv.get('is_confusable_pair', False)}")
                print(f"  Confidence gap: {sv.get('confidence_gap', 0):.3f}")

                if sv.get("marker_scores"):
                    print(f"  Marker scores: {sv['marker_scores']}")

                if sv.get("corrected_prediction"):
                    print(f"  ✅ Corrected to: {sv['corrected_prediction']}")

    # Calculate overall improvement
    print("\n" + "=" * 70)
    print("Improvement Summary")
    print("=" * 70)

    luad_results = by_true.get("LUAD", [])
    lusc_results = by_true.get("LUSC", [])

    # LUSC->LUAD errors
    lusc_to_luad = [r for r in lusc_results if r["predicted"] == "LUAD"]
    lusc_corrected = [r for r in lusc_to_luad
                     if r.get("secondary_validation") and
                     r["secondary_validation"].get("corrected_prediction") == "LUSC"]

    # LUAD->LUSC errors
    luad_to_lusc = [r for r in luad_results if r["predicted"] == "LUSC"]
    luad_corrected = [r for r in luad_to_lusc
                     if r.get("secondary_validation") and
                     r["secondary_validation"].get("corrected_prediction") == "LUAD"]

    print(f"\nLUSC→LUAD errors: {len(lusc_to_luad)}")
    print(f"  Would be corrected: {len(lusc_corrected)}")

    print(f"\nLUAD→LUSC errors: {len(luad_to_lusc)}")
    print(f"  Would be corrected: {len(luad_corrected)}")

    # Calculate potential F1 improvement
    if luad_results and lusc_results:
        # Original metrics
        luad_tp = sum(1 for r in luad_results if r["predicted"] == "LUAD")
        luad_fn = len(luad_results) - luad_tp
        lusc_fp = sum(1 for r in lusc_results if r["predicted"] == "LUAD")

        lusc_tp = sum(1 for r in lusc_results if r["predicted"] == "LUSC")
        lusc_fn = len(lusc_results) - lusc_tp
        luad_fp = sum(1 for r in luad_results if r["predicted"] == "LUSC")

        # Original F1
        luad_precision = luad_tp / (luad_tp + lusc_fp) if (luad_tp + lusc_fp) > 0 else 0
        luad_recall = luad_tp / (luad_tp + luad_fn) if (luad_tp + luad_fn) > 0 else 0
        luad_f1 = 2 * luad_precision * luad_recall / (luad_precision + luad_recall) if (luad_precision + luad_recall) > 0 else 0

        lusc_precision = lusc_tp / (lusc_tp + luad_fp) if (lusc_tp + luad_fp) > 0 else 0
        lusc_recall = lusc_tp / (lusc_tp + lusc_fn) if (lusc_tp + lusc_fn) > 0 else 0
        lusc_f1 = 2 * lusc_precision * lusc_recall / (lusc_precision + lusc_recall) if (lusc_precision + lusc_recall) > 0 else 0

        print(f"\nCurrent F1 on test set:")
        print(f"  LUAD: {luad_f1:.3f} (P={luad_precision:.3f}, R={luad_recall:.3f})")
        print(f"  LUSC: {lusc_f1:.3f} (P={lusc_precision:.3f}, R={lusc_recall:.3f})")

        # After correction
        if len(lusc_corrected) > 0 or len(luad_corrected) > 0:
            # Adjusted metrics
            adj_luad_tp = luad_tp + len(luad_corrected)
            adj_lusc_fp = lusc_fp - len(lusc_corrected)
            adj_luad_fn = luad_fn - len(luad_corrected)

            adj_lusc_tp = lusc_tp + len(lusc_corrected)
            adj_luad_fp = luad_fp - len(luad_corrected)
            adj_lusc_fn = lusc_fn - len(lusc_corrected)

            adj_luad_precision = adj_luad_tp / (adj_luad_tp + adj_lusc_fp) if (adj_luad_tp + adj_lusc_fp) > 0 else 0
            adj_luad_recall = adj_luad_tp / (adj_luad_tp + adj_luad_fn) if (adj_luad_tp + adj_luad_fn) > 0 else 0
            adj_luad_f1 = 2 * adj_luad_precision * adj_luad_recall / (adj_luad_precision + adj_luad_recall) if (adj_luad_precision + adj_luad_recall) > 0 else 0

            adj_lusc_precision = adj_lusc_tp / (adj_lusc_tp + adj_luad_fp) if (adj_lusc_tp + adj_luad_fp) > 0 else 0
            adj_lusc_recall = adj_lusc_tp / (adj_lusc_tp + adj_lusc_fn) if (adj_lusc_tp + adj_lusc_fn) > 0 else 0
            adj_lusc_f1 = 2 * adj_lusc_precision * adj_lusc_recall / (adj_lusc_precision + adj_lusc_recall) if (adj_lusc_precision + adj_lusc_recall) > 0 else 0

            print(f"\nProjected F1 with secondary validation:")
            print(f"  LUAD: {adj_luad_f1:.3f} (Δ={adj_luad_f1 - luad_f1:+.3f})")
            print(f"  LUSC: {adj_lusc_f1:.3f} (Δ={adj_lusc_f1 - lusc_f1:+.3f})")


if __name__ == "__main__":
    main()
