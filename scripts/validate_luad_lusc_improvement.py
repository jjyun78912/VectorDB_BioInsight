#!/usr/bin/env python3
"""
LUAD/LUSC Classification Improvement Validation
===============================================

Run actual predictions on TCGA test data with secondary validation.
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


def load_and_predict(classifier, count_matrix_path: Path, true_label: str, max_samples: int = None):
    """Load data and run predictions."""
    # Load count matrix (genes x samples)
    df = pd.read_csv(count_matrix_path, index_col=0)

    # Get sample columns
    sample_cols = df.columns.tolist()
    if max_samples:
        sample_cols = sample_cols[:max_samples]

    print(f"  Loaded {len(sample_cols)} samples for {true_label}")
    print(f"  Gene format: {df.index[0]} (total genes: {len(df)})")

    results = []

    for sample_id in sample_cols:
        # Prepare as DataFrame with genes x samples format
        # (the predict() function expects Gene x Sample matrix)
        sample_df = df[[sample_id]]  # Keep as genes x 1 sample

        # Run prediction
        try:
            pred_results = classifier.predict(sample_df, sample_ids=[sample_id])

            if pred_results and len(pred_results) > 0:
                result = pred_results[0]
                results.append({
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "predicted": result.predicted_cancer,
                    "confidence": result.confidence,
                    "secondary_validation": result.secondary_validation,
                    "warnings": result.warnings,
                })
        except Exception as e:
            print(f"    Error predicting {sample_id}: {e}")

    return results


def analyze_results(all_results):
    """Analyze prediction results."""
    # Group by true label
    by_true_label = defaultdict(list)
    for r in all_results:
        by_true_label[r["true_label"]].append(r)

    # Calculate metrics
    metrics = {}

    for label, results in by_true_label.items():
        total = len(results)
        correct = sum(1 for r in results if r["predicted"] == label)
        misclassified = [r for r in results if r["predicted"] != label]

        # Count corrections by secondary validation
        corrected = sum(
            1 for r in results
            if r.get("secondary_validation") and
               r["secondary_validation"].get("corrected_prediction") == label
        )

        metrics[label] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "misclassified": len(misclassified),
            "misclassified_as": defaultdict(int),
            "corrections_by_sv": corrected,
        }

        for r in misclassified:
            metrics[label]["misclassified_as"][r["predicted"]] += 1

    return metrics


def main():
    """Run LUAD/LUSC validation."""
    print("=" * 70)
    print("LUAD/LUSC Classification Improvement Validation")
    print("=" * 70)

    model_dir = Path("models/rnaseq/pancancer")
    data_dir = Path("data/tcga/pancancer_test")

    # Load classifier
    print("\n1. Loading Pan-Cancer Classifier...")
    classifier = PanCancerClassifier(model_dir)
    classifier.load()
    print("   ✅ Model loaded")

    # Load and predict LUAD samples
    print("\n2. Running predictions on LUAD test samples...")
    luad_path = data_dir / "LUAD" / "count_matrix.csv"
    luad_results = load_and_predict(classifier, luad_path, "LUAD", max_samples=None)  # All samples

    # Load and predict LUSC samples
    print("\n3. Running predictions on LUSC test samples...")
    lusc_path = data_dir / "LUSC" / "count_matrix.csv"
    lusc_results = load_and_predict(classifier, lusc_path, "LUSC", max_samples=None)  # All samples

    # Combine results
    all_results = luad_results + lusc_results

    # Analyze
    print("\n4. Analyzing results...")
    metrics = analyze_results(all_results)

    # Print results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for label in ["LUAD", "LUSC"]:
        m = metrics.get(label, {})
        print(f"\n{label}:")
        print(f"  Total samples: {m.get('total', 0)}")
        print(f"  Correct: {m.get('correct', 0)}")
        print(f"  Accuracy: {m.get('accuracy', 0):.1%}")
        print(f"  Misclassified: {m.get('misclassified', 0)}")

        if m.get("misclassified_as"):
            print(f"  Misclassified as:")
            for pred_label, count in m["misclassified_as"].items():
                print(f"    - {pred_label}: {count}")

        if m.get("corrections_by_sv", 0) > 0:
            print(f"  ✅ Corrections by secondary validation: {m['corrections_by_sv']}")

    # Show detailed misclassifications
    print("\n" + "=" * 70)
    print("Detailed Misclassifications")
    print("=" * 70)

    for r in all_results:
        if r["predicted"] != r["true_label"]:
            print(f"\n  {r['sample_id']}:")
            print(f"    True: {r['true_label']}")
            print(f"    Predicted: {r['predicted']} ({r['confidence']:.1%})")

            sv = r.get("secondary_validation", {})
            if sv:
                print(f"    Secondary validation:")
                print(f"      - Is confusable pair: {sv.get('is_confusable_pair', False)}")
                print(f"      - Confidence gap: {sv.get('confidence_gap', 0):.2f}")

                if sv.get("marker_scores"):
                    print(f"      - Marker scores: {sv['marker_scores']}")

                if sv.get("corrected_prediction"):
                    print(f"      ✅ Corrected to: {sv['corrected_prediction']}")

            if r.get("warnings"):
                print(f"    Warnings: {r['warnings']}")

    # Calculate improvement
    print("\n" + "=" * 70)
    print("Improvement Analysis")
    print("=" * 70)

    luad_m = metrics.get("LUAD", {})
    lusc_m = metrics.get("LUSC", {})

    # LUSC→LUAD errors (the main problem: 7 cases historically)
    lusc_to_luad = lusc_m.get("misclassified_as", {}).get("LUAD", 0)
    lusc_corrected = lusc_m.get("corrections_by_sv", 0)

    # LUAD→LUSC errors (2 cases historically)
    luad_to_lusc = luad_m.get("misclassified_as", {}).get("LUSC", 0)
    luad_corrected = luad_m.get("corrections_by_sv", 0)

    print(f"\nLUSC→LUAD misclassifications: {lusc_to_luad}")
    print(f"  - Corrected by secondary validation: {lusc_corrected}")

    print(f"\nLUAD→LUSC misclassifications: {luad_to_lusc}")
    print(f"  - Corrected by secondary validation: {luad_corrected}")

    # Estimate F1 improvement
    print("\n" + "=" * 70)
    print("Estimated Impact on F1 Scores")
    print("=" * 70)

    # Original F1 scores from evaluation report
    original_luad_f1 = 0.910
    original_lusc_f1 = 0.877

    print(f"\nOriginal F1 scores (from evaluation):")
    print(f"  LUAD F1: {original_luad_f1:.3f}")
    print(f"  LUSC F1: {original_lusc_f1:.3f}")

    # Calculate potential improvement
    # If secondary validation can correct even 50% of confusions...
    total_luad = luad_m.get("total", 20)
    total_lusc = lusc_m.get("total", 20)
    correct_luad = luad_m.get("correct", 0)
    correct_lusc = lusc_m.get("correct", 0)

    # Current accuracy on test samples
    test_luad_acc = correct_luad / total_luad if total_luad > 0 else 0
    test_lusc_acc = correct_lusc / total_lusc if total_lusc > 0 else 0

    print(f"\nTest sample accuracy (n={total_luad + total_lusc}):")
    print(f"  LUAD: {test_luad_acc:.1%} ({correct_luad}/{total_luad})")
    print(f"  LUSC: {test_lusc_acc:.1%} ({correct_lusc}/{total_lusc})")


if __name__ == "__main__":
    main()
