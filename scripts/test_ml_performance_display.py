#!/usr/bin/env python3
"""
ML Performance Metrics Display Test
====================================

Tests that model_performance is correctly populated and displayed
for all 17 cancer types.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


def test_all_cancer_types():
    """Test ML prediction with model_performance for all cancer types."""
    print("=" * 70)
    print("ML Model Performance Metrics Test")
    print("=" * 70)

    model_dir = Path("models/rnaseq/pancancer")
    test_data_dir = Path("data/tcga/pancancer_test_20")

    # Load classifier
    print("\n1. Loading Pan-Cancer Classifier...")
    classifier = PanCancerClassifier(model_dir)
    classifier.load()
    print("   ✅ Model loaded")
    print(f"   - Training metrics: {'loaded' if classifier.training_metrics else 'not loaded'}")
    print(f"   - Evaluation metrics: {'loaded' if classifier.evaluation_metrics else 'not loaded'}")

    # Get all cancer types
    cancer_types = sorted([d.name for d in test_data_dir.iterdir() if d.is_dir()])
    print(f"\n2. Testing {len(cancer_types)} cancer types...")

    results = []

    for cancer_type in cancer_types:
        count_path = test_data_dir / cancer_type / "count_matrix.csv"
        if not count_path.exists():
            continue

        # Load counts
        counts = pd.read_csv(count_path, index_col=0)

        # Take only 2 samples for quick test
        sample_cols = counts.columns[:2]
        sample_df = counts[sample_cols]

        # Predict
        try:
            pred_results = classifier.predict(sample_df, sample_ids=list(sample_cols))

            if pred_results and len(pred_results) > 0:
                result = pred_results[0]

                # Check model_performance
                perf = result.model_performance

                correct = result.predicted_cancer == cancer_type

                results.append({
                    'true_cancer': cancer_type,
                    'predicted': result.predicted_cancer,
                    'confidence': result.confidence,
                    'correct': correct,
                    'has_performance': perf is not None,
                    'overall_accuracy': perf.get('overall', {}).get('accuracy', 0) if perf else 0,
                    'overall_f1': perf.get('overall', {}).get('f1_macro', 0) if perf else 0,
                    'overall_mcc': perf.get('overall', {}).get('mcc', 0) if perf else 0,
                    'overall_pr_auc': perf.get('overall', {}).get('pr_auc_macro', 0) if perf else 0,
                    'per_class_f1': perf.get('per_class', {}).get('f1', 0) if perf else 0,
                    'per_class_precision': perf.get('per_class', {}).get('precision', 0) if perf else 0,
                    'per_class_recall': perf.get('per_class', {}).get('recall', 0) if perf else 0,
                    'per_class_pr_auc': perf.get('per_class', {}).get('pr_auc', 0) if perf else 0,
                    'per_class_roc_auc': perf.get('per_class', {}).get('roc_auc', 0) if perf else 0,
                })

                status = "✅" if correct else "❌"
                perf_status = "📊" if perf else "⚠️"
                print(f"   {status} {cancer_type}: {result.predicted_cancer} ({result.confidence*100:.1f}%) {perf_status}")

        except Exception as e:
            print(f"   ❌ {cancer_type}: Error - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    correct_count = sum(1 for r in results if r['correct'])
    perf_count = sum(1 for r in results if r['has_performance'])

    print(f"\n📊 Prediction Accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"📊 Performance Metrics Available: {perf_count}/{len(results)}")

    # Display performance metrics table
    print("\n" + "=" * 70)
    print("Model Performance Scorecard (per predicted cancer type)")
    print("=" * 70)

    print(f"\n{'Cancer':<8} {'Pred':<8} {'Conf':>6} {'Acc':>7} {'F1':>7} {'MCC':>7} {'PR-AUC':>7}")
    print("-" * 60)

    for r in results:
        mark = "✅" if r['correct'] else "❌"
        print(f"{r['true_cancer']:<8} {r['predicted']:<8} {r['confidence']*100:>5.1f}% "
              f"{r['overall_accuracy']*100:>6.1f}% {r['overall_f1']*100:>6.1f}% "
              f"{r['overall_mcc']:>6.3f} {r['overall_pr_auc']*100:>6.1f}% {mark}")

    # Per-class metrics for predicted cancer
    print("\n" + "=" * 70)
    print("Per-Class Performance (for predicted cancer type)")
    print("=" * 70)

    print(f"\n{'Cancer':<8} {'Pred':<8} {'F1':>7} {'Prec':>7} {'Recall':>7} {'PR-AUC':>7} {'ROC-AUC':>7}")
    print("-" * 65)

    for r in results:
        mark = "✅" if r['correct'] else "❌"
        print(f"{r['true_cancer']:<8} {r['predicted']:<8} "
              f"{r['per_class_f1']*100:>6.1f}% {r['per_class_precision']*100:>6.1f}% "
              f"{r['per_class_recall']*100:>6.1f}% {r['per_class_pr_auc']*100:>6.1f}% "
              f"{r['per_class_roc_auc']*100:>6.1f}% {mark}")

    # Show sample JSON output
    print("\n" + "=" * 70)
    print("Sample model_performance JSON (BRCA)")
    print("=" * 70)

    for r in results:
        if r['true_cancer'] == 'BRCA' and r['has_performance']:
            # Re-predict to get full performance dict
            counts = pd.read_csv(test_data_dir / "BRCA" / "count_matrix.csv", index_col=0)
            pred = classifier.predict(counts[counts.columns[:1]])[0]
            print(json.dumps(pred.model_performance, indent=2))
            break

    return results


if __name__ == "__main__":
    test_all_cancer_types()
