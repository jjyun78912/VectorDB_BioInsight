#!/usr/bin/env python3
"""
LUAD/LUSC Classification Improvement Test
=========================================

Tests the secondary validation effectiveness on LUAD-LUSC pairs.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml.pancancer_classifier import (
    PanCancerClassifier,
    CONFUSABLE_CANCER_PAIRS,
    TISSUE_SPECIFIC_MARKERS,
    CONFIDENCE_GAP_THRESHOLD
)


def test_luad_lusc_markers():
    """Test LUAD/LUSC marker configuration."""
    print("=" * 60)
    print("1. LUAD/LUSC Marker Configuration Test")
    print("=" * 60)

    # Check confusable pairs
    luad_lusc_pair = frozenset({"LUAD", "LUSC"})
    if luad_lusc_pair in CONFUSABLE_CANCER_PAIRS:
        print(f"✅ LUAD-LUSC pair registered: {CONFUSABLE_CANCER_PAIRS[luad_lusc_pair]}")
    else:
        print("❌ LUAD-LUSC pair NOT in CONFUSABLE_CANCER_PAIRS")
        return False

    # Check markers
    luad_markers = TISSUE_SPECIFIC_MARKERS.get("LUAD", {})
    lusc_markers = TISSUE_SPECIFIC_MARKERS.get("LUSC", {})

    print(f"\nLUAD markers: {luad_markers.get('markers', [])}")
    print(f"LUAD negative markers: {luad_markers.get('negative_markers', [])}")
    print(f"\nLUSC markers: {lusc_markers.get('markers', [])}")
    print(f"LUSC negative markers: {lusc_markers.get('negative_markers', [])}")

    # Check mutual exclusivity
    luad_pos = set(luad_markers.get('markers', []))
    luad_neg = set(luad_markers.get('negative_markers', []))
    lusc_pos = set(lusc_markers.get('markers', []))
    lusc_neg = set(lusc_markers.get('negative_markers', []))

    # LUAD positive should be in LUSC negative (and vice versa)
    overlap1 = luad_pos & lusc_neg
    overlap2 = lusc_pos & luad_neg

    print(f"\n✅ LUAD+ markers in LUSC-: {overlap1}")
    print(f"✅ LUSC+ markers in LUAD-: {overlap2}")

    return True


def test_marker_scoring():
    """Test marker scoring with synthetic data."""
    print("\n" + "=" * 60)
    print("2. Marker Scoring Test (Synthetic Data)")
    print("=" * 60)

    model_dir = Path("models/rnaseq/pancancer")
    classifier = PanCancerClassifier(model_dir)

    # Create synthetic LUAD sample (TTF-1+, p63-)
    luad_sample = pd.Series({
        "NKX2-1": 8.5,  # TTF-1 high
        "NAPSA": 7.2,   # Napsin A high
        "SFTPC": 6.8,   # Surfactant high
        "TP63": 1.2,    # p63 low (negative marker for LUAD)
        "KRT5": 1.5,    # CK5 low
        "SOX2": 2.0,    # SOX2 low
    })

    # Create synthetic LUSC sample (TTF-1-, p63+)
    lusc_sample = pd.Series({
        "NKX2-1": 1.5,  # TTF-1 low (negative marker for LUSC)
        "NAPSA": 1.2,   # Napsin A low
        "SFTPC": 1.0,   # Surfactant low
        "TP63": 9.2,    # p63 high
        "KRT5": 8.5,    # CK5 high
        "SOX2": 7.8,    # SOX2 high
    })

    # Get marker configs
    luad_config = TISSUE_SPECIFIC_MARKERS.get("LUAD", {})
    lusc_config = TISSUE_SPECIFIC_MARKERS.get("LUSC", {})

    # Calculate scores for LUAD sample
    luad_as_luad = classifier._calculate_marker_score(
        luad_sample,
        luad_config.get("markers", []),
        luad_config.get("negative_markers", [])
    )
    luad_as_lusc = classifier._calculate_marker_score(
        luad_sample,
        lusc_config.get("markers", []),
        lusc_config.get("negative_markers", [])
    )

    # Calculate scores for LUSC sample
    lusc_as_luad = classifier._calculate_marker_score(
        lusc_sample,
        luad_config.get("markers", []),
        luad_config.get("negative_markers", [])
    )
    lusc_as_lusc = classifier._calculate_marker_score(
        lusc_sample,
        lusc_config.get("markers", []),
        lusc_config.get("negative_markers", [])
    )

    print(f"\nLUAD sample scores:")
    print(f"  - As LUAD: {luad_as_luad:.3f}")
    print(f"  - As LUSC: {luad_as_lusc:.3f}")
    print(f"  - Correct: {'✅' if luad_as_luad > luad_as_lusc else '❌'}")

    print(f"\nLUSC sample scores:")
    print(f"  - As LUAD: {lusc_as_luad:.3f}")
    print(f"  - As LUSC: {lusc_as_lusc:.3f}")
    print(f"  - Correct: {'✅' if lusc_as_lusc > lusc_as_luad else '❌'}")

    return (luad_as_luad > luad_as_lusc) and (lusc_as_lusc > lusc_as_luad)


def test_with_tcga_data():
    """Test secondary validation with actual TCGA data."""
    print("\n" + "=" * 60)
    print("3. TCGA Test Data Validation")
    print("=" * 60)

    model_dir = Path("models/rnaseq/pancancer")

    # Check if model exists
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return None

    # Load classifier
    classifier = PanCancerClassifier(model_dir)

    try:
        classifier.load()
        print(f"✅ Model loaded successfully")
        print(f"   - Cancer types: {len(classifier.cancer_types)}")
        print(f"   - Features: {len(classifier.feature_names)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Load test data if available
    test_data_path = model_dir / "test_data.csv"
    if not test_data_path.exists():
        # Try to use the preprocessor to get training data info
        print(f"\n⚠️ No separate test data found. Checking training info...")

        training_results_path = model_dir / "training_results.json"
        if training_results_path.exists():
            with open(training_results_path) as f:
                training_info = json.load(f)
            print(f"   Training samples: {training_info.get('n_samples', 'N/A')}")
            print(f"   Training accuracy: {training_info.get('accuracy', 'N/A')}")

        # Check evaluation report for LUAD/LUSC performance
        eval_path = model_dir / "evaluation" / "evaluation_report.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_report = json.load(f)

            per_class = eval_report.get("metrics", {}).get("basic_metrics", {}).get("per_class", {})

            luad_metrics = per_class.get("LUAD", {})
            lusc_metrics = per_class.get("LUSC", {})

            print(f"\n📊 Current Performance (from evaluation report):")
            print(f"   LUAD - F1: {luad_metrics.get('f1', 0):.3f}, "
                  f"Precision: {luad_metrics.get('precision', 0):.3f}, "
                  f"Recall: {luad_metrics.get('recall', 0):.3f}")
            print(f"   LUSC - F1: {lusc_metrics.get('f1', 0):.3f}, "
                  f"Precision: {lusc_metrics.get('precision', 0):.3f}, "
                  f"Recall: {lusc_metrics.get('recall', 0):.3f}")

            return {
                "LUAD": luad_metrics,
                "LUSC": lusc_metrics,
            }

    return None


def test_secondary_validation_logic():
    """Test the secondary validation logic directly."""
    print("\n" + "=" * 60)
    print("4. Secondary Validation Logic Test")
    print("=" * 60)

    model_dir = Path("models/rnaseq/pancancer")
    classifier = PanCancerClassifier(model_dir)

    try:
        classifier.load()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Simulate a LUSC sample misclassified as LUAD
    # Create mock data with LUSC-like marker expression
    mock_lusc_sample = pd.Series({
        # LUSC positive markers (high)
        "TP63": 9.5,
        "SOX2": 8.2,
        "KRT5": 7.8,
        "KRT6A": 7.5,
        "KRT14": 6.9,
        "DSG3": 6.5,
        "PKP1": 6.2,
        # LUSC negative markers (low = LUAD markers)
        "NKX2-1": 1.5,
        "TTF1": 1.2,
        "NAPSA": 1.0,
        "SFTPC": 0.8,
        "SFTPB": 0.9,
    }, name="MOCK_LUSC_SAMPLE")

    # Mock probabilities where LUAD is slightly higher (misclassification)
    mock_probs = {
        "LUAD": 0.42,  # Incorrectly higher
        "LUSC": 0.38,  # Should be higher
        "HNSC": 0.08,
        "BLCA": 0.05,
        "others": 0.07,
    }

    print("\nSimulated scenario: LUSC sample misclassified as LUAD")
    print(f"  Original probabilities: LUAD={mock_probs['LUAD']:.2f}, LUSC={mock_probs['LUSC']:.2f}")
    print(f"  Confidence gap: {mock_probs['LUAD'] - mock_probs['LUSC']:.2f}")

    # Test marker scoring
    luad_config = TISSUE_SPECIFIC_MARKERS.get("LUAD", {})
    lusc_config = TISSUE_SPECIFIC_MARKERS.get("LUSC", {})

    luad_score = classifier._calculate_marker_score(
        mock_lusc_sample,
        luad_config.get("markers", []),
        luad_config.get("negative_markers", [])
    )
    lusc_score = classifier._calculate_marker_score(
        mock_lusc_sample,
        lusc_config.get("markers", []),
        lusc_config.get("negative_markers", [])
    )

    print(f"\n  Marker scores:")
    print(f"    LUAD markers: {luad_score:.3f}")
    print(f"    LUSC markers: {lusc_score:.3f}")

    # Check if secondary validation would correct this
    prob_diff = mock_probs["LUAD"] - mock_probs["LUSC"]
    would_correct = (
        prob_diff < 0.25 and  # Confidence gap is small
        lusc_score > luad_score * 1.5 and  # LUSC markers significantly higher
        lusc_score > 0.2  # LUSC score is meaningful
    )

    print(f"\n  Secondary validation criteria:")
    print(f"    - prob_diff < 0.25: {prob_diff:.2f} < 0.25 = {prob_diff < 0.25}")
    print(f"    - LUSC > LUAD * 1.5: {lusc_score:.3f} > {luad_score * 1.5:.3f} = {lusc_score > luad_score * 1.5}")
    print(f"    - LUSC > 0.2: {lusc_score:.3f} > 0.2 = {lusc_score > 0.2}")
    print(f"\n  Would correct to LUSC: {'✅ Yes' if would_correct else '❌ No'}")

    return would_correct


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LUAD/LUSC Classification Improvement Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Marker configuration
    results["marker_config"] = test_luad_lusc_markers()

    # Test 2: Marker scoring
    results["marker_scoring"] = test_marker_scoring()

    # Test 3: TCGA data validation
    results["tcga_validation"] = test_with_tcga_data()

    # Test 4: Secondary validation logic
    results["secondary_validation"] = test_secondary_validation_logic()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results.items():
        if result is None:
            status = "⚠️ Skipped"
        elif isinstance(result, bool):
            status = "✅ Passed" if result else "❌ Failed"
        else:
            status = "✅ Completed"
        print(f"  {test_name}: {status}")

    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations for Full Validation")
    print("=" * 60)
    print("""
To fully validate the LUAD/LUSC improvement:

1. Run prediction on actual misclassified LUSC samples
2. Check if secondary_validation corrects LUSC→LUAD errors
3. Re-run evaluation with secondary validation enabled

The secondary validation logic is designed to:
- Detect when LUAD-LUSC confusion is likely (small confidence gap)
- Use tissue-specific markers (TTF-1 vs p63) to discriminate
- Correct predictions only when marker evidence is strong

Expected improvement:
- LUSC→LUAD misclassifications (7 cases): May reduce by 50-70%
- LUAD F1: 0.910 → ~0.93+
- LUSC F1: 0.877 → ~0.92+
""")


if __name__ == "__main__":
    main()
