#!/usr/bin/env python3
"""
Test script for the unified report system.

Tests both Bulk and Single-cell report generation with existing test data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rnaseq_pipeline.reports import UnifiedReportGenerator, ReportConfig


def test_bulk_report():
    """Test bulk RNA-seq report generation."""
    print("=" * 60)
    print("Testing Bulk RNA-seq Report Generation")
    print("=" * 60)

    # Find test data directory
    test_dirs = [
        project_root / "rnaseq_test_results" / "tcga_brca_v3" / "run_20260111_002156" / "accumulated",
        project_root / "rnaseq_test_results" / "tcga_kirc_test",
        project_root / "rnaseq_test_results" / "tcga_luad_test",
    ]

    input_dir = None
    for d in test_dirs:
        if d.exists():
            input_dir = d
            break

    if input_dir is None:
        print("No test data found. Skipping bulk report test.")
        return None

    print(f"Input directory: {input_dir}")

    output_dir = project_root / "rnaseq_test_results" / "unified_report_test" / "bulk"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ReportConfig(
        report_title="BRCA Bulk RNA-seq Analysis Report",
        data_type="bulk",
        cancer_type="BRCA",
        language="ko",
    )

    generator = UnifiedReportGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config
    )

    report_path = generator.generate()
    print(f"\n✅ Report generated: {report_path}")
    print(f"   File size: {report_path.stat().st_size / 1024:.1f} KB")

    return report_path


def test_singlecell_report():
    """Test single-cell RNA-seq report generation."""
    print("\n" + "=" * 60)
    print("Testing Single-cell RNA-seq Report Generation")
    print("=" * 60)

    # Find test data directory
    test_dirs = [
        project_root / "rnaseq_test_results" / "singlecell_test",
        project_root / "rnaseq_test_results" / "singlecell_real_cancer",
        project_root / "rnaseq_test_results" / "singlecell_enhanced_test",
    ]

    input_dir = None
    for d in test_dirs:
        if d.exists():
            input_dir = d
            break

    if input_dir is None:
        print("No single-cell test data found. Skipping.")
        return None

    print(f"Input directory: {input_dir}")

    output_dir = project_root / "rnaseq_test_results" / "unified_report_test" / "singlecell"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ReportConfig(
        report_title="Single-cell RNA-seq Analysis Report",
        data_type="singlecell",
        cancer_type="unknown",
        language="ko",
    )

    generator = UnifiedReportGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config
    )

    report_path = generator.generate()
    print(f"\n✅ Report generated: {report_path}")
    print(f"   File size: {report_path.stat().st_size / 1024:.1f} KB")

    return report_path


def test_adapter():
    """Test the report adapter for backward compatibility."""
    print("\n" + "=" * 60)
    print("Testing Report Adapter")
    print("=" * 60)

    from rnaseq_pipeline.reports.adapter import ReportAdapter

    test_dirs = [
        project_root / "rnaseq_test_results" / "tcga_brca_v3" / "run_20260111_002156" / "accumulated",
    ]

    input_dir = None
    for d in test_dirs:
        if d.exists():
            input_dir = d
            break

    if input_dir is None:
        print("No test data found. Skipping adapter test.")
        return None

    output_dir = project_root / "rnaseq_test_results" / "unified_report_test" / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = ReportAdapter(input_dir, output_dir)
    result = adapter.generate_bulk_report(cancer_type="BRCA")

    print(f"\n✅ Adapter test passed")
    print(f"   Report path: {result['report_path']}")
    print(f"   System: {result['system']}")

    return result


def main():
    print("\n🧬 Unified Report System Test\n")

    results = {
        "bulk": test_bulk_report(),
        "singlecell": test_singlecell_report(),
        "adapter": test_adapter(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "⏭️ SKIP"
        print(f"  {test_name}: {status}")

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()
