"""
Test script for the RNA-seq Pipeline

Creates sample data and runs the full pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rnaseq_pipeline.orchestrator import RNAseqPipeline, create_sample_data


def test_full_pipeline():
    """Test the complete pipeline with sample data."""
    print("=" * 60)
    print("RNA-seq Pipeline Test")
    print("=" * 60)

    # Setup directories
    base_dir = Path(__file__).parent
    input_dir = base_dir / "test_input"
    output_dir = base_dir / "test_output"

    # Create sample data
    print("\n1. Creating sample data...")
    create_sample_data(input_dir, n_genes=500, n_samples=8)

    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = RNAseqPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        config={
            "cancer_type": "lung_cancer",
            "padj_cutoff": 0.05,
            "log2fc_cutoff": 1.0,
            "correlation_threshold": 0.6,  # Lower for test data
            "top_hub_count": 10
        }
    )

    # Run pipeline
    print("\n3. Running pipeline...")
    results = pipeline.run()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Run directory: {pipeline.run_dir}")
    print(f"Completed agents: {results['completed_agents']}")
    print(f"Failed agents: {results['failed_agents']}")

    # Check outputs
    print("\n4. Checking outputs...")

    expected_files = [
        "agent1_deg/deg_significant.csv",
        "agent2_network/hub_genes.csv",
        "agent3_pathway/pathway_summary.csv",
        "agent4_validation/integrated_gene_table.csv",
        "agent4_validation/interpretation_report.json",
        "agent5_visualization/figures/volcano_plot.png",
        "agent6_report/report.html"
    ]

    for f in expected_files:
        filepath = pipeline.run_dir / f
        if filepath.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")

    # Show key statistics
    print("\n5. Key Statistics:")
    for agent, result in results['agent_results'].items():
        print(f"\n  {agent}:")
        for key, value in result.items():
            if isinstance(value, (int, float, str)):
                print(f"    {key}: {value}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print(f"View report: {pipeline.run_dir / 'agent6_report' / 'report.html'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    test_full_pipeline()
