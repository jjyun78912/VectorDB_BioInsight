#!/usr/bin/env python3
"""
Test script for visualization with hover labels.

Tests the modified visualization code that shows gene names on hover
instead of directly on the plots.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rnaseq_pipeline.agents.agent5_visualization import VisualizationAgent


def test_visualization():
    """Test visualization agent with hover labels."""
    print("=" * 60)
    print("Testing Visualization with Hover Labels")
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
        print("No test data found. Skipping.")
        return None

    print(f"Input directory: {input_dir}")

    output_dir = project_root / "rnaseq_test_results" / "viz_hover_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "cancer_type": "BRCA",
        "generate_interactive": True,
        "n_top_genes_label": 10,  # For old code reference, should be overridden
    }

    agent = VisualizationAgent(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config
    )

    # Load data first
    if not agent.validate_inputs():
        print("Warning: Some input files missing, but continuing with available data")

    result = agent.run()

    print(f"\n✅ Visualization completed!")
    print(f"   Static figures: {len(result.get('figures_generated', []))}")
    print(f"   Interactive files: {len(result.get('interactive_files', []))}")
    print(f"   Failed: {len(result.get('failed_figures', []))}")

    # Check generated files
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        png_files = list(figures_dir.glob("*.png"))
        html_files = list(figures_dir.glob("*.html"))
        print(f"\n   PNG files: {len(png_files)}")
        for f in png_files[:5]:
            print(f"     - {f.name}")
        print(f"\n   HTML files (interactive): {len(html_files)}")
        for f in html_files:
            print(f"     - {f.name}")

    return result


if __name__ == "__main__":
    print("\n🧬 Visualization Hover Test\n")
    result = test_visualization()

    if result:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n⚠️ Test skipped - no data found")
