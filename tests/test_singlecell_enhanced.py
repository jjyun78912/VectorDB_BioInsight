"""
Test enhanced single-cell pipeline with trajectory and cell-cell interaction.
"""

import sys
sys.path.insert(0, '/Users/admin/VectorDB_BioInsight')

from pathlib import Path
import shutil
from datetime import datetime

def test_enhanced_pipeline():
    """Test the complete enhanced single-cell pipeline."""
    from rnaseq_pipeline.agents.agent_singlecell import SingleCellAgent
    from rnaseq_pipeline.agents.agent_singlecell_report import SingleCellReportAgent

    # Use existing h5ad from previous test
    source_h5ad = Path("/Users/admin/VectorDB_BioInsight/rnaseq_test_results/singlecell_enhanced_test2/run_20260120_190935/adata.h5ad")

    if not source_h5ad.exists():
        print("❌ Source h5ad not found. Please run initial test first.")
        return False

    # Create new test directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/Users/admin/VectorDB_BioInsight/rnaseq_test_results/singlecell_trajectory_test/run_{timestamp}")
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Copy h5ad to input
    print(f"Copying h5ad to {input_dir}...")
    shutil.copy(source_h5ad, input_dir / "adata.h5ad")

    # Run enhanced single-cell pipeline
    print("\n" + "="*60)
    print("Running Enhanced Single-Cell Pipeline")
    print("="*60)

    config = {
        "cancer_type": "LIHC",
        "tissue_type": "liver",
        "enable_cancer_prediction": True,
        "enable_driver_matching": True,
        "enable_pathway_analysis": True,
        "enable_trajectory": True,          # NEW
        "enable_cell_interaction": True,    # NEW
        "model_dir": "/Users/admin/VectorDB_BioInsight/models/rnaseq/pancancer",
    }

    agent = SingleCellAgent(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config
    )

    result = agent.run()

    if result.get("status") != "success":
        print(f"❌ Pipeline failed: {result.get('error')}")
        return False

    print(f"\n✅ Pipeline completed successfully!")

    # Check outputs
    print("\n📁 Output files:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  - {f.name}: {size:,} bytes")

    # Check trajectory results
    trajectory_file = output_dir / "trajectory_pseudotime.csv"
    if trajectory_file.exists():
        import pandas as pd
        traj_df = pd.read_csv(trajectory_file)
        print(f"\n📈 Trajectory Analysis:")
        print(f"   - Cells with pseudotime: {len(traj_df)}")
        print(f"   - Pseudotime range: {traj_df['dpt_pseudotime'].min():.3f} - {traj_df['dpt_pseudotime'].max():.3f}")
    else:
        print("\n⚠️ Trajectory file not found")

    # Check cell-cell interaction results
    interaction_file = output_dir / "cell_interactions.csv"
    if interaction_file.exists():
        import pandas as pd
        int_df = pd.read_csv(interaction_file)
        print(f"\n🔗 Cell-Cell Interaction:")
        print(f"   - Total interactions: {len(int_df)}")
        if len(int_df) > 0:
            print(f"   - Sample interactions:")
            for _, row in int_df.head(5).iterrows():
                print(f"     {row.get('source_cell_type', 'N/A')} -> {row.get('target_cell_type', 'N/A')}: {row.get('ligand', 'N/A')}-{row.get('receptor', 'N/A')}")
    else:
        print("\n⚠️ Cell interaction file not found")

    # Generate report
    print("\n" + "="*60)
    print("Generating HTML Report")
    print("="*60)

    report_agent = SingleCellReportAgent(
        input_dir=output_dir,
        output_dir=output_dir,
        config={"cancer_type": "LIHC"}
    )

    report_result = report_agent.run()

    if report_result.get("status") == "success":
        print(f"✅ Report generated: {output_dir}/report.html")
    else:
        print(f"⚠️ Report generation: {report_result.get('error', 'unknown error')}")

    print(f"\n📂 Results directory: {output_dir}")
    return True


if __name__ == "__main__":
    success = test_enhanced_pipeline()
    print(f"\n{'='*60}")
    print(f"Test Result: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"{'='*60}")
