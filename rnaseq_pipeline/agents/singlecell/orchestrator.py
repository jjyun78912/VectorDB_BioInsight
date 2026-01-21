"""
Single-Cell RNA-seq Pipeline Orchestrator

Coordinates the 6-agent pipeline for single-cell RNA-seq analysis.

Pipeline Flow:
    Agent 1 (QC) → Agent 2 (Cluster) → Agent 3 (Pathway) →
    Agent 4 (Trajectory) → Agent 5 (CNV/ML) → Agent 6 (Report)

Usage:
    from rnaseq_pipeline.agents.singlecell import SingleCellOrchestrator

    orchestrator = SingleCellOrchestrator(
        input_dir=Path("./input"),
        output_dir=Path("./output"),
        config={"cancer_type": "BRCA"}
    )
    results = orchestrator.run()
"""

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import logging

from .agent1_qc import SingleCellQCAgent
from .agent2_cluster import SingleCellClusterAgent
from .agent3_pathway import SingleCellPathwayAgent
from .agent4_trajectory import SingleCellTrajectoryAgent
from .agent5_cnv_ml import SingleCellCNVMLAgent
from .agent6_report import SingleCellReportAgent


class SingleCellOrchestrator:
    """
    Orchestrator for 6-Agent Single-Cell RNA-seq Pipeline.

    Manages the sequential execution of:
    - Agent 1: QC & Preprocessing
    - Agent 2: Clustering & Annotation (ML #1: Cell Type)
    - Agent 3: Pathway & Validation
    - Agent 4: Trajectory & Dynamics
    - Agent 5: CNV & ML Prediction (ML #2: Cancer Type, ML #3: Malignancy)
    - Agent 6: Visualization & Report
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ):
        """
        Initialize orchestrator.

        Args:
            input_dir: Directory containing input files (count_matrix.csv or h5ad)
            output_dir: Directory for output files
            config: Configuration dictionary for all agents
            progress_callback: Optional callback for progress updates
                              Signature: callback(agent_name, percent, message)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.progress_callback = progress_callback

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("SingleCellOrchestrator")
        self.logger.setLevel(logging.INFO)

        # Add file handler
        log_file = self.output_dir / "pipeline.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # Pipeline state
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run(
        self,
        skip_agents: Optional[List[int]] = None,
        stop_after: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            skip_agents: List of agent numbers to skip (e.g., [4] to skip trajectory)
            stop_after: Stop after this agent number (e.g., 3 to stop after pathway)

        Returns:
            Dictionary with results from all agents
        """
        self.start_time = datetime.now()
        skip_agents = skip_agents or []

        self.logger.info("=" * 70)
        self.logger.info("Starting Single-Cell 6-Agent Pipeline")
        self.logger.info(f"Input: {self.input_dir}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("=" * 70)

        try:
            # Agent 1: QC & Preprocessing
            if 1 not in skip_agents:
                self._run_agent1()
            if stop_after == 1:
                return self._finalize_results()

            # Agent 2: Clustering & Annotation
            if 2 not in skip_agents:
                self._run_agent2()
            if stop_after == 2:
                return self._finalize_results()

            # Agent 3: Pathway & Validation
            if 3 not in skip_agents:
                self._run_agent3()
            if stop_after == 3:
                return self._finalize_results()

            # Agent 4: Trajectory & Dynamics
            if 4 not in skip_agents:
                self._run_agent4()
            if stop_after == 4:
                return self._finalize_results()

            # Agent 5: CNV & ML Prediction
            if 5 not in skip_agents:
                self._run_agent5()
            if stop_after == 5:
                return self._finalize_results()

            # Agent 6: Report Generation
            if 6 not in skip_agents:
                self._run_agent6()

            return self._finalize_results()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.results['error'] = str(e)
            return self._finalize_results()

    def _report_progress(self, agent_name: str, percent: int, message: str):
        """Report progress to callback if available."""
        self.logger.info(f"[{agent_name}] {percent}% - {message}")
        if self.progress_callback:
            self.progress_callback(agent_name, percent, message)

    def _run_agent1(self):
        """Run Agent 1: QC & Preprocessing."""
        self._report_progress("Agent1_QC", 0, "Starting QC & Preprocessing")

        agent1_output = self.output_dir / "agent1_qc"
        agent1_output.mkdir(exist_ok=True)

        # Get agent-specific config
        agent_config = self._get_agent_config(1)

        agent = SingleCellQCAgent(
            input_dir=self.input_dir,
            output_dir=agent1_output,
            config=agent_config
        )

        if not agent.validate_inputs():
            raise ValueError("Agent 1 input validation failed")

        self._report_progress("Agent1_QC", 50, "Running QC pipeline")
        result = agent.run()

        self.results['agent1'] = result
        self._report_progress("Agent1_QC", 100, f"QC complete: {result.get('n_cells', 0)} cells")

    def _run_agent2(self):
        """Run Agent 2: Clustering & Annotation."""
        self._report_progress("Agent2_Cluster", 0, "Starting Clustering & Annotation")

        agent2_output = self.output_dir / "agent2_cluster"
        agent2_output.mkdir(exist_ok=True)

        # Input from Agent 1
        agent1_output = self.output_dir / "agent1_qc"

        agent_config = self._get_agent_config(2)

        agent = SingleCellClusterAgent(
            input_dir=agent1_output,
            output_dir=agent2_output,
            config=agent_config
        )

        if not agent.validate_inputs():
            raise ValueError("Agent 2 input validation failed")

        self._report_progress("Agent2_Cluster", 50, "Clustering and annotating")
        result = agent.run()

        self.results['agent2'] = result
        self._report_progress(
            "Agent2_Cluster", 100,
            f"Found {result.get('n_clusters', 0)} clusters, {result.get('n_cell_types', 0)} cell types"
        )

    def _run_agent3(self):
        """Run Agent 3: Pathway & Validation."""
        self._report_progress("Agent3_Pathway", 0, "Starting Pathway Analysis")

        agent3_output = self.output_dir / "agent3_pathway"
        agent3_output.mkdir(exist_ok=True)

        # Input from Agent 2
        agent2_output = self.output_dir / "agent2_cluster"

        agent_config = self._get_agent_config(3)

        agent = SingleCellPathwayAgent(
            input_dir=agent2_output,
            output_dir=agent3_output,
            config=agent_config
        )

        if not agent.validate_inputs():
            raise ValueError("Agent 3 input validation failed")

        self._report_progress("Agent3_Pathway", 50, "Running pathway enrichment")
        result = agent.run()

        self.results['agent3'] = result
        self._report_progress(
            "Agent3_Pathway", 100,
            f"Found {result.get('n_pathways_found', 0)} pathways, {result.get('n_driver_genes', 0)} drivers"
        )

    def _run_agent4(self):
        """Run Agent 4: Trajectory & Dynamics."""
        self._report_progress("Agent4_Trajectory", 0, "Starting Trajectory Analysis")

        agent4_output = self.output_dir / "agent4_trajectory"
        agent4_output.mkdir(exist_ok=True)

        # Input from Agent 3 (or 2)
        agent3_output = self.output_dir / "agent3_pathway"
        if not agent3_output.exists():
            agent3_output = self.output_dir / "agent2_cluster"

        agent_config = self._get_agent_config(4)

        agent = SingleCellTrajectoryAgent(
            input_dir=agent3_output,
            output_dir=agent4_output,
            config=agent_config
        )

        if not agent.validate_inputs():
            raise ValueError("Agent 4 input validation failed")

        self._report_progress("Agent4_Trajectory", 50, "Computing trajectory and pseudotime")
        result = agent.run()

        self.results['agent4'] = result
        self._report_progress(
            "Agent4_Trajectory", 100,
            f"Trajectory: pseudotime={result.get('has_pseudotime', False)}, PAGA={result.get('has_paga', False)}"
        )

    def _run_agent5(self):
        """Run Agent 5: CNV & ML Prediction."""
        self._report_progress("Agent5_CNV_ML", 0, "Starting CNV & ML Prediction")

        agent5_output = self.output_dir / "agent5_cnv_ml"
        agent5_output.mkdir(exist_ok=True)

        # Input from Agent 4 (or 3/2)
        for name in ["agent4_trajectory", "agent3_pathway", "agent2_cluster"]:
            input_dir = self.output_dir / name
            if input_dir.exists():
                break

        agent_config = self._get_agent_config(5)

        agent = SingleCellCNVMLAgent(
            input_dir=input_dir,
            output_dir=agent5_output,
            config=agent_config
        )

        if not agent.validate_inputs():
            raise ValueError("Agent 5 input validation failed")

        self._report_progress("Agent5_CNV_ML", 30, "ML #2: Predicting cancer type")
        self._report_progress("Agent5_CNV_ML", 60, "ML #3: Detecting malignant cells")

        result = agent.run()

        self.results['agent5'] = result

        cancer_pred = result.get('cancer_prediction') or {}
        n_mal = result.get('n_malignant_cells', 0)
        self._report_progress(
            "Agent5_CNV_ML", 100,
            f"Predicted: {cancer_pred.get('predicted_type', 'Skipped')}, {n_mal} malignant cells"
        )

    def _run_agent6(self):
        """Run Agent 6: Report Generation."""
        self._report_progress("Agent6_Report", 0, "Starting Report Generation")

        agent6_output = self.output_dir / "report"
        agent6_output.mkdir(exist_ok=True)

        # Collect all outputs for report
        self._consolidate_outputs(agent6_output)

        agent_config = self._get_agent_config(6)
        agent_config.update({
            "report_title": self.config.get("report_title", "Single-Cell RNA-seq Analysis Report"),
            "cancer_type": self.config.get("cancer_type", "Unknown"),
            "sample_id": self.config.get("sample_id", "Sample"),
            "language": self.config.get("language", "ko")
        })

        agent = SingleCellReportAgent(
            input_dir=agent6_output,
            output_dir=agent6_output,
            config=agent_config
        )

        if not agent.validate_inputs():
            raise ValueError("Agent 6 input validation failed")

        self._report_progress("Agent6_Report", 50, "Generating visualizations and HTML report")
        result = agent.run()

        self.results['agent6'] = result
        self._report_progress("Agent6_Report", 100, "Report generated successfully")

    def _consolidate_outputs(self, output_dir: Path):
        """Copy all outputs from previous agents to report directory."""
        # List of files to copy
        files_to_copy = [
            "qc_statistics.json",
            "cell_cycle_info.json",
            "celltype_predictions.json",
            "cluster_markers.csv",
            "top_markers_summary.csv",
            "cell_composition.csv",
            "umap_coordinates.csv",
            "cluster_pathways.csv",
            "driver_genes.csv",
            "tme_scores.csv",
            "pathway_summary.json",
            "trajectory_results.json",
            "pseudotime_values.csv",
            "cancer_prediction.json",
            "cnv_results.json",
            "cnv_scores.csv",
            "malignant_cells.csv",
            "malignant_results.json",
        ]

        # H5ad files to check (in priority order)
        h5ad_files = [
            "adata_cnv.h5ad",
            "adata_trajectory.h5ad",
            "adata_clustered.h5ad",
            "adata_qc.h5ad"
        ]

        # Search in agent directories
        for agent_dir in ["agent5_cnv_ml", "agent4_trajectory", "agent3_pathway", "agent2_cluster", "agent1_qc"]:
            src_dir = self.output_dir / agent_dir
            if not src_dir.exists():
                continue

            # Copy regular files
            for filename in files_to_copy:
                src_file = src_dir / filename
                if src_file.exists() and not (output_dir / filename).exists():
                    shutil.copy(src_file, output_dir / filename)

            # Copy figures
            src_figures = src_dir / "figures"
            if src_figures.exists():
                dst_figures = output_dir / "figures"
                dst_figures.mkdir(exist_ok=True)
                for fig_file in src_figures.glob("*"):
                    if not (dst_figures / fig_file.name).exists():
                        shutil.copy(fig_file, dst_figures / fig_file.name)

        # Copy h5ad (prefer most processed version)
        for h5ad_name in h5ad_files:
            for agent_dir in ["agent5_cnv_ml", "agent4_trajectory", "agent3_pathway", "agent2_cluster", "agent1_qc"]:
                h5ad_path = self.output_dir / agent_dir / h5ad_name
                if h5ad_path.exists():
                    shutil.copy(h5ad_path, output_dir / h5ad_name)
                    break
            else:
                continue
            break

    def _get_agent_config(self, agent_num: int) -> Dict[str, Any]:
        """Get configuration for specific agent."""
        # Base config
        agent_config = {}

        # Agent-specific config keys
        config_keys = {
            1: ["min_genes_per_cell", "max_genes_per_cell", "max_mito_percent",
                "enable_doublet_detection", "enable_cell_cycle_scoring",
                "n_top_genes", "hvg_flavor"],
            2: ["clustering_resolution", "clustering_method", "annotation_method",
                "celltypist_model", "batch_key", "use_harmony"],
            3: ["pathway_databases", "pathway_top_genes", "enable_driver_matching",
                "enable_tme_scoring"],
            4: ["trajectory_method", "root_cluster", "compute_pseudotime",
                "compute_velocity"],
            5: ["enable_cancer_prediction", "enable_cnv_inference",
                "enable_malignant_detection", "model_dir"],
            6: ["report_title", "cancer_type", "sample_id", "language",
                "generate_interactive_plots"]
        }

        # Extract relevant config
        for key in config_keys.get(agent_num, []):
            if key in self.config:
                agent_config[key] = self.config[key]

        return agent_config

    def _finalize_results(self) -> Dict[str, Any]:
        """Finalize and save results."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        self.results['pipeline'] = {
            "status": "success" if 'error' not in self.results else "failed",
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
            "config": self.config,
            "output_dir": str(self.output_dir)
        }

        # Save pipeline results
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self.logger.info("=" * 70)
        self.logger.info(f"Pipeline complete in {duration:.1f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 70)

        return self.results

    def get_report_path(self) -> Optional[Path]:
        """Get path to generated HTML report."""
        report_path = self.output_dir / "report" / "singlecell_report.html"
        if report_path.exists():
            return report_path
        return None


def run_singlecell_pipeline(
    input_dir: str,
    output_dir: str,
    cancer_type: str = "Unknown",
    sample_id: str = "Sample",
    language: str = "ko",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run the single-cell pipeline.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        cancer_type: Cancer type for context
        sample_id: Sample identifier
        language: Report language ("ko" or "en")
        **kwargs: Additional configuration options

    Returns:
        Pipeline results dictionary
    """
    config = {
        "cancer_type": cancer_type,
        "sample_id": sample_id,
        "language": language,
        **kwargs
    }

    orchestrator = SingleCellOrchestrator(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        config=config
    )

    return orchestrator.run()
