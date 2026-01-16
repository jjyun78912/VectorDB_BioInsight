"""
RNA-seq Pipeline Orchestrator

Coordinates the execution of RNA-seq analysis pipelines.
Automatically detects data type (bulk vs single-cell) and routes to appropriate pipeline.

Usage:
    from rnaseq_pipeline import RNAseqPipeline

    pipeline = RNAseqPipeline(
        input_dir="./data",
        output_dir="./results",
        config={"cancer_type": "lung_cancer"}
    )

    # Run full pipeline (auto-detect bulk vs single-cell)
    results = pipeline.run()

    # Force specific pipeline type
    pipeline = RNAseqPipeline(..., pipeline_type="bulk")  # or "singlecell"

    # Or run specific agents (bulk only)
    pipeline.run_agent("agent1_deg")
    pipeline.run_from("agent3_pathway")  # Resume from agent 3

Bulk RNA-seq (2-Step, 6 Agents):
    DEG → Network → Pathway → Validation → Visualization → Report

Single-cell RNA-seq (1-Step):
    SingleCellAgent (QC → Clustering → Annotation → DEG → Report)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import logging

from .agents import (
    DEGAgent,
    NetworkAgent,
    PathwayAgent,
    ValidationAgent,
    VisualizationAgent,
    ReportAgent,
    SingleCellAgent
)
from .utils.data_type_detector import DataTypeDetector, detect_data_type


class RNAseqPipeline:
    """Orchestrator for the RNA-seq analysis pipeline."""

    # Bulk RNA-seq pipeline (6 agents)
    BULK_AGENT_ORDER = [
        "agent1_deg",
        "agent2_network",
        "agent3_pathway",
        "agent4_validation",
        "agent5_visualization",
        "agent6_report"
    ]

    BULK_AGENT_CLASSES = {
        "agent1_deg": DEGAgent,
        "agent2_network": NetworkAgent,
        "agent3_pathway": PathwayAgent,
        "agent4_validation": ValidationAgent,
        "agent5_visualization": VisualizationAgent,
        "agent6_report": ReportAgent
    }

    # Single-cell pipeline (1 agent)
    SINGLECELL_AGENT_ORDER = ["singlecell"]
    SINGLECELL_AGENT_CLASSES = {"singlecell": SingleCellAgent}

    # Legacy compatibility
    AGENT_ORDER = BULK_AGENT_ORDER
    AGENT_CLASSES = BULK_AGENT_CLASSES

    # Define which outputs each agent needs from previous agents
    AGENT_DEPENDENCIES = {
        "agent1_deg": [],
        "agent2_network": ["normalized_counts.csv", "deg_significant.csv"],
        "agent3_pathway": ["deg_significant.csv"],
        "agent4_validation": ["deg_significant.csv", "hub_genes.csv", "gene_to_pathway.csv"],
        "agent5_visualization": [
            "deg_all_results.csv", "deg_significant.csv", "normalized_counts.csv",
            "hub_genes.csv", "network_edges.csv", "pathway_summary.csv",
            "integrated_gene_table.csv"
        ],
        "agent6_report": ["*"],  # All outputs
        "singlecell": []  # Single-cell is self-contained
    }

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        pipeline_type: Optional[Literal["bulk", "singlecell", "auto"]] = "auto"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.pipeline_type_override = pipeline_type

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging first (needed by _determine_pipeline_type)
        self.logger = self._setup_logging()

        # Detect data type (requires logger)
        self.detection_result = None
        self.pipeline_type = self._determine_pipeline_type()

        # Track execution state
        self.execution_state = {
            "run_id": timestamp,
            "start_time": None,
            "end_time": None,
            "completed_agents": [],
            "failed_agents": [],
            "agent_results": {}
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup pipeline-level logging."""
        logger = logging.getLogger("rnaseq_pipeline")
        logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.run_dir / "pipeline.log"
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _determine_pipeline_type(self) -> str:
        """Determine pipeline type based on data or override."""
        # If explicitly specified
        if self.pipeline_type_override and self.pipeline_type_override != "auto":
            self.logger.info(f"Pipeline type override: {self.pipeline_type_override}")
            return self.pipeline_type_override

        # Auto-detect
        try:
            self.detection_result = detect_data_type(self.input_dir)
            detected_type = self.detection_result.get("data_type", "bulk")

            self.logger.info(f"Auto-detected data type: {detected_type}")
            self.logger.info(f"Detection confidence: {self.detection_result.get('confidence', 0):.2f}")
            self.logger.info(f"Samples/Cells: {self.detection_result.get('n_samples', 0)}")

            return detected_type
        except Exception as e:
            self.logger.warning(f"Detection failed, defaulting to bulk: {e}")
            return "bulk"

    def get_agent_order(self) -> List[str]:
        """Get agent order based on pipeline type."""
        if self.pipeline_type == "singlecell":
            return self.SINGLECELL_AGENT_ORDER
        return self.BULK_AGENT_ORDER

    def get_agent_classes(self) -> Dict:
        """Get agent classes based on pipeline type."""
        if self.pipeline_type == "singlecell":
            return self.SINGLECELL_AGENT_CLASSES
        return self.BULK_AGENT_CLASSES

    def _get_agent_input_dir(self, agent_name: str) -> Path:
        """Determine input directory for an agent."""
        # Single-cell agent uses input directly
        if agent_name == "singlecell":
            return self.input_dir

        # Bulk: first agent uses input
        if agent_name == "agent1_deg":
            return self.input_dir

        # For subsequent agents, use accumulated outputs
        return self.run_dir / "accumulated"

    def _accumulate_outputs(self, agent_name: str) -> None:
        """Copy agent outputs to accumulated directory for next agents."""
        accumulated_dir = self.run_dir / "accumulated"
        accumulated_dir.mkdir(exist_ok=True)

        agent_output_dir = self.run_dir / agent_name

        if not agent_output_dir.exists():
            return

        # Copy all CSV and JSON files
        for pattern in ["*.csv", "*.json"]:
            for f in agent_output_dir.glob(pattern):
                dest = accumulated_dir / f.name
                if not dest.exists():  # Don't overwrite
                    shutil.copy2(f, dest)

        # Copy figures directory if exists
        figures_dir = agent_output_dir / "figures"
        if figures_dir.exists():
            dest_figures = accumulated_dir / "figures"
            if dest_figures.exists():
                shutil.rmtree(dest_figures)
            shutil.copytree(figures_dir, dest_figures)

    def _copy_initial_inputs(self) -> None:
        """Copy initial input files to accumulated directory."""
        accumulated_dir = self.run_dir / "accumulated"
        accumulated_dir.mkdir(exist_ok=True)

        # Copy input files
        for f in self.input_dir.glob("*.csv"):
            dest = accumulated_dir / f.name
            shutil.copy2(f, dest)

        for f in self.input_dir.glob("*.json"):
            dest = accumulated_dir / f.name
            shutil.copy2(f, dest)

    def run_agent(self, agent_name: str, config_override: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a single agent."""
        agent_classes = self.get_agent_classes()

        if agent_name not in agent_classes:
            raise ValueError(f"Unknown agent: {agent_name}")

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Running {agent_name} (pipeline: {self.pipeline_type})")
        self.logger.info(f"{'='*60}")

        # Merge configs
        agent_config = {**self.config, **(config_override or {})}

        # Get directories
        input_dir = self._get_agent_input_dir(agent_name)
        output_dir = self.run_dir / agent_name

        # Instantiate and run
        AgentClass = agent_classes[agent_name]
        agent = AgentClass(
            input_dir=input_dir,
            output_dir=output_dir,
            config=agent_config
        )

        try:
            results = agent.execute()
            self.execution_state["completed_agents"].append(agent_name)
            self.execution_state["agent_results"][agent_name] = results

            # Accumulate outputs for next agents
            self._accumulate_outputs(agent_name)

            return results

        except Exception as e:
            self.logger.error(f"Agent {agent_name} failed: {e}")
            self.execution_state["failed_agents"].append(agent_name)
            raise

    def run(self, stop_after: Optional[str] = None) -> Dict[str, Any]:
        """Run the full pipeline or until a specific agent."""
        self.execution_state["start_time"] = datetime.now().isoformat()
        self.execution_state["pipeline_type"] = self.pipeline_type
        self.execution_state["detection_result"] = self.detection_result

        self.logger.info("Starting RNA-seq Pipeline")
        self.logger.info(f"Pipeline type: {self.pipeline_type}")
        self.logger.info(f"Run directory: {self.run_dir}")

        # Copy initial inputs
        self._copy_initial_inputs()

        # Get appropriate agent order
        agent_order = self.get_agent_order()

        # Determine which agents to run
        if stop_after:
            stop_idx = agent_order.index(stop_after) + 1
            agents_to_run = agent_order[:stop_idx]
        else:
            agents_to_run = agent_order

        self.logger.info(f"Agents to run: {agents_to_run}")

        # Run agents
        for agent_name in agents_to_run:
            try:
                self.run_agent(agent_name)
            except Exception as e:
                self.logger.error(f"Pipeline stopped at {agent_name}: {e}")
                break

        # Finalize
        self.execution_state["end_time"] = datetime.now().isoformat()
        self._save_execution_state()

        self.logger.info(f"{'='*60}")
        self.logger.info("Pipeline Complete")
        self.logger.info(f"Completed: {len(self.execution_state['completed_agents'])} agents")
        self.logger.info(f"Failed: {len(self.execution_state['failed_agents'])} agents")
        self.logger.info(f"Results: {self.run_dir}")
        self.logger.info(f"{'='*60}")

        return self.execution_state

    def run_from(self, agent_name: str) -> Dict[str, Any]:
        """Resume pipeline from a specific agent (bulk only)."""
        agent_order = self.get_agent_order()

        if agent_name not in agent_order:
            raise ValueError(f"Unknown agent: {agent_name}")

        start_idx = agent_order.index(agent_name)
        agents_to_run = agent_order[start_idx:]

        self.logger.info(f"Resuming from {agent_name}")

        for agent in agents_to_run:
            try:
                self.run_agent(agent)
            except Exception as e:
                self.logger.error(f"Pipeline stopped at {agent}: {e}")
                break

        self._save_execution_state()
        return self.execution_state

    def run_singlecell(self) -> Dict[str, Any]:
        """Run single-cell pipeline explicitly."""
        self.pipeline_type = "singlecell"
        return self.run()

    def _save_execution_state(self) -> None:
        """Save execution state to JSON."""
        state_file = self.run_dir / "pipeline_summary.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_state, f, indent=2, default=str)


def create_sample_data(output_dir: Path, n_genes: int = 1000, n_samples: int = 10) -> None:
    """Create sample data for testing the pipeline."""
    import numpy as np
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Generate gene names (mix of known cancer genes and random)
    known_genes = ['TP53', 'KRAS', 'EGFR', 'MYC', 'BRCA1', 'BRCA2', 'PIK3CA',
                   'PTEN', 'RB1', 'APC', 'BRAF', 'CDK4', 'CDKN2A', 'ERBB2',
                   'ALK', 'ROS1', 'MET', 'VEGFA', 'CD274', 'CTLA4']
    random_genes = [f'GENE{i}' for i in range(n_genes - len(known_genes))]
    genes = known_genes + random_genes

    # Generate sample names
    n_tumor = n_samples // 2
    n_normal = n_samples - n_tumor
    tumor_samples = [f'Tumor_{i+1}' for i in range(n_tumor)]
    normal_samples = [f'Normal_{i+1}' for i in range(n_normal)]
    samples = tumor_samples + normal_samples

    # Generate count matrix with clear differential expression
    # Higher base expression for more reliable statistics
    base_counts = np.random.negative_binomial(20, 0.05, size=(len(genes), len(samples)))

    # Add strong differential expression for known genes in tumor samples
    for i, gene in enumerate(genes[:20]):  # First 20 are known genes
        # Strong fold changes
        fold_change = [4, 5, 6, 0.15, 0.2, 0.25][i % 6]
        for j, sample in enumerate(samples):
            if 'Tumor' in sample:
                base_counts[i, j] = int(base_counts[i, j] * fold_change)

    # Add some moderately differentially expressed random genes
    for i in range(30, 80):  # 50 additional DEGs
        fold_change = np.random.choice([2.5, 3, 0.33, 0.4])
        for j, sample in enumerate(samples):
            if 'Tumor' in sample:
                base_counts[i, j] = int(base_counts[i, j] * fold_change)

    # Create DataFrame
    count_df = pd.DataFrame(base_counts, columns=samples)
    count_df.insert(0, 'gene_id', genes)

    # Create metadata
    meta_df = pd.DataFrame({
        'sample_id': samples,
        'condition': ['tumor'] * n_tumor + ['normal'] * n_normal,
        'batch': ['batch1'] * (n_samples // 2) + ['batch2'] * (n_samples - n_samples // 2)
    })

    # Save files
    count_df.to_csv(output_dir / 'count_matrix.csv', index=False)
    meta_df.to_csv(output_dir / 'metadata.csv', index=False)

    # Save config
    config = {
        "contrast": ["tumor", "normal"],
        "cancer_type": "lung_cancer",
        "padj_cutoff": 0.05,
        "log2fc_cutoff": 1.0
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Sample data created in {output_dir}")
    print(f"  - count_matrix.csv: {len(genes)} genes x {len(samples)} samples")
    print(f"  - metadata.csv: {len(samples)} samples")
    print(f"  - config.json: analysis configuration")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RNA-seq Analysis Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--cancer-type", "-c", default="lung_cancer", help="Cancer type")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    parser.add_argument("--agent", help="Run specific agent only")
    parser.add_argument("--from-agent", help="Resume from specific agent")

    args = parser.parse_args()

    if args.create_sample:
        create_sample_data(Path(args.input))
    else:
        config = {"cancer_type": args.cancer_type}
        pipeline = RNAseqPipeline(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            config=config
        )

        if args.agent:
            pipeline.run_agent(args.agent)
        elif args.from_agent:
            pipeline.run_from(args.from_agent)
        else:
            pipeline.run()
