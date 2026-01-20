"""
RNA-seq Pipeline Orchestrator

Coordinates the execution of RNA-seq analysis pipelines.
Automatically detects data type and routes to appropriate pipeline.

Usage:
    from rnaseq_pipeline import RNAseqPipeline

    pipeline = RNAseqPipeline(
        input_dir="./data",
        output_dir="./results",
        config={"cancer_type": "lung_cancer"}
    )

    # Run full pipeline (auto-detect data type)
    results = pipeline.run()

    # Force specific pipeline type
    pipeline = RNAseqPipeline(..., pipeline_type="bulk")  # or "singlecell", "multiomic"

    # Or run specific agents (bulk only)
    pipeline.run_agent("agent1_deg")
    pipeline.run_from("agent3_pathway")  # Resume from agent 3

Pipeline Types:
================

1. Bulk RNA-seq (1-Step, Expression Only):
   DEG → Network → Pathway → Validation → Visualization → Report
   => Driver gene PREDICTION only (DB matching)

2. Bulk + WGS/WES (2-Step, Multi-omic):
   Step 1: Bulk RNA-seq pipeline
   Step 2: Variant Analysis → Integrated Driver Analysis
   => Driver gene IDENTIFICATION (mutation + expression)

3. Single-cell RNA-seq (1-Step):
   SingleCellAgent (QC → Clustering → Annotation → DEG) → Report
   => Cell type specific expression
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
    SingleCellAgent,
    SingleCellReportAgent,
    VariantAgent,
    IntegratedDriverAgent,
)
from .utils.data_type_detector import DataTypeDetector, detect_data_type


class RNAseqPipeline:
    """Orchestrator for the RNA-seq analysis pipeline."""

    # =========================================================================
    # Pipeline Type 1: Bulk RNA-seq (1-Step, Expression Only)
    # => Driver gene PREDICTION only (no mutation data)
    # =========================================================================
    BULK_AGENT_ORDER = [
        "agent1_deg",
        "agent2_network",
        "agent3_pathway",
        "agent4_validation",      # Validation Stage 1: DEG/Network/Pathway
        "agent5_visualization",
        "agent4_validation_ml",   # Validation Stage 2: ML Prediction (after ML runs)
        "agent6_report"
    ]

    BULK_AGENT_CLASSES = {
        "agent1_deg": DEGAgent,
        "agent2_network": NetworkAgent,
        "agent3_pathway": PathwayAgent,
        "agent4_validation": ValidationAgent,
        "agent4_validation_ml": ValidationAgent,  # Same class, different config
        "agent5_visualization": VisualizationAgent,
        "agent6_report": ReportAgent
    }

    # =========================================================================
    # Pipeline Type 2: Bulk + WGS/WES (2-Step, Multi-omic)
    # => Driver gene IDENTIFICATION (mutation + expression evidence)
    # =========================================================================
    MULTIOMIC_AGENT_ORDER = [
        # Step 1: Bulk RNA-seq analysis
        "agent1_deg",
        "agent2_network",
        "agent3_pathway",
        "agent4_validation",
        "agent5_visualization",
        # Step 2: Variant analysis + Integration
        "agent_variant",
        "agent_integrated_driver",
        # Final report
        "agent6_report"
    ]

    MULTIOMIC_AGENT_CLASSES = {
        "agent1_deg": DEGAgent,
        "agent2_network": NetworkAgent,
        "agent3_pathway": PathwayAgent,
        "agent4_validation": ValidationAgent,
        "agent5_visualization": VisualizationAgent,
        "agent_variant": VariantAgent,
        "agent_integrated_driver": IntegratedDriverAgent,
        "agent6_report": ReportAgent
    }

    # =========================================================================
    # Pipeline Type 3: Single-cell RNA-seq (1-Step)
    # =========================================================================
    SINGLECELL_AGENT_ORDER = ["singlecell", "singlecell_report"]
    SINGLECELL_AGENT_CLASSES = {
        "singlecell": SingleCellAgent,
        "singlecell_report": SingleCellReportAgent
    }

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
        "agent4_validation_ml": ["cancer_prediction.json", "integrated_gene_table.csv"],
        "agent_variant": [],  # Uses VCF/MAF from input
        "agent_integrated_driver": ["deg_significant.csv", "hub_genes.csv", "driver_mutations.csv"],
        "agent6_report": ["*"],  # All outputs
        "singlecell": [],
        "singlecell_report": ["cluster_markers.csv", "cell_composition.csv"]
    }

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        pipeline_type: Optional[Literal["bulk", "singlecell", "multiomic", "auto"]] = "auto"
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
        """Determine pipeline type based on data or override.

        Pipeline Types:
        - bulk: Expression-only analysis, driver gene PREDICTION
        - multiomic: Bulk + WGS/WES, driver gene IDENTIFICATION
        - singlecell: Single-cell analysis
        """
        # If explicitly specified
        if self.pipeline_type_override and self.pipeline_type_override != "auto":
            self.logger.info(f"Pipeline type override: {self.pipeline_type_override}")
            return self.pipeline_type_override

        # Check for VCF/MAF files (indicates multi-omic data)
        vcf_files = list(self.input_dir.glob("*.vcf")) + list(self.input_dir.glob("*.vcf.gz"))
        maf_files = list(self.input_dir.glob("*.maf")) + list(self.input_dir.glob("*.maf.gz"))

        has_variant_data = len(vcf_files) > 0 or len(maf_files) > 0

        if has_variant_data:
            self.logger.info(f"Detected variant data files:")
            for f in vcf_files + maf_files:
                self.logger.info(f"  - {f.name}")
            self.logger.info("Selecting MULTIOMIC pipeline (Bulk + WGS/WES)")
            self.detection_result = {
                "data_type": "multiomic",
                "vcf_files": [str(f) for f in vcf_files],
                "maf_files": [str(f) for f in maf_files],
                "confidence": 1.0
            }
            return "multiomic"

        # Auto-detect bulk vs single-cell
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
        elif self.pipeline_type == "multiomic":
            return self.MULTIOMIC_AGENT_ORDER
        return self.BULK_AGENT_ORDER

    def get_agent_classes(self) -> Dict:
        """Get agent classes based on pipeline type."""
        if self.pipeline_type == "singlecell":
            return self.SINGLECELL_AGENT_CLASSES
        elif self.pipeline_type == "multiomic":
            return self.MULTIOMIC_AGENT_CLASSES
        return self.BULK_AGENT_CLASSES

    def _get_agent_input_dir(self, agent_name: str) -> Path:
        """Determine input directory for an agent."""
        # Single-cell agent uses input directly
        if agent_name == "singlecell":
            return self.input_dir

        # Bulk: first agent uses input
        if agent_name == "agent1_deg":
            return self.input_dir

        # Variant agent needs access to original VCF/MAF files from input
        if agent_name == "agent_variant":
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

        # Special handling for validation_ml (2nd validation stage)
        if agent_name == "agent4_validation_ml":
            agent_config["validation_stage"] = 2
            agent_config["validate_ml_prediction"] = True
            self.logger.info("Running Validation Stage 2: ML Prediction validation")

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

    def _predict_cancer_type(self) -> Optional[Dict[str, Any]]:
        """
        Predict cancer type using Pan-Cancer ML model.

        This is called at the START of the pipeline to:
        1. Classify the tumor samples into one of 17 cancer types
        2. Store prediction results for Driver analysis and Report
        3. Update config with predicted cancer_type (or validate user-specified type)

        Returns:
            Prediction results dict or None if prediction fails
        """
        # Check if cancer_type is already specified by user
        user_specified_cancer = self.config.get('cancer_type', 'unknown')
        is_user_specified = user_specified_cancer and user_specified_cancer.lower() != 'unknown'

        if is_user_specified:
            self.logger.info(f"User-specified cancer type: {user_specified_cancer}")
            self.logger.info("Running ML prediction for validation...")

        try:
            from .ml.pancancer_classifier import PanCancerClassifier
            import pandas as pd
        except ImportError as e:
            self.logger.warning(f"Pan-Cancer classifier not available: {e}")
            return None

        # Check if model exists
        model_dir = Path(__file__).parent.parent / "models" / "rnaseq" / "pancancer"
        if not model_dir.exists():
            self.logger.warning(f"Pan-Cancer model not found at {model_dir}")
            return None

        # Load count matrix
        accumulated_dir = self.run_dir / "accumulated"
        count_file = accumulated_dir / "count_matrix.csv"
        if not count_file.exists():
            count_file = self.input_dir / "count_matrix.csv"

        if not count_file.exists():
            self.logger.warning("Count matrix not found for cancer type prediction")
            return None

        self.logger.info("=" * 60)
        self.logger.info("Running Cancer Type Prediction (Pan-Cancer ML Model)")
        self.logger.info("=" * 60)

        try:
            # Load count matrix
            counts = pd.read_csv(count_file, index_col=0)
            self.logger.info(f"Loaded count matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")

            # Filter to tumor samples only (if metadata available)
            metadata_file = accumulated_dir / "metadata.csv"
            if not metadata_file.exists():
                metadata_file = self.input_dir / "metadata.csv"

            tumor_samples = None
            if metadata_file.exists():
                metadata = pd.read_csv(metadata_file)
                condition_col = self.config.get('condition_column', 'condition')
                contrast = self.config.get('contrast', ['tumor', 'normal'])
                if condition_col in metadata.columns:
                    tumor_cond = contrast[0] if isinstance(contrast, list) else 'tumor'
                    sample_col = 'sample_id' if 'sample_id' in metadata.columns else metadata.columns[0]
                    tumor_samples = metadata[metadata[condition_col] == tumor_cond][sample_col].tolist()
                    self.logger.info(f"Found {len(tumor_samples)} tumor samples")

            # Use tumor samples if available
            if tumor_samples:
                available_samples = [s for s in tumor_samples if s in counts.columns]
                if available_samples:
                    counts = counts[available_samples]
                    self.logger.info(f"Using {len(available_samples)} tumor samples for prediction")

            # Run prediction
            classifier = PanCancerClassifier(str(model_dir))
            classifier.load()

            results = classifier.predict(counts, top_k=5)

            # Aggregate results (majority vote or highest confidence)
            if results:
                # Count predictions
                from collections import Counter
                predictions = [r.predicted_cancer for r in results if not r.is_unknown]

                if predictions:
                    cancer_counts = Counter(predictions)
                    predicted_cancer, count = cancer_counts.most_common(1)[0]

                    # Get average confidence for the predicted cancer
                    confidences = [r.confidence for r in results if r.predicted_cancer == predicted_cancer]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                    # Get cancer info
                    cancer_info = classifier.cancer_info.get(predicted_cancer, {})
                    korean_name = cancer_info.get('korean', predicted_cancer)

                    prediction_result = {
                        'predicted_cancer': predicted_cancer,
                        'predicted_cancer_korean': korean_name,
                        'confidence': avg_confidence,
                        'sample_count': len(results),
                        'agreement_count': count,
                        'agreement_ratio': count / len(predictions) if predictions else 0,
                        'top_k_summary': [
                            {
                                'cancer': c,
                                'count': cnt,
                                'ratio': cnt / len(predictions)
                            }
                            for c, cnt in cancer_counts.most_common(5)
                        ],
                        'all_results': [r.to_dict() for r in results]  # All samples for condition inference
                    }

                    # Handle user-specified vs ML-predicted cancer type
                    if is_user_specified:
                        # Keep user-specified cancer type but record ML prediction for validation
                        prediction_result['user_specified_cancer'] = user_specified_cancer
                        prediction_result['ml_predicted_cancer'] = predicted_cancer
                        prediction_result['prediction_matches_user'] = (
                            user_specified_cancer.upper() == predicted_cancer.upper()
                        )

                        # Don't override user-specified cancer type
                        self.config['cancer_prediction'] = prediction_result

                        if prediction_result['prediction_matches_user']:
                            self.logger.info(f"✅ ML Prediction MATCHES user-specified: {predicted_cancer} ({korean_name})")
                        else:
                            self.logger.warning(f"⚠️ ML Prediction DIFFERS from user-specified!")
                            self.logger.warning(f"   User specified: {user_specified_cancer}")
                            self.logger.warning(f"   ML predicted: {predicted_cancer} ({korean_name})")

                        self.logger.info(f"   Confidence: {avg_confidence:.2%}")
                        self.logger.info(f"   Agreement: {count}/{len(predictions)} samples ({count/len(predictions)*100:.1f}%)")
                    else:
                        # Update config with predicted cancer type (no user specification)
                        self.config['cancer_type'] = predicted_cancer
                        self.config['cancer_type_korean'] = korean_name
                        self.config['cancer_prediction'] = prediction_result

                        self.logger.info(f"🎯 Predicted Cancer Type: {predicted_cancer} ({korean_name})")
                        self.logger.info(f"   Confidence: {avg_confidence:.2%}")
                        self.logger.info(f"   Agreement: {count}/{len(predictions)} samples ({count/len(predictions)*100:.1f}%)")

                    # Save prediction results
                    prediction_file = self.run_dir / "cancer_prediction.json"
                    with open(prediction_file, 'w', encoding='utf-8') as f:
                        json.dump(prediction_result, f, indent=2, ensure_ascii=False)

                    return prediction_result
                else:
                    self.logger.warning("All samples classified as UNKNOWN")
            else:
                self.logger.warning("No prediction results returned")

        except Exception as e:
            self.logger.error(f"Cancer type prediction failed: {e}")
            import traceback
            traceback.print_exc()

        return None

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

        # ===== STEP 0: Predict Cancer Type =====
        # Run cancer type prediction BEFORE any agents
        cancer_prediction = self._predict_cancer_type()
        if cancer_prediction:
            self.execution_state["cancer_prediction"] = cancer_prediction
            self.logger.info(f"Cancer type set to: {self.config.get('cancer_type')}")

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

    def run_multiomic(self) -> Dict[str, Any]:
        """Run multi-omic pipeline explicitly (Bulk RNA-seq + WGS/WES).

        This pipeline combines:
        1. Bulk RNA-seq analysis (DEG, Network, Pathway, Validation, Visualization)
        2. WGS/WES variant analysis (VCF/MAF parsing, driver mutation detection)
        3. Integrated driver identification (mutation + expression evidence)

        Returns:
            Execution state with integrated driver analysis results
        """
        self.pipeline_type = "multiomic"
        self.logger.info("=" * 60)
        self.logger.info("MULTI-OMIC PIPELINE: Bulk RNA-seq + WGS/WES Integration")
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Bulk RNA-seq analysis (Expression-based)")
        self.logger.info("Step 2: Variant analysis (Mutation-based)")
        self.logger.info("Step 3: Integrated driver identification")
        self.logger.info("=" * 60)
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
    parser.add_argument("--pipeline", "-p", choices=["auto", "bulk", "singlecell", "multiomic"],
                       default="auto", help="Pipeline type (auto-detected by default)")
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
            config=config,
            pipeline_type=args.pipeline
        )

        if args.agent:
            pipeline.run_agent(args.agent)
        elif args.from_agent:
            pipeline.run_from(args.from_agent)
        else:
            pipeline.run()
