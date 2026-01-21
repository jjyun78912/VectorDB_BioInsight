"""
Single-cell RNA-seq Analysis - 6 Agent Architecture

This package implements a modular 6-agent pipeline for single-cell RNA-seq analysis:

- Agent 1 (QC): Quality control, filtering, normalization
- Agent 2 (Cluster): Clustering, cell type annotation, CellTypist ML
- Agent 3 (Pathway): Pathway enrichment, database validation
- Agent 4 (Trajectory): Pseudotime, trajectory inference, RNA velocity
- Agent 5 (CNV/ML): CNV inference, cancer type prediction, malignancy detection
- Agent 6 (Report): Visualization and HTML report generation

Each agent follows the BaseAgent interface and can be run independently or
orchestrated through SingleCellOrchestrator.
"""

from .agent1_qc import SingleCellQCAgent
from .agent2_cluster import SingleCellClusterAgent
from .agent3_pathway import SingleCellPathwayAgent
from .agent4_trajectory import SingleCellTrajectoryAgent
from .agent5_cnv_ml import SingleCellCNVMLAgent
from .agent6_report import SingleCellReportAgent
from .orchestrator import SingleCellOrchestrator

__all__ = [
    'SingleCellQCAgent',
    'SingleCellClusterAgent',
    'SingleCellPathwayAgent',
    'SingleCellTrajectoryAgent',
    'SingleCellCNVMLAgent',
    'SingleCellReportAgent',
    'SingleCellOrchestrator'
]
