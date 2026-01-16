"""
RNA-seq Pipeline Agents

Each agent handles a specific step of the analysis:

Bulk RNA-seq (2-Step):
- Agent 1: DEG Analysis (DESeq2)
- Agent 2: Network Analysis (Hub genes)
- Agent 3: Pathway Enrichment (GO/KEGG)
- Agent 4: DB Validation & Interpretation
- Agent 5: Visualization
- Agent 6: Report Generation

Single-cell RNA-seq (1-Step):
- SingleCellAgent: Scanpy-based unified pipeline
"""

from .agent1_deg import DEGAgent
from .agent2_network import NetworkAgent
from .agent3_pathway import PathwayAgent
from .agent4_validation import ValidationAgent
from .agent5_visualization import VisualizationAgent
from .agent6_report import ReportAgent
from .agent_singlecell import SingleCellAgent

__all__ = [
    # Bulk RNA-seq agents
    "DEGAgent",
    "NetworkAgent",
    "PathwayAgent",
    "ValidationAgent",
    "VisualizationAgent",
    "ReportAgent",
    # Single-cell agent
    "SingleCellAgent"
]
