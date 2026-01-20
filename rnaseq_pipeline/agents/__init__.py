"""
RNA-seq Pipeline Agents

Each agent handles a specific step of the analysis:

Bulk RNA-seq (1-Step, Expression Only):
- Agent 1: DEG Analysis (DESeq2)
- Agent 2: Network Analysis (Hub genes)
- Agent 3: Pathway Enrichment (GO/KEGG)
- Agent 4: DB Validation & Interpretation
- Agent 5: Visualization
- Agent 6: Report Generation
=> Driver gene PREDICTION only (no mutation data)

Bulk RNA-seq + WGS/WES (2-Step, Multi-omic):
- Step 1: Bulk RNA-seq pipeline (above)
- Step 2: Variant Analysis + Integrated Driver Analysis
=> Driver gene IDENTIFICATION (mutation + expression evidence)

Single-cell RNA-seq (1-Step):
- SingleCellAgent: Scanpy-based unified pipeline
- SingleCellReportAgent: Report generation

WGS/WES Variant Analysis:
- VariantAgent: Somatic variant analysis from VCF/MAF
- IntegratedDriverAgent: Multi-omic driver identification
"""

from .agent1_deg import DEGAgent
from .agent2_network import NetworkAgent
from .agent3_pathway import PathwayAgent
from .agent4_validation import ValidationAgent
from .agent5_visualization import VisualizationAgent
from .agent6_report import ReportAgent
from .agent_singlecell import SingleCellAgent
from .agent_singlecell_report import SingleCellReportAgent
from .agent_variant import VariantAgent
from .agent_integrated_driver import IntegratedDriverAgent

__all__ = [
    # Bulk RNA-seq agents
    "DEGAgent",
    "NetworkAgent",
    "PathwayAgent",
    "ValidationAgent",
    "VisualizationAgent",
    "ReportAgent",
    # Single-cell agents
    "SingleCellAgent",
    "SingleCellReportAgent",
    # WGS/WES agents
    "VariantAgent",
    "IntegratedDriverAgent",
]
