"""
RNA-seq Cancer Analysis Pipeline

A modular pipeline for RNA-seq analysis with 6 specialized agents:
1. DEG Analysis (DESeq2)
2. Network Analysis (Hub Gene Detection)
3. Pathway Enrichment (GO/KEGG)
4. DB Validation & Interpretation
5. Visualization
6. Report Generation

Each agent has clear input/output specs and can be run independently.
"""

__version__ = "1.0.0"
__author__ = "BioInsight AI"
