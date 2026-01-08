"""
Pytest configuration and fixtures for BioInsight tests.
"""
import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture
def sample_paper_text():
    """Sample paper text for testing."""
    return """
    Abstract

    In this study, we investigated the role of KRAS mutations in pancreatic cancer
    development. We analyzed 150 patient samples using RNA-seq and identified
    significant gene expression changes associated with KRAS G12D mutations.

    Introduction

    Pancreatic ductal adenocarcinoma (PDAC) is one of the most lethal cancers,
    with a 5-year survival rate of less than 10%. KRAS mutations are found in
    over 90% of PDAC cases.

    Methods

    RNA was extracted from tumor samples using TRIzol reagent. Libraries were
    prepared using the Illumina TruSeq kit. Sequencing was performed on the
    NovaSeq 6000 platform.

    Results

    We identified 256 differentially expressed genes (DEGs) with log2 fold change
    > 2 and adjusted p-value < 0.05. Pathway analysis revealed enrichment in
    MAPK signaling and cell cycle regulation.

    Discussion

    Our findings suggest that KRAS-driven transcriptional changes contribute to
    the aggressive phenotype of PDAC. These results may inform targeted
    therapeutic strategies.

    References

    1. Smith et al. (2020) Cancer Cell
    2. Jones et al. (2021) Nature Medicine
    """


@pytest.fixture
def sample_deg_results():
    """Sample DEG analysis results for testing."""
    import pandas as pd
    return pd.DataFrame({
        "gene_id": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        "gene_symbol": ["KRAS", "TP53", "BRCA1", "EGFR", "MYC"],
        "log2FoldChange": [2.5, -1.8, 1.2, 3.1, -2.0],
        "padj": [0.001, 0.01, 0.05, 0.001, 0.02],
        "baseMean": [1000, 500, 750, 1200, 800]
    })


@pytest.fixture
def sample_pubmed_metadata():
    """Sample PubMed paper metadata."""
    return {
        "pmid": "12345678",
        "title": "KRAS mutations in pancreatic cancer",
        "abstract": "We investigated KRAS mutations in PDAC...",
        "authors": ["Smith J", "Jones A", "Brown K"],
        "journal": "Cancer Research",
        "year": 2024,
        "doi": "10.1158/0008-5472.CAN-23-1234",
        "keywords": ["KRAS", "pancreatic cancer", "RNA-seq"]
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
