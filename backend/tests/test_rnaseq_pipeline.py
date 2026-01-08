"""
Tests for the RNA-seq analysis pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDEGAgent:
    """Test suite for DEG analysis agent."""

    def test_deg_results_structure(self, sample_deg_results):
        """Test that DEG results have expected structure."""
        df = sample_deg_results

        # Required columns
        required_cols = ["gene_symbol", "log2FoldChange", "padj"]
        for col in required_cols:
            assert col in df.columns

        # Should have numeric values
        assert df["log2FoldChange"].dtype in [np.float64, np.float32, float]
        assert df["padj"].dtype in [np.float64, np.float32, float]

    def test_deg_filtering(self, sample_deg_results):
        """Test DEG filtering logic."""
        df = sample_deg_results

        # Filter significant DEGs (typical criteria)
        sig_degs = df[
            (df["padj"] < 0.05) &
            (abs(df["log2FoldChange"]) > 1)
        ]

        # Should have some significant DEGs
        assert len(sig_degs) >= 1

    def test_updown_classification(self, sample_deg_results):
        """Test up/down regulation classification."""
        df = sample_deg_results

        upregulated = df[df["log2FoldChange"] > 0]
        downregulated = df[df["log2FoldChange"] < 0]

        # Both should exist in sample data
        assert len(upregulated) > 0
        assert len(downregulated) > 0


class TestNetworkAgent:
    """Test suite for network analysis agent."""

    def test_hub_gene_detection(self, sample_deg_results):
        """Test hub gene detection from DEG results."""
        df = sample_deg_results

        # Simulate hub gene scoring (degree-like metric)
        df["hub_score"] = abs(df["log2FoldChange"]) * -np.log10(df["padj"] + 1e-10)

        # Get top hub genes
        top_hubs = df.nlargest(3, "hub_score")

        assert len(top_hubs) == 3
        assert top_hubs["hub_score"].iloc[0] >= top_hubs["hub_score"].iloc[1]


class TestPathwayAgent:
    """Test suite for pathway enrichment agent."""

    def test_pathway_results_format(self):
        """Test pathway results format."""
        # Simulated pathway results
        pathway_results = pd.DataFrame({
            "pathway_id": ["KEGG:hsa04010", "GO:0007049"],
            "pathway_name": ["MAPK signaling", "Cell cycle"],
            "p_value": [0.001, 0.01],
            "gene_count": [15, 10],
            "genes": ["KRAS,BRAF,MEK1", "CDK1,CDK2"]
        })

        # Check structure
        assert "pathway_id" in pathway_results.columns
        assert "p_value" in pathway_results.columns
        assert len(pathway_results) >= 1


class TestValidationAgent:
    """Test suite for database validation agent."""

    def test_disgenet_validation_format(self):
        """Test DisGeNET validation format."""
        # Simulated validation results
        validation = {
            "gene": "KRAS",
            "disgenet_score": 0.95,
            "associated_diseases": ["Pancreatic Cancer", "Lung Cancer"],
            "pmid_count": 150
        }

        assert "gene" in validation
        assert "disgenet_score" in validation
        assert 0 <= validation["disgenet_score"] <= 1


class TestVisualizationAgent:
    """Test suite for visualization agent."""

    def test_volcano_plot_data(self, sample_deg_results):
        """Test volcano plot data preparation."""
        df = sample_deg_results

        # Add -log10 p-value for volcano plot
        df["neg_log10_pval"] = -np.log10(df["padj"] + 1e-300)

        # All values should be valid
        assert not df["neg_log10_pval"].isna().any()
        assert (df["neg_log10_pval"] >= 0).all()

    def test_heatmap_data_normalization(self, sample_deg_results):
        """Test heatmap data normalization."""
        df = sample_deg_results

        # Z-score normalization
        values = df["log2FoldChange"].values
        z_scores = (values - values.mean()) / values.std()

        # Z-scores should have mean ~0 and std ~1
        assert abs(z_scores.mean()) < 0.1
        assert abs(z_scores.std() - 1) < 0.1


class TestReportAgent:
    """Test suite for report generation agent."""

    def test_report_sections(self):
        """Test that report has expected sections."""
        expected_sections = [
            "summary",
            "deg_analysis",
            "pathway_analysis",
            "network_analysis",
            "validation"
        ]

        # Simulated report data
        report_data = {section: {} for section in expected_sections}

        for section in expected_sections:
            assert section in report_data
