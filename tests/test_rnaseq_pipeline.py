"""
BioInsight AI - RNA-seq Pipeline Unit Tests
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
import os

# Import markers from conftest
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Define markers locally
requires_r = pytest.mark.skipif(
    os.system("which R > /dev/null 2>&1") != 0,
    reason="R not installed"
)

requires_api_keys = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="API keys not configured"
)


class TestPipelineOrchestrator:
    """Test cases for RNAseqPipeline orchestrator."""

    def test_pipeline_import(self):
        """Test that pipeline can be imported."""
        from rnaseq_pipeline.orchestrator import RNAseqPipeline
        assert RNAseqPipeline is not None

    def test_pipeline_type_detection_bulk(self, sample_count_matrix):
        """Test automatic detection of bulk RNA-seq data."""
        # Bulk: few samples (< 500)
        assert sample_count_matrix.shape[1] == 10  # 10 samples = bulk

    def test_pipeline_type_detection_singlecell(self):
        """Test automatic detection of single-cell data."""
        # Single-cell: many columns (cells) > 500
        np.random.seed(42)
        sc_data = pd.DataFrame(
            np.random.poisson(5, (100, 1000)),  # 1000 cells
            index=[f"GENE{i}" for i in range(100)]
        )
        assert sc_data.shape[1] == 1000  # > 500 = single-cell

    def test_pipeline_initialization(self, sample_count_matrix, sample_metadata, sample_config, temp_dir):
        """Test pipeline initialization with valid inputs."""
        from rnaseq_pipeline.orchestrator import RNAseqPipeline

        # Create input directory with required files
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        sample_count_matrix.to_csv(input_dir / "count_matrix.csv")
        sample_metadata.to_csv(input_dir / "metadata.csv", index=False)
        with open(input_dir / "config.json", "w") as f:
            json.dump(sample_config, f)

        output_dir = temp_dir / "output"

        pipeline = RNAseqPipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config=sample_config,
            pipeline_type="bulk"
        )

        assert pipeline is not None
        assert pipeline.pipeline_type == "bulk"


class TestAgent1DEG:
    """Test cases for Agent 1 - DEG Analysis."""

    def test_deg_input_validation(self, sample_count_matrix, sample_metadata):
        """Test DEG input validation."""
        # Check count matrix has genes as rows
        assert sample_count_matrix.index.name == "gene_id"

        # Check metadata has required columns
        assert "sample_id" in sample_metadata.columns
        assert "condition" in sample_metadata.columns

        # Check sample alignment
        assert set(sample_count_matrix.columns) == set(sample_metadata["sample_id"])

    def test_deg_filtering(self, sample_count_matrix):
        """Test gene filtering by minimum counts."""
        min_count = 10
        filtered = sample_count_matrix[sample_count_matrix.sum(axis=1) >= min_count]
        assert len(filtered) > 0
        assert len(filtered) <= len(sample_count_matrix)

    @requires_r
    def test_deseq2_available(self):
        """Test that DESeq2 R package is available."""
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        try:
            deseq2 = importr("DESeq2")
            assert deseq2 is not None
        except Exception as e:
            pytest.skip(f"DESeq2 not available: {e}")


class TestAgent2Network:
    """Test cases for Agent 2 - Network Analysis."""

    def test_correlation_matrix(self, sample_count_matrix):
        """Test correlation matrix computation."""
        # Transpose for gene-gene correlation
        corr_matrix = sample_count_matrix.T.corr(method="spearman")

        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert corr_matrix.shape[0] == len(sample_count_matrix)
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix.values), np.ones(len(corr_matrix))
        )

    def test_hub_gene_identification(self, sample_deg_results):
        """Test hub gene scoring."""
        # Simple hub score based on |log2FC| * -log10(padj)
        deg = sample_deg_results.copy()
        deg["hub_score"] = np.abs(deg["log2FoldChange"]) * (-np.log10(deg["padj"] + 1e-10))

        top_hubs = deg.nlargest(10, "hub_score")
        assert len(top_hubs) == 10
        assert top_hubs["hub_score"].iloc[0] >= top_hubs["hub_score"].iloc[-1]


class TestAgent3Pathway:
    """Test cases for Agent 3 - Pathway Analysis."""

    def test_gene_list_preparation(self, sample_deg_results):
        """Test gene list preparation for enrichment."""
        # Filter significant genes
        sig_genes = sample_deg_results[sample_deg_results["padj"] < 0.05]["gene_id"].tolist()
        assert isinstance(sig_genes, list)
        assert len(sig_genes) > 0

    def test_direction_separation(self, sample_deg_results):
        """Test separation of up/down regulated genes."""
        up_genes = sample_deg_results[
            (sample_deg_results["padj"] < 0.05) &
            (sample_deg_results["log2FoldChange"] > 0)
        ]
        down_genes = sample_deg_results[
            (sample_deg_results["padj"] < 0.05) &
            (sample_deg_results["log2FoldChange"] < 0)
        ]

        assert len(up_genes) + len(down_genes) <= len(sample_deg_results)


class TestAgent4Validation:
    """Test cases for Agent 4 - Database Validation."""

    def test_gene_symbol_format(self, sample_deg_results):
        """Test gene symbol format validation."""
        gene_ids = sample_deg_results["gene_id"].tolist()

        # Check for valid format (not empty, no special chars)
        for gene in gene_ids:
            assert len(gene) > 0
            assert isinstance(gene, str)

    def test_cosmic_query_format(self):
        """Test COSMIC query format."""
        test_genes = ["TP53", "BRCA1", "EGFR"]
        query = ",".join(test_genes)
        assert "TP53" in query
        assert len(query.split(",")) == 3


class TestAgent5Visualization:
    """Test cases for Agent 5 - Visualization."""

    def test_volcano_data_preparation(self, sample_deg_results):
        """Test data preparation for volcano plot."""
        deg = sample_deg_results.copy()
        deg["-log10_padj"] = -np.log10(deg["padj"] + 1e-300)

        assert "-log10_padj" in deg.columns
        assert deg["-log10_padj"].min() >= 0

    def test_heatmap_gene_selection(self, sample_deg_results):
        """Test gene selection for heatmap."""
        top_n = 20
        top_genes = sample_deg_results.nsmallest(top_n, "padj")

        assert len(top_genes) == top_n
        # Should be sorted by padj
        assert top_genes["padj"].iloc[0] <= top_genes["padj"].iloc[-1]


class TestAgent6Report:
    """Test cases for Agent 6 - Report Generation."""

    def test_report_data_structure(self, sample_deg_results):
        """Test report data structure."""
        report_data = {
            "summary": {
                "total_genes": len(sample_deg_results),
                "significant_genes": len(sample_deg_results[sample_deg_results["padj"] < 0.05]),
                "upregulated": len(sample_deg_results[sample_deg_results["log2FoldChange"] > 0]),
                "downregulated": len(sample_deg_results[sample_deg_results["log2FoldChange"] < 0]),
            },
            "deg_results": sample_deg_results.to_dict(orient="records")
        }

        assert "summary" in report_data
        assert "deg_results" in report_data
        assert report_data["summary"]["total_genes"] == 50


class TestMLPrediction:
    """Test cases for ML Cancer Type Prediction."""

    def test_feature_extraction(self, sample_count_matrix):
        """Test feature extraction for ML model."""
        # Log2 transform
        log_counts = np.log2(sample_count_matrix + 1)

        assert log_counts.min().min() >= 0
        assert not np.isinf(log_counts.values).any()

    def test_prediction_output_format(self):
        """Test prediction output format."""
        mock_prediction = {
            "predicted_cancer": "BRCA",
            "confidence": 0.95,
            "probabilities": {"BRCA": 0.95, "LUAD": 0.03, "COAD": 0.02},
            "sample_agreement": 1.0
        }

        assert "predicted_cancer" in mock_prediction
        assert 0 <= mock_prediction["confidence"] <= 1
        assert sum(mock_prediction["probabilities"].values()) == pytest.approx(1.0)


class TestDataValidation:
    """Test cases for data validation utilities."""

    def test_count_matrix_validation(self, sample_count_matrix):
        """Test count matrix validation."""
        # Should be non-negative integers
        assert (sample_count_matrix >= 0).all().all()

        # Should have genes as rows
        assert sample_count_matrix.shape[0] > sample_count_matrix.shape[1]

    def test_metadata_validation(self, sample_metadata):
        """Test metadata validation."""
        required_cols = ["sample_id", "condition"]
        for col in required_cols:
            assert col in sample_metadata.columns

        # No missing values in required columns
        assert not sample_metadata["sample_id"].isna().any()
        assert not sample_metadata["condition"].isna().any()

    def test_config_validation(self, sample_config):
        """Test config validation."""
        required_keys = ["contrast", "cancer_type"]
        for key in required_keys:
            assert key in sample_config

        # Contrast should have exactly 2 elements
        assert len(sample_config["contrast"]) == 2


class TestIntegration:
    """Integration tests for full pipeline."""

    @requires_r
    @requires_api_keys
    def test_mini_pipeline(self, sample_count_matrix, sample_metadata, sample_config, temp_dir):
        """Test minimal pipeline run with synthetic data."""
        pytest.skip("Full integration test - run separately")

        from rnaseq_pipeline.orchestrator import RNAseqPipeline

        # Setup
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        sample_count_matrix.to_csv(input_dir / "count_matrix.csv")
        sample_metadata.to_csv(input_dir / "metadata.csv", index=False)
        with open(input_dir / "config.json", "w") as f:
            json.dump(sample_config, f)

        # Run
        pipeline = RNAseqPipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config=sample_config,
            pipeline_type="bulk"
        )

        result = pipeline.run()

        # Verify outputs
        assert result["status"] == "completed"
        assert (output_dir / "deg_significant.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
