"""
BioInsight AI - Test Configuration and Fixtures
"""
import os
import sys
import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_count_matrix():
    """Generate a small synthetic count matrix for testing."""
    np.random.seed(42)
    n_genes = 100
    n_samples = 10

    # Generate gene names
    genes = [f"GENE{i}" for i in range(n_genes)]

    # Generate sample names (5 tumor, 5 normal)
    samples = [f"TUMOR_{i}" for i in range(5)] + [f"NORMAL_{i}" for i in range(5)]

    # Generate count data (Poisson-like)
    counts = np.random.negative_binomial(n=10, p=0.3, size=(n_genes, n_samples))

    # Add differential expression for some genes
    # First 20 genes upregulated in tumor
    counts[:20, :5] = counts[:20, :5] * 3
    # Genes 20-40 downregulated in tumor
    counts[20:40, :5] = counts[20:40, :5] // 3 + 1

    df = pd.DataFrame(counts, index=genes, columns=samples)
    df.index.name = "gene_id"
    return df


@pytest.fixture
def sample_metadata():
    """Generate matching metadata for sample_count_matrix."""
    samples = [f"TUMOR_{i}" for i in range(5)] + [f"NORMAL_{i}" for i in range(5)]
    conditions = ["tumor"] * 5 + ["normal"] * 5
    batches = ["batch1"] * 10

    return pd.DataFrame({
        "sample_id": samples,
        "condition": conditions,
        "batch": batches
    })


@pytest.fixture
def sample_config():
    """Generate sample config for pipeline."""
    return {
        "contrast": ["tumor", "normal"],
        "cancer_type": "test_cancer",
        "padj_cutoff": 0.05,
        "log2fc_cutoff": 1.0
    }


@pytest.fixture
def sample_deg_results():
    """Generate sample DEG results."""
    np.random.seed(42)
    n_genes = 50

    return pd.DataFrame({
        "gene_id": [f"GENE{i}" for i in range(n_genes)],
        "log2FoldChange": np.random.randn(n_genes) * 2,
        "padj": np.random.uniform(0, 0.1, n_genes),
        "baseMean": np.random.uniform(100, 10000, n_genes),
        "direction": ["up" if i < 25 else "down" for i in range(n_genes)]
    })


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-google")


# Skip markers for tests requiring specific resources
requires_gpu = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="GPU not available"
)

requires_api_keys = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="API keys not configured"
)

requires_r = pytest.mark.skipif(
    os.system("which R > /dev/null 2>&1") != 0,
    reason="R not installed"
)
