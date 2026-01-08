"""
Tests for the RAG Pipeline module.

Uses mocking to avoid external dependencies (ChromaDB, Gemini API).
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


# ═══════════════════════════════════════════════════════════════
# Test Citation dataclass
# ═══════════════════════════════════════════════════════════════

class TestCitation:
    """Test suite for Citation dataclass."""

    def test_citation_creation(self):
        """Test creating a citation."""
        from backend.app.core.rag_pipeline import Citation

        citation = Citation(
            paper_title="KRAS in Pancreatic Cancer",
            section="Results",
            content_preview="The study found...",
            relevance_score=85.0,
            doi="10.1234/example",
            year="2024"
        )

        assert citation.paper_title == "KRAS in Pancreatic Cancer"
        assert citation.section == "Results"
        assert citation.relevance_score == 85.0

    def test_citation_format_with_doi_and_year(self):
        """Test citation formatting with DOI and year."""
        from backend.app.core.rag_pipeline import Citation

        citation = Citation(
            paper_title="Test Paper",
            section="Methods",
            content_preview="...",
            relevance_score=80.0,
            doi="10.1234/test",
            year="2024"
        )

        formatted = citation.format(1)

        assert "[1]" in formatted
        assert "Test Paper" in formatted
        assert "(2024)" in formatted
        assert "(DOI: 10.1234/test)" in formatted
        assert "Section: Methods" in formatted

    def test_citation_format_without_doi(self):
        """Test citation formatting without DOI."""
        from backend.app.core.rag_pipeline import Citation

        citation = Citation(
            paper_title="Test Paper",
            section="Methods",
            content_preview="...",
            relevance_score=80.0
        )

        formatted = citation.format(2)

        assert "[2]" in formatted
        assert "Test Paper" in formatted
        assert "DOI" not in formatted

    def test_citation_format_without_year(self):
        """Test citation formatting without year."""
        from backend.app.core.rag_pipeline import Citation

        citation = Citation(
            paper_title="Test Paper",
            section="Methods",
            content_preview="...",
            relevance_score=80.0,
            doi="10.1234/test"
        )

        formatted = citation.format(1)

        # Year should not appear as empty parentheses
        assert "Test Paper (DOI:" in formatted or "Test Paper\n" in formatted


# ═══════════════════════════════════════════════════════════════
# Test RAGResponse dataclass
# ═══════════════════════════════════════════════════════════════

class TestRAGResponse:
    """Test suite for RAGResponse dataclass."""

    def test_response_creation(self):
        """Test creating a RAG response."""
        from backend.app.core.rag_pipeline import RAGResponse, Citation

        citations = [
            Citation("Paper 1", "Results", "...", 85.0),
            Citation("Paper 2", "Methods", "...", 75.0)
        ]

        response = RAGResponse(
            answer="KRAS is a key oncogene.",
            citations=citations,
            query="What is KRAS?",
            context_used="Context..."
        )

        assert response.answer == "KRAS is a key oncogene."
        assert len(response.citations) == 2
        assert response.query == "What is KRAS?"

    def test_response_format_with_citations(self):
        """Test response formatting with citations."""
        from backend.app.core.rag_pipeline import RAGResponse, Citation

        citations = [
            Citation("Paper 1", "Results", "...", 85.0),
        ]

        response = RAGResponse(
            answer="The answer is here.",
            citations=citations
        )

        formatted = response.format()

        assert "The answer is here." in formatted
        assert "References" in formatted
        assert "[1]" in formatted

    def test_response_format_without_citations(self):
        """Test response formatting without citations."""
        from backend.app.core.rag_pipeline import RAGResponse

        response = RAGResponse(
            answer="No sources found.",
            citations=[]
        )

        formatted = response.format()

        assert "No sources found." in formatted
        assert "References" not in formatted

    def test_response_default_values(self):
        """Test response with default values."""
        from backend.app.core.rag_pipeline import RAGResponse

        response = RAGResponse(answer="Simple answer.")

        assert response.citations == []
        assert response.query == ""
        assert response.context_used == ""


# ═══════════════════════════════════════════════════════════════
# Test BioRAGPipeline (with mocking)
# ═══════════════════════════════════════════════════════════════

class TestBioRAGPipeline:
    """Test suite for BioRAGPipeline class."""

    @pytest.fixture
    def mock_search_result(self):
        """Create mock search result."""
        from backend.app.core.vector_store import SearchResult
        return SearchResult(
            content="KRAS mutations were found in 85% of samples.",
            metadata={
                "paper_title": "Pancreatic Cancer Study",
                "section": "Results",
                "year": "2024",
                "doi": "10.1234/test"
            },
            relevance_score=85.0,
            distance=0.15
        )

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_pipeline_initialization(self, mock_llm_class, mock_create_vs):
        """Test pipeline initialization."""
        mock_create_vs.return_value = Mock()
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="pancreatic_cancer",
            api_key="test-api-key"
        )

        assert pipeline.disease_domain == "pancreatic_cancer"
        mock_create_vs.assert_called_once_with(disease_domain="pancreatic_cancer")

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.GOOGLE_API_KEY', None)
    def test_pipeline_raises_without_api_key(self, mock_create_vs):
        """Test that pipeline raises error without API key."""
        mock_create_vs.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        with pytest.raises(ValueError, match="GOOGLE_API_KEY not set"):
            BioRAGPipeline(disease_domain="cancer")

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_retrieve_without_section(self, mock_llm_class, mock_create_vs):
        """Test retrieval without section filter."""
        mock_vs = Mock()
        mock_vs.search.return_value = []
        mock_create_vs.return_value = mock_vs
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        pipeline._retrieve("What is KRAS?")

        mock_vs.search.assert_called_once()

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_retrieve_with_section(self, mock_llm_class, mock_create_vs):
        """Test retrieval with section filter."""
        mock_vs = Mock()
        mock_vs.search_by_section.return_value = []
        mock_create_vs.return_value = mock_vs
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        pipeline._retrieve("What methods were used?", section="Methods")

        mock_vs.search_by_section.assert_called_once()

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_format_context_empty(self, mock_llm_class, mock_create_vs):
        """Test context formatting with no results."""
        mock_create_vs.return_value = Mock()
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        context = pipeline._format_context([])

        assert "No relevant documents found" in context

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_format_context_with_results(self, mock_llm_class, mock_create_vs, mock_search_result):
        """Test context formatting with results."""
        mock_create_vs.return_value = Mock()
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        context = pipeline._format_context([mock_search_result])

        assert "[1]" in context
        assert "Pancreatic Cancer Study" in context
        assert "(2024)" in context
        assert "Results" in context
        assert "KRAS mutations" in context

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_create_citations(self, mock_llm_class, mock_create_vs, mock_search_result):
        """Test citation creation from results."""
        mock_create_vs.return_value = Mock()
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        citations = pipeline._create_citations([mock_search_result])

        assert len(citations) == 1
        assert citations[0].paper_title == "Pancreatic Cancer Study"
        assert citations[0].section == "Results"
        assert citations[0].doi == "10.1234/test"
        assert citations[0].year == "2024"

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_query_no_results(self, mock_llm_class, mock_create_vs):
        """Test query when no results found."""
        mock_vs = Mock()
        mock_vs.search.return_value = []
        mock_create_vs.return_value = mock_vs
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        response = pipeline.query("What is KRAS?")

        assert "찾을 수 없습니다" in response.answer
        assert len(response.citations) == 0

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_query_with_results(self, mock_llm_class, mock_create_vs, mock_search_result):
        """Test query with results."""
        mock_vs = Mock()
        mock_vs.search.return_value = [mock_search_result]
        mock_create_vs.return_value = mock_vs

        # Mock the LLM chain response
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        # Mock the chain invoke
        pipeline.chain = Mock()
        pipeline.chain.invoke.return_value = "KRAS is an important oncogene."

        response = pipeline.query("What is KRAS?")

        assert response.answer == "KRAS is an important oncogene."
        assert len(response.citations) == 1
        assert response.query == "What is KRAS?"

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_query_include_context(self, mock_llm_class, mock_create_vs, mock_search_result):
        """Test query with context included."""
        mock_vs = Mock()
        mock_vs.search.return_value = [mock_search_result]
        mock_create_vs.return_value = mock_vs
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        pipeline.chain = Mock()
        pipeline.chain.invoke.return_value = "Answer here."

        response = pipeline.query("What is KRAS?", include_context=True)

        assert response.context_used != ""
        assert "Pancreatic Cancer Study" in response.context_used

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_query_with_followup(self, mock_llm_class, mock_create_vs, mock_search_result):
        """Test follow-up query with context."""
        mock_vs = Mock()
        mock_vs.search.return_value = [mock_search_result]
        mock_create_vs.return_value = mock_vs
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import BioRAGPipeline

        pipeline = BioRAGPipeline(
            disease_domain="cancer",
            api_key="test-key"
        )

        pipeline.chain = Mock()
        pipeline.chain.invoke.return_value = "Follow-up answer."

        response = pipeline.query_with_followup(
            "What about treatment?",
            previous_context="KRAS is an oncogene."
        )

        assert response.answer == "Follow-up answer."


# ═══════════════════════════════════════════════════════════════
# Test create_rag_pipeline function
# ═══════════════════════════════════════════════════════════════

class TestCreateRagPipeline:
    """Test suite for create_rag_pipeline function."""

    @patch('backend.app.core.rag_pipeline.create_vector_store')
    @patch('backend.app.core.rag_pipeline.ChatGoogleGenerativeAI')
    def test_create_pipeline(self, mock_llm_class, mock_create_vs):
        """Test convenience function for creating pipeline."""
        mock_create_vs.return_value = Mock()
        mock_llm_class.return_value = Mock()

        from backend.app.core.rag_pipeline import create_rag_pipeline

        pipeline = create_rag_pipeline(
            disease_domain="lung_cancer",
            api_key="test-key",
            top_k=10
        )

        assert pipeline.disease_domain == "lung_cancer"
        assert pipeline.top_k == 10
