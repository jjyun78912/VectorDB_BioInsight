"""
Tests for the Paper Agent module.

Uses mocking to avoid external dependencies (ChromaDB, Gemini API).
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# Test JunkContentValidator
# ═══════════════════════════════════════════════════════════════

class TestJunkContentValidator:
    """Test suite for JunkContentValidator."""

    @pytest.fixture
    def validator(self):
        """Import validator class."""
        from backend.app.core.paper_agent import JunkContentValidator
        return JunkContentValidator

    def test_detects_author_contributions(self, validator):
        """Test detection of author contributions section."""
        text = """Author Contributions: A.K. contributed to writing, editing,
        review, conceptualization, and methodology. B.L. handled validation
        and investigation."""
        assert validator.is_junk(text) is True

    def test_detects_reference_format(self, validator):
        """Test detection of reference-style content."""
        text = "1. Smith AB, Jones CD. A study on cancer research. Nature 2024."
        assert validator.is_junk(text) is True

    def test_allows_normal_content(self, validator):
        """Test that normal scientific content passes."""
        text = """The results show that KRAS mutations were found in 85% of
        pancreatic cancer samples. This finding suggests a key role in
        tumor progression."""
        assert validator.is_junk(text) is False

    def test_allows_methods_section(self, validator):
        """Test that methods section with single keywords passes."""
        text = """We used RNA sequencing methodology to analyze gene expression
        patterns in tumor samples collected from 50 patients."""
        # Only has 'methodology' - below threshold of 4
        assert validator.is_junk(text) is False

    def test_relevance_check_matches_terms(self, validator):
        """Test relevance check finds matching terms."""
        content = "KRAS mutations drive tumor growth in pancreatic cancer"
        question = "What causes tumor growth in cancer?"
        assert validator.is_relevant_to_question(content, question) is True

    def test_relevance_check_no_match(self, validator):
        """Test relevance check with no matching terms."""
        content = "The weather today is sunny and warm"
        question = "What are KRAS mutations?"
        assert validator.is_relevant_to_question(content, question) is False

    def test_relevance_check_ignores_stop_words(self, validator):
        """Test that stop words are ignored in relevance check."""
        content = "some random content about biology genes"
        question = "What are the genes in this study?"
        # 'what', 'are', 'the', 'this', 'study' are stop words
        # 'genes' should match
        assert validator.is_relevant_to_question(content, question) is True


# ═══════════════════════════════════════════════════════════════
# Test ConfidenceEstimator
# ═══════════════════════════════════════════════════════════════

class TestConfidenceEstimator:
    """Test suite for ConfidenceEstimator."""

    @pytest.fixture
    def estimator(self):
        """Import estimator class."""
        from backend.app.core.paper_agent import ConfidenceEstimator
        return ConfidenceEstimator

    def test_base_confidence(self, estimator):
        """Test base confidence for neutral answer."""
        answer = "The study found significant results in gene expression."
        context = "Gene expression analysis..."
        confidence = estimator.estimate(answer, context)
        assert confidence == pytest.approx(0.7, abs=0.01)

    def test_uncertainty_lowers_confidence(self, estimator):
        """Test that uncertainty phrases lower confidence."""
        answer = "I cannot find specific information about this topic."
        context = "Some context..."
        confidence = estimator.estimate(answer, context)
        assert confidence < 0.7

    def test_citations_increase_confidence(self, estimator):
        """Test that citations increase confidence."""
        answer = "The results show significant findings [1][2][3]."
        context = "Some context..."
        confidence = estimator.estimate(answer, context)
        assert confidence > 0.7

    def test_confidence_capped_at_1(self, estimator):
        """Test that confidence is capped at 1.0."""
        answer = "Results are clear [1][2][3][4][5][6][7][8]."
        context = "Some context..."
        confidence = estimator.estimate(answer, context)
        assert confidence <= 1.0

    def test_confidence_minimum_zero(self, estimator):
        """Test that confidence doesn't go below 0."""
        answer = "I cannot find this. It may not be mentioned. Possibly unclear."
        context = ""
        confidence = estimator.estimate(answer, context)
        assert confidence >= 0.0


# ═══════════════════════════════════════════════════════════════
# Test PromptTemplates
# ═══════════════════════════════════════════════════════════════

class TestPromptTemplates:
    """Test suite for PromptTemplates."""

    @pytest.fixture
    def templates(self):
        """Import templates class."""
        from backend.app.core.paper_agent import PromptTemplates
        return PromptTemplates

    def test_has_qa_prompts(self, templates):
        """Test that QA prompts exist."""
        assert hasattr(templates, 'QA_SYSTEM_PROMPT')
        assert hasattr(templates, 'QA_HUMAN_PROMPT')
        assert "{context}" in templates.QA_HUMAN_PROMPT
        assert "{question}" in templates.QA_HUMAN_PROMPT

    def test_has_english_summary_prompts(self, templates):
        """Test English summary prompts."""
        assert "en" in templates.SUMMARIZE_PROMPTS
        assert "system" in templates.SUMMARIZE_PROMPTS["en"]
        assert "human" in templates.SUMMARIZE_PROMPTS["en"]

    def test_has_korean_summary_prompts(self, templates):
        """Test Korean summary prompts."""
        assert "ko" in templates.SUMMARIZE_PROMPTS
        assert "한국어" in templates.SUMMARIZE_PROMPTS["ko"]["system"]


# ═══════════════════════════════════════════════════════════════
# Test PaperSession dataclass
# ═══════════════════════════════════════════════════════════════

class TestPaperSession:
    """Test suite for PaperSession dataclass."""

    def test_session_creation(self):
        """Test creating a paper session."""
        from backend.app.core.paper_agent import PaperSession
        session = PaperSession(
            session_id="abc123",
            paper_title="Test Paper",
            collection_name="test_collection",
            chunks_count=10
        )
        assert session.session_id == "abc123"
        assert session.paper_title == "Test Paper"
        assert session.chunks_count == 10
        assert session.created_at is not None


# ═══════════════════════════════════════════════════════════════
# Test AgentResponse dataclass
# ═══════════════════════════════════════════════════════════════

class TestAgentResponse:
    """Test suite for AgentResponse dataclass."""

    def test_response_creation(self):
        """Test creating an agent response."""
        from backend.app.core.paper_agent import AgentResponse
        response = AgentResponse(
            answer="This is the answer.",
            sources=[{"section": "Results", "excerpt": "..."}],
            confidence=0.85,
            is_answerable=True
        )
        assert response.answer == "This is the answer."
        assert len(response.sources) == 1
        assert response.confidence == 0.85
        assert response.is_answerable is True


# ═══════════════════════════════════════════════════════════════
# Test PaperAgent (with mocking)
# ═══════════════════════════════════════════════════════════════

class TestPaperAgent:
    """Test suite for PaperAgent class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock()
        mock_store.count = 10
        mock_store.search.return_value = []
        mock_store.add_chunks.return_value = 5
        return mock_store

    @pytest.fixture
    def mock_search_result(self):
        """Create mock search result."""
        from backend.app.core.vector_store import SearchResult
        return SearchResult(
            content="KRAS mutations were found in 85% of samples.",
            metadata={"section": "Results", "paper_title": "Test Paper"},
            relevance_score=75.0,
            distance=0.25
        )

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_agent_initialization(self, mock_vs_class):
        """Test agent initialization."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        assert agent.session_id == "test123"
        assert agent.paper_title == "Test Paper"
        assert agent.collection_name == "paper_session_test123"

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_add_chunks(self, mock_vs_class):
        """Test adding chunks to agent."""
        mock_store = Mock()
        mock_store.add_chunks.return_value = 5
        mock_vs_class.return_value = mock_store

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        chunks = [{"content": "chunk1"}, {"content": "chunk2"}]
        count = agent.add_chunks(chunks)

        mock_store.add_chunks.assert_called_once()
        assert count == 5

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_query_no_results(self, mock_vs_class):
        """Test query when no results found."""
        mock_store = Mock()
        mock_store.search.return_value = []
        mock_vs_class.return_value = mock_store

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        response = agent.query("What is KRAS?")

        assert response.is_answerable is False
        assert response.confidence == 0.0
        assert len(response.sources) == 0

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_filter_results_removes_junk(self, mock_vs_class):
        """Test that filter_results removes junk content."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import PaperAgent
        from backend.app.core.vector_store import SearchResult

        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        # Create mix of good and junk results
        results = [
            SearchResult(
                content="KRAS mutations found in samples.",
                metadata={"section": "Results"},
                relevance_score=80.0,
                distance=0.2
            ),
            SearchResult(
                content="Author: writing, editing, review, conceptualization, methodology, validation",
                metadata={"section": "Author Contributions"},
                relevance_score=70.0,
                distance=0.3
            )
        ]

        filtered = agent._filter_results(results, "What is KRAS?")

        # Should only have the non-junk result
        assert len(filtered) == 1
        assert "KRAS" in filtered[0].content

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_build_context(self, mock_vs_class):
        """Test context building from results."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import PaperAgent
        from backend.app.core.vector_store import SearchResult

        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        results = [
            SearchResult(
                content="First result content.",
                metadata={"section": "Introduction"},
                relevance_score=80.0,
                distance=0.2
            ),
            SearchResult(
                content="Second result content.",
                metadata={"section": "Results"},
                relevance_score=70.0,
                distance=0.3
            )
        ]

        context = agent._build_context(results)

        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "Introduction" in context
        assert "Results" in context

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_get_session_info(self, mock_vs_class):
        """Test getting session info."""
        mock_store = Mock()
        mock_store.count = 15
        mock_vs_class.return_value = mock_store

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        info = agent.get_session_info()

        assert info["session_id"] == "test123"
        assert info["paper_title"] == "Test Paper"
        assert info["chunks_count"] == 15

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_cleanup(self, mock_vs_class):
        """Test session cleanup."""
        mock_store = Mock()
        mock_vs_class.return_value = mock_store

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        agent.cleanup()

        mock_store.reset.assert_called_once()

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_parse_summary_response_json(self, mock_vs_class):
        """Test parsing JSON summary response."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        json_response = '''```json
        {
            "summary": "This is a summary.",
            "key_findings": ["Finding 1", "Finding 2"],
            "methodology": "RNA-seq analysis"
        }
        ```'''

        result = agent._parse_summary_response(json_response)

        assert result["summary"] == "This is a summary."
        assert len(result["key_findings"]) == 2
        assert result["methodology"] == "RNA-seq analysis"

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_parse_summary_response_invalid_json(self, mock_vs_class):
        """Test parsing invalid JSON falls back gracefully."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import PaperAgent
        agent = PaperAgent(session_id="test123", paper_title="Test Paper")

        invalid_response = "This is just plain text, not JSON."

        result = agent._parse_summary_response(invalid_response)

        assert "This is just plain text" in result["summary"]
        assert result["key_findings"] == []


# ═══════════════════════════════════════════════════════════════
# Test Session Manager Functions
# ═══════════════════════════════════════════════════════════════

class TestSessionManager:
    """Test suite for session manager functions."""

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_create_paper_session(self, mock_vs_class):
        """Test creating a paper session."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import create_paper_session, _sessions

        session_id = create_paper_session("Test Paper")

        assert session_id is not None
        assert len(session_id) == 8  # UUID[:8]
        assert session_id in _sessions

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_get_paper_agent(self, mock_vs_class):
        """Test getting an existing paper agent."""
        mock_vs_class.return_value = Mock()

        from backend.app.core.paper_agent import (
            create_paper_session, get_paper_agent
        )

        session_id = create_paper_session("Test Paper")
        agent = get_paper_agent(session_id)

        assert agent is not None
        assert agent.paper_title == "Test Paper"

    def test_get_paper_agent_not_found(self):
        """Test getting non-existent agent."""
        from backend.app.core.paper_agent import get_paper_agent

        agent = get_paper_agent("nonexistent")

        assert agent is None

    @patch('backend.app.core.paper_agent.BioVectorStore')
    def test_delete_paper_session(self, mock_vs_class):
        """Test deleting a paper session."""
        mock_store = Mock()
        mock_vs_class.return_value = mock_store

        from backend.app.core.paper_agent import (
            create_paper_session, delete_paper_session, get_paper_agent
        )

        session_id = create_paper_session("Test Paper")
        delete_paper_session(session_id)

        assert get_paper_agent(session_id) is None
        mock_store.reset.assert_called()
