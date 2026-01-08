"""
Tests for the Web Crawler Agent module.

Uses mocking to avoid external API calls (PubMed, CrossRef, Semantic Scholar).
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import asyncio


# ═══════════════════════════════════════════════════════════════
# Test PaperScorer
# ═══════════════════════════════════════════════════════════════

class TestPaperScorer:
    """Test suite for PaperScorer class."""

    @pytest.fixture
    def scorer(self):
        """Import scorer class."""
        from backend.app.core.web_crawler_agent import PaperScorer
        return PaperScorer()

    def test_recency_score_current_year(self, scorer):
        """Test recency score for current year papers."""
        current_year = datetime.now().year
        score = scorer.calculate_recency_score(current_year)
        assert score == 100.0

    def test_recency_score_one_year_old(self, scorer):
        """Test recency score for 1-year-old papers."""
        last_year = datetime.now().year - 1
        score = scorer.calculate_recency_score(last_year)
        assert score == 80.0

    def test_recency_score_five_years_old(self, scorer):
        """Test recency score for 5-year-old papers."""
        year = datetime.now().year - 5
        score = scorer.calculate_recency_score(year)
        assert score == 0.0  # 5 * 20 = 100, so 100 - 100 = 0

    def test_recency_score_no_year(self, scorer):
        """Test recency score with no year provided."""
        score = scorer.calculate_recency_score(0)
        assert score == 0.0

    def test_citation_velocity_no_data(self, scorer):
        """Test citation velocity with no data."""
        velocity = scorer.calculate_citation_velocity({})
        assert velocity == 0.0

    def test_citation_velocity_accelerating(self, scorer):
        """Test citation velocity for accelerating citations."""
        current_year = datetime.now().year
        citations = {
            str(current_year - 3): 10,  # Older
            str(current_year - 2): 15,  # Older
            str(current_year - 1): 30,  # Recent
            str(current_year): 50,      # Recent
        }
        velocity = scorer.calculate_citation_velocity(citations)
        # Recent: 80, Older: 25 -> velocity = 80/25 = 3.2
        assert velocity > 1.0  # Accelerating

    def test_citation_velocity_decelerating(self, scorer):
        """Test citation velocity for decelerating citations."""
        current_year = datetime.now().year
        citations = {
            str(current_year - 3): 50,  # Older
            str(current_year - 2): 40,  # Older
            str(current_year - 1): 10,  # Recent
            str(current_year): 5,       # Recent
        }
        velocity = scorer.calculate_citation_velocity(citations)
        # Recent: 15, Older: 90 -> velocity = 15/90 = 0.167
        assert velocity < 1.0  # Decelerating

    def test_citation_velocity_only_recent(self, scorer):
        """Test citation velocity with only recent citations."""
        current_year = datetime.now().year
        citations = {
            str(current_year): 50,
            str(current_year - 1): 30,
        }
        velocity = scorer.calculate_citation_velocity(citations)
        # Only recent citations = very hot (new paper)
        assert velocity == 5.0

    def test_trend_score_calculation(self, scorer):
        """Test trend score calculation."""
        score = scorer.calculate_trend_score(
            citation_velocity=2.0,
            publication_surge=1.5,
            citation_count=100,
            recency_score=80.0
        )

        assert 0.0 <= score <= 100.0
        assert score > 0  # Should have a positive score

    def test_trend_score_with_no_velocity(self, scorer):
        """Test trend score when no velocity data."""
        score = scorer.calculate_trend_score(
            citation_velocity=0.0,
            publication_surge=0.0,
            citation_count=50,
            recency_score=60.0
        )

        # Falls back to citation + recency formula
        assert 0.0 <= score <= 100.0

    def test_trend_score_new_paper_no_citations(self, scorer):
        """Test trend score for new paper with no citations."""
        score = scorer.calculate_trend_score(
            citation_velocity=0.0,
            publication_surge=0.0,
            citation_count=0,
            recency_score=100.0
        )

        # Only recency contributes
        assert score == 50.0  # recency * 0.5


# ═══════════════════════════════════════════════════════════════
# Test FetchedPaper dataclass
# ═══════════════════════════════════════════════════════════════

class TestFetchedPaper:
    """Test suite for FetchedPaper dataclass."""

    def test_paper_creation(self):
        """Test creating a fetched paper."""
        from backend.app.core.web_crawler_agent import FetchedPaper

        paper = FetchedPaper(
            id="10.1234/test",
            source="pubmed",
            title="Test Paper Title",
            authors=["Author A", "Author B"],
            abstract="This is the abstract.",
            journal="Nature",
            year=datetime.now().year,
            doi="10.1234/test",
            pmid="12345678"
        )

        assert paper.title == "Test Paper Title"
        assert paper.source == "pubmed"
        assert len(paper.authors) == 2
        assert paper.fetched_at != ""

    def test_paper_recency_score_calculated(self):
        """Test that recency score is auto-calculated."""
        from backend.app.core.web_crawler_agent import FetchedPaper

        current_year = datetime.now().year
        paper = FetchedPaper(
            id="test",
            source="pubmed",
            title="Test",
            year=current_year
        )

        assert paper.recency_score == 100.0

    def test_paper_trend_score_with_citations(self):
        """Test trend score calculation with citation data."""
        from backend.app.core.web_crawler_agent import FetchedPaper

        current_year = datetime.now().year
        paper = FetchedPaper(
            id="test",
            source="pubmed",
            title="Test",
            year=current_year,
            citation_count=100,
            citations_by_year={
                str(current_year): 50,
                str(current_year - 1): 30,
            }
        )

        assert paper.trend_score > 0
        assert paper.citation_velocity > 0

    def test_paper_to_dict(self):
        """Test converting paper to dictionary."""
        from backend.app.core.web_crawler_agent import FetchedPaper

        paper = FetchedPaper(
            id="test",
            source="pubmed",
            title="Test Paper"
        )

        paper_dict = paper.to_dict()

        assert isinstance(paper_dict, dict)
        assert paper_dict["id"] == "test"
        assert paper_dict["title"] == "Test Paper"

    def test_paper_default_values(self):
        """Test paper default field values."""
        from backend.app.core.web_crawler_agent import FetchedPaper

        paper = FetchedPaper(
            id="test",
            source="crossref",
            title="Minimal Paper"
        )

        assert paper.authors == []
        assert paper.abstract == ""
        assert paper.citation_count == 0
        assert paper.is_core_paper is False


# ═══════════════════════════════════════════════════════════════
# Test WebCrawlerAgent (with mocking)
# ═══════════════════════════════════════════════════════════════

class TestWebCrawlerAgent:
    """Test suite for WebCrawlerAgent class."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        from backend.app.core.web_crawler_agent import WebCrawlerAgent
        return WebCrawlerAgent(ncbi_api_key="test-key", timeout=10)

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.ncbi_api_key == "test-key"
        assert agent.rate_limits["pubmed"] == 3  # Higher with API key

    def test_agent_initialization_no_key(self):
        """Test agent initialization without API key."""
        from backend.app.core.web_crawler_agent import WebCrawlerAgent

        agent = WebCrawlerAgent()

        assert agent.ncbi_api_key is None
        assert agent.rate_limits["pubmed"] == 1  # Lower without API key

    def test_rate_limits_configured(self, agent):
        """Test rate limits are configured correctly."""
        # With API key, pubmed rate should be 3
        assert agent.rate_limits["pubmed"] == 3
        assert agent.rate_limits["crossref"] == 5
        assert agent.rate_limits["semantic_scholar"] == 5

    def test_last_request_time_tracking(self, agent):
        """Test last request time dictionary exists."""
        assert isinstance(agent.last_request_time, dict)

    def test_cache_initialized(self, agent):
        """Test caches are initialized."""
        assert isinstance(agent._trending_cache, dict)
        assert isinstance(agent._surge_cache, dict)


# ═══════════════════════════════════════════════════════════════
# Test Trending Categories
# ═══════════════════════════════════════════════════════════════

class TestTrendingCategories:
    """Test suite for trending category configurations."""

    def test_all_categories_defined(self):
        """Test that all expected categories are defined."""
        from backend.app.core.web_crawler_agent import TRENDING_CATEGORIES

        expected_categories = [
            "oncology",
            "immunotherapy",
            "gene_therapy",
            "neurology",
            "infectious_disease",
            "ai_medicine",
            "genomics",
            "drug_discovery"
        ]

        for category in expected_categories:
            assert category in TRENDING_CATEGORIES

    def test_category_queries_have_year_filter(self):
        """Test that category queries include year filter."""
        from backend.app.core.web_crawler_agent import TRENDING_CATEGORIES

        current_year = datetime.now().year

        for category, query in TRENDING_CATEGORIES.items():
            # Should have year filter
            assert "pdat" in query or str(current_year) in query

    def test_year_filter_function(self):
        """Test dynamic year filter generation."""
        from backend.app.core.web_crawler_agent import _get_year_filter

        year_filter = _get_year_filter()
        current_year = datetime.now().year

        assert str(current_year) in year_filter
        assert str(current_year - 1) in year_filter


# ═══════════════════════════════════════════════════════════════
# Test Major Journals List
# ═══════════════════════════════════════════════════════════════

class TestMajorJournals:
    """Test suite for major journals configuration."""

    def test_major_journals_not_empty(self):
        """Test that major journals list is populated."""
        from backend.app.core.web_crawler_agent import MAJOR_JOURNALS

        assert len(MAJOR_JOURNALS) > 0

    def test_top_journals_included(self):
        """Test that top journals are included."""
        from backend.app.core.web_crawler_agent import MAJOR_JOURNALS

        top_journals = ["Nature", "Science", "Cell", "The Lancet"]

        for journal in top_journals:
            assert journal in MAJOR_JOURNALS

    def test_journal_filter_built(self):
        """Test that journal filter is built correctly."""
        from backend.app.core.web_crawler_agent import MAJOR_JOURNAL_FILTER

        assert '"Nature"[Journal]' in MAJOR_JOURNAL_FILTER
        assert " OR " in MAJOR_JOURNAL_FILTER


# ═══════════════════════════════════════════════════════════════
# Test API URL Constants
# ═══════════════════════════════════════════════════════════════

class TestAPIConstants:
    """Test suite for API URL constants."""

    def test_pubmed_urls_defined(self):
        """Test PubMed API URLs are defined."""
        from backend.app.core.web_crawler_agent import (
            PUBMED_SEARCH_URL,
            PUBMED_FETCH_URL
        )

        assert "ncbi.nlm.nih.gov" in PUBMED_SEARCH_URL
        assert "ncbi.nlm.nih.gov" in PUBMED_FETCH_URL

    def test_crossref_url_defined(self):
        """Test CrossRef API URL is defined."""
        from backend.app.core.web_crawler_agent import CROSSREF_API

        assert "crossref.org" in CROSSREF_API

    def test_semantic_scholar_url_defined(self):
        """Test Semantic Scholar API URL is defined."""
        from backend.app.core.web_crawler_agent import SEMANTIC_SCHOLAR_API

        assert "semanticscholar.org" in SEMANTIC_SCHOLAR_API
