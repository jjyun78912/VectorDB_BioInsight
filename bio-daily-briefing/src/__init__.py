"""
BIO Daily Briefing - Auto Trend Newsletter System v3

Features:
- Multi-source data aggregation (FDA, ClinicalTrials, bioRxiv, PubMed)
- Newspaper-style HTML newsletter
- PDF download support
- AI-powered editor comments
"""

from .pubmed_fetcher import PubMedFetcher, Paper
from .trend_analyzer import TrendAnalyzer, Trend
from .ai_summarizer import AISummarizer, NewsArticle
from .newsletter_generator import NewsletterGenerator
from .prioritizer import NewsPrioritizer, convert_to_newsletter_format

__all__ = [
    "PubMedFetcher",
    "Paper",
    "TrendAnalyzer",
    "Trend",
    "AISummarizer",
    "NewsArticle",
    "NewsletterGenerator",
    "NewsPrioritizer",
    "convert_to_newsletter_format",
]

__version__ = "3.0.0"
