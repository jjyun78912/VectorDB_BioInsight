"""
BIO Daily Briefing - Auto Trend Newsletter System
"""

from .pubmed_fetcher import PubMedFetcher, Paper
from .trend_analyzer import TrendAnalyzer, Trend
from .ai_summarizer import AISummarizer, NewsArticle
from .newsletter_generator import NewsletterGenerator, NewsletterData

__all__ = [
    "PubMedFetcher",
    "Paper",
    "TrendAnalyzer",
    "Trend",
    "AISummarizer",
    "NewsArticle",
    "NewsletterGenerator",
    "NewsletterData",
]

__version__ = "2.0.0"
