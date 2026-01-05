"""
News Prioritizer - Ranking and headline selection for newsletter

Prioritizes news across multiple sources:
1. FDA (Regulatory) - Highest priority for approvals
2. ClinicalTrials.gov (Clinical) - Phase 3 results
3. bioRxiv/medRxiv (Preprints) - Breaking research
4. PubMed (Research) - High-impact journals
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PrioritizedNews:
    """News item with priority scoring."""
    source: str
    category: str  # regulatory, clinical, research, preprint
    title: str
    summary: str
    link: str
    date: str
    priority_score: int
    is_headline_candidate: bool
    metadata: Dict[str, Any]


class NewsPrioritizer:
    """Prioritizes and ranks news from multiple sources."""

    # Priority weights by source
    SOURCE_WEIGHTS = {
        "FDA": 100,
        "ClinicalTrials": 80,
        "PubMed": 60,
        "bioRxiv": 50,
        "medRxiv": 50,
    }

    # Priority weights by news type
    TYPE_WEIGHTS = {
        # FDA types
        "drug_approval": 150,
        "biologics_approval": 150,
        "rejection": 120,
        "safety_warning": 130,
        "designation": 80,

        # Clinical trial types
        "phase3_positive": 140,
        "phase3_negative": 130,
        "phase3_completed": 100,
        "new_trial": 70,
        "stopped": 110,

        # Research types
        "research": 50,
        "preprint": 40,
    }

    # Hot keywords that boost priority
    HOT_KEYWORDS = [
        "glp-1", "semaglutide", "tirzepatide", "ozempic", "wegovy", "mounjaro",
        "crispr", "gene therapy", "gene editing",
        "car-t", "car t", "cell therapy",
        "mrna", "vaccine",
        "antibody-drug conjugate", "adc",
        "alzheimer", "lecanemab", "donanemab",
        "checkpoint inhibitor", "pd-1", "pd-l1", "keytruda", "opdivo",
        "bispecific", "obesity", "diabetes",
        "first-in-class", "breakthrough", "novel",
    ]

    # Big pharma companies
    BIG_PHARMA = [
        "pfizer", "novartis", "roche", "lilly", "eli lilly", "novo nordisk",
        "merck", "bristol-myers", "bms", "astrazeneca", "johnson & johnson",
        "j&j", "sanofi", "gsk", "glaxosmithkline", "abbvie", "gilead",
        "amgen", "moderna", "biontech", "regeneron", "vertex",
    ]

    # High-impact journals
    HIGH_IMPACT_JOURNALS = [
        "nature", "science", "cell", "lancet",
        "new england journal of medicine", "nejm",
        "jama", "bmj", "nature medicine", "nature genetics",
        "nature biotechnology", "cell stem cell",
        "journal of clinical oncology", "blood",
        "circulation", "gut", "annals of oncology",
    ]

    def calculate_priority(self, news: Dict[str, Any]) -> int:
        """
        Calculate priority score for a news item.

        Args:
            news: News item dictionary with source, type, title, etc.

        Returns:
            Priority score (higher is more important)
        """
        score = 0
        text = f"{news.get('title', '')} {news.get('summary', '')}".lower()

        # 1. Source weight
        source = news.get("source", "")
        score += self.SOURCE_WEIGHTS.get(source, 20)

        # 2. Type weight
        news_type = news.get("type", "")
        score += self.TYPE_WEIGHTS.get(news_type, 10)

        # 3. Hot keyword bonus
        for keyword in self.HOT_KEYWORDS:
            if keyword in text:
                score += 30
                break

        # 4. Big pharma bonus
        for company in self.BIG_PHARMA:
            if company in text:
                score += 25
                break

        # 5. High-impact journal bonus (for research)
        journal = news.get("metadata", {}).get("journal", "").lower()
        for hj in self.HIGH_IMPACT_JOURNALS:
            if hj in journal:
                score += 40
                break

        # 6. "First" or "breakthrough" bonus
        if "first" in text or "breakthrough" in text or "novel" in text:
            score += 50

        # 7. Recency bonus (within 24 hours)
        date_str = news.get("date", "")
        if date_str:
            try:
                news_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                days_old = (datetime.now() - news_date).days
                if days_old == 0:
                    score += 30
                elif days_old <= 1:
                    score += 20
                elif days_old <= 3:
                    score += 10
            except:
                pass

        return score

    def prioritize_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize a list of news items.

        Args:
            news_items: List of news item dictionaries

        Returns:
            List sorted by priority (highest first)
        """
        for news in news_items:
            news["priority_score"] = self.calculate_priority(news)

        return sorted(news_items, key=lambda x: x.get("priority_score", 0), reverse=True)

    def select_headline(
        self,
        regulatory: List[Dict[str, Any]],
        clinical: List[Dict[str, Any]],
        research: List[Dict[str, Any]],
        preprints: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most important news as headline.

        Priority order:
        1. FDA approvals
        2. FDA safety warnings
        3. Major clinical trial results
        4. Breakthrough research
        5. Top preprint

        Args:
            regulatory: FDA news items
            clinical: Clinical trial items
            research: Research papers
            preprints: Preprint papers

        Returns:
            Best headline candidate or None
        """
        # Priority 1: FDA approvals
        approvals = [
            n for n in regulatory
            if "approval" in n.get("type", "").lower()
        ]
        if approvals:
            prioritized = self.prioritize_news(approvals)
            return prioritized[0] if prioritized else None

        # Priority 2: FDA safety warnings / rejections
        warnings = [
            n for n in regulatory
            if any(w in n.get("type", "").lower() for w in ["warning", "safety", "reject"])
        ]
        if warnings:
            prioritized = self.prioritize_news(warnings)
            return prioritized[0] if prioritized else None

        # Priority 3: Phase 3 positive results
        phase3_positive = [
            n for n in clinical
            if "positive" in n.get("type", "").lower()
        ]
        if phase3_positive:
            prioritized = self.prioritize_news(phase3_positive)
            return prioritized[0] if prioritized else None

        # Priority 4: Any clinical trial results
        if clinical:
            prioritized = self.prioritize_news(clinical)
            return prioritized[0] if prioritized else None

        # Priority 5: High-impact research
        if research:
            prioritized = self.prioritize_news(research)
            return prioritized[0] if prioritized else None

        # Priority 6: Top preprint
        if preprints:
            prioritized = self.prioritize_news(preprints)
            return prioritized[0] if prioritized else None

        # Fallback: Any regulatory news
        if regulatory:
            prioritized = self.prioritize_news(regulatory)
            return prioritized[0] if prioritized else None

        return None

    def categorize_news(
        self,
        news_items: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize news items by type.

        Args:
            news_items: List of news items

        Returns:
            Dictionary with categories as keys
        """
        categories = {
            "regulatory": [],
            "clinical": [],
            "research": [],
            "preprints": [],
        }

        for news in news_items:
            source = news.get("source", "")
            news_type = news.get("type", "")

            if source == "FDA":
                categories["regulatory"].append(news)
            elif source == "ClinicalTrials":
                categories["clinical"].append(news)
            elif source in ["bioRxiv", "medRxiv"]:
                categories["preprints"].append(news)
            elif "preprint" in news_type.lower():
                categories["preprints"].append(news)
            else:
                categories["research"].append(news)

        # Sort each category by priority
        for category in categories:
            categories[category] = self.prioritize_news(categories[category])

        return categories


# Utility function for newsletter data conversion
def convert_to_newsletter_format(briefing_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert aggregator output to newsletter generator format.

    Args:
        briefing_data: Output from NewsAggregator.aggregate_daily()

    Returns:
        Data formatted for NewsletterGenerator.generate()
    """
    prioritizer = NewsPrioritizer()

    # Convert regulatory news
    regulatory = []
    for news in briefing_data.get("regulatory", []):
        status_map = {
            "drug_approval": "approved",
            "biologics_approval": "approved",
            "safety_warning": "warning",
            "rejection": "rejected",
            "designation": "fast_track",
        }
        regulatory.append({
            "status": status_map.get(news.get("type", ""), "pending"),
            "title": news.get("title", ""),
            "description": news.get("summary", "")[:200],
        })

    # Convert clinical trials
    clinical_trials = []
    for key in ["phase3_results", "new_trials", "terminated"]:
        for trial in briefing_data.get("clinical_trials", {}).get(key, []):
            type_map = {
                "phase3_results": "phase3_positive",
                "new_trials": "new_trial",
                "terminated": "stopped",
            }
            clinical_trials.append({
                "type": type_map.get(key, "phase3_completed"),
                "title": trial.get("title", ""),
                "description": trial.get("summary", "")[:200],
            })

    # Convert research papers
    research = []
    for key in ["high_impact", "preprints"]:
        for paper in briefing_data.get("research", {}).get(key, []):
            journal = paper.get("metadata", {}).get("journal", "")
            if not journal:
                journal = paper.get("source", "")
            research.append({
                "journal": journal,
                "title": paper.get("title", ""),
                "insight": paper.get("summary", "")[:150],
            })

    # Convert hot topics
    hot_topics = []
    for topic in briefing_data.get("hot_topics", []):
        change = getattr(topic, 'week_change', 0) if hasattr(topic, 'week_change') else topic.get('week_change', 0)
        hot_topics.append({
            "name": topic.get("keyword", "") if isinstance(topic, dict) else getattr(topic, 'keyword', ''),
            "count": topic.get("count", 0) if isinstance(topic, dict) else getattr(topic, 'count', 0),
            "change": int(change),
            "event": None,
            "event_type": None,
        })

    # Select headline
    headline_data = briefing_data.get("headline", {})
    headline = {
        "title": headline_data.get("title", "오늘의 바이오 뉴스"),
        "summary": headline_data.get("summary", ""),
    }

    # Editor comment
    editor_comment = briefing_data.get("editor_comment", "")

    return {
        "total_papers": briefing_data.get("stats", {}).get("papers", 0),
        "headline": headline,
        "regulatory": regulatory[:3],
        "clinical_trials": clinical_trials[:3],
        "research": research[:4],
        "hot_topics": hot_topics[:5],
        "editor": {
            "quote": editor_comment[:300] if editor_comment else "",
            "note": "",
        }
    }


if __name__ == "__main__":
    # Test prioritization
    test_news = [
        {
            "source": "FDA",
            "type": "drug_approval",
            "title": "FDA Approves First GLP-1 Drug for Obesity",
            "summary": "First breakthrough approval for weight loss",
            "date": "2025-01-05",
        },
        {
            "source": "ClinicalTrials",
            "type": "phase3_positive",
            "title": "Pfizer Phase 3 Trial Shows Positive Results",
            "summary": "Cancer immunotherapy trial success",
            "date": "2025-01-04",
        },
        {
            "source": "PubMed",
            "type": "research",
            "title": "CRISPR Gene Editing Study",
            "summary": "Novel gene editing approach",
            "metadata": {"journal": "Nature"},
            "date": "2025-01-03",
        },
    ]

    prioritizer = NewsPrioritizer()
    prioritized = prioritizer.prioritize_news(test_news)

    print("Prioritized News:")
    for i, news in enumerate(prioritized, 1):
        print(f"{i}. [{news['priority_score']}] {news['title'][:50]}...")
