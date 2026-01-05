"""
Trend Analyzer - Hybrid Hot Topic Analysis

2-Track System:
1. Predefined Hot Topics: 업계 주목 키워드 (GLP-1, CAR-T, CRISPR 등)
2. Emerging Trends: 급상승 키워드 자동 감지
"""

import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

from .pubmed_fetcher import Paper
from .config.hot_topics import PREDEFINED_HOT_TOPICS, is_in_predefined, EMERGING_BLACKLIST


@dataclass
class PredefinedTrend:
    """Represents a predefined hot topic trend."""
    name: str
    count: int
    previous_count: int = 0
    week_ago_count: int = 0
    category: str = ""
    why_hot: str = ""
    representative_papers: List[Paper] = field(default_factory=list)

    @property
    def week_change(self) -> float:
        """Calculate week-over-week change percentage."""
        if self.week_ago_count == 0:
            return 100.0 if self.count > 0 else 0.0
        return ((self.count - self.week_ago_count) / self.week_ago_count) * 100

    @property
    def change_label(self) -> str:
        """Get formatted change label."""
        change = self.week_change
        if change >= 50:
            return f"🔥 +{change:.0f}%"
        elif change >= 10:
            return f"⬆️ +{change:.0f}%"
        elif change <= -10:
            return f"⬇️ {change:.0f}%"
        else:
            return "➡️ 유지"

    @property
    def trend_indicator(self) -> str:
        """Get trend indicator emoji only."""
        change = self.week_change
        if change >= 50:
            return "🔥"
        elif change >= 10:
            return "⬆️"
        elif change <= -10:
            return "⬇️"
        else:
            return "➡️"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "previous_count": self.previous_count,
            "week_ago_count": self.week_ago_count,
            "week_change": round(self.week_change, 1),
            "change_label": self.change_label,
            "trend_indicator": self.trend_indicator,
            "category": self.category,
            "why_hot": self.why_hot,
            "paper_count": len(self.representative_papers),
        }


@dataclass
class EmergingTrend:
    """Represents an auto-detected emerging trend."""
    name: str
    count: int
    previous_count: int = 0
    is_new: bool = False
    representative_papers: List[Paper] = field(default_factory=list)

    @property
    def change_label(self) -> str:
        """Get formatted change label."""
        if self.is_new:
            return "🆕 신규"
        return "🔥 급상승"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "previous_count": self.previous_count,
            "is_new": self.is_new,
            "change_label": self.change_label,
            "paper_count": len(self.representative_papers),
        }


@dataclass
class Trend:
    """Legacy Trend class for backward compatibility."""
    keyword: str
    count: int
    previous_count: int = 0
    week_ago_count: int = 0
    representative_papers: List[Paper] = field(default_factory=list)
    category: str = ""
    why_hot: str = ""
    is_predefined: bool = True
    is_emerging: bool = False

    @property
    def day_change(self) -> float:
        if self.previous_count == 0:
            return 100.0 if self.count > 0 else 0.0
        return ((self.count - self.previous_count) / self.previous_count) * 100

    @property
    def is_first_tracking(self) -> bool:
        """Check if this is the first time tracking this keyword."""
        return self.week_ago_count == 0 and self.previous_count == 0

    @property
    def week_change(self) -> float:
        if self.week_ago_count == 0:
            return 0.0  # No comparison available
        return ((self.count - self.week_ago_count) / self.week_ago_count) * 100

    @property
    def trend_indicator(self) -> str:
        if self.is_emerging:
            return "🆕" if self.previous_count < 5 else "🔥"
        if self.is_first_tracking:
            return "📊"  # First tracking indicator
        change = self.week_change
        if change >= 50:
            return "🔥"
        elif change >= 10:
            return "⬆️"
        elif change <= -10:
            return "⬇️"
        else:
            return "➡️"

    @property
    def change_label(self) -> str:
        if self.is_emerging:
            return "🆕 신규 급상승" if self.previous_count < 5 else "🔥 급상승"
        if self.is_first_tracking:
            return "📊 추적 시작"  # First tracking label
        change = self.week_change
        if change >= 50:
            return f"🔥 +{change:.0f}%"
        elif change >= 10:
            return f"⬆️ +{change:.0f}%"
        elif change <= -10:
            return f"⬇️ {change:.0f}%"
        else:
            return "➡️ 유지"

    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "count": self.count,
            "previous_count": self.previous_count,
            "week_ago_count": self.week_ago_count,
            "day_change": round(self.day_change, 1),
            "week_change": round(self.week_change, 1),
            "trend_indicator": self.trend_indicator,
            "change_label": self.change_label,
            "category": self.category,
            "why_hot": self.why_hot,
            "is_predefined": self.is_predefined,
            "is_emerging": self.is_emerging,
            "paper_count": len(self.representative_papers),
        }


@dataclass
class TrendResult:
    """Combined result of hybrid trend analysis."""
    predefined_trends: List[PredefinedTrend]
    emerging_trends: List[EmergingTrend]

    def to_unified_trends(self) -> List[Trend]:
        """Convert to unified Trend list for backward compatibility."""
        trends = []

        # Add predefined trends
        for pt in self.predefined_trends:
            trends.append(Trend(
                keyword=pt.name,
                count=pt.count,
                previous_count=pt.previous_count,
                week_ago_count=pt.week_ago_count,
                representative_papers=pt.representative_papers,
                category=pt.category,
                why_hot=pt.why_hot,
                is_predefined=True,
                is_emerging=False,
            ))

        # Add emerging trends
        for et in self.emerging_trends:
            trends.append(Trend(
                keyword=et.name,
                count=et.count,
                previous_count=et.previous_count,
                week_ago_count=0,
                representative_papers=et.representative_papers,
                category="",
                why_hot="급상승 감지된 키워드",
                is_predefined=False,
                is_emerging=True,
            ))

        return trends


class TrendAnalyzer:
    """Hybrid trend analyzer with predefined + emerging detection."""

    # Stopwords for filtering
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "this", "that", "these", "those", "it", "its", "we", "our", "their",
        "study", "studies", "research", "analysis", "results", "findings",
        "patients", "cells", "using", "based", "novel", "new", "showed",
        "humans", "animals", "male", "female", "adult", "aged",
        "cell", "cancer", "tumor", "tumors", "protein", "proteins",
        "gene", "genes", "treatment", "therapy", "prognosis", "diagnosis",
    }

    # Generic MeSH terms to exclude from emerging trends
    # Combine with EMERGING_BLACKLIST from hot_topics.py
    GENERIC_MESH_BLACKLIST = {
        # Too generic - appear in most papers
        "humans", "animals", "male", "female", "adult", "aged", "child",
        "adolescent", "young adult", "middle aged", "infant", "pregnancy",
        # Generic methods/approaches
        "antineoplastic agents", "cell line, tumor", "molecular structure",
        "structure-activity relationship", "dose-response relationship, drug",
        "drug screening assays, antitumor", "cell proliferation",
        "signal transduction", "mice", "rats", "in vitro techniques",
        "mice, inbred balb c", "mice, inbred c57bl", "mice, nude",
        # Generic outcomes
        "apoptosis", "cell survival", "cell death", "neoplasms",
        "treatment outcome", "prognosis", "risk factors",
        "reactive oxygen species", "oxidative stress",
        # Generic lab terms
        "cells, cultured", "molecular docking simulation", "kinetics",
        "substrate specificity", "binding sites", "catalytic domain",
        "enzyme inhibitors", "protein binding", "drug design",
        "drug delivery systems", "nanoparticles", "drug discovery",
        "fluorescent dyes", "spectrometry, fluorescence", "limit of detection",
        "spectrum analysis, raman", "chromatography, high pressure liquid",
        # Generic research terms
        "retrospective studies", "prospective studies", "cohort studies",
        "cross-sectional studies", "case-control studies",
        "randomized controlled trials as topic", "clinical trials as topic",
        "surveys and questionnaires", "follow-up studies",
        # Countries/demographics
        "china", "united states", "japan", "europe", "korea", "india",
        "greece", "germany", "france", "brazil", "australia",
        # Generic biology
        "phosphorylation", "gene expression regulation", "transcription factors",
        "cell movement", "cell differentiation", "cell adhesion",
        "dna", "rna", "proteins", "lipids", "carbohydrates",
        # Additional terms from hot_topics.py EMERGING_BLACKLIST
        "disease models, animal", "animal experimentation",
        "mutation", "phenotype", "genotype", "biomarkers",
        "carcinogenesis", "metastasis", "tumor burden",
    } | EMERGING_BLACKLIST  # Merge with hot_topics blacklist

    MIN_KEYWORD_LENGTH = 3
    MIN_PAPER_COUNT = 5  # Minimum papers to be considered a trend

    def __init__(self, history_dir: Optional[Path] = None):
        self.history_dir = history_dir or Path(__file__).parent.parent / "output" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def analyze_hybrid(
        self,
        papers: List[Paper],
        top_predefined: int = 5,
        max_emerging: int = 3,
    ) -> TrendResult:
        """
        Perform hybrid trend analysis.

        Args:
            papers: List of Paper objects
            top_predefined: Number of top predefined trends to return
            max_emerging: Maximum emerging trends to detect

        Returns:
            TrendResult with predefined and emerging trends
        """
        # Load history
        yesterday = datetime.now() - timedelta(days=1)
        week_ago = datetime.now() - timedelta(days=7)
        yesterday_counts = self.load_history(yesterday)
        week_ago_counts = self.load_history(week_ago)

        # === Track 1: Predefined Hot Topics ===
        predefined_counts = self._count_predefined_topics(papers)
        predefined_trends = []

        for topic_name, count in predefined_counts.items():
            if count < self.MIN_PAPER_COUNT:
                continue

            topic_info = PREDEFINED_HOT_TOPICS.get(topic_name, {})

            trend = PredefinedTrend(
                name=topic_name,
                count=count,
                previous_count=yesterday_counts.get(f"predefined_{topic_name}", 0),
                week_ago_count=week_ago_counts.get(f"predefined_{topic_name}", 0),
                category=topic_info.get("category", ""),
                why_hot=topic_info.get("why_hot", ""),
            )

            # Find representative papers
            trend.representative_papers = self._find_papers_for_topic(topic_name, papers)
            predefined_trends.append(trend)

        # Sort by count
        predefined_trends.sort(key=lambda x: x.count, reverse=True)
        predefined_trends = predefined_trends[:top_predefined]

        # === Track 2: Emerging Trends ===
        all_keywords = self._extract_all_keywords(papers)
        emerging_trends = []

        for keyword, count in all_keywords.items():
            # Skip if in predefined topics
            if is_in_predefined(keyword):
                continue

            # Skip if too few papers
            if count < self.MIN_PAPER_COUNT:
                continue

            prev_count = yesterday_counts.get(f"keyword_{keyword}", 0)

            # Detect emerging: 100%+ increase or new appearance
            is_new = prev_count < 3
            is_rising = prev_count > 0 and (count / prev_count) >= 2.0

            if is_new and count >= self.MIN_PAPER_COUNT:
                trend = EmergingTrend(
                    name=keyword,
                    count=count,
                    previous_count=prev_count,
                    is_new=True,
                )
                trend.representative_papers = self._find_papers_for_keyword(keyword, papers)
                emerging_trends.append(trend)
            elif is_rising:
                trend = EmergingTrend(
                    name=keyword,
                    count=count,
                    previous_count=prev_count,
                    is_new=False,
                )
                trend.representative_papers = self._find_papers_for_keyword(keyword, papers)
                emerging_trends.append(trend)

        # Sort emerging by count
        emerging_trends.sort(key=lambda x: x.count, reverse=True)
        emerging_trends = emerging_trends[:max_emerging]

        # Save today's counts
        self._save_counts(predefined_counts, all_keywords)

        return TrendResult(
            predefined_trends=predefined_trends,
            emerging_trends=emerging_trends,
        )

    def get_hot_topics(
        self,
        papers: List[Paper],
        top_n: int = 5,
        min_count: int = 2,
    ) -> List[Trend]:
        """
        Backward-compatible method that returns unified Trend list.

        Args:
            papers: List of Paper objects
            top_n: Number of top trends to return
            min_count: Minimum count (not used, kept for compatibility)

        Returns:
            List of Trend objects
        """
        result = self.analyze_hybrid(papers, top_predefined=top_n, max_emerging=2)
        return result.to_unified_trends()

    # Short keywords that need word boundary matching to avoid false positives
    # e.g., "BiTE" should not match "exhibited", "inhibited"
    SHORT_KEYWORDS_NEED_BOUNDARY = {
        "bite", "aav", "lnp", "adc", "car", "nk", "rna", "dna",
    }

    def _is_word_match(self, keyword: str, text: str) -> bool:
        """
        Check if keyword appears as a whole word in text.
        Uses word boundary regex for short keywords to avoid false positives.

        Examples:
            - "BiTE" should NOT match "exhibited" or "inhibited"
            - "GLP-1" should match "GLP-1 agonist"
            - "bispecific" should match "bispecific antibody"
        """
        kw_lower = keyword.lower()
        text_lower = text.lower()

        # For short keywords (<=4 chars), use word boundary matching
        if len(kw_lower) <= 4 or kw_lower in self.SHORT_KEYWORDS_NEED_BOUNDARY:
            # Use regex with word boundaries
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            return bool(re.search(pattern, text_lower))
        else:
            # For longer keywords, simple substring matching is usually safe
            return kw_lower in text_lower

    def _count_predefined_topics(self, papers: List[Paper]) -> Dict[str, int]:
        """Count papers matching each predefined topic with content relevance validation."""
        counts = {name: 0 for name in PREDEFINED_HOT_TOPICS.keys()}

        for paper in papers:
            # Primary content: title + abstract (most reliable for relevance)
            primary_text = f"{paper.title} {paper.abstract}"
            # Secondary content: MeSH terms + keywords (metadata)
            secondary_text = f"{' '.join(paper.mesh_terms)} {' '.join(paper.keywords)}"

            for topic_name, topic_info in PREDEFINED_HOT_TOPICS.items():
                matched = False
                for keyword in topic_info["keywords"]:
                    # Use word boundary matching for accuracy
                    if self._is_word_match(keyword, primary_text):
                        matched = True
                        break
                    # Also accept MeSH/keyword metadata matches
                    elif self._is_word_match(keyword, secondary_text):
                        matched = True
                        break

                if matched:
                    counts[topic_name] += 1

        return counts

    def _validate_paper_relevance(self, paper: Paper, keywords: List[str], min_matches: int = 1) -> bool:
        """
        Validate that a paper is genuinely relevant to the topic keywords.
        Uses word boundary matching for short keywords.

        Args:
            paper: Paper object
            keywords: List of topic keywords
            min_matches: Minimum keyword matches required

        Returns:
            True if paper is relevant
        """
        text = f"{paper.title} {paper.abstract}"
        match_count = 0

        for keyword in keywords:
            if self._is_word_match(keyword, text):
                match_count += 1
                if match_count >= min_matches:
                    return True

        return False

    def _extract_all_keywords(self, papers: List[Paper]) -> Dict[str, int]:
        """Extract all keywords from papers with frequency counts."""
        keyword_counts = Counter()

        for paper in papers:
            # MeSH terms (high quality)
            for mesh in paper.mesh_terms:
                normalized = mesh.lower().strip()
                if self._is_valid_keyword(normalized):
                    keyword_counts[normalized] += 1

            # Author keywords
            for kw in paper.keywords:
                normalized = kw.lower().strip()
                if self._is_valid_keyword(normalized):
                    keyword_counts[normalized] += 1

        return dict(keyword_counts)

    def _is_valid_keyword(self, keyword: str) -> bool:
        """Check if keyword is valid for trend analysis."""
        keyword_lower = keyword.lower()

        if len(keyword) < self.MIN_KEYWORD_LENGTH:
            return False
        if keyword_lower in self.STOPWORDS:
            return False
        # Filter generic MeSH terms (blacklist)
        if keyword_lower in self.GENERIC_MESH_BLACKLIST:
            return False
        # Filter generic terms
        generic = {"cell line", "cell lines", "signal transduction", "molecular structure"}
        if keyword_lower in generic:
            return False
        return True

    def _find_papers_for_topic(self, topic_name: str, papers: List[Paper], max_papers: int = 3) -> List[Paper]:
        """Find representative papers for a predefined topic with strict relevance validation."""
        topic_info = PREDEFINED_HOT_TOPICS.get(topic_name, {})
        keywords = topic_info.get("keywords", [])

        scored = []
        for paper in papers:
            # Must pass relevance validation first (uses word boundary matching)
            if not self._validate_paper_relevance(paper, keywords):
                continue

            text = f"{paper.title} {paper.abstract}"
            score = 0
            matched_keywords = []

            for kw in keywords:
                # Use word boundary matching for scoring too
                if self._is_word_match(kw, paper.title):
                    score += 10
                    matched_keywords.append(kw)
                elif self._is_word_match(kw, text):
                    score += 5
                    matched_keywords.append(kw)

            if score > 0:
                scored.append((score, len(matched_keywords), paper))

        # Sort by score, then by number of matched keywords
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, _, p in scored[:max_papers]]

    def _find_papers_for_keyword(self, keyword: str, papers: List[Paper], max_papers: int = 3) -> List[Paper]:
        """Find representative papers for a keyword."""
        keyword_lower = keyword.lower()
        scored = []

        for paper in papers:
            score = 0

            if keyword_lower in paper.title.lower():
                score += 10
            if keyword_lower in paper.abstract.lower():
                score += 5
            if any(keyword_lower in m.lower() for m in paper.mesh_terms):
                score += 3
            if any(keyword_lower in k.lower() for k in paper.keywords):
                score += 2

            if score > 0:
                scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:max_papers]]

    # Mapping from new topic names to old topic names for history migration
    TOPIC_NAME_MIGRATION = {
        "GLP-1/비만치료": ["GLP-1", "비만"],  # Merged topic
        "COVID-19": ["COVID-19"],  # New topic
        "Long COVID": ["Long COVID"],
    }

    def load_history(self, date: Optional[datetime] = None) -> Dict[str, int]:
        """Load counts from history file with migration support."""
        if date is None:
            date = datetime.now() - timedelta(days=1)

        filename = f"trends_{date.strftime('%Y%m%d')}.json"
        filepath = self.history_dir / filename

        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both old and new format
                if "predefined" in data:
                    # New format with predefined section
                    result = {}
                    predefined_data = data.get("predefined", {})

                    # Handle topic name migrations
                    for new_name, old_names in self.TOPIC_NAME_MIGRATION.items():
                        # Sum up old topic counts for merged topics
                        total = 0
                        for old_name in old_names:
                            if old_name in predefined_data:
                                total += predefined_data[old_name]
                        if total > 0:
                            result[f"predefined_{new_name}"] = total

                    # Copy other predefined topics directly
                    for k, v in predefined_data.items():
                        # Skip old names that were migrated
                        is_migrated = any(k in old_names for old_names in self.TOPIC_NAME_MIGRATION.values())
                        if not is_migrated:
                            result[f"predefined_{k}"] = v

                    # Copy keywords
                    for k, v in data.get("keywords", {}).items():
                        result[f"keyword_{k}"] = v
                    return result
                else:
                    # Old format - treat as keywords
                    return {f"keyword_{k}": v for k, v in data.items()}

        return {}

    def _save_counts(self, predefined: Dict[str, int], keywords: Dict[str, int]):
        """Save counts to history file."""
        today = datetime.now()
        filename = f"trends_{today.strftime('%Y%m%d')}.json"
        filepath = self.history_dir / filename

        data = {
            "date": today.strftime("%Y-%m-%d"),
            "predefined": predefined,
            "keywords": keywords,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_history(self, keyword_counts: Dict[str, int], date: Optional[datetime] = None):
        """Legacy method for backward compatibility."""
        if date is None:
            date = datetime.now()

        filename = f"trends_{date.strftime('%Y%m%d')}.json"
        filepath = self.history_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(keyword_counts, f, ensure_ascii=False, indent=2)

    def generate_trend_summary(self, trends: List[Trend]) -> str:
        """Generate a text summary of trends."""
        lines = ["## 오늘의 핫 키워드\n"]

        predefined = [t for t in trends if t.is_predefined]
        emerging = [t for t in trends if t.is_emerging]

        if predefined:
            lines.append("### [업계 주목 키워드]")
            for i, t in enumerate(predefined, 1):
                lines.append(f"{i}위  {t.keyword:<20} {t.count}건  {t.change_label}")

        if emerging:
            lines.append("\n### [🆕 급상승 감지]")
            for t in emerging:
                lines.append(f"• {t.keyword:<20} {t.count}건  {t.change_label}")

        return "\n".join(lines)


async def main():
    """Test the hybrid trend analyzer."""
    from .pubmed_fetcher import PubMedFetcher

    print("=" * 60)
    print("BIO Daily Briefing - Hybrid Trend Analysis Test")
    print("=" * 60)

    # Fetch papers
    fetcher = PubMedFetcher()
    print("\nFetching papers from PubMed (last 48 hours)...")
    papers = await fetcher.fetch_recent_papers(max_results=100, days=2)
    print(f"Fetched {len(papers)} papers")

    # Analyze trends
    analyzer = TrendAnalyzer()
    result = analyzer.analyze_hybrid(papers, top_predefined=5, max_emerging=3)

    # Print predefined trends
    print("\n" + "=" * 60)
    print("[업계 주목 키워드]")
    print("=" * 60)

    for i, t in enumerate(result.predefined_trends, 1):
        print(f"\n{i}위  {t.name}")
        print(f"    논문 수: {t.count}건")
        print(f"    변화: {t.change_label}")
        print(f"    카테고리: {t.category}")
        print(f"    왜 핫한가: {t.why_hot}")
        if t.representative_papers:
            print(f"    대표 논문: {t.representative_papers[0].title[:50]}...")

    # Print emerging trends
    if result.emerging_trends:
        print("\n" + "=" * 60)
        print("[🆕 급상승 감지]")
        print("=" * 60)

        for t in result.emerging_trends:
            print(f"\n• {t.name}")
            print(f"    논문 수: {t.count}건")
            print(f"    상태: {t.change_label}")
            if t.representative_papers:
                print(f"    대표 논문: {t.representative_papers[0].title[:50]}...")

    # Test backward compatibility
    print("\n" + "=" * 60)
    print("[Unified Trends (backward compatible)]")
    print("=" * 60)

    unified = result.to_unified_trends()
    for t in unified:
        print(f"  {t.trend_indicator} {t.keyword}: {t.count}건")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
