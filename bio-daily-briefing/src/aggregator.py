"""
News Aggregator - Multi-source Integration

Combines news from multiple sources:
1. FDA (Regulatory)
2. ClinicalTrials.gov (Clinical)
3. bioRxiv/medRxiv (Preprints)
4. PubMed (Research)

Provides prioritization and headline selection.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .sources.fda_fetcher import FDAFetcher, FDANews
from .sources.clinicaltrials_fetcher import ClinicalTrialsFetcher, ClinicalTrial
from .sources.biorxiv_fetcher import BioRxivFetcher, Preprint
from .pubmed_fetcher import PubMedFetcher, Paper
from .trend_analyzer import TrendAnalyzer, Trend


@dataclass
class AggregatedNews:
    """Unified news item from any source."""
    source: str  # FDA, ClinicalTrials, bioRxiv, medrxiv, PubMed
    news_type: str  # approval, phase3_result, preprint, research
    title: str
    summary: str
    link: str
    date: str
    priority: int = 0

    # Source-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "type": self.news_type,
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "date": self.date,
            "priority": self.priority,
            "metadata": self.metadata,
        }


@dataclass
class DailyBriefingData:
    """Complete daily briefing data structure."""
    date: str
    issue_number: int

    # Headline (most important news)
    headline: Optional[AggregatedNews] = None

    # Categorized news
    regulatory_news: List[AggregatedNews] = field(default_factory=list)
    clinical_trials: Dict[str, List[AggregatedNews]] = field(default_factory=dict)
    research: Dict[str, List[AggregatedNews]] = field(default_factory=dict)

    # Trends
    hot_topics: List[Trend] = field(default_factory=list)

    # Stats
    total_fda: int = 0
    total_trials: int = 0
    total_preprints: int = 0
    total_papers: int = 0

    # Editor comment (AI generated)
    editor_comment: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "issue_number": self.issue_number,
            "headline": self.headline.to_dict() if self.headline else None,
            "regulatory": [n.to_dict() for n in self.regulatory_news],
            "clinical_trials": {
                k: [n.to_dict() for n in v]
                for k, v in self.clinical_trials.items()
            },
            "research": {
                k: [n.to_dict() for n in v]
                for k, v in self.research.items()
            },
            "hot_topics": [t.to_dict() for t in self.hot_topics],
            "stats": {
                "fda": self.total_fda,
                "trials": self.total_trials,
                "preprints": self.total_preprints,
                "papers": self.total_papers,
            },
            "editor_comment": self.editor_comment,
        }


class NewsAggregator:
    """Aggregates news from multiple sources."""

    def __init__(self):
        self.fda_fetcher = FDAFetcher()
        self.trials_fetcher = ClinicalTrialsFetcher()
        self.biorxiv_fetcher = BioRxivFetcher()
        self.pubmed_fetcher = PubMedFetcher()
        self.trend_analyzer = TrendAnalyzer()

    async def aggregate_daily(
        self,
        fda_hours: int = 72,
        trials_days: int = 30,
        preprint_days: int = 3,
        pubmed_days: int = 2,
        issue_number: int = 1,
    ) -> DailyBriefingData:
        """
        Aggregate news from all sources for daily briefing.

        Args:
            fda_hours: Look back for FDA news
            trials_days: Look back for clinical trials
            preprint_days: Look back for preprints
            pubmed_days: Look back for PubMed papers
            issue_number: Issue number for the briefing

        Returns:
            DailyBriefingData with all aggregated content
        """
        print("=" * 60)
        print(f"BIO Daily Briefing v2 - 멀티소스 수집")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)

        # Initialize briefing data
        briefing = DailyBriefingData(
            date=datetime.now().strftime("%Y%m%d"),
            issue_number=issue_number,
        )

        # === 1. FDA News (Highest Priority) ===
        print("\n[1/4] FDA 뉴스 수집...")
        fda_news = await self.fda_fetcher.fetch_recent(hours=fda_hours)
        briefing.total_fda = len(fda_news)

        # Convert to unified format
        regulatory_news = self._convert_fda_news(fda_news)
        briefing.regulatory_news = regulatory_news[:10]  # Top 10

        # === 2. Clinical Trials ===
        print("\n[2/4] ClinicalTrials.gov 수집...")
        trials_data = await self.trials_fetcher.fetch_all(
            results_days=trials_days,
            new_trials_days=14
        )

        # Phase 3 results
        phase3_results = self._convert_trials(trials_data.get("phase3_results", []))
        briefing.clinical_trials["phase3_results"] = phase3_results[:5]

        # New trials
        new_trials = self._convert_trials(trials_data.get("new_trials", []))
        briefing.clinical_trials["new_trials"] = new_trials[:5]

        # Terminated
        terminated = self._convert_trials(trials_data.get("terminated", []))
        briefing.clinical_trials["terminated"] = terminated[:3]

        briefing.total_trials = (
            len(trials_data.get("phase3_results", [])) +
            len(trials_data.get("new_trials", [])) +
            len(trials_data.get("terminated", []))
        )

        # === 3. Preprints ===
        print("\n[3/4] bioRxiv/medRxiv 수집...")
        preprints = await self.biorxiv_fetcher.fetch_all_servers(days=preprint_days)
        top_preprints = self.biorxiv_fetcher.get_top_preprints(
            preprints, n=10, require_keywords=True
        )

        preprint_news = self._convert_preprints(top_preprints)
        briefing.research["preprints"] = preprint_news
        briefing.total_preprints = len(preprints)

        # === 4. PubMed (Peer-reviewed) ===
        print("\n[4/4] PubMed 논문 수집...")
        papers = await self.pubmed_fetcher.fetch_comprehensive(
            days=pubmed_days,
            max_total=300
        )
        briefing.total_papers = len(papers)

        # Analyze trends from PubMed papers
        hot_topics = self.trend_analyzer.get_hot_topics(papers, top_n=5, min_count=3)
        briefing.hot_topics = hot_topics

        # High-impact journal papers
        high_impact_papers = self._filter_high_impact_papers(papers)
        paper_news = self._convert_papers(high_impact_papers[:10])
        briefing.research["high_impact"] = paper_news

        # === Select Headline ===
        briefing.headline = self._select_headline(
            regulatory_news,
            phase3_results,
            preprint_news
        )

        # === Summary ===
        print("\n" + "=" * 60)
        print("수집 완료:")
        print(f"  FDA 뉴스: {briefing.total_fda}건")
        print(f"  임상시험: {briefing.total_trials}건")
        print(f"  프리프린트: {briefing.total_preprints}건")
        print(f"  PubMed 논문: {briefing.total_papers}건")
        print("=" * 60)

        return briefing

    def _convert_fda_news(self, fda_news: List[FDANews]) -> List[AggregatedNews]:
        """Convert FDA news to unified format."""
        result = []
        for news in fda_news:
            result.append(AggregatedNews(
                source="FDA",
                news_type=news.source_type,
                title=news.title,
                summary=news.summary,
                link=news.link,
                date=news.date.strftime("%Y-%m-%d"),
                priority=news.priority,
                metadata={
                    "drug_name": news.drug_name,
                    "company": news.company,
                    "indication": news.indication,
                }
            ))
        return result

    def _convert_trials(self, trials: List[ClinicalTrial]) -> List[AggregatedNews]:
        """Convert clinical trials to unified format."""
        result = []
        for trial in trials:
            result.append(AggregatedNews(
                source="ClinicalTrials",
                news_type=f"phase3_{trial.result_type or 'study'}",
                title=trial.title,
                summary=f"Sponsor: {trial.sponsor}. Conditions: {', '.join(trial.conditions[:2])}",
                link=f"https://clinicaltrials.gov/study/{trial.nct_id}",
                date=trial.results_date or trial.completion_date or "",
                priority=trial.priority,
                metadata={
                    "nct_id": trial.nct_id,
                    "phase": trial.phase,
                    "sponsor": trial.sponsor,
                    "conditions": trial.conditions,
                    "interventions": trial.interventions,
                    "enrollment": trial.enrollment,
                    "has_results": trial.has_results,
                }
            ))
        return result

    def _convert_preprints(self, preprints: List[Preprint]) -> List[AggregatedNews]:
        """Convert preprints to unified format."""
        result = []
        for p in preprints:
            result.append(AggregatedNews(
                source=p.server,
                news_type="preprint",
                title=p.title,
                summary=p.abstract[:300] + "..." if len(p.abstract) > 300 else p.abstract,
                link=f"https://www.{p.server}.org/content/{p.doi}",
                date=p.date,
                priority=p.priority,
                metadata={
                    "doi": p.doi,
                    "authors": p.authors,
                    "category": p.category,
                    "institution": p.institution,
                }
            ))
        return result

    def _convert_papers(self, papers: List[Paper]) -> List[AggregatedNews]:
        """Convert PubMed papers to unified format."""
        result = []
        for paper in papers:
            result.append(AggregatedNews(
                source="PubMed",
                news_type="research",
                title=paper.title,
                summary=paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract,
                link=f"https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}",
                date=paper.pub_date,
                priority=50,  # Base priority
                metadata={
                    "pmid": paper.pmid,
                    "journal": paper.journal,
                    "authors": paper.authors[:3],
                    "doi": paper.doi,
                }
            ))
        return result

    def _filter_high_impact_papers(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers from high-impact journals."""
        high_impact_journals = {
            "nature", "science", "cell", "lancet",
            "new england journal of medicine", "nejm",
            "jama", "bmj", "nature medicine", "nature genetics",
            "nature biotechnology", "cell stem cell",
            "journal of clinical oncology", "blood",
            "circulation", "gut", "annals of oncology",
        }

        result = []
        for paper in papers:
            journal_lower = paper.journal.lower()
            for hj in high_impact_journals:
                if hj in journal_lower:
                    result.append(paper)
                    break

        return result

    def _select_headline(
        self,
        regulatory: List[AggregatedNews],
        trials: List[AggregatedNews],
        preprints: List[AggregatedNews]
    ) -> Optional[AggregatedNews]:
        """Select the most important news as headline."""

        # Priority 1: FDA approvals
        approvals = [n for n in regulatory if "approval" in n.news_type]
        if approvals:
            return max(approvals, key=lambda x: x.priority)

        # Priority 2: FDA safety warnings
        warnings = [n for n in regulatory if "warning" in n.news_type or "safety" in n.news_type]
        if warnings:
            return max(warnings, key=lambda x: x.priority)

        # Priority 3: Major clinical trial results
        if trials:
            return max(trials, key=lambda x: x.priority)

        # Priority 4: Top preprint
        if preprints:
            return max(preprints, key=lambda x: x.priority)

        # Priority 5: Any FDA news
        if regulatory:
            return max(regulatory, key=lambda x: x.priority)

        return None


async def main():
    """Test the aggregator."""
    print("=" * 60)
    print("News Aggregator Test")
    print("=" * 60)

    aggregator = NewsAggregator()
    briefing = await aggregator.aggregate_daily(
        fda_hours=168,  # 1 week for testing
        trials_days=60,
        preprint_days=7,
        pubmed_days=3,
        issue_number=1,
    )

    # Print headline
    if briefing.headline:
        print(f"\n🚨 오늘의 헤드라인:")
        print(f"   [{briefing.headline.source}] {briefing.headline.title}")

    # Print regulatory
    print(f"\n⚡ 규제/승인 소식 ({len(briefing.regulatory_news)}건):")
    for news in briefing.regulatory_news[:3]:
        print(f"  • {news.title[:60]}...")

    # Print trials
    results = briefing.clinical_trials.get("phase3_results", [])
    print(f"\n📊 Phase 3 결과 ({len(results)}건):")
    for news in results[:3]:
        print(f"  • {news.title[:60]}...")

    # Print hot topics
    print(f"\n📈 핫토픽:")
    for trend in briefing.hot_topics:
        print(f"  {trend.trend_indicator} {trend.keyword}: {trend.count}건")


if __name__ == "__main__":
    asyncio.run(main())
