"""
FDA News Fetcher

Fetches drug approvals, safety warnings, and regulatory news from FDA RSS feeds.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, field
import httpx
import feedparser
from bs4 import BeautifulSoup
import re


@dataclass
class FDANews:
    """Represents an FDA news item."""
    title: str
    summary: str
    link: str
    date: datetime
    source_type: str  # drug_approval, safety, press_release
    priority: int = 0

    # Extracted metadata
    drug_name: Optional[str] = None
    company: Optional[str] = None
    indication: Optional[str] = None
    approval_type: Optional[str] = None  # new_drug, biologics, supplement

    def to_dict(self) -> dict:
        return {
            "source": "FDA",
            "type": self.source_type,
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "date": self.date.isoformat(),
            "priority": self.priority,
            "drug_name": self.drug_name,
            "company": self.company,
            "indication": self.indication,
            "approval_type": self.approval_type,
        }


class FDAFetcher:
    """Fetches FDA regulatory news from RSS feeds."""

    # FDA RSS feeds (verified working URLs)
    RSS_FEEDS = {
        "drugs": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
        "press_releases": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
        "medwatch": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml",
        "recalls": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml",
    }

    # Hot keywords for priority scoring
    HOT_KEYWORDS = [
        "glp-1", "semaglutide", "tirzepatide", "ozempic", "wegovy", "mounjaro",
        "crispr", "gene therapy", "car-t", "car t", "cell therapy",
        "mrna", "vaccine", "antibody-drug conjugate", "adc",
        "alzheimer", "lecanemab", "donanemab",
        "checkpoint inhibitor", "pd-1", "pd-l1",
        "bispecific", "obesity", "diabetes", "cancer",
    ]

    # Big pharma companies
    BIG_PHARMA = [
        "pfizer", "novartis", "roche", "lilly", "eli lilly", "novo nordisk",
        "merck", "bristol-myers", "bms", "astrazeneca", "johnson & johnson",
        "j&j", "sanofi", "gsk", "glaxosmithkline", "abbvie", "gilead",
        "amgen", "moderna", "biontech", "regeneron",
    ]

    def __init__(self):
        self.timeout = httpx.Timeout(30.0)

    async def fetch_recent(self, hours: int = 72) -> List[FDANews]:
        """
        Fetch recent FDA news from all RSS feeds.

        Args:
            hours: Look back period in hours

        Returns:
            List of FDANews items sorted by priority
        """
        all_items = []
        cutoff = datetime.now() - timedelta(hours=hours)

        print(f"\n[FDA 뉴스 수집 - 최근 {hours}시간]")

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                items = await self._fetch_feed(feed_name, feed_url, cutoff)
                all_items.extend(items)
                if items:
                    print(f"  ✅ {feed_name}: {len(items)}건")
            except Exception as e:
                print(f"  ❌ {feed_name}: 오류 - {e}")

        # Deduplicate by title
        seen_titles = set()
        unique_items = []
        for item in all_items:
            title_key = item.title.lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)

        # Sort by priority
        unique_items.sort(key=lambda x: x.priority, reverse=True)

        print(f"  총 FDA 뉴스: {len(unique_items)}건")
        return unique_items

    async def _fetch_feed(
        self,
        feed_name: str,
        feed_url: str,
        cutoff: datetime
    ) -> List[FDANews]:
        """Fetch and parse a single RSS feed."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(feed_url)
            response.raise_for_status()

        feed = feedparser.parse(response.text)
        items = []

        for entry in feed.entries:
            # Parse date
            pub_date = self._parse_date(entry)
            if pub_date and pub_date < cutoff:
                continue

            if pub_date is None:
                pub_date = datetime.now()

            # Determine source type
            source_type = self._determine_source_type(feed_name, entry)

            # Create news item
            news = FDANews(
                title=entry.get("title", ""),
                summary=self._clean_summary(entry.get("summary", "")),
                link=entry.get("link", ""),
                date=pub_date,
                source_type=source_type,
            )

            # Extract metadata
            self._extract_metadata(news)

            # Calculate priority
            news.priority = self._calculate_priority(news)

            items.append(news)

        return items

    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse publication date from feed entry."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        return None

    def _determine_source_type(self, feed_name: str, entry) -> str:
        """Determine the type of FDA news."""
        title = entry.get("title", "").lower()

        if "approv" in title:
            if "biologic" in title or "bla" in title:
                return "biologics_approval"
            return "drug_approval"
        if "warning" in title or "safety" in title or "recall" in title:
            return "safety_warning"
        if "complete response" in title or "reject" in title:
            return "rejection"
        if "breakthrough" in title or "fast track" in title or "accelerated" in title:
            return "designation"
        if "advisory" in title or "committee" in title:
            return "advisory"

        return feed_name

    def _clean_summary(self, summary: str) -> str:
        """Clean HTML from summary."""
        soup = BeautifulSoup(summary, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # Limit length
        if len(text) > 500:
            text = text[:497] + "..."
        return text

    def _extract_metadata(self, news: FDANews):
        """Extract drug name, company, indication from title/summary."""
        text = f"{news.title} {news.summary}".lower()

        # Extract company name
        for company in self.BIG_PHARMA:
            if company in text:
                news.company = company.title()
                break

        # Extract common drug names (simplified)
        drug_patterns = [
            r"(semaglutide|tirzepatide|liraglutide|ozempic|wegovy|mounjaro)",
            r"(lecanemab|donanemab|aducanumab)",
            r"(pembrolizumab|nivolumab|keytruda|opdivo)",
        ]

        for pattern in drug_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                news.drug_name = match.group(1).title()
                break

    def _calculate_priority(self, news: FDANews) -> int:
        """Calculate news priority score."""
        score = 0
        text = f"{news.title} {news.summary}".lower()

        # Source type scoring
        type_scores = {
            "drug_approval": 100,
            "biologics_approval": 100,
            "rejection": 90,
            "safety_warning": 85,
            "designation": 70,
            "advisory": 60,
        }
        score += type_scores.get(news.source_type, 20)

        # "First" or "breakthrough" bonus
        if "first" in text or "breakthrough" in text or "novel" in text:
            score += 50

        # Hot keyword bonus
        for keyword in self.HOT_KEYWORDS:
            if keyword in text:
                score += 20
                break

        # Big pharma bonus
        for company in self.BIG_PHARMA:
            if company in text:
                score += 15
                break

        return score

    def get_headline_news(self, news_items: List[FDANews]) -> Optional[FDANews]:
        """Get the most important news item as headline."""
        if not news_items:
            return None

        # Prioritize approvals
        approvals = [n for n in news_items if "approval" in n.source_type]
        if approvals:
            return max(approvals, key=lambda x: x.priority)

        # Then safety warnings
        warnings = [n for n in news_items if "warning" in n.source_type or "safety" in n.source_type]
        if warnings:
            return max(warnings, key=lambda x: x.priority)

        # Otherwise, highest priority
        return news_items[0]

    def filter_by_type(self, news_items: List[FDANews], types: List[str]) -> List[FDANews]:
        """Filter news by source type."""
        return [n for n in news_items if n.source_type in types]


async def main():
    """Test FDA fetcher."""
    print("=" * 60)
    print("FDA News Fetcher Test")
    print("=" * 60)

    fetcher = FDAFetcher()
    news_items = await fetcher.fetch_recent(hours=168)  # 1 week for testing

    print(f"\nTotal items: {len(news_items)}")

    # Show approvals
    approvals = fetcher.filter_by_type(news_items, ["drug_approval", "biologics_approval"])
    if approvals:
        print(f"\n승인 소식 ({len(approvals)}건):")
        for item in approvals[:5]:
            print(f"  • {item.title[:70]}...")
            print(f"    Priority: {item.priority}, Date: {item.date.strftime('%Y-%m-%d')}")

    # Show safety warnings
    warnings = fetcher.filter_by_type(news_items, ["safety_warning"])
    if warnings:
        print(f"\n안전성 경고 ({len(warnings)}건):")
        for item in warnings[:3]:
            print(f"  • {item.title[:70]}...")

    # Headline
    headline = fetcher.get_headline_news(news_items)
    if headline:
        print(f"\n🚨 오늘의 헤드라인:")
        print(f"   {headline.title}")


if __name__ == "__main__":
    asyncio.run(main())
