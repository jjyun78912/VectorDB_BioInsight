"""
bioRxiv/medRxiv Preprint Fetcher

Fetches recent preprints from bioRxiv and medRxiv APIs.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, field
import httpx


@dataclass
class Preprint:
    """Represents a bioRxiv/medRxiv preprint."""
    doi: str
    title: str
    abstract: str
    authors: str
    category: str
    server: str  # biorxiv or medrxiv
    date: str
    version: int = 1
    priority: int = 0

    # Optional metadata
    institution: Optional[str] = None
    published_doi: Optional[str] = None  # If published in journal

    def to_dict(self) -> dict:
        return {
            "source": self.server,
            "type": "preprint",
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract[:500],
            "authors": self.authors,
            "category": self.category,
            "date": self.date,
            "version": self.version,
            "priority": self.priority,
            "link": f"https://www.{self.server}.org/content/{self.doi}",
        }


class BioRxivFetcher:
    """Fetches preprints from bioRxiv and medRxiv APIs."""

    BASE_URL = "https://api.biorxiv.org/details"

    # Relevant categories for bio/healthcare
    RELEVANT_CATEGORIES = {
        # bioRxiv categories
        "cancer biology",
        "cell biology",
        "genetics",
        "genomics",
        "immunology",
        "microbiology",
        "molecular biology",
        "neuroscience",
        "pharmacology and toxicology",
        "biochemistry",
        "bioinformatics",
        "developmental biology",
        "synthetic biology",
        # medRxiv categories
        "oncology",
        "infectious diseases",
        "genetic and genomic medicine",
        "hematology",
        "endocrinology",
        "neurology",
        "psychiatry and clinical psychology",
        "cardiovascular medicine",
        "gastroenterology",
        "respiratory medicine",
        "allergy and immunology",
    }

    # Hot keywords for priority scoring
    HOT_KEYWORDS = [
        "glp-1", "semaglutide", "tirzepatide", "obesity",
        "crispr", "gene editing", "base editing", "prime editing",
        "car-t", "car t", "cell therapy", "gene therapy",
        "mrna", "lipid nanoparticle", "lnp",
        "alzheimer", "amyloid", "tau",
        "pd-1", "pd-l1", "checkpoint", "immunotherapy",
        "adc", "antibody-drug conjugate", "bispecific",
        "single-cell", "spatial transcriptomics",
        "alphafold", "protein structure", "ai drug",
        "microbiome", "gut-brain",
    ]

    # Major institutions
    MAJOR_INSTITUTIONS = [
        "harvard", "mit", "stanford", "yale", "columbia", "berkeley",
        "oxford", "cambridge", "imperial", "ucl",
        "nih", "cdc", "fda",
        "broad institute", "sanger", "embl",
    ]

    def __init__(self):
        self.timeout = httpx.Timeout(60.0)

    async def fetch_recent(
        self,
        days: int = 3,
        server: str = "biorxiv",
        filter_categories: bool = True
    ) -> List[Preprint]:
        """
        Fetch recent preprints from bioRxiv or medRxiv.

        Args:
            days: Look back period in days
            server: "biorxiv" or "medrxiv"
            filter_categories: Only include relevant categories

        Returns:
            List of Preprint objects
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/{server}/{start_date}/{end_date}"

        print(f"\n[{server} 프리프린트 - 최근 {days}일]")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # API returns paginated results, need to fetch all
                all_preprints = []
                cursor = 0

                while True:
                    response = await client.get(f"{url}/{cursor}")

                    if response.status_code != 200:
                        print(f"  ⚠️ API 오류: {response.status_code}")
                        break

                    data = response.json()
                    collection = data.get("collection", [])

                    if not collection:
                        break

                    for item in collection:
                        preprint = self._parse_preprint(item, server)

                        # Filter by category
                        if filter_categories:
                            if preprint.category.lower() not in self.RELEVANT_CATEGORIES:
                                continue

                        all_preprints.append(preprint)

                    # Check for more pages
                    messages = data.get("messages", [])
                    if not messages or "No posts found" in str(messages):
                        break

                    cursor += len(collection)

                    # Limit total fetched
                    if len(all_preprints) >= 200:
                        break

                    await asyncio.sleep(0.5)  # Rate limiting

                print(f"  {server}: {len(all_preprints)}건")
                return all_preprints

        except Exception as e:
            print(f"  ⚠️ 오류: {e}")
            return []

    async def fetch_all_servers(self, days: int = 3) -> List[Preprint]:
        """
        Fetch from both bioRxiv and medRxiv.

        Args:
            days: Look back period

        Returns:
            Combined list of preprints, sorted by priority
        """
        biorxiv = await self.fetch_recent(days=days, server="biorxiv")
        await asyncio.sleep(1)  # Rate limiting between servers
        medrxiv = await self.fetch_recent(days=days, server="medrxiv")

        all_preprints = biorxiv + medrxiv

        # Sort by priority
        all_preprints.sort(key=lambda x: x.priority, reverse=True)

        print(f"  총 프리프린트: {len(all_preprints)}건")
        return all_preprints

    def _parse_preprint(self, item: dict, server: str) -> Preprint:
        """Parse preprint from API response."""
        preprint = Preprint(
            doi=item.get("doi", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", ""),
            authors=item.get("authors", ""),
            category=item.get("category", ""),
            server=server,
            date=item.get("date", ""),
            version=int(item.get("version", 1)),
            published_doi=item.get("published", None) if item.get("published") != "NA" else None,
        )

        # Extract institution from author affiliations
        author_text = item.get("author_corresponding_institution", "")
        if author_text:
            preprint.institution = author_text

        # Calculate priority
        preprint.priority = self._calculate_priority(preprint)

        return preprint

    def _calculate_priority(self, preprint: Preprint) -> int:
        """Calculate preprint priority score."""
        score = 0
        text = f"{preprint.title} {preprint.abstract}".lower()

        # Hot keywords bonus
        for keyword in self.HOT_KEYWORDS:
            if keyword in text:
                score += 30
                break

        # Category bonus for highly relevant fields
        high_priority_categories = {
            "cancer biology", "oncology", "immunology",
            "genetics", "genomics", "neuroscience",
        }
        if preprint.category.lower() in high_priority_categories:
            score += 20

        # Major institution bonus
        if preprint.institution:
            inst_lower = preprint.institution.lower()
            for inst in self.MAJOR_INSTITUTIONS:
                if inst in inst_lower:
                    score += 15
                    break

        # medRxiv bonus for clinical relevance
        if preprint.server == "medrxiv":
            score += 10

        # Already published penalty (not new)
        if preprint.published_doi:
            score -= 20

        # Version 1 bonus (truly new)
        if preprint.version == 1:
            score += 10

        return score

    def get_top_preprints(
        self,
        preprints: List[Preprint],
        n: int = 10,
        require_keywords: bool = False
    ) -> List[Preprint]:
        """
        Get top N preprints by priority.

        Args:
            preprints: List of preprints
            n: Number to return
            require_keywords: Only include preprints matching hot keywords

        Returns:
            Top N preprints
        """
        if require_keywords:
            filtered = []
            for p in preprints:
                text = f"{p.title} {p.abstract}".lower()
                for keyword in self.HOT_KEYWORDS:
                    if keyword in text:
                        filtered.append(p)
                        break
            preprints = filtered

        return sorted(preprints, key=lambda x: x.priority, reverse=True)[:n]


async def main():
    """Test bioRxiv fetcher."""
    print("=" * 60)
    print("bioRxiv/medRxiv Fetcher Test")
    print("=" * 60)

    fetcher = BioRxivFetcher()

    # Fetch from both servers
    preprints = await fetcher.fetch_all_servers(days=3)

    # Show top preprints
    top = fetcher.get_top_preprints(preprints, n=10, require_keywords=True)

    print(f"\n🔬 주목할 프리프린트 ({len(top)}건):")
    for p in top:
        print(f"\n  [{p.server}] {p.title[:60]}...")
        print(f"    Category: {p.category}")
        print(f"    Authors: {p.authors[:50]}...")
        print(f"    Priority: {p.priority}")
        print(f"    DOI: {p.doi}")

    # Category distribution
    print(f"\n카테고리 분포:")
    from collections import Counter
    categories = Counter(p.category for p in preprints)
    for cat, count in categories.most_common(10):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
