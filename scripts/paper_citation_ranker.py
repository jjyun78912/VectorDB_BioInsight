"""
논문 인용 수 기반 랭킹 시스템.

Semantic Scholar API를 사용하여 VectorDB에 인덱싱된 논문들의
인용 수를 수집하고 품질 점수를 계산합니다.

Usage:
    python scripts/paper_citation_ranker.py --collection rnaseq_pancreatic_cancer
    python scripts/paper_citation_ranker.py --all-rnaseq
    python scripts/paper_citation_ranker.py --top 20 --collection bio_papers_breast_cancer
"""

import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import CHROMA_DIR, DATA_DIR
import chromadb


@dataclass
class PaperScore:
    """논문 품질 점수 데이터."""
    pmid: str
    title: str
    citation_count: int
    year: int
    has_fulltext: bool
    journal: str
    quality_score: float  # 종합 품질 점수 (0-100)

    # 세부 점수
    citation_score: float  # 인용 기반 점수 (0-40)
    recency_score: float   # 최신성 점수 (0-30)
    content_score: float   # 콘텐츠 품질 점수 (0-30)


class PaperCitationRanker:
    """논문 인용 수 기반 랭킹 시스템."""

    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"

    def __init__(self, collection_name: str):
        """
        Initialize paper ranker.

        Args:
            collection_name: ChromaDB collection name
        """
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception as e:
            raise ValueError(f"Collection not found: {collection_name}. Error: {e}")

        # Cache directory
        self.cache_dir = DATA_DIR / "citation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{collection_name}_citations.json"

        # Load cache
        self.citation_cache = self._load_cache()

        # Rate limiting for Semantic Scholar API (more conservative)
        # Free tier: ~100 requests per 5 minutes, but often stricter
        self.request_delay = 5.0  # 5 seconds between requests to be safe
        self.last_request_time = 0

        print(f"Initialized ranker for: {collection_name}")
        print(f"Collection has {self.collection.count()} chunks")

    def _load_cache(self) -> Dict:
        """Load citation cache from file."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save citation cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.citation_cache, f, indent=2, ensure_ascii=False)

    def _rate_limit(self):
        """Enforce rate limiting for API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def get_unique_papers(self) -> List[Dict]:
        """Get unique papers from collection."""
        results = self.collection.get(include=['metadatas'])

        papers = {}
        for meta in results.get('metadatas', []):
            if meta and 'pmid' in meta:
                pmid = meta['pmid']
                if pmid not in papers:
                    papers[pmid] = {
                        'pmid': pmid,
                        'title': meta.get('title', ''),
                        'year': meta.get('year', 0),
                        'journal': meta.get('journal', ''),
                        'has_fulltext': meta.get('has_fulltext', False),
                        'chunk_count': 1
                    }
                else:
                    papers[pmid]['chunk_count'] += 1

        return list(papers.values())

    def fetch_citation_count(self, pmid: str) -> Optional[int]:
        """
        Fetch citation count from Semantic Scholar API.

        Args:
            pmid: PubMed ID

        Returns:
            Citation count or None if not found
        """
        # Check cache first
        if pmid in self.citation_cache:
            cache_entry = self.citation_cache[pmid]
            # Cache valid for 7 days
            cache_age = time.time() - cache_entry.get('timestamp', 0)
            if cache_age < 7 * 24 * 60 * 60:
                return cache_entry.get('citation_count')

        # Rate limiting
        self._rate_limit()

        try:
            url = f"{self.SEMANTIC_SCHOLAR_API}/PMID:{pmid}"
            params = {"fields": "citationCount,title,year,venue"}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                citation_count = data.get('citationCount', 0)

                # Update cache
                self.citation_cache[pmid] = {
                    'citation_count': citation_count,
                    'title': data.get('title', ''),
                    'year': data.get('year'),
                    'venue': data.get('venue', ''),
                    'timestamp': time.time()
                }
                self._save_cache()

                return citation_count

            elif response.status_code == 404:
                # Paper not found in Semantic Scholar
                self.citation_cache[pmid] = {
                    'citation_count': 0,
                    'timestamp': time.time()
                }
                self._save_cache()
                return 0

            else:
                print(f"  API error for PMID {pmid}: {response.status_code}")
                return None

        except Exception as e:
            print(f"  Error fetching citation for PMID {pmid}: {e}")
            return None

    def calculate_quality_score(self, paper: Dict, citation_count: int) -> PaperScore:
        """
        Calculate comprehensive quality score for a paper.

        Scoring breakdown:
        - Citation score (0-40): Based on citation count
        - Recency score (0-30): Based on publication year
        - Content score (0-30): Based on fulltext availability and chunk count

        Args:
            paper: Paper metadata dict
            citation_count: Number of citations

        Returns:
            PaperScore object
        """
        current_year = datetime.now().year
        pub_year = int(paper.get('year', 0)) if paper.get('year') else 0

        # 1. Citation score (0-40)
        # Log scale: 0 citations = 0, 10 = 20, 100 = 30, 1000+ = 40
        if citation_count <= 0:
            citation_score = 0
        elif citation_count < 10:
            citation_score = citation_count * 2  # 0-20
        elif citation_count < 100:
            citation_score = 20 + (citation_count - 10) * 0.11  # 20-30
        elif citation_count < 1000:
            citation_score = 30 + (citation_count - 100) * 0.011  # 30-40
        else:
            citation_score = 40

        # 2. Recency score (0-30)
        # Recent papers get higher scores
        if pub_year == 0:
            recency_score = 15  # Unknown year, neutral score
        else:
            years_old = current_year - pub_year
            if years_old <= 1:
                recency_score = 30
            elif years_old <= 2:
                recency_score = 27
            elif years_old <= 3:
                recency_score = 24
            elif years_old <= 5:
                recency_score = 20
            elif years_old <= 10:
                recency_score = 15
            else:
                recency_score = max(5, 15 - (years_old - 10))

        # 3. Content score (0-30)
        # Based on fulltext availability and content richness
        has_fulltext = paper.get('has_fulltext', False)
        chunk_count = paper.get('chunk_count', 1)

        if has_fulltext:
            content_score = 20  # Base score for fulltext
            # Bonus for rich content (many chunks = detailed paper)
            if chunk_count > 100:
                content_score += 10
            elif chunk_count > 50:
                content_score += 7
            elif chunk_count > 20:
                content_score += 5
            else:
                content_score += 3
        else:
            # Abstract only
            content_score = 10

        # Total quality score
        quality_score = citation_score + recency_score + content_score

        return PaperScore(
            pmid=paper['pmid'],
            title=paper.get('title', '')[:100],
            citation_count=citation_count,
            year=pub_year,
            has_fulltext=has_fulltext,
            journal=paper.get('journal', '')[:50],
            quality_score=round(quality_score, 1),
            citation_score=round(citation_score, 1),
            recency_score=round(recency_score, 1),
            content_score=round(content_score, 1)
        )

    def rank_papers(self, top_n: int = None, fetch_citations: bool = True) -> List[PaperScore]:
        """
        Rank all papers in collection by quality score.

        Args:
            top_n: Return only top N papers (None = all)
            fetch_citations: Whether to fetch citations from API

        Returns:
            List of PaperScore objects sorted by quality_score descending
        """
        papers = self.get_unique_papers()
        print(f"\nFound {len(papers)} unique papers in collection")

        scores = []

        for i, paper in enumerate(papers):
            pmid = paper['pmid']

            if fetch_citations:
                print(f"  [{i+1}/{len(papers)}] Fetching citations for PMID {pmid}...", end=" ")
                citation_count = self.fetch_citation_count(pmid)
                if citation_count is None:
                    citation_count = 0
                print(f"{citation_count} citations")
            else:
                # Use cached value
                cached = self.citation_cache.get(pmid, {})
                citation_count = cached.get('citation_count', 0)

            score = self.calculate_quality_score(paper, citation_count)
            scores.append(score)

        # Sort by quality score descending
        scores.sort(key=lambda x: x.quality_score, reverse=True)

        if top_n:
            scores = scores[:top_n]

        return scores

    def print_ranking(self, scores: List[PaperScore]):
        """Print ranking table."""
        print("\n" + "=" * 100)
        print(f"📊 논문 품질 랭킹: {self.collection_name}")
        print("=" * 100)
        print(f"{'Rank':>4} | {'PMID':>10} | {'Citations':>9} | {'Year':>4} | {'Score':>5} | {'Title':<50}")
        print("-" * 100)

        for i, score in enumerate(scores, 1):
            title = score.title[:47] + "..." if len(score.title) > 50 else score.title
            fulltext_marker = "📄" if score.has_fulltext else "📋"
            print(f"{i:>4} | {score.pmid:>10} | {score.citation_count:>9} | {score.year:>4} | {score.quality_score:>5} | {fulltext_marker} {title:<47}")

        print("-" * 100)

        # Statistics
        if scores:
            avg_citations = sum(s.citation_count for s in scores) / len(scores)
            avg_score = sum(s.quality_score for s in scores) / len(scores)
            fulltext_ratio = sum(1 for s in scores if s.has_fulltext) / len(scores) * 100

            print(f"\n📈 통계:")
            print(f"  평균 인용 수: {avg_citations:.1f}")
            print(f"  평균 품질 점수: {avg_score:.1f}/100")
            print(f"  Full-text 비율: {fulltext_ratio:.1f}%")
            print(f"  최고 인용 논문: PMID {scores[0].pmid} ({scores[0].citation_count} citations)")

    def export_ranking(self, scores: List[PaperScore], output_file: Path = None):
        """Export ranking to JSON file."""
        if output_file is None:
            output_file = self.cache_dir / f"{self.collection_name}_ranking.json"

        export_data = {
            'collection': self.collection_name,
            'generated_at': datetime.now().isoformat(),
            'total_papers': len(scores),
            'papers': [asdict(s) for s in scores]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Ranking exported to: {output_file}")


def rank_all_rnaseq_collections(top_n: int = 20):
    """Rank papers in all RNA-seq collections."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collections = client.list_collections()

    rnaseq_collections = [c.name for c in collections if c.name.startswith('rnaseq_')]

    print(f"\n🧬 RNA-seq 컬렉션 전체 랭킹 ({len(rnaseq_collections)}개)")
    print("=" * 80)

    all_results = {}

    for col_name in sorted(rnaseq_collections):
        print(f"\n\n{'='*60}")
        print(f"Processing: {col_name}")
        print(f"{'='*60}")

        try:
            ranker = PaperCitationRanker(col_name)
            scores = ranker.rank_papers(top_n=top_n, fetch_citations=True)
            ranker.print_ranking(scores)
            ranker.export_ranking(scores)

            all_results[col_name] = {
                'total_papers': len(scores),
                'avg_citations': sum(s.citation_count for s in scores) / len(scores) if scores else 0,
                'avg_score': sum(s.quality_score for s in scores) / len(scores) if scores else 0,
                'top_paper': scores[0].pmid if scores else None,
                'top_citations': scores[0].citation_count if scores else 0
            }
        except Exception as e:
            print(f"Error processing {col_name}: {e}")
            all_results[col_name] = {'error': str(e)}

    # Summary
    print("\n\n" + "=" * 80)
    print("📊 전체 RNA-seq 컬렉션 요약")
    print("=" * 80)
    print(f"{'Collection':<30} | {'Papers':>6} | {'Avg Cit':>8} | {'Avg Score':>9} | {'Top PMID':>10}")
    print("-" * 80)

    for col_name, result in sorted(all_results.items()):
        if 'error' in result:
            print(f"{col_name:<30} | {'ERROR':>6}")
        else:
            print(f"{col_name:<30} | {result['total_papers']:>6} | {result['avg_citations']:>8.1f} | {result['avg_score']:>9.1f} | {result['top_paper']:>10}")


def main():
    parser = argparse.ArgumentParser(description="Paper citation ranking system")
    parser.add_argument("--collection", type=str, help="Collection name to rank")
    parser.add_argument("--all-rnaseq", action="store_true", help="Rank all RNA-seq collections")
    parser.add_argument("--top", type=int, default=20, help="Number of top papers to show")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached citations only")
    parser.add_argument("--list", action="store_true", help="List all collections")

    args = parser.parse_args()

    if args.list:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collections = client.list_collections()
        print("\n📚 Available collections:")
        for col in sorted(collections, key=lambda x: x.name):
            print(f"  {col.name}: {col.count()} chunks")
        return

    if args.all_rnaseq:
        rank_all_rnaseq_collections(top_n=args.top)
    elif args.collection:
        ranker = PaperCitationRanker(args.collection)
        scores = ranker.rank_papers(top_n=args.top, fetch_citations=not args.no_fetch)
        ranker.print_ranking(scores)
        ranker.export_ranking(scores)
    else:
        parser.print_help()
        print("\n예시:")
        print("  python scripts/paper_citation_ranker.py --collection rnaseq_breast_cancer")
        print("  python scripts/paper_citation_ranker.py --all-rnaseq --top 10")
        print("  python scripts/paper_citation_ranker.py --list")


if __name__ == "__main__":
    main()
