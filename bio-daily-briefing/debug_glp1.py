#!/usr/bin/env python3
"""Debug script to test GLP-1 search directly on PubMed."""

import asyncio
import httpx
from datetime import datetime, timedelta

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def search_pubmed_direct(query: str, days: int = 2) -> dict:
    """Search PubMed directly with a specific query."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_filter = f'("{start_date.strftime("%Y/%m/%d")}"[PDAT] : "{end_date.strftime("%Y/%m/%d")}"[PDAT])'

    full_query = f"{query} AND {date_filter}"

    params = {
        "db": "pubmed",
        "term": full_query,
        "retmax": 100,
        "retmode": "json",
        "sort": "pub_date",
        "email": "test@example.com",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        response.raise_for_status()
        data = response.json()

    count = data.get("esearchresult", {}).get("count", "0")
    pmids = data.get("esearchresult", {}).get("idlist", [])

    return {
        "query": query,
        "full_query": full_query,
        "count": int(count),
        "pmids": pmids[:10],
    }


async def main():
    print("=" * 70)
    print("PubMed Direct Search Test - Last 48 hours")
    print("=" * 70)

    # Test queries for predefined hot topics
    test_queries = [
        # GLP-1 관련
        ("GLP-1[tiab]", "GLP-1 (title/abstract)"),
        ("semaglutide[tiab]", "semaglutide"),
        ("tirzepatide[tiab]", "tirzepatide"),
        ("Ozempic[tiab] OR Wegovy[tiab]", "Ozempic/Wegovy"),
        ("glucagon-like peptide[tiab]", "glucagon-like peptide"),

        # 주요 핫토픽
        ("CAR-T[tiab] OR CAR T cell[tiab]", "CAR-T"),
        ("CRISPR[tiab]", "CRISPR"),
        ("checkpoint inhibitor[tiab] OR PD-1[tiab] OR PD-L1[tiab]", "면역관문억제제"),
        ("bispecific antibody[tiab] OR bispecific[tiab]", "이중항체"),
        ("mRNA vaccine[tiab] OR mRNA therapy[tiab]", "mRNA"),

        # 비만/대사
        ("obesity treatment[tiab]", "비만 치료"),
        ("weight loss drug[tiab]", "체중감량약"),

        # AI
        ("AlphaFold[tiab]", "AlphaFold"),
        ("AI drug discovery[tiab]", "AI 신약개발"),

        # 기타
        ("Alzheimer[tiab]", "알츠하이머"),
        ("microbiome therapy[tiab]", "마이크로바이옴"),
    ]

    print("\n[개별 키워드 검색 결과 - 48시간]")
    print("-" * 70)

    total = 0
    for query, label in test_queries:
        result = await search_pubmed_direct(query, days=2)
        count = result["count"]
        total += count
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {label:<25}: {count:>4}건")
        if count > 0 and count <= 5:
            print(f"      PMIDs: {result['pmids']}")

    print(f"\n  총 검색 결과: {total}건")

    # 48시간이 너무 짧으면 7일로 테스트
    print("\n" + "=" * 70)
    print("[7일로 확장 테스트]")
    print("-" * 70)

    key_queries = [
        ("GLP-1[tiab]", "GLP-1"),
        ("semaglutide[tiab]", "semaglutide"),
        ("CAR-T[tiab]", "CAR-T"),
        ("obesity[tiab]", "obesity"),
    ]

    for query, label in key_queries:
        result = await search_pubmed_direct(query, days=7)
        print(f"  {label:<15}: {result['count']:>4}건 (7일)")

    # 현재 DEFAULT_QUERY 테스트
    print("\n" + "=" * 70)
    print("[현재 DEFAULT_QUERY 테스트]")
    print("-" * 70)

    from src.pubmed_fetcher import PubMedFetcher
    fetcher = PubMedFetcher()

    result = await search_pubmed_direct(fetcher.DEFAULT_QUERY, days=2)
    print(f"  DEFAULT_QUERY: {result['count']}건 (48시간)")

    # GLP-1이 포함된 확장 쿼리 테스트
    extended_query = fetcher.DEFAULT_QUERY.replace(")", " OR GLP-1[tiab] OR semaglutide[tiab])")
    result2 = await search_pubmed_direct(extended_query, days=2)
    print(f"  확장 쿼리 (+GLP-1): {result2['count']}건 (48시간)")


if __name__ == "__main__":
    asyncio.run(main())
