#!/usr/bin/env python3
"""Debug script for trend analysis."""

import asyncio
from src.pubmed_fetcher import PubMedFetcher
from src.config.hot_topics import PREDEFINED_HOT_TOPICS


async def debug_keyword_matching():
    """Debug keyword matching for predefined topics."""
    print("=" * 70)
    print("BIO Daily Briefing - Keyword Matching Debug")
    print("=" * 70)

    # Fetch papers
    fetcher = PubMedFetcher()
    print("\n[1] Fetching 100 papers from PubMed (last 48 hours)...")
    papers = await fetcher.fetch_recent_papers(max_results=100, days=2)
    print(f"    Fetched {len(papers)} papers")

    # Debug: Show sample paper content
    print("\n[2] Sample paper content (first 3):")
    for i, paper in enumerate(papers[:3]):
        print(f"\n  Paper {i+1}: {paper.title[:60]}...")
        print(f"    MeSH terms: {paper.mesh_terms[:5]}")
        print(f"    Keywords: {paper.keywords[:5]}")

    # Count predefined topics
    print("\n[3] Predefined topic keyword matching:")
    print("-" * 70)

    topic_results = {}
    for topic_name, topic_info in PREDEFINED_HOT_TOPICS.items():
        keywords = topic_info["keywords"]
        count = 0
        matched_papers = []

        for paper in papers:
            text = f"{paper.title} {paper.abstract} {' '.join(paper.mesh_terms)} {' '.join(paper.keywords)}".lower()

            for keyword in keywords:
                if keyword.lower() in text:
                    count += 1
                    matched_papers.append((paper.pmid, keyword))
                    break  # Only count once per paper

        topic_results[topic_name] = {
            "count": count,
            "keywords": keywords[:3],
            "matched": matched_papers[:3],
        }

    # Sort by count
    sorted_topics = sorted(topic_results.items(), key=lambda x: x[1]["count"], reverse=True)

    for topic_name, data in sorted_topics:
        status = "✅" if data["count"] > 0 else "❌"
        print(f"  {status} {topic_name:<20} : {data['count']:>3}건  (keywords: {data['keywords'][:2]})")
        if data["matched"]:
            for pmid, kw in data["matched"][:2]:
                print(f"      └ PMID:{pmid} matched '{kw}'")

    # Debug: Check specific keywords
    print("\n[4] Direct keyword search in all papers:")
    print("-" * 70)

    test_keywords = ["GLP-1", "glp-1", "semaglutide", "CAR-T", "car-t", "CRISPR", "crispr", "PD-1", "pd-1"]

    for kw in test_keywords:
        count = 0
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            if kw.lower() in text:
                count += 1
        print(f"  '{kw}' : {count}건")

    # Debug: MeSH term analysis
    print("\n[5] Top MeSH terms in fetched papers:")
    print("-" * 70)

    mesh_counts = {}
    for paper in papers:
        for mesh in paper.mesh_terms:
            mesh_lower = mesh.lower()
            mesh_counts[mesh_lower] = mesh_counts.get(mesh_lower, 0) + 1

    top_mesh = sorted(mesh_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for mesh, count in top_mesh:
        print(f"  {mesh:<50} : {count}건")

    # Debug: Check if any paper contains specific terms
    print("\n[6] Papers containing specific drug names:")
    print("-" * 70)

    drug_names = ["semaglutide", "ozempic", "wegovy", "mounjaro", "pembrolizumab", "keytruda"]
    for drug in drug_names:
        matches = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            if drug.lower() in text:
                matches.append(paper.pmid)
        if matches:
            print(f"  ✅ {drug}: {len(matches)}건 - PMIDs: {matches[:3]}")
        else:
            print(f"  ❌ {drug}: 0건")


if __name__ == "__main__":
    asyncio.run(debug_keyword_matching())
