#!/usr/bin/env python3
"""
BIO Daily Briefing v2 Test Script

Tests the multi-source fetching and newsletter generation.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sources.fda_fetcher import FDAFetcher
from src.sources.clinicaltrials_fetcher import ClinicalTrialsFetcher
from src.sources.biorxiv_fetcher import BioRxivFetcher
from src.aggregator import NewsAggregator
from src.newsletter_v2 import NewsletterGeneratorV2


async def test_fda():
    """Test FDA fetcher."""
    print("\n" + "=" * 60)
    print("🏥 FDA Fetcher Test")
    print("=" * 60)

    fetcher = FDAFetcher()
    news = await fetcher.fetch_recent(hours=168)  # 1 week

    print(f"\n총 수집: {len(news)}건")

    # Show by type
    types = {}
    for n in news:
        types[n.source_type] = types.get(n.source_type, 0) + 1

    print("\n유형별 분포:")
    for t, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t}: {count}건")

    # Top items
    if news:
        print("\n🔝 Top 5 FDA 뉴스:")
        for n in news[:5]:
            print(f"  [{n.priority}] {n.title[:60]}...")

    return len(news) > 0


async def test_clinicaltrials():
    """Test ClinicalTrials fetcher."""
    print("\n" + "=" * 60)
    print("📊 ClinicalTrials.gov Fetcher Test")
    print("=" * 60)

    fetcher = ClinicalTrialsFetcher()
    data = await fetcher.fetch_all(results_days=60, new_trials_days=30)

    results = data.get("phase3_results", [])
    new_trials = data.get("new_trials", [])
    terminated = data.get("terminated", [])

    print(f"\nPhase 3 결과: {len(results)}건")
    print(f"신규 임상: {len(new_trials)}건")
    print(f"중단/실패: {len(terminated)}건")

    if results:
        print("\n🔝 Top Phase 3 결과:")
        for t in sorted(results, key=lambda x: x.priority, reverse=True)[:3]:
            print(f"  [{t.priority}] {t.title[:50]}...")
            print(f"      Sponsor: {t.sponsor}")

    return len(results) > 0 or len(new_trials) > 0


async def test_biorxiv():
    """Test bioRxiv/medRxiv fetcher."""
    print("\n" + "=" * 60)
    print("🔬 bioRxiv/medRxiv Fetcher Test")
    print("=" * 60)

    fetcher = BioRxivFetcher()
    preprints = await fetcher.fetch_all_servers(days=3)

    print(f"\n총 수집: {len(preprints)}건")

    # By server
    biorxiv_count = len([p for p in preprints if p.server == "biorxiv"])
    medrxiv_count = len([p for p in preprints if p.server == "medrxiv"])
    print(f"  bioRxiv: {biorxiv_count}건")
    print(f"  medRxiv: {medrxiv_count}건")

    # Top with hot keywords
    top = fetcher.get_top_preprints(preprints, n=5, require_keywords=True)
    if top:
        print("\n🔝 핫키워드 프리프린트:")
        for p in top:
            print(f"  [{p.priority}] {p.title[:50]}...")
            print(f"      Category: {p.category}")

    return len(preprints) > 0


async def test_aggregator():
    """Test full aggregator."""
    print("\n" + "=" * 60)
    print("🔄 Full Aggregator Test")
    print("=" * 60)

    aggregator = NewsAggregator()
    briefing = await aggregator.aggregate_daily(
        fda_hours=168,
        trials_days=60,
        preprint_days=5,
        pubmed_days=2,
        issue_number=999,  # Test issue
    )

    print("\n📋 Aggregation Results:")
    print(f"  FDA News: {briefing.total_fda}건")
    print(f"  Clinical Trials: {briefing.total_trials}건")
    print(f"  Preprints: {briefing.total_preprints}건")
    print(f"  PubMed Papers: {briefing.total_papers}건")

    if briefing.headline:
        print(f"\n🚨 Headline: {briefing.headline.title[:60]}...")

    if briefing.hot_topics:
        print(f"\n📈 Hot Topics:")
        for t in briefing.hot_topics[:5]:
            print(f"  {t.trend_indicator} {t.keyword}: {t.count}건")

    return briefing


async def test_newsletter_generation(briefing):
    """Test newsletter generation."""
    print("\n" + "=" * 60)
    print("📰 Newsletter Generation Test")
    print("=" * 60)

    # Add sample editor comment
    briefing.editor_comment = """오늘의 BIO 데일리 브리핑 v2 테스트입니다.

**멀티소스 분석**을 통해 FDA, ClinicalTrials.gov, bioRxiv/medRxiv, PubMed 4개 소스에서 데이터를 수집했습니다.

주요 업데이트:
- FDA 규제 뉴스 실시간 반영
- Phase 3 임상시험 결과 추적
- 프리프린트 속보 모니터링
- PubMed 핫토픽 분석

내일도 함께해주세요! 🏥"""

    generator = NewsletterGeneratorV2()
    html_path, json_path = generator.generate_and_save(briefing)

    print(f"\n✅ Newsletter generated:")
    print(f"   HTML: {html_path}")
    print(f"   JSON: {json_path}")

    return html_path, json_path


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 BIO Daily Briefing v2 - 전체 테스트")
    print("=" * 60)

    results = {}

    # Individual fetcher tests
    try:
        results["FDA"] = await test_fda()
    except Exception as e:
        print(f"\n❌ FDA Test Failed: {e}")
        results["FDA"] = False

    try:
        results["ClinicalTrials"] = await test_clinicaltrials()
    except Exception as e:
        print(f"\n❌ ClinicalTrials Test Failed: {e}")
        results["ClinicalTrials"] = False

    try:
        results["bioRxiv"] = await test_biorxiv()
    except Exception as e:
        print(f"\n❌ bioRxiv Test Failed: {e}")
        results["bioRxiv"] = False

    # Full aggregator test
    try:
        briefing = await test_aggregator()
        results["Aggregator"] = briefing is not None
    except Exception as e:
        print(f"\n❌ Aggregator Test Failed: {e}")
        results["Aggregator"] = False
        briefing = None

    # Newsletter generation
    if briefing:
        try:
            html_path, json_path = await test_newsletter_generation(briefing)
            results["Newsletter"] = True
        except Exception as e:
            print(f"\n❌ Newsletter Test Failed: {e}")
            results["Newsletter"] = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\n{'✅ All tests passed!' if all_passed else '⚠️ Some tests failed'}")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
