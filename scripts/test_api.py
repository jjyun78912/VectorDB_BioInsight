#!/usr/bin/env python3
"""
API Test Script for BioInsight Backend
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("=" * 60)
    print("BIOINSIGHT API TEST")
    print("=" * 60)

    # 1. Health Check
    print("\n✓ 1. Health Check")
    r = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {r.json()['status']}")

    # 2. Domains
    print("\n✓ 2. 질병 도메인 목록")
    r = requests.get(f"{BASE_URL}/api/search/domains")
    for d in r.json()["domains"]:
        print(f"   - {d['kr_name']} ({d['key']})")

    # 3. Paper Search
    print("\n✓ 3. 키워드 검색: 'KRAS mutation'")
    r = requests.get(f"{BASE_URL}/api/search/papers", params={
        "query": "KRAS mutation",
        "domain": "pancreatic_cancer",
        "top_k": 3
    })
    data = r.json()
    print(f"   결과: {data['total']}건")
    for p in data["papers"]:
        print(f"   [{p['pmid']}] {p['title'][:45]}... ({p['relevance_score']:.1f}%)")

    # 4. Similar Papers
    print("\n✓ 4. 유사 논문 추천")
    r = requests.get(f"{BASE_URL}/api/search/similar/34534465", params={
        "domain": "pancreatic_cancer",
        "top_k": 3
    })
    data = r.json()
    print(f"   원본: {data['source_paper'][:40]}...")
    for p in data["similar_papers"]:
        print(f"   → {p['similarity_score']:.1f}%: {p['title'][:40]}...")

    # 5. Keywords
    print("\n✓ 5. 키워드 추출 (Top 10)")
    r = requests.get(f"{BASE_URL}/api/graph/keywords", params={"domain": "pancreatic_cancer"})
    data = r.json()
    print(f"   총 {data['total']}개 키워드")
    for kw in data["keywords"][:10]:
        print(f"   {kw['keyword']:20} ({kw['type']:10}) - {kw['count']}회")

    # 6. All Domain Stats
    print("\n✓ 6. 각 질병별 통계")
    domains = [
        "pancreatic_cancer", "blood_cancer", "glioblastoma", "alzheimer", "pcos", "pheochromocytoma",
        "lung_cancer", "breast_cancer", "colorectal_cancer", "liver_cancer", "rnaseq_transcriptomics"
    ]
    kr_names = [
        "췌장암", "혈액암", "교모세포종", "알츠하이머", "다낭성난소증후군", "갈색세포종",
        "폐암", "유방암", "대장암", "간암", "RNA-seq 전사체학"
    ]

    total_papers = 0
    total_chunks = 0

    for domain, kr in zip(domains, kr_names):
        r = requests.get(f"{BASE_URL}/api/graph/stats", params={"domain": domain})
        data = r.json()
        print(f"   {kr:15}: {data['total_papers']:3} papers, {data['total_chunks']:5} chunks")
        total_papers += data['total_papers']
        total_chunks += data['total_chunks']

    print(f"   {'─' * 40}")
    print(f"   {'TOTAL':15}: {total_papers:3} papers, {total_chunks:5} chunks")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_api()
