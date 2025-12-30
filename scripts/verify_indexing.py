#!/usr/bin/env python3
"""
VectorDB 인덱싱 검증 스크립트

논문별 섹션 분리, 청킹, 메타데이터가 제대로 저장되었는지 확인
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import create_vector_store
from collections import defaultdict
import json


def verify_indexing():
    """인덱싱 상태를 상세 검증"""

    print("=" * 70)
    print("VectorDB 인덱싱 검증 리포트")
    print("=" * 70)

    # Vector Store 연결
    vs = create_vector_store(disease_domain="pheochromocytoma")

    # 모든 데이터 가져오기
    all_data = vs.collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    total_chunks = len(all_data["ids"])
    print(f"\n총 저장된 Chunks: {total_chunks}")
    embeddings = all_data.get('embeddings')
    emb_dim = len(embeddings[0]) if embeddings is not None and len(embeddings) > 0 else 'N/A'
    print(f"Embedding 차원: {emb_dim}")

    # =========================================
    # 1. 논문별 분석
    # =========================================
    print("\n" + "=" * 70)
    print("1. 논문별 Chunk 분포")
    print("=" * 70)

    papers = defaultdict(lambda: {"chunks": 0, "sections": defaultdict(int)})

    for i, meta in enumerate(all_data["metadatas"]):
        paper_title = meta.get("paper_title", "Unknown")[:50]
        section = meta.get("section", "Unknown")
        papers[paper_title]["chunks"] += 1
        papers[paper_title]["sections"][section] += 1
        papers[paper_title]["doi"] = meta.get("doi", "")

    for paper, info in papers.items():
        print(f"\n📄 {paper}...")
        print(f"   DOI: {info['doi']}")
        print(f"   총 Chunks: {info['chunks']}")
        print(f"   섹션별 분포:")
        for section, count in sorted(info["sections"].items()):
            print(f"      - {section}: {count} chunks")

    # =========================================
    # 2. 섹션별 분석
    # =========================================
    print("\n" + "=" * 70)
    print("2. 전체 섹션별 Chunk 분포")
    print("=" * 70)

    section_counts = defaultdict(int)
    for meta in all_data["metadatas"]:
        section = meta.get("section", "Unknown")
        section_counts[section] += 1

    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_chunks) * 100
        bar = "█" * int(pct / 2)
        print(f"  {section:25} {count:4} ({pct:5.1f}%) {bar}")

    # =========================================
    # 3. 메타데이터 필드 확인
    # =========================================
    print("\n" + "=" * 70)
    print("3. 메타데이터 필드 검증")
    print("=" * 70)

    required_fields = ["paper_title", "section", "doi", "disease_domain", "chunk_index", "source_file"]
    field_coverage = {field: 0 for field in required_fields}

    for meta in all_data["metadatas"]:
        for field in required_fields:
            if meta.get(field):
                field_coverage[field] += 1

    print("\n  필드명                    | 채워진 비율")
    print("  " + "-" * 45)
    for field, count in field_coverage.items():
        pct = (count / total_chunks) * 100
        status = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"
        print(f"  {status} {field:22} | {count:4}/{total_chunks} ({pct:.1f}%)")

    # =========================================
    # 4. 샘플 Chunk 확인
    # =========================================
    print("\n" + "=" * 70)
    print("4. 샘플 Chunk 상세 확인 (각 섹션별 1개)")
    print("=" * 70)

    shown_sections = set()
    for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
        section = meta.get("section", "Unknown")
        if section not in shown_sections and len(shown_sections) < 5:
            shown_sections.add(section)
            print(f"\n--- [{section}] 섹션 샘플 ---")
            print(f"Paper: {meta.get('paper_title', 'Unknown')[:60]}...")
            print(f"Chunk Index: {meta.get('chunk_index', 'N/A')}")
            print(f"Disease Domain: {meta.get('disease_domain', 'N/A')}")
            print(f"Content Preview ({len(doc)} chars):")
            print(f"  \"{doc[:200]}...\"")

    # =========================================
    # 5. Embedding 검증
    # =========================================
    print("\n" + "=" * 70)
    print("5. Embedding 벡터 검증")
    print("=" * 70)

    if embeddings is not None and len(embeddings) > 0:
        sample_emb = embeddings[0]
        print(f"\n  Embedding 차원: {len(sample_emb)}")
        print(f"  샘플 벡터 (처음 10개 값):")
        print(f"  {sample_emb[:10]}")

        # 벡터 정규화 확인
        import math
        magnitude = math.sqrt(sum(x**2 for x in sample_emb))
        print(f"  벡터 크기 (L2 norm): {magnitude:.4f}")

    # =========================================
    # 6. 요약
    # =========================================
    print("\n" + "=" * 70)
    print("6. 검증 요약")
    print("=" * 70)

    issues = []
    if len(section_counts) < 3:
        issues.append("⚠️ 섹션 다양성 부족 (3개 미만)")
    if field_coverage["section"] / total_chunks < 0.9:
        issues.append("⚠️ 섹션 메타데이터 누락 있음")
    if embeddings is None or len(embeddings) == 0:
        issues.append("❌ Embedding이 저장되지 않음")

    if not issues:
        print("\n  ✅ 모든 검증 통과!")
        print(f"  - {len(papers)}개 논문 인덱싱 완료")
        print(f"  - {len(section_counts)}개 섹션 유형 감지")
        print(f"  - {total_chunks}개 Chunk + Embedding 저장")
        print(f"  - 메타데이터 정상 저장")
    else:
        print("\n  발견된 이슈:")
        for issue in issues:
            print(f"    {issue}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    verify_indexing()
