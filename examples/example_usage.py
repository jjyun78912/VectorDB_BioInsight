#!/usr/bin/env python3
"""
VectorDB BioInsight - 사용 예제

Pheochromocytoma (갈색세포종) 논문 Vector DB 사용법
"""
import sys
sys.path.insert(0, '..')

from src.indexer import create_indexer
from src.search import create_searcher
from src.vector_store import create_vector_store


def example_index_papers():
    """논문 PDF 인덱싱 예제"""
    print("=" * 60)
    print("1. 논문 인덱싱")
    print("=" * 60)

    # Pheochromocytoma 전용 인덱서 생성
    indexer = create_indexer(disease_domain="pheochromocytoma")

    # 단일 PDF 인덱싱
    # result = indexer.index_pdf("./data/papers/your_paper.pdf")

    # 디렉토리 전체 인덱싱
    # results = indexer.index_directory("./data/papers/pheochromocytoma")

    # 통계 확인
    stats = indexer.get_stats()
    print(f"Collection: {stats['collection_name']}")
    print(f"Total Papers: {stats['total_papers']}")
    print(f"Total Chunks: {stats['total_chunks']}")


def example_search():
    """유사도 검색 예제"""
    print("\n" + "=" * 60)
    print("2. 유사도 검색")
    print("=" * 60)

    searcher = create_searcher(disease_domain="pheochromocytoma")

    # 기본 검색
    print("\n[기본 검색] 'RET mutation in pheochromocytoma'")
    results = searcher.search(
        query="RET mutation in pheochromocytoma",
        top_k=3
    )
    print(searcher.format_results(results))

    # 섹션 필터 검색
    print("\n[Methods 섹션만 검색] 'RNA sequencing protocol'")
    results = searcher.search(
        query="RNA sequencing protocol",
        top_k=3,
        section_filter="Methods"
    )
    print(searcher.format_results(results))

    # Results/Discussion 검색
    print("\n[Results & Discussion 검색] 'patient survival outcome'")
    results = searcher.search_results_discussion(
        query="patient survival outcome",
        top_k=3
    )
    print(searcher.format_results(results))


def example_metadata_filtering():
    """메타데이터 필터링 예제"""
    print("\n" + "=" * 60)
    print("3. 메타데이터 기반 필터링")
    print("=" * 60)

    vector_store = create_vector_store(disease_domain="pheochromocytoma")

    # 특정 연도 논문만 검색
    results = vector_store.search(
        query="catecholamine synthesis",
        where={"year": "2023"},
        top_k=3
    )
    print("\n[2023년 논문만 검색]")
    for r in results:
        print(f"  - {r.metadata.get('paper_title', 'Unknown')[:50]}...")

    # 특정 섹션 검색
    results = vector_store.search(
        query="genetic mutation analysis",
        where={"section": "Results"},
        top_k=3
    )
    print("\n[Results 섹션만 검색]")
    for r in results:
        print(f"  - Score: {r.relevance_score:.3f} | {r.content[:100]}...")


def example_list_papers():
    """인덱싱된 논문 목록 조회"""
    print("\n" + "=" * 60)
    print("4. 인덱싱된 논문 목록")
    print("=" * 60)

    vector_store = create_vector_store(disease_domain="pheochromocytoma")
    papers = vector_store.get_all_papers()

    print(f"총 {len(papers)}개 논문 인덱싱됨:\n")
    for paper in papers:
        print(f"• {paper['title']}")
        if paper['doi']:
            print(f"  DOI: {paper['doi']}")
        print()


if __name__ == "__main__":
    print("VectorDB BioInsight - Pheochromocytoma Domain")
    print("=" * 60)

    # 인덱싱 예제 (논문 PDF가 있을 때)
    example_index_papers()

    # 검색 예제 (인덱싱 후 실행)
    # example_search()
    # example_metadata_filtering()
    # example_list_papers()
