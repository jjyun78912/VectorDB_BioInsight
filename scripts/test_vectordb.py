"""
Test script for BioInsight VectorDB.
Verifies that all disease collections are working properly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import BioVectorStore
from src.config import CHROMA_DIR

# Disease collections to test
DISEASES = [
    ("pancreatic_cancer", "췌장암", "pancreatic cancer treatment"),
    ("blood_cancer", "혈액암", "leukemia chemotherapy"),
    ("glioblastoma", "교모세포종", "glioblastoma temozolomide"),
    ("alzheimer", "알츠하이머", "amyloid beta tau protein"),
    ("pcos", "다낭성난소증후군", "polycystic ovary insulin resistance"),
    ("pheochromocytoma", "갈색세포종", "pheochromocytoma paraganglioma catecholamine"),
]


def test_collection(disease_key: str, kr_name: str, test_query: str):
    """Test a single disease collection."""
    print(f"\n{'='*60}")
    print(f"Testing: {kr_name} ({disease_key})")
    print(f"{'='*60}")

    try:
        # Initialize vector store
        vs = BioVectorStore(
            disease_domain=disease_key,
            persist_directory=CHROMA_DIR
        )

        # Get stats
        stats = vs.get_collection_stats()
        print(f"✓ Collection: {stats['collection_name']}")
        print(f"  - Total chunks: {stats['total_chunks']}")
        print(f"  - Total papers: {stats['total_papers']}")
        print(f"  - Embedding model: {stats['embedding_model']}")

        # Test search
        print(f"\n  Query: \"{test_query}\"")
        results = vs.search(test_query, top_k=3)

        if results:
            print(f"  Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                title = r.metadata.get('paper_title', 'Unknown')[:50]
                section = r.metadata.get('section', 'Unknown')
                score = r.relevance_score
                print(f"    [{i}] {title}...")
                print(f"        Section: {section} | Relevance: {score:.1f}%")
                print(f"        Preview: {r.content[:100]}...")
        else:
            print("  ⚠ No results found")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("BIOINSIGHT VECTORDB - TEST SUITE")
    print("="*70)

    results = {}

    for disease_key, kr_name, test_query in DISEASES:
        success = test_collection(disease_key, kr_name, test_query)
        results[disease_key] = success

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for disease_key, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {disease_key}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All tests passed! VectorDB is working correctly.")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
