#!/usr/bin/env python3
"""
Index existing papers from data/papers/ into ChromaDB.

This script indexes papers that have already been downloaded
but were never indexed into the vector database.

Usage:
    python scripts/index_existing_papers.py
    python scripts/index_existing_papers.py --folder breast_cancer
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pubmed_collector import PubMedCollector, DISEASE_CONFIGS
from backend.app.core.config import PAPERS_DIR


def index_folder(folder_name: str) -> dict:
    """Index papers from a specific folder."""
    folder_path = PAPERS_DIR / folder_name

    if not folder_path.exists():
        return {"success": False, "error": f"Folder not found: {folder_path}"}

    json_files = [f for f in folder_path.glob("*.json") if f.name != "_index.json"]

    if not json_files:
        return {"success": False, "error": "No JSON files found"}

    print(f"\n{'='*60}")
    print(f"Indexing: {folder_name} ({len(json_files)} papers)")
    print(f"{'='*60}")

    # Check if folder is in DISEASE_CONFIGS
    if folder_name not in DISEASE_CONFIGS:
        print(f"  Warning: {folder_name} not in DISEASE_CONFIGS, skipping...")
        return {"success": False, "error": "Not in DISEASE_CONFIGS", "papers": len(json_files)}

    try:
        collector = PubMedCollector(folder_name)
        collector.index_to_vectordb()

        stats = collector.vector_store.get_collection_stats()
        return {
            "success": True,
            "papers": len(json_files),
            "chunks": stats.get("total_chunks", 0)
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"success": False, "error": str(e), "papers": len(json_files)}


def index_all_papers():
    """Index all papers from all disease folders."""
    print("\n" + "="*70)
    print("BioInsight VectorDB - Batch Paper Indexing")
    print("="*70)

    # Get all disease folders
    folders = [d for d in PAPERS_DIR.iterdir() if d.is_dir()]

    print(f"Found {len(folders)} disease folders:")
    for folder in sorted(folders):
        json_count = len([f for f in folder.glob("*.json") if f.name != "_index.json"])
        config_status = "✓" if folder.name in DISEASE_CONFIGS else "✗"
        print(f"  {config_status} {folder.name}: {json_count} papers")

    print("="*70)

    results = {}
    total_papers = 0
    total_chunks = 0

    for folder in sorted(folders):
        folder_name = folder.name

        # Skip rnaseq folder if empty or if it's a parent folder
        if folder_name == "rnaseq":
            continue

        result = index_folder(folder_name)
        results[folder_name] = result

        if result.get("success"):
            total_papers += result.get("papers", 0)
            total_chunks += result.get("chunks", 0)

    # Print summary
    print("\n" + "="*70)
    print("INDEXING SUMMARY")
    print("="*70)

    success_count = 0
    for folder_name, result in results.items():
        if result.get("success"):
            print(f"  ✓ {folder_name:25}: {result['papers']:3} papers -> {result['chunks']:,} chunks")
            success_count += 1
        else:
            error = result.get("error", "Unknown error")
            papers = result.get("papers", 0)
            if "Not in DISEASE_CONFIGS" in error:
                print(f"  ⚠ {folder_name:25}: {papers:3} papers (skipped - not configured)")
            else:
                print(f"  ✗ {folder_name:25}: FAILED - {error}")

    print("-" * 70)
    print(f"  {'TOTAL':25}: {total_papers} papers -> {total_chunks:,} chunks")
    print(f"  Successful folders: {success_count}/{len(results)}")
    print("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Index existing papers into ChromaDB"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Specific folder to index (default: all)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available folders and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable folders in data/papers:")
        for folder in sorted(PAPERS_DIR.iterdir()):
            if folder.is_dir():
                json_count = len([f for f in folder.glob("*.json") if f.name != "_index.json"])
                config_status = "✓" if folder.name in DISEASE_CONFIGS else "✗"
                print(f"  {config_status} {folder.name}: {json_count} papers")
        return

    if args.folder:
        index_folder(args.folder)
    else:
        index_all_papers()


if __name__ == "__main__":
    main()
