"""
RNA-seq 특화 논문 수집 스크립트.

각 암종별로 RNA-seq, transcriptomics, DEG, gene expression 관련
핵심 논문들을 수집하여 VectorDB에 인덱싱합니다.

Usage:
    python scripts/collect_rnaseq_papers.py --all --count 50
    python scripts/collect_rnaseq_papers.py --cancer pancreatic_cancer --count 50
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pubmed_collector import PubMedCollector, PaperInfo, DISEASE_CONFIGS
from backend.app.core.config import PAPERS_DIR, CHROMA_DIR
from backend.app.core.vector_store import BioVectorStore

# RNA-seq 특화 쿼리 (암종별)
RNASEQ_CANCER_QUERIES = {
    "pancreatic_cancer": {
        "name": "Pancreatic Cancer RNA-seq",
        "query": "(pancreatic cancer[Title/Abstract] OR PDAC[Title/Abstract] OR pancreatic adenocarcinoma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract] OR DEG[Title/Abstract]) AND (KRAS OR TP53 OR SMAD4 OR CDKN2A OR hub gene OR driver gene OR biomarker OR prognosis)",
        "kr_name": "췌장암 RNA-seq"
    },
    "breast_cancer": {
        "name": "Breast Cancer RNA-seq",
        "query": "(breast cancer[Title/Abstract] OR breast carcinoma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (BRCA1 OR BRCA2 OR HER2 OR ESR1 OR triple negative OR PAM50 OR molecular subtype OR hub gene OR prognosis)",
        "kr_name": "유방암 RNA-seq"
    },
    "lung_cancer": {
        "name": "Lung Cancer RNA-seq",
        "query": "(lung cancer[Title/Abstract] OR NSCLC[Title/Abstract] OR lung adenocarcinoma[Title/Abstract] OR LUAD[Title/Abstract] OR lung squamous[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (EGFR OR ALK OR KRAS OR TP53 OR hub gene OR driver mutation OR biomarker)",
        "kr_name": "폐암 RNA-seq"
    },
    "colorectal_cancer": {
        "name": "Colorectal Cancer RNA-seq",
        "query": "(colorectal cancer[Title/Abstract] OR colon cancer[Title/Abstract] OR CRC[Title/Abstract] OR rectal cancer[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (APC OR KRAS OR BRAF OR MSI OR CMS OR hub gene OR prognosis)",
        "kr_name": "대장암 RNA-seq"
    },
    "liver_cancer": {
        "name": "Liver Cancer RNA-seq",
        "query": "(hepatocellular carcinoma[Title/Abstract] OR HCC[Title/Abstract] OR liver cancer[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (TP53 OR CTNNB1 OR AFP OR hub gene OR prognosis OR TCGA)",
        "kr_name": "간암 RNA-seq"
    },
    "glioblastoma": {
        "name": "Glioblastoma RNA-seq",
        "query": "(glioblastoma[Title/Abstract] OR GBM[Title/Abstract] OR glioma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (IDH OR MGMT OR EGFR OR TERT OR molecular subtype OR hub gene OR prognosis)",
        "kr_name": "교모세포종 RNA-seq"
    },
    "blood_cancer": {
        "name": "Blood Cancer RNA-seq",
        "query": "(leukemia[Title/Abstract] OR lymphoma[Title/Abstract] OR AML[Title/Abstract] OR ALL[Title/Abstract] OR CLL[Title/Abstract] OR myeloma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (mutation OR fusion gene OR hub gene OR prognosis OR molecular classification)",
        "kr_name": "혈액암 RNA-seq"
    },
    "thyroid_cancer": {
        "name": "Thyroid Cancer RNA-seq",
        "query": "(thyroid cancer[Title/Abstract] OR papillary thyroid[Title/Abstract] OR PTC[Title/Abstract] OR thyroid carcinoma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (BRAF OR RET OR RAS OR hub gene OR prognosis OR molecular subtype)",
        "kr_name": "갑상선암 RNA-seq",
        "tcga_code": "THCA"
    },
    # ============================================================
    # ML Pan-Cancer 17종 중 누락된 9개 암종 추가
    # ============================================================
    "bladder_cancer": {
        "name": "Bladder Cancer RNA-seq",
        "query": "(bladder cancer[Title/Abstract] OR urothelial carcinoma[Title/Abstract] OR BLCA[Title/Abstract] OR bladder urothelial[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (FGFR3 OR TP53 OR RB1 OR molecular subtype OR hub gene OR prognosis OR TCGA)",
        "kr_name": "방광암 RNA-seq",
        "tcga_code": "BLCA"
    },
    "head_neck_cancer": {
        "name": "Head and Neck Cancer RNA-seq",
        "query": "(head and neck cancer[Title/Abstract] OR HNSCC[Title/Abstract] OR head neck squamous[Title/Abstract] OR oral cancer[Title/Abstract] OR oropharyngeal[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (HPV OR TP53 OR CDKN2A OR PIK3CA OR hub gene OR prognosis OR molecular subtype)",
        "kr_name": "두경부암 RNA-seq",
        "tcga_code": "HNSC"
    },
    "kidney_cancer": {
        "name": "Kidney Cancer RNA-seq",
        "query": "(kidney cancer[Title/Abstract] OR renal cell carcinoma[Title/Abstract] OR clear cell renal[Title/Abstract] OR ccRCC[Title/Abstract] OR KIRC[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (VHL OR PBRM1 OR BAP1 OR hub gene OR prognosis OR molecular subtype OR TCGA)",
        "kr_name": "신장암 RNA-seq",
        "tcga_code": "KIRC"
    },
    "low_grade_glioma": {
        "name": "Low Grade Glioma RNA-seq",
        "query": "(low grade glioma[Title/Abstract] OR LGG[Title/Abstract] OR diffuse glioma[Title/Abstract] OR astrocytoma[Title/Abstract] OR oligodendroglioma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (IDH1 OR IDH2 OR 1p19q OR ATRX OR hub gene OR prognosis OR molecular classification)",
        "kr_name": "저등급 신경교종 RNA-seq",
        "tcga_code": "LGG"
    },
    "ovarian_cancer": {
        "name": "Ovarian Cancer RNA-seq",
        "query": "(ovarian cancer[Title/Abstract] OR ovarian carcinoma[Title/Abstract] OR high grade serous[Title/Abstract] OR HGSOC[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (BRCA1 OR BRCA2 OR TP53 OR homologous recombination OR hub gene OR prognosis OR molecular subtype)",
        "kr_name": "난소암 RNA-seq",
        "tcga_code": "OV"
    },
    "prostate_cancer": {
        "name": "Prostate Cancer RNA-seq",
        "query": "(prostate cancer[Title/Abstract] OR prostate adenocarcinoma[Title/Abstract] OR PRAD[Title/Abstract] OR castration resistant[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (AR OR TMPRSS2 OR ERG OR PTEN OR hub gene OR prognosis OR Gleason OR molecular subtype)",
        "kr_name": "전립선암 RNA-seq",
        "tcga_code": "PRAD"
    },
    "melanoma": {
        "name": "Melanoma RNA-seq",
        "query": "(melanoma[Title/Abstract] OR cutaneous melanoma[Title/Abstract] OR SKCM[Title/Abstract] OR skin melanoma[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (BRAF OR NRAS OR CDKN2A OR immune checkpoint OR hub gene OR prognosis OR molecular subtype)",
        "kr_name": "피부흑색종 RNA-seq",
        "tcga_code": "SKCM"
    },
    "stomach_cancer": {
        "name": "Stomach Cancer RNA-seq",
        "query": "(gastric cancer[Title/Abstract] OR stomach cancer[Title/Abstract] OR gastric adenocarcinoma[Title/Abstract] OR STAD[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (CDH1 OR TP53 OR HER2 OR MSI OR EBV OR hub gene OR prognosis OR Lauren classification)",
        "kr_name": "위암 RNA-seq",
        "tcga_code": "STAD"
    },
    "uterine_cancer": {
        "name": "Uterine Cancer RNA-seq",
        "query": "(endometrial cancer[Title/Abstract] OR uterine cancer[Title/Abstract] OR endometrial carcinoma[Title/Abstract] OR UCEC[Title/Abstract]) AND (RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptome[Title/Abstract] OR gene expression profiling[Title/Abstract] OR differential expression[Title/Abstract]) AND (PTEN OR PIK3CA OR TP53 OR MSI OR POLE OR hub gene OR prognosis OR molecular subtype)",
        "kr_name": "자궁내막암 RNA-seq",
        "tcga_code": "UCEC"
    }
}


class RNAseqPaperCollector(PubMedCollector):
    """RNA-seq 특화 논문 수집기."""

    def __init__(self, cancer_key: str, api_key: str = None):
        """
        Initialize RNA-seq paper collector.

        Args:
            cancer_key: Cancer type key from RNASEQ_CANCER_QUERIES
            api_key: Optional NCBI API key
        """
        if cancer_key not in RNASEQ_CANCER_QUERIES:
            raise ValueError(f"Unknown cancer type: {cancer_key}. Valid options: {list(RNASEQ_CANCER_QUERIES.keys())}")

        self.cancer_key = cancer_key
        self.rnaseq_config = RNASEQ_CANCER_QUERIES[cancer_key]
        self.api_key = api_key or os.getenv("NCBI_API_KEY", "")

        # Set up directories - use rnaseq subdirectory
        self.disease_dir = PAPERS_DIR / "rnaseq" / cancer_key
        self.disease_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.request_delay = 0.34 if self.api_key else 1.0

        # Collection name with rnaseq prefix
        collection_name = f"rnaseq_{cancer_key}"

        # Initialize vector store
        from backend.app.core.vector_store import BioVectorStore
        from backend.app.core.text_splitter import BioPaperSplitter

        self.vector_store = BioVectorStore(
            collection_name=collection_name,
            persist_directory=CHROMA_DIR
        )

        self.text_splitter = BioPaperSplitter()

        # Override disease_config for search
        self.disease_config = self.rnaseq_config
        self.disease_key = cancer_key

        print(f"Initialized RNA-seq collector for: {self.rnaseq_config['name']} ({self.rnaseq_config['kr_name']})")
        print(f"Papers directory: {self.disease_dir}")
        print(f"Vector store collection: {self.vector_store.collection_name}")


def collect_all_rnaseq_papers(target_per_cancer: int = 50):
    """Collect RNA-seq papers for all cancer types."""
    print("\n" + "=" * 70)
    print("🧬 RNA-seq 특화 논문 수집 시작")
    print("=" * 70)
    print(f"암종당 목표 논문 수: {target_per_cancer}")
    print(f"대상 암종: {list(RNASEQ_CANCER_QUERIES.keys())}")
    print("=" * 70 + "\n")

    results = {}
    total_papers = 0
    total_chunks = 0

    for cancer_key in RNASEQ_CANCER_QUERIES.keys():
        try:
            print(f"\n{'='*60}")
            print(f"수집 중: {RNASEQ_CANCER_QUERIES[cancer_key]['kr_name']}")
            print(f"{'='*60}")

            collector = RNAseqPaperCollector(cancer_key)
            papers = collector.collect_papers(target_count=target_per_cancer)
            collector.index_to_vectordb(papers)

            chunk_count = collector.vector_store.count
            results[cancer_key] = {
                "success": True,
                "papers": len(papers),
                "chunks": chunk_count
            }
            total_papers += len(papers)
            total_chunks += chunk_count

            print(f"✓ {cancer_key}: {len(papers)}편, {chunk_count} chunks")

        except Exception as e:
            print(f"✗ ERROR collecting {cancer_key}: {e}")
            results[cancer_key] = {
                "success": False,
                "error": str(e)
            }

    # Print summary
    print("\n" + "=" * 70)
    print("📊 RNA-seq 논문 수집 완료 요약")
    print("=" * 70)

    for cancer_key, result in results.items():
        name = RNASEQ_CANCER_QUERIES[cancer_key]["kr_name"]
        if result["success"]:
            print(f"  ✓ {name:20} : {result['papers']:3}편 → {result['chunks']:,} chunks")
        else:
            print(f"  ✗ {name:20} : FAILED - {result['error']}")

    print("-" * 70)
    print(f"  {'TOTAL':20} : {total_papers}편 → {total_chunks:,} chunks")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect RNA-seq focused papers for cancer research")
    parser.add_argument("--cancer", type=str, help="Specific cancer type to collect")
    parser.add_argument("--all", action="store_true", help="Collect all cancer types")
    parser.add_argument("--count", type=int, default=50, help="Number of papers per cancer type")
    parser.add_argument("--list", action="store_true", help="List available cancer types")

    args = parser.parse_args()

    if args.list:
        print("\n🧬 RNA-seq 수집 가능 암종:")
        for key, config in RNASEQ_CANCER_QUERIES.items():
            print(f"  {key:25} : {config['name']} ({config['kr_name']})")
        return

    if args.all:
        collect_all_rnaseq_papers(target_per_cancer=args.count)
    elif args.cancer:
        collector = RNAseqPaperCollector(args.cancer)
        papers = collector.collect_papers(target_count=args.count)
        collector.index_to_vectordb(papers)
    else:
        parser.print_help()
        print("\n예시:")
        print("  python scripts/collect_rnaseq_papers.py --all --count 50")
        print("  python scripts/collect_rnaseq_papers.py --cancer pancreatic_cancer --count 50")


if __name__ == "__main__":
    main()
