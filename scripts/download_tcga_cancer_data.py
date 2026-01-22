#!/usr/bin/env python3
"""
TCGA Cancer RNA-seq Data Downloader
===================================

GDC API를 사용하여 TCGA 암종별 RNA-seq 데이터를 다운로드합니다.
이미 전처리된 HTSeq-Counts/FPKM 데이터를 가져옵니다.

사용법:
    python scripts/download_tcga_cancer_data.py --cancer PAAD --max-samples 200
    python scripts/download_tcga_cancer_data.py --all-low-performance
    python scripts/download_tcga_cancer_data.py --list
"""

import os
import sys
import json
import gzip
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GDC API endpoints
GDC_API = "https://api.gdc.cancer.gov"
GDC_FILES = f"{GDC_API}/files"
GDC_DATA = f"{GDC_API}/data"

# TCGA 암종 정보
TCGA_CANCER_TYPES = {
    # 저성능 암종 (Priority 1)
    'PAAD': {'full_name': 'Pancreatic Adenocarcinoma', 'priority': 1},
    'PRAD': {'full_name': 'Prostate Adenocarcinoma', 'priority': 1},
    'BLCA': {'full_name': 'Bladder Urothelial Carcinoma', 'priority': 1},
    'OV': {'full_name': 'Ovarian Serous Cystadenocarcinoma', 'priority': 1},
    # 중간 성능 (Priority 2)
    'LUSC': {'full_name': 'Lung Squamous Cell Carcinoma', 'priority': 2},
    'KIRC': {'full_name': 'Kidney Renal Clear Cell Carcinoma', 'priority': 2},
    'LIHC': {'full_name': 'Liver Hepatocellular Carcinoma', 'priority': 2},
    'LUAD': {'full_name': 'Lung Adenocarcinoma', 'priority': 2},
    'HNSC': {'full_name': 'Head and Neck Squamous Cell Carcinoma', 'priority': 2},
    'STAD': {'full_name': 'Stomach Adenocarcinoma', 'priority': 2},
    'UCEC': {'full_name': 'Uterine Corpus Endometrial Carcinoma', 'priority': 2},
    'SKCM': {'full_name': 'Skin Cutaneous Melanoma', 'priority': 2},
    'GBM': {'full_name': 'Glioblastoma Multiforme', 'priority': 2},
    'LGG': {'full_name': 'Brain Lower Grade Glioma', 'priority': 2},
    # 고성능 (Priority 3) - 참고용
    'BRCA': {'full_name': 'Breast Invasive Carcinoma', 'priority': 3},
    'COAD': {'full_name': 'Colon Adenocarcinoma', 'priority': 3},
    'THCA': {'full_name': 'Thyroid Carcinoma', 'priority': 3},
}


class TCGADownloader:
    """GDC API를 통한 TCGA 데이터 다운로더"""

    def __init__(self, output_dir: str = "data/tcga"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def query_files(self, project_id: str, data_type: str = "Gene Expression Quantification",
                   workflow_type: str = "STAR - Counts",
                   max_files: int = 200) -> List[Dict]:
        """GDC에서 파일 목록 쿼리"""

        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": f"TCGA-{project_id}"
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "data_type",
                        "value": data_type
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": workflow_type
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "data_format",
                        "value": "TSV"
                    }
                }
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
            "format": "JSON",
            "size": max_files
        }

        try:
            response = requests.get(GDC_FILES, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            files = data.get('data', {}).get('hits', [])
            logger.info(f"Found {len(files)} files for TCGA-{project_id}")
            return files

        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def download_file(self, file_id: str, output_path: Path) -> bool:
        """단일 파일 다운로드"""
        try:
            response = requests.get(f"{GDC_DATA}/{file_id}",
                                   headers={"Content-Type": "application/json"},
                                   timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True

        except Exception as e:
            logger.error(f"Download error for {file_id}: {e}")
            return False

    def download_batch(self, file_ids: List[str], cancer_code: str) -> List[Path]:
        """배치 다운로드 (POST 요청)"""
        output_path = self.output_dir / cancer_code
        output_path.mkdir(parents=True, exist_ok=True)

        # 이미 다운로드된 파일 확인
        downloaded = []
        to_download = []

        for fid in file_ids:
            local_file = output_path / f"{fid}.tsv.gz"
            if local_file.exists():
                downloaded.append(local_file)
            else:
                to_download.append(fid)

        if not to_download:
            logger.info(f"All {len(file_ids)} files already downloaded")
            return downloaded

        logger.info(f"Downloading {len(to_download)} files...")

        # 배치로 다운로드 (최대 100개씩)
        batch_size = 50
        for i in range(0, len(to_download), batch_size):
            batch = to_download[i:i+batch_size]

            payload = {"ids": batch}

            try:
                response = requests.post(
                    GDC_DATA,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=300
                )
                response.raise_for_status()

                # tar.gz 형태로 받음
                import tarfile
                import io

                tar_data = io.BytesIO(response.content)
                with tarfile.open(fileobj=tar_data, mode='r:gz') as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            # 파일 추출
                            file_content = tar.extractfile(member)
                            if file_content:
                                # 파일명에서 file_id 추출
                                file_name = Path(member.name).name
                                local_path = output_path / file_name

                                with open(local_path, 'wb') as f:
                                    f.write(file_content.read())
                                downloaded.append(local_path)

                logger.info(f"Downloaded batch {i//batch_size + 1}: {len(batch)} files")
                time.sleep(1)

            except Exception as e:
                logger.error(f"Batch download error: {e}")
                # 개별 다운로드 시도
                for fid in batch:
                    local_file = output_path / f"{fid}.tsv"
                    if self.download_file(fid, local_file):
                        downloaded.append(local_file)
                    time.sleep(0.5)

        return downloaded

    def merge_expression_files(self, file_paths: List[Path], cancer_code: str) -> pd.DataFrame:
        """다운로드된 파일들을 하나의 행렬로 병합"""
        logger.info(f"Merging {len(file_paths)} expression files...")

        all_data = {}
        gene_names = None

        for fp in file_paths:
            try:
                # TSV 파일 읽기 (gzip 또는 plain)
                if fp.suffix == '.gz':
                    df = pd.read_csv(fp, sep='\t', compression='gzip', comment='#')
                else:
                    df = pd.read_csv(fp, sep='\t', comment='#')

                # Gene ID와 Count 컬럼 찾기
                if 'gene_id' in df.columns:
                    gene_col = 'gene_id'
                elif 'gene_name' in df.columns:
                    gene_col = 'gene_name'
                else:
                    gene_col = df.columns[0]

                # Count 컬럼 찾기 (unstranded, stranded_first 등)
                count_cols = ['unstranded', 'stranded_first', 'stranded_second',
                             'tpm_unstranded', 'fpkm_unstranded']
                count_col = None
                for cc in count_cols:
                    if cc in df.columns:
                        count_col = cc
                        break

                if count_col is None:
                    count_col = df.columns[-1]  # 마지막 컬럼 사용

                # 샘플명 추출 (파일명에서)
                sample_id = fp.stem.replace('.tsv', '')

                # 데이터 추가
                series = df.set_index(gene_col)[count_col]
                all_data[sample_id] = series

                if gene_names is None:
                    gene_names = series.index

            except Exception as e:
                logger.warning(f"Error reading {fp}: {e}")

        if not all_data:
            return pd.DataFrame()

        # 병합
        merged = pd.DataFrame(all_data)

        # ENSG ID에서 버전 제거
        if merged.index.str.startswith('ENSG').any():
            merged.index = merged.index.str.split('.').str[0]

        logger.info(f"Merged: {merged.shape[0]} genes x {merged.shape[1]} samples")

        return merged

    def download_cancer_data(self, cancer_code: str, max_samples: int = 200) -> Dict:
        """암종별 TCGA 데이터 다운로드"""
        if cancer_code not in TCGA_CANCER_TYPES:
            logger.error(f"Unknown cancer code: {cancer_code}")
            return {}

        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading TCGA-{cancer_code} ({TCGA_CANCER_TYPES[cancer_code]['full_name']})")
        logger.info(f"{'='*60}")

        # 파일 쿼리
        files = self.query_files(cancer_code, max_files=max_samples)

        if not files:
            logger.warning(f"No files found for {cancer_code}")
            return {}

        # Primary Tumor 샘플만 필터링
        primary_files = []
        for f in files:
            cases = f.get('cases', [])
            if cases:
                samples = cases[0].get('samples', [])
                if samples:
                    sample_type = samples[0].get('sample_type', '')
                    if 'Primary Tumor' in sample_type or 'Tumor' in sample_type:
                        primary_files.append(f)

        logger.info(f"Primary tumor samples: {len(primary_files)}")

        if not primary_files:
            primary_files = files  # Fallback

        # 파일 ID 추출
        file_ids = [f['file_id'] for f in primary_files[:max_samples]]

        # 다운로드
        downloaded = self.download_batch(file_ids, cancer_code)

        # 병합
        if downloaded:
            merged = self.merge_expression_files(downloaded, cancer_code)

            if len(merged) > 0:
                # 저장
                output_file = self.output_dir / cancer_code / "expression_matrix.csv"
                merged.to_csv(output_file)
                logger.info(f"Saved: {output_file}")

                # 메타데이터 저장
                meta = {
                    'cancer_code': cancer_code,
                    'full_name': TCGA_CANCER_TYPES[cancer_code]['full_name'],
                    'n_samples': merged.shape[1],
                    'n_genes': merged.shape[0],
                    'files': [str(p) for p in downloaded]
                }

                meta_file = self.output_dir / cancer_code / "metadata.json"
                with open(meta_file, 'w') as f:
                    json.dump(meta, f, indent=2)

                return meta

        return {}

    def download_low_performance_cancers(self, max_samples: int = 200) -> Dict:
        """저성능 암종 일괄 다운로드"""
        low_perf = [c for c, info in TCGA_CANCER_TYPES.items() if info['priority'] == 1]

        results = {}
        for cancer in low_perf:
            result = self.download_cancer_data(cancer, max_samples)
            results[cancer] = result
            time.sleep(2)

        return results


def main():
    parser = argparse.ArgumentParser(description='TCGA Cancer Data Downloader')
    parser.add_argument('--cancer', type=str, help='Cancer code (e.g., PAAD)')
    parser.add_argument('--all-low-performance', action='store_true',
                       help='Download all low-performance cancers')
    parser.add_argument('--max-samples', type=int, default=200,
                       help='Maximum samples per cancer')
    parser.add_argument('--output-dir', type=str, default='data/tcga',
                       help='Output directory')
    parser.add_argument('--list', action='store_true', help='List cancer types')

    args = parser.parse_args()

    if args.list:
        print("\n=== TCGA Cancer Types ===\n")
        print(f"{'Code':<8} {'Name':<45} {'Priority'}")
        print("-" * 65)
        for code, info in sorted(TCGA_CANCER_TYPES.items(),
                                key=lambda x: (x[1]['priority'], x[0])):
            priority_str = {1: '🔴 High', 2: '🟡 Medium', 3: '🟢 Low'}[info['priority']]
            print(f"{code:<8} {info['full_name']:<45} {priority_str}")
        return

    downloader = TCGADownloader(output_dir=args.output_dir)

    if args.all_low_performance:
        results = downloader.download_low_performance_cancers(max_samples=args.max_samples)

        print("\n=== Download Summary ===")
        for cancer, result in results.items():
            print(f"{cancer}: {result.get('n_samples', 0)} samples")

    elif args.cancer:
        downloader.download_cancer_data(args.cancer.upper(), max_samples=args.max_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
