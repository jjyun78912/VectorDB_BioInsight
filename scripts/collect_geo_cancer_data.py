#!/usr/bin/env python3
"""
GEO Cancer RNA-seq Data Collector
=================================

저성능 암종(PAAD, PRAD, BLCA, OV 등)의 GEO 데이터를 자동 수집합니다.

사용법:
    python scripts/collect_geo_cancer_data.py --cancer PAAD --max-series 5
    python scripts/collect_geo_cancer_data.py --all-low-performance
    python scripts/collect_geo_cancer_data.py --list
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
from io import StringIO
import time
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GEO 검색 쿼리 (암종별)
CANCER_SEARCH_QUERIES = {
    # 저성능 암종 (우선순위 높음)
    'PAAD': {
        'query': '(pancreatic cancer[Title] OR pancreatic adenocarcinoma[Title] OR PDAC[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'PAAD',
        'full_name': 'Pancreatic Adenocarcinoma',
        'priority': 1
    },
    'PRAD': {
        'query': '(prostate cancer[Title] OR prostate adenocarcinoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'PRAD',
        'full_name': 'Prostate Adenocarcinoma',
        'priority': 1
    },
    'BLCA': {
        'query': '(bladder cancer[Title] OR urothelial carcinoma[Title] OR bladder urothelial[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'BLCA',
        'full_name': 'Bladder Urothelial Carcinoma',
        'priority': 1
    },
    'OV': {
        'query': '(ovarian cancer[Title] OR ovarian carcinoma[Title] OR ovarian serous[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'OV',
        'full_name': 'Ovarian Serous Cystadenocarcinoma',
        'priority': 1
    },
    # 중간 성능 암종
    'LUSC': {
        'query': '(lung squamous[Title] OR squamous cell lung[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'LUSC',
        'full_name': 'Lung Squamous Cell Carcinoma',
        'priority': 2
    },
    'KIRC': {
        'query': '(renal cell carcinoma[Title] OR kidney cancer[Title] OR clear cell renal[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'KIRC',
        'full_name': 'Kidney Renal Clear Cell Carcinoma',
        'priority': 2
    },
    'LIHC': {
        'query': '(hepatocellular carcinoma[Title] OR liver cancer[Title] OR HCC[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'LIHC',
        'full_name': 'Liver Hepatocellular Carcinoma',
        'priority': 2
    },
    'LUAD': {
        'query': '(lung adenocarcinoma[Title] OR lung cancer adenocarcinoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'LUAD',
        'full_name': 'Lung Adenocarcinoma',
        'priority': 2
    },
    # 고성능 암종 (참고용)
    'BRCA': {
        'query': '(breast cancer[Title] OR breast carcinoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'BRCA',
        'full_name': 'Breast Invasive Carcinoma',
        'priority': 3
    },
    'COAD': {
        'query': '(colon cancer[Title] OR colorectal cancer[Title] OR colon adenocarcinoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'COAD',
        'full_name': 'Colon Adenocarcinoma',
        'priority': 3
    },
    'HNSC': {
        'query': '(head and neck cancer[Title] OR head neck squamous[Title] OR HNSCC[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'HNSC',
        'full_name': 'Head and Neck Squamous Cell Carcinoma',
        'priority': 2
    },
    'STAD': {
        'query': '(gastric cancer[Title] OR stomach cancer[Title] OR gastric adenocarcinoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'STAD',
        'full_name': 'Stomach Adenocarcinoma',
        'priority': 2
    },
    'UCEC': {
        'query': '(endometrial cancer[Title] OR uterine cancer[Title] OR endometrial carcinoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'UCEC',
        'full_name': 'Uterine Corpus Endometrial Carcinoma',
        'priority': 2
    },
    'THCA': {
        'query': '(thyroid cancer[Title] OR thyroid carcinoma[Title] OR papillary thyroid[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'THCA',
        'full_name': 'Thyroid Carcinoma',
        'priority': 3
    },
    'SKCM': {
        'query': '(melanoma[Title] OR skin cutaneous melanoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'SKCM',
        'full_name': 'Skin Cutaneous Melanoma',
        'priority': 2
    },
    'GBM': {
        'query': '(glioblastoma[Title] OR GBM[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'GBM',
        'full_name': 'Glioblastoma Multiforme',
        'priority': 2
    },
    'LGG': {
        'query': '(low grade glioma[Title] OR diffuse glioma[Title] OR astrocytoma[Title]) AND RNA-seq[Title/Abstract] AND Homo sapiens[Organism]',
        'tcga_code': 'LGG',
        'full_name': 'Brain Lower Grade Glioma',
        'priority': 2
    }
}


class GEODataCollector:
    """GEO 데이터베이스에서 암종별 RNA-seq 데이터 수집"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

    def __init__(self, output_dir: str = "data/geo_cancer"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get('NCBI_API_KEY', '')

    def search_geo_series(self, query: str, max_results: int = 20) -> List[str]:
        """GEO에서 시리즈 검색"""
        params = {
            'db': 'gds',
            'term': query + ' AND GSE[ETYP]',
            'retmax': max_results,
            'retmode': 'json',
            'usehistory': 'y'
        }
        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
            response.raise_for_status()
            data = response.json()

            id_list = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(id_list)} GEO entries")

            # GDS ID를 GSE ID로 변환
            gse_ids = []
            for gds_id in id_list:
                gse = self._get_gse_from_gds(gds_id)
                if gse:
                    gse_ids.append(gse)
                time.sleep(0.35)  # Rate limiting

            return list(set(gse_ids))  # 중복 제거

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _get_gse_from_gds(self, gds_id: str) -> Optional[str]:
        """GDS ID에서 GSE ID 추출"""
        params = {
            'db': 'gds',
            'id': gds_id,
            'retmode': 'xml'
        }
        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = requests.get(f"{self.BASE_URL}/esummary.fcgi", params=params)
            response.raise_for_status()

            # XML 파싱
            root = ET.fromstring(response.content)
            for item in root.findall('.//Item[@Name="GSE"]'):
                return f"GSE{item.text}"

            # Accession에서 직접 추출 시도
            for item in root.findall('.//Item[@Name="Accession"]'):
                if item.text and item.text.startswith('GSE'):
                    return item.text

        except Exception as e:
            logger.debug(f"Could not get GSE from GDS {gds_id}: {e}")

        return None

    def get_series_info(self, gse_id: str) -> Optional[Dict]:
        """GSE 시리즈 정보 조회"""
        # GEO Query로 직접 검색
        params = {
            'db': 'gds',
            'term': f'{gse_id}[Accession]',
            'retmax': 1,
            'retmode': 'json'
        }
        if self.api_key:
            params['api_key'] = self.api_key

        try:
            # 먼저 검색
            response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])

            if not id_list:
                return None

            # Summary 조회
            sum_params = {
                'db': 'gds',
                'id': ','.join(id_list),
                'retmode': 'json'
            }
            if self.api_key:
                sum_params['api_key'] = self.api_key

            response = requests.get(f"{self.BASE_URL}/esummary.fcgi", params=sum_params)
            summary = response.json()

            result = summary.get('result', {})
            for uid in id_list:
                if uid in result:
                    info = result[uid]
                    return {
                        'gse_id': gse_id,
                        'title': info.get('title', ''),
                        'summary': info.get('summary', ''),
                        'n_samples': info.get('n_samples', 0),
                        'gpl': info.get('gpl', ''),
                        'taxon': info.get('taxon', ''),
                        'gdstype': info.get('gdstype', ''),
                        'pdat': info.get('pdat', '')
                    }

        except Exception as e:
            logger.error(f"Error getting series info for {gse_id}: {e}")

        return None

    def download_expression_matrix(self, gse_id: str, cancer_type: str) -> Optional[Path]:
        """발현 행렬 다운로드"""
        output_path = self.output_dir / cancer_type / gse_id
        output_path.mkdir(parents=True, exist_ok=True)

        # 이미 다운로드된 경우 스킵
        matrix_file = output_path / "expression_matrix.csv"
        if matrix_file.exists():
            logger.info(f"Already downloaded: {gse_id}")
            return matrix_file

        # GEO Series Matrix 파일 다운로드 시도
        gse_prefix = gse_id[:len(gse_id)-3] + "nnn"
        matrix_url = f"{self.GEO_FTP}/{gse_prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"

        try:
            logger.info(f"Downloading {gse_id} matrix...")
            response = requests.get(matrix_url, timeout=60)

            if response.status_code == 200:
                # Gzip 압축 해제 및 파싱
                content = gzip.decompress(response.content).decode('utf-8')

                # 메타데이터와 발현 데이터 분리
                lines = content.split('\n')
                data_start = 0
                metadata = {}

                for i, line in enumerate(lines):
                    if line.startswith('!Sample_'):
                        key = line.split('\t')[0].replace('!', '')
                        values = line.strip().split('\t')[1:]
                        metadata[key] = values
                    elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                        data_start = i
                        break

                # 발현 데이터 추출
                data_lines = '\n'.join(lines[data_start:])
                df = pd.read_csv(StringIO(data_lines), sep='\t', index_col=0)

                # 빈 행 제거
                df = df.dropna(how='all')

                if len(df) > 0:
                    df.to_csv(matrix_file)

                    # 메타데이터 저장
                    meta_df = pd.DataFrame(metadata)
                    meta_df.to_csv(output_path / "metadata.csv", index=False)

                    logger.info(f"Saved {gse_id}: {df.shape[0]} genes x {df.shape[1]} samples")
                    return matrix_file
                else:
                    logger.warning(f"No expression data in {gse_id}")

            else:
                logger.warning(f"Matrix not available for {gse_id} (HTTP {response.status_code})")

        except Exception as e:
            logger.error(f"Error downloading {gse_id}: {e}")

        return None

    def collect_cancer_data(self, cancer_code: str, max_series: int = 5,
                           min_samples: int = 10) -> Dict:
        """특정 암종의 GEO 데이터 수집"""
        if cancer_code not in CANCER_SEARCH_QUERIES:
            logger.error(f"Unknown cancer code: {cancer_code}")
            return {}

        cancer_info = CANCER_SEARCH_QUERIES[cancer_code]
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting {cancer_info['full_name']} ({cancer_code}) data...")
        logger.info(f"{'='*60}")

        # GEO 검색
        gse_ids = self.search_geo_series(cancer_info['query'], max_results=max_series * 3)
        logger.info(f"Found {len(gse_ids)} candidate series")

        collected = {
            'cancer_code': cancer_code,
            'full_name': cancer_info['full_name'],
            'series': [],
            'total_samples': 0
        }

        for gse_id in gse_ids[:max_series * 2]:  # 후보 더 많이 확인
            if len(collected['series']) >= max_series:
                break

            time.sleep(0.5)  # Rate limiting

            # 시리즈 정보 확인
            info = self.get_series_info(gse_id)
            if not info:
                continue

            # 필터링: 샘플 수, 종
            if info.get('n_samples', 0) < min_samples:
                logger.info(f"Skipping {gse_id}: only {info.get('n_samples', 0)} samples")
                continue

            if 'Homo sapiens' not in str(info.get('taxon', '')):
                logger.info(f"Skipping {gse_id}: not human")
                continue

            # 다운로드
            matrix_path = self.download_expression_matrix(gse_id, cancer_code)

            if matrix_path and matrix_path.exists():
                df = pd.read_csv(matrix_path, index_col=0)
                n_samples = df.shape[1]

                collected['series'].append({
                    'gse_id': gse_id,
                    'title': info.get('title', ''),
                    'n_samples': n_samples,
                    'n_genes': df.shape[0],
                    'path': str(matrix_path)
                })
                collected['total_samples'] += n_samples

                logger.info(f"✓ Collected {gse_id}: {n_samples} samples")

            time.sleep(1)  # Rate limiting

        # 결과 저장
        summary_path = self.output_dir / cancer_code / "collection_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(collected, f, indent=2)

        logger.info(f"\nCollected {len(collected['series'])} series, "
                   f"{collected['total_samples']} total samples for {cancer_code}")

        return collected

    def collect_low_performance_cancers(self, max_series_per_cancer: int = 5) -> Dict:
        """저성능 암종 데이터 일괄 수집"""
        low_perf_cancers = [c for c, info in CANCER_SEARCH_QUERIES.items()
                           if info['priority'] == 1]

        all_results = {}
        for cancer in low_perf_cancers:
            result = self.collect_cancer_data(cancer, max_series=max_series_per_cancer)
            all_results[cancer] = result
            time.sleep(2)  # 암종 간 rate limiting

        return all_results

    def merge_collected_data(self, cancer_codes: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """수집된 데이터를 하나의 행렬로 병합"""
        if cancer_codes is None:
            cancer_codes = list(CANCER_SEARCH_QUERIES.keys())

        all_data = []
        all_labels = []

        for cancer in cancer_codes:
            cancer_dir = self.output_dir / cancer
            if not cancer_dir.exists():
                continue

            summary_file = cancer_dir / "collection_summary.json"
            if not summary_file.exists():
                continue

            with open(summary_file) as f:
                summary = json.load(f)

            for series in summary.get('series', []):
                matrix_path = Path(series['path'])
                if matrix_path.exists():
                    df = pd.read_csv(matrix_path, index_col=0)

                    # 각 샘플에 라벨 추가
                    for col in df.columns:
                        all_data.append(df[col])
                        all_labels.append({
                            'sample_id': col,
                            'cancer_type': cancer,
                            'gse_id': series['gse_id']
                        })

        if not all_data:
            logger.warning("No data collected")
            return pd.DataFrame(), pd.DataFrame()

        # 병합
        merged = pd.concat(all_data, axis=1)
        labels_df = pd.DataFrame(all_labels)

        logger.info(f"Merged data: {merged.shape[0]} genes x {merged.shape[1]} samples")
        logger.info(f"Cancer distribution:\n{labels_df['cancer_type'].value_counts()}")

        return merged, labels_df


def main():
    parser = argparse.ArgumentParser(description='GEO Cancer RNA-seq Data Collector')
    parser.add_argument('--cancer', type=str, help='Cancer code (e.g., PAAD, PRAD)')
    parser.add_argument('--all-low-performance', action='store_true',
                       help='Collect all low-performance cancers')
    parser.add_argument('--max-series', type=int, default=5,
                       help='Maximum series per cancer')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples per series')
    parser.add_argument('--output-dir', type=str, default='data/geo_cancer',
                       help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available cancer codes')
    parser.add_argument('--merge', action='store_true', help='Merge collected data')

    args = parser.parse_args()

    if args.list:
        print("\n=== Available Cancer Codes ===\n")
        print(f"{'Code':<8} {'Name':<45} {'Priority'}")
        print("-" * 65)
        for code, info in sorted(CANCER_SEARCH_QUERIES.items(),
                                 key=lambda x: (x[1]['priority'], x[0])):
            priority_str = {1: '🔴 High', 2: '🟡 Medium', 3: '🟢 Low'}[info['priority']]
            print(f"{code:<8} {info['full_name']:<45} {priority_str}")
        print("\nPriority 1 (High): Low external validation accuracy - needs more data")
        print("Priority 2 (Medium): Moderate accuracy - could benefit from more data")
        print("Priority 3 (Low): Good accuracy - sufficient data")
        return

    collector = GEODataCollector(output_dir=args.output_dir)

    if args.merge:
        merged, labels = collector.merge_collected_data()
        if len(merged) > 0:
            merged.to_csv(Path(args.output_dir) / "merged_expression.csv")
            labels.to_csv(Path(args.output_dir) / "merged_labels.csv", index=False)
            print(f"Saved merged data to {args.output_dir}/")
        return

    if args.all_low_performance:
        results = collector.collect_low_performance_cancers(max_series_per_cancer=args.max_series)

        print("\n=== Collection Summary ===")
        for cancer, result in results.items():
            print(f"{cancer}: {result.get('total_samples', 0)} samples from "
                  f"{len(result.get('series', []))} series")

    elif args.cancer:
        collector.collect_cancer_data(
            args.cancer,
            max_series=args.max_series,
            min_samples=args.min_samples
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
