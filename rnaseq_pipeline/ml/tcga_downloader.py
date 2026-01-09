"""
TCGA Data Downloader
====================

GDC API를 사용하여 TCGA RNA-seq 데이터를 다운로드합니다.

Supported Projects:
- TCGA-PAAD: 췌장암 (Pancreatic Adenocarcinoma)
- TCGA-LUAD: 폐 선암 (Lung Adenocarcinoma)
- TCGA-BRCA: 유방암 (Breast Cancer)
- TCGA-COAD: 대장암 (Colon Adenocarcinoma)
"""

import os
import json
import gzip
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCGADownloader:
    """TCGA GDC API를 통한 RNA-seq 데이터 다운로드"""

    GDC_API_BASE = "https://api.gdc.cancer.gov"

    # TCGA 프로젝트 코드 매핑
    CANCER_TYPES = {
        "pancreatic": "TCGA-PAAD",
        "lung": "TCGA-LUAD",
        "breast": "TCGA-BRCA",
        "colon": "TCGA-COAD",
        "liver": "TCGA-LIHC",
        "stomach": "TCGA-STAD",
        "kidney": "TCGA-KIRC",
        "prostate": "TCGA-PRAD",
    }

    # 샘플 타입 코드
    SAMPLE_TYPES = {
        "01": "Primary Tumor",
        "02": "Recurrent Tumor",
        "03": "Primary Blood Derived Cancer",
        "06": "Metastatic",
        "11": "Solid Tissue Normal",
    }

    def __init__(self, output_dir: str = "data/tcga"):
        """
        Args:
            output_dir: 데이터 저장 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_project_code(self, cancer_type: str) -> str:
        """암 종류에서 TCGA 프로젝트 코드 반환"""
        cancer_type_lower = cancer_type.lower()

        # 직접 매핑 확인
        if cancer_type_lower in self.CANCER_TYPES:
            return self.CANCER_TYPES[cancer_type_lower]

        # TCGA-XXX 형식이면 그대로 반환
        if cancer_type.upper().startswith("TCGA-"):
            return cancer_type.upper()

        raise ValueError(f"Unknown cancer type: {cancer_type}. "
                        f"Available: {list(self.CANCER_TYPES.keys())}")

    def _query_files(self, project: str, data_type: str = "Gene Expression Quantification",
                    workflow: str = "STAR - Counts", limit: int = 1000) -> List[Dict]:
        """GDC API로 파일 목록 조회"""

        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
                {"op": "=", "content": {"field": "data_type", "value": data_type}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.case_id,cases.samples.sample_type,cases.samples.portions.analytes.aliquots.submitter_id",
            "format": "JSON",
            "size": limit,
        }

        response = requests.get(f"{self.GDC_API_BASE}/files", params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("data", {}).get("hits", [])

    def _download_file(self, file_id: str, output_path: Path) -> bool:
        """단일 파일 다운로드"""
        try:
            url = f"{self.GDC_API_BASE}/data/{file_id}"
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to download {file_id}: {e}")
            return False

    def _parse_counts_file(self, file_path: Path) -> pd.Series:
        """STAR counts 파일 파싱 (gene_id -> count)"""
        try:
            # .gz 파일 처리
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep='\t', comment='#', header=None,
                                    names=['gene_id', 'gene_name', 'gene_type',
                                          'unstranded', 'stranded_first', 'stranded_second',
                                          'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'],
                                    skiprows=6)
            else:
                df = pd.read_csv(file_path, sep='\t', comment='#', header=None,
                                names=['gene_id', 'gene_name', 'gene_type',
                                      'unstranded', 'stranded_first', 'stranded_second',
                                      'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'],
                                skiprows=6)

            # gene_id를 인덱스로, unstranded counts 사용
            counts = df.set_index('gene_id')['unstranded']
            return counts

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return pd.Series()

    def _get_sample_type(self, barcode: str) -> str:
        """TCGA barcode에서 샘플 타입 추출 (예: TCGA-XX-XXXX-01A -> "01")"""
        try:
            parts = barcode.split('-')
            if len(parts) >= 4:
                sample_code = parts[3][:2]
                return sample_code
        except:
            pass
        return "unknown"

    def download_project(self, cancer_type: str,
                        max_samples: int = 200,
                        include_normal: bool = True,
                        n_workers: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        TCGA 프로젝트의 RNA-seq 데이터 다운로드

        Args:
            cancer_type: 암 종류 (예: "pancreatic", "lung")
            max_samples: 최대 샘플 수 (0 = 전체 다운로드)
            include_normal: 정상 샘플 포함 여부
            n_workers: 병렬 다운로드 워커 수

        Returns:
            (count_matrix, metadata) 튜플
        """
        project = self.get_project_code(cancer_type)
        logger.info(f"Downloading {project} data...")

        # 프로젝트별 디렉토리 생성
        project_dir = self.output_dir / project
        raw_dir = project_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 파일 목록 조회 (max_samples=0이면 전체 다운로드, limit=5000)
        query_limit = 5000 if max_samples == 0 else max_samples * 2
        logger.info(f"Querying GDC API for available files (limit={query_limit})...")
        files = self._query_files(project, limit=query_limit)

        if not files:
            raise ValueError(f"No files found for {project}")

        logger.info(f"Found {len(files)} files")

        # 다운로드할 파일 정보 수집
        download_tasks = []
        sample_info = []

        for file_info in files:
            file_id = file_info['file_id']
            file_name = file_info['file_name']

            # 케이스 및 샘플 정보 추출
            cases = file_info.get('cases', [])
            if not cases:
                continue

            case = cases[0]
            case_id = case.get('case_id', '')

            # 샘플 타입 확인
            samples = case.get('samples', [])
            if samples:
                sample_type = samples[0].get('sample_type', '')

                # 정상 샘플 제외 옵션
                if not include_normal and 'Normal' in sample_type:
                    continue

                # aliquot에서 barcode 추출
                portions = samples[0].get('portions', [])
                if portions:
                    analytes = portions[0].get('analytes', [])
                    if analytes:
                        aliquots = analytes[0].get('aliquots', [])
                        if aliquots:
                            barcode = aliquots[0].get('submitter_id', '')
                        else:
                            barcode = case_id
                    else:
                        barcode = case_id
                else:
                    barcode = case_id
            else:
                sample_type = 'Unknown'
                barcode = case_id

            output_path = raw_dir / file_name

            download_tasks.append({
                'file_id': file_id,
                'file_name': file_name,
                'output_path': output_path,
            })

            sample_info.append({
                'file_name': file_name,
                'case_id': case_id,
                'barcode': barcode,
                'sample_type': sample_type,
                'label': 0 if 'Normal' in sample_type else 1,  # 0: Normal, 1: Tumor
            })

            # max_samples=0이면 제한 없음
            if max_samples > 0 and len(download_tasks) >= max_samples:
                break

        logger.info(f"Downloading {len(download_tasks)} files...")

        # 병렬 다운로드
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._download_file, task['file_id'], task['output_path']): task
                for task in download_tasks
                if not task['output_path'].exists()  # 이미 존재하면 스킵
            }

            # 이미 존재하는 파일 추가
            for task in download_tasks:
                if task['output_path'].exists():
                    downloaded_files.append(task['file_name'])

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                task = futures[future]
                try:
                    if future.result():
                        downloaded_files.append(task['file_name'])
                except Exception as e:
                    logger.error(f"Error: {e}")

        logger.info(f"Downloaded {len(downloaded_files)} files")

        # 메타데이터 저장
        metadata = pd.DataFrame(sample_info)
        metadata = metadata[metadata['file_name'].isin(downloaded_files)]
        metadata.to_csv(project_dir / "metadata.csv", index=False)

        # Count matrix 생성
        logger.info("Building count matrix...")
        count_matrix = self._build_count_matrix(raw_dir, metadata)
        count_matrix.to_csv(project_dir / "count_matrix.csv")

        logger.info(f"Count matrix shape: {count_matrix.shape}")
        logger.info(f"Samples - Tumor: {(metadata['label'] == 1).sum()}, Normal: {(metadata['label'] == 0).sum()}")

        return count_matrix, metadata

    def _build_count_matrix(self, raw_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
        """개별 count 파일들을 하나의 matrix로 합침"""

        count_dict = {}

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Parsing files"):
            file_path = raw_dir / row['file_name']
            if file_path.exists():
                counts = self._parse_counts_file(file_path)
                if not counts.empty:
                    count_dict[row['barcode']] = counts

        if not count_dict:
            raise ValueError("No valid count files found")

        # DataFrame으로 변환 (genes x samples)
        count_matrix = pd.DataFrame(count_dict)

        # NaN을 0으로
        count_matrix = count_matrix.fillna(0).astype(int)

        return count_matrix

    def load_project(self, cancer_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """이미 다운로드된 프로젝트 데이터 로드"""
        project = self.get_project_code(cancer_type)
        project_dir = self.output_dir / project

        if not project_dir.exists():
            raise FileNotFoundError(f"Project data not found: {project_dir}")

        count_matrix = pd.read_csv(project_dir / "count_matrix.csv", index_col=0)
        metadata = pd.read_csv(project_dir / "metadata.csv")

        return count_matrix, metadata


def main():
    """CLI 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Download TCGA RNA-seq data")
    parser.add_argument("--cancer", "-c", type=str, default="pancreatic",
                       help="Cancer type (pancreatic, lung, breast, etc.)")
    parser.add_argument("--max-samples", "-n", type=int, default=200,
                       help="Maximum number of samples")
    parser.add_argument("--output", "-o", type=str, default="data/tcga",
                       help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of parallel workers")

    args = parser.parse_args()

    downloader = TCGADownloader(output_dir=args.output)

    print(f"\n{'='*60}")
    print(f"  TCGA Data Downloader")
    print(f"{'='*60}")
    print(f"  Cancer Type: {args.cancer}")
    print(f"  Max Samples: {args.max_samples}")
    print(f"  Output Dir:  {args.output}")
    print(f"{'='*60}\n")

    count_matrix, metadata = downloader.download_project(
        cancer_type=args.cancer,
        max_samples=args.max_samples,
        n_workers=args.workers,
    )

    print(f"\n{'='*60}")
    print(f"  Download Complete!")
    print(f"{'='*60}")
    print(f"  Count Matrix: {count_matrix.shape[0]} genes x {count_matrix.shape[1]} samples")
    print(f"  Tumor samples:  {(metadata['label'] == 1).sum()}")
    print(f"  Normal samples: {(metadata['label'] == 0).sum()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
