"""
Pan-Cancer TCGA Downloader
===========================

TCGA 33개 암종 전체 데이터를 다운로드하여 Multi-class 분류용 데이터셋 구축

Supported Projects (33 cancer types):
- TCGA-ACC, TCGA-BLCA, TCGA-BRCA, TCGA-CESC, TCGA-CHOL, TCGA-COAD,
- TCGA-DLBC, TCGA-ESCA, TCGA-GBM, TCGA-HNSC, TCGA-KICH, TCGA-KIRC,
- TCGA-KIRP, TCGA-LAML, TCGA-LGG, TCGA-LIHC, TCGA-LUAD, TCGA-LUSC,
- TCGA-MESO, TCGA-OV, TCGA-PAAD, TCGA-PCPG, TCGA-PRAD, TCGA-READ,
- TCGA-SARC, TCGA-SKCM, TCGA-STAD, TCGA-TGCT, TCGA-THCA, TCGA-THYM,
- TCGA-UCEC, TCGA-UCS, TCGA-UVM
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
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PanCancerDownloader:
    """TCGA 전체 암종 다운로드 및 Pan-Cancer 데이터셋 구축"""

    GDC_API_BASE = "https://api.gdc.cancer.gov"

    # TCGA 33개 암종 (전체)
    ALL_CANCER_TYPES = {
        # Adrenal & Endocrine
        "ACC": {"name": "Adrenocortical Carcinoma", "korean": "부신피질암"},
        "PCPG": {"name": "Pheochromocytoma & Paraganglioma", "korean": "갈색세포종"},
        "THCA": {"name": "Thyroid Carcinoma", "korean": "갑상선암"},

        # Brain & CNS
        "GBM": {"name": "Glioblastoma Multiforme", "korean": "교모세포종"},
        "LGG": {"name": "Lower Grade Glioma", "korean": "저등급 신경교종"},

        # Breast
        "BRCA": {"name": "Breast Invasive Carcinoma", "korean": "유방암"},

        # GI Tract
        "COAD": {"name": "Colon Adenocarcinoma", "korean": "대장암"},
        "READ": {"name": "Rectum Adenocarcinoma", "korean": "직장암"},
        "ESCA": {"name": "Esophageal Carcinoma", "korean": "식도암"},
        "STAD": {"name": "Stomach Adenocarcinoma", "korean": "위암"},
        "CHOL": {"name": "Cholangiocarcinoma", "korean": "담관암"},
        "LIHC": {"name": "Liver Hepatocellular Carcinoma", "korean": "간암"},
        "PAAD": {"name": "Pancreatic Adenocarcinoma", "korean": "췌장암"},

        # GU Tract
        "BLCA": {"name": "Bladder Urothelial Carcinoma", "korean": "방광암"},
        "KICH": {"name": "Kidney Chromophobe", "korean": "신장 혐색소세포암"},
        "KIRC": {"name": "Kidney Clear Cell Carcinoma", "korean": "신장 투명세포암"},
        "KIRP": {"name": "Kidney Papillary Cell Carcinoma", "korean": "신장 유두세포암"},
        "PRAD": {"name": "Prostate Adenocarcinoma", "korean": "전립선암"},
        "TGCT": {"name": "Testicular Germ Cell Tumor", "korean": "고환암"},

        # Gynecologic
        "CESC": {"name": "Cervical Cancer", "korean": "자궁경부암"},
        "OV": {"name": "Ovarian Serous Cystadenocarcinoma", "korean": "난소암"},
        "UCEC": {"name": "Uterine Corpus Endometrial Carcinoma", "korean": "자궁내막암"},
        "UCS": {"name": "Uterine Carcinosarcoma", "korean": "자궁 암육종"},

        # Head & Neck
        "HNSC": {"name": "Head & Neck Squamous Cell Carcinoma", "korean": "두경부암"},

        # Hematologic
        "DLBC": {"name": "Diffuse Large B-cell Lymphoma", "korean": "미만성 대B세포 림프종"},
        "LAML": {"name": "Acute Myeloid Leukemia", "korean": "급성 골수성 백혈병"},
        "THYM": {"name": "Thymoma", "korean": "흉선종"},

        # Lung
        "LUAD": {"name": "Lung Adenocarcinoma", "korean": "폐 선암"},
        "LUSC": {"name": "Lung Squamous Cell Carcinoma", "korean": "폐 편평세포암"},
        "MESO": {"name": "Mesothelioma", "korean": "중피종"},

        # Skin
        "SKCM": {"name": "Skin Cutaneous Melanoma", "korean": "피부 흑색종"},
        "UVM": {"name": "Uveal Melanoma", "korean": "포도막 흑색종"},

        # Soft Tissue
        "SARC": {"name": "Sarcoma", "korean": "육종"},
    }

    # 주요 암종 (빈도순, 학습에 우선 사용)
    PRIMARY_CANCER_TYPES = [
        "BRCA",  # 유방암 (~1100 samples)
        "LUAD",  # 폐 선암 (~500)
        "LUSC",  # 폐 편평세포암 (~500)
        "KIRC",  # 신장암 (~530)
        "THCA",  # 갑상선암 (~500)
        "PRAD",  # 전립선암 (~500)
        "LIHC",  # 간암 (~370)
        "COAD",  # 대장암 (~450)
        "STAD",  # 위암 (~415)
        "BLCA",  # 방광암 (~400)
        "HNSC",  # 두경부암 (~520)
        "OV",    # 난소암 (~370)
        "UCEC",  # 자궁내막암 (~550)
        "GBM",   # 교모세포종 (~150)
        "LGG",   # 저등급 신경교종 (~500)
        "PAAD",  # 췌장암 (~180)
        "SKCM",  # 흑색종 (~470)
    ]

    def __init__(self, output_dir: str = "data/tcga/pancancer"):
        """
        Args:
            output_dir: 데이터 저장 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _query_files(self, project: str, limit: int = 1000) -> List[Dict]:
        """GDC API로 파일 목록 조회"""
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.case_id,cases.samples.sample_type,cases.samples.portions.analytes.aliquots.submitter_id",
            "format": "JSON",
            "size": limit,
        }

        response = requests.get(f"{self.GDC_API_BASE}/files", params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        return data.get("data", {}).get("hits", [])

    def _download_file(self, file_id: str, output_path: Path) -> bool:
        """단일 파일 다운로드"""
        try:
            if output_path.exists():
                return True

            url = f"{self.GDC_API_BASE}/data/{file_id}"
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to download {file_id}: {e}")
            return False

    def _parse_counts_file(self, file_path: Path) -> pd.Series:
        """STAR counts 파일 파싱"""
        try:
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

            counts = df.set_index('gene_id')['unstranded']
            return counts

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return pd.Series()

    def download_cancer_type(self, cancer_code: str,
                            max_samples: int = 100,
                            n_workers: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        특정 암종 다운로드

        Args:
            cancer_code: TCGA 암종 코드 (예: "BRCA", "LUAD")
            max_samples: 최대 샘플 수 (0 = 전체)
            n_workers: 병렬 워커 수

        Returns:
            (count_matrix, metadata)
        """
        project = f"TCGA-{cancer_code}"
        cancer_info = self.ALL_CANCER_TYPES.get(cancer_code, {"name": cancer_code, "korean": cancer_code})

        logger.info(f"Downloading {project} ({cancer_info['korean']})...")

        # 디렉토리 생성
        cancer_dir = self.output_dir / cancer_code
        raw_dir = cancer_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 파일 목록 조회
        query_limit = 1000 if max_samples == 0 else max_samples * 2
        files = self._query_files(project, limit=query_limit)

        if not files:
            logger.warning(f"No files found for {project}")
            return pd.DataFrame(), pd.DataFrame()

        # 다운로드 태스크 수집
        download_tasks = []
        sample_info = []

        for file_info in files:
            file_id = file_info['file_id']
            file_name = file_info['file_name']

            cases = file_info.get('cases', [])
            if not cases:
                continue

            case = cases[0]
            case_id = case.get('case_id', '')
            samples = case.get('samples', [])

            if samples:
                sample_type = samples[0].get('sample_type', '')

                # barcode 추출
                portions = samples[0].get('portions', [])
                barcode = case_id
                if portions:
                    analytes = portions[0].get('analytes', [])
                    if analytes:
                        aliquots = analytes[0].get('aliquots', [])
                        if aliquots:
                            barcode = aliquots[0].get('submitter_id', case_id)
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
                'cancer_type': cancer_code,
                'cancer_name': cancer_info['name'],
                'cancer_korean': cancer_info['korean'],
                'is_tumor': 0 if 'Normal' in sample_type else 1,
            })

            if max_samples > 0 and len(download_tasks) >= max_samples:
                break

        logger.info(f"Downloading {len(download_tasks)} files for {cancer_code}...")

        # 병렬 다운로드
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._download_file, task['file_id'], task['output_path']): task
                for task in download_tasks
                if not task['output_path'].exists()
            }

            for task in download_tasks:
                if task['output_path'].exists():
                    downloaded_files.append(task['file_name'])

            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Downloading {cancer_code}"):
                task = futures[future]
                try:
                    if future.result():
                        downloaded_files.append(task['file_name'])
                except Exception as e:
                    logger.error(f"Error: {e}")

        # 메타데이터
        metadata = pd.DataFrame(sample_info)
        metadata = metadata[metadata['file_name'].isin(downloaded_files)]
        metadata.to_csv(cancer_dir / "metadata.csv", index=False)

        # Count matrix 생성
        logger.info(f"Building count matrix for {cancer_code}...")
        count_matrix = self._build_count_matrix(raw_dir, metadata)

        if not count_matrix.empty:
            count_matrix.to_csv(cancer_dir / "count_matrix.csv")

        logger.info(f"{cancer_code}: {count_matrix.shape[1]} samples "
                   f"(Tumor: {(metadata['is_tumor'] == 1).sum()}, "
                   f"Normal: {(metadata['is_tumor'] == 0).sum()})")

        return count_matrix, metadata

    def _build_count_matrix(self, raw_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
        """개별 count 파일들을 하나의 matrix로 합침"""
        count_dict = {}

        for _, row in tqdm(metadata.iterrows(), total=len(metadata),
                          desc="Parsing files", leave=False):
            file_path = raw_dir / row['file_name']
            if file_path.exists():
                counts = self._parse_counts_file(file_path)
                if not counts.empty:
                    count_dict[row['barcode']] = counts

        if not count_dict:
            return pd.DataFrame()

        count_matrix = pd.DataFrame(count_dict)
        count_matrix = count_matrix.fillna(0).astype(int)

        return count_matrix

    def download_pancancer(self,
                          cancer_types: Optional[List[str]] = None,
                          samples_per_cancer: int = 100,
                          n_workers: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pan-Cancer 데이터셋 다운로드 (여러 암종 통합)

        Args:
            cancer_types: 다운로드할 암종 목록 (None이면 PRIMARY_CANCER_TYPES 사용)
            samples_per_cancer: 암종당 최대 샘플 수
            n_workers: 병렬 워커 수

        Returns:
            (combined_count_matrix, combined_metadata)
        """
        if cancer_types is None:
            cancer_types = self.PRIMARY_CANCER_TYPES

        logger.info(f"\n{'='*60}")
        logger.info(f"  Pan-Cancer Dataset Download")
        logger.info(f"  Cancer types: {len(cancer_types)}")
        logger.info(f"  Samples per type: {samples_per_cancer}")
        logger.info(f"{'='*60}\n")

        all_matrices = []
        all_metadata = []

        for i, cancer_code in enumerate(cancer_types, 1):
            logger.info(f"\n[{i}/{len(cancer_types)}] Processing {cancer_code}...")

            try:
                count_matrix, metadata = self.download_cancer_type(
                    cancer_code,
                    max_samples=samples_per_cancer,
                    n_workers=n_workers
                )

                if not count_matrix.empty:
                    all_matrices.append(count_matrix)
                    all_metadata.append(metadata)

            except Exception as e:
                logger.error(f"Failed to download {cancer_code}: {e}")
                continue

        if not all_matrices:
            raise ValueError("No data downloaded")

        # 통합
        logger.info("\nCombining all cancer types...")

        # 공통 유전자만 사용
        common_genes = set(all_matrices[0].index)
        for matrix in all_matrices[1:]:
            common_genes &= set(matrix.index)
        common_genes = sorted(list(common_genes))

        logger.info(f"Common genes across all cancer types: {len(common_genes)}")

        # 통합 matrix
        combined_matrices = [m.loc[common_genes] for m in all_matrices]
        combined_matrix = pd.concat(combined_matrices, axis=1)

        # 통합 metadata
        combined_metadata = pd.concat(all_metadata, ignore_index=True)

        # 레이블 인코딩 (암종별 숫자)
        cancer_labels = {cancer: i for i, cancer in enumerate(cancer_types)}
        combined_metadata['cancer_label'] = combined_metadata['cancer_type'].map(cancer_labels)

        # 저장
        combined_matrix.to_csv(self.output_dir / "pancancer_counts.csv")
        combined_metadata.to_csv(self.output_dir / "pancancer_metadata.csv", index=False)

        # 레이블 매핑 저장
        label_mapping = {
            'cancer_to_label': cancer_labels,
            'label_to_cancer': {v: k for k, v in cancer_labels.items()},
            'cancer_info': {k: self.ALL_CANCER_TYPES.get(k, {}) for k in cancer_types},
            'n_classes': len(cancer_types),
            'download_date': datetime.now().isoformat(),
            'samples_per_cancer': samples_per_cancer,
        }

        with open(self.output_dir / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"  Pan-Cancer Download Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"  Total samples: {combined_matrix.shape[1]}")
        logger.info(f"  Total genes: {combined_matrix.shape[0]}")
        logger.info(f"  Cancer types: {len(cancer_types)}")
        logger.info(f"  Saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        # 암종별 샘플 수 출력
        print("\nSamples per cancer type:")
        print("-" * 40)
        for cancer in cancer_types:
            count = (combined_metadata['cancer_type'] == cancer).sum()
            info = self.ALL_CANCER_TYPES.get(cancer, {})
            korean = info.get('korean', cancer)
            print(f"  {cancer:6s} ({korean}): {count:4d}")

        return combined_matrix, combined_metadata

    def load_pancancer(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """저장된 Pan-Cancer 데이터 로드"""
        count_matrix = pd.read_csv(self.output_dir / "pancancer_counts.csv", index_col=0)
        metadata = pd.read_csv(self.output_dir / "pancancer_metadata.csv")

        with open(self.output_dir / "label_mapping.json", 'r') as f:
            label_mapping = json.load(f)

        return count_matrix, metadata, label_mapping


def main():
    """CLI 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Download Pan-Cancer TCGA data")
    parser.add_argument("--cancers", "-c", type=str, nargs='+', default=None,
                       help="Cancer types to download (default: PRIMARY_CANCER_TYPES)")
    parser.add_argument("--samples", "-n", type=int, default=100,
                       help="Samples per cancer type")
    parser.add_argument("--output", "-o", type=str, default="data/tcga/pancancer",
                       help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--list", action="store_true",
                       help="List all available cancer types")

    args = parser.parse_args()

    downloader = PanCancerDownloader(output_dir=args.output)

    if args.list:
        print("\nAvailable TCGA Cancer Types (33):")
        print("-" * 60)
        for code, info in sorted(downloader.ALL_CANCER_TYPES.items()):
            print(f"  {code:6s}: {info['name']:45s} ({info['korean']})")
        return

    count_matrix, metadata = downloader.download_pancancer(
        cancer_types=args.cancers,
        samples_per_cancer=args.samples,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
