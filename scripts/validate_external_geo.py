#!/usr/bin/env python3
"""
External GEO Dataset Validation
================================

Pan-Cancer 모델을 외부 GEO 데이터셋으로 검증

Usage:
    python scripts/validate_external_geo.py --gse GSE96058 --cancer BRCA
    python scripts/validate_external_geo.py --gse GSE62944 --cancer multi
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging
import json
from datetime import datetime

# GEOparse
try:
    import GEOparse
    HAS_GEOPARSE = True
except ImportError:
    HAS_GEOPARSE = False
    print("GEOparse not installed. Run: pip install GEOparse")

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml import PanCancerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 알려진 GEO 데이터셋 (검증용)
KNOWN_GEO_DATASETS = {
    # 유방암
    'GSE96058': {'cancer': 'BRCA', 'platform': 'RNA-seq', 'samples': 3273, 'desc': 'BRCA RNA-seq'},
    'GSE81538': {'cancer': 'BRCA', 'platform': 'RNA-seq', 'samples': 405, 'desc': 'BRCA TNBC'},

    # 폐암
    'GSE81089': {'cancer': 'LUAD', 'platform': 'RNA-seq', 'samples': 199, 'desc': 'LUAD RNA-seq'},
    'GSE40419': {'cancer': 'LUAD', 'platform': 'RNA-seq', 'samples': 87, 'desc': 'LUAD paired'},

    # 대장암
    'GSE39582': {'cancer': 'COAD', 'platform': 'Microarray', 'samples': 566, 'desc': 'CRC microarray'},

    # 간암
    'GSE14520': {'cancer': 'LIHC', 'platform': 'Microarray', 'samples': 445, 'desc': 'HCC microarray'},

    # 췌장암
    'GSE62452': {'cancer': 'PAAD', 'platform': 'Microarray', 'samples': 130, 'desc': 'PDAC microarray'},

    # 멀티 암종
    'GSE62944': {'cancer': 'multi', 'platform': 'RNA-seq', 'samples': 9264, 'desc': 'TCGA reprocessed'},
}


class GEODataLoader:
    """GEO 데이터셋 로더"""

    def __init__(self, cache_dir: str = "data/geo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_gse(self, gse_id: str) -> Optional[Any]:
        """GSE 데이터셋 다운로드"""
        if not HAS_GEOPARSE:
            logger.error("GEOparse not installed")
            return None

        logger.info(f"Downloading {gse_id}...")

        try:
            gse = GEOparse.get_GEO(
                geo=gse_id,
                destdir=str(self.cache_dir),
                silent=True
            )
            logger.info(f"Downloaded {gse_id} successfully")
            return gse
        except Exception as e:
            logger.error(f"Failed to download {gse_id}: {e}")
            return None

    def extract_expression_matrix(self, gse) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        GSE에서 발현 매트릭스 추출

        Returns:
            (expression_df, metadata_df)
        """
        if gse is None:
            return None, None

        # GSM (샘플) 데이터 추출
        gsm_names = list(gse.gsms.keys())

        if not gsm_names:
            logger.error("No samples found in GSE")
            return None, None

        # 첫 번째 샘플로 구조 확인
        first_gsm = gse.gsms[gsm_names[0]]

        # 발현 데이터 추출
        expression_data = {}
        metadata_list = []

        for gsm_name in gsm_names:
            gsm = gse.gsms[gsm_name]

            # 발현 값 추출
            if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
                # ID_REF와 VALUE 컬럼 확인
                if 'ID_REF' in gsm.table.columns and 'VALUE' in gsm.table.columns:
                    expr_series = gsm.table.set_index('ID_REF')['VALUE']
                    expression_data[gsm_name] = expr_series
                elif len(gsm.table.columns) >= 2:
                    # 첫 번째 컬럼을 ID로, 두 번째를 VALUE로
                    expr_series = gsm.table.set_index(gsm.table.columns[0])[gsm.table.columns[1]]
                    expression_data[gsm_name] = expr_series

            # 메타데이터 추출
            meta = {
                'sample_id': gsm_name,
                'title': gsm.metadata.get('title', [''])[0],
                'source': gsm.metadata.get('source_name_ch1', [''])[0] if 'source_name_ch1' in gsm.metadata else '',
                'characteristics': gsm.metadata.get('characteristics_ch1', []),
            }
            metadata_list.append(meta)

        if not expression_data:
            logger.error("No expression data extracted")
            return None, None

        # DataFrame 생성
        expr_df = pd.DataFrame(expression_data)
        meta_df = pd.DataFrame(metadata_list)

        logger.info(f"Extracted expression matrix: {expr_df.shape}")
        logger.info(f"Samples: {len(meta_df)}")

        return expr_df, meta_df

    def infer_cancer_type(self, metadata: pd.DataFrame) -> pd.Series:
        """
        메타데이터에서 암종 추론

        Returns:
            샘플별 암종 레이블
        """
        cancer_keywords = {
            'BRCA': ['breast', 'mammary', 'brca'],
            'LUAD': ['lung adenocarcinoma', 'luad', 'lung adeno'],
            'LUSC': ['lung squamous', 'lusc', 'sqcc'],
            'COAD': ['colon', 'colorectal', 'crc', 'coad'],
            'LIHC': ['liver', 'hepatocellular', 'hcc', 'lihc'],
            'PAAD': ['pancrea', 'pdac', 'paad'],
            'GBM': ['glioblastoma', 'gbm', 'glioma'],
            'OV': ['ovarian', 'ovary'],
            'PRAD': ['prostate', 'prad'],
            'KIRC': ['kidney', 'renal', 'kirc'],
            'THCA': ['thyroid', 'thca'],
            'SKCM': ['melanoma', 'skin', 'skcm'],
            'STAD': ['stomach', 'gastric', 'stad'],
            'HNSC': ['head neck', 'hnsc', 'oral', 'pharyn'],
            'BLCA': ['bladder', 'blca', 'urothelial'],
            'LGG': ['low grade glioma', 'lgg', 'astrocytoma'],
            'UCEC': ['endometri', 'ucec', 'uterine'],
        }

        labels = []
        for _, row in metadata.iterrows():
            # title과 source에서 키워드 검색
            text = f"{row.get('title', '')} {row.get('source', '')}".lower()

            # characteristics도 확인
            chars = row.get('characteristics', [])
            if isinstance(chars, list):
                text += ' ' + ' '.join(str(c).lower() for c in chars)

            found_cancer = 'UNKNOWN'
            for cancer, keywords in cancer_keywords.items():
                if any(kw in text for kw in keywords):
                    found_cancer = cancer
                    break

            labels.append(found_cancer)

        return pd.Series(labels, index=metadata.index)


class GeneIDMapper:
    """유전자 ID 매핑 (Probe ID -> Ensembl ID)"""

    def __init__(self):
        self.probe_to_ensembl: Dict[str, str] = {}
        self.symbol_to_ensembl: Dict[str, str] = {}

    def load_platform_annotation(self, gpl_id: str, cache_dir: str = "data/geo_cache"):
        """GPL 플랫폼 주석 로드"""
        cache_path = Path(cache_dir) / f"{gpl_id}_annotation.csv"

        if cache_path.exists():
            logger.info(f"Loading cached annotation: {cache_path}")
            df = pd.read_csv(cache_path)
            self._build_mapping(df)
            return

        try:
            logger.info(f"Downloading {gpl_id} annotation...")
            gpl = GEOparse.get_GEO(geo=gpl_id, destdir=cache_dir, silent=True)

            if hasattr(gpl, 'table') and gpl.table is not None:
                df = gpl.table
                df.to_csv(cache_path, index=False)
                self._build_mapping(df)
        except Exception as e:
            logger.warning(f"Failed to load {gpl_id}: {e}")

    def _build_mapping(self, df: pd.DataFrame):
        """주석 테이블에서 매핑 구축"""
        # 일반적인 컬럼 이름들
        probe_cols = ['ID', 'PROBE_ID', 'probe_id']
        ensembl_cols = ['ENSEMBL', 'Ensembl', 'ensembl_gene_id', 'ENSEMBL_ID']
        symbol_cols = ['Gene Symbol', 'GENE_SYMBOL', 'Symbol', 'gene_symbol']

        probe_col = None
        ensembl_col = None
        symbol_col = None

        for col in probe_cols:
            if col in df.columns:
                probe_col = col
                break

        for col in ensembl_cols:
            if col in df.columns:
                ensembl_col = col
                break

        for col in symbol_cols:
            if col in df.columns:
                symbol_col = col
                break

        if probe_col and ensembl_col:
            for _, row in df.iterrows():
                probe = str(row[probe_col])
                ensembl = str(row[ensembl_col])
                if ensembl and ensembl != 'nan':
                    self.probe_to_ensembl[probe] = ensembl

        if symbol_col and ensembl_col:
            for _, row in df.iterrows():
                symbol = str(row[symbol_col])
                ensembl = str(row[ensembl_col])
                if symbol and symbol != 'nan' and ensembl and ensembl != 'nan':
                    self.symbol_to_ensembl[symbol.upper()] = ensembl

        logger.info(f"Built mapping: {len(self.probe_to_ensembl)} probes, {len(self.symbol_to_ensembl)} symbols")

    def map_to_ensembl(self, gene_ids: List[str]) -> Dict[str, str]:
        """유전자 ID를 Ensembl ID로 매핑"""
        mapping = {}
        for gene_id in gene_ids:
            # Probe ID 먼저 시도
            if gene_id in self.probe_to_ensembl:
                mapping[gene_id] = self.probe_to_ensembl[gene_id]
            # Symbol 시도
            elif gene_id.upper() in self.symbol_to_ensembl:
                mapping[gene_id] = self.symbol_to_ensembl[gene_id.upper()]
            # 이미 Ensembl ID인 경우
            elif gene_id.startswith('ENSG'):
                mapping[gene_id] = gene_id

        return mapping


def preprocess_geo_for_pancancer(
    expression_df: pd.DataFrame,
    model_genes: List[str],
    gene_mapper: Optional[GeneIDMapper] = None
) -> pd.DataFrame:
    """
    GEO 데이터를 Pan-Cancer 모델 입력 형식으로 변환

    Args:
        expression_df: GEO 발현 매트릭스 (genes x samples)
        model_genes: 모델에서 사용하는 유전자 목록 (Ensembl ID)
        gene_mapper: 유전자 ID 매퍼

    Returns:
        변환된 발현 매트릭스
    """
    logger.info(f"Preprocessing GEO data: {expression_df.shape}")

    # 유전자 ID 매핑
    geo_genes = expression_df.index.tolist()

    if gene_mapper:
        gene_mapping = gene_mapper.map_to_ensembl(geo_genes)
    else:
        # 직접 매핑 시도 (Ensembl ID인 경우)
        gene_mapping = {g: g for g in geo_genes if g.startswith('ENSG')}

    logger.info(f"Mapped {len(gene_mapping)}/{len(geo_genes)} genes to Ensembl IDs")

    # 모델 유전자와 매칭
    matched_genes = []
    for model_gene in model_genes:
        # 버전 제거 (ENSG00000123456.1 -> ENSG00000123456)
        model_gene_base = model_gene.split('.')[0]

        for geo_gene, ensembl_id in gene_mapping.items():
            ensembl_base = ensembl_id.split('.')[0]
            if ensembl_base == model_gene_base:
                matched_genes.append((model_gene, geo_gene))
                break

    logger.info(f"Matched {len(matched_genes)}/{len(model_genes)} model genes")

    if len(matched_genes) < 100:
        logger.warning("Very few genes matched. Check gene ID format.")

    # 새 데이터프레임 생성
    result_data = {}
    for model_gene, geo_gene in matched_genes:
        result_data[model_gene] = expression_df.loc[geo_gene]

    # 누락된 유전자는 0으로 채움
    for model_gene in model_genes:
        if model_gene not in result_data:
            result_data[model_gene] = pd.Series(0.0, index=expression_df.columns)

    result_df = pd.DataFrame(result_data).T  # genes x samples

    # 순서 맞추기
    result_df = result_df.loc[model_genes]

    logger.info(f"Final preprocessed shape: {result_df.shape}")

    return result_df


def validate_on_geo(
    gse_id: str,
    model_dir: str = "models/rnaseq/pancancer",
    expected_cancer: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    GEO 데이터셋으로 모델 검증

    Args:
        gse_id: GEO Series ID (예: GSE96058)
        model_dir: Pan-Cancer 모델 경로
        expected_cancer: 예상 암종 (검증용)
        output_dir: 결과 저장 경로

    Returns:
        검증 결과
    """
    results = {
        'gse_id': gse_id,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',
    }

    # 1. 모델 로드
    logger.info("Loading Pan-Cancer classifier...")
    classifier = PanCancerClassifier(model_dir)
    classifier.load()
    model_genes = classifier.preprocessor.selected_genes

    # 2. GEO 데이터 다운로드
    loader = GEODataLoader()
    gse = loader.download_gse(gse_id)

    if gse is None:
        results['status'] = 'failed'
        results['error'] = 'Failed to download GSE'
        return results

    # 3. 발현 데이터 추출
    expr_df, meta_df = loader.extract_expression_matrix(gse)

    if expr_df is None:
        results['status'] = 'failed'
        results['error'] = 'Failed to extract expression data'
        return results

    results['n_samples'] = expr_df.shape[1]
    results['n_genes_original'] = expr_df.shape[0]

    # 4. 암종 레이블 추론
    cancer_labels = loader.infer_cancer_type(meta_df)
    results['inferred_cancers'] = cancer_labels.value_counts().to_dict()

    # 5. 유전자 ID 매핑 및 전처리
    gene_mapper = GeneIDMapper()

    # GPL 플랫폼 정보 확인
    if hasattr(gse, 'gpls') and gse.gpls:
        gpl_id = list(gse.gpls.keys())[0]
        gene_mapper.load_platform_annotation(gpl_id)

    processed_df = preprocess_geo_for_pancancer(expr_df, model_genes, gene_mapper)
    results['n_genes_matched'] = (processed_df.sum(axis=1) > 0).sum()

    # 6. 예측
    logger.info("Running predictions...")
    predictions = classifier.predict(processed_df)

    # 7. 결과 분석
    pred_cancers = [p.predicted_cancer for p in predictions]
    pred_confidence = [p.confidence for p in predictions]
    is_unknown = [p.is_unknown for p in predictions]

    results['predictions'] = {
        'cancer_distribution': pd.Series(pred_cancers).value_counts().to_dict(),
        'mean_confidence': float(np.mean(pred_confidence)),
        'unknown_rate': float(np.mean(is_unknown)),
    }

    # 8. 정확도 계산 (알려진 레이블이 있는 경우)
    if expected_cancer and expected_cancer != 'multi':
        # 단일 암종 데이터셋
        correct = sum(1 for p in pred_cancers if p == expected_cancer)
        accuracy = correct / len(pred_cancers)
        results['accuracy'] = float(accuracy)
        results['expected_cancer'] = expected_cancer
        logger.info(f"Accuracy for {expected_cancer}: {accuracy:.2%}")
    else:
        # 추론된 레이블과 비교
        known_samples = cancer_labels[cancer_labels != 'UNKNOWN']
        if len(known_samples) > 0:
            correct = sum(
                1 for i, label in known_samples.items()
                if pred_cancers[i] == label
            )
            accuracy = correct / len(known_samples)
            results['accuracy_on_known'] = float(accuracy)
            logger.info(f"Accuracy on known samples: {accuracy:.2%}")

    results['status'] = 'completed'

    # 9. 결과 저장
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 상세 결과 저장
        detailed_results = pd.DataFrame({
            'sample_id': meta_df['sample_id'],
            'predicted_cancer': pred_cancers,
            'confidence': pred_confidence,
            'is_unknown': is_unknown,
            'inferred_label': cancer_labels.values,
        })
        detailed_results.to_csv(output_path / f"{gse_id}_predictions.csv", index=False)

        # 요약 저장
        with open(output_path / f"{gse_id}_summary.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    return results


def print_validation_report(results: Dict[str, Any]):
    """검증 결과 리포트 출력"""
    print("\n" + "="*70)
    print(f"  External Validation Report: {results['gse_id']}")
    print("="*70)

    print(f"\n  Status: {results['status']}")
    print(f"  Samples: {results.get('n_samples', 'N/A')}")
    print(f"  Genes matched: {results.get('n_genes_matched', 'N/A')}/{results.get('n_genes_original', 'N/A')}")

    if 'inferred_cancers' in results:
        print("\n  Inferred Cancer Types:")
        for cancer, count in results['inferred_cancers'].items():
            print(f"    - {cancer}: {count}")

    if 'predictions' in results:
        print("\n  Prediction Results:")
        print(f"    Mean confidence: {results['predictions']['mean_confidence']:.2%}")
        print(f"    Unknown rate: {results['predictions']['unknown_rate']:.2%}")
        print("\n    Cancer distribution:")
        for cancer, count in results['predictions']['cancer_distribution'].items():
            print(f"      - {cancer}: {count}")

    if 'accuracy' in results:
        print(f"\n  ✅ Accuracy: {results['accuracy']:.2%}")
    elif 'accuracy_on_known' in results:
        print(f"\n  ✅ Accuracy (known samples): {results['accuracy_on_known']:.2%}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Validate Pan-Cancer model on external GEO dataset")
    parser.add_argument('--gse', type=str, default=None, help='GEO Series ID (e.g., GSE96058)')
    parser.add_argument('--cancer', type=str, default=None, help='Expected cancer type')
    parser.add_argument('--model', type=str, default='models/rnaseq/pancancer', help='Model directory')
    parser.add_argument('--output', type=str, default='validation_results', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List known GEO datasets')

    args = parser.parse_args()

    if args.list:
        print("\nKnown GEO datasets for validation:")
        print("-"*70)
        for gse_id, info in KNOWN_GEO_DATASETS.items():
            print(f"  {gse_id}: {info['desc']} ({info['cancer']}, {info['samples']} samples)")
        print("-"*70)
        return

    if args.gse is None:
        parser.error("--gse is required unless --list is specified")

    # 검증 실행
    results = validate_on_geo(
        gse_id=args.gse,
        model_dir=args.model,
        expected_cancer=args.cancer,
        output_dir=args.output
    )

    print_validation_report(results)


if __name__ == "__main__":
    main()
