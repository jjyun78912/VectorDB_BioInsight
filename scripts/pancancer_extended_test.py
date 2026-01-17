#!/usr/bin/env python3
"""
Pan-Cancer Extended Validation Script
======================================

암종별로 더 많은 샘플을 다운로드하여 모델 성능을 상세 검증

기능:
1. 암종별 N개 샘플 다운로드 (기존 데이터 외 신규)
2. 모델 예측 및 정확도 측정
3. Confusion Matrix 및 per-class metrics
4. 오분류 케이스 상세 분석
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import defaultdict

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml import PanCancerClassifier
from rnaseq_pipeline.ml.pancancer_downloader import PanCancerDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PanCancerExtendedTest:
    """Pan-Cancer 확장 테스트"""

    def __init__(self,
                 model_dir: str = "models/rnaseq/pancancer",
                 test_data_dir: str = "data/tcga/pancancer_test"):
        self.model_dir = Path(model_dir)
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        self.classifier: Optional[PanCancerClassifier] = None
        self.downloader: Optional[PanCancerDownloader] = None

    def load_model(self):
        """모델 로드"""
        logger.info("Loading Pan-Cancer classifier...")
        self.classifier = PanCancerClassifier(str(self.model_dir))
        self.classifier.load()

        info = self.classifier.get_model_info()
        logger.info(f"Model loaded: {info['n_classes']} classes, {info['n_genes']} genes")
        return info

    def download_test_samples(self,
                              samples_per_cancer: int = 10,
                              cancer_types: Optional[List[str]] = None,
                              skip_existing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        테스트용 샘플 다운로드 (학습 데이터와 별도)

        Args:
            samples_per_cancer: 암종당 샘플 수
            cancer_types: 테스트할 암종 목록 (None이면 모델의 17종)
            skip_existing: 기존 데이터 있으면 스킵
        """
        if cancer_types is None:
            # 모델의 17종 사용
            cancer_types = [
                "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LGG",
                "LIHC", "LUAD", "LUSC", "OV", "PAAD", "PRAD", "SKCM",
                "STAD", "THCA", "UCEC"
            ]

        # 캐시 확인
        counts_path = self.test_data_dir / "test_counts.csv"
        meta_path = self.test_data_dir / "test_metadata.csv"

        if skip_existing and counts_path.exists() and meta_path.exists():
            logger.info("Loading existing test data...")
            counts = pd.read_csv(counts_path, index_col=0)
            metadata = pd.read_csv(meta_path)

            # 요청한 샘플 수 확인
            existing_per_cancer = metadata.groupby('cancer_type').size().to_dict()
            need_more = any(existing_per_cancer.get(c, 0) < samples_per_cancer
                          for c in cancer_types)

            if not need_more:
                logger.info(f"Existing test data sufficient: {len(metadata)} samples")
                return counts, metadata
            else:
                logger.info("Need to download additional samples...")

        # 다운로더 초기화
        self.downloader = PanCancerDownloader(output_dir=str(self.test_data_dir))

        logger.info(f"\n{'='*60}")
        logger.info(f"  Downloading Test Samples")
        logger.info(f"  Cancer types: {len(cancer_types)}")
        logger.info(f"  Samples per type: {samples_per_cancer}")
        logger.info(f"{'='*60}\n")

        all_matrices = []
        all_metadata = []

        for i, cancer_code in enumerate(cancer_types, 1):
            logger.info(f"[{i}/{len(cancer_types)}] Downloading {cancer_code}...")

            try:
                count_matrix, metadata = self.downloader.download_cancer_type(
                    cancer_code,
                    max_samples=samples_per_cancer,
                    n_workers=4
                )

                if not count_matrix.empty:
                    all_matrices.append(count_matrix)
                    all_metadata.append(metadata)
                    logger.info(f"  → {len(metadata)} samples downloaded")
            except Exception as e:
                logger.error(f"Failed to download {cancer_code}: {e}")
                continue

        if not all_matrices:
            raise ValueError("No test data downloaded")

        # 통합
        common_genes = set(all_matrices[0].index)
        for matrix in all_matrices[1:]:
            common_genes &= set(matrix.index)
        common_genes = sorted(list(common_genes))

        combined_matrices = [m.loc[common_genes] for m in all_matrices]
        combined_counts = pd.concat(combined_matrices, axis=1)
        combined_metadata = pd.concat(all_metadata, ignore_index=True)

        # 저장
        combined_counts.to_csv(counts_path)
        combined_metadata.to_csv(meta_path, index=False)

        logger.info(f"\nTest data saved: {len(combined_metadata)} samples, {len(common_genes)} genes")

        return combined_counts, combined_metadata

    def run_validation(self,
                       counts: pd.DataFrame,
                       metadata: pd.DataFrame,
                       verbose: bool = True) -> Dict:
        """
        전체 검증 실행

        Returns:
            검증 결과 딕셔너리
        """
        if self.classifier is None:
            self.load_model()

        logger.info(f"\n{'='*60}")
        logger.info(f"  Running Validation")
        logger.info(f"  Samples: {len(metadata)}")
        logger.info(f"  Cancer types: {metadata['cancer_type'].nunique()}")
        logger.info(f"{'='*60}\n")

        # 샘플 ID 매핑
        sample_ids = counts.columns.tolist()
        barcode_to_cancer = dict(zip(metadata['barcode'], metadata['cancer_type']))

        # 예측
        logger.info("Running predictions...")
        results = self.classifier.predict(counts, sample_ids)

        # 결과 분석
        predictions = []
        for r in results:
            actual = barcode_to_cancer.get(r.sample_id, 'UNKNOWN')
            predictions.append({
                'sample_id': r.sample_id,
                'actual': actual,
                'predicted': r.predicted_cancer,
                'confidence': r.confidence,
                'confidence_level': r.confidence_level,
                'ensemble_agreement': r.ensemble_agreement,
                'is_unknown': r.is_unknown,
                'is_correct': r.predicted_cancer == actual,
                'top_2': r.top_k_predictions[1]['cancer'] if len(r.top_k_predictions) > 1 else None,
                'top_2_prob': r.top_k_predictions[1]['probability'] if len(r.top_k_predictions) > 1 else 0,
                'confidence_gap': r.confidence_gap,
                'is_confusable_pair': r.is_confusable_pair,
            })

        pred_df = pd.DataFrame(predictions)

        # 전체 정확도
        overall_accuracy = pred_df['is_correct'].mean()

        # 암종별 정확도
        per_cancer_accuracy = pred_df.groupby('actual')['is_correct'].agg(['mean', 'sum', 'count'])
        per_cancer_accuracy.columns = ['accuracy', 'correct', 'total']
        per_cancer_accuracy = per_cancer_accuracy.sort_values('accuracy')

        # Confusion Matrix
        cancer_types = sorted(pred_df['actual'].unique())
        confusion = pd.crosstab(pred_df['actual'], pred_df['predicted'],
                               margins=True, margins_name='Total')

        # 오분류 케이스
        misclassified = pred_df[~pred_df['is_correct']].copy()

        # 결과 출력
        if verbose:
            self._print_results(overall_accuracy, per_cancer_accuracy,
                               confusion, misclassified, pred_df)

        return {
            'overall_accuracy': overall_accuracy,
            'per_cancer_accuracy': per_cancer_accuracy.to_dict(),
            'confusion_matrix': confusion.to_dict(),
            'predictions': pred_df.to_dict('records'),
            'misclassified': misclassified.to_dict('records'),
            'n_samples': len(pred_df),
            'n_correct': pred_df['is_correct'].sum(),
            'n_wrong': (~pred_df['is_correct']).sum(),
            'timestamp': datetime.now().isoformat(),
        }

    def _print_results(self, overall_accuracy, per_cancer_accuracy,
                       confusion, misclassified, pred_df):
        """결과 출력"""

        print(f"\n{'='*70}")
        print(f"  VALIDATION RESULTS")
        print(f"{'='*70}")

        # 전체 정확도
        n_total = len(pred_df)
        n_correct = pred_df['is_correct'].sum()
        print(f"\n  Overall Accuracy: {overall_accuracy:.1%} ({n_correct}/{n_total})")

        # 암종별 정확도
        print(f"\n  Per-Cancer Accuracy:")
        print(f"  {'-'*50}")
        print(f"  {'Cancer':<10} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print(f"  {'-'*50}")

        for cancer, row in per_cancer_accuracy.iterrows():
            bar = "█" * int(row['accuracy'] * 20) + "░" * (20 - int(row['accuracy'] * 20))
            print(f"  {cancer:<10} {row['accuracy']:>9.1%} {int(row['correct']):>10} {int(row['total']):>10}  {bar}")

        # 오분류 분석
        if len(misclassified) > 0:
            print(f"\n  Misclassified Cases ({len(misclassified)}):")
            print(f"  {'-'*70}")
            print(f"  {'Sample':<25} {'Actual':<8} {'Predicted':<8} {'Conf':>7} {'Top-2':<8} {'Gap':>6}")
            print(f"  {'-'*70}")

            for _, row in misclassified.iterrows():
                sample = row['sample_id'][:23] + '..' if len(row['sample_id']) > 25 else row['sample_id']
                confusable = " ⚠️" if row['is_confusable_pair'] else ""
                print(f"  {sample:<25} {row['actual']:<8} {row['predicted']:<8} "
                      f"{row['confidence']:>6.1%} {row['top_2'] or '-':<8} {row['confidence_gap']:>5.1%}{confusable}")

            # 오분류 패턴 분석
            print(f"\n  Misclassification Patterns:")
            patterns = misclassified.groupby(['actual', 'predicted']).size().sort_values(ascending=False)
            for (actual, predicted), count in patterns.head(10).items():
                print(f"    {actual} → {predicted}: {count}회")

        # 신뢰도 분포
        print(f"\n  Confidence Distribution:")
        for level in ['high', 'medium', 'low', 'unknown']:
            count = (pred_df['confidence_level'] == level).sum()
            pct = count / len(pred_df) * 100
            print(f"    {level:>8}: {count:>4} ({pct:>5.1f}%)")

        # 혼동 가능 암종 쌍 분석
        confusable_cases = pred_df[pred_df['is_confusable_pair']]
        if len(confusable_cases) > 0:
            print(f"\n  Confusable Cancer Pair Cases: {len(confusable_cases)}")
            confusable_correct = confusable_cases['is_correct'].mean()
            print(f"    Accuracy in confusable pairs: {confusable_correct:.1%}")

        print(f"\n{'='*70}\n")

    def save_report(self, results: Dict, output_path: Optional[str] = None):
        """결과 저장"""
        if output_path is None:
            output_path = self.test_data_dir / "validation_report.json"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Report saved to: {output_path}")

        # CSV 형식 저장 (predictions)
        pred_df = pd.DataFrame(results['predictions'])
        pred_df.to_csv(output_path.parent / "predictions.csv", index=False)

        return output_path


def main():
    """CLI 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Pan-Cancer Extended Validation")
    parser.add_argument("--samples", "-n", type=int, default=10,
                       help="Samples per cancer type (default: 10)")
    parser.add_argument("--cancers", "-c", type=str, nargs='+', default=None,
                       help="Cancer types to test (default: all 17)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, use existing data only")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output report path")
    parser.add_argument("--model", "-m", type=str, default="models/rnaseq/pancancer",
                       help="Model directory")
    parser.add_argument("--data-dir", "-d", type=str, default="data/tcga/pancancer_test",
                       help="Test data directory")

    args = parser.parse_args()

    # 테스터 초기화
    tester = PanCancerExtendedTest(
        model_dir=args.model,
        test_data_dir=args.data_dir
    )

    # 모델 로드
    tester.load_model()

    # 테스트 데이터 준비
    if args.skip_download:
        counts_path = Path(args.data_dir) / "test_counts.csv"
        meta_path = Path(args.data_dir) / "test_metadata.csv"

        if not counts_path.exists():
            print("No existing test data found. Run without --skip-download first.")
            return

        counts = pd.read_csv(counts_path, index_col=0)
        metadata = pd.read_csv(meta_path)
    else:
        counts, metadata = tester.download_test_samples(
            samples_per_cancer=args.samples,
            cancer_types=args.cancers,
            skip_existing=True
        )

    # 검증 실행
    results = tester.run_validation(counts, metadata)

    # 결과 저장
    tester.save_report(results, args.output)

    print(f"\n✅ Validation complete!")
    print(f"   Overall Accuracy: {results['overall_accuracy']:.1%}")
    print(f"   Samples: {results['n_samples']}")
    print(f"   Correct: {results['n_correct']}")
    print(f"   Wrong: {results['n_wrong']}")


if __name__ == "__main__":
    main()
