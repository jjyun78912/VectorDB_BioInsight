#!/usr/bin/env python3
"""
RNA-seq Classifier Training Script
===================================

TCGA 데이터를 다운로드하고 CatBoost 분류기를 학습합니다.

Usage:
    python scripts/train_rnaseq_classifier.py --cancer pancreatic
    python scripts/train_rnaseq_classifier.py --cancer lung --max-samples 300
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train RNA-seq classifier with TCGA data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train pancreatic cancer classifier
  python scripts/train_rnaseq_classifier.py --cancer pancreatic

  # Train with more samples and skip optimization
  python scripts/train_rnaseq_classifier.py --cancer lung --max-samples 300 --no-optimize

Supported cancer types:
  - pancreatic (TCGA-PAAD)
  - lung (TCGA-LUAD)
  - breast (TCGA-BRCA)
  - colon (TCGA-COAD)
  - liver (TCGA-LIHC)
        """
    )

    parser.add_argument('--cancer', '-c', type=str, default='pancreatic',
                       help='Cancer type (default: pancreatic)')
    parser.add_argument('--max-samples', '-n', type=int, default=200,
                       help='Maximum samples to download (default: 200)')
    parser.add_argument('--data-dir', '-d', type=str, default='data/tcga',
                       help='TCGA data directory')
    parser.add_argument('--output', '-o', type=str, default='models/rnaseq',
                       help='Output model directory')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Skip hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Optuna trials for optimization (default: 30)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download if data exists')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Download workers (default: 4)')

    args = parser.parse_args()

    print(f"""
{'='*60}
  BioInsight AI - RNA-seq Classifier Training
{'='*60}
  Cancer Type:    {args.cancer}
  Max Samples:    {args.max_samples}
  Data Dir:       {args.data_dir}
  Output Dir:     {args.output}
  Optimization:   {'Disabled' if args.no_optimize else f'Enabled ({args.n_trials} trials)'}
{'='*60}
""")

    # 의존성 확인
    try:
        from rnaseq_pipeline.ml import (
            TCGADownloader, RNAseqPreprocessor, CatBoostTrainer, SHAPExplainer
        )
        from rnaseq_pipeline.ml.predictor import train_and_save_model
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nPlease install required packages:")
        print("  pip install catboost shap optuna scikit-learn pandas numpy tqdm")
        sys.exit(1)

    # 1. TCGA 데이터 다운로드/로드
    print("\n[Step 1] Loading TCGA data...")

    downloader = TCGADownloader(output_dir=args.data_dir)

    try:
        if args.skip_download:
            counts, metadata = downloader.load_project(args.cancer)
            print(f"Loaded existing data: {counts.shape[0]} genes x {counts.shape[1]} samples")
        else:
            try:
                counts, metadata = downloader.load_project(args.cancer)
                print(f"Found existing data: {counts.shape[0]} genes x {counts.shape[1]} samples")
            except FileNotFoundError:
                print(f"Downloading TCGA-{args.cancer.upper()} data...")
                counts, metadata = downloader.download_project(
                    cancer_type=args.cancer,
                    max_samples=args.max_samples,
                    include_normal=True,
                    n_workers=args.workers,
                )
    except Exception as e:
        logger.error(f"Failed to load/download data: {e}")
        sys.exit(1)

    # 데이터 통계 출력
    n_tumor = (metadata['label'] == 1).sum()
    n_normal = (metadata['label'] == 0).sum()
    print(f"\nData loaded:")
    print(f"  - Total samples: {len(metadata)}")
    print(f"  - Tumor:  {n_tumor}")
    print(f"  - Normal: {n_normal}")
    print(f"  - Genes:  {counts.shape[0]}")

    # 클래스 불균형 경고
    if n_normal < 5:
        print(f"\n⚠️ Warning: Very few normal samples ({n_normal}). Model may be biased.")
        print("  Consider using different cancer type or including more samples.")

    # 2. 모델 학습
    print("\n[Step 2] Training classifier...")

    output_dir = Path(args.output) / args.cancer
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = train_and_save_model(
            cancer_type=args.cancer,
            counts=counts,
            metadata=metadata,
            output_dir=str(output_dir),
            optimize_hyperparams=not args.no_optimize,
            n_optuna_trials=args.n_trials,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. 결과 요약
    print(f"""
{'='*60}
  Training Complete!
{'='*60}
  Model saved to: {output_dir}

  Performance:
    - CV AUC:       {results.get('cv_auc', 0):.4f}
    - Test AUC:     {results.get('test_auc', 0):.4f}
    - Test Accuracy: {results.get('test_accuracy', 0):.4f}

  Files:
    - model.cbm          : CatBoost model
    - preprocessor.joblib: Data preprocessor
    - model_metadata.json: Model info
    - shap_importance.csv: Feature importance
    - plots/             : SHAP visualizations

  Usage:
    from rnaseq_pipeline.ml import RNAseqPredictor

    predictor = RNAseqPredictor("{output_dir}")
    results = predictor.predict(your_count_matrix)
{'='*60}
""")


if __name__ == "__main__":
    main()
