"""
RNA-seq Predictor API
=====================

학습된 모델을 사용하여 새 샘플 예측 및 설명

Features:
- 단일/배치 예측
- SHAP 기반 설명
- Gene Status Card 생성
- 불확실성 명시 (Guardrail)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import json
import logging

from .preprocessor import RNAseqPreprocessor
from .trainer import CatBoostTrainer
from .explainer import SHAPExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """예측 결과"""
    sample_id: str
    prediction: int  # 0: Normal, 1: Tumor
    probability: float
    confidence: str  # "low", "medium", "high"
    top_genes: List[Dict[str, Any]]
    warnings: List[str]

    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'prediction': self.prediction,
            'prediction_label': 'Tumor' if self.prediction == 1 else 'Normal',
            'probability': self.probability,
            'confidence': self.confidence,
            'top_genes': self.top_genes,
            'warnings': self.warnings,
        }


class RNAseqPredictor:
    """RNA-seq 샘플 분류 예측기"""

    # 신뢰도 임계값
    CONFIDENCE_THRESHOLDS = {
        'high': 0.9,
        'medium': 0.7,
        'low': 0.0,
    }

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: 학습된 모델 디렉토리 경로
        """
        self.model_dir = Path(model_dir)
        self.preprocessor: Optional[RNAseqPreprocessor] = None
        self.trainer: Optional[CatBoostTrainer] = None
        self.explainer: Optional[SHAPExplainer] = None
        self.metadata: Dict = {}
        self._loaded = False

    def load(self):
        """모델 및 관련 컴포넌트 로드"""
        logger.info(f"Loading model from {self.model_dir}")

        # 메타데이터 로드
        metadata_path = self.model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        # Preprocessor 로드
        preprocessor_path = self.model_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            self.preprocessor = RNAseqPreprocessor.load(str(preprocessor_path))
        else:
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        # Trainer (CatBoost 모델) 로드
        self.trainer = CatBoostTrainer.load(str(self.model_dir))

        # SHAP Explainer 초기화
        self.explainer = SHAPExplainer(
            self.trainer.model,
            feature_names=self.preprocessor.selected_genes
        )

        self._loaded = True
        logger.info("Model loaded successfully")

    def predict(self, counts: pd.DataFrame,
                sample_ids: Optional[List[str]] = None,
                explain: bool = True,
                top_k_genes: int = 10) -> List[PredictionResult]:
        """
        새 샘플 예측

        Args:
            counts: Gene x Sample count matrix
            sample_ids: 샘플 ID 목록 (없으면 컬럼 이름 사용)
            explain: SHAP 설명 포함 여부
            top_k_genes: 상위 유전자 수

        Returns:
            예측 결과 리스트
        """
        if not self._loaded:
            self.load()

        # 샘플 ID 설정
        if sample_ids is None:
            sample_ids = counts.columns.tolist()

        # 전처리
        X = self.preprocessor.transform(counts)

        # 예측
        probabilities = self.trainer.predict_proba(X)[:, 1]
        predictions = (probabilities > 0.5).astype(int)

        # SHAP Explainer 초기화 (배경 데이터 없이)
        if explain and self.explainer.explainer is None:
            # 현재 데이터를 배경으로 사용
            self.explainer.fit(X, sample_size=min(len(X), 50))

        results = []
        for i, (sample_id, pred, prob) in enumerate(zip(sample_ids, predictions, probabilities)):
            # 신뢰도 계산
            confidence = self._calculate_confidence(prob)

            # 경고 메시지
            warnings = self._generate_warnings(prob, confidence)

            # SHAP 설명
            top_genes = []
            if explain:
                sample_explanation = self.explainer.explain_sample(
                    X[i], top_k=top_k_genes
                )
                top_genes = sample_explanation['top_genes']

            result = PredictionResult(
                sample_id=sample_id,
                prediction=int(pred),
                probability=float(prob),
                confidence=confidence,
                top_genes=top_genes,
                warnings=warnings,
            )
            results.append(result)

        return results

    def predict_single(self, counts: pd.Series,
                      sample_id: str = "sample",
                      explain: bool = True) -> PredictionResult:
        """
        단일 샘플 예측

        Args:
            counts: 유전자별 count (Series)
            sample_id: 샘플 ID
            explain: SHAP 설명 포함

        Returns:
            예측 결과
        """
        df = pd.DataFrame({sample_id: counts})
        results = self.predict(df, [sample_id], explain=explain)
        return results[0]

    def _calculate_confidence(self, probability: float) -> str:
        """확률에서 신뢰도 레벨 계산"""
        # 0.5에서 멀수록 높은 신뢰도
        distance_from_uncertain = abs(probability - 0.5) * 2

        if distance_from_uncertain >= self.CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif distance_from_uncertain >= self.CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'

    def _generate_warnings(self, probability: float, confidence: str) -> List[str]:
        """경고 메시지 생성 (Guardrail)"""
        warnings = [
            "⚠️ 이 예측은 참고용이며, 진단 목적으로 사용할 수 없습니다.",
        ]

        if confidence == 'low':
            warnings.append("⚠️ 신뢰도가 낮습니다. 결과 해석에 주의가 필요합니다.")

        if 0.4 <= probability <= 0.6:
            warnings.append("⚠️ 확률이 경계값 근처입니다. 추가 검증을 권장합니다.")

        return warnings

    def get_gene_status_card(self, gene_name: str,
                            counts: pd.DataFrame,
                            extra_info: Optional[Dict] = None) -> str:
        """
        특정 유전자의 Status Card 생성

        Args:
            gene_name: 유전자 이름
            counts: Count matrix
            extra_info: 추가 정보 (DB 검증 결과 등)

        Returns:
            포맷된 Status Card
        """
        if not self._loaded:
            self.load()

        X = self.preprocessor.transform(counts)

        if self.explainer.explainer is None:
            self.explainer.fit(X, sample_size=min(len(X), 50))

        return self.explainer.generate_gene_status_card(
            gene_name, X, extra_info=extra_info
        )

    def batch_predict(self, counts: pd.DataFrame,
                     output_path: Optional[str] = None) -> pd.DataFrame:
        """
        배치 예측 및 결과 저장

        Args:
            counts: Count matrix
            output_path: 결과 저장 경로 (optional)

        Returns:
            예측 결과 DataFrame
        """
        results = self.predict(counts, explain=False)

        df = pd.DataFrame([r.to_dict() for r in results])

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")

        return df

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self._loaded:
            self.load()

        return {
            'cancer_type': self.metadata.get('cancer_type', 'unknown'),
            'n_genes': len(self.preprocessor.selected_genes) if self.preprocessor else 0,
            'training_samples': self.metadata.get('n_training_samples', 0),
            'cv_auc': self.metadata.get('cv_auc', 0),
            'training_date': self.metadata.get('training_date', 'unknown'),
            'model_version': self.metadata.get('version', '1.0'),
        }

    def export_feature_importance(self, output_path: str, top_k: int = 100):
        """특징 중요도 내보내기"""
        if not self._loaded:
            self.load()

        importance = self.trainer.get_feature_importance(top_k=top_k)
        importance.to_csv(output_path, index=False)
        logger.info(f"Feature importance saved to {output_path}")


def train_and_save_model(cancer_type: str,
                         counts: pd.DataFrame,
                         metadata: pd.DataFrame,
                         output_dir: str,
                         optimize_hyperparams: bool = True,
                         n_optuna_trials: int = 30) -> Dict[str, Any]:
    """
    전체 학습 파이프라인 실행 및 모델 저장

    Args:
        cancer_type: 암 종류
        counts: Count matrix (genes x samples)
        metadata: 메타데이터 (sample_id, label 포함)
        output_dir: 저장 디렉토리
        optimize_hyperparams: 하이퍼파라미터 최적화 여부
        n_optuna_trials: Optuna 시도 횟수

    Returns:
        학습 결과
    """
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training RNA-seq Classifier: {cancer_type}")
    logger.info(f"{'='*60}")

    # 레이블 정렬
    sample_order = counts.columns.tolist()
    metadata_indexed = metadata.set_index('barcode') if 'barcode' in metadata.columns else metadata.set_index(metadata.columns[0])

    labels = []
    valid_samples = []
    for sample in sample_order:
        if sample in metadata_indexed.index:
            labels.append(metadata_indexed.loc[sample, 'label'])
            valid_samples.append(sample)

    counts = counts[valid_samples]
    labels = np.array(labels)

    logger.info(f"Samples: {len(labels)} (Tumor: {(labels==1).sum()}, Normal: {(labels==0).sum()})")

    # 1. 전처리
    logger.info("\n[1/4] Preprocessing...")
    preprocessor = RNAseqPreprocessor(
        min_counts=10,
        min_samples_pct=0.2,
        log_transform=True,
        normalize="cpm",
        n_top_genes=3000,
        feature_selection="anova",
    )

    X_train, X_test, y_train, y_test = preprocessor.fit_transform(counts, labels)
    preprocessor.save(str(output_dir / "preprocessor.joblib"))

    # 2. 모델 학습
    logger.info("\n[2/4] Training model...")
    trainer = CatBoostTrainer(task_type="CPU", verbose=True)

    if optimize_hyperparams:
        logger.info("Running hyperparameter optimization...")
        best_params = trainer.optimize_hyperparameters(
            X_train, y_train,
            n_trials=n_optuna_trials,
            feature_names=preprocessor.selected_genes,
        )
    else:
        best_params = trainer.get_default_params()

    # 교차 검증
    cv_results = trainer.cross_validate(
        X_train, y_train,
        feature_names=preprocessor.selected_genes,
        params=best_params,
    )

    # 최종 모델 학습
    trainer.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        feature_names=preprocessor.selected_genes,
        params=best_params,
    )

    # 3. 평가
    logger.info("\n[3/4] Evaluating...")
    test_metrics = trainer.evaluate(X_test, y_test)
    trainer.save(str(output_dir))

    # 4. SHAP 분석
    logger.info("\n[4/4] SHAP analysis...")
    explainer = SHAPExplainer(trainer.model, preprocessor.selected_genes)
    explainer.fit(X_train)

    # SHAP plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    explainer.plot_summary(X_test, str(plots_dir / "shap_summary.png"))
    global_importance = explainer.get_global_importance(X_test)
    global_importance.to_csv(output_dir / "shap_importance.csv", index=False)

    # 메타데이터 저장
    model_metadata = {
        'cancer_type': cancer_type,
        'version': '1.0',
        'training_date': datetime.now().isoformat(),
        'n_training_samples': len(y_train),
        'n_test_samples': len(y_test),
        'n_genes': len(preprocessor.selected_genes),
        'cv_auc': cv_results['roc_auc']['mean'],
        'test_auc': test_metrics['roc_auc'],
        'test_accuracy': test_metrics['accuracy'],
        'best_params': best_params,
    }

    with open(output_dir / "model_metadata.json", 'w') as f:
        json.dump(model_metadata, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"  CV AUC:   {cv_results['roc_auc']['mean']:.4f} (+/- {cv_results['roc_auc']['std']:.4f})")
    logger.info(f"  Test AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Model saved to: {output_dir}")
    logger.info(f"{'='*60}\n")

    return model_metadata


def main():
    """CLI 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="RNA-seq ML Predictor")
    subparsers = parser.add_subparsers(dest='command')

    # train 명령
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--cancer', '-c', type=str, required=True,
                             help='Cancer type (pancreatic, lung, etc.)')
    train_parser.add_argument('--data-dir', '-d', type=str, default='data/tcga',
                             help='TCGA data directory')
    train_parser.add_argument('--output', '-o', type=str, default='models/rnaseq',
                             help='Output directory')
    train_parser.add_argument('--no-optimize', action='store_true',
                             help='Skip hyperparameter optimization')

    # predict 명령
    predict_parser = subparsers.add_parser('predict', help='Predict new samples')
    predict_parser.add_argument('--model', '-m', type=str, required=True,
                               help='Model directory')
    predict_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input count matrix (CSV)')
    predict_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output predictions (CSV)')

    args = parser.parse_args()

    if args.command == 'train':
        from .tcga_downloader import TCGADownloader

        downloader = TCGADownloader(output_dir=args.data_dir)
        counts, metadata = downloader.load_project(args.cancer)

        output_dir = Path(args.output) / args.cancer
        train_and_save_model(
            cancer_type=args.cancer,
            counts=counts,
            metadata=metadata,
            output_dir=str(output_dir),
            optimize_hyperparams=not args.no_optimize,
        )

    elif args.command == 'predict':
        predictor = RNAseqPredictor(args.model)
        counts = pd.read_csv(args.input, index_col=0)
        predictor.batch_predict(counts, args.output)


if __name__ == "__main__":
    main()
