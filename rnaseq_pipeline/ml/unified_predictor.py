"""
Unified RNA-seq Predictor
=========================

통합 예측 API:
- Pan-Cancer 멀티클래스 분류
- 암종별 Binary 분류 (세부 예측)
- Unknown/OOD 탐지
- Ensemble 결과 통합
- Gene Status Card + SHAP

사용 예시:
```python
from rnaseq_pipeline.ml import UnifiedPredictor

predictor = UnifiedPredictor()
result = predictor.predict(counts_df)

print(result.primary_prediction)      # "BRCA" or "UNKNOWN"
print(result.confidence)              # 0.87
print(result.is_unknown)              # False
print(result.top_predictions)         # [{"cancer": "BRCA", "prob": 0.87}, ...]
print(result.detailed_analysis)       # Binary 분류 결과 (해당 암종 모델)
```
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

from .pancancer_classifier import PanCancerClassifier, ClassificationResult
from .predictor import RNAseqPredictor, PredictionResult
from .explainer import SHAPExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedPredictionResult:
    """통합 예측 결과"""
    sample_id: str

    # Pan-Cancer 분류 결과
    predicted_cancer: str
    predicted_cancer_korean: str
    confidence: float
    confidence_level: str  # "high", "medium", "low", "unknown"
    is_unknown: bool

    # Top-K 예측
    top_predictions: List[Dict[str, Any]]

    # Ensemble 정보
    ensemble_agreement: float
    ensemble_models: List[str]

    # 세부 분석 (해당 암종 Binary 모델이 있는 경우)
    detailed_analysis: Optional[Dict[str, Any]] = None

    # SHAP 기반 주요 유전자
    top_genes: List[Dict[str, Any]] = field(default_factory=list)

    # 경고 및 권장사항
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'predicted_cancer': self.predicted_cancer,
            'predicted_cancer_korean': self.predicted_cancer_korean,
            'confidence': round(self.confidence, 4),
            'confidence_level': self.confidence_level,
            'is_unknown': self.is_unknown,
            'top_predictions': self.top_predictions,
            'ensemble_agreement': round(self.ensemble_agreement, 4),
            'ensemble_models': self.ensemble_models,
            'detailed_analysis': self.detailed_analysis,
            'top_genes': self.top_genes,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
        }

    def summary(self) -> str:
        """결과 요약 문자열"""
        lines = [
            f"{'='*60}",
            f"  Sample: {self.sample_id}",
            f"{'='*60}",
            "",
            f"  PREDICTION",
            f"  ─────────────────────────────────",
        ]

        if self.is_unknown:
            lines.append(f"  암종: ❓ UNKNOWN (신뢰도 부족)")
            lines.append(f"  상위 후보:")
            for pred in self.top_predictions[:3]:
                lines.append(f"    • {pred['cancer']} ({pred['cancer_korean']}): {pred['probability']:.1%}")
        else:
            lines.append(f"  암종: {self.predicted_cancer} ({self.predicted_cancer_korean})")
            lines.append(f"  신뢰도: {self.confidence:.1%} [{self.confidence_level.upper()}]")

        lines.extend([
            "",
            f"  ENSEMBLE ({len(self.ensemble_models)} models)",
            f"  ─────────────────────────────────",
            f"  모델 일치도: {self.ensemble_agreement:.1%}",
            f"  사용 모델: {', '.join(self.ensemble_models)}",
        ])

        if self.top_genes:
            lines.extend([
                "",
                f"  TOP CONTRIBUTING GENES",
                f"  ─────────────────────────────────",
            ])
            for gene in self.top_genes[:5]:
                direction = "↑" if gene.get('direction', 'up') == 'up' else "↓"
                lines.append(f"  {direction} {gene['gene']}: SHAP={gene.get('shap_value', 0):.4f}")

        if self.warnings:
            lines.extend([
                "",
                f"  WARNINGS",
                f"  ─────────────────────────────────",
            ])
            for warning in self.warnings:
                lines.append(f"  {warning}")

        if self.recommendations:
            lines.extend([
                "",
                f"  RECOMMENDATIONS",
                f"  ─────────────────────────────────",
            ])
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        lines.append(f"{'='*60}")

        return "\n".join(lines)


class UnifiedPredictor:
    """
    통합 RNA-seq 예측기

    기능:
    1. Pan-Cancer 멀티클래스 분류 (33개 암종)
    2. Unknown/OOD 탐지
    3. 특정 암종 Binary 모델로 세부 분석 (선택적)
    4. Ensemble 기반 신뢰도 평가
    5. SHAP 기반 주요 유전자 식별
    """

    # 사용 가능한 Binary 모델
    AVAILABLE_BINARY_MODELS = ['breast', 'pancreatic']

    def __init__(self,
                 pancancer_model_dir: str = "models/rnaseq/pancancer",
                 binary_models_dir: str = "models/rnaseq"):
        """
        Args:
            pancancer_model_dir: Pan-Cancer 모델 경로
            binary_models_dir: Binary 모델들 경로 (암종별 하위 폴더)
        """
        self.pancancer_model_dir = Path(pancancer_model_dir)
        self.binary_models_dir = Path(binary_models_dir)

        self.pancancer_classifier: Optional[PanCancerClassifier] = None
        self.binary_predictors: Dict[str, RNAseqPredictor] = {}

        self.is_loaded = False

    def load(self, load_binary: bool = True):
        """모델 로드"""
        logger.info("Loading Unified Predictor...")

        # Pan-Cancer 모델 로드
        if self.pancancer_model_dir.exists():
            logger.info(f"  Loading Pan-Cancer model from {self.pancancer_model_dir}")
            self.pancancer_classifier = PanCancerClassifier(str(self.pancancer_model_dir))
            self.pancancer_classifier.load()
        else:
            logger.warning(f"  Pan-Cancer model not found at {self.pancancer_model_dir}")

        # Binary 모델들 로드
        if load_binary:
            for cancer_type in self.AVAILABLE_BINARY_MODELS:
                model_path = self.binary_models_dir / cancer_type
                if model_path.exists():
                    logger.info(f"  Loading {cancer_type} binary model")
                    predictor = RNAseqPredictor(str(model_path))
                    try:
                        predictor.load()
                        self.binary_predictors[cancer_type] = predictor
                    except Exception as e:
                        logger.warning(f"  Failed to load {cancer_type} model: {e}")

        self.is_loaded = True
        logger.info("Unified Predictor loaded successfully")

    def predict(self,
                counts: pd.DataFrame,
                sample_ids: Optional[List[str]] = None,
                include_binary_analysis: bool = True,
                top_k: int = 5,
                top_genes: int = 10) -> List[UnifiedPredictionResult]:
        """
        통합 예측 수행

        Args:
            counts: Gene x Sample count matrix
            sample_ids: 샘플 ID 목록
            include_binary_analysis: Binary 모델 세부 분석 포함 여부
            top_k: Top-K 예측 수
            top_genes: 상위 유전자 수

        Returns:
            통합 예측 결과 리스트
        """
        if not self.is_loaded:
            self.load()

        if sample_ids is None:
            sample_ids = counts.columns.tolist()

        results = []

        # Pan-Cancer 분류
        if self.pancancer_classifier:
            pancancer_results = self.pancancer_classifier.predict(counts, sample_ids, top_k)
        else:
            pancancer_results = None

        for i, sample_id in enumerate(sample_ids):
            # Pan-Cancer 결과
            if pancancer_results:
                pc_result = pancancer_results[i]
                predicted_cancer = pc_result.predicted_cancer
                predicted_korean = pc_result.predicted_cancer_korean
                confidence = pc_result.confidence
                confidence_level = pc_result.confidence_level
                is_unknown = pc_result.is_unknown
                top_preds = pc_result.top_k_predictions
                ensemble_agreement = pc_result.ensemble_agreement
                warnings = pc_result.warnings.copy()
            else:
                predicted_cancer = "UNKNOWN"
                predicted_korean = "알 수 없음"
                confidence = 0.0
                confidence_level = "unknown"
                is_unknown = True
                top_preds = []
                ensemble_agreement = 0.0
                warnings = ["⚠️ Pan-Cancer 모델이 로드되지 않았습니다."]

            # Ensemble 모델 정보
            if self.pancancer_classifier and self.pancancer_classifier.ensemble:
                ensemble_models = list(self.pancancer_classifier.ensemble.models.keys())
            else:
                ensemble_models = []

            # Binary 세부 분석 (해당 암종 모델이 있는 경우)
            detailed_analysis = None
            gene_shap_info = []

            if include_binary_analysis and not is_unknown:
                # 암종 코드를 모델 키로 변환
                cancer_key = predicted_cancer.lower()

                # 특수 매핑
                cancer_model_map = {
                    'brca': 'breast',
                    'paad': 'pancreatic',
                }
                cancer_key = cancer_model_map.get(cancer_key, cancer_key)

                if cancer_key in self.binary_predictors:
                    try:
                        binary_predictor = self.binary_predictors[cancer_key]
                        sample_counts = counts[[sample_id]]
                        binary_result = binary_predictor.predict(sample_counts, [sample_id],
                                                                 explain=True, top_k_genes=top_genes)[0]

                        detailed_analysis = {
                            'binary_model': cancer_key,
                            'is_tumor': binary_result.prediction == 1,
                            'tumor_probability': binary_result.probability,
                            'binary_confidence': binary_result.confidence,
                        }

                        gene_shap_info = binary_result.top_genes

                    except Exception as e:
                        logger.warning(f"Binary analysis failed for {sample_id}: {e}")

            # 권장사항 생성
            recommendations = self._generate_recommendations(
                predicted_cancer, confidence_level, is_unknown, detailed_analysis
            )

            result = UnifiedPredictionResult(
                sample_id=sample_id,
                predicted_cancer=predicted_cancer,
                predicted_cancer_korean=predicted_korean,
                confidence=confidence,
                confidence_level=confidence_level,
                is_unknown=is_unknown,
                top_predictions=top_preds,
                ensemble_agreement=ensemble_agreement,
                ensemble_models=ensemble_models,
                detailed_analysis=detailed_analysis,
                top_genes=gene_shap_info,
                warnings=warnings,
                recommendations=recommendations,
            )

            results.append(result)

        return results

    def _generate_recommendations(self,
                                  cancer: str,
                                  confidence_level: str,
                                  is_unknown: bool,
                                  detailed_analysis: Optional[Dict]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        if is_unknown:
            recommendations.extend([
                "추가 샘플 분석을 통해 데이터 품질 확인",
                "다른 분석 방법 (예: 조직 병리학) 병행 권장",
                "샘플이 학습 데이터에 포함되지 않은 희귀 암종일 수 있음",
            ])
        else:
            if confidence_level == 'low':
                recommendations.append("낮은 신뢰도로 인해 추가 검증 권장")

            if confidence_level in ['low', 'medium']:
                recommendations.append(f"조직 병리학적 확인 권장")

            if detailed_analysis:
                if detailed_analysis.get('is_tumor'):
                    recommendations.append(f"{cancer} 특이적 마커 검사 고려")
                else:
                    recommendations.append("정상 조직 가능성, 종양 여부 재검토")

        recommendations.append("이 분석은 참고용이며 임상 진단을 대체할 수 없습니다")

        return recommendations

    def predict_single(self, counts: pd.Series,
                       sample_id: str = "sample") -> UnifiedPredictionResult:
        """단일 샘플 예측"""
        df = pd.DataFrame({sample_id: counts})
        results = self.predict(df, [sample_id])
        return results[0]

    def get_available_models(self) -> Dict[str, Any]:
        """사용 가능한 모델 정보"""
        info = {
            'pancancer': {
                'available': self.pancancer_classifier is not None,
                'path': str(self.pancancer_model_dir),
            },
            'binary_models': {},
        }

        for cancer_type in self.AVAILABLE_BINARY_MODELS:
            model_path = self.binary_models_dir / cancer_type
            info['binary_models'][cancer_type] = {
                'available': cancer_type in self.binary_predictors,
                'path': str(model_path),
            }

        return info

    def batch_predict(self, counts: pd.DataFrame,
                      output_path: Optional[str] = None) -> pd.DataFrame:
        """
        배치 예측 및 결과 저장

        Args:
            counts: Count matrix
            output_path: CSV 저장 경로

        Returns:
            예측 결과 DataFrame
        """
        results = self.predict(counts, include_binary_analysis=False)

        df = pd.DataFrame([{
            'sample_id': r.sample_id,
            'predicted_cancer': r.predicted_cancer,
            'predicted_cancer_korean': r.predicted_cancer_korean,
            'confidence': r.confidence,
            'confidence_level': r.confidence_level,
            'is_unknown': r.is_unknown,
            'ensemble_agreement': r.ensemble_agreement,
        } for r in results])

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")

        return df

    def generate_report(self, result: UnifiedPredictionResult) -> str:
        """상세 리포트 생성"""
        return result.summary()


def demo_prediction():
    """데모 예측"""
    logger.info("\n" + "="*60)
    logger.info("  Unified Predictor Demo")
    logger.info("="*60 + "\n")

    # 더미 데이터 생성
    np.random.seed(42)
    n_genes = 1000
    gene_ids = [f"ENSG{i:08d}" for i in range(n_genes)]

    sample_data = {
        'sample_1': np.random.poisson(100, n_genes),
        'sample_2': np.random.poisson(50, n_genes),
    }

    counts = pd.DataFrame(sample_data, index=gene_ids)

    logger.info(f"Demo data shape: {counts.shape}")
    logger.info("Note: Using random data for demonstration")
    logger.info("Pan-Cancer model needs to be trained first.\n")

    # 예측 시도
    predictor = UnifiedPredictor()

    try:
        predictor.load(load_binary=True)
        results = predictor.predict(counts)

        for result in results:
            print(result.summary())
            print()

    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}")
        logger.info("\nTo train the Pan-Cancer model:")
        logger.info("  1. python -m rnaseq_pipeline.ml.pancancer_downloader")
        logger.info("  2. python -m rnaseq_pipeline.ml.pancancer_classifier train -d data/tcga/pancancer")


if __name__ == "__main__":
    demo_prediction()
