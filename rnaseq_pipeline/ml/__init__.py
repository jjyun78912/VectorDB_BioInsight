"""
RNA-seq ML Module
=================

Multi-class Pan-Cancer 분류 + Binary 분류 + Unknown 탐지 + Ensemble

Components:
- tcga_downloader: TCGA 단일 암종 데이터 다운로드
- pancancer_downloader: TCGA 33개 암종 Pan-Cancer 데이터 다운로드
- preprocessor: 데이터 전처리 파이프라인
- trainer: CatBoost 모델 학습
- explainer: SHAP 기반 설명
- predictor: 암종별 Binary 예측 API
- pancancer_classifier: Pan-Cancer Multi-class 분류기 + Unknown 탐지
- unified_predictor: 통합 예측 API (Pan-Cancer + Binary + Ensemble)

Usage:
```python
from rnaseq_pipeline.ml import UnifiedPredictor

# 통합 예측기 (권장)
predictor = UnifiedPredictor()
result = predictor.predict(counts_df)

print(result.predicted_cancer)      # "BRCA" or "UNKNOWN"
print(result.confidence)            # 0.87
print(result.is_unknown)            # False
print(result.top_predictions)       # [{"cancer": "BRCA", ...}, ...]
```
"""

# Binary Classification (기존)
from .tcga_downloader import TCGADownloader
from .preprocessor import RNAseqPreprocessor
from .trainer import CatBoostTrainer
from .explainer import SHAPExplainer
from .predictor import RNAseqPredictor

# Pan-Cancer Multi-class Classification (신규)
from .pancancer_downloader import PanCancerDownloader
from .pancancer_classifier import (
    PanCancerClassifier,
    PanCancerPreprocessor,
    EnsembleClassifier,
    ClassificationResult,
    train_pancancer_classifier,
)

# Unified Predictor (통합)
from .unified_predictor import UnifiedPredictor, UnifiedPredictionResult

__all__ = [
    # Binary Classification
    "TCGADownloader",
    "RNAseqPreprocessor",
    "CatBoostTrainer",
    "SHAPExplainer",
    "RNAseqPredictor",

    # Pan-Cancer Multi-class
    "PanCancerDownloader",
    "PanCancerClassifier",
    "PanCancerPreprocessor",
    "EnsembleClassifier",
    "ClassificationResult",
    "train_pancancer_classifier",

    # Unified
    "UnifiedPredictor",
    "UnifiedPredictionResult",
]
