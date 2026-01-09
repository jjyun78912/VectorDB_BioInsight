"""
RNA-seq ML Module
=================

CatBoost 기반 샘플 분류 및 SHAP 설명 모듈

Components:
- tcga_downloader: TCGA 데이터 다운로드
- preprocessor: 데이터 전처리 파이프라인
- trainer: CatBoost 모델 학습
- explainer: SHAP 기반 설명
- predictor: 예측 API
"""

from .tcga_downloader import TCGADownloader
from .preprocessor import RNAseqPreprocessor
from .trainer import CatBoostTrainer
from .explainer import SHAPExplainer
from .predictor import RNAseqPredictor

__all__ = [
    "TCGADownloader",
    "RNAseqPreprocessor",
    "CatBoostTrainer",
    "SHAPExplainer",
    "RNAseqPredictor",
]
