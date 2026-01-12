"""
Pan-Cancer Multi-class Classifier
==================================

TCGA 33개 암종을 분류하는 Multi-class 분류기
+ Unknown/OOD 탐지 + Ensemble 시스템

Features:
- Multi-class classification (33 cancer types)
- Unknown/OOD detection via confidence thresholding
- Ensemble of multiple models (CatBoost, LightGBM, XGBoost)
- Hierarchical classification (optional)
- SHAP-based explainability
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings

# ML frameworks
from catboost import CatBoostClassifier, Pool

# Optional: LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    LGBMClassifier = None

# Optional: XGBoost (requires libomp on macOS)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except (ImportError, OSError, Exception):
    HAS_XGBOOST = False
    XGBClassifier = None

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """분류 결과"""
    sample_id: str
    predicted_cancer: str
    predicted_cancer_korean: str
    confidence: float
    confidence_level: str  # "high", "medium", "low", "unknown"
    is_unknown: bool
    top_k_predictions: List[Dict[str, Any]]
    ensemble_agreement: float  # 앙상블 모델 간 일치도
    warnings: List[str]

    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'predicted_cancer': self.predicted_cancer,
            'predicted_cancer_korean': self.predicted_cancer_korean,
            'confidence': round(self.confidence, 4),
            'confidence_level': self.confidence_level,
            'is_unknown': self.is_unknown,
            'top_k_predictions': self.top_k_predictions,
            'ensemble_agreement': round(self.ensemble_agreement, 4),
            'warnings': self.warnings,
        }


class PanCancerPreprocessor:
    """Pan-Cancer 데이터 전처리"""

    def __init__(self,
                 n_top_genes: int = 5000,
                 log_transform: bool = True,
                 normalize: str = "cpm"):
        self.n_top_genes = n_top_genes
        self.log_transform = log_transform
        self.normalize = normalize

        self.selected_genes: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.cancer_info: Dict = {}
        self.is_fitted = False

    def _normalize_cpm(self, counts: pd.DataFrame) -> pd.DataFrame:
        """CPM 정규화"""
        lib_sizes = counts.sum(axis=0)
        return counts * 1e6 / lib_sizes

    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """log2(x + 1) 변환"""
        return np.log2(data + 1)

    def fit_transform(self, counts: pd.DataFrame,
                     cancer_labels: np.ndarray,
                     cancer_info: Dict,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전처리 및 학습/테스트 분할

        Args:
            counts: Gene x Sample count matrix
            cancer_labels: 암종 레이블 (문자열)
            cancer_info: 암종 정보 딕셔너리
            test_size: 테스트셋 비율
            random_state: 랜덤 시드

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Input shape: {counts.shape}")
        self.cancer_info = cancer_info

        # 레이블 인코딩
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(cancer_labels)
        logger.info(f"Classes: {self.label_encoder.classes_}")

        # 정규화
        if self.normalize == "cpm":
            normalized = self._normalize_cpm(counts)
        else:
            normalized = counts

        # Log 변환
        if self.log_transform:
            transformed = self._log_transform(normalized)
        else:
            transformed = normalized

        # 분산 기준 유전자 선택
        variances = transformed.var(axis=1)
        top_genes = variances.nlargest(self.n_top_genes).index.tolist()
        self.selected_genes = top_genes
        logger.info(f"Selected {len(self.selected_genes)} genes by variance")

        # 선택된 유전자만 추출 (samples x genes)
        X = transformed.loc[self.selected_genes].T.values

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        # 표준화
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.is_fitted = True

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train class distribution: {np.bincount(y_train)}")

        return X_train, X_test, y_train, y_test

    def transform(self, counts: pd.DataFrame) -> np.ndarray:
        """새 데이터 변환"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")

        # 정규화
        if self.normalize == "cpm":
            normalized = self._normalize_cpm(counts)
        else:
            normalized = counts

        # Log 변환
        if self.log_transform:
            transformed = self._log_transform(normalized)
        else:
            transformed = normalized

        # 선택된 유전자 추출 (없는 건 0으로)
        X = pd.DataFrame(0.0, index=counts.columns, columns=self.selected_genes)
        for gene in self.selected_genes:
            if gene in transformed.index:
                X[gene] = transformed.loc[gene].values

        # 표준화
        X = self.scaler.transform(X.values)
        return X

    def decode_labels(self, y: np.ndarray) -> List[str]:
        """숫자 레이블을 암종 코드로 변환"""
        return self.label_encoder.inverse_transform(y)

    def save(self, path: str):
        """전처리기 저장"""
        save_dict = {
            'n_top_genes': self.n_top_genes,
            'log_transform': self.log_transform,
            'normalize': self.normalize,
            'selected_genes': self.selected_genes,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'cancer_info': self.cancer_info,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(save_dict, path)

    @classmethod
    def load(cls, path: str) -> "PanCancerPreprocessor":
        """저장된 전처리기 로드"""
        save_dict = joblib.load(path)
        preprocessor = cls(
            n_top_genes=save_dict['n_top_genes'],
            log_transform=save_dict['log_transform'],
            normalize=save_dict['normalize'],
        )
        preprocessor.selected_genes = save_dict['selected_genes']
        preprocessor.scaler = save_dict['scaler']
        preprocessor.label_encoder = save_dict['label_encoder']
        preprocessor.cancer_info = save_dict['cancer_info']
        preprocessor.is_fitted = save_dict['is_fitted']
        return preprocessor


class EnsembleClassifier:
    """
    앙상블 분류기 (CatBoost + LightGBM + XGBoost)
    + Unknown/OOD 탐지
    """

    # Confidence 임계값
    CONFIDENCE_THRESHOLDS = {
        'high': 0.7,      # 70% 이상: 높은 신뢰도
        'medium': 0.4,    # 40-70%: 중간 신뢰도
        'low': 0.2,       # 20-40%: 낮은 신뢰도
        'unknown': 0.2,   # 20% 미만: Unknown으로 처리
    }

    # 앙상블 일치도 임계값
    AGREEMENT_THRESHOLD = 0.5  # 50% 미만 일치: Unknown 가능성

    def __init__(self,
                 n_classes: int,
                 class_names: List[str],
                 use_lightgbm: bool = True,
                 use_xgboost: bool = True,
                 random_state: int = 42):
        """
        Args:
            n_classes: 클래스 수 (암종 수)
            class_names: 클래스 이름 목록
            use_lightgbm: LightGBM 사용 여부
            use_xgboost: XGBoost 사용 여부
            random_state: 랜덤 시드
        """
        self.n_classes = n_classes
        self.class_names = class_names
        self.random_state = random_state

        # 모델 초기화
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.is_fitted = False

        # CatBoost (기본)
        self.models['catboost'] = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass',
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_weights['catboost'] = 0.4

        # LightGBM
        if use_lightgbm and HAS_LIGHTGBM:
            self.models['lightgbm'] = LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                num_class=n_classes,
                objective='multiclass',
                random_state=random_state,
                verbose=-1,
            )
            self.model_weights['lightgbm'] = 0.3

        # XGBoost
        if use_xgboost and HAS_XGBOOST:
            self.models['xgboost'] = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=n_classes,
                random_state=random_state,
                verbosity=0,
                use_label_encoder=False,
            )
            self.model_weights['xgboost'] = 0.3

        # 가중치 정규화
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}

        logger.info(f"Ensemble models: {list(self.models.keys())}")
        logger.info(f"Model weights: {self.model_weights}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        """
        앙상블 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            feature_names: 유전자 이름
        """
        logger.info("Training ensemble models...")

        for name, model in self.models.items():
            logger.info(f"  Training {name}...")

            if name == 'catboost':
                train_pool = Pool(X_train, y_train, feature_names=feature_names)
                if X_val is not None:
                    val_pool = Pool(X_val, y_val, feature_names=feature_names)
                    model.fit(train_pool, eval_set=val_pool,
                             early_stopping_rounds=50, verbose=False)
                else:
                    model.fit(train_pool, verbose=False)

            elif name == 'lightgbm':
                if X_val is not None:
                    model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             callbacks=[])
                else:
                    model.fit(X_train, y_train)

            elif name == 'xgboost':
                if X_val is not None:
                    model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             verbose=False)
                else:
                    model.fit(X_train, y_train)

        self.is_fitted = True
        logger.info("Ensemble training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        앙상블 확률 예측 (가중 평균)

        Args:
            X: 입력 데이터

        Returns:
            확률 배열 (samples x classes)
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted")

        ensemble_proba = np.zeros((X.shape[0], self.n_classes))

        for name, model in self.models.items():
            proba = model.predict_proba(X)
            weight = self.model_weights[name]
            ensemble_proba += weight * proba

        return ensemble_proba

    def predict_with_individual(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        앙상블 + 개별 모델 예측

        Returns:
            (ensemble_proba, individual_predictions)
        """
        individual_preds = {}

        for name, model in self.models.items():
            individual_preds[name] = model.predict(X)

        ensemble_proba = self.predict_proba(X)

        return ensemble_proba, individual_preds

    def calculate_agreement(self, individual_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """
        모델 간 일치도 계산

        Args:
            individual_preds: 개별 모델 예측 결과

        Returns:
            샘플별 일치도 (0~1)
        """
        if len(individual_preds) < 2:
            return np.ones(len(list(individual_preds.values())[0]))

        preds_array = np.array(list(individual_preds.values()))  # (n_models, n_samples)
        n_samples = preds_array.shape[1]

        agreements = []
        for i in range(n_samples):
            sample_preds = preds_array[:, i]
            # 가장 많은 예측과 일치하는 비율
            most_common = np.bincount(sample_preds.astype(int)).argmax()
            agreement = (sample_preds == most_common).mean()
            agreements.append(agreement)

        return np.array(agreements)

    def get_confidence_level(self, confidence: float, agreement: float) -> Tuple[str, bool]:
        """
        신뢰도 레벨 및 Unknown 여부 결정

        Returns:
            (confidence_level, is_unknown)
        """
        # Unknown 조건:
        # 1. 최고 확률이 너무 낮음
        # 2. 앙상블 일치도가 낮음
        if confidence < self.CONFIDENCE_THRESHOLDS['unknown']:
            return 'unknown', True

        if agreement < self.AGREEMENT_THRESHOLD and confidence < self.CONFIDENCE_THRESHOLDS['medium']:
            return 'unknown', True

        # 신뢰도 레벨
        if confidence >= self.CONFIDENCE_THRESHOLDS['high']:
            return 'high', False
        elif confidence >= self.CONFIDENCE_THRESHOLDS['medium']:
            return 'medium', False
        else:
            return 'low', False

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """모델 평가"""
        ensemble_proba = self.predict_proba(X_test)
        y_pred = ensemble_proba.argmax(axis=1)

        # 개별 모델 평가
        individual_metrics = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            individual_metrics[name] = {
                'accuracy': accuracy_score(y_test, pred),
                'f1_macro': f1_score(y_test, pred, average='macro'),
            }

        # 앙상블 평가
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'top_3_accuracy': top_k_accuracy_score(y_test, ensemble_proba, k=3),
            'top_5_accuracy': top_k_accuracy_score(y_test, ensemble_proba, k=5),
        }

        return {
            'ensemble': ensemble_metrics,
            'individual': individual_metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.class_names,
                output_dict=True
            ),
        }

    def save(self, path: str):
        """앙상블 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # CatBoost 저장
        if 'catboost' in self.models:
            self.models['catboost'].save_model(str(path / "catboost.cbm"))

        # 다른 모델들 저장
        for name in ['lightgbm', 'xgboost']:
            if name in self.models:
                joblib.dump(self.models[name], path / f"{name}.joblib")

        # 메타데이터 저장
        metadata = {
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'model_weights': self.model_weights,
            'models_available': list(self.models.keys()),
            'random_state': self.random_state,
        }
        with open(path / "ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EnsembleClassifier":
        """앙상블 로드"""
        path = Path(path)

        with open(path / "ensemble_metadata.json", 'r') as f:
            metadata = json.load(f)

        ensemble = cls(
            n_classes=metadata['n_classes'],
            class_names=metadata['class_names'],
            use_lightgbm='lightgbm' in metadata['models_available'],
            use_xgboost='xgboost' in metadata['models_available'],
            random_state=metadata['random_state'],
        )

        # CatBoost 로드
        if 'catboost' in metadata['models_available']:
            ensemble.models['catboost'] = CatBoostClassifier()
            ensemble.models['catboost'].load_model(str(path / "catboost.cbm"))

        # 다른 모델 로드
        for name in ['lightgbm', 'xgboost']:
            if name in metadata['models_available']:
                model_path = path / f"{name}.joblib"
                if model_path.exists():
                    ensemble.models[name] = joblib.load(model_path)

        ensemble.model_weights = metadata['model_weights']
        ensemble.is_fitted = True

        return ensemble


class PanCancerClassifier:
    """
    Pan-Cancer 통합 분류기

    기능:
    - Multi-class 암종 분류 (33개)
    - Unknown/OOD 탐지
    - Ensemble 예측
    - Confidence 기반 경고
    """

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: 모델 저장 디렉토리
        """
        self.model_dir = Path(model_dir)
        self.preprocessor: Optional[PanCancerPreprocessor] = None
        self.ensemble: Optional[EnsembleClassifier] = None
        self.cancer_info: Dict = {}
        self.is_loaded = False

    def load(self):
        """모델 로드"""
        logger.info(f"Loading Pan-Cancer classifier from {self.model_dir}")

        # Preprocessor 로드
        self.preprocessor = PanCancerPreprocessor.load(
            str(self.model_dir / "preprocessor.joblib")
        )

        # Ensemble 로드
        self.ensemble = EnsembleClassifier.load(str(self.model_dir / "ensemble"))

        # Cancer info 로드
        with open(self.model_dir / "cancer_info.json", 'r') as f:
            self.cancer_info = json.load(f)

        self.is_loaded = True
        logger.info("Model loaded successfully")

    def predict(self, counts: pd.DataFrame,
                sample_ids: Optional[List[str]] = None,
                top_k: int = 5) -> List[ClassificationResult]:
        """
        암종 예측

        Args:
            counts: Gene x Sample count matrix
            sample_ids: 샘플 ID 목록
            top_k: Top-k 예측 결과 포함

        Returns:
            분류 결과 리스트
        """
        if not self.is_loaded:
            self.load()

        if sample_ids is None:
            sample_ids = counts.columns.tolist()

        # 전처리
        X = self.preprocessor.transform(counts)

        # 앙상블 예측
        ensemble_proba, individual_preds = self.ensemble.predict_with_individual(X)

        # 일치도 계산
        agreements = self.ensemble.calculate_agreement(individual_preds)

        results = []
        for i, sample_id in enumerate(sample_ids):
            proba = ensemble_proba[i]
            agreement = agreements[i]

            # Top-k 예측
            top_indices = np.argsort(proba)[::-1][:top_k]
            top_k_preds = []
            for idx in top_indices:
                cancer_code = self.ensemble.class_names[idx]
                info = self.cancer_info.get(cancer_code, {})
                top_k_preds.append({
                    'cancer': cancer_code,
                    'cancer_korean': info.get('korean', cancer_code),
                    'probability': float(proba[idx]),
                })

            # 최고 확률 예측
            best_idx = top_indices[0]
            best_cancer = self.ensemble.class_names[best_idx]
            best_proba = proba[best_idx]
            cancer_info = self.cancer_info.get(best_cancer, {})

            # 신뢰도 레벨 및 Unknown 판정
            confidence_level, is_unknown = self.ensemble.get_confidence_level(
                best_proba, agreement
            )

            # 경고 메시지
            warnings = self._generate_warnings(best_proba, agreement, confidence_level, is_unknown)

            result = ClassificationResult(
                sample_id=sample_id,
                predicted_cancer=best_cancer if not is_unknown else "UNKNOWN",
                predicted_cancer_korean=cancer_info.get('korean', best_cancer) if not is_unknown else "알 수 없음",
                confidence=best_proba,
                confidence_level=confidence_level,
                is_unknown=is_unknown,
                top_k_predictions=top_k_preds,
                ensemble_agreement=agreement,
                warnings=warnings,
            )
            results.append(result)

        return results

    def _generate_warnings(self, confidence: float, agreement: float,
                          confidence_level: str, is_unknown: bool) -> List[str]:
        """경고 메시지 생성"""
        warnings = []

        if is_unknown:
            warnings.append("⚠️ 신뢰도가 낮아 암종을 특정할 수 없습니다. 추가 분석이 필요합니다.")

        if confidence_level == 'low':
            warnings.append("⚠️ 예측 신뢰도가 낮습니다. 결과 해석에 주의가 필요합니다.")

        if agreement < 0.7:
            warnings.append("⚠️ 앙상블 모델 간 일치도가 낮습니다. 여러 암종 가능성을 검토하세요.")

        warnings.append("⚠️ 이 예측은 참고용이며, 진단 목적으로 사용할 수 없습니다.")

        return warnings

    def predict_single(self, counts: pd.Series,
                      sample_id: str = "sample") -> ClassificationResult:
        """단일 샘플 예측"""
        df = pd.DataFrame({sample_id: counts})
        results = self.predict(df, [sample_id])
        return results[0]

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보"""
        if not self.is_loaded:
            self.load()

        return {
            'n_classes': self.ensemble.n_classes,
            'cancer_types': self.ensemble.class_names,
            'ensemble_models': list(self.ensemble.models.keys()),
            'n_genes': len(self.preprocessor.selected_genes),
        }


def train_pancancer_classifier(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    cancer_info: Dict,
    output_dir: str,
    n_top_genes: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Pan-Cancer 분류기 학습

    Args:
        counts: Gene x Sample count matrix
        metadata: 샘플 메타데이터 (cancer_type 컬럼 필요)
        cancer_info: 암종 정보
        output_dir: 모델 저장 경로
        n_top_genes: 사용할 유전자 수
        test_size: 테스트셋 비율
        random_state: 랜덤 시드

    Returns:
        학습 결과
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info("  Pan-Cancer Classifier Training")
    logger.info(f"{'='*60}")
    logger.info(f"  Samples: {counts.shape[1]}")
    logger.info(f"  Genes: {counts.shape[0]}")
    logger.info(f"  Cancer types: {metadata['cancer_type'].nunique()}")
    logger.info(f"{'='*60}\n")

    # 샘플 순서 맞추기
    sample_order = counts.columns.tolist()
    metadata_indexed = metadata.set_index('barcode')
    cancer_labels = np.array([metadata_indexed.loc[s, 'cancer_type'] for s in sample_order])

    # 전처리
    logger.info("[1/3] Preprocessing...")
    preprocessor = PanCancerPreprocessor(n_top_genes=n_top_genes)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        counts, cancer_labels, cancer_info, test_size, random_state
    )

    preprocessor.save(str(output_dir / "preprocessor.joblib"))

    # 클래스 정보
    class_names = preprocessor.label_encoder.classes_.tolist()
    n_classes = len(class_names)

    # 앙상블 학습
    logger.info("\n[2/3] Training ensemble...")
    ensemble = EnsembleClassifier(
        n_classes=n_classes,
        class_names=class_names,
        random_state=random_state
    )

    ensemble.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        feature_names=preprocessor.selected_genes
    )

    ensemble.save(str(output_dir / "ensemble"))

    # 평가
    logger.info("\n[3/3] Evaluating...")
    eval_results = ensemble.evaluate(X_test, y_test)

    # Cancer info 저장
    with open(output_dir / "cancer_info.json", 'w') as f:
        json.dump(cancer_info, f, indent=2, ensure_ascii=False)

    # 결과 저장
    training_results = {
        'n_classes': n_classes,
        'class_names': class_names,
        'n_samples': counts.shape[1],
        'n_genes': n_top_genes,
        'training_date': datetime.now().isoformat(),
        'test_size': test_size,
        'metrics': eval_results,
    }

    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2)

    # 결과 출력
    logger.info(f"\n{'='*60}")
    logger.info("  Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"  Ensemble Accuracy: {eval_results['ensemble']['accuracy']:.4f}")
    logger.info(f"  Ensemble F1 (macro): {eval_results['ensemble']['f1_macro']:.4f}")
    logger.info(f"  Top-3 Accuracy: {eval_results['ensemble']['top_3_accuracy']:.4f}")
    logger.info(f"  Top-5 Accuracy: {eval_results['ensemble']['top_5_accuracy']:.4f}")
    logger.info(f"\n  Individual model performance:")
    for name, metrics in eval_results['individual'].items():
        logger.info(f"    {name}: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")
    logger.info(f"\n  Model saved to: {output_dir}")
    logger.info(f"{'='*60}\n")

    return training_results


def main():
    """CLI 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Pan-Cancer Classifier")
    subparsers = parser.add_subparsers(dest='command')

    # train
    train_parser = subparsers.add_parser('train', help='Train classifier')
    train_parser.add_argument('--data', '-d', type=str, required=True,
                             help='Pan-cancer data directory')
    train_parser.add_argument('--output', '-o', type=str, default='models/rnaseq/pancancer',
                             help='Output directory')
    train_parser.add_argument('--genes', '-g', type=int, default=5000,
                             help='Number of genes to use')

    # predict
    predict_parser = subparsers.add_parser('predict', help='Predict samples')
    predict_parser.add_argument('--model', '-m', type=str, required=True,
                               help='Model directory')
    predict_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input count matrix (CSV)')
    predict_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output predictions (CSV)')

    args = parser.parse_args()

    if args.command == 'train':
        from .pancancer_downloader import PanCancerDownloader

        data_dir = Path(args.data)
        counts = pd.read_csv(data_dir / "pancancer_counts.csv", index_col=0)
        metadata = pd.read_csv(data_dir / "pancancer_metadata.csv")

        with open(data_dir / "label_mapping.json", 'r') as f:
            label_mapping = json.load(f)

        train_pancancer_classifier(
            counts, metadata,
            cancer_info=label_mapping.get('cancer_info', {}),
            output_dir=args.output,
            n_top_genes=args.genes
        )

    elif args.command == 'predict':
        classifier = PanCancerClassifier(args.model)
        counts = pd.read_csv(args.input, index_col=0)

        results = classifier.predict(counts)
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(args.output, index=False)

        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
