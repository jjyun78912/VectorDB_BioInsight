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


# ═══════════════════════════════════════════════════════════════════════════════
# 혼동 가능 암종 쌍 및 조직 특이 마커 정의
# ═══════════════════════════════════════════════════════════════════════════════

# 생물학적으로 유사하여 혼동되기 쉬운 암종 쌍
CONFUSABLE_CANCER_PAIRS = {
    # ★★★ 폐암 (가장 중요 - LUAD vs LUSC 구분 어려움)
    frozenset({"LUAD", "LUSC"}): "lung_cancer",        # 폐선암 vs 폐편평 (같은 장기, 다른 조직형)
    # 편평상피암(SCC) 계열 - 같은 조직학적 기원
    frozenset({"HNSC", "LUSC"}): "squamous_cell",      # 두경부 vs 폐편평
    frozenset({"HNSC", "SKCM"}): "skin_mucosal",       # 두경부 vs 흑색종 (점막 흑색종)
    frozenset({"LUSC", "BLCA"}): "squamous_cell",      # 폐편평 vs 방광 (일부 SCC)
    frozenset({"SKCM", "HNSC"}): "skin_mucosal",       # 흑색종 vs 두경부
    # 선암(Adenocarcinoma) 계열
    frozenset({"LUAD", "PAAD"}): "adenocarcinoma",     # 폐선암 vs 췌장암
    frozenset({"LUAD", "BLCA"}): "adenocarcinoma",     # 폐선암 vs 방광 (일부 선암)
    frozenset({"COAD", "STAD"}): "gi_adenocarcinoma",  # 대장 vs 위
    # 부인과 암종
    frozenset({"OV", "UCEC"}): "gynecologic",          # 난소 vs 자궁
}

# 조직 특이적 마커 유전자 (ENSEMBL ID → Symbol 매핑 포함)
TISSUE_SPECIFIC_MARKERS = {
    # ★★★ 폐선암 vs 폐편평 구분 마커 (강화됨)
    "LUAD": {
        # TTF-1 (NKX2-1) = 폐선암의 gold standard 마커
        # NAPSA (Napsin A) = 폐선암 특이
        # Surfactant proteins = 폐포 II형 세포 기원
        # MUC1 = 선암 뮤신
        "markers": ["NKX2-1", "TTF1", "NAPSA", "SFTPC", "SFTPB", "SFTPA1", "MUC1", "CK7", "KRT7"],
        "negative_markers": ["TP63", "KRT5", "KRT6A", "SOX2"],  # 음성이어야 함
        "description": "폐선암 특이 전사인자 및 계면활성제 (TTF-1+, Napsin A+, p63-)",
    },
    "LUSC": {
        # p63 (TP63) = 편평상피암 gold standard
        # SOX2 = 편평상피 분화
        # High-molecular-weight keratins (KRT5, KRT6) = 편평상피 특이
        # p40 (ΔNp63) = 편평상피암 특이
        "markers": ["TP63", "SOX2", "KRT5", "KRT6A", "KRT14", "DSG3", "PKP1"],
        "negative_markers": ["NKX2-1", "TTF1", "NAPSA", "SFTPC"],  # 음성이어야 함
        "description": "폐편평상피암 각질화 마커 (p63+, CK5/6+, TTF-1-)",
    },
    # 두경부 특이 마커
    "HNSC": {
        "markers": ["TP63", "SOX2", "KRT5", "CDKN2A", "HPV16"],  # HPV 관련
        "description": "두경부 편평상피암 마커",
    },
    # 흑색종 특이 마커
    "SKCM": {
        "markers": ["MLANA", "MITF", "TYR", "S100B", "PMEL"],  # Melan-A, MITF
        "description": "멜라닌 생성 경로 마커",
    },
    # 췌장 특이 마커
    "PAAD": {
        "markers": ["PDX1", "KRAS", "SMAD4", "CDKN2A", "MUC1"],
        "description": "췌장 분화 및 드라이버 마커",
    },
    # 대장 특이 마커
    "COAD": {
        "markers": ["CDX2", "MUC2", "SATB2", "CK20"],
        "description": "대장 분화 마커",
    },
    # 위 특이 마커
    "STAD": {
        "markers": ["MUC5AC", "MUC6", "CDH1", "CLDN18"],
        "description": "위 뮤신 및 접착 분자",
    },
    # 난소 특이 마커
    "OV": {
        "markers": ["PAX8", "WT1", "CA125", "HE4"],
        "description": "난소암 전사인자 및 마커",
    },
    # 자궁 특이 마커
    "UCEC": {
        "markers": ["PAX8", "ER", "PR", "PTEN", "MSH6"],
        "description": "자궁내막 호르몬 수용체",
    },
    # 방광 특이 마커
    "BLCA": {
        "markers": ["GATA3", "UPKB", "KRT20", "FGFR3"],
        "description": "요로상피 마커",
    },
}

# 신뢰도 갭 임계값 (Top1 - Top2 < 이 값이면 혼동 가능)
CONFIDENCE_GAP_THRESHOLD = 0.15  # 15% 미만 차이면 불확실


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
    # 2차 검증 결과 (v2 추가)
    secondary_validation: Optional[Dict[str, Any]] = None
    confidence_gap: float = 0.0  # Top1 - Top2 신뢰도 차이
    is_confusable_pair: bool = False  # 혼동 가능 암종 쌍 여부

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
            'secondary_validation': self.secondary_validation,
            'confidence_gap': round(self.confidence_gap, 4),
            'is_confusable_pair': self.is_confusable_pair,
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

        # Gene ID 변환 매핑
        self.symbol_to_ensembl: Optional[Dict[str, str]] = None
        self.ensembl_to_symbol: Optional[Dict[str, str]] = None

    def _load_gene_mapping(self, model_dir: Path):
        """Gene Symbol <-> ENSEMBL ID 매핑 로드"""
        mapping_file = model_dir / 'symbol_to_model_ensembl.json'
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                self.symbol_to_ensembl = json.load(f)
            self.ensembl_to_symbol = {v: k for k, v in self.symbol_to_ensembl.items()}
            logger.info(f"Loaded gene mapping: {len(self.symbol_to_ensembl)} symbols")
        else:
            logger.warning(f"Gene mapping file not found: {mapping_file}")

    def _convert_gene_ids(self, counts: pd.DataFrame) -> pd.DataFrame:
        """
        Gene ID 변환 → 모델이 기대하는 ENSEMBL ID로 변환

        지원 형식:
        - Entrez ID (숫자): mygene API로 ENSEMBL ID 변환
        - Gene Symbol: symbol_to_ensembl 매핑으로 변환
        - ENSEMBL ID: 그대로 사용
        """
        if self.symbol_to_ensembl is None:
            return counts

        # 입력 데이터의 유전자 ID 형식 감지 (더 많은 샘플로 정확히 판단)
        sample_genes = counts.index[:100].tolist() if len(counts.index) >= 100 else counts.index.tolist()

        # Count each type
        ensembl_count = sum(1 for g in sample_genes if str(g).startswith('ENSG'))
        numeric_count = sum(1 for g in sample_genes if str(g).isdigit())
        symbol_count = len(sample_genes) - ensembl_count - numeric_count

        # Determine format by majority (allow mixed data)
        total = len(sample_genes)
        is_ensembl = ensembl_count / total > 0.7  # >70% must be ENSEMBL to be considered ENSEMBL format
        is_entrez = numeric_count / total > 0.7  # >70% must be numeric for Entrez

        logger.info(f"Gene ID format detection: ENSEMBL={ensembl_count}, Entrez={numeric_count}, Symbol={symbol_count} (of {total})")

        if is_entrez:
            # Entrez ID -> ENSEMBL ID 변환
            logger.info("Input is in Entrez ID format, converting to ENSEMBL ID...")

            entrez_ids = [str(g) for g in counts.index if str(g).isdigit()]
            entrez_to_ensembl = {}

            # 1. 먼저 로컬 캐시 확인
            cache_file = Path(__file__).parent.parent.parent / "models" / "rnaseq" / "pancancer" / "entrez_to_ensembl_cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_mapping = json.load(f)
                    # 캐시된 매핑 사용
                    for entrez_id in entrez_ids:
                        if entrez_id in cached_mapping:
                            entrez_to_ensembl[entrez_id] = cached_mapping[entrez_id]
                    logger.info(f"Loaded {len(entrez_to_ensembl)} mappings from local cache")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")

            # 2. 캐시에 없는 ID는 mygene API로 조회
            missing_ids = [eid for eid in entrez_ids if eid not in entrez_to_ensembl]
            if missing_ids and len(missing_ids) > 0:
                try:
                    import mygene
                    mg = mygene.MyGeneInfo()

                    batch_size = 1000
                    new_mappings = {}

                    for i in range(0, len(missing_ids), batch_size):
                        batch = missing_ids[i:i+batch_size]
                        results = mg.querymany(batch, scopes='entrezgene',
                                             fields='ensembl.gene,symbol',
                                             species='human', verbose=False)
                        for r in results:
                            if 'query' not in r:
                                continue
                            query = r['query']

                            # ensembl ID가 있으면 사용
                            if 'ensembl' in r:
                                ensembl_data = r['ensembl']
                                if isinstance(ensembl_data, list):
                                    ensembl_id = ensembl_data[0].get('gene') if ensembl_data else None
                                else:
                                    ensembl_id = ensembl_data.get('gene')
                                if ensembl_id:
                                    # 모델의 ENSEMBL ID 형식과 매칭 (버전 포함)
                                    for model_ensembl in self.symbol_to_ensembl.values():
                                        if model_ensembl.startswith(ensembl_id):
                                            new_mappings[query] = model_ensembl
                                            break
                                    else:
                                        new_mappings[query] = ensembl_id

                            # ensembl이 없으면 symbol을 통해 변환
                            if query not in new_mappings and 'symbol' in r:
                                symbol = r['symbol']
                                if symbol in self.symbol_to_ensembl:
                                    new_mappings[query] = self.symbol_to_ensembl[symbol]

                    entrez_to_ensembl.update(new_mappings)
                    logger.info(f"Fetched {len(new_mappings)} new mappings from mygene API")

                    # 캐시 업데이트
                    if new_mappings:
                        try:
                            existing_cache = {}
                            if cache_file.exists():
                                with open(cache_file, 'r') as f:
                                    existing_cache = json.load(f)
                            existing_cache.update(new_mappings)
                            with open(cache_file, 'w') as f:
                                json.dump(existing_cache, f)
                            logger.info(f"Updated cache with {len(new_mappings)} new entries")
                        except Exception as e:
                            logger.warning(f"Failed to update cache: {e}")

                except Exception as e:
                    logger.warning(f"mygene API failed: {e}, using cached data only")

            if entrez_to_ensembl:
                logger.info(f"Total Entrez → ENSEMBL mappings: {len(entrez_to_ensembl)}")

                # Convert counts
                converted_index = []
                converted_data = []
                conversion_stats = {'matched': 0, 'unmatched': 0}

                for gene in counts.index:
                    gene_str = str(gene)
                    if gene_str in entrez_to_ensembl:
                        ensembl_id = entrez_to_ensembl[gene_str]
                        converted_index.append(ensembl_id)
                        converted_data.append(counts.loc[gene].values)
                        conversion_stats['matched'] += 1
                    else:
                        conversion_stats['unmatched'] += 1

                logger.info(f"Gene ID conversion (Entrez → ENSEMBL): {conversion_stats['matched']} matched, "
                           f"{conversion_stats['unmatched']} unmatched")

                if converted_data:
                    converted_df = pd.DataFrame(
                        converted_data,
                        index=converted_index,
                        columns=counts.columns
                    )
                    return converted_df

        if is_ensembl:
            # ENSEMBL ID가 이미 사용중 - 그대로 반환
            logger.info("Input is already in ENSEMBL format")
            return counts

        # Gene Symbol -> ENSEMBL 변환
        logger.info("Input appears to be Gene Symbol, converting to ENSEMBL ID...")
        converted_index = []
        converted_data = []
        conversion_stats = {'matched': 0, 'unmatched': 0}

        for gene in counts.index:
            gene_str = str(gene)
            if gene_str in self.symbol_to_ensembl:
                ensembl_id = self.symbol_to_ensembl[gene_str]
                converted_index.append(ensembl_id)
                converted_data.append(counts.loc[gene].values)
                conversion_stats['matched'] += 1
            else:
                conversion_stats['unmatched'] += 1

        logger.info(f"Gene ID conversion (Symbol → ENSEMBL): {conversion_stats['matched']} matched, "
                   f"{conversion_stats['unmatched']} unmatched")

        if converted_data:
            converted_df = pd.DataFrame(
                converted_data,
                index=converted_index,
                columns=counts.columns
            )
            return converted_df
        else:
            logger.warning("No genes matched during conversion!")
            return counts

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

        # Gene ID 변환 (Symbol -> ENSEMBL)
        counts = self._convert_gene_ids(counts)

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

        # 중복 gene index 처리 (평균 사용)
        if transformed.index.duplicated().any():
            logger.info(f"Found {transformed.index.duplicated().sum()} duplicate gene indices, averaging...")
            transformed = transformed.groupby(transformed.index).mean()

        # ENSEMBL ID 버전 번호 제거하여 매칭 (ENSG00000186081.12 -> ENSG00000186081)
        # 입력 데이터의 버전 제거
        input_gene_map = {}  # base_id -> full_id
        for gene_id in transformed.index:
            if isinstance(gene_id, str) and gene_id.startswith('ENSG'):
                base_id = gene_id.split('.')[0]
                input_gene_map[base_id] = gene_id

        # 모델 selected_genes의 버전 제거 매핑
        model_gene_map = {}  # base_id -> versioned_id
        for gene_id in self.selected_genes:
            if isinstance(gene_id, str) and gene_id.startswith('ENSG'):
                base_id = gene_id.split('.')[0]
                model_gene_map[base_id] = gene_id

        # 선택된 유전자 추출 (없는 건 0으로)
        X = pd.DataFrame(0.0, index=counts.columns, columns=self.selected_genes)
        matched_genes = 0
        for gene in self.selected_genes:
            # 1. 정확히 일치하는 경우
            if gene in transformed.index:
                gene_values = transformed.loc[gene]
                if isinstance(gene_values, pd.Series):
                    X[gene] = gene_values.values
                else:
                    X[gene] = gene_values.iloc[0].values
                matched_genes += 1
            # 2. 버전 번호 제거 후 매칭 시도
            elif isinstance(gene, str) and gene.startswith('ENSG'):
                base_id = gene.split('.')[0]
                if base_id in input_gene_map:
                    input_gene = input_gene_map[base_id]
                    gene_values = transformed.loc[input_gene]
                    if isinstance(gene_values, pd.Series):
                        X[gene] = gene_values.values
                    else:
                        X[gene] = gene_values.iloc[0].values
                    matched_genes += 1

        logger.info(f"Feature matching: {matched_genes}/{len(self.selected_genes)} genes "
                   f"({matched_genes/len(self.selected_genes)*100:.1f}%)")

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

        # Gene mapping 로드 (모델 디렉토리에서)
        model_dir = Path(path).parent
        preprocessor._load_gene_mapping(model_dir)

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
                top_k: int = 5,
                use_secondary_validation: bool = True) -> List[ClassificationResult]:
        """
        암종 예측 (v2: 2차 검증 포함)

        Args:
            counts: Gene x Sample count matrix
            sample_ids: 샘플 ID 목록
            top_k: Top-k 예측 결과 포함
            use_secondary_validation: 혼동 가능 암종에 대한 2차 검증 수행

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
            cancer_info_dict = self.cancer_info.get(best_cancer, {})

            # Top2 확률 (신뢰도 갭 계산용)
            second_best_proba = proba[top_indices[1]] if len(top_indices) > 1 else 0.0
            confidence_gap = best_proba - second_best_proba
            second_best_cancer = self.ensemble.class_names[top_indices[1]] if len(top_indices) > 1 else None

            # 혼동 가능 암종 쌍 체크
            is_confusable = False
            confusable_type = None
            if second_best_cancer:
                pair = frozenset({best_cancer, second_best_cancer})
                if pair in CONFUSABLE_CANCER_PAIRS:
                    is_confusable = True
                    confusable_type = CONFUSABLE_CANCER_PAIRS[pair]

            # 2차 검증 수행 조건:
            # 1. 신뢰도 갭이 작음 (15% 미만) 또는
            # 2. 혼동 가능 암종 쌍이며 신뢰도가 낮음 (70% 미만)
            secondary_validation = None
            final_cancer = best_cancer

            needs_secondary = (
                use_secondary_validation and
                (confidence_gap < CONFIDENCE_GAP_THRESHOLD or
                 (is_confusable and best_proba < 0.70))
            )

            if needs_secondary and second_best_cancer:
                secondary_validation = self._perform_secondary_validation(
                    counts.iloc[:, i] if isinstance(counts.iloc[:, i], pd.Series) else counts[sample_id],
                    best_cancer,
                    second_best_cancer,
                    best_proba,
                    second_best_proba
                )

                # 2차 검증 결과로 예측 변경 여부 결정
                if secondary_validation and secondary_validation.get('corrected_prediction'):
                    final_cancer = secondary_validation['corrected_prediction']
                    cancer_info_dict = self.cancer_info.get(final_cancer, {})

            # 신뢰도 레벨 및 Unknown 판정
            confidence_level, is_unknown = self.ensemble.get_confidence_level(
                best_proba, agreement
            )

            # 경고 메시지
            warnings = self._generate_warnings(
                best_proba, agreement, confidence_level, is_unknown,
                confidence_gap, is_confusable, secondary_validation
            )

            result = ClassificationResult(
                sample_id=sample_id,
                predicted_cancer=final_cancer if not is_unknown else "UNKNOWN",
                predicted_cancer_korean=cancer_info_dict.get('korean', final_cancer) if not is_unknown else "알 수 없음",
                confidence=best_proba,
                confidence_level=confidence_level,
                is_unknown=is_unknown,
                top_k_predictions=top_k_preds,
                ensemble_agreement=agreement,
                warnings=warnings,
                secondary_validation=secondary_validation,
                confidence_gap=confidence_gap,
                is_confusable_pair=is_confusable,
            )
            results.append(result)

        return results

    def _perform_secondary_validation(self,
                                      sample_expr: pd.Series,
                                      cancer1: str,
                                      cancer2: str,
                                      prob1: float,
                                      prob2: float) -> Dict[str, Any]:
        """
        2차 검증: 조직 특이 마커 발현으로 혼동 암종 구분

        Args:
            sample_expr: 샘플 발현 데이터 (gene_id -> expression)
            cancer1: Top-1 예측 암종
            cancer2: Top-2 예측 암종
            prob1: Top-1 확률
            prob2: Top-2 확률

        Returns:
            2차 검증 결과
        """
        validation = {
            'cancer1': cancer1,
            'cancer2': cancer2,
            'prob1': prob1,
            'prob2': prob2,
            'marker_scores': {},
            'corrected_prediction': None,
            'correction_reason': None,
        }

        # 각 암종의 마커 점수 계산 (positive + negative markers)
        for cancer in [cancer1, cancer2]:
            if cancer in TISSUE_SPECIFIC_MARKERS:
                marker_info = TISSUE_SPECIFIC_MARKERS[cancer]
                markers = marker_info['markers']
                negative_markers = marker_info.get('negative_markers', [])
                score = self._calculate_marker_score(sample_expr, markers, negative_markers)
                validation['marker_scores'][cancer] = score

        # 마커 점수 비교하여 보정 여부 결정
        if len(validation['marker_scores']) == 2:
            score1 = validation['marker_scores'].get(cancer1, 0)
            score2 = validation['marker_scores'].get(cancer2, 0)

            # ★★★ LUAD-LUSC 전용 보정 로직 (더 엄격한 기준)
            is_lung_cancer_pair = frozenset({cancer1, cancer2}) == frozenset({"LUAD", "LUSC"})

            if is_lung_cancer_pair:
                # 폐암 쌍은 마커 점수 차이가 1.5배 이상이면 보정 (더 민감하게)
                prob_diff = prob1 - prob2
                if prob_diff < 0.25 and score2 > score1 * 1.5 and score2 > 0.2:
                    validation['corrected_prediction'] = cancer2
                    validation['correction_reason'] = (
                        f"폐암 마커 기반 보정: {cancer1}({score1:.2f}) → {cancer2}({score2:.2f})"
                    )
                    logger.info(f"Lung cancer secondary validation: {cancer1} → {cancer2} "
                               f"(marker scores: {score1:.2f} vs {score2:.2f})")
            else:
                # 일반 조건: prob 차이가 20% 미만이고 마커 점수 차이가 2배 이상이면 보정
                prob_diff = prob1 - prob2
                if prob_diff < 0.20 and score2 > score1 * 2 and score2 > 0.3:
                    validation['corrected_prediction'] = cancer2
                    validation['correction_reason'] = (
                        f"마커 점수 기반 보정: {cancer1}({score1:.2f}) → {cancer2}({score2:.2f})"
                    )
                    logger.info(f"Secondary validation: {cancer1} → {cancer2} "
                           f"(marker scores: {score1:.2f} vs {score2:.2f})")

        return validation

    def _calculate_marker_score(self, sample_expr: pd.Series, markers: List[str],
                                  negative_markers: List[str] = None) -> float:
        """
        마커 유전자 발현 점수 계산 (positive + negative markers 고려)

        높은 발현 = 높은 점수 (정규화된 값 기준)
        negative_markers가 높으면 점수 차감
        """
        if sample_expr is None or len(sample_expr) == 0:
            return 0.0

        def get_expression(marker):
            """마커의 발현값 조회"""
            # Symbol로 직접 찾기
            if marker in sample_expr.index:
                return sample_expr[marker]
            # ENSEMBL 매핑이 있으면 사용
            if hasattr(self, 'symbol_to_ensembl') and self.symbol_to_ensembl:
                ensembl = self.symbol_to_ensembl.get(marker)
                if ensembl and ensembl in sample_expr.index:
                    return sample_expr[ensembl]
            return None

        # Positive markers 점수 계산
        pos_expressions = []
        for marker in markers:
            expr = get_expression(marker)
            if expr is not None:
                pos_expressions.append(expr)

        if not pos_expressions:
            return 0.0

        # 평균 발현량을 0-1 점수로 정규화
        avg_pos_expr = np.mean(pos_expressions)
        pos_score = min(1.0, avg_pos_expr / 10.0) if avg_pos_expr > 0 else 0.0

        # Negative markers 점수 계산 (높으면 감점)
        neg_score = 0.0
        if negative_markers:
            neg_expressions = []
            for marker in negative_markers:
                expr = get_expression(marker)
                if expr is not None:
                    neg_expressions.append(expr)

            if neg_expressions:
                avg_neg_expr = np.mean(neg_expressions)
                # 음성 마커가 높으면 감점 (최대 0.5점 감점)
                neg_score = min(0.5, avg_neg_expr / 20.0) if avg_neg_expr > 0 else 0.0

        # 최종 점수 = positive - negative (최소 0)
        final_score = max(0.0, pos_score - neg_score)

        return final_score

    def _generate_warnings(self, confidence: float, agreement: float,
                          confidence_level: str, is_unknown: bool,
                          confidence_gap: float = 1.0,
                          is_confusable: bool = False,
                          secondary_validation: Optional[Dict[str, Any]] = None) -> List[str]:
        """경고 메시지 생성

        Args:
            confidence: 예측 신뢰도
            agreement: 앙상블 일치도
            confidence_level: 신뢰 수준 (high/medium/low)
            is_unknown: Unknown 예측 여부
            confidence_gap: 1위-2위 확률 차이
            is_confusable: 혼동 가능 암종 쌍 여부
            secondary_validation: 2차 검증 결과
        """
        warnings = []

        if is_unknown:
            warnings.append("⚠️ 신뢰도가 낮아 암종을 특정할 수 없습니다. 추가 분석이 필요합니다.")

        if confidence_level == 'low':
            warnings.append("⚠️ 예측 신뢰도가 낮습니다. 결과 해석에 주의가 필요합니다.")

        if agreement < 0.7:
            warnings.append("⚠️ 앙상블 모델 간 일치도가 낮습니다. 여러 암종 가능성을 검토하세요.")

        # 혼동 가능 암종 쌍에 대한 경고
        if is_confusable and confidence_gap < CONFIDENCE_GAP_THRESHOLD:
            warnings.append(f"⚠️ 1-2위 예측 확률 차이가 작습니다 ({confidence_gap:.1%}). "
                          "유사한 조직학적 특성의 암종 간 혼동 가능성이 있습니다.")

        # 2차 검증 결과에 따른 경고
        if secondary_validation:
            if secondary_validation.get('override_applied'):
                original = secondary_validation.get('original_prediction')
                final = secondary_validation.get('final_prediction')
                warnings.append(f"ℹ️ 조직 특이적 마커 분석으로 예측이 {original}에서 {final}로 "
                              "보정되었습니다. 병리 조직검사로 확인이 권장됩니다.")
            elif secondary_validation.get('validation_status') == 'ambiguous':
                warnings.append("⚠️ 마커 기반 2차 검증에서 명확한 결론을 얻지 못했습니다. "
                              "추가 면역조직화학 검사가 권장됩니다.")

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

    def explain_with_shap(self,
                         counts: pd.DataFrame,
                         sample_ids: Optional[List[str]] = None,
                         cancer_type: Optional[str] = None,
                         top_k_genes: int = 20,
                         save_plots: bool = False,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        SHAP 기반 예측 설명

        Args:
            counts: Gene x Sample count matrix
            sample_ids: 샘플 ID 목록
            cancer_type: 특정 암종에 대한 설명 (None이면 전체)
            top_k_genes: 상위 유전자 수
            save_plots: 플롯 저장 여부
            output_dir: 플롯 저장 경로

        Returns:
            SHAP 분석 결과
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Run: pip install shap")
            return {'error': 'SHAP not installed'}

        if not self.is_loaded:
            self.load()

        if sample_ids is None:
            sample_ids = counts.columns.tolist()

        # 전처리
        X = self.preprocessor.transform(counts)
        gene_names = self.preprocessor.selected_genes

        # CatBoost 모델 사용 (앙상블 중 기본)
        model = self.ensemble.models.get('catboost')
        if model is None:
            return {'error': 'CatBoost model not found'}

        logger.info("Computing SHAP values (this may take a while)...")

        # SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Multi-class인 경우 shap_values는 (n_classes, n_samples, n_features)
        # 또는 list of arrays
        if isinstance(shap_values, list):
            # 각 클래스별 SHAP values
            shap_values_array = np.array(shap_values)  # (n_classes, n_samples, n_features)
        else:
            shap_values_array = shap_values

        results = {
            'sample_ids': sample_ids,
            'gene_names': gene_names,
            'class_names': self.ensemble.class_names,
        }

        # 전체 중요도 (모든 클래스 평균)
        if len(shap_values_array.shape) == 3:
            # Multi-class: 절대값 평균
            global_importance = np.abs(shap_values_array).mean(axis=(0, 1))
        else:
            global_importance = np.abs(shap_values_array).mean(axis=0)

        top_indices = np.argsort(global_importance)[::-1][:top_k_genes]
        results['global_top_genes'] = [
            {
                'gene': gene_names[i],
                'importance': float(global_importance[i]),
                'rank': rank + 1
            }
            for rank, i in enumerate(top_indices)
        ]

        logger.info(f"Top 5 global genes: {[g['gene'] for g in results['global_top_genes'][:5]]}")

        # 특정 암종별 중요 유전자
        if cancer_type and cancer_type in self.ensemble.class_names:
            class_idx = self.ensemble.class_names.index(cancer_type)

            if len(shap_values_array.shape) == 3:
                class_shap = shap_values_array[class_idx]  # (n_samples, n_features)
            else:
                class_shap = shap_values_array

            class_importance = np.abs(class_shap).mean(axis=0)
            class_top_indices = np.argsort(class_importance)[::-1][:top_k_genes]

            results[f'{cancer_type}_top_genes'] = [
                {
                    'gene': gene_names[i],
                    'importance': float(class_importance[i]),
                    'mean_shap': float(class_shap[:, i].mean()),
                    'direction': 'up' if class_shap[:, i].mean() > 0 else 'down',
                    'rank': rank + 1
                }
                for rank, i in enumerate(class_top_indices)
            ]

            logger.info(f"Top 5 genes for {cancer_type}: {[g['gene'] for g in results[f'{cancer_type}_top_genes'][:5]]}")

        # 플롯 저장
        if save_plots and output_dir:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            results['plots'] = {}

            try:
                # Global summary plot - bar chart 형식 (shape 문제 없음)
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values_array,
                    X,
                    feature_names=gene_names,
                    max_display=top_k_genes,
                    plot_type="bar",
                    show=False
                )
                plt.tight_layout()
                plt.savefig(output_path / "shap_global_summary.png", dpi=150, bbox_inches='tight')
                plt.close()
                results['plots']['global_summary'] = str(output_path / "shap_global_summary.png")
                logger.info(f"Global summary plot saved")
            except Exception as e:
                logger.warning(f"Failed to save global summary plot: {e}")

            # Cancer-specific plot
            if cancer_type and cancer_type in self.ensemble.class_names:
                try:
                    class_idx = self.ensemble.class_names.index(cancer_type)
                    plt.figure(figsize=(12, 10))
                    if len(shap_values_array.shape) == 3:
                        class_shap = shap_values_array[class_idx]
                        shap.summary_plot(
                            class_shap,
                            X,
                            feature_names=gene_names,
                            max_display=top_k_genes,
                            plot_type="bar",
                            show=False
                        )
                    plt.title(f"SHAP Summary - {cancer_type}")
                    plt.tight_layout()
                    plt.savefig(output_path / f"shap_{cancer_type}_summary.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    results['plots'][f'{cancer_type}_summary'] = str(output_path / f"shap_{cancer_type}_summary.png")
                    logger.info(f"{cancer_type} summary plot saved")
                except Exception as e:
                    logger.warning(f"Failed to save {cancer_type} plot: {e}")

            logger.info(f"SHAP plots saved to {output_path}")

        return results

    def get_cancer_specific_genes(self,
                                  counts: pd.DataFrame,
                                  cancer_types: Optional[List[str]] = None,
                                  top_k: int = 10) -> Dict[str, List[Dict]]:
        """
        각 암종별 특이적 유전자 추출

        Args:
            counts: Gene x Sample count matrix
            cancer_types: 분석할 암종 목록 (None이면 전체)
            top_k: 암종당 상위 유전자 수

        Returns:
            암종별 특이적 유전자 딕셔너리
        """
        if not self.is_loaded:
            self.load()

        if cancer_types is None:
            cancer_types = self.ensemble.class_names

        results = {}
        for cancer_type in cancer_types:
            if cancer_type in self.ensemble.class_names:
                shap_result = self.explain_with_shap(
                    counts,
                    cancer_type=cancer_type,
                    top_k_genes=top_k,
                    save_plots=False
                )
                key = f'{cancer_type}_top_genes'
                if key in shap_result:
                    results[cancer_type] = shap_result[key]

        return results


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
