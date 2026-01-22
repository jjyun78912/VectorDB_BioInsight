"""
RNA-seq Data Preprocessor
=========================

RNA-seq count data를 ML 학습에 적합하게 전처리합니다.

Data Leakage 방지 순서 (v2.0):
1. Train/Test split (환자 단위)
2. Train 데이터만으로 HVG selection
3. Train 데이터만으로 PCA fitting (optional)
4. Test에 transform만 적용
5. 평가

기존 순서 (v1.0, deprecated):
1. Low-expression gene filtering
2. Normalization (CPM, log2)
3. Feature selection (ALL data - DATA LEAKAGE!)
4. Train/test split
5. Standardization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RNAseqPreprocessor:
    """RNA-seq 데이터 전처리 파이프라인 (Data Leakage 방지 v2.0)"""

    def __init__(self,
                 min_counts: int = 10,
                 min_samples_pct: float = 0.2,
                 log_transform: bool = True,
                 normalize: str = "cpm",
                 n_top_genes: int = 5000,
                 feature_selection: str = "variance",
                 use_pca: bool = False,
                 n_components: int = 100):
        """
        Args:
            min_counts: 최소 count 수 (이하는 필터링)
            min_samples_pct: 최소 샘플 비율 (이 비율 이상에서 발현되어야 함)
            log_transform: log2 변환 여부
            normalize: 정규화 방법 ("cpm", "tpm", "none")
            n_top_genes: 선택할 상위 유전자 수 (HVG)
            feature_selection: 특징 선택 방법 ("variance", "anova", "none")
            use_pca: PCA 차원 축소 사용 여부
            n_components: PCA 컴포넌트 수
        """
        self.min_counts = min_counts
        self.min_samples_pct = min_samples_pct
        self.log_transform = log_transform
        self.normalize = normalize
        self.n_top_genes = n_top_genes
        self.feature_selection = feature_selection
        self.use_pca = use_pca
        self.n_components = n_components

        # 학습 시 저장되는 정보 (Train 데이터에서만 fitting)
        self.selected_genes: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.gene_stats: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def _filter_low_expression(self, counts: pd.DataFrame) -> pd.DataFrame:
        """낮은 발현량 유전자 필터링"""
        min_samples = int(counts.shape[1] * self.min_samples_pct)

        # 각 유전자가 min_counts 이상인 샘플 수 계산
        expressed_samples = (counts >= self.min_counts).sum(axis=1)

        # 필터링
        keep_genes = expressed_samples >= min_samples
        filtered = counts.loc[keep_genes]

        logger.info(f"Filtered genes: {counts.shape[0]} -> {filtered.shape[0]} "
                   f"(removed {counts.shape[0] - filtered.shape[0]})")

        return filtered

    def _normalize_cpm(self, counts: pd.DataFrame) -> pd.DataFrame:
        """CPM (Counts Per Million) 정규화"""
        lib_sizes = counts.sum(axis=0)
        cpm = counts * 1e6 / lib_sizes
        return cpm

    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """log2(x + 1) 변환"""
        return np.log2(data + 1)

    def _select_top_variance_genes(self, data: pd.DataFrame, n_top: int) -> List[str]:
        """분산 기준 상위 유전자 선택"""
        variances = data.var(axis=1)
        top_genes = variances.nlargest(n_top).index.tolist()
        return top_genes

    def _select_anova_genes(self, data: pd.DataFrame, labels: np.ndarray,
                           n_top: int) -> List[str]:
        """ANOVA F-value 기준 상위 유전자 선택"""
        selector = SelectKBest(f_classif, k=min(n_top, data.shape[0]))
        selector.fit(data.T, labels)

        # 선택된 유전자
        mask = selector.get_support()
        selected = data.index[mask].tolist()

        # F-score로 정렬
        scores = pd.Series(selector.scores_, index=data.index)
        selected = scores.loc[selected].nlargest(n_top).index.tolist()

        return selected

    def fit_transform(self, counts: pd.DataFrame, labels: np.ndarray,
                     test_size: float = 0.2, random_state: int = 42,
                     patient_ids: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전처리 파이프라인 학습 및 변환 (Data Leakage 방지 v2.0)

        올바른 순서:
        1. Train/Test split (환자 단위)
        2. Train 데이터만으로 HVG selection
        3. Train 데이터만으로 PCA fitting (optional)
        4. Test에 transform만 적용
        5. 평가

        Args:
            counts: Gene x Sample count matrix
            labels: 샘플 레이블 (0/1 또는 multi-class)
            test_size: 테스트셋 비율
            random_state: 랜덤 시드
            patient_ids: 환자 ID 배열 (환자 단위 split용, 없으면 샘플 단위 split)

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Input shape: {counts.shape} (genes x samples)")
        logger.info("Using Data Leakage Prevention Pipeline v2.0")

        # ========================================
        # STEP 1: Train/Test Split (환자 단위)
        # ========================================
        sample_names = counts.columns.tolist()
        n_samples = len(sample_names)

        if patient_ids is not None:
            # 환자 단위 split (같은 환자의 샘플은 같은 set에)
            logger.info(f"Patient-level split: {len(np.unique(patient_ids))} unique patients")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(sample_names, labels, groups=patient_ids))
        else:
            # 샘플 단위 stratified split
            logger.info("Sample-level stratified split (no patient IDs provided)")
            indices = np.arange(n_samples)
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state,
                stratify=labels
            )

        train_samples = [sample_names[i] for i in train_idx]
        test_samples = [sample_names[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        logger.info(f"Split: Train={len(train_samples)}, Test={len(test_samples)}")

        # Train/Test 데이터 분리
        counts_train = counts[train_samples]
        counts_test = counts[test_samples]

        # ========================================
        # STEP 2: 기본 전처리 (Train에서 파라미터 학습)
        # ========================================
        # 2a. Low-expression filtering (Train 데이터 기준)
        filtered_train = self._filter_low_expression(counts_train)
        # Test도 같은 유전자만 사용
        common_genes = [g for g in filtered_train.index if g in counts_test.index]
        filtered_train = filtered_train.loc[common_genes]
        filtered_test = counts_test.loc[common_genes]
        logger.info(f"After filtering: {len(common_genes)} genes")

        # 2b. Normalization
        if self.normalize == "cpm":
            normalized_train = self._normalize_cpm(filtered_train)
            normalized_test = self._normalize_cpm(filtered_test)
        else:
            normalized_train = filtered_train
            normalized_test = filtered_test

        # 2c. Log transform
        if self.log_transform:
            transformed_train = self._log_transform(normalized_train)
            transformed_test = self._log_transform(normalized_test)
        else:
            transformed_train = normalized_train
            transformed_test = normalized_test

        # ========================================
        # STEP 3: HVG Selection (Train 데이터만 사용!)
        # ========================================
        if self.feature_selection == "variance":
            # Train 데이터의 분산으로만 유전자 선택
            self.selected_genes = self._select_top_variance_genes(
                transformed_train, self.n_top_genes
            )
        elif self.feature_selection == "anova":
            # Train 데이터와 Train 레이블로만 유전자 선택
            self.selected_genes = self._select_anova_genes(
                transformed_train, y_train, self.n_top_genes
            )
        else:
            self.selected_genes = transformed_train.index.tolist()[:self.n_top_genes]

        logger.info(f"Selected {len(self.selected_genes)} HVGs (from TRAIN data only)")

        # 선택된 유전자만 추출 (samples x genes)
        X_train_raw = transformed_train.loc[self.selected_genes].T.values
        X_test_raw = transformed_test.loc[self.selected_genes].T.values

        # ========================================
        # STEP 4: Standardization (Train에서 fit, Test에 transform)
        # ========================================
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)  # fit on TRAIN only
        X_test = self.scaler.transform(X_test_raw)        # transform only

        logger.info(f"Standardization: fit on train, transform on test")

        # ========================================
        # STEP 5: PCA (Optional, Train에서 fit)
        # ========================================
        if self.use_pca:
            n_comp = min(self.n_components, X_train.shape[0], X_train.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=random_state)
            X_train = self.pca.fit_transform(X_train)  # fit on TRAIN only
            X_test = self.pca.transform(X_test)        # transform only
            logger.info(f"PCA: {n_comp} components, explained variance: "
                       f"{self.pca.explained_variance_ratio_.sum():.2%}")

        # ========================================
        # 유전자 통계 저장 (Train 데이터 기준)
        # ========================================
        self.gene_stats = pd.DataFrame({
            'gene': self.selected_genes,
            'mean': transformed_train.loc[self.selected_genes].mean(axis=1).values,
            'std': transformed_train.loc[self.selected_genes].std(axis=1).values,
            'variance': transformed_train.loc[self.selected_genes].var(axis=1).values,
        })

        self.is_fitted = True

        logger.info(f"Final shapes: Train={X_train.shape}, Test={X_test.shape}")
        logger.info(f"Train labels: {np.bincount(y_train.astype(int))}")
        logger.info(f"Test labels: {np.bincount(y_test.astype(int))}")

        return X_train, X_test, y_train, y_test

    def transform(self, counts: pd.DataFrame) -> np.ndarray:
        """
        학습된 파이프라인으로 새 데이터 변환 (transform only, no fitting)

        Args:
            counts: Gene x Sample count matrix

        Returns:
            변환된 데이터 (samples x genes or samples x n_components if PCA)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        # 선택된 유전자 추출 (없는 유전자는 0으로 채움)
        available_genes = [g for g in self.selected_genes if g in counts.index]
        missing_genes = [g for g in self.selected_genes if g not in counts.index]

        if missing_genes:
            logger.warning(f"Missing {len(missing_genes)} genes, filling with 0")

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

        # 선택된 유전자 순서대로 추출
        X = pd.DataFrame(0.0, index=counts.columns, columns=self.selected_genes, dtype=float)
        for gene in available_genes:
            X[gene] = transformed.loc[gene].values

        # 표준화 (train에서 학습된 scaler로 transform만)
        X = self.scaler.transform(X.values)

        # PCA (train에서 학습된 PCA로 transform만)
        if self.use_pca and self.pca is not None:
            X = self.pca.transform(X)

        return X

    def get_cv_splits(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Stratified K-Fold splits 생성"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
        return list(skf.split(X, y))

    def save(self, path: str):
        """전처리기 저장"""
        save_dict = {
            'version': '2.0',  # Data Leakage Prevention version
            'min_counts': self.min_counts,
            'min_samples_pct': self.min_samples_pct,
            'log_transform': self.log_transform,
            'normalize': self.normalize,
            'n_top_genes': self.n_top_genes,
            'feature_selection': self.feature_selection,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'selected_genes': self.selected_genes,
            'scaler': self.scaler,
            'pca': self.pca,
            'gene_stats': self.gene_stats,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(save_dict, path)
        logger.info(f"Preprocessor v2.0 saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RNAseqPreprocessor":
        """저장된 전처리기 로드"""
        save_dict = joblib.load(path)

        # v1.0 호환성
        use_pca = save_dict.get('use_pca', False)
        n_components = save_dict.get('n_components', 100)

        preprocessor = cls(
            min_counts=save_dict['min_counts'],
            min_samples_pct=save_dict['min_samples_pct'],
            log_transform=save_dict['log_transform'],
            normalize=save_dict['normalize'],
            n_top_genes=save_dict['n_top_genes'],
            feature_selection=save_dict['feature_selection'],
            use_pca=use_pca,
            n_components=n_components,
        )

        preprocessor.selected_genes = save_dict['selected_genes']
        preprocessor.scaler = save_dict['scaler']
        preprocessor.pca = save_dict.get('pca', None)
        preprocessor.gene_stats = save_dict['gene_stats']
        preprocessor.is_fitted = save_dict['is_fitted']

        version = save_dict.get('version', '1.0')
        logger.info(f"Preprocessor v{version} loaded from {path}")
        return preprocessor

    def get_gene_names(self) -> List[str]:
        """선택된 유전자 이름 목록 반환"""
        return self.selected_genes if self.selected_genes else []

    def summary(self) -> Dict[str, Any]:
        """전처리 설정 요약"""
        summary_dict = {
            'version': '2.0 (Data Leakage Prevention)',
            'min_counts': self.min_counts,
            'min_samples_pct': self.min_samples_pct,
            'log_transform': self.log_transform,
            'normalize': self.normalize,
            'n_top_genes': self.n_top_genes,
            'feature_selection': self.feature_selection,
            'use_pca': self.use_pca,
            'n_components': self.n_components if self.use_pca else None,
            'n_selected_genes': len(self.selected_genes) if self.selected_genes else 0,
            'is_fitted': self.is_fitted,
        }
        if self.use_pca and self.pca is not None:
            summary_dict['pca_explained_variance'] = float(self.pca.explained_variance_ratio_.sum())
        return summary_dict
