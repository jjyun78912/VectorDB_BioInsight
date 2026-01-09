"""
RNA-seq Data Preprocessor
=========================

RNA-seq count data를 ML 학습에 적합하게 전처리합니다.

Steps:
1. Low-expression gene filtering
2. Normalization (CPM, log2)
3. Batch effect correction (optional)
4. Feature selection
5. Train/test split with stratification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RNAseqPreprocessor:
    """RNA-seq 데이터 전처리 파이프라인"""

    def __init__(self,
                 min_counts: int = 10,
                 min_samples_pct: float = 0.2,
                 log_transform: bool = True,
                 normalize: str = "cpm",
                 n_top_genes: int = 5000,
                 feature_selection: str = "variance"):
        """
        Args:
            min_counts: 최소 count 수 (이하는 필터링)
            min_samples_pct: 최소 샘플 비율 (이 비율 이상에서 발현되어야 함)
            log_transform: log2 변환 여부
            normalize: 정규화 방법 ("cpm", "tpm", "none")
            n_top_genes: 선택할 상위 유전자 수
            feature_selection: 특징 선택 방법 ("variance", "anova", "none")
        """
        self.min_counts = min_counts
        self.min_samples_pct = min_samples_pct
        self.log_transform = log_transform
        self.normalize = normalize
        self.n_top_genes = n_top_genes
        self.feature_selection = feature_selection

        # 학습 시 저장되는 정보
        self.selected_genes: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
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
                     test_size: float = 0.2, random_state: int = 42
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전처리 파이프라인 학습 및 변환

        Args:
            counts: Gene x Sample count matrix
            labels: 샘플 레이블 (0/1)
            test_size: 테스트셋 비율
            random_state: 랜덤 시드

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Input shape: {counts.shape}")

        # 1. Low-expression filtering
        filtered = self._filter_low_expression(counts)

        # 2. Normalization
        if self.normalize == "cpm":
            normalized = self._normalize_cpm(filtered)
        else:
            normalized = filtered

        # 3. Log transform
        if self.log_transform:
            transformed = self._log_transform(normalized)
        else:
            transformed = normalized

        # 4. Feature selection
        if self.feature_selection == "variance":
            self.selected_genes = self._select_top_variance_genes(
                transformed, self.n_top_genes
            )
        elif self.feature_selection == "anova":
            self.selected_genes = self._select_anova_genes(
                transformed, labels, self.n_top_genes
            )
        else:
            self.selected_genes = transformed.index.tolist()[:self.n_top_genes]

        logger.info(f"Selected {len(self.selected_genes)} genes")

        # 선택된 유전자만 추출 (samples x genes)
        X = transformed.loc[self.selected_genes].T.values

        # 5. Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=random_state,
            stratify=labels
        )

        # 6. Standardization
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # 유전자 통계 저장
        self.gene_stats = pd.DataFrame({
            'gene': self.selected_genes,
            'mean': transformed.loc[self.selected_genes].mean(axis=1).values,
            'std': transformed.loc[self.selected_genes].std(axis=1).values,
            'variance': transformed.loc[self.selected_genes].var(axis=1).values,
        })

        self.is_fitted = True

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train labels: {np.bincount(y_train.astype(int))}")
        logger.info(f"Test labels: {np.bincount(y_test.astype(int))}")

        return X_train, X_test, y_train, y_test

    def transform(self, counts: pd.DataFrame) -> np.ndarray:
        """
        학습된 파이프라인으로 새 데이터 변환

        Args:
            counts: Gene x Sample count matrix

        Returns:
            변환된 데이터 (samples x genes)
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
        X = pd.DataFrame(0, index=counts.columns, columns=self.selected_genes)
        for gene in available_genes:
            X[gene] = transformed.loc[gene].values

        # 표준화
        X = self.scaler.transform(X.values)

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
            'min_counts': self.min_counts,
            'min_samples_pct': self.min_samples_pct,
            'log_transform': self.log_transform,
            'normalize': self.normalize,
            'n_top_genes': self.n_top_genes,
            'feature_selection': self.feature_selection,
            'selected_genes': self.selected_genes,
            'scaler': self.scaler,
            'gene_stats': self.gene_stats,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(save_dict, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RNAseqPreprocessor":
        """저장된 전처리기 로드"""
        save_dict = joblib.load(path)

        preprocessor = cls(
            min_counts=save_dict['min_counts'],
            min_samples_pct=save_dict['min_samples_pct'],
            log_transform=save_dict['log_transform'],
            normalize=save_dict['normalize'],
            n_top_genes=save_dict['n_top_genes'],
            feature_selection=save_dict['feature_selection'],
        )

        preprocessor.selected_genes = save_dict['selected_genes']
        preprocessor.scaler = save_dict['scaler']
        preprocessor.gene_stats = save_dict['gene_stats']
        preprocessor.is_fitted = save_dict['is_fitted']

        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor

    def get_gene_names(self) -> List[str]:
        """선택된 유전자 이름 목록 반환"""
        return self.selected_genes if self.selected_genes else []

    def summary(self) -> Dict[str, Any]:
        """전처리 설정 요약"""
        return {
            'min_counts': self.min_counts,
            'min_samples_pct': self.min_samples_pct,
            'log_transform': self.log_transform,
            'normalize': self.normalize,
            'n_top_genes': self.n_top_genes,
            'feature_selection': self.feature_selection,
            'n_selected_genes': len(self.selected_genes) if self.selected_genes else 0,
            'is_fitted': self.is_fitted,
        }
