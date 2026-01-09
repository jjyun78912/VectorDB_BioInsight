"""
SHAP Explainer Module
=====================

SHAP (SHapley Additive exPlanations)를 사용한 모델 설명

Features:
- Global feature importance
- Local (per-sample) explanations
- Visualization (summary plot, force plot, waterfall)
- Gene Status Card generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP 기반 모델 설명"""

    def __init__(self, model: CatBoostClassifier,
                 feature_names: Optional[List[str]] = None):
        """
        Args:
            model: 학습된 CatBoost 모델
            feature_names: 특징 이름 (유전자 이름)
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: Optional[float] = None

    def fit(self, X_background: np.ndarray, sample_size: int = 100):
        """
        SHAP Explainer 초기화

        Args:
            X_background: 배경 데이터 (전체 학습 데이터 또는 샘플)
            sample_size: 배경 데이터 샘플 수
        """
        # 배경 데이터 샘플링
        if len(X_background) > sample_size:
            indices = np.random.choice(len(X_background), sample_size, replace=False)
            background = X_background[indices]
        else:
            background = X_background

        # TreeExplainer 사용 (CatBoost에 최적화)
        self.explainer = shap.TreeExplainer(self.model)
        self.base_value = self.explainer.expected_value

        if isinstance(self.base_value, np.ndarray):
            # Binary classification: 양성 클래스 기준
            if len(self.base_value) > 1:
                self.base_value = self.base_value[1]
            else:
                self.base_value = self.base_value[0]

        logger.info(f"SHAP Explainer initialized. Base value: {self.base_value:.4f}")

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        샘플에 대한 SHAP 값 계산

        Args:
            X: 설명할 데이터 (samples x features)

        Returns:
            SHAP values (samples x features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call fit() first.")

        shap_values = self.explainer.shap_values(X)

        # Binary classification: 양성 클래스 SHAP 값 사용
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        self.shap_values = shap_values
        return shap_values

    def get_global_importance(self, X: np.ndarray, top_k: int = 50) -> pd.DataFrame:
        """
        전역 특징 중요도 (평균 |SHAP|)

        Args:
            X: 데이터
            top_k: 상위 k개 반환

        Returns:
            유전자별 SHAP 중요도 DataFrame
        """
        shap_values = self.explain(X)

        # 평균 절대값
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        if self.feature_names:
            df = pd.DataFrame({
                'gene': self.feature_names,
                'mean_abs_shap': mean_abs_shap,
            })
        else:
            df = pd.DataFrame({
                'gene': [f'gene_{i}' for i in range(len(mean_abs_shap))],
                'mean_abs_shap': mean_abs_shap,
            })

        df = df.sort_values('mean_abs_shap', ascending=False)
        return df.head(top_k)

    def explain_sample(self, X_sample: np.ndarray,
                       top_k: int = 20) -> Dict[str, Any]:
        """
        단일 샘플에 대한 상세 설명

        Args:
            X_sample: 샘플 데이터 (1 x features)
            top_k: 상위 영향력 유전자 수

        Returns:
            샘플 설명 딕셔너리
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.explain(X_sample)[0]

        # 예측 확률
        prob = self.model.predict_proba(X_sample)[0, 1]
        pred = int(prob > 0.5)

        # 상위 영향력 유전자
        indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

        top_genes = []
        for idx in indices:
            gene_name = self.feature_names[idx] if self.feature_names else f'gene_{idx}'
            top_genes.append({
                'gene': gene_name,
                'shap_value': float(shap_values[idx]),
                'direction': 'up' if shap_values[idx] > 0 else 'down',
                'feature_value': float(X_sample[0, idx]),
            })

        return {
            'prediction': pred,
            'probability': float(prob),
            'base_value': float(self.base_value),
            'top_genes': top_genes,
            'total_shap': float(shap_values.sum()),
        }

    def generate_gene_status_card(self, gene_name: str,
                                   X: np.ndarray,
                                   shap_values: Optional[np.ndarray] = None,
                                   extra_info: Optional[Dict] = None) -> str:
        """
        유전자 Status Card 생성

        Args:
            gene_name: 유전자 이름
            X: 데이터
            shap_values: SHAP 값 (없으면 계산)
            extra_info: 추가 정보 (DB 검증 결과 등)

        Returns:
            포맷된 Status Card 문자열
        """
        if shap_values is None:
            shap_values = self.explain(X)

        if self.feature_names is None or gene_name not in self.feature_names:
            return f"Gene {gene_name} not found in feature set."

        gene_idx = self.feature_names.index(gene_name)

        # SHAP 통계
        gene_shap = shap_values[:, gene_idx]
        mean_shap = gene_shap.mean()
        abs_mean_shap = np.abs(gene_shap).mean()

        # SHAP 순위
        global_importance = np.abs(shap_values).mean(axis=0)
        rank = int((global_importance > abs_mean_shap).sum()) + 1

        # 발현 통계
        gene_values = X[:, gene_idx]
        mean_expr = gene_values.mean()
        std_expr = gene_values.std()

        card = f"""
{'═' * 60}
  Gene Status Card: {gene_name}
{'═' * 60}

  ML PREDICTION (CatBoost + SHAP)
  ─────────────────────────────────
  Mean SHAP Value: {mean_shap:+.4f}
  Mean |SHAP|:     {abs_mean_shap:.4f}
  Global Rank:     #{rank} / {len(self.feature_names)}
  Direction:       {'↑ 암 발생에 기여' if mean_shap > 0 else '↓ 정상 유지에 기여'}

  EXPRESSION (Preprocessed)
  ─────────────────────────────────
  Mean: {mean_expr:.4f}
  Std:  {std_expr:.4f}
"""

        if extra_info:
            card += f"""
  DATABASE VALIDATION
  ─────────────────────────────────
"""
            for source, info in extra_info.items():
                card += f"  [{source}] {info}\n"

        card += f"""
  LIMITATIONS (Guardrail)
  ─────────────────────────────────
  ⚠️ ML 예측은 진단이 아니며, 참고용입니다.
  ⚠️ SHAP 순위는 "분류 기여도"이며 생물학적 중요도와 다를 수 있습니다.

  SUGGESTED VALIDATIONS
  ─────────────────────────────────
  • 실험적 검증: RT-qPCR, Western blot
  • 문헌 검토: PubMed에서 {gene_name} + cancer 검색
  • 기능 연구: Knockdown/Overexpression 실험
{'═' * 60}
"""
        return card

    def plot_summary(self, X: np.ndarray, output_path: str,
                    max_display: int = 20, plot_type: str = "dot"):
        """
        SHAP Summary Plot 저장

        Args:
            X: 데이터
            output_path: 저장 경로
            max_display: 표시할 유전자 수
            plot_type: "dot" or "bar"
        """
        shap_values = self.explain(X)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Summary plot saved to {output_path}")

    def plot_waterfall(self, X_sample: np.ndarray, output_path: str,
                       max_display: int = 15):
        """
        단일 샘플에 대한 Waterfall Plot

        Args:
            X_sample: 샘플 데이터
            output_path: 저장 경로
            max_display: 표시할 유전자 수
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.explain(X_sample)

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.base_value,
                data=X_sample[0],
                feature_names=self.feature_names,
            ),
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Waterfall plot saved to {output_path}")

    def plot_force(self, X_sample: np.ndarray, output_path: str):
        """
        Force Plot (HTML로 저장)

        Args:
            X_sample: 샘플 데이터
            output_path: 저장 경로 (.html)
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.explain(X_sample)

        force_plot = shap.force_plot(
            self.base_value,
            shap_values[0],
            X_sample[0],
            feature_names=self.feature_names,
        )

        shap.save_html(output_path, force_plot)
        logger.info(f"Force plot saved to {output_path}")

    def plot_dependence(self, X: np.ndarray, gene_name: str,
                        output_path: str, interaction_gene: Optional[str] = None):
        """
        Dependence Plot (유전자별 SHAP vs 발현량)

        Args:
            X: 데이터
            gene_name: 타겟 유전자
            output_path: 저장 경로
            interaction_gene: 상호작용 유전자 (색상)
        """
        if self.feature_names is None or gene_name not in self.feature_names:
            logger.error(f"Gene {gene_name} not found")
            return

        shap_values = self.explain(X)
        gene_idx = self.feature_names.index(gene_name)

        plt.figure(figsize=(10, 6))

        if interaction_gene and interaction_gene in self.feature_names:
            interaction_idx = self.feature_names.index(interaction_gene)
            shap.dependence_plot(
                gene_idx, shap_values, X,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False,
            )
        else:
            shap.dependence_plot(
                gene_idx, shap_values, X,
                feature_names=self.feature_names,
                interaction_index="auto",
                show=False,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Dependence plot saved to {output_path}")

    def export_explanations(self, X: np.ndarray,
                            sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        모든 샘플의 SHAP 값을 DataFrame으로 내보내기

        Args:
            X: 데이터
            sample_ids: 샘플 ID 목록

        Returns:
            SHAP 값 DataFrame (samples x genes)
        """
        shap_values = self.explain(X)

        columns = self.feature_names if self.feature_names else [
            f'gene_{i}' for i in range(shap_values.shape[1])
        ]

        df = pd.DataFrame(shap_values, columns=columns)

        if sample_ids:
            df.index = sample_ids

        return df

    def save(self, path: str):
        """Explainer 상태 저장"""
        save_dict = {
            'feature_names': self.feature_names,
            'base_value': float(self.base_value) if self.base_value else None,
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        logger.info(f"Explainer config saved to {path}")

    @classmethod
    def from_model(cls, model: CatBoostClassifier,
                   feature_names: Optional[List[str]] = None,
                   X_background: Optional[np.ndarray] = None) -> "SHAPExplainer":
        """
        모델로부터 Explainer 생성

        Args:
            model: CatBoost 모델
            feature_names: 특징 이름
            X_background: 배경 데이터

        Returns:
            초기화된 SHAPExplainer
        """
        explainer = cls(model, feature_names)
        if X_background is not None:
            explainer.fit(X_background)
        return explainer
