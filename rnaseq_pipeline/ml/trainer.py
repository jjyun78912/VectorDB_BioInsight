"""
CatBoost Model Trainer
======================

CatBoost를 사용한 RNA-seq 샘플 분류 모델 학습

Features:
- Hyperparameter optimization (Optuna)
- Cross-validation
- Early stopping
- Model saving/loading
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import optuna
from optuna.samplers import TPESampler
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatBoostTrainer:
    """CatBoost 분류기 학습 및 최적화"""

    def __init__(self,
                 task_type: str = "CPU",
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Args:
            task_type: "CPU" or "GPU"
            random_state: 랜덤 시드
            verbose: 상세 로그 출력
        """
        self.task_type = task_type
        self.random_state = random_state
        self.verbose = verbose

        self.model: Optional[CatBoostClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.training_history: List[Dict] = []
        self.cv_results: Optional[Dict] = None

    def _create_model(self, params: Dict) -> CatBoostClassifier:
        """CatBoost 모델 생성"""
        default_params = {
            'task_type': self.task_type,
            'random_seed': self.random_state,
            'verbose': False,
            'allow_writing_files': False,
        }
        default_params.update(params)
        return CatBoostClassifier(**default_params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None,
              params: Optional[Dict] = None) -> CatBoostClassifier:
        """
        모델 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터 (optional)
            y_val: 검증 레이블 (optional)
            feature_names: 특징 이름 목록
            params: 하이퍼파라미터

        Returns:
            학습된 CatBoost 모델
        """
        if params is None:
            params = self.get_default_params()

        self.feature_names = feature_names
        self.model = self._create_model(params)

        # Pool 생성
        train_pool = Pool(X_train, y_train, feature_names=feature_names)

        if X_val is not None and y_val is not None:
            val_pool = Pool(X_val, y_val, feature_names=feature_names)
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=50,
                verbose=100 if self.verbose else False,
            )
        else:
            self.model.fit(train_pool, verbose=100 if self.verbose else False)

        self.best_params = params

        logger.info(f"Model trained. Best iteration: {self.model.best_iteration_}")

        return self.model

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      n_splits: int = 5,
                      feature_names: Optional[List[str]] = None,
                      params: Optional[Dict] = None) -> Dict[str, float]:
        """
        교차 검증 수행

        Args:
            X: 전체 데이터
            y: 전체 레이블
            n_splits: fold 수
            feature_names: 특징 이름
            params: 하이퍼파라미터

        Returns:
            교차 검증 결과 (평균 메트릭)
        """
        if params is None:
            params = self.get_default_params()

        self.feature_names = feature_names
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=self.random_state)

        metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'roc_auc': []
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self._create_model(params)
            train_pool = Pool(X_train, y_train, feature_names=feature_names)
            val_pool = Pool(X_val, y_val, feature_names=feature_names)

            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=50,
                verbose=False,
            )

            # 예측
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            # 메트릭 계산
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            metrics['roc_auc'].append(roc_auc_score(y_val, y_prob))

            if self.verbose:
                logger.info(f"Fold {fold + 1}: AUC = {metrics['roc_auc'][-1]:.4f}")

        # 평균 계산
        self.cv_results = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in metrics.items()
        }

        logger.info(f"\nCV Results:")
        for metric, values in self.cv_results.items():
            logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")

        return self.cv_results

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                 n_trials: int = 50,
                                 n_splits: int = 3,
                                 feature_names: Optional[List[str]] = None,
                                 timeout: Optional[int] = None) -> Dict:
        """
        Optuna를 사용한 하이퍼파라미터 최적화

        Args:
            X: 학습 데이터
            y: 레이블
            n_trials: 시도 횟수
            n_splits: CV fold 수
            feature_names: 특징 이름
            timeout: 최대 시간 (초)

        Returns:
            최적 파라미터
        """
        self.feature_names = feature_names

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            }

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=self.random_state)

            auc_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = self._create_model(params)
                train_pool = Pool(X_train, y_train, feature_names=feature_names)
                val_pool = Pool(X_val, y_val, feature_names=feature_names)

                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=30,
                    verbose=False,
                )

                y_prob = model.predict_proba(X_val)[:, 1]
                auc_scores.append(roc_auc_score(y_val, y_prob))

            return np.mean(auc_scores)

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        if self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                      show_progress_bar=True)

        self.best_params = study.best_params
        logger.info(f"\nBest params: {self.best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")

        return self.best_params

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            X_test: 테스트 데이터
            y_test: 테스트 레이블

        Returns:
            평가 메트릭
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }

        logger.info(f"\nTest Evaluation:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def get_feature_importance(self, top_k: int = 50) -> pd.DataFrame:
        """
        특징 중요도 반환

        Args:
            top_k: 상위 k개 반환

        Returns:
            특징 중요도 DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.get_feature_importance()

        if self.feature_names:
            df = pd.DataFrame({
                'gene': self.feature_names,
                'importance': importance,
            })
        else:
            df = pd.DataFrame({
                'gene': [f'gene_{i}' for i in range(len(importance))],
                'importance': importance,
            })

        df = df.sort_values('importance', ascending=False)
        return df.head(top_k)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측"""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)

    def save(self, path: str):
        """모델 및 설정 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # CatBoost 모델 저장
        if self.model:
            self.model.save_model(str(path / "model.cbm"))

        # 메타데이터 저장
        metadata = {
            'task_type': self.task_type,
            'random_state': self.random_state,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'cv_results': self.cv_results,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CatBoostTrainer":
        """저장된 모델 로드"""
        path = Path(path)

        # 메타데이터 로드
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        trainer = cls(
            task_type=metadata['task_type'],
            random_state=metadata['random_state'],
        )

        trainer.best_params = metadata['best_params']
        trainer.feature_names = metadata['feature_names']
        trainer.cv_results = metadata['cv_results']

        # CatBoost 모델 로드
        if (path / "model.cbm").exists():
            trainer.model = CatBoostClassifier()
            trainer.model.load_model(str(path / "model.cbm"))

        logger.info(f"Model loaded from {path}")
        return trainer

    @staticmethod
    def get_default_params() -> Dict:
        """기본 하이퍼파라미터"""
        return {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
        }
