#!/usr/bin/env python3
"""
Pan-Cancer 17종 모델 학습 스크립트
==================================

새로 다운로드한 TCGA 17종 데이터로 Pan-Cancer 분류기를 학습합니다.

암종 목록 (17종):
- BLCA, BRCA, COAD, GBM, HNSC, KIRC, LGG, LIHC
- LUAD, LUSC, OV, PAAD, PRAD, SKCM, STAD, THCA, UCEC
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
)
from catboost import CatBoostClassifier, Pool
import joblib
import optuna
from optuna.samplers import TPESampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 17종 암종 코드
CANCER_TYPES = [
    'BLCA', 'BRCA', 'COAD', 'GBM', 'HNSC', 'KIRC', 'LGG', 'LIHC',
    'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC'
]


def load_tcga_data(cancer_code: str, data_dir: str = "data/tcga") -> Optional[pd.DataFrame]:
    """단일 암종 데이터 로드 (개별 TSV 파일에서 TPM 추출)"""
    cancer_dir = Path(data_dir) / cancer_code
    tsv_files = list(cancer_dir.glob("*.tsv"))

    if not tsv_files:
        logger.warning(f"No TSV files found for {cancer_code}: {cancer_dir}")
        return None

    all_samples = {}

    for tsv_file in tsv_files:
        try:
            # TSV 파일 읽기 (첫 줄 주석 건너뜀)
            df = pd.read_csv(tsv_file, sep='\t', comment='#')

            # N_unmapped 등 메타 행 제거
            df = df[~df['gene_id'].str.startswith('N_')]

            # gene_name과 tpm_unstranded 열 추출
            if 'gene_name' in df.columns and 'tpm_unstranded' in df.columns:
                sample_id = tsv_file.stem.split('.')[0][:8]  # 샘플 ID 축약
                tpm_data = df.set_index('gene_name')['tpm_unstranded']
                all_samples[sample_id] = tpm_data
        except Exception as e:
            logger.warning(f"Error reading {tsv_file.name}: {e}")
            continue

    if not all_samples:
        logger.warning(f"No valid samples for {cancer_code}")
        return None

    # DataFrame으로 통합 (genes x samples)
    combined = pd.DataFrame(all_samples)

    # 중복 유전자 인덱스 제거 (첫 번째만 유지)
    if combined.index.duplicated().any():
        n_dup = combined.index.duplicated().sum()
        logger.debug(f"{cancer_code}: Removing {n_dup} duplicate gene indices")
        combined = combined[~combined.index.duplicated(keep='first')]

    logger.info(f"Loaded {cancer_code}: {combined.shape[0]} genes x {combined.shape[1]} samples (TPM)")

    return combined


def load_all_tcga_data(cancer_types: List[str] = CANCER_TYPES,
                       data_dir: str = "data/tcga") -> Tuple[pd.DataFrame, pd.Series]:
    """모든 암종 데이터를 로드하고 통합"""
    all_data = []
    all_labels = []

    for cancer in cancer_types:
        df = load_tcga_data(cancer, data_dir)
        if df is not None:
            # Transpose: genes x samples -> samples x genes
            df_t = df.T
            all_data.append(df_t)
            all_labels.extend([cancer] * len(df_t))

    # Concatenate all - 공통 유전자 먼저 찾기
    common_genes = set(all_data[0].columns)
    for df in all_data[1:]:
        common_genes = common_genes.intersection(set(df.columns))

    logger.info(f"Common genes across all cancer types: {len(common_genes)}")

    # 공통 유전자만 선택하여 concat
    all_data_filtered = [df[list(common_genes)] for df in all_data]
    combined = pd.concat(all_data_filtered, axis=0)
    labels = pd.Series(all_labels, index=combined.index)

    logger.info(f"Combined data: {combined.shape[0]} samples x {combined.shape[1]} genes")
    logger.info(f"Label distribution:\n{labels.value_counts().sort_index()}")

    return combined, labels


def preprocess_data(X: pd.DataFrame, y: pd.Series,
                    n_top_genes: int = 5000,
                    variance_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder, StandardScaler]:
    """데이터 전처리"""
    logger.info("Preprocessing data...")

    # 결측치 0으로 대체
    X = X.fillna(0)

    # 숫자형으로 변환
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 1. Log2 변환 (TPM -> log2(TPM + 1))
    X_log = np.log2(X + 1)

    # 2. 분산 기반 유전자 선택
    gene_vars = X_log.var(axis=0)
    gene_vars = gene_vars.sort_values(ascending=False)

    # 분산 임계값 이상 & 상위 N개 선택
    high_var_genes = gene_vars[gene_vars > variance_threshold]
    selected_genes = high_var_genes.head(n_top_genes).index.tolist()

    logger.info(f"Selected {len(selected_genes)} genes (variance > {variance_threshold})")

    X_selected = X_log[selected_genes].values

    # 3. 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 4. 레이블 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info(f"Classes: {label_encoder.classes_}")

    return X_scaled, y_encoded, selected_genes, label_encoder, scaler


def optuna_objective(trial, X_train, y_train, X_val, y_val, n_classes):
    """Optuna 하이퍼파라미터 최적화 objective"""
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),  # 축소: 빠른 학습
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 8),  # 축소
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 5, log=True),
        'border_count': trial.suggest_int('border_count', 64, 200),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 0.8),
        'random_strength': trial.suggest_float('random_strength', 1e-6, 5, log=True),
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU',
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy


def train_with_optuna(X_train, y_train, X_val, y_val, n_classes: int, n_trials: int = 50):
    """Optuna로 하이퍼파라미터 최적화 후 학습"""
    logger.info(f"Starting Optuna HPO with {n_trials} trials...")

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, n_classes),
        n_trials=n_trials,
        show_progress_bar=True
    )

    logger.info(f"Best trial accuracy: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # 최적 파라미터로 최종 모델 학습
    best_params = study.best_params
    best_params.update({
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': 100,
        'task_type': 'CPU',
    })

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    return final_model, study.best_params


def cross_validate(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                   best_params: Dict = None) -> Dict:
    """Stratified K-Fold Cross Validation"""
    logger.info(f"Running {n_splits}-fold cross validation...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {
        'accuracy': [],
        'f1_macro': [],
        'mcc': [],
    }

    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 모델 학습
        params = best_params.copy() if best_params else {}
        params.update({
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': 42,
            'verbose': False,
            'task_type': 'CPU',
            'iterations': params.get('iterations', 1000),
        })

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        mcc = matthews_corrcoef(y_val, y_pred)

        metrics['accuracy'].append(acc)
        metrics['f1_macro'].append(f1)
        metrics['mcc'].append(mcc)

        logger.info(f"Fold {fold+1}: Accuracy={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

    # 평균 및 표준편차
    summary = {}
    for metric, values in metrics.items():
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    logger.info(f"\nCV Summary:")
    logger.info(f"  Accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
    logger.info(f"  F1 Macro: {summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['std']:.4f}")
    logger.info(f"  MCC:      {summary['mcc']['mean']:.4f} ± {summary['mcc']['std']:.4f}")

    return summary


def save_model(model, label_encoder, scaler, selected_genes, best_params, cv_results,
               output_dir: str = "models/rnaseq/pancancer_v3"):
    """모델 및 관련 파일 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. CatBoost 모델 저장
    model.save_model(str(output_path / "catboost_model.cbm"))
    logger.info(f"Saved CatBoost model")

    # 2. Preprocessor 저장 (scikit-learn 호환)
    preprocessor = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'selected_genes': selected_genes,
        'n_genes': len(selected_genes),
    }
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")
    logger.info(f"Saved preprocessor")

    # 3. 메타데이터 저장
    metadata = {
        'created_at': datetime.now().isoformat(),
        'n_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'n_genes': len(selected_genes),
        'best_params': best_params,
        'cv_results': {
            k: {'mean': v['mean'], 'std': v['std']}
            for k, v in cv_results.items()
        },
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata")

    # 4. Gene list 저장
    with open(output_path / "selected_genes.txt", 'w') as f:
        f.write('\n'.join(selected_genes))

    logger.info(f"Model saved to {output_path}")

    return output_path


def main():
    """메인 실행"""
    logger.info("=" * 60)
    logger.info("Pan-Cancer 17종 모델 학습 시작")
    logger.info("=" * 60)

    # 1. 데이터 로드
    X, y = load_all_tcga_data(CANCER_TYPES)

    # 2. 전처리
    X_processed, y_encoded, selected_genes, label_encoder, scaler = preprocess_data(
        X, y, n_top_genes=5000, variance_threshold=0.1
    )

    # 3. Train/Val/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Optuna HPO
    n_classes = len(label_encoder.classes_)
    model, best_params = train_with_optuna(X_train, y_train, X_val, y_val, n_classes, n_trials=15)  # 빠른 학습

    # 5. Test set 평가
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    logger.info(f"\nTest Set Results:")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  F1 Macro: {test_f1:.4f}")
    logger.info(f"  MCC:      {test_mcc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    # 6. Cross Validation (전체 데이터)
    cv_results = cross_validate(X_processed, y_encoded, n_splits=5, best_params=best_params)

    # 7. 최종 모델 학습 (전체 데이터)
    logger.info("\nTraining final model on all data...")
    final_params = best_params.copy()
    final_params.update({
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': 100,
        'task_type': 'CPU',
    })

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_processed, y_encoded)

    # 8. 모델 저장
    output_path = save_model(
        final_model, label_encoder, scaler, selected_genes,
        best_params, cv_results, output_dir="models/rnaseq/pancancer_v3"
    )

    logger.info("\n" + "=" * 60)
    logger.info("학습 완료!")
    logger.info(f"모델 저장 위치: {output_path}")
    logger.info("=" * 60)

    return final_model, cv_results


if __name__ == "__main__":
    main()
