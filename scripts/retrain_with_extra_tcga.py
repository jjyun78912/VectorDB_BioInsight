#!/usr/bin/env python3
"""
Pan-Cancer Model Retraining with Extra TCGA Data
=================================================

저성능 암종의 추가 TCGA 데이터로 Pan-Cancer 모델을 재학습합니다.

기존 모델:
- 17종 암종, ~11,000 samples
- 외부 검증 정확도: 61%

개선 목표:
- 저성능 암종(PAAD, PRAD, BLCA, OV) 데이터 추가
- 데이터 균형 맞춤
- Domain adaptation 적용
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from catboost import CatBoostClassifier
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_existing_model_data():
    """기존 모델의 학습 데이터 로드 (preprocessor에서 gene list 가져오기)"""
    model_dir = Path("models/rnaseq/pancancer")

    preprocessor_path = model_dir / "preprocessor.joblib"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        selected_genes = preprocessor.get('selected_genes', [])
        logger.info(f"Loaded {len(selected_genes)} genes from existing model")
        return selected_genes

    return None


def load_extra_tcga_data(cancer_code: str, data_dir: str = "data/tcga") -> Tuple[pd.DataFrame, List[str]]:
    """추가 TCGA 데이터 로드"""
    data_path = Path(data_dir) / cancer_code / "expression_matrix.csv"

    if not data_path.exists():
        logger.warning(f"No data found for {cancer_code}")
        return None, []

    df = pd.read_csv(data_path, index_col=0)

    # ENSG ID를 gene symbol로 변환 시도 (옵션)
    # 지금은 ENSG ID 그대로 사용

    logger.info(f"Loaded {cancer_code}: {df.shape[0]} genes x {df.shape[1]} samples")

    return df, df.columns.tolist()


def load_external_validation_data():
    """외부 검증 데이터 (GSE293591) 로드"""
    data_dir = Path("data")

    tpm_file = data_dir / "GSE293591_TPM_all_samples.tsv"
    if not tpm_file.exists():
        tpm_file = data_dir / "GSE293591_TPM_all_samples.tsv.gz"

    if not tpm_file.exists():
        logger.warning("External validation data not found")
        return None, None

    # TPM 데이터 로드
    df = pd.read_csv(tpm_file, sep='\t', index_col=0)

    # 메타데이터 로드
    meta_file = data_dir / "GSE293591_metadata_curated.csv"
    if meta_file.exists():
        meta_df = pd.read_csv(meta_file)
    else:
        logger.warning("Metadata not found")
        return df, None

    return df, meta_df


def prepare_training_data(extra_cancers: List[str] = ['PAAD', 'PRAD', 'BLCA', 'OV'],
                         n_top_genes: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """학습 데이터 준비 (추가 TCGA + 기존 데이터)"""

    all_dfs = {}

    # 1. 추가 TCGA 데이터 로드 및 공통 유전자 찾기
    for cancer in extra_cancers:
        df, samples = load_extra_tcga_data(cancer)
        if df is not None:
            all_dfs[cancer] = df

    if not all_dfs:
        logger.error("No training data available")
        return None, None, None, None, None, None, None

    # 2. 공통 유전자 찾기
    common_genes = None
    for cancer, df in all_dfs.items():
        if common_genes is None:
            common_genes = set(df.index)
        else:
            common_genes = common_genes.intersection(set(df.index))

    common_genes = sorted(list(common_genes))
    logger.info(f"Common genes across all cancers: {len(common_genes)}")

    # 3. 공통 유전자만 추출하여 병합
    all_data = []
    all_labels = []

    for cancer, df in all_dfs.items():
        df_common = df.loc[common_genes]
        for col in df_common.columns:
            all_data.append(df_common[col].values)
            all_labels.append(cancer)

    # 4. numpy array로 변환
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels)

    logger.info(f"Combined data: {X.shape[0]} samples, {X.shape[1]} genes")
    logger.info(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

    # 3. Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Gene selection (variance-based on training data)
    # Train/test split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # HVG selection on train only
    variances = np.var(X_train, axis=0)
    top_gene_idx = np.argsort(variances)[-n_top_genes:]

    X_train_hvg = X_train[:, top_gene_idx]
    X_test_hvg = X_test[:, top_gene_idx]

    # 5. Standardization (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hvg)
    X_test_scaled = scaler.transform(X_test_hvg)

    logger.info(f"After HVG selection: {X_train_scaled.shape[1]} genes")

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            le, scaler, top_gene_idx)


def train_model(X_train: np.ndarray, y_train: np.ndarray,
               class_names: List[str]) -> CatBoostClassifier:
    """CatBoost 모델 학습"""

    # Class weights 계산
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = total / (len(class_counts) * class_counts)

    logger.info(f"Class weights: {dict(zip(range(len(class_weights)), class_weights.round(2)))}")

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        class_weights=class_weights.tolist(),
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type='CPU'
    )

    # K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cv_scores.append(acc)
        logger.info(f"Fold {fold+1}: {acc:.4f}")

    logger.info(f"CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final model on all training data
    model.fit(X_train, y_train, verbose=100)

    return model


def evaluate_on_external(model, scaler, top_gene_idx, le,
                        gene_names: List[str]) -> Dict:
    """외부 검증 데이터로 평가"""

    ext_data, ext_meta = load_external_validation_data()

    if ext_data is None:
        logger.warning("External validation skipped")
        return {}

    # Gene matching (ENSG IDs should match)
    # 외부 데이터도 동일한 gene index 사용

    # TODO: 실제 구현 시 gene mapping 필요
    logger.info("External validation: Gene mapping required")

    return {}


def save_model(model, scaler, le, top_gene_idx,
              output_dir: str = "models/rnaseq/pancancer_v2"):
    """모델 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CatBoost 모델
    model.save_model(str(output_path / "catboost_model.cbm"))

    # 전처리기
    preprocessor = {
        'scaler': scaler,
        'label_encoder': le,
        'top_gene_idx': top_gene_idx,
        'selected_genes': top_gene_idx.tolist() if hasattr(top_gene_idx, 'tolist') else top_gene_idx,
        'version': '2.0'
    }
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")

    # 메타데이터
    meta = {
        'model_type': 'CatBoost Pan-Cancer v2',
        'n_classes': len(le.classes_),
        'class_names': le.classes_.tolist(),
        'n_features': len(top_gene_idx) if hasattr(top_gene_idx, '__len__') else top_gene_idx,
    }
    with open(output_path / "model_info.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved to {output_path}")


def main():
    logger.info("="*60)
    logger.info("Pan-Cancer Model Retraining with Extra TCGA Data")
    logger.info("="*60)

    # 1. 데이터 준비
    result = prepare_training_data(
        extra_cancers=['PAAD', 'PRAD', 'BLCA', 'OV'],
        n_top_genes=5000
    )

    if result[0] is None:
        logger.error("Failed to prepare training data")
        return

    X_train, X_test, y_train, y_test, le, scaler, top_gene_idx = result

    # 2. 모델 학습
    logger.info("\nTraining model...")
    model = train_model(X_train, y_train, le.classes_.tolist())

    # 3. 테스트 평가
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"\nTest Results:")
    logger.info(f"Accuracy: {test_acc:.4f}")
    logger.info(f"F1 (macro): {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 4. 모델 저장
    save_model(model, scaler, le, top_gene_idx)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
