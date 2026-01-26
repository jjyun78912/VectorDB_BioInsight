#!/usr/bin/env python3
"""
Transfer Learning: TCGA + External Data Mixed Training
=======================================================

TCGA 데이터와 GSE293591 외부 데이터를 혼합하여 새로운 Pan-Cancer 모델을 학습합니다.

전략:
1. TCGA 데이터 로드 (17개 암종)
2. GSE293591 외부 데이터 추가 (13개 암종, 약 10% 비율)
3. Data Leakage 방지 전처리 (Split first → HVG → Scale)
4. CatBoost 학습 (class weight 조정)
5. 외부 검증 및 저장
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from catboost import CatBoostClassifier, Pool
from rnaseq_pipeline.ml.domain_adapter import DomainAdapter

# GSE293591 매핑
DIAGNOSIS_TO_TCGA = {
    'Breast cancer': 'BRCA',
    'Colorectal Adenocarcinoma': 'COAD',
    'Lung Adenocarcinoma': 'LUAD',
    'Gastric Adenocarcinoma': 'STAD',
    'Squamous Cell Carcinoma of Lung': 'LUSC',
    'Pancreatic Adenocarcinoma': 'PAAD',
    'Liver carcinoma and Cholangiocarcinoma': 'LIHC',
    'Prostate adenocarcinoma': 'PRAD',
    'Renal cell carcinoma': 'KIRC',
    'Esophageal Adenocarcinoma': 'STAD',
    'Urothelial_cancer': 'BLCA',
    'High grade serous carcinoma': 'OV',
    'Endometrial carcinoma': 'UCEC',
    'Squamous Cell Carcinoma (other than Lung)': 'HNSC',
}


def load_tcga_data():
    """TCGA Pan-Cancer 데이터 로드"""
    print("\n[1/6] Loading TCGA data...")

    data_dir = Path("data/tcga/pancancer")

    # Count matrix (gene_id × samples)
    counts = pd.read_csv(data_dir / "pancancer_counts.csv", index_col=0)
    metadata = pd.read_csv(data_dir / "pancancer_metadata.csv")

    with open(data_dir / "label_mapping.json") as f:
        label_mapping = json.load(f)

    print(f"  TCGA: {counts.shape[1]} samples, {counts.shape[0]} genes")
    print(f"  Cancer types: {metadata['cancer_type'].nunique()}")

    return counts, metadata, label_mapping


def load_external_data():
    """GSE293591 외부 데이터 로드 및 전처리"""
    print("\n[2/6] Loading external data (GSE293591)...")

    data_dir = Path("data/external_validation/GSE293591")

    expr = pd.read_csv(data_dir / "GSE293591_TPM_all_samples.tsv", sep='\t', index_col=0)
    meta = pd.read_csv(data_dir / "metadata.csv")

    # Gene symbol → ENSEMBL 매핑 로드
    model_dir = Path("models/rnaseq/pancancer")
    with open(model_dir / "symbol_to_model_ensembl.json") as f:
        symbol_map = json.load(f)

    # Symbol → ENSEMBL 변환
    mapped_genes = []
    mapped_indices = []
    for gene in expr.index:
        if gene in symbol_map:
            mapped_genes.append(symbol_map[gene])
            mapped_indices.append(gene)

    mapped_expr = expr.loc[mapped_indices].copy()
    mapped_expr.index = mapped_genes

    if mapped_expr.index.duplicated().any():
        mapped_expr = mapped_expr.groupby(mapped_expr.index).mean()

    # Sample → Cancer mapping
    sample_cancer = {}
    for i, row in meta.iterrows():
        diagnosis = row['diagnosis']
        tcga = DIAGNOSIS_TO_TCGA.get(diagnosis, None)
        if tcga:
            col_name = f"Sample_{i+1:03d}"
            if col_name in mapped_expr.columns:
                sample_cancer[col_name] = tcga

    # 유효 샘플만 선택
    valid_samples = list(sample_cancer.keys())
    valid_labels = [sample_cancer[s] for s in valid_samples]

    external_counts = mapped_expr[valid_samples]

    print(f"  External: {external_counts.shape[1]} samples, {external_counts.shape[0]} genes")
    print(f"  Cancer types: {len(set(valid_labels))}")

    return external_counts, valid_samples, valid_labels


def merge_datasets(tcga_counts, tcga_meta, external_counts, external_samples, external_labels,
                   external_ratio=0.15):
    """
    TCGA와 외부 데이터 병합

    Args:
        external_ratio: 외부 데이터 비율 (0.15 = 15%)
    """
    print("\n[3/6] Merging TCGA + External data...")

    # TCGA 샘플 추출
    tcga_samples = tcga_counts.columns.tolist()
    tcga_meta_indexed = tcga_meta.set_index('barcode')
    tcga_labels = [tcga_meta_indexed.loc[s, 'cancer_type'] for s in tcga_samples]

    # 공통 유전자 찾기
    common_genes = list(set(tcga_counts.index) & set(external_counts.index))
    print(f"  Common genes: {len(common_genes)}")

    # 외부 데이터 서브샘플링 (TCGA 대비 비율 맞춤)
    n_external_target = int(len(tcga_samples) * external_ratio)
    if len(external_samples) > n_external_target:
        # Stratified 서브샘플링
        np.random.seed(42)
        indices = np.random.choice(len(external_samples), n_external_target, replace=False)
        external_samples = [external_samples[i] for i in indices]
        external_labels = [external_labels[i] for i in indices]

    print(f"  TCGA samples: {len(tcga_samples)}")
    print(f"  External samples: {len(external_samples)} ({len(external_samples)/len(tcga_samples)*100:.1f}%)")

    # 데이터 병합
    tcga_subset = tcga_counts.loc[common_genes, tcga_samples]
    external_subset = external_counts.loc[common_genes, external_samples]

    # TPM이 아닌 외부 데이터는 log transform 필요
    # TCGA는 raw counts, external은 TPM이므로 별도 처리
    # → 두 데이터 모두 CPM + log2로 통일

    # TCGA: raw counts → CPM → log2
    tcga_lib_sizes = tcga_subset.sum(axis=0)
    tcga_cpm = tcga_subset * 1e6 / tcga_lib_sizes
    tcga_log = np.log2(tcga_cpm + 1)

    # External: TPM → log2 (이미 정규화됨)
    external_log = np.log2(external_subset + 1)

    # Quantile normalization 적용 (외부 데이터만)
    adapter = DomainAdapter(method='quantile')
    external_normalized = adapter.transform(external_log)

    # 병합
    all_samples = tcga_samples + external_samples
    all_labels = tcga_labels + external_labels

    merged_counts = pd.concat([tcga_log, external_normalized], axis=1)

    # Source 구분
    source_labels = ['TCGA'] * len(tcga_samples) + ['External'] * len(external_samples)

    print(f"  Merged: {merged_counts.shape[1]} total samples")

    return merged_counts, all_samples, all_labels, source_labels, common_genes


def train_transfer_model(merged_counts, all_labels, source_labels, common_genes,
                         n_top_genes=5000, test_size=0.2, random_state=42):
    """
    Transfer Learning 모델 학습 (Data Leakage 방지)
    """
    print("\n[4/6] Training Transfer Learning model...")

    # Label encoding
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(all_labels)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)

    print(f"  Classes: {n_classes}")

    # ========================================
    # STEP 1: Train/Test Split FIRST (Data Leakage 방지!)
    # ========================================
    sample_indices = np.arange(len(all_labels))
    train_idx, test_idx = train_test_split(
        sample_indices, test_size=test_size, random_state=random_state,
        stratify=y_all
    )

    train_samples = [merged_counts.columns[i] for i in train_idx]
    test_samples = [merged_counts.columns[i] for i in test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    # Source 분포 확인
    train_sources = [source_labels[i] for i in train_idx]
    test_sources = [source_labels[i] for i in test_idx]
    print(f"  Train: {len(train_samples)} (TCGA: {train_sources.count('TCGA')}, External: {train_sources.count('External')})")
    print(f"  Test: {len(test_samples)} (TCGA: {test_sources.count('TCGA')}, External: {test_sources.count('External')})")

    # ========================================
    # STEP 2: HVG Selection on TRAIN only
    # ========================================
    train_data = merged_counts[train_samples]
    test_data = merged_counts[test_samples]

    # Train 데이터의 분산으로 유전자 선택
    variances = train_data.var(axis=1)
    selected_genes = variances.nlargest(n_top_genes).index.tolist()
    print(f"  Selected {len(selected_genes)} HVGs (from TRAIN only)")

    X_train_raw = train_data.loc[selected_genes].T.values
    X_test_raw = test_data.loc[selected_genes].T.values

    # ========================================
    # STEP 3: Standardization (fit on TRAIN only)
    # ========================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"  Standardization: fit on train, transform on test")

    # ========================================
    # STEP 4: CatBoost Training
    # ========================================
    print(f"\n  Training CatBoost...")

    # Class weights 계산 (불균형 보정)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        class_weights=class_weights.tolist(),
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False,
        early_stopping_rounds=50,
    )

    train_pool = Pool(X_train, y_train, feature_names=selected_genes)
    test_pool = Pool(X_test, y_test, feature_names=selected_genes)

    model.fit(train_pool, eval_set=test_pool, verbose=100)

    # ========================================
    # STEP 5: Evaluation
    # ========================================
    print(f"\n  Evaluating...")

    y_pred = model.predict(X_test).flatten().astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n  Test Accuracy: {acc:.1%}")
    print(f"  Test F1 (macro): {f1:.1%}")

    # External 샘플만 따로 평가
    external_mask = np.array([s == 'External' for s in test_sources])
    if external_mask.sum() > 0:
        y_test_ext = y_test[external_mask]
        y_pred_ext = y_pred[external_mask]
        acc_ext = accuracy_score(y_test_ext, y_pred_ext)
        print(f"  External-only Accuracy: {acc_ext:.1%} ({external_mask.sum()} samples)")

    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'selected_genes': selected_genes,
        'metrics': {
            'accuracy': acc,
            'f1_macro': f1,
            'external_accuracy': acc_ext if external_mask.sum() > 0 else None,
        },
        'class_names': class_names,
    }


def validate_on_gse293591(trained_result, merged_counts, all_labels, source_labels, all_samples):
    """
    학습에 사용하지 않은 GSE293591 샘플로 검증
    (전체 데이터셋 재로드하여 검증)
    """
    print("\n[5/6] Validating on full GSE293591...")

    # GSE293591 전체 데이터 로드
    external_counts, external_samples, external_labels = load_external_data()

    model = trained_result['model']
    scaler = trained_result['scaler']
    label_encoder = trained_result['label_encoder']
    selected_genes = trained_result['selected_genes']

    # 유효 암종만 선택
    valid_cancers = set(label_encoder.classes_)

    results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for i, (sample, label) in enumerate(zip(external_samples, external_labels)):
        if label not in valid_cancers:
            continue

        # Feature 추출
        X = pd.DataFrame(0.0, index=[sample], columns=selected_genes)
        for gene in selected_genes:
            if gene in external_counts.index:
                X.loc[sample, gene] = external_counts.loc[gene, sample]

        # Log transform (TPM → log2)
        X_log = np.log2(X + 1)

        # Quantile normalization
        adapter = DomainAdapter(method='quantile')
        X_norm = adapter.transform(X_log.T).T

        # Standardization
        X_scaled = scaler.transform(X_norm.values)

        # Prediction
        pred = model.predict(X_scaled).flatten()[0]
        true_label = label_encoder.transform([label])[0]

        results[label]['total'] += 1
        if pred == true_label:
            results[label]['correct'] += 1

    # 결과 출력
    print(f"\n  {'Cancer':<8} {'Accuracy':>10} {'Samples':>10}")
    print(f"  {'-'*30}")

    total_correct = 0
    total_samples = 0

    for cancer in sorted(results.keys()):
        r = results[cancer]
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"  {cancer:<8} {acc:>9.1f}% {r['total']:>10}")
        total_correct += r['correct']
        total_samples += r['total']

    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"  {'-'*30}")
    print(f"  {'OVERALL':<8} {overall_acc:>9.1f}% {total_samples:>10}")

    return {'overall_accuracy': overall_acc, 'per_cancer': dict(results)}


def save_transfer_model(trained_result, validation_result, output_dir):
    """
    Transfer Learning 모델 저장
    """
    print("\n[6/6] Saving Transfer Learning model...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    # Model 저장
    trained_result['model'].save_model(str(output_dir / "catboost.cbm"))

    # Preprocessor 저장
    preprocessor = {
        'scaler': trained_result['scaler'],
        'label_encoder': trained_result['label_encoder'],
        'selected_genes': trained_result['selected_genes'],
        'n_top_genes': len(trained_result['selected_genes']),
        'log_transform': True,
        'normalize': 'cpm',
        'is_fitted': True,
    }
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    # Metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'Transfer Learning (TCGA + GSE293591)',
        'n_classes': len(trained_result['class_names']),
        'class_names': trained_result['class_names'],
        'training_metrics': trained_result['metrics'],
        'validation_metrics': validation_result,
    }

    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Ensemble 메타데이터 (기존 구조 호환)
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(exist_ok=True)

    trained_result['model'].save_model(str(ensemble_dir / "catboost.cbm"))

    ensemble_meta = {
        'n_classes': len(trained_result['class_names']),
        'class_names': trained_result['class_names'],
        'model_weights': {'catboost': 1.0},
        'models_available': ['catboost'],
        'random_state': 42,
    }

    with open(ensemble_dir / "ensemble_metadata.json", 'w') as f:
        json.dump(ensemble_meta, f, indent=2)

    # Cancer info
    cancer_info = {cancer: {'korean': cancer} for cancer in trained_result['class_names']}
    with open(output_dir / "cancer_info.json", 'w') as f:
        json.dump(cancer_info, f, indent=2, ensure_ascii=False)

    print(f"  Model saved to: {output_dir}")

    return output_dir


def main():
    print("="*70)
    print("Transfer Learning: TCGA + External Data")
    print("="*70)

    # 1. TCGA 데이터 로드
    tcga_counts, tcga_meta, label_mapping = load_tcga_data()

    # 2. 외부 데이터 로드
    external_counts, external_samples, external_labels = load_external_data()

    # 3. 데이터 병합
    merged_counts, all_samples, all_labels, source_labels, common_genes = merge_datasets(
        tcga_counts, tcga_meta,
        external_counts, external_samples, external_labels,
        external_ratio=0.15  # 외부 데이터 15% 비율
    )

    # 4. 모델 학습
    trained_result = train_transfer_model(
        merged_counts, all_labels, source_labels, common_genes,
        n_top_genes=5000, test_size=0.2
    )

    # 5. GSE293591 전체 검증
    validation_result = validate_on_gse293591(
        trained_result, merged_counts, all_labels, source_labels, all_samples
    )

    # 6. 모델 저장
    output_dir = save_transfer_model(
        trained_result, validation_result,
        output_dir="models/rnaseq/pancancer_transfer"
    )

    print("\n" + "="*70)
    print("Transfer Learning Complete!")
    print("="*70)
    print(f"\nTest Accuracy: {trained_result['metrics']['accuracy']:.1%}")
    print(f"External Validation: {validation_result['overall_accuracy']:.1f}%")
    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()
