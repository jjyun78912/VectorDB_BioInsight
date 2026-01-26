#!/usr/bin/env python3
"""
Fine-tune Pan-Cancer Model with External Data
==============================================

GSE293591 외부 데이터로 기존 Pan-Cancer 모델을 Fine-tuning합니다.

전략:
1. 기존 모델 로드
2. 외부 데이터 전처리 (Quantile normalization)
3. Train/Test split (80/20)
4. CatBoost 모델 추가 학습 (warm start)
5. 성능 평가 및 저장
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from catboost import CatBoostClassifier, Pool
from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier, PanCancerPreprocessor
from rnaseq_pipeline.ml.domain_adapter import DomainAdapter

# Mapping from GSE293591 diagnosis to TCGA cancer codes
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
    'Esophageal Adenocarcinoma': 'STAD',  # ESCA → STAD
    'Urothelial_cancer': 'BLCA',
    'High grade serous carcinoma': 'OV',
    'Endometrial carcinoma': 'UCEC',
    'Squamous Cell Carcinoma (other than Lung)': 'HNSC',
}


def load_gse293591():
    """Load GSE293591 data and metadata."""
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data/external_validation/GSE293591"

    expr = pd.read_csv(data_dir / "GSE293591_TPM_all_samples.tsv", sep='\t', index_col=0)
    meta = pd.read_csv(data_dir / "metadata.csv")

    return expr, meta


def prepare_finetuning_data(classifier, expr, meta, test_size=0.2, random_state=42):
    """
    Fine-tuning용 데이터 준비

    1. Gene symbol → ENSEMBL 매핑
    2. Log transform + Quantile normalization
    3. Label encoding
    4. Train/Test split
    """
    print("\n[1/4] Preparing fine-tuning data...")

    model_dir = classifier.model_dir

    # Gene symbol → ENSEMBL 매핑
    with open(model_dir / "symbol_to_model_ensembl.json") as f:
        symbol_map = json.load(f)

    mapped_genes = []
    mapped_indices = []
    for gene in expr.index:
        if gene in symbol_map:
            mapped_genes.append(symbol_map[gene])
            mapped_indices.append(gene)

    mapped_expr = expr.loc[mapped_indices].copy()
    mapped_expr.index = mapped_genes

    # 중복 유전자 평균
    if mapped_expr.index.duplicated().any():
        mapped_expr = mapped_expr.groupby(mapped_expr.index).mean()

    print(f"  Mapped {len(mapped_expr)} genes to ENSEMBL")

    # Log transform
    log_expr = np.log2(mapped_expr + 1)

    # Quantile normalization
    adapter = DomainAdapter(method='quantile')
    adapted_expr = adapter.transform(log_expr)
    print(f"  Applied quantile normalization")

    # Sample → Cancer type 매핑
    sample_to_cancer = {}
    for i, row in meta.iterrows():
        diagnosis = row['diagnosis']
        tcga = DIAGNOSIS_TO_TCGA.get(diagnosis, None)
        if tcga:
            col_name = f"Sample_{i+1:03d}"
            if col_name in adapted_expr.columns:
                sample_to_cancer[col_name] = tcga

    # 유효한 샘플만 선택 (모델이 학습한 암종만)
    valid_cancers = set(classifier.preprocessor.label_encoder.classes_)
    valid_samples = []
    valid_labels = []

    for sample, cancer in sample_to_cancer.items():
        if cancer in valid_cancers:
            valid_samples.append(sample)
            valid_labels.append(cancer)

    print(f"  Valid samples: {len(valid_samples)} (out of {len(sample_to_cancer)})")

    # 데이터 필터링
    X_df = adapted_expr[valid_samples]
    y_labels = np.array(valid_labels)

    # Label encoding
    y_encoded = classifier.preprocessor.label_encoder.transform(y_labels)

    # Feature 추출 (모델이 학습한 유전자만)
    selected_genes = classifier.preprocessor.selected_genes

    X = pd.DataFrame(0.0, index=valid_samples, columns=selected_genes)
    matched = 0
    for gene in selected_genes:
        if gene in X_df.index:
            X[gene] = X_df.loc[gene].values
            matched += 1
        elif gene.split('.')[0] in [g.split('.')[0] for g in X_df.index]:
            # 버전 번호 무시 매칭
            base_gene = gene.split('.')[0]
            for idx_gene in X_df.index:
                if idx_gene.split('.')[0] == base_gene:
                    X[gene] = X_df.loc[idx_gene].values
                    matched += 1
                    break

    print(f"  Feature matching: {matched}/{len(selected_genes)} genes")

    # Standardization (기존 scaler 사용)
    X_scaled = classifier.preprocessor.scaler.transform(X.values)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 클래스 분포
    train_counts = np.bincount(y_train, minlength=len(valid_cancers))
    print(f"  Train class distribution:")
    for i, count in enumerate(train_counts):
        if count > 0:
            cancer = classifier.preprocessor.label_encoder.inverse_transform([i])[0]
            print(f"    {cancer}: {count}")

    return X_train, X_test, y_train, y_test, selected_genes


def finetune_model(classifier, X_train, y_train, X_test, y_test, feature_names,
                   learning_rate=0.01, iterations=100):
    """
    CatBoost 모델 Fine-tuning

    기존 모델을 기반으로 추가 학습 (warm start 효과)
    """
    print("\n[2/4] Fine-tuning CatBoost model...")

    # 기존 CatBoost 모델 가져오기
    original_model = classifier.ensemble.models['catboost']

    # 새 모델 생성 (기존 모델 복사)
    finetuned_model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=learning_rate,  # 낮은 학습률로 미세조정
        loss_function='MultiClass',
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        # Fine-tuning 옵션
        early_stopping_rounds=20,
    )

    # 학습 데이터 풀
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    test_pool = Pool(X_test, y_test, feature_names=feature_names)

    # Fine-tuning (init_model로 기존 모델 사용)
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max iterations: {iterations}")

    finetuned_model.fit(
        train_pool,
        eval_set=test_pool,
        init_model=original_model,  # 기존 모델에서 시작
        verbose=False
    )

    print(f"  Fine-tuning complete!")

    return finetuned_model


def evaluate_finetuned_model(classifier, finetuned_model, X_test, y_test):
    """
    Fine-tuned 모델 평가
    """
    print("\n[3/4] Evaluating fine-tuned model...")

    # 기존 모델 예측
    original_model = classifier.ensemble.models['catboost']
    y_pred_original = original_model.predict(X_test).flatten().astype(int)

    # Fine-tuned 모델 예측
    y_pred_finetuned = finetuned_model.predict(X_test).flatten().astype(int)

    # 성능 비교
    acc_original = accuracy_score(y_test, y_pred_original)
    acc_finetuned = accuracy_score(y_test, y_pred_finetuned)

    f1_original = f1_score(y_test, y_pred_original, average='macro')
    f1_finetuned = f1_score(y_test, y_pred_finetuned, average='macro')

    print(f"\n  Performance Comparison:")
    print(f"  {'Metric':<15} {'Original':>12} {'Fine-tuned':>12} {'Change':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<15} {acc_original:>11.1%} {acc_finetuned:>11.1%} {acc_finetuned-acc_original:>+9.1%}")
    print(f"  {'F1 (macro)':<15} {f1_original:>11.1%} {f1_finetuned:>11.1%} {f1_finetuned-f1_original:>+9.1%}")

    # 클래스별 성능
    class_names = classifier.preprocessor.label_encoder.classes_
    report = classification_report(y_test, y_pred_finetuned,
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)

    print(f"\n  Per-class F1 (Fine-tuned):")
    for cancer in class_names:
        if cancer in report:
            f1 = report[cancer]['f1-score']
            support = report[cancer]['support']
            if support > 0:
                print(f"    {cancer}: {f1:.1%} (n={int(support)})")

    return {
        'original': {'accuracy': acc_original, 'f1_macro': f1_original},
        'finetuned': {'accuracy': acc_finetuned, 'f1_macro': f1_finetuned},
        'classification_report': report
    }


def save_finetuned_model(classifier, finetuned_model, eval_results, output_dir):
    """
    Fine-tuned 모델 저장
    """
    print("\n[4/4] Saving fine-tuned model...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CatBoost 모델 저장
    model_path = output_dir / "catboost_finetuned.cbm"
    finetuned_model.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # 메타데이터 저장
    metadata = {
        'base_model': str(classifier.model_dir),
        'finetuning_date': datetime.now().isoformat(),
        'finetuning_data': 'GSE293591',
        'performance': eval_results,
    }

    with open(output_dir / "finetuning_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Metadata saved: {output_dir / 'finetuning_metadata.json'}")

    return output_dir


def validate_on_full_dataset(classifier, finetuned_model, test_ratio=0.3):
    """
    전체 GSE293591 데이터셋에서 검증 (Fine-tuning에 사용하지 않은 샘플 포함)
    """
    print("\n" + "="*70)
    print("Validation on Full External Dataset")
    print("="*70)

    # 데이터 로드
    expr, meta = load_gse293591()
    model_dir = classifier.model_dir

    # 전처리
    with open(model_dir / "symbol_to_model_ensembl.json") as f:
        symbol_map = json.load(f)

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

    # Quantile normalization
    log_expr = np.log2(mapped_expr + 1)
    adapter = DomainAdapter(method='quantile')
    adapted_expr = adapter.transform(log_expr)

    # 샘플별 예측
    valid_cancers = set(classifier.preprocessor.label_encoder.classes_)
    results_original = defaultdict(lambda: {'correct': 0, 'total': 0})
    results_finetuned = defaultdict(lambda: {'correct': 0, 'total': 0})

    original_model = classifier.ensemble.models['catboost']

    for i, row in meta.iterrows():
        diagnosis = row['diagnosis']
        tcga = DIAGNOSIS_TO_TCGA.get(diagnosis, None)
        if tcga and tcga in valid_cancers:
            col_name = f"Sample_{i+1:03d}"
            if col_name not in adapted_expr.columns:
                continue

            # Feature 추출
            sample_data = adapted_expr[col_name]
            X = pd.DataFrame(0.0, index=[col_name], columns=classifier.preprocessor.selected_genes)
            for gene in classifier.preprocessor.selected_genes:
                if gene in sample_data.index:
                    X.loc[col_name, gene] = sample_data[gene]

            X_scaled = classifier.preprocessor.scaler.transform(X.values)

            # 예측
            pred_original = original_model.predict(X_scaled).flatten()[0]
            pred_finetuned = finetuned_model.predict(X_scaled).flatten()[0]

            true_label = classifier.preprocessor.label_encoder.transform([tcga])[0]

            results_original[tcga]['total'] += 1
            results_finetuned[tcga]['total'] += 1

            if pred_original == true_label:
                results_original[tcga]['correct'] += 1
            if pred_finetuned == true_label:
                results_finetuned[tcga]['correct'] += 1

    # 결과 출력
    print(f"\n{'Cancer':<8} {'Original':>12} {'Fine-tuned':>12} {'Change':>10}")
    print("-" * 45)

    total_orig = 0
    total_fine = 0
    total_n = 0

    for cancer in sorted(results_original.keys()):
        orig = results_original[cancer]
        fine = results_finetuned[cancer]

        acc_orig = orig['correct'] / orig['total'] * 100 if orig['total'] > 0 else 0
        acc_fine = fine['correct'] / fine['total'] * 100 if fine['total'] > 0 else 0

        change = acc_fine - acc_orig
        marker = "+" if change > 0 else ""

        print(f"{cancer:<8} {acc_orig:>10.1f}% {acc_fine:>10.1f}% {marker}{change:>8.1f}%")

        total_orig += orig['correct']
        total_fine += fine['correct']
        total_n += orig['total']

    print("-" * 45)
    overall_orig = total_orig / total_n * 100
    overall_fine = total_fine / total_n * 100
    change = overall_fine - overall_orig
    marker = "+" if change > 0 else ""
    print(f"{'OVERALL':<8} {overall_orig:>10.1f}% {overall_fine:>10.1f}% {marker}{change:>8.1f}%")

    return {
        'original_accuracy': overall_orig,
        'finetuned_accuracy': overall_fine,
        'improvement': change
    }


def main():
    print("="*70)
    print("Pan-Cancer Model Fine-tuning with GSE293591")
    print("="*70)

    # 모델 로드
    model_dir = Path("models/rnaseq/pancancer")
    classifier = PanCancerClassifier(model_dir)
    classifier.load()

    print(f"\nBase model: {model_dir}")
    print(f"Cancer types: {len(classifier.preprocessor.label_encoder.classes_)}")

    # 데이터 로드
    expr, meta = load_gse293591()
    print(f"External data: {expr.shape[1]} samples, {expr.shape[0]} genes")

    # Fine-tuning 데이터 준비
    X_train, X_test, y_train, y_test, feature_names = prepare_finetuning_data(
        classifier, expr, meta, test_size=0.2, random_state=42
    )

    # Fine-tuning
    finetuned_model = finetune_model(
        classifier, X_train, y_train, X_test, y_test, feature_names,
        learning_rate=0.01,  # 낮은 학습률
        iterations=200       # 적은 iterations
    )

    # 평가
    eval_results = evaluate_finetuned_model(classifier, finetuned_model, X_test, y_test)

    # 저장
    output_dir = save_finetuned_model(
        classifier, finetuned_model, eval_results,
        output_dir="models/rnaseq/pancancer/finetuned"
    )

    # 전체 데이터셋 검증
    full_validation = validate_on_full_dataset(classifier, finetuned_model)

    print("\n" + "="*70)
    print("Fine-tuning Complete!")
    print("="*70)
    print(f"\nImprovement: {full_validation['original_accuracy']:.1f}% → {full_validation['finetuned_accuracy']:.1f}%")
    print(f"Change: {full_validation['improvement']:+.1f}%")
    print(f"\nFine-tuned model saved to: {output_dir}")


if __name__ == "__main__":
    main()
