#!/usr/bin/env python3
"""
Pan-Cancer 학습 결과 시각화
===========================

학습 완료 후 다음 그래프를 생성:
1. Confusion Matrix (히트맵)
2. Per-class Accuracy 바 차트
3. ROC Curve (multiclass)
4. Top-5 Accuracy 비교
5. 학습 요약 대시보드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_training_results(model_dir: str = "models/rnaseq/pancancer"):
    """학습 결과 로드"""
    model_path = Path(model_dir)

    with open(model_path / "training_results.json") as f:
        results = json.load(f)

    with open(model_path / "cancer_info.json") as f:
        cancer_info = json.load(f)

    return results, cancer_info


def plot_confusion_matrix(results: dict, cancer_info: dict, output_dir: str = "figures"):
    """Confusion Matrix 히트맵"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = np.array(results['metrics']['confusion_matrix'])
    class_names = results['class_names']

    # 정규화된 confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 원본 confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # 정규화된 confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion Matrix saved: {output_path / 'confusion_matrix.png'}")
    return str(output_path / "confusion_matrix.png")


def plot_per_class_accuracy(results: dict, cancer_info: dict, output_dir: str = "figures"):
    """암종별 정확도 바 차트"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = np.array(results['metrics']['confusion_matrix'])
    class_names = results['class_names']

    # 클래스별 정확도 계산
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)

    # 한글 이름 추가
    labels_korean = [f"{name}\n({cancer_info.get(name, name)})" for name in class_names]

    # 정확도 순으로 정렬
    sorted_idx = np.argsort(per_class_acc)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.RdYlGn(per_class_acc[sorted_idx])
    bars = ax.barh(range(len(class_names)), per_class_acc[sorted_idx], color=colors)

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([labels_korean[i] for i in sorted_idx])
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Cancer Type Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)

    # 정확도 값 표시
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc[sorted_idx])):
        ax.text(acc + 0.01, i, f'{acc:.1%}', va='center', fontsize=9)

    # 평균선
    mean_acc = np.mean(per_class_acc)
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.1%}')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / "per_class_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Per-class Accuracy saved: {output_path / 'per_class_accuracy.png'}")
    return str(output_path / "per_class_accuracy.png")


def plot_training_summary(results: dict, cancer_info: dict, output_dir: str = "figures"):
    """학습 요약 대시보드"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))

    # 레이아웃 설정
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. 주요 메트릭 (게이지 스타일)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = results['metrics']['ensemble']
    accuracy = metrics['accuracy']

    # 원형 게이지
    theta = np.linspace(0, np.pi, 100)
    r_inner, r_outer = 0.6, 1.0

    # 배경
    ax1.fill_between(theta, r_inner, r_outer, alpha=0.2, color='gray')
    # 값
    theta_val = np.linspace(0, np.pi * accuracy, 100)
    ax1.fill_between(theta_val, r_inner, r_outer, alpha=0.8, color='#2ecc71')

    ax1.set_xlim(-0.1, np.pi + 0.1)
    ax1.set_ylim(0, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.text(np.pi/2, 0.3, f'{accuracy:.1%}', ha='center', va='center', fontsize=32, fontweight='bold')
    ax1.text(np.pi/2, -0.1, 'Accuracy', ha='center', va='center', fontsize=14)
    ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold', pad=20)

    # 2. Top-K Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    top_k = {
        'Top-1': metrics['accuracy'],
        'Top-3': metrics['top_3_accuracy'],
        'Top-5': metrics['top_5_accuracy'],
    }

    colors = ['#3498db', '#2ecc71', '#27ae60']
    bars = ax2.bar(top_k.keys(), top_k.values(), color=colors, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, top_k.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Top-K Accuracy', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. 모델 정보
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    info_text = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 Training Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━

    📁 Samples: {results['n_samples']:,}
    🧬 Genes: {results['n_genes']:,}
    🏥 Cancer Types: {results['n_classes']}
    📅 Date: {results['training_date'][:10]}

    ━━━━━━━━━━━━━━━━━━━━━━━━━
    📈 Performance Metrics
    ━━━━━━━━━━━━━━━━━━━━━━━━━

    Accuracy: {metrics['accuracy']:.2%}
    F1 (macro): {metrics['f1_macro']:.2%}
    Top-3 Acc: {metrics['top_3_accuracy']:.2%}
    Top-5 Acc: {metrics['top_5_accuracy']:.2%}
    """

    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # 4. Confusion Matrix (미니)
    ax4 = fig.add_subplot(gs[1, :2])
    cm = np.array(results['metrics']['confusion_matrix'])
    class_names = results['class_names']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(cm_normalized, annot=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax4, cbar_kws={'label': 'Accuracy'})
    ax4.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Predicted', fontsize=12)
    ax4.set_ylabel('Actual', fontsize=12)
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    ax4.tick_params(axis='y', rotation=0, labelsize=9)

    # 5. 클래스별 샘플 수 (파이 차트)
    ax5 = fig.add_subplot(gs[1, 2])
    samples_per_class = cm.sum(axis=1)

    # 상위 5개만 표시, 나머지는 'Others'
    sorted_idx = np.argsort(samples_per_class)[::-1]
    top_5_idx = sorted_idx[:5]
    other_sum = samples_per_class[sorted_idx[5:]].sum()

    pie_labels = [class_names[i] for i in top_5_idx] + ['Others']
    pie_values = [samples_per_class[i] for i in top_5_idx] + [other_sum]
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))

    wedges, texts, autotexts = ax5.pie(pie_values, labels=pie_labels, autopct='%1.0f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 9})
    ax5.set_title('Sample Distribution', fontsize=14, fontweight='bold')

    plt.suptitle('Pan-Cancer Classifier Training Results', fontsize=18, fontweight='bold', y=1.02)

    plt.savefig(output_path / "training_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Training Dashboard saved: {output_path / 'training_dashboard.png'}")
    return str(output_path / "training_dashboard.png")


def plot_misclassification_analysis(results: dict, cancer_info: dict, output_dir: str = "figures"):
    """오분류 분석"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = np.array(results['metrics']['confusion_matrix'])
    class_names = results['class_names']
    n_classes = len(class_names)

    # 오분류 쌍 찾기
    misclass_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                misclass_pairs.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j],
                    'true_korean': cancer_info.get(class_names[i], class_names[i]),
                    'pred_korean': cancer_info.get(class_names[j], class_names[j]),
                })

    if not misclass_pairs:
        print("✅ No misclassifications found!")
        return None

    # 오분류 수 기준 정렬
    misclass_pairs.sort(key=lambda x: x['count'], reverse=True)
    top_misclass = misclass_pairs[:10]

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"{m['true']}→{m['pred']}" for m in top_misclass]
    counts = [m['count'] for m in top_misclass]

    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(top_misclass)))
    bars = ax.barh(range(len(top_misclass)), counts, color=colors)

    ax.set_yticks(range(len(top_misclass)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Misclassification Count', fontsize=12)
    ax.set_title('Top 10 Misclassification Pairs', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for i, (bar, m) in enumerate(zip(bars, top_misclass)):
        ax.text(bar.get_width() + 0.1, i,
                f"{m['true_korean']} → {m['pred_korean']}",
                va='center', fontsize=9, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path / "misclassification_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Misclassification Analysis saved: {output_path / 'misclassification_analysis.png'}")
    return str(output_path / "misclassification_analysis.png")


def main():
    """메인 실행"""
    print("=" * 60)
    print("  Pan-Cancer Training Visualization")
    print("=" * 60)
    print()

    # 결과 로드
    try:
        results, cancer_info = load_training_results()
        print(f"✅ Loaded training results")
        print(f"   - Samples: {results['n_samples']}")
        print(f"   - Classes: {results['n_classes']}")
        print(f"   - Accuracy: {results['metrics']['ensemble']['accuracy']:.2%}")
        print()
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return

    # 출력 디렉토리
    output_dir = "models/rnaseq/pancancer/figures"

    # 시각화 생성
    files = []

    # 1. Confusion Matrix
    files.append(plot_confusion_matrix(results, cancer_info, output_dir))

    # 2. Per-class Accuracy
    files.append(plot_per_class_accuracy(results, cancer_info, output_dir))

    # 3. Training Dashboard
    files.append(plot_training_summary(results, cancer_info, output_dir))

    # 4. Misclassification Analysis
    misclass_file = plot_misclassification_analysis(results, cancer_info, output_dir)
    if misclass_file:
        files.append(misclass_file)

    print()
    print("=" * 60)
    print("  ✅ All visualizations generated!")
    print("=" * 60)
    print()
    print("Generated files:")
    for f in files:
        if f:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
