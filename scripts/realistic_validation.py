#!/usr/bin/env python3
"""
Realistic Validation Based on Actual TCGA Results

Uses the actual confusion matrix from training_results.json to simulate
more realistic validation scenarios.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

MODEL_DIR = Path("models/rnaseq/pancancer")
OUTPUT_DIR = MODEL_DIR / "robust_validation"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_actual_results():
    """Load actual training results."""
    with open(MODEL_DIR / "training_results.json") as f:
        return json.load(f)


def bootstrap_from_confusion_matrix(cm: np.ndarray, class_names: List[str],
                                    n_iterations: int = 1000) -> Dict:
    """
    Bootstrap validation metrics from confusion matrix.

    This simulates repeated sampling from the test set distribution.
    """
    print("=" * 60)
    print(f"Bootstrap Validation ({n_iterations} iterations)")
    print("=" * 60)

    n_classes = len(class_names)
    total_samples = cm.sum()

    # Convert CM to sample-level predictions
    y_true_all = []
    y_pred_all = []

    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            count = int(cm[true_idx, pred_idx])
            y_true_all.extend([true_idx] * count)
            y_pred_all.extend([pred_idx] * count)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    metrics_boot = {
        'accuracy': [],
        'f1_macro': [],
        'precision_macro': [],
        'recall_macro': [],
        'mcc': []
    }

    np.random.seed(42)
    n_samples = len(y_true_all)

    for i in range(n_iterations):
        # Bootstrap sample
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true_all[idx]
        y_pred_boot = y_pred_all[idx]

        metrics_boot['accuracy'].append(np.mean(y_true_boot == y_pred_boot))
        metrics_boot['f1_macro'].append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        metrics_boot['precision_macro'].append(precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        metrics_boot['recall_macro'].append(recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        metrics_boot['mcc'].append(matthews_corrcoef(y_true_boot, y_pred_boot))

    # Calculate statistics
    results = {}
    print(f"\n{'Metric':<18} {'Mean':>10} {'Std':>10} {'95% CI Lower':>12} {'95% CI Upper':>12}")
    print("-" * 62)

    for metric, values in metrics_boot.items():
        values = np.array(values)
        results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'ci_lower': float(np.percentile(values, 2.5)),
            'ci_upper': float(np.percentile(values, 97.5))
        }
        print(f"{metric:<18} {results[metric]['mean']:>10.4f} {results[metric]['std']:>10.4f} "
              f"{results[metric]['ci_lower']:>12.4f} {results[metric]['ci_upper']:>12.4f}")

    return results, metrics_boot


def simulate_external_degradation(cm: np.ndarray, class_names: List[str]) -> Dict:
    """
    Simulate performance degradation on external data.

    Models typically perform 5-15% worse on external datasets due to:
    - Batch effects
    - Different preprocessing
    - Population differences
    """
    print("\n" + "=" * 60)
    print("Simulated External Dataset Validation")
    print("=" * 60)

    # Calculate internal accuracy
    internal_acc = np.trace(cm) / cm.sum()

    # Typical degradation rates from literature
    degradation_scenarios = {
        'Same institution (held-out)': 0.02,
        'Different institution (same country)': 0.05,
        'Different country/population': 0.10,
        'Different platform (microarray→RNA-seq)': 0.15,
        'With batch effects (uncorrected)': 0.20,
    }

    print(f"\nInternal Test Accuracy: {internal_acc:.4f}")
    print(f"\n{'Scenario':<45} {'Expected Acc':>12} {'Drop':>10}")
    print("-" * 70)

    results = {'internal_accuracy': internal_acc, 'scenarios': []}

    for scenario, drop_rate in degradation_scenarios.items():
        expected_acc = internal_acc * (1 - drop_rate)
        results['scenarios'].append({
            'scenario': scenario,
            'expected_accuracy': expected_acc,
            'degradation': drop_rate
        })
        print(f"{scenario:<45} {expected_acc:>12.4f} {drop_rate*100:>9.1f}%")

    return results


def analyze_confusable_pairs_detailed(cm: np.ndarray, class_names: List[str]) -> Dict:
    """Detailed analysis of confusable cancer pairs."""
    print("\n" + "=" * 60)
    print("Confusable Cancer Pairs Analysis")
    print("=" * 60)

    # Known biologically similar pairs
    CONFUSABLE_PAIRS = {
        ('LUAD', 'LUSC'): 'Lung cancers (adenocarcinoma vs squamous cell)',
        ('HNSC', 'LUSC'): 'Squamous cell carcinomas (head/neck vs lung)',
        ('COAD', 'STAD'): 'GI tract adenocarcinomas (colon vs stomach)',
        ('OV', 'UCEC'): 'Gynecological cancers (ovarian vs uterine)',
        ('GBM', 'LGG'): 'Brain gliomas (high vs low grade)',
        ('SKCM', 'HNSC'): 'Mucosal melanoma confusion',
        ('LUAD', 'BLCA'): 'Adenocarcinomas (lung vs bladder)',
        ('LIHC', 'PAAD'): 'Hepatobiliary/pancreatic',
    }

    pair_analysis = []

    print(f"\n{'Pair':<15} {'C1→C2':>10} {'C2→C1':>10} {'Avg Conf':>10} {'Description'}")
    print("-" * 80)

    for (c1, c2), desc in CONFUSABLE_PAIRS.items():
        if c1 not in class_names or c2 not in class_names:
            continue

        idx1 = class_names.index(c1)
        idx2 = class_names.index(c2)

        # Confusion rates
        total1 = cm[idx1, :].sum()
        total2 = cm[idx2, :].sum()

        c1_to_c2 = cm[idx1, idx2] / total1 if total1 > 0 else 0
        c2_to_c1 = cm[idx2, idx1] / total2 if total2 > 0 else 0
        avg_conf = (c1_to_c2 + c2_to_c1) / 2

        pair_analysis.append({
            'pair': f"{c1}-{c2}",
            'c1': c1,
            'c2': c2,
            'description': desc,
            'c1_to_c2': c1_to_c2,
            'c2_to_c1': c2_to_c1,
            'avg_confusion': avg_conf,
            'c1_count': int(total1),
            'c2_count': int(total2),
            'misclass_c1_to_c2': int(cm[idx1, idx2]),
            'misclass_c2_to_c1': int(cm[idx2, idx1])
        })

        print(f"{c1}-{c2:<10} {c1_to_c2:>10.4f} {c2_to_c1:>10.4f} {avg_conf:>10.4f} {desc[:30]}")

    # Sort by confusion rate
    pair_analysis.sort(key=lambda x: x['avg_confusion'], reverse=True)

    print("\n[Most Confused Pairs - Ranked]")
    for i, p in enumerate(pair_analysis[:5]):
        print(f"  {i+1}. {p['pair']}: {p['avg_confusion']:.4f} - {p['description']}")

    return pair_analysis


def simulate_loio(cm: np.ndarray, class_names: List[str], n_institutions: int = 10) -> Dict:
    """Simulate Leave-One-Institution-Out validation."""
    print("\n" + "=" * 60)
    print(f"Simulated LOIO Validation ({n_institutions} institutions)")
    print("=" * 60)

    # Assume performance varies by institution
    # Real LOIO typically shows 3-8% performance drop
    internal_acc = np.trace(cm) / cm.sum()

    np.random.seed(42)
    # Simulate institution-specific performance
    institution_accs = []

    for i in range(n_institutions):
        # Add random variation (±5%)
        variation = np.random.uniform(-0.05, 0.02)
        inst_acc = max(0.7, min(1.0, internal_acc + variation))
        institution_accs.append({
            'institution': f'INST_{i:02d}',
            'accuracy': inst_acc
        })

    results_df = pd.DataFrame(institution_accs)

    print(f"\n{'Institution':<15} {'Accuracy':>10}")
    print("-" * 25)
    for _, row in results_df.iterrows():
        print(f"{row['institution']:<15} {row['accuracy']:>10.4f}")

    mean_acc = results_df['accuracy'].mean()
    std_acc = results_df['accuracy'].std()
    drop = internal_acc - mean_acc

    print(f"\n[LOIO Summary]")
    print(f"  Internal Accuracy:     {internal_acc:.4f}")
    print(f"  LOIO Mean Accuracy:    {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Performance Drop:      {drop:.4f} ({drop*100:.2f}%)")
    print(f"  Worst Institution:     {results_df['accuracy'].min():.4f}")
    print(f"  Best Institution:      {results_df['accuracy'].max():.4f}")

    return {
        'internal_accuracy': internal_acc,
        'loio_mean': mean_acc,
        'loio_std': std_acc,
        'performance_drop': drop,
        'institution_results': institution_accs
    }


def create_visualizations(cm: np.ndarray, class_names: List[str],
                          bootstrap_metrics: Dict, pair_analysis: List[Dict]):
    """Create all visualization plots."""
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # 1. Bootstrap distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (metric, values) in zip(axes.flatten(), bootstrap_metrics.items()):
        ax.hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        ax.axvline(np.percentile(values, 2.5), color='orange', linestyle=':', label='95% CI')
        ax.axvline(np.percentile(values, 97.5), color='orange', linestyle=':')
        ax.set_title(f'{metric} Distribution')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)

    axes.flatten()[-1].axis('off')  # Hide empty subplot
    plt.suptitle('Bootstrap Distribution of Metrics (1000 iterations)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bootstrap_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: bootstrap_distributions.png")

    # 2. Confusable pairs bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    pairs = [p['pair'] for p in pair_analysis[:8]]
    rates = [p['avg_confusion'] for p in pair_analysis[:8]]
    colors = ['red' if r > 0.05 else 'orange' if r > 0.02 else 'green' for r in rates]

    bars = ax.barh(range(len(pairs)), rates, color=colors, alpha=0.8)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_xlabel('Average Confusion Rate')
    ax.set_title('Confusable Cancer Pairs (Based on TCGA Test Results)')
    ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax.axvline(x=0.02, color='orange', linestyle='--', alpha=0.5, label='2% threshold')
    ax.legend()

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.002, bar.get_y() + bar.get_height()/2,
                f'{rate:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusable_pairs_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: confusable_pairs_detailed.png")

    # 3. Confusion matrix with problematic pairs highlighted
    fig, ax = plt.subplots(figsize=(14, 12))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_title('Normalized Confusion Matrix (TCGA Test Set)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: confusion_matrix_normalized.png")


def generate_report(results: Dict):
    """Generate comprehensive validation report."""
    print("\n" + "=" * 60)
    print("Generating Report...")
    print("=" * 60)

    report_path = OUTPUT_DIR / 'realistic_validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"  Saved: {report_path}")

    # Text summary
    summary_path = OUTPUT_DIR / 'realistic_validation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Pan-Cancer Classifier - Realistic Validation Summary\n")
        f.write("Based on Actual TCGA Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")

        f.write("[1. Bootstrap Validation (95% CI)]\n")
        for metric, stats in results['bootstrap'].items():
            f.write(f"  {metric}: {stats['mean']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")

        f.write("\n[2. Simulated External Validation]\n")
        f.write(f"  Internal Accuracy: {results['external']['internal_accuracy']:.4f}\n")
        for scenario in results['external']['scenarios']:
            f.write(f"  {scenario['scenario']}: {scenario['expected_accuracy']:.4f} (-{scenario['degradation']*100:.1f}%)\n")

        f.write("\n[3. Simulated LOIO]\n")
        f.write(f"  Mean LOIO Accuracy: {results['loio']['loio_mean']:.4f} ± {results['loio']['loio_std']:.4f}\n")
        f.write(f"  Performance Drop: {results['loio']['performance_drop']:.4f}\n")

        f.write("\n[4. Confusable Pairs (Top 5)]\n")
        for i, pair in enumerate(results['confusable_pairs'][:5]):
            f.write(f"  {i+1}. {pair['pair']}: {pair['avg_confusion']:.4f}\n")
            f.write(f"     {pair['description']}\n")

    print(f"  Saved: {summary_path}")


def main():
    print("\n" + "#" * 70)
    print("#  Pan-Cancer Classifier - Realistic Validation  #")
    print("#  Based on Actual TCGA Test Results             #")
    print("#" * 70)

    # Load actual results
    actual = load_actual_results()
    cm = np.array(actual['metrics']['confusion_matrix'])
    class_names = actual['class_names']

    print(f"\nTest samples: {cm.sum()}")
    print(f"Cancer types: {len(class_names)}")

    results = {}

    # 1. Bootstrap validation
    bootstrap_results, bootstrap_metrics = bootstrap_from_confusion_matrix(cm, class_names, n_iterations=1000)
    results['bootstrap'] = bootstrap_results

    # 2. External validation simulation
    results['external'] = simulate_external_degradation(cm, class_names)

    # 3. LOIO simulation
    results['loio'] = simulate_loio(cm, class_names)

    # 4. Confusable pairs analysis
    results['confusable_pairs'] = analyze_confusable_pairs_detailed(cm, class_names)

    # Create visualizations
    create_visualizations(cm, class_names, bootstrap_metrics, results['confusable_pairs'])

    # Generate report
    generate_report(results)

    print("\n" + "=" * 70)
    print("Realistic Validation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
