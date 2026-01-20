#!/usr/bin/env python3
"""
Quick Robust Validation for Pan-Cancer Classifier
- Uses fewer features and iterations for speed
- Still demonstrates the validation concepts
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Faster than CatBoost for demo
import warnings
warnings.filterwarnings('ignore')


class QuickRobustValidator:
    """Quick validation using RandomForest for speed."""

    CONFUSABLE_PAIRS = {
        ('LUAD', 'LUSC'): 'Lung cancers (adenocarcinoma vs squamous)',
        ('HNSC', 'LUSC'): 'Squamous cell carcinomas',
        ('COAD', 'STAD'): 'GI tract adenocarcinomas',
        ('OV', 'UCEC'): 'Gynecological (Müllerian origin)',
        ('GBM', 'LGG'): 'Brain gliomas (grade difference)',
    }

    def __init__(self, model_dir: str = "models/rnaseq/pancancer"):
        self.model_dir = Path(model_dir)
        self.output_dir = self.model_dir / "robust_validation"
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self._load_data()

    def _load_data(self):
        """Load or generate data."""
        print("=" * 60)
        print("Loading Data...")
        print("=" * 60)

        with open(self.model_dir / "training_results.json") as f:
            self.training_results = json.load(f)

        self.class_names = self.training_results.get("class_names", [])
        print(f"  Classes: {len(self.class_names)} cancer types")

        # Generate synthetic data (faster than loading real TCGA)
        n_samples = 2000  # Reduced for speed
        n_genes = 500     # Reduced features

        print(f"  Generating synthetic data: {n_samples} samples, {n_genes} genes")

        np.random.seed(42)
        n_per_class = n_samples // len(self.class_names)

        X_list = []
        y_list = []

        for i, cancer in enumerate(self.class_names):
            # Each cancer has distinct pattern
            mean_shift = np.random.randn(n_genes) * 2.0
            # Add some class-specific signature genes
            signature_genes = np.random.choice(n_genes, 20, replace=False)
            mean_shift[signature_genes] += np.random.randn(20) * 3.0

            X_class = np.random.randn(n_per_class, n_genes) * 0.5 + mean_shift
            X_list.append(X_class)
            y_list.extend([cancer] * n_per_class)

        self.X = np.vstack(X_list)
        self.y = np.array(y_list)

        # Shuffle
        idx = np.random.permutation(len(self.y))
        self.X = self.X[idx]
        self.y = self.y[idx]

        # Create institution labels (10 institutions)
        self.institutions = np.array([f"INST_{i % 10:02d}" for i in range(len(self.y))])
        np.random.shuffle(self.institutions)

        print(f"  Data shape: {self.X.shape}")

    def run_repeated_cv(self, n_splits: int = 5, n_repeats: int = 5) -> Dict:
        """Run repeated stratified K-fold CV."""
        print("\n" + "=" * 60)
        print(f"Running {n_splits}-Fold CV × {n_repeats} Repeats...")
        print("=" * 60)

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        all_metrics = defaultdict(list)

        total_folds = n_splits * n_repeats
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(self.X, y_encoded)):
            if (fold_idx + 1) % 5 == 0:
                print(f"  Progress: {fold_idx + 1}/{total_folds}")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Use RandomForest (faster)
            model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            all_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            all_metrics['f1_macro'].append(f1_score(y_val, y_pred, average='macro'))
            all_metrics['precision'].append(precision_score(y_val, y_pred, average='macro'))
            all_metrics['recall'].append(recall_score(y_val, y_pred, average='macro'))
            all_metrics['mcc'].append(matthews_corrcoef(y_val, y_pred))

        # Summary
        summary = {}
        print("\n[Repeated CV Results]")
        print(f"  {'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI Lower':>12} {'95% CI Upper':>12}")
        print("  " + "-" * 60)

        for metric, values in all_metrics.items():
            values = np.array(values)
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, 2.5)),
                'ci_upper': float(np.percentile(values, 97.5))
            }
            print(f"  {metric:<15} {summary[metric]['mean']:>10.4f} {summary[metric]['std']:>10.4f} "
                  f"{summary[metric]['ci_lower']:>12.4f} {summary[metric]['ci_upper']:>12.4f}")

        self._plot_cv_results(all_metrics, n_splits, n_repeats)
        self.results['repeated_cv'] = {'summary': summary, 'n_splits': n_splits, 'n_repeats': n_repeats}
        return summary

    def _plot_cv_results(self, metrics: Dict, n_splits: int, n_repeats: int):
        """Plot CV results."""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_df = pd.DataFrame(metrics)
        bp = ax.boxplot([metrics_df[col] for col in metrics_df.columns],
                       labels=metrics_df.columns, patch_artist=True)

        colors = ['steelblue', 'forestgreen', 'coral', 'purple', 'gold']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Score')
        ax.set_title(f'{n_splits}-Fold CV × {n_repeats} Repeats: Metric Distribution')
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'repeated_cv_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: repeated_cv_results.png")

    def run_loio_validation(self) -> Dict:
        """Leave-One-Institution-Out validation."""
        print("\n" + "=" * 60)
        print("Running Leave-One-Institution-Out Validation...")
        print("=" * 60)

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)

        unique_institutions = np.unique(self.institutions)
        print(f"  Testing {len(unique_institutions)} institutions")

        results = []

        for inst in unique_institutions:
            test_mask = self.institutions == inst
            train_mask = ~test_mask

            X_train, X_test = self.X[train_mask], self.X[test_mask]
            y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({'institution': inst, 'n_samples': int(test_mask.sum()),
                          'accuracy': acc, 'f1_macro': f1})
            print(f"  {inst}: n={test_mask.sum():4d}, Acc={acc:.4f}, F1={f1:.4f}")

        results_df = pd.DataFrame(results)

        # Compare to standard CV
        std_cv_acc = self.results.get('repeated_cv', {}).get('summary', {}).get('accuracy', {}).get('mean', 0.95)
        loio_mean = results_df['accuracy'].mean()
        drop = std_cv_acc - loio_mean

        print("\n[LOIO Summary]")
        print(f"  Mean Accuracy: {loio_mean:.4f} ± {results_df['accuracy'].std():.4f}")
        print(f"  Standard CV:   {std_cv_acc:.4f}")
        print(f"  Performance Drop: {drop:.4f} ({drop*100:.2f}%)")

        self._plot_loio_results(results_df)

        self.results['loio'] = {
            'results': results,
            'mean_accuracy': float(loio_mean),
            'std_accuracy': float(results_df['accuracy'].std()),
            'performance_drop': float(drop)
        }
        return self.results['loio']

    def _plot_loio_results(self, results_df: pd.DataFrame):
        """Plot LOIO results."""
        fig, ax = plt.subplots(figsize=(12, 5))

        x = range(len(results_df))
        bars = ax.bar(x, results_df['accuracy'], color='steelblue', alpha=0.7)

        mean_acc = results_df['accuracy'].mean()
        ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(results_df['institution'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Leave-One-Institution-Out Validation')
        ax.legend()
        ax.set_ylim(0.7, 1.0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'loio_validation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: loio_validation.png")

    def simulate_external_validation(self, noise_level: float = 0.3) -> Dict:
        """Simulate external dataset with noise."""
        print("\n" + "=" * 60)
        print(f"Simulating External Dataset (noise={noise_level})...")
        print("=" * 60)

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)

        # Train on full data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=42, n_jobs=-1)
        model.fit(X_scaled, y_encoded)

        # Internal test
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2,
                                                random_state=42, stratify=y_encoded)
        internal_acc = accuracy_score(y_test, model.predict(X_test))

        # External test with noise
        np.random.seed(456)
        n_external = len(self.X) // 5
        sample_idx = np.random.choice(len(self.X), n_external, replace=False)

        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        noise_results = []

        for nl in noise_levels:
            X_noisy = self.X[sample_idx].copy()
            X_noisy += np.random.randn(*X_noisy.shape) * nl
            X_noisy_scaled = scaler.transform(X_noisy)

            y_ext = y_encoded[sample_idx]
            acc = accuracy_score(y_ext, model.predict(X_noisy_scaled))
            noise_results.append({'noise_level': nl, 'accuracy': acc})

        print("\n[Noise Robustness]")
        print(f"  {'Noise':<10} {'Accuracy':>10} {'Drop':>10}")
        print("  " + "-" * 30)
        for r in noise_results:
            drop = internal_acc - r['accuracy']
            print(f"  {r['noise_level']:<10.1f} {r['accuracy']:>10.4f} {drop:>10.4f}")

        self._plot_external_validation(noise_results, internal_acc)

        self.results['external'] = {
            'internal_accuracy': float(internal_acc),
            'noise_results': noise_results
        }
        return self.results['external']

    def _plot_external_validation(self, noise_results: List[Dict], internal_acc: float):
        """Plot external validation."""
        fig, ax = plt.subplots(figsize=(10, 6))

        noise_levels = [r['noise_level'] for r in noise_results]
        accuracies = [r['accuracy'] for r in noise_results]

        ax.plot(noise_levels, accuracies, 'o-', markersize=10, linewidth=2, color='steelblue')
        ax.axhline(y=internal_acc, color='green', linestyle='--', label=f'Internal: {internal_acc:.3f}')
        ax.fill_between(noise_levels, accuracies, internal_acc, alpha=0.3, color='orange')

        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Robustness to External Data Variability')
        ax.legend()
        ax.set_ylim(0.3, 1.0)

        for nl, acc in zip(noise_levels, accuracies):
            ax.annotate(f'{acc:.2f}', (nl, acc), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'external_validation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: external_validation.png")

    def analyze_confusable_pairs(self) -> Dict:
        """Analyze confusion between similar cancer types."""
        print("\n" + "=" * 60)
        print("Analyzing Confusable Cancer Pairs...")
        print("=" * 60)

        le = LabelEncoder()
        le.fit(self.class_names)
        y_encoded = le.fit_transform(self.y)

        # Run CV to get predictions
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_y_true = []
        all_y_pred = []

        for train_idx, val_idx in skf.split(self.X, y_encoded):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)

        y_true_labels = le.inverse_transform(all_y_true)
        y_pred_labels = le.inverse_transform(all_y_pred)

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.class_names)

        # Analyze pairs
        pair_analysis = []
        print("\n[Confusable Pair Analysis]")

        for (c1, c2), desc in self.CONFUSABLE_PAIRS.items():
            if c1 not in self.class_names or c2 not in self.class_names:
                continue

            idx1 = self.class_names.index(c1)
            idx2 = self.class_names.index(c2)

            c1_to_c2 = cm[idx1, idx2] / cm[idx1, :].sum() if cm[idx1, :].sum() > 0 else 0
            c2_to_c1 = cm[idx2, idx1] / cm[idx2, :].sum() if cm[idx2, :].sum() > 0 else 0

            pair_analysis.append({
                'pair': f"{c1}-{c2}",
                'description': desc,
                f'{c1}_to_{c2}': float(c1_to_c2),
                f'{c2}_to_{c1}': float(c2_to_c1),
                'total_confusion': float((c1_to_c2 + c2_to_c1) / 2)
            })

            print(f"\n  {c1} ↔ {c2} ({desc})")
            print(f"    {c1} → {c2}: {c1_to_c2:.4f}")
            print(f"    {c2} → {c1}: {c2_to_c1:.4f}")

        pair_analysis.sort(key=lambda x: x['total_confusion'], reverse=True)

        self._plot_confusable_pairs(pair_analysis, cm)

        self.results['confusable_pairs'] = {
            'analysis': pair_analysis,
            'confusion_matrix': cm.tolist()
        }
        return self.results['confusable_pairs']

    def _plot_confusable_pairs(self, pair_analysis: List[Dict], cm: np.ndarray):
        """Plot confusable pairs."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        pairs = [p['pair'] for p in pair_analysis]
        rates = [p['total_confusion'] for p in pair_analysis]
        colors = ['red' if r > 0.05 else 'orange' if r > 0.02 else 'green' for r in rates]

        axes[0].barh(range(len(pairs)), rates, color=colors)
        axes[0].set_yticks(range(len(pairs)))
        axes[0].set_yticklabels(pairs)
        axes[0].set_xlabel('Average Confusion Rate')
        axes[0].set_title('Confusable Cancer Pairs')
        axes[0].axvline(x=0.05, color='red', linestyle='--', alpha=0.5)

        # Confusion matrix heatmap (normalized)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=False, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1], cbar_kws={'label': 'Proportion'})
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusable_pairs.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: confusable_pairs.png")

    def generate_report(self):
        """Generate summary report."""
        print("\n" + "=" * 60)
        print("Generating Report...")
        print("=" * 60)

        report = {
            'date': datetime.now().isoformat(),
            'model': 'Pan-Cancer Classifier',
            'note': 'Quick validation with synthetic data for demonstration',
            'results': self.results
        }

        with open(self.output_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        # Text summary
        with open(self.output_dir / 'validation_summary.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Pan-Cancer Classifier - Robust Validation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {report['date']}\n")
            f.write("Note: Quick validation with synthetic data\n\n")

            if 'repeated_cv' in self.results:
                cv = self.results['repeated_cv']['summary']
                f.write("[1. Repeated Cross-Validation]\n")
                f.write(f"  Accuracy: {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}\n")
                f.write(f"  F1 Macro: {cv['f1_macro']['mean']:.4f} ± {cv['f1_macro']['std']:.4f}\n")
                f.write(f"  MCC:      {cv['mcc']['mean']:.4f} ± {cv['mcc']['std']:.4f}\n")
                f.write(f"  95% CI:   [{cv['accuracy']['ci_lower']:.4f}, {cv['accuracy']['ci_upper']:.4f}]\n\n")

            if 'loio' in self.results:
                loio = self.results['loio']
                f.write("[2. Leave-One-Institution-Out]\n")
                f.write(f"  Mean Accuracy: {loio['mean_accuracy']:.4f} ± {loio['std_accuracy']:.4f}\n")
                f.write(f"  Performance Drop: {loio['performance_drop']:.4f}\n\n")

            if 'external' in self.results:
                ext = self.results['external']
                f.write("[3. External Validation (Noise Robustness)]\n")
                f.write(f"  Internal Accuracy: {ext['internal_accuracy']:.4f}\n")
                for nr in ext['noise_results']:
                    f.write(f"  Noise {nr['noise_level']:.1f}: {nr['accuracy']:.4f}\n")
                f.write("\n")

            if 'confusable_pairs' in self.results:
                f.write("[4. Confusable Cancer Pairs]\n")
                for pair in self.results['confusable_pairs']['analysis'][:5]:
                    f.write(f"  {pair['pair']}: {pair['total_confusion']:.4f}\n")

        print(f"  Saved: validation_report.json")
        print(f"  Saved: validation_summary.txt")

    def run_all(self):
        """Run all validations."""
        print("\n" + "#" * 60)
        print("#  Pan-Cancer Classifier - Quick Robust Validation  #")
        print("#" * 60)

        self.run_repeated_cv(n_splits=5, n_repeats=5)
        self.run_loio_validation()
        self.simulate_external_validation()
        self.analyze_confusable_pairs()
        self.generate_report()

        print("\n" + "=" * 60)
        print("Validation Complete!")
        print(f"Results: {self.output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    validator = QuickRobustValidator()
    validator.run_all()
