#!/usr/bin/env python3
"""
RNA-seq Quality Control Module

Provides comprehensive QC metrics:
- Library size distribution
- Gene detection rate
- Sample correlation
- Outlier detection
- Low-count filtering

Author: BioInsight AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import logging

logger = logging.getLogger(__name__)


@dataclass
class QCMetrics:
    """Quality control metrics for a sample"""
    sample_id: str
    total_counts: int
    detected_genes: int
    detection_rate: float
    mean_expression: float
    median_expression: float
    zero_fraction: float
    cv: float  # Coefficient of variation
    is_outlier: bool = False
    outlier_reasons: List[str] = field(default_factory=list)


@dataclass
class QCReport:
    """Complete QC report"""
    # Dataset level
    total_samples: int
    total_genes: int

    # Sample metrics
    sample_metrics: List[QCMetrics]

    # Filtering
    samples_passed: int
    samples_failed: int
    genes_before_filter: int
    genes_after_filter: int

    # Summary stats
    median_library_size: float
    median_detection_rate: float
    outlier_samples: List[str]

    def to_dict(self) -> Dict:
        return {
            'total_samples': self.total_samples,
            'total_genes': self.total_genes,
            'samples_passed': self.samples_passed,
            'samples_failed': self.samples_failed,
            'genes_before_filter': self.genes_before_filter,
            'genes_after_filter': self.genes_after_filter,
            'median_library_size': self.median_library_size,
            'median_detection_rate': self.median_detection_rate,
            'outlier_samples': self.outlier_samples
        }


class RNAseqQC:
    """RNA-seq Quality Control"""

    def __init__(self, output_dir: str = "qc_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # QC thresholds
        self.thresholds = {
            'min_counts': 10,           # Minimum counts per gene
            'min_samples_expressed': 3, # Minimum samples where gene is expressed
            'min_library_size': 1e5,    # Minimum total counts per sample
            'min_detection_rate': 0.1,  # Minimum fraction of genes detected
            'max_zero_fraction': 0.95,  # Maximum fraction of zeros per gene
            'outlier_mad': 3.0          # MAD threshold for outlier detection
        }

    def calculate_sample_metrics(
        self,
        counts: pd.DataFrame
    ) -> List[QCMetrics]:
        """
        Calculate QC metrics for each sample

        Args:
            counts: Count matrix (genes x samples)

        Returns:
            List of QCMetrics for each sample
        """
        metrics = []

        for sample in counts.columns:
            sample_counts = counts[sample]

            total_counts = sample_counts.sum()
            detected = (sample_counts > 0).sum()
            detection_rate = detected / len(sample_counts)
            mean_expr = sample_counts[sample_counts > 0].mean()
            median_expr = sample_counts[sample_counts > 0].median()
            zero_frac = (sample_counts == 0).sum() / len(sample_counts)
            cv = sample_counts.std() / mean_expr if mean_expr > 0 else 0

            metrics.append(QCMetrics(
                sample_id=sample,
                total_counts=int(total_counts),
                detected_genes=int(detected),
                detection_rate=detection_rate,
                mean_expression=mean_expr,
                median_expression=median_expr,
                zero_fraction=zero_frac,
                cv=cv
            ))

        return metrics

    def detect_outliers(
        self,
        metrics: List[QCMetrics],
        method: str = 'mad'
    ) -> List[QCMetrics]:
        """
        Detect outlier samples using various methods

        Args:
            metrics: List of sample QC metrics
            method: 'mad' (Median Absolute Deviation) or 'iqr'

        Returns:
            Updated metrics with outlier flags
        """
        # Extract values for outlier detection
        library_sizes = np.array([m.total_counts for m in metrics])
        detection_rates = np.array([m.detection_rate for m in metrics])

        # Log transform library sizes
        log_sizes = np.log10(library_sizes + 1)

        for i, m in enumerate(metrics):
            reasons = []

            if method == 'mad':
                # MAD-based outlier detection
                median_size = np.median(log_sizes)
                mad_size = np.median(np.abs(log_sizes - median_size))

                if mad_size > 0:
                    z_size = (log_sizes[i] - median_size) / (1.4826 * mad_size)
                    if abs(z_size) > self.thresholds['outlier_mad']:
                        reasons.append(f"Library size outlier (z={z_size:.2f})")

                # Detection rate
                median_det = np.median(detection_rates)
                mad_det = np.median(np.abs(detection_rates - median_det))

                if mad_det > 0:
                    z_det = (detection_rates[i] - median_det) / (1.4826 * mad_det)
                    if z_det < -self.thresholds['outlier_mad']:
                        reasons.append(f"Low detection rate (z={z_det:.2f})")

            # Absolute thresholds
            if m.total_counts < self.thresholds['min_library_size']:
                reasons.append(f"Library size below {self.thresholds['min_library_size']:.0e}")

            if m.detection_rate < self.thresholds['min_detection_rate']:
                reasons.append(f"Detection rate below {self.thresholds['min_detection_rate']:.1%}")

            m.is_outlier = len(reasons) > 0
            m.outlier_reasons = reasons

        return metrics

    def filter_genes(
        self,
        counts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter low-expressed genes

        Args:
            counts: Count matrix (genes x samples)

        Returns:
            Filtered counts and filter statistics
        """
        n_genes_before = len(counts)

        # Filter by minimum counts
        gene_totals = counts.sum(axis=1)
        genes_min_counts = gene_totals >= self.thresholds['min_counts']

        # Filter by expression in minimum samples
        genes_expressed = (counts > 0).sum(axis=1)
        genes_min_samples = genes_expressed >= self.thresholds['min_samples_expressed']

        # Filter by maximum zero fraction
        zero_fractions = (counts == 0).sum(axis=1) / counts.shape[1]
        genes_max_zeros = zero_fractions <= self.thresholds['max_zero_fraction']

        # Combined filter
        keep_genes = genes_min_counts & genes_min_samples & genes_max_zeros

        filtered_counts = counts.loc[keep_genes]

        stats = {
            'genes_before': n_genes_before,
            'genes_after': len(filtered_counts),
            'removed_low_counts': (~genes_min_counts).sum(),
            'removed_low_samples': (~genes_min_samples & genes_min_counts).sum(),
            'removed_high_zeros': (~genes_max_zeros & genes_min_counts & genes_min_samples).sum()
        }

        logger.info(f"Gene filtering: {n_genes_before} -> {len(filtered_counts)} genes")

        return filtered_counts, stats

    def calculate_sample_correlation(
        self,
        counts: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Calculate sample-sample correlation matrix
        """
        log_counts = np.log2(counts + 1)
        return log_counts.corr(method=method)

    def run_full_qc(
        self,
        counts: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None
    ) -> Tuple[QCReport, pd.DataFrame]:
        """
        Run complete QC analysis

        Args:
            counts: Count matrix (genes x samples)
            metadata: Sample metadata

        Returns:
            QC report and filtered counts
        """
        logger.info("=" * 70)
        logger.info("RNA-seq QUALITY CONTROL")
        logger.info("=" * 70)

        # 1. Calculate sample metrics
        logger.info("\n1. Calculating sample metrics...")
        metrics = self.calculate_sample_metrics(counts)

        # 2. Detect outliers
        logger.info("2. Detecting outliers...")
        metrics = self.detect_outliers(metrics)

        outlier_samples = [m.sample_id for m in metrics if m.is_outlier]
        logger.info(f"   Found {len(outlier_samples)} outlier samples")

        # 3. Filter genes
        logger.info("3. Filtering genes...")
        filtered_counts, filter_stats = self.filter_genes(counts)

        # 4. Create report
        library_sizes = [m.total_counts for m in metrics]
        detection_rates = [m.detection_rate for m in metrics]

        report = QCReport(
            total_samples=len(counts.columns),
            total_genes=len(counts),
            sample_metrics=metrics,
            samples_passed=len([m for m in metrics if not m.is_outlier]),
            samples_failed=len(outlier_samples),
            genes_before_filter=filter_stats['genes_before'],
            genes_after_filter=filter_stats['genes_after'],
            median_library_size=np.median(library_sizes),
            median_detection_rate=np.median(detection_rates),
            outlier_samples=outlier_samples
        )

        # 5. Generate visualizations
        logger.info("4. Generating QC visualizations...")
        self.plot_qc_summary(counts, metrics, metadata)

        # 6. Save report
        self.save_qc_report(report)

        logger.info("\n✓ QC complete")

        return report, filtered_counts

    def plot_qc_summary(
        self,
        counts: pd.DataFrame,
        metrics: List[QCMetrics],
        metadata: Optional[pd.DataFrame] = None
    ):
        """Generate QC visualization plots"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Library size distribution
        ax = axes[0, 0]
        library_sizes = [m.total_counts for m in metrics]
        colors = ['red' if m.is_outlier else 'steelblue' for m in metrics]

        ax.bar(range(len(library_sizes)), library_sizes, color=colors, alpha=0.7)
        ax.axhline(self.thresholds['min_library_size'], color='red',
                   linestyle='--', label=f"Min threshold ({self.thresholds['min_library_size']:.0e})")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Total Counts')
        ax.set_title('Library Size Distribution')
        ax.legend()
        ax.set_yscale('log')

        # 2. Detection rate
        ax = axes[0, 1]
        detection_rates = [m.detection_rate for m in metrics]
        colors = ['red' if m.is_outlier else 'steelblue' for m in metrics]

        ax.bar(range(len(detection_rates)), detection_rates, color=colors, alpha=0.7)
        ax.axhline(self.thresholds['min_detection_rate'], color='red',
                   linestyle='--', label=f"Min threshold ({self.thresholds['min_detection_rate']:.0%})")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Gene Detection Rate')
        ax.legend()

        # 3. Library size vs Detection rate
        ax = axes[0, 2]
        outliers = [m.is_outlier for m in metrics]

        scatter = ax.scatter(
            library_sizes,
            detection_rates,
            c=['red' if o else 'steelblue' for o in outliers],
            alpha=0.7, s=50
        )
        ax.set_xlabel('Library Size')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Library Size vs Detection Rate')
        ax.set_xscale('log')

        # Add sample labels for outliers
        for m in metrics:
            if m.is_outlier:
                ax.annotate(m.sample_id, (m.total_counts, m.detection_rate),
                           fontsize=7, alpha=0.7)

        # 4. Count distribution per sample (boxplot)
        ax = axes[1, 0]
        log_counts = np.log2(counts + 1)

        # Sample a subset if too many samples
        if len(counts.columns) > 20:
            sample_subset = counts.columns[::len(counts.columns)//20]
        else:
            sample_subset = counts.columns

        log_counts[sample_subset].boxplot(ax=ax, rot=45)
        ax.set_ylabel('log₂(counts + 1)')
        ax.set_title('Count Distribution per Sample')

        # 5. Gene count distribution
        ax = axes[1, 1]
        gene_totals = counts.sum(axis=1)
        ax.hist(np.log10(gene_totals + 1), bins=50, alpha=0.7, color='steelblue')
        ax.axvline(np.log10(self.thresholds['min_counts'] + 1), color='red',
                   linestyle='--', label='Filter threshold')
        ax.set_xlabel('log₁₀(Total Counts + 1)')
        ax.set_ylabel('Number of Genes')
        ax.set_title('Gene Expression Distribution')
        ax.legend()

        # 6. Sample correlation heatmap
        ax = axes[1, 2]
        corr_matrix = self.calculate_sample_correlation(counts)

        # Use only subset for visualization
        if len(corr_matrix) > 30:
            idx = list(range(0, len(corr_matrix), len(corr_matrix)//30))
            corr_subset = corr_matrix.iloc[idx, idx]
        else:
            corr_subset = corr_matrix

        sns.heatmap(corr_subset, cmap='RdYlBu_r', center=0.8,
                   vmin=0.5, vmax=1.0, ax=ax,
                   xticklabels=True, yticklabels=True)
        ax.set_title('Sample Correlation')

        plt.tight_layout()

        # Save
        fig.savefig(self.output_dir / 'qc_summary.png', dpi=300, bbox_inches='tight')
        logger.info(f"   Saved: {self.output_dir / 'qc_summary.png'}")

        plt.close(fig)

    def save_qc_report(self, report: QCReport):
        """Save QC report to files"""

        # 1. Summary text report
        lines = [
            "=" * 70,
            "RNA-seq QUALITY CONTROL REPORT",
            "=" * 70,
            "",
            "DATASET SUMMARY",
            "-" * 40,
            f"Total samples: {report.total_samples}",
            f"Total genes: {report.total_genes}",
            "",
            "SAMPLE QC",
            "-" * 40,
            f"Samples passed: {report.samples_passed}",
            f"Samples failed (outliers): {report.samples_failed}",
            f"Median library size: {report.median_library_size:,.0f}",
            f"Median detection rate: {report.median_detection_rate:.2%}",
            "",
            "GENE FILTERING",
            "-" * 40,
            f"Genes before filter: {report.genes_before_filter}",
            f"Genes after filter: {report.genes_after_filter}",
            f"Genes removed: {report.genes_before_filter - report.genes_after_filter}",
            ""
        ]

        if report.outlier_samples:
            lines.extend([
                "OUTLIER SAMPLES",
                "-" * 40
            ])
            for sample_id in report.outlier_samples:
                metric = next(m for m in report.sample_metrics if m.sample_id == sample_id)
                lines.append(f"  {sample_id}:")
                for reason in metric.outlier_reasons:
                    lines.append(f"    - {reason}")
            lines.append("")

        lines.extend([
            "SAMPLE METRICS",
            "-" * 40
        ])

        # Create metrics table
        metrics_df = pd.DataFrame([
            {
                'sample_id': m.sample_id,
                'total_counts': m.total_counts,
                'detected_genes': m.detected_genes,
                'detection_rate': f"{m.detection_rate:.2%}",
                'is_outlier': m.is_outlier
            }
            for m in report.sample_metrics
        ])

        lines.append(metrics_df.to_string(index=False))

        # Save text report
        report_path = self.output_dir / 'qc_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"   Saved: {report_path}")

        # Save metrics CSV
        metrics_df.to_csv(self.output_dir / 'sample_metrics.csv', index=False)
        logger.info(f"   Saved: {self.output_dir / 'sample_metrics.csv'}")


def run_qc_pipeline(
    counts: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    output_dir: str = "qc_results"
) -> Tuple[QCReport, pd.DataFrame]:
    """
    Convenience function to run QC pipeline

    Args:
        counts: Count matrix (genes x samples)
        metadata: Optional sample metadata
        output_dir: Output directory

    Returns:
        Tuple of (QC report, filtered counts)
    """
    qc = RNAseqQC(output_dir=output_dir)
    return qc.run_full_qc(counts, metadata)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Create test data
    n_genes, n_samples = 1000, 20
    counts = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, (n_genes, n_samples)),
        index=[f"Gene_{i}" for i in range(n_genes)],
        columns=[f"Sample_{i}" for i in range(n_samples)]
    )

    # Add an outlier sample
    counts['Sample_19'] = counts['Sample_19'] * 0.01  # Low library size

    # Run QC
    report, filtered_counts = run_qc_pipeline(
        counts,
        output_dir='/Users/admin/VectorDB_BioInsight/rnaseq_test_results/test_run/qc'
    )

    print(f"\nQC Summary:")
    print(f"  Samples passed: {report.samples_passed}/{report.total_samples}")
    print(f"  Genes retained: {report.genes_after_filter}/{report.genes_before_filter}")
    print(f"  Outliers: {report.outlier_samples}")
