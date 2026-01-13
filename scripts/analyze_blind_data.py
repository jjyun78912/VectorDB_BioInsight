#!/usr/bin/env python3
"""
BLIND 마이크로어레이 데이터 분석 스크립트
=========================================

Affymetrix 마이크로어레이 데이터를 유전자 심볼로 변환 후
Pan-Cancer ML 모델로 암종 예측 수행
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml import UnifiedPredictor
from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


class BlindDataAnalyzer:
    """BLIND 마이크로어레이 데이터 분석기"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load probe-to-gene annotation
        self.annotation_path = Path("rnaseq_test_results/geo_lung_cancer/cache/GPL570_annotation.csv")
        self.probe_to_gene = self._load_annotation()

        # Load symbol to ENSEMBL mapping (for model input)
        self.symbol_to_ensembl = self._load_symbol_mapping()
        print(f"Loaded {len(self.symbol_to_ensembl)} symbol->ENSEMBL mappings")

        # Initialize PanCancer classifier directly
        print("Loading Pan-Cancer classifier...")
        self.model_dir = Path("models/rnaseq/pancancer")
        self.classifier = PanCancerClassifier(str(self.model_dir))
        self.classifier.load()
        print("Pan-Cancer classifier loaded!")

        # Get model's expected genes
        self.model_genes = self._get_model_genes()
        print(f"Model expects {len(self.model_genes)} genes")

    def _load_annotation(self) -> Dict[str, str]:
        """Load Affymetrix probe to gene symbol mapping"""
        if self.annotation_path.exists():
            df = pd.read_csv(self.annotation_path)
            return dict(zip(df['probe_id'], df['gene_symbol']))

        # Fallback: try to download from GEO
        print("Downloading GPL570 annotation from GEO...")
        import GEOparse
        gpl = GEOparse.get_GEO(geo='GPL570', destdir=str(self.annotation_path.parent))

        # Extract probe to gene mapping
        probe_to_gene = {}
        for probe_id, row in gpl.table.iterrows():
            gene = row.get('Gene Symbol', '')
            if gene and '///' not in gene:
                probe_to_gene[probe_id] = gene.split(' /// ')[0]

        # Save for future use
        df = pd.DataFrame(list(probe_to_gene.items()), columns=['probe_id', 'gene_symbol'])
        df.to_csv(self.annotation_path, index=False)

        return probe_to_gene

    def _load_symbol_mapping(self) -> Dict[str, str]:
        """Load gene symbol to ENSEMBL ID mapping"""
        mapping_path = Path("models/rnaseq/pancancer/symbol_to_model_ensembl.json")
        if mapping_path.exists():
            with open(mapping_path) as f:
                return json.load(f)
        return {}

    def _get_model_genes(self) -> List[str]:
        """Get genes expected by the model"""
        model_dir = Path("models/rnaseq/pancancer")

        # Try to load from preprocessor
        import joblib
        preprocessor_path = model_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            if hasattr(preprocessor, 'selected_genes'):
                return preprocessor.selected_genes

        # Fallback: use gene mapping files
        symbol_map_path = model_dir / "symbol_to_model_ensembl.json"
        if symbol_map_path.exists():
            with open(symbol_map_path) as f:
                return list(json.load(f).keys())

        return []

    def convert_probes_to_genes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert probe IDs to gene symbols (Gene x Sample format for model)

        Model expects: Gene (index) x Sample (columns) format
        Input: Sample (rows) x Probe (columns) format
        Output: Gene (rows/index) x Sample (columns) format
        """
        # Input format: rows=samples (GSM*), columns=probes

        # Get probe columns (exclude 'name' column)
        probe_cols = [c for c in df.columns if c != 'name']

        # Get sample IDs
        sample_ids = df['name'].tolist() if 'name' in df.columns else df.index.tolist()

        # Step 1: Map probes to gene symbols
        probe_to_symbol_filtered = {}
        for probe in probe_cols:
            gene = self.probe_to_gene.get(probe)
            if gene and isinstance(gene, str) and gene.strip():
                probe_to_symbol_filtered[probe] = gene

        # Step 2: Group probes by gene symbol
        symbol_to_probes = {}  # symbol -> list of probes
        for probe, symbol in probe_to_symbol_filtered.items():
            if symbol not in symbol_to_probes:
                symbol_to_probes[symbol] = []
            symbol_to_probes[symbol].append(probe)

        print(f"  Probes mapped to symbols: {len(probe_to_symbol_filtered)}")
        print(f"  Unique gene symbols: {len(symbol_to_probes)}")

        # Step 3: Average expression for probes mapping to same gene
        # Create Gene x Sample matrix
        gene_data = {}
        for symbol, probes in symbol_to_probes.items():
            # Average across probes for each sample
            gene_data[symbol] = df[probes].mean(axis=1).values

        # Create dataframe: rows=genes, columns=samples
        gene_df = pd.DataFrame(gene_data, index=sample_ids).T
        print(f"  Output shape: {gene_df.shape} (genes x samples)")

        return gene_df

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single BLIND file"""
        file_name = file_path.stem
        print(f"\n{'='*60}")
        print(f"Analyzing: {file_name}")
        print(f"{'='*60}")

        # Load data
        df = pd.read_csv(file_path)
        n_samples = len(df)
        n_probes = len(df.columns) - 1  # Exclude 'name' column
        print(f"Loaded: {n_samples} samples, {n_probes} probes")

        # Convert to genes (Gene x Sample format)
        gene_df = self.convert_probes_to_genes(df)
        n_genes = len(gene_df.index)  # genes are in index now
        sample_ids = gene_df.columns.tolist()
        print(f"Converted to: {n_genes} genes x {len(sample_ids)} samples")

        # Check overlap with model genes
        overlap_pct = 0
        if self.model_genes:
            overlap = set(gene_df.index) & set(self.model_genes)
            overlap_pct = len(overlap) / len(self.model_genes) * 100
            print(f"Gene overlap with model: {len(overlap)}/{len(self.model_genes)} ({overlap_pct:.1f}%)")

        # Run batch prediction (all samples at once)
        results = []
        predictions_summary = {}

        try:
            print(f"  Running batch prediction for {n_samples} samples...")
            # PanCancerClassifier.predict expects Gene x Sample format
            prediction_results = self.classifier.predict(gene_df, sample_ids=sample_ids)

            for result in prediction_results:
                result_dict = {
                    'sample_id': result.sample_id,
                    'predicted_cancer': result.predicted_cancer,
                    'predicted_cancer_korean': result.predicted_cancer_korean,
                    'confidence': result.confidence,
                    'confidence_level': result.confidence_level,
                    'is_unknown': result.is_unknown,
                    'top_predictions': result.top_k_predictions[:3],
                    'ensemble_agreement': result.ensemble_agreement,
                }
                results.append(result_dict)

                # Count predictions
                pred = result.predicted_cancer
                if pred not in predictions_summary:
                    predictions_summary[pred] = 0
                predictions_summary[pred] += 1

            print(f"  Batch prediction complete!")

        except Exception as e:
            print(f"  Error in batch prediction: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: mark all as error
            for sample_id in sample_ids:
                results.append({
                    'sample_id': sample_id,
                    'predicted_cancer': 'ERROR',
                    'confidence': 0,
                    'error': str(e)
                })

        # Calculate statistics
        valid_results = [r for r in results if r.get('predicted_cancer') != 'ERROR']
        confidences = [r['confidence'] for r in valid_results if 'confidence' in r]

        avg_confidence = np.mean(confidences) if confidences else 0
        unknown_count = sum(1 for r in valid_results if r.get('is_unknown', False))

        # Determine most likely cancer type
        if predictions_summary:
            # Exclude UNKNOWN for determining primary prediction
            non_unknown = {k: v for k, v in predictions_summary.items() if k != 'UNKNOWN'}
            if non_unknown:
                primary_prediction = max(non_unknown, key=non_unknown.get)
                primary_count = non_unknown[primary_prediction]
            else:
                primary_prediction = 'UNKNOWN'
                primary_count = predictions_summary.get('UNKNOWN', 0)
        else:
            primary_prediction = 'N/A'
            primary_count = 0

        summary = {
            'file_name': file_name,
            'n_samples': n_samples,
            'n_probes': n_probes,
            'n_genes': n_genes,
            'gene_overlap_pct': overlap_pct if self.model_genes else 0,
            'primary_prediction': primary_prediction,
            'primary_prediction_count': primary_count,
            'primary_prediction_pct': primary_count / n_samples * 100 if n_samples > 0 else 0,
            'prediction_distribution': predictions_summary,
            'average_confidence': avg_confidence,
            'unknown_count': unknown_count,
            'unknown_pct': unknown_count / n_samples * 100 if n_samples > 0 else 0,
            'samples': results
        }

        # Print summary
        print(f"\n--- Results for {file_name} ---")
        print(f"Primary Prediction: {primary_prediction} ({primary_count}/{n_samples}, {primary_count/n_samples*100:.1f}%)")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Unknown samples: {unknown_count} ({unknown_count/n_samples*100:.1f}%)")
        print(f"Distribution: {predictions_summary}")

        # Save individual file results
        file_output_dir = self.output_dir / file_name
        file_output_dir.mkdir(exist_ok=True)

        with open(file_output_dir / 'prediction_results.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save as CSV for easy viewing
        results_df = pd.DataFrame(results)
        results_df.to_csv(file_output_dir / 'predictions.csv', index=False)

        return summary

    def generate_summary_report(self, all_results: List[Dict]) -> None:
        """Generate overall summary report"""
        print(f"\n{'='*70}")
        print("BLIND DATA ANALYSIS - SUMMARY REPORT")
        print(f"{'='*70}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Files: {len(all_results)}")
        print()

        # Overall statistics
        total_samples = sum(r['n_samples'] for r in all_results)
        print(f"Total Samples Analyzed: {total_samples}")
        print()

        # Per-file summary table
        print("=" * 90)
        print(f"{'File':<12} {'Samples':>8} {'Primary':>10} {'Count':>6} {'%':>8} {'Avg Conf':>10} {'Unknown%':>10}")
        print("-" * 90)

        for r in all_results:
            print(f"{r['file_name']:<12} {r['n_samples']:>8} {r['primary_prediction']:>10} "
                  f"{r['primary_prediction_count']:>6} {r['primary_prediction_pct']:>7.1f}% "
                  f"{r['average_confidence']:>10.3f} {r['unknown_pct']:>9.1f}%")

        print("=" * 90)
        print()

        # Overall prediction distribution
        overall_dist = {}
        for r in all_results:
            for cancer, count in r['prediction_distribution'].items():
                if cancer not in overall_dist:
                    overall_dist[cancer] = 0
                overall_dist[cancer] += count

        print("Overall Prediction Distribution:")
        for cancer, count in sorted(overall_dist.items(), key=lambda x: -x[1]):
            pct = count / total_samples * 100
            print(f"  {cancer}: {count} ({pct:.1f}%)")

        # Save summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_files': len(all_results),
            'total_samples': total_samples,
            'overall_distribution': overall_dist,
            'per_file_results': all_results
        }

        with open(self.output_dir / 'summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {self.output_dir}")


def main():
    """Main entry point"""
    # Define input files
    blind_files = [
        Path("/Users/admin/blind_data/BLIND_A.csv"),
        Path("/Users/admin/blind_data/BLIND_B.csv"),
        Path("/Users/admin/blind_data/BLIND_C.csv"),
        Path("/Users/admin/blind_data/BLIND_D.csv"),
        Path("/Users/admin/blind_data/BLIND_E.csv"),
    ]

    # Check files exist
    for f in blind_files:
        if not f.exists():
            print(f"ERROR: File not found: {f}")
            return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"rnaseq_test_results/blind_analysis/run_{timestamp}")

    # Initialize analyzer
    analyzer = BlindDataAnalyzer(output_dir)

    # Analyze each file
    all_results = []
    for file_path in blind_files:
        result = analyzer.analyze_file(file_path)
        all_results.append(result)

    # Generate summary report
    analyzer.generate_summary_report(all_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
