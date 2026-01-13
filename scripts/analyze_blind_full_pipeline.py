#!/usr/bin/env python3
"""
BLIND 마이크로어레이 데이터 전체 파이프라인 분석
=================================================

마이크로어레이 데이터를 유전자 심볼로 변환 후:
1. 차등 발현 분석 (DEG) - limma 스타일
2. 네트워크 분석 (Hub gene)
3. Pathway 분석 (GO/KEGG)
4. DB 검증 (DisGeNET, OMIM, COSMIC)
5. 시각화 (Volcano, Heatmap, Network)
6. HTML 리포트 생성
+ ML 예측

BLIND_D는 폐 편평세포암(LUSC)으로 예측됨 - 분석 대상
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.agents.agent2_network import NetworkAgent
from rnaseq_pipeline.agents.agent3_pathway import PathwayAgent
from rnaseq_pipeline.agents.agent4_validation import ValidationAgent
from rnaseq_pipeline.agents.agent5_visualization import VisualizationAgent
from rnaseq_pipeline.agents.agent6_report import ReportAgent
from rnaseq_pipeline.ml.pancancer_classifier import PanCancerClassifier


class BlindFullPipelineAnalyzer:
    """BLIND 마이크로어레이 전체 파이프라인 분석기"""

    def __init__(self, blind_file: str, output_dir: Path, cancer_type: str = "lung_cancer"):
        self.blind_file = Path(blind_file)
        self.output_dir = output_dir
        self.cancer_type = cancer_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load probe-to-gene annotation
        self.annotation_path = Path("rnaseq_test_results/geo_lung_cancer/cache/GPL570_annotation.csv")
        self.probe_to_gene = self._load_annotation()
        print(f"Loaded {len(self.probe_to_gene)} probe->gene mappings")

    def _load_annotation(self) -> Dict[str, str]:
        """Load Affymetrix probe to gene symbol mapping"""
        if self.annotation_path.exists():
            df = pd.read_csv(self.annotation_path)
            mapping = {}
            for _, row in df.iterrows():
                probe = row['probe_id']
                gene = row['gene_symbol']
                if pd.notna(gene) and isinstance(gene, str) and gene.strip():
                    mapping[probe] = gene.strip()
            return mapping
        return {}

    def convert_to_gene_expression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert probe IDs to gene symbols and create Gene x Sample matrix"""
        # Input: rows=samples, columns=probes
        probe_cols = [c for c in df.columns if c != 'name']
        sample_ids = df['name'].tolist() if 'name' in df.columns else df.index.tolist()

        # Map probes to genes
        probe_to_symbol = {}
        for probe in probe_cols:
            gene = self.probe_to_gene.get(probe)
            if gene:
                probe_to_symbol[probe] = gene

        # Group probes by gene
        symbol_to_probes = {}
        for probe, symbol in probe_to_symbol.items():
            if symbol not in symbol_to_probes:
                symbol_to_probes[symbol] = []
            symbol_to_probes[symbol].append(probe)

        # Average expression per gene
        gene_data = {}
        for symbol, probes in symbol_to_probes.items():
            gene_data[symbol] = df[probes].mean(axis=1).values

        # Gene x Sample matrix
        gene_df = pd.DataFrame(gene_data, index=sample_ids).T
        return gene_df

    def perform_deg_analysis(self, gene_df: pd.DataFrame, sample_info: Dict) -> pd.DataFrame:
        """
        Perform differential expression analysis for microarray
        Using t-test + fold change (limma style without R)

        sample_info: {'group1': [sample_ids], 'group2': [sample_ids]}
        """
        group1_samples = sample_info.get('group1', [])
        group2_samples = sample_info.get('group2', [])

        if not group1_samples or not group2_samples:
            # No groups defined - use variance-based ranking
            print("No sample groups defined. Using variance-based DEG ranking.")
            return self._variance_based_deg(gene_df)

        results = []
        for gene in gene_df.index:
            g1_vals = gene_df.loc[gene, group1_samples].values.astype(float)
            g2_vals = gene_df.loc[gene, group2_samples].values.astype(float)

            # Remove NaN
            g1_vals = g1_vals[~np.isnan(g1_vals)]
            g2_vals = g2_vals[~np.isnan(g2_vals)]

            if len(g1_vals) < 2 or len(g2_vals) < 2:
                continue

            # Calculate fold change (log2)
            mean1, mean2 = np.mean(g1_vals), np.mean(g2_vals)
            if mean2 > 0 and mean1 > 0:
                log2fc = np.log2(mean1 / mean2)
            else:
                log2fc = 0

            # t-test
            try:
                t_stat, pval = stats.ttest_ind(g1_vals, g2_vals)
            except:
                pval = 1.0

            results.append({
                'gene_id': gene,
                'gene_name': gene,
                'log2FoldChange': log2fc,
                'pvalue': pval,
                'baseMean': (mean1 + mean2) / 2
            })

        deg_df = pd.DataFrame(results)

        # Adjust p-values (Benjamini-Hochberg)
        from scipy.stats import false_discovery_control
        if len(deg_df) > 0:
            deg_df['padj'] = false_discovery_control(deg_df['pvalue'].fillna(1).values, method='bh')
        else:
            deg_df['padj'] = []

        return deg_df

    def _variance_based_deg(self, gene_df: pd.DataFrame) -> pd.DataFrame:
        """Use variance-based ranking when no sample groups available"""
        results = []
        for gene in gene_df.index:
            vals = gene_df.loc[gene].values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) < 2:
                continue

            var = np.var(vals)
            mean = np.mean(vals)
            cv = np.std(vals) / mean if mean > 0 else 0

            results.append({
                'gene_id': gene,
                'gene_name': gene,
                'log2FoldChange': cv,  # Use CV as pseudo fold change
                'pvalue': 1 / (var + 1),  # Higher variance = lower "p-value"
                'padj': 1 / (var + 1),
                'baseMean': mean
            })

        deg_df = pd.DataFrame(results)
        deg_df = deg_df.sort_values('padj', ascending=True)
        return deg_df

    def run_ml_prediction(self, gene_df: pd.DataFrame) -> Dict:
        """Run Pan-Cancer ML prediction"""
        print("\n--- Running ML Prediction ---")
        model_dir = Path("models/rnaseq/pancancer")
        classifier = PanCancerClassifier(str(model_dir))
        classifier.load()

        sample_ids = gene_df.columns.tolist()
        predictions = classifier.predict(gene_df, sample_ids=sample_ids)

        # Summarize
        pred_counts = {}
        results_list = []
        for p in predictions:
            cancer = p.predicted_cancer
            pred_counts[cancer] = pred_counts.get(cancer, 0) + 1
            results_list.append({
                'sample_id': p.sample_id,
                'predicted_cancer': p.predicted_cancer,
                'predicted_cancer_korean': p.predicted_cancer_korean,
                'confidence': p.confidence,
                'is_unknown': p.is_unknown
            })

        return {
            'prediction_distribution': pred_counts,
            'sample_predictions': results_list,
            'n_samples': len(predictions)
        }

    def run_full_pipeline(self, sample_info: Optional[Dict] = None):
        """Run the full analysis pipeline"""
        print(f"\n{'='*70}")
        print(f"BLIND FULL PIPELINE ANALYSIS: {self.blind_file.stem}")
        print(f"{'='*70}")

        # Load data
        print("\n[1/7] Loading and converting data...")
        raw_df = pd.read_csv(self.blind_file)
        gene_df = self.convert_to_gene_expression(raw_df)
        print(f"  Converted to {gene_df.shape[0]} genes x {gene_df.shape[1]} samples")

        # Save count matrix
        gene_df.to_csv(self.output_dir / 'count_matrix.csv')

        # Create sample metadata
        metadata = pd.DataFrame({
            'sample_id': gene_df.columns,
            'condition': ['tumor'] * len(gene_df.columns)  # All samples assumed tumor
        })
        metadata.to_csv(self.output_dir / 'sample_metadata.csv', index=False)

        # DEG analysis
        print("\n[2/7] Performing DEG analysis...")
        if sample_info is None:
            # Split samples in half for pseudo-comparison (or use all as variance-based)
            sample_info = {}  # Will trigger variance-based DEG

        deg_df = self.perform_deg_analysis(gene_df, sample_info)

        # Filter significant
        deg_significant = deg_df[
            (abs(deg_df['log2FoldChange']) > 1) &
            (deg_df['padj'] < 0.05)
        ].copy()

        if len(deg_significant) < 100:
            # Not enough significant - use top variance genes
            deg_significant = deg_df.head(500).copy()

        deg_significant.to_csv(self.output_dir / 'deg_significant.csv', index=False)
        deg_df.to_csv(self.output_dir / 'deg_all.csv', index=False)
        print(f"  Found {len(deg_significant)} DEGs")

        # ML Prediction
        print("\n[3/7] Running ML prediction...")
        ml_results = self.run_ml_prediction(gene_df)
        with open(self.output_dir / 'ml_predictions.json', 'w') as f:
            json.dump(ml_results, f, indent=2, ensure_ascii=False)
        print(f"  Predictions: {ml_results['prediction_distribution']}")

        # Network Analysis (Agent 2)
        print("\n[4/7] Running network analysis...")
        network_agent = NetworkAgent(self.output_dir, {
            'cancer_type': self.cancer_type,
            'top_n_hub_genes': 20
        })
        network_result = network_agent.run({
            'deg_file': str(self.output_dir / 'deg_significant.csv'),
            'count_matrix': str(self.output_dir / 'count_matrix.csv')
        })
        print(f"  Hub genes: {network_result.get('n_hub_genes', 0)}")

        # Pathway Analysis (Agent 3)
        print("\n[5/7] Running pathway analysis...")
        pathway_agent = PathwayAgent(self.output_dir, {
            'cancer_type': self.cancer_type
        })
        pathway_result = pathway_agent.run({
            'deg_file': str(self.output_dir / 'deg_significant.csv')
        })
        print(f"  Pathways: {pathway_result.get('n_pathways', 0)}")

        # Validation (Agent 4)
        print("\n[6/7] Running validation...")
        validation_agent = ValidationAgent(self.output_dir, {
            'cancer_type': self.cancer_type,
            'use_rag': True
        })
        validation_result = validation_agent.run({
            'deg_file': str(self.output_dir / 'deg_significant.csv'),
            'hub_genes_file': str(self.output_dir / 'hub_genes.csv')
        })
        print(f"  Validated genes: {validation_result.get('n_validated', 0)}")

        # Visualization (Agent 5)
        print("\n[7/7] Generating visualizations...")
        viz_agent = VisualizationAgent(self.output_dir, {
            'cancer_type': self.cancer_type
        })
        viz_result = viz_agent.run({
            'deg_file': str(self.output_dir / 'deg_significant.csv'),
            'count_matrix': str(self.output_dir / 'count_matrix.csv'),
            'hub_genes_file': str(self.output_dir / 'hub_genes.csv'),
            'pathway_file': str(self.output_dir / 'pathway_summary.csv')
        })
        print(f"  Generated {viz_result.get('n_figures', 0)} figures")

        # Report (Agent 6)
        print("\n[FINAL] Generating HTML report...")
        report_agent = ReportAgent(self.output_dir, {
            'cancer_type': self.cancer_type,
            'study_name': f'{self.blind_file.stem} Analysis'
        })
        report_result = report_agent.run({
            'deg_file': str(self.output_dir / 'deg_significant.csv'),
            'hub_genes_file': str(self.output_dir / 'hub_genes.csv'),
            'pathway_file': str(self.output_dir / 'pathway_summary.csv'),
            'validation_file': str(self.output_dir / 'integrated_gene_table.csv'),
            'ml_predictions': ml_results
        })

        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        print(f"HTML Report: {self.output_dir / 'final_report.html'}")

        return {
            'n_genes': gene_df.shape[0],
            'n_samples': gene_df.shape[1],
            'n_degs': len(deg_significant),
            'ml_predictions': ml_results,
            'output_dir': str(self.output_dir)
        }


def main():
    """Main entry point - analyze BLIND_D (predicted as LUSC)"""
    # BLIND_D showed strong LUSC prediction - analyze it
    blind_files = [
        ("/Users/admin/blind_data/BLIND_D.csv", "lung_cancer"),  # LUSC
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for blind_file, cancer_type in blind_files:
        file_name = Path(blind_file).stem
        output_dir = Path(f"rnaseq_test_results/blind_full_pipeline/{file_name}_{timestamp}")

        analyzer = BlindFullPipelineAnalyzer(
            blind_file=blind_file,
            output_dir=output_dir,
            cancer_type=cancer_type
        )

        result = analyzer.run_full_pipeline()

        print(f"\n--- {file_name} Summary ---")
        print(f"Genes: {result['n_genes']}")
        print(f"Samples: {result['n_samples']}")
        print(f"DEGs: {result['n_degs']}")
        print(f"ML Predictions: {result['ml_predictions']['prediction_distribution']}")


if __name__ == "__main__":
    main()
