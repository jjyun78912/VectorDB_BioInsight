#!/usr/bin/env python3
"""
Enhanced RNA-seq Analysis Pipeline (v2.0)

Improvements over v1:
- Quality Control module
- Publication-quality visualizations
- Gene Cards with disease associations
- HTML interactive reports
- Multi-format output (TXT, JSON, HTML)

Author: BioInsight AI
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import json

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhanced modules
from visualizations import RNAseqVisualizer, create_all_visualizations
from disease_database import DiseaseDatabase, GeneCard, get_gene_cards_summary
from qc_module import RNAseqQC, run_qc_pipeline
from html_report import HTMLReportGenerator, generate_html_report


class EnhancedRNAseqPipeline:
    """
    Enhanced RNA-seq analysis pipeline with:
    - QC module
    - Visualizations
    - Disease associations
    - HTML reports
    """

    def __init__(self, output_dir: str = "rnaseq_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize modules
        self.qc = RNAseqQC(output_dir=str(self.output_dir / "qc"))
        self.visualizer = RNAseqVisualizer(output_dir=str(self.output_dir / "figures"))
        self.disease_db = DiseaseDatabase()
        self.report_generator = HTMLReportGenerator(output_dir=str(self.output_dir / "reports"))

        # Data storage
        self.raw_counts = None
        self.filtered_counts = None
        self.normalized_counts = None
        self.metadata = None
        self.deg_results = None
        self.hub_genes = None
        self.gene_cards = None
        self.qc_report = None
        self.pathway_results = None

        logger.info(f"Enhanced Pipeline initialized. Output: {self.output_dir}")

    # =========================================================================
    # STEP 1: Data Loading
    # =========================================================================

    def generate_synthetic_counts(
        self,
        n_genes: int = 5000,
        n_tumor: int = 20,
        n_normal: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic RNA-seq count data for testing."""
        logger.info("=" * 70)
        logger.info("STEP 1: Generating Synthetic Count Data")
        logger.info("=" * 70)

        np.random.seed(42)
        n_samples = n_tumor + n_normal

        # Known cancer genes to embed
        cancer_genes = [
            'EGFR', 'KRAS', 'TP53', 'ALK', 'MET', 'BRAF', 'PIK3CA', 'PTEN',
            'RB1', 'CDKN2A', 'MYC', 'CCND1', 'CDK4', 'ERBB2', 'VEGFA',
            'TOP2A', 'BIRC5', 'AURKA', 'CDC20', 'UBE2C', 'CCNB1', 'BUB1',
            'FOXM1', 'E2F1', 'MCM2', 'TYMS', 'SPP1', 'MMP1', 'MMP9', 'COL1A1',
            'BRCA1', 'BRCA2', 'APC', 'VHL', 'NF1', 'STK11', 'BCL2', 'BAX',
            'AKT1', 'MTOR', 'HIF1A', 'RET', 'ROS1'
        ]

        gene_names = cancer_genes.copy()
        for i in range(n_genes - len(cancer_genes)):
            gene_names.append(f"GENE_{i:04d}")

        # Generate negative binomial counts
        counts = np.random.negative_binomial(n=5, p=0.3, size=(n_genes, n_samples))

        # Add differential expression for cancer genes
        for i, gene in enumerate(cancer_genes):
            if i % 2 == 0:  # Up in tumor
                counts[i, :n_tumor] = counts[i, :n_tumor] * np.random.randint(3, 6)
            else:  # Down in tumor
                counts[i, n_tumor:] = counts[i, n_tumor:] * np.random.randint(3, 6)

        # Sample names
        tumor_samples = [f"TUMOR_{i:02d}" for i in range(n_tumor)]
        normal_samples = [f"NORMAL_{i:02d}" for i in range(n_normal)]
        sample_names = tumor_samples + normal_samples

        counts_df = pd.DataFrame(counts, index=gene_names, columns=sample_names)
        metadata_df = pd.DataFrame({
            'sample_id': sample_names,
            'condition': ['tumor'] * n_tumor + ['normal'] * n_normal,
            'batch': ['batch1'] * (n_samples // 2) + ['batch2'] * (n_samples - n_samples // 2)
        })

        logger.info(f"Generated count matrix: {counts_df.shape}")
        logger.info(f"Tumor samples: {n_tumor}, Normal samples: {n_normal}")
        logger.info(f"Cancer genes embedded: {len(cancer_genes)}")

        counts_df.to_csv(self.output_dir / "raw_counts.csv")
        metadata_df.to_csv(self.output_dir / "metadata.csv", index=False)

        self.raw_counts = counts_df
        self.metadata = metadata_df

        return counts_df, metadata_df

    def load_count_matrix(
        self,
        count_file: str,
        metadata_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-computed count matrix and metadata."""
        logger.info("=" * 70)
        logger.info("STEP 1: Loading Count Matrix")
        logger.info("=" * 70)

        counts = pd.read_csv(count_file, index_col=0)
        metadata = pd.read_csv(metadata_file)

        logger.info(f"Count matrix: {counts.shape} (genes x samples)")
        logger.info(f"Metadata: {metadata.shape}")

        self.raw_counts = counts
        self.metadata = metadata

        return counts, metadata

    # =========================================================================
    # STEP 2: Quality Control
    # =========================================================================

    def run_quality_control(
        self,
        counts: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """Run QC analysis and filter data."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Quality Control")
        logger.info("=" * 70)

        qc_report, filtered_counts = self.qc.run_full_qc(counts, metadata)

        self.qc_report = qc_report.to_dict()
        self.filtered_counts = filtered_counts

        logger.info(f"\nQC Summary:")
        logger.info(f"  Samples passed: {qc_report.samples_passed}/{qc_report.total_samples}")
        logger.info(f"  Genes retained: {qc_report.genes_after_filter}/{qc_report.genes_before_filter}")

        return self.qc_report, filtered_counts

    # =========================================================================
    # STEP 3: DESeq2 Analysis
    # =========================================================================

    def run_deseq2(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        condition_col: str = 'condition',
        reference_level: str = 'normal'
    ) -> pd.DataFrame:
        """Run DESeq2 differential expression analysis via rpy2."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: DESeq2 Differential Expression Analysis")
        logger.info("=" * 70)

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr
            from rpy2.robjects.conversion import localconverter

            converter = ro.default_converter + pandas2ri.converter

            logger.info("Loading R packages...")
            base = importr('base')
            deseq2 = importr('DESeq2')
            logger.info("  ✓ DESeq2 loaded")

            # Align samples
            common_samples = list(set(counts.columns) & set(metadata['sample_id']))
            counts_aligned = counts[common_samples]
            meta_aligned = metadata[metadata['sample_id'].isin(common_samples)].copy()
            meta_aligned = meta_aligned.set_index('sample_id').loc[common_samples].reset_index()

            logger.info(f"\nSamples: {len(common_samples)}")
            logger.info(f"Genes: {len(counts_aligned)}")

            with localconverter(converter):
                counts_r = ro.conversion.py2rpy(counts_aligned)
                meta_r = ro.conversion.py2rpy(meta_aligned.set_index('sample_id'))

            # Run DESeq2 in R
            ro.r('''
            run_deseq2_analysis <- function(counts, coldata, condition_col, ref_level) {
                library(DESeq2)
                count_matrix <- as.matrix(counts)
                mode(count_matrix) <- "integer"
                common_samples <- intersect(colnames(count_matrix), rownames(coldata))
                count_matrix <- count_matrix[, common_samples, drop=FALSE]
                coldata <- coldata[common_samples, , drop=FALSE]
                coldata[[condition_col]] <- factor(coldata[[condition_col]])
                coldata[[condition_col]] <- relevel(coldata[[condition_col]], ref = ref_level)
                design_formula <- as.formula(paste("~", condition_col))
                dds <- DESeqDataSetFromMatrix(
                    countData = count_matrix,
                    colData = coldata,
                    design = design_formula
                )
                keep <- rowSums(counts(dds)) >= 10
                dds <- dds[keep,]
                dds <- DESeq(dds)
                res <- results(dds, alpha = 0.05)
                res_df <- as.data.frame(res)
                res_df$gene <- rownames(res_df)
                norm_counts <- as.data.frame(counts(dds, normalized=TRUE))
                return(list(results=res_df, normalized_counts=norm_counts))
            }
            ''')

            run_deseq2_r = ro.globalenv['run_deseq2_analysis']

            logger.info("\nRunning DESeq2...")
            results = run_deseq2_r(counts_r, meta_r, condition_col, reference_level)

            with localconverter(converter):
                results_df = ro.conversion.rpy2py(results.rx2('results'))
                norm_counts = ro.conversion.rpy2py(results.rx2('normalized_counts'))

            results_df = results_df.dropna(subset=['padj'])
            results_df = results_df.sort_values('padj')

            # Save normalized counts
            self.normalized_counts = pd.DataFrame(
                norm_counts,
                index=counts_aligned.index[:len(norm_counts)],
                columns=common_samples
            )
            self.normalized_counts.to_csv(self.output_dir / "normalized_counts.csv")

            # Filter significant DEGs
            sig_results = results_df[
                (results_df['padj'] < 0.05) &
                (abs(results_df['log2FoldChange']) > 1.0)
            ]

            logger.info(f"\n=== DESeq2 Results ===")
            logger.info(f"Total genes tested: {len(results_df)}")
            logger.info(f"Significant DEGs (padj<0.05, |log2FC|>1): {len(sig_results)}")
            logger.info(f"Up-regulated: {len(sig_results[sig_results['log2FoldChange'] > 0])}")
            logger.info(f"Down-regulated: {len(sig_results[sig_results['log2FoldChange'] < 0])}")

            results_df.to_csv(self.output_dir / "deseq2_all_results.csv", index=False)
            sig_results.to_csv(self.output_dir / "deseq2_significant.csv", index=False)

            self.deg_results = results_df

            return results_df

        except Exception as e:
            logger.error(f"DESeq2 failed: {e}")
            logger.info("Falling back to scipy-based analysis...")
            return self._run_scipy_deg(counts, metadata, condition_col)

    def _run_scipy_deg(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        condition_col: str = 'condition'
    ) -> pd.DataFrame:
        """Fallback DEG analysis using scipy."""
        from scipy import stats
        from scipy.stats import rankdata

        common_samples = list(set(counts.columns) & set(metadata['sample_id']))
        counts_subset = counts[common_samples]
        meta_subset = metadata[metadata['sample_id'].isin(common_samples)].set_index('sample_id')

        counts_log = np.log2(counts_subset + 1)

        conditions = meta_subset[condition_col].unique()
        if 'tumor' in conditions:
            group1 = meta_subset[meta_subset[condition_col] == 'tumor'].index.tolist()
            group2 = meta_subset[meta_subset[condition_col] == 'normal'].index.tolist()
        else:
            group1 = meta_subset[meta_subset[condition_col] == conditions[0]].index.tolist()
            group2 = meta_subset[meta_subset[condition_col] == conditions[1]].index.tolist()

        results = []
        for gene in counts_log.index:
            expr1 = counts_log.loc[gene, group1].values
            expr2 = counts_log.loc[gene, group2].values
            log2fc = np.mean(expr1) - np.mean(expr2)

            try:
                stat, pvalue = stats.mannwhitneyu(expr1, expr2, alternative='two-sided')
            except:
                pvalue = 1.0

            results.append({
                'gene': gene,
                'log2FoldChange': log2fc,
                'pvalue': pvalue,
                'baseMean': np.mean(np.concatenate([expr1, expr2]))
            })

        results_df = pd.DataFrame(results)

        n = len(results_df)
        ranked = rankdata(results_df['pvalue'])
        results_df['padj'] = np.minimum(1, results_df['pvalue'] * n / ranked)
        results_df = results_df.sort_values('padj')

        results_df.to_csv(self.output_dir / "deseq2_all_results.csv", index=False)

        self.deg_results = results_df
        self.normalized_counts = counts_log

        return results_df

    # =========================================================================
    # STEP 4: Network Analysis
    # =========================================================================

    def build_coexpression_network(
        self,
        deg_results: pd.DataFrame,
        counts: pd.DataFrame,
        top_n: int = 300,
        corr_threshold: float = 0.6
    ) -> Tuple:
        """Build co-expression network from DEGs."""
        import networkx as nx

        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Co-expression Network Analysis")
        logger.info("=" * 70)

        sig_degs = deg_results[
            (deg_results['padj'] < 0.05) &
            (abs(deg_results['log2FoldChange']) > 1.0)
        ].head(top_n)

        genes = sig_degs['gene'].tolist()
        genes_in_counts = [g for g in genes if g in counts.index]

        if len(genes_in_counts) < 50:
            extra = [g for g in counts.index if g not in genes_in_counts][:200]
            genes_in_counts.extend(extra)

        logger.info(f"Genes for network: {len(genes_in_counts)}")

        if self.normalized_counts is not None:
            expr_data = self.normalized_counts.loc[[g for g in genes_in_counts if g in self.normalized_counts.index]]
        else:
            expr_data = np.log2(counts.loc[genes_in_counts] + 1)

        logger.info("Computing Spearman correlations...")
        corr_matrix = expr_data.T.corr(method='spearman')

        G = nx.Graph()
        G.add_nodes_from(genes_in_counts)

        edge_count = 0
        for i, g1 in enumerate(genes_in_counts):
            for j, g2 in enumerate(genes_in_counts):
                if i < j and g1 in corr_matrix.index and g2 in corr_matrix.columns:
                    corr = corr_matrix.loc[g1, g2]
                    if abs(corr) > corr_threshold:
                        G.add_edge(g1, g2, weight=abs(corr))
                        edge_count += 1

        logger.info(f"Network: {G.number_of_nodes()} nodes, {edge_count} edges")

        logger.info("\nCalculating centrality measures...")

        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        try:
            pagerank = nx.pagerank(G, weight='weight')
        except:
            pagerank = {n: 1/len(G) for n in G.nodes()}

        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector = {n: 0 for n in G.nodes()}

        hub_results = []
        for gene in G.nodes():
            deg_info = deg_results[deg_results['gene'] == gene]
            log2fc = deg_info['log2FoldChange'].values[0] if len(deg_info) > 0 else 0
            padj = deg_info['padj'].values[0] if len(deg_info) > 0 else 1

            hub_results.append({
                'gene': gene,
                'degree': degree.get(gene, 0),
                'betweenness': betweenness.get(gene, 0),
                'pagerank': pagerank.get(gene, 0),
                'eigenvector': eigenvector.get(gene, 0),
                'log2FoldChange': log2fc,
                'padj': padj
            })

        hub_df = pd.DataFrame(hub_results)

        for col in ['degree', 'betweenness', 'pagerank', 'eigenvector']:
            max_val = hub_df[col].max()
            hub_df[f'{col}_norm'] = hub_df[col] / max_val if max_val > 0 else 0

        hub_df['network_score'] = (
            hub_df['degree_norm'] * 0.3 +
            hub_df['betweenness_norm'] * 0.25 +
            hub_df['pagerank_norm'] * 0.25 +
            hub_df['eigenvector_norm'] * 0.2
        )

        hub_df['expr_score'] = (1 - hub_df['padj'].clip(upper=1)) * np.abs(hub_df['log2FoldChange']) / 5
        hub_df['composite_score'] = hub_df['network_score'] * 0.6 + hub_df['expr_score'] * 0.4
        hub_df = hub_df.sort_values('composite_score', ascending=False)

        hub_df.to_csv(self.output_dir / "hub_genes.csv", index=False)

        self.hub_genes = hub_df
        self.network = G

        return G, hub_df

    # =========================================================================
    # STEP 5: Pathway Enrichment
    # =========================================================================

    def run_enrichment(
        self,
        deg_results: pd.DataFrame,
        top_n: int = 200
    ) -> pd.DataFrame:
        """Run pathway enrichment analysis using gseapy."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Pathway Enrichment Analysis")
        logger.info("=" * 70)

        try:
            import gseapy as gp

            sig_genes = deg_results[
                (deg_results['padj'] < 0.05) &
                (abs(deg_results['log2FoldChange']) > 1.0)
            ]['gene'].tolist()[:top_n]

            logger.info(f"Running enrichment for {len(sig_genes)} genes...")

            enr = gp.enrichr(
                gene_list=sig_genes,
                gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021'],
                organism='human',
                outdir=None,
                no_plot=True,
                verbose=False
            )

            results = enr.results
            sig_pathways = results[results['Adjusted P-value'] < 0.05]

            logger.info(f"\nSignificant pathways: {len(sig_pathways)}")

            if len(sig_pathways) > 0:
                sig_pathways.to_csv(self.output_dir / "pathway_enrichment.csv", index=False)

            self.pathway_results = sig_pathways

            return sig_pathways

        except Exception as e:
            logger.warning(f"Enrichment failed: {e}")
            self.pathway_results = pd.DataFrame()
            return pd.DataFrame()

    # =========================================================================
    # STEP 6: Gene Cards & Disease Associations
    # =========================================================================

    def create_gene_cards(
        self,
        deg_results: pd.DataFrame,
        top_n: int = 50
    ) -> List[GeneCard]:
        """Create gene cards with disease associations."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Gene Status Cards & Disease Associations")
        logger.info("=" * 70)

        gene_cards = self.disease_db.create_gene_cards_from_deg(
            deg_results,
            top_n=top_n
        )

        # Get disease enrichment
        sig_genes = deg_results[
            (deg_results['padj'] < 0.05) &
            (abs(deg_results['log2FoldChange']) > 1.0)
        ]['gene'].tolist()

        disease_enrichment = self.disease_db.get_disease_enrichment(sig_genes)

        if len(disease_enrichment) > 0:
            logger.info(f"\nDisease Enrichment:")
            logger.info(disease_enrichment.head(10).to_string())
            disease_enrichment.to_csv(self.output_dir / "disease_enrichment.csv", index=False)

        # Validate hub genes
        if self.hub_genes is not None:
            validation = self.disease_db.validate_hub_genes(self.hub_genes)
            logger.info(f"\nHub Gene Validation:")
            logger.info(f"  Validated: {validation['validated_count']}/{validation['total_analyzed']} ({validation['validation_rate']:.1%})")
            logger.info(f"  Oncogenes: {len(validation['oncogenes'])}")
            logger.info(f"  Tumor suppressors: {len(validation['tumor_suppressors'])}")
            logger.info(f"  With therapeutics: {len(validation['with_therapeutics'])}")

        self.gene_cards = gene_cards

        # Save summary
        summary = get_gene_cards_summary(gene_cards)
        with open(self.output_dir / "gene_cards_summary.txt", 'w') as f:
            f.write(summary)

        return gene_cards

    # =========================================================================
    # STEP 7: Visualizations
    # =========================================================================

    def create_visualizations(
        self,
        deg_results: pd.DataFrame,
        hub_genes: pd.DataFrame,
        expression_matrix: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Create all visualizations."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: Creating Visualizations")
        logger.info("=" * 70)

        figures = create_all_visualizations(
            deg_results=deg_results,
            hub_genes=hub_genes,
            expression_matrix=expression_matrix,
            metadata=metadata,
            output_dir=str(self.output_dir / "figures")
        )

        return figures

    # =========================================================================
    # STEP 8: Report Generation
    # =========================================================================

    def generate_reports(self) -> Dict[str, str]:
        """Generate all report formats."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 8: Generating Reports")
        logger.info("=" * 70)

        reports = {}

        # 1. Text report
        text_report = self._generate_text_report()
        text_path = self.output_dir / "analysis_report.txt"
        with open(text_path, 'w') as f:
            f.write(text_report)
        reports['text'] = str(text_path)
        logger.info(f"  ✓ Text report: {text_path}")

        # 2. JSON report
        json_report = self._generate_json_report()
        json_path = self.output_dir / "analysis_report.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        reports['json'] = str(json_path)
        logger.info(f"  ✓ JSON report: {json_path}")

        # 3. HTML report
        html_path = generate_html_report(
            deg_results=self.deg_results,
            hub_genes=self.hub_genes,
            gene_cards=self.gene_cards,
            pathway_results=self.pathway_results,
            qc_report=self.qc_report,
            figures_dir=str(self.output_dir / "figures"),
            output_dir=str(self.output_dir / "reports"),
            title="RNA-seq Analysis Report"
        )
        reports['html'] = html_path
        logger.info(f"  ✓ HTML report: {html_path}")

        return reports

    def _generate_text_report(self) -> str:
        """Generate text report."""
        sig_degs = self.deg_results[
            (self.deg_results['padj'] < 0.05) &
            (abs(self.deg_results['log2FoldChange']) > 1.0)
        ]

        report = f"""
================================================================================
          ENHANCED RNA-seq ANALYSIS PIPELINE REPORT
          Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. DATA SUMMARY
---------------
Total genes analyzed: {len(self.deg_results)}
Samples: {len(self.raw_counts.columns) if self.raw_counts is not None else 'N/A'}

2. QUALITY CONTROL
------------------
Samples passed: {self.qc_report.get('samples_passed', 'N/A') if self.qc_report else 'N/A'}
Genes before filter: {self.qc_report.get('genes_before_filter', 'N/A') if self.qc_report else 'N/A'}
Genes after filter: {self.qc_report.get('genes_after_filter', 'N/A') if self.qc_report else 'N/A'}

3. DIFFERENTIAL EXPRESSION (DESeq2)
-----------------------------------
Significant DEGs (padj<0.05, |log2FC|>1): {len(sig_degs)}
Up-regulated: {len(sig_degs[sig_degs['log2FoldChange'] > 0])}
Down-regulated: {len(sig_degs[sig_degs['log2FoldChange'] < 0])}

Top 10 DEGs:
{sig_degs[['gene', 'log2FoldChange', 'padj']].head(10).to_string()}

4. NETWORK ANALYSIS
-------------------
Genes in network: {len(self.hub_genes)}

Top 10 Hub Genes:
{self.hub_genes[['gene', 'degree', 'composite_score', 'log2FoldChange']].head(10).to_string()}

5. GENE STATUS CARDS
--------------------
Total cards created: {len(self.gene_cards) if self.gene_cards else 0}
With disease associations: {len([c for c in self.gene_cards if c.diseases]) if self.gene_cards else 0}
With therapeutics: {len([c for c in self.gene_cards if c.therapeutics]) if self.gene_cards else 0}

6. PATHWAY ENRICHMENT
---------------------
Significant pathways: {len(self.pathway_results) if self.pathway_results is not None else 0}

7. OUTPUT FILES
---------------
{self.output_dir}/
├── raw_counts.csv
├── normalized_counts.csv
├── metadata.csv
├── deseq2_all_results.csv
├── deseq2_significant.csv
├── hub_genes.csv
├── pathway_enrichment.csv
├── disease_enrichment.csv
├── gene_cards_summary.txt
├── qc/
│   ├── qc_summary.png
│   └── qc_report.txt
├── figures/
│   ├── volcano_plot.png
│   ├── ma_plot.png
│   ├── heatmap.png
│   ├── pca_plot.png
│   ├── network_plot.png
│   └── dashboard.png
└── reports/
    └── analysis_report.html

================================================================================
                    ANALYSIS COMPLETED SUCCESSFULLY
================================================================================
"""
        return report

    def _generate_json_report(self) -> Dict:
        """Generate JSON report."""
        sig_degs = self.deg_results[
            (self.deg_results['padj'] < 0.05) &
            (abs(self.deg_results['log2FoldChange']) > 1.0)
        ]

        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'pipeline_version': '2.0',
                'output_dir': str(self.output_dir)
            },
            'summary': {
                'total_genes': len(self.deg_results),
                'significant_degs': len(sig_degs),
                'upregulated': int(len(sig_degs[sig_degs['log2FoldChange'] > 0])),
                'downregulated': int(len(sig_degs[sig_degs['log2FoldChange'] < 0])),
                'hub_genes': len(self.hub_genes),
                'gene_cards': len(self.gene_cards) if self.gene_cards else 0
            },
            'qc': self.qc_report,
            'top_degs': sig_degs.head(20).to_dict('records'),
            'top_hub_genes': self.hub_genes.head(20).to_dict('records'),
            'gene_cards': [c.to_dict() for c in self.gene_cards[:20]] if self.gene_cards else []
        }

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    def run_full_pipeline(
        self,
        use_synthetic: bool = True,
        count_file: str = None,
        metadata_file: str = None,
        skip_qc: bool = False,
        skip_viz: bool = False
    ) -> Dict:
        """
        Run the complete enhanced RNA-seq analysis pipeline.

        Args:
            use_synthetic: Use synthetic data for testing
            count_file: Path to count matrix file
            metadata_file: Path to metadata file
            skip_qc: Skip QC step
            skip_viz: Skip visualization step

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 70)
        logger.info("ENHANCED RNA-seq ANALYSIS PIPELINE v2.0")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}\n")

        try:
            # Step 1: Load or generate data
            if use_synthetic:
                counts, metadata = self.generate_synthetic_counts()
            elif count_file and metadata_file:
                counts, metadata = self.load_count_matrix(count_file, metadata_file)
            else:
                raise ValueError("Provide count_file and metadata_file, or use use_synthetic=True")

            # Step 2: QC
            if not skip_qc:
                qc_report, filtered_counts = self.run_quality_control(counts, metadata)
                counts = filtered_counts
            else:
                self.filtered_counts = counts

            # Step 3: DESeq2 analysis
            deg_results = self.run_deseq2(counts, metadata)

            # Step 4: Network analysis
            G, hub_df = self.build_coexpression_network(deg_results, counts)

            # Step 5: Pathway enrichment
            pathway_results = self.run_enrichment(deg_results)

            # Step 6: Gene cards
            gene_cards = self.create_gene_cards(deg_results)

            # Step 7: Visualizations
            if not skip_viz:
                expr_matrix = self.normalized_counts if self.normalized_counts is not None else counts
                figures = self.create_visualizations(deg_results, hub_df, expr_matrix, metadata)

            # Step 8: Reports
            reports = self.generate_reports()

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info(f"\nOpen the HTML report: {reports['html']}")

            return {
                'deg_results': deg_results,
                'hub_genes': hub_df,
                'pathway_results': pathway_results,
                'gene_cards': gene_cards,
                'qc_report': self.qc_report,
                'reports': reports
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Run the enhanced pipeline with synthetic data for testing."""

    output_dir = "/Users/admin/VectorDB_BioInsight/rnaseq_test_results/enhanced_run"

    pipeline = EnhancedRNAseqPipeline(output_dir=output_dir)
    results = pipeline.run_full_pipeline(use_synthetic=True)

    return results


if __name__ == "__main__":
    main()
