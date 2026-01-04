#!/usr/bin/env python3
"""
Complete RNA-seq Analysis Pipeline

Features:
1. SRA/GEO Data Download (pysradb, GEOparse)
2. Preprocessing & Normalization
3. Differential Expression with DESeq2 (via rpy2)
4. Co-expression Network Analysis
5. Hub Gene Identification
6. Pathway Enrichment (gseapy)
7. Validation & Report Generation

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

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RNAseqPipeline:
    """Complete RNA-seq analysis pipeline."""

    def __init__(self, output_dir: str = "rnaseq_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.raw_counts = None
        self.normalized_counts = None
        self.metadata = None
        self.deg_results = None
        self.hub_genes = None

        logger.info(f"Pipeline initialized. Output: {self.output_dir}")

    # =========================================================================
    # STEP 1: Data Download
    # =========================================================================

    def download_from_sra(self, srp_id: str, max_samples: int = 10) -> pd.DataFrame:
        """
        Download RNA-seq data from SRA using pysradb.

        Args:
            srp_id: SRA Project ID (e.g., SRP123456)
            max_samples: Maximum number of samples to download

        Returns:
            DataFrame with sample metadata
        """
        from pysradb.sraweb import SRAweb

        logger.info("=" * 70)
        logger.info("STEP 1: Downloading from SRA")
        logger.info("=" * 70)
        logger.info(f"Project: {srp_id}")

        sra = SRAweb()

        # Get project metadata
        logger.info("Fetching project metadata...")
        try:
            metadata = sra.sra_metadata(srp_id, detailed=True)
            logger.info(f"Found {len(metadata)} samples")

            if len(metadata) > max_samples:
                logger.info(f"Limiting to {max_samples} samples")
                metadata = metadata.head(max_samples)

            # Save metadata
            metadata.to_csv(self.output_dir / "sra_metadata.csv", index=False)
            logger.info(f"Metadata saved: {self.output_dir / 'sra_metadata.csv'}")

            # Display sample info
            display_cols = ['run_accession', 'experiment_title', 'library_strategy']
            available_cols = [c for c in display_cols if c in metadata.columns]
            logger.info(f"\nSample preview:\n{metadata[available_cols].head(10).to_string()}")

            return metadata

        except Exception as e:
            logger.error(f"SRA download failed: {e}")
            raise

    def download_fastq_files(self, run_accessions: List[str],
                             output_dir: Optional[str] = None) -> List[str]:
        """
        Download FASTQ files for given SRA run accessions.

        Note: This requires prefetch and fasterq-dump from SRA Toolkit.
        """
        import subprocess

        if output_dir is None:
            output_dir = self.output_dir / "fastq"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"\nDownloading FASTQ files to {output_dir}")
        downloaded_files = []

        for srr in run_accessions:
            logger.info(f"  Downloading {srr}...")
            try:
                # Check if SRA Toolkit is available
                result = subprocess.run(
                    ["which", "fasterq-dump"],
                    capture_output=True, text=True
                )

                if result.returncode != 0:
                    logger.warning("SRA Toolkit not found. Install with: brew install sratoolkit")
                    logger.info(f"  Manual download: https://www.ncbi.nlm.nih.gov/sra/{srr}")
                    continue

                # Download using fasterq-dump
                subprocess.run([
                    "fasterq-dump", srr,
                    "-O", str(output_dir),
                    "-p"  # progress
                ], check=True)

                downloaded_files.append(f"{output_dir}/{srr}.fastq")
                logger.info(f"  ✓ {srr} downloaded")

            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ Failed to download {srr}: {e}")

        return downloaded_files

    def load_count_matrix(self, count_file: str, metadata_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load pre-computed count matrix and metadata.

        Args:
            count_file: Path to count matrix (genes x samples)
            metadata_file: Path to sample metadata

        Returns:
            Tuple of (counts_df, metadata_df)
        """
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

    def generate_synthetic_counts(self, n_genes: int = 5000,
                                   n_tumor: int = 20,
                                   n_normal: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic RNA-seq count data for testing.
        """
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
            'FOXM1', 'E2F1', 'MCM2', 'TYMS', 'SPP1', 'MMP1', 'MMP9', 'COL1A1'
        ]

        # Gene names
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

        # Create DataFrames
        counts_df = pd.DataFrame(counts, index=gene_names, columns=sample_names)
        metadata_df = pd.DataFrame({
            'sample_id': sample_names,
            'condition': ['tumor'] * n_tumor + ['normal'] * n_normal,
            'batch': ['batch1'] * (n_samples // 2) + ['batch2'] * (n_samples - n_samples // 2)
        })

        logger.info(f"Generated count matrix: {counts_df.shape}")
        logger.info(f"Tumor samples: {n_tumor}, Normal samples: {n_normal}")
        logger.info(f"Cancer genes embedded: {len(cancer_genes)}")

        # Save
        counts_df.to_csv(self.output_dir / "raw_counts.csv")
        metadata_df.to_csv(self.output_dir / "metadata.csv", index=False)

        self.raw_counts = counts_df
        self.metadata = metadata_df

        return counts_df, metadata_df

    # =========================================================================
    # STEP 2: DESeq2 Differential Expression Analysis
    # =========================================================================

    def run_deseq2(self, counts: pd.DataFrame, metadata: pd.DataFrame,
                   condition_col: str = 'condition',
                   reference_level: str = 'normal') -> pd.DataFrame:
        """
        Run DESeq2 differential expression analysis via rpy2.

        Args:
            counts: Count matrix (genes x samples)
            metadata: Sample metadata with condition column
            condition_col: Column name for condition
            reference_level: Reference level for comparison

        Returns:
            DataFrame with DESeq2 results
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: DESeq2 Differential Expression Analysis")
        logger.info("=" * 70)

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr
            from rpy2.robjects.conversion import localconverter

            # Use context manager for pandas conversion (new rpy2 style)
            converter = ro.default_converter + pandas2ri.converter

            # Import R packages
            logger.info("Loading R packages...")
            base = importr('base')
            deseq2 = importr('DESeq2')

            logger.info("  ✓ DESeq2 loaded")

            # Prepare data
            # Align samples
            common_samples = list(set(counts.columns) & set(metadata['sample_id']))
            counts_aligned = counts[common_samples]
            meta_aligned = metadata[metadata['sample_id'].isin(common_samples)].copy()
            meta_aligned = meta_aligned.set_index('sample_id').loc[common_samples].reset_index()

            logger.info(f"\nSamples: {len(common_samples)}")
            logger.info(f"Genes: {len(counts_aligned)}")

            # Convert to R objects using context manager
            # Note: DESeq2 expects counts as genes (rows) x samples (columns)
            # The counts_aligned DataFrame is already in this format
            with localconverter(converter):
                # Convert counts - keep as genes x samples (no transpose needed for conversion)
                counts_r = ro.conversion.py2rpy(counts_aligned)
                # Convert metadata - set sample_id as index for proper rownames
                meta_r = ro.conversion.py2rpy(meta_aligned.set_index('sample_id'))

            # Run DESeq2 in R
            ro.r('''
            run_deseq2_analysis <- function(counts, coldata, condition_col, ref_level) {
                library(DESeq2)

                # Convert to matrix and ensure integer counts
                count_matrix <- as.matrix(counts)
                mode(count_matrix) <- "integer"

                # Ensure coldata rownames match count matrix column names
                # and are in the same order
                common_samples <- intersect(colnames(count_matrix), rownames(coldata))
                count_matrix <- count_matrix[, common_samples, drop=FALSE]
                coldata <- coldata[common_samples, , drop=FALSE]

                # Ensure condition is a factor with correct reference
                coldata[[condition_col]] <- factor(coldata[[condition_col]])
                coldata[[condition_col]] <- relevel(coldata[[condition_col]], ref = ref_level)

                # Create formula
                design_formula <- as.formula(paste("~", condition_col))

                # Create DESeq2 object
                dds <- DESeqDataSetFromMatrix(
                    countData = count_matrix,
                    colData = coldata,
                    design = design_formula
                )

                # Filter low counts
                keep <- rowSums(counts(dds)) >= 10
                dds <- dds[keep,]

                # Run DESeq2
                dds <- DESeq(dds)

                # Get results
                res <- results(dds, alpha = 0.05)
                res_df <- as.data.frame(res)
                res_df$gene <- rownames(res_df)

                # Also get normalized counts
                norm_counts <- as.data.frame(counts(dds, normalized=TRUE))

                return(list(results=res_df, normalized_counts=norm_counts))
            }
            ''')

            run_deseq2_r = ro.globalenv['run_deseq2_analysis']

            logger.info("\nRunning DESeq2...")
            results = run_deseq2_r(counts_r, meta_r, condition_col, reference_level)

            # Convert results back to pandas
            with localconverter(converter):
                results_df = ro.conversion.rpy2py(results.rx2('results'))
                norm_counts = ro.conversion.rpy2py(results.rx2('normalized_counts'))

            # Clean up results
            results_df = results_df.dropna(subset=['padj'])
            results_df = results_df.rename(columns={
                'log2FoldChange': 'log2FoldChange',
                'pvalue': 'pvalue',
                'padj': 'padj',
                'baseMean': 'baseMean'
            })

            # Sort by adjusted p-value
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

            logger.info(f"\nTop 15 DEGs:")
            display_cols = ['gene', 'log2FoldChange', 'padj', 'baseMean']
            logger.info(f"\n{sig_results[display_cols].head(15).to_string()}")

            # Save results
            results_df.to_csv(self.output_dir / "deseq2_all_results.csv", index=False)
            sig_results.to_csv(self.output_dir / "deseq2_significant.csv", index=False)

            self.deg_results = results_df

            logger.info(f"\n✓ DESeq2 analysis complete")

            return results_df

        except ImportError as e:
            logger.error(f"rpy2 or DESeq2 not available: {e}")
            logger.info("Falling back to scipy-based analysis...")
            return self.run_scipy_deg(counts, metadata, condition_col)

        except Exception as e:
            logger.error(f"DESeq2 failed: {e}")
            logger.info("Falling back to scipy-based analysis...")
            return self.run_scipy_deg(counts, metadata, condition_col)

    def run_scipy_deg(self, counts: pd.DataFrame, metadata: pd.DataFrame,
                      condition_col: str = 'condition') -> pd.DataFrame:
        """
        Fallback DEG analysis using scipy (Wilcoxon rank-sum test).
        """
        from scipy import stats
        from scipy.stats import rankdata

        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Differential Expression (scipy fallback)")
        logger.info("=" * 70)

        # Get sample groups
        common_samples = list(set(counts.columns) & set(metadata['sample_id']))
        counts_subset = counts[common_samples]
        meta_subset = metadata[metadata['sample_id'].isin(common_samples)].set_index('sample_id')

        # Log2 transform
        counts_log = np.log2(counts_subset + 1)

        # Get group samples
        conditions = meta_subset[condition_col].unique()
        if 'tumor' in conditions:
            group1 = meta_subset[meta_subset[condition_col] == 'tumor'].index.tolist()
            group2 = meta_subset[meta_subset[condition_col] == 'normal'].index.tolist()
        else:
            group1 = meta_subset[meta_subset[condition_col] == conditions[0]].index.tolist()
            group2 = meta_subset[meta_subset[condition_col] == conditions[1]].index.tolist()

        logger.info(f"Group 1: {len(group1)} samples")
        logger.info(f"Group 2: {len(group2)} samples")

        # DEG analysis
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

        # FDR correction
        n = len(results_df)
        ranked = rankdata(results_df['pvalue'])
        results_df['padj'] = np.minimum(1, results_df['pvalue'] * n / ranked)
        results_df = results_df.sort_values('padj')

        # Significant DEGs
        sig_results = results_df[
            (results_df['padj'] < 0.05) &
            (abs(results_df['log2FoldChange']) > 1.0)
        ]

        logger.info(f"\nTotal genes: {len(results_df)}")
        logger.info(f"Significant DEGs: {len(sig_results)}")

        # Save
        results_df.to_csv(self.output_dir / "deg_results.csv", index=False)
        sig_results.to_csv(self.output_dir / "deg_significant.csv", index=False)

        self.deg_results = results_df

        return results_df

    # =========================================================================
    # STEP 3: Network Analysis
    # =========================================================================

    def build_coexpression_network(self, deg_results: pd.DataFrame,
                                    counts: pd.DataFrame,
                                    top_n: int = 300,
                                    corr_threshold: float = 0.6) -> Tuple:
        """
        Build co-expression network from DEGs.
        """
        import networkx as nx

        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Co-expression Network Analysis")
        logger.info("=" * 70)

        # Get top DEGs
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

        # Get expression data
        if self.normalized_counts is not None:
            expr_data = self.normalized_counts.loc[genes_in_counts]
        else:
            expr_data = np.log2(counts.loc[genes_in_counts] + 1)

        # Compute correlation matrix
        logger.info("Computing Spearman correlations...")
        corr_matrix = expr_data.T.corr(method='spearman')

        # Build network
        G = nx.Graph()
        G.add_nodes_from(genes_in_counts)

        edge_count = 0
        for i, g1 in enumerate(genes_in_counts):
            for j, g2 in enumerate(genes_in_counts):
                if i < j:
                    corr = corr_matrix.loc[g1, g2]
                    if abs(corr) > corr_threshold:
                        G.add_edge(g1, g2, weight=abs(corr))
                        edge_count += 1

        logger.info(f"Network: {G.number_of_nodes()} nodes, {edge_count} edges")

        # Calculate centrality measures
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

        # Combine results
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

        # Normalize and compute composite score
        for col in ['degree', 'betweenness', 'pagerank', 'eigenvector']:
            max_val = hub_df[col].max()
            hub_df[f'{col}_norm'] = hub_df[col] / max_val if max_val > 0 else 0

        hub_df['network_score'] = (
            hub_df['degree_norm'] * 0.3 +
            hub_df['betweenness_norm'] * 0.25 +
            hub_df['pagerank_norm'] * 0.25 +
            hub_df['eigenvector_norm'] * 0.2
        )

        # Add expression significance score
        hub_df['expr_score'] = (1 - hub_df['padj'].clip(upper=1)) * np.abs(hub_df['log2FoldChange']) / 5

        # Composite score
        hub_df['composite_score'] = hub_df['network_score'] * 0.6 + hub_df['expr_score'] * 0.4

        hub_df = hub_df.sort_values('composite_score', ascending=False)

        logger.info(f"\nTop 15 Hub Genes:")
        display_cols = ['gene', 'degree', 'betweenness', 'log2FoldChange', 'composite_score']
        logger.info(f"\n{hub_df[display_cols].head(15).to_string()}")

        # Save
        hub_df.to_csv(self.output_dir / "hub_genes.csv", index=False)

        self.hub_genes = hub_df

        return G, hub_df

    # =========================================================================
    # STEP 4: Pathway Enrichment
    # =========================================================================

    def run_enrichment(self, deg_results: pd.DataFrame,
                       top_n: int = 200) -> pd.DataFrame:
        """
        Run pathway enrichment analysis using gseapy.
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Pathway Enrichment Analysis")
        logger.info("=" * 70)

        try:
            import gseapy as gp

            # Get significant DEGs
            sig_genes = deg_results[
                (deg_results['padj'] < 0.05) &
                (abs(deg_results['log2FoldChange']) > 1.0)
            ]['gene'].tolist()[:top_n]

            logger.info(f"Running enrichment for {len(sig_genes)} genes...")

            # Run Enrichr
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
                logger.info(f"\nTop 10 pathways:")
                logger.info(f"\n{sig_pathways[['Term', 'Adjusted P-value']].head(10).to_string()}")

                sig_pathways.to_csv(self.output_dir / "pathway_enrichment.csv", index=False)

            return sig_pathways

        except Exception as e:
            logger.warning(f"Enrichment failed: {e}")
            return pd.DataFrame()

    # =========================================================================
    # STEP 5: Validation & Report
    # =========================================================================

    def validate_results(self, hub_df: pd.DataFrame) -> set:
        """
        Validate hub genes against known cancer genes.
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Validation")
        logger.info("=" * 70)

        known_cancer_genes = {
            'EGFR', 'KRAS', 'TP53', 'ALK', 'ROS1', 'MET', 'BRAF', 'PIK3CA',
            'PTEN', 'RB1', 'CDKN2A', 'STK11', 'KEAP1', 'NF1', 'ERBB2', 'RET',
            'MYC', 'CCND1', 'CDK4', 'MDM2', 'SOX2', 'TERT', 'FOXM1', 'E2F1',
            'TOP2A', 'BIRC5', 'AURKA', 'CDC20', 'UBE2C', 'CCNB1', 'BUB1',
            'KIF11', 'TYMS', 'MCM2', 'SPP1', 'MMP1', 'MMP9', 'COL1A1',
            'VEGFA', 'HIF1A', 'BCL2', 'BAX', 'AKT1', 'MTOR'
        }

        top_hubs = set(hub_df.head(50)['gene'].tolist())
        validated = top_hubs & known_cancer_genes

        logger.info(f"\nTop 50 hub genes analyzed")
        logger.info(f"Known cancer genes in database: {len(known_cancer_genes)}")
        logger.info(f"Validated hub genes: {len(validated)} ({len(validated)/50*100:.1f}%)")

        if validated:
            logger.info(f"\nValidated genes:")
            for gene in sorted(validated):
                hub_row = hub_df[hub_df['gene'] == gene]
                if len(hub_row) > 0:
                    score = hub_row.iloc[0]['composite_score']
                    log2fc = hub_row.iloc[0]['log2FoldChange']
                    direction = "UP" if log2fc > 0 else "DOWN"
                    logger.info(f"  - {gene}: score={score:.4f}, {direction} (log2FC={log2fc:.2f})")

        return validated

    def generate_report(self, validated_genes: set) -> str:
        """
        Generate comprehensive analysis report.
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Report Generation")
        logger.info("=" * 70)

        deg_sig = self.deg_results[
            (self.deg_results['padj'] < 0.05) &
            (abs(self.deg_results['log2FoldChange']) > 1.0)
        ]

        report = f"""
================================================================================
          RNA-seq ANALYSIS PIPELINE REPORT
          Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. DATA SUMMARY
---------------
Total genes analyzed: {len(self.deg_results)}
Samples: {len(self.raw_counts.columns) if self.raw_counts is not None else 'N/A'}

2. DIFFERENTIAL EXPRESSION (DESeq2)
-----------------------------------
Significant DEGs (padj<0.05, |log2FC|>1): {len(deg_sig)}
Up-regulated: {len(deg_sig[deg_sig['log2FoldChange'] > 0])}
Down-regulated: {len(deg_sig[deg_sig['log2FoldChange'] < 0])}

Top 10 DEGs:
{deg_sig[['gene', 'log2FoldChange', 'padj']].head(10).to_string()}

3. NETWORK ANALYSIS
-------------------
Genes in network: {len(self.hub_genes)}

Top 10 Hub Genes:
{self.hub_genes[['gene', 'degree', 'composite_score', 'log2FoldChange']].head(10).to_string()}

4. VALIDATION
-------------
Known cancer genes identified: {len(validated_genes)}/50 ({len(validated_genes)/50*100:.1f}%)
Validated genes: {', '.join(sorted(validated_genes)) if validated_genes else 'None'}

5. OUTPUT FILES
---------------
{self.output_dir}/
├── raw_counts.csv (or normalized_counts.csv)
├── metadata.csv
├── deseq2_all_results.csv
├── deseq2_significant.csv
├── hub_genes.csv
├── pathway_enrichment.csv
└── analysis_report.txt

================================================================================
                    ANALYSIS COMPLETED SUCCESSFULLY
================================================================================
"""

        logger.info(report)

        with open(self.output_dir / "analysis_report.txt", 'w') as f:
            f.write(report)

        return report

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    def run_full_pipeline(self, use_synthetic: bool = True,
                          count_file: str = None,
                          metadata_file: str = None) -> Dict:
        """
        Run the complete RNA-seq analysis pipeline.

        Args:
            use_synthetic: Use synthetic data for testing
            count_file: Path to count matrix file
            metadata_file: Path to metadata file

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 70)
        logger.info("COMPLETE RNA-seq ANALYSIS PIPELINE")
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

            # Step 2: DESeq2 analysis
            deg_results = self.run_deseq2(counts, metadata)

            # Step 3: Network analysis
            G, hub_df = self.build_coexpression_network(deg_results, counts)

            # Step 4: Pathway enrichment
            try:
                pathway_results = self.run_enrichment(deg_results)
            except:
                pathway_results = pd.DataFrame()

            # Step 5: Validation
            validated = self.validate_results(hub_df)

            # Step 6: Report
            report = self.generate_report(validated)

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)

            return {
                'deg_results': deg_results,
                'hub_genes': hub_df,
                'pathway_results': pathway_results,
                'validated_genes': validated,
                'report': report
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Run the pipeline with synthetic data for testing."""

    output_dir = "/Users/admin/VectorDB_BioInsight/rnaseq_test_results/full_pipeline"

    pipeline = RNAseqPipeline(output_dir=output_dir)
    results = pipeline.run_full_pipeline(use_synthetic=True)

    return results


if __name__ == "__main__":
    main()
