"""
Agent 1: Differential Expression Gene (DEG) Analysis

Uses DESeq2 via rpy2 to perform differential expression analysis.

Input:
- count_matrix.csv: Gene expression count matrix (genes × samples)
- metadata.csv: Sample metadata with condition column
- config.json: Analysis parameters

Output:
- deg_all_results.csv: Full DESeq2 results
- deg_significant.csv: Filtered significant DEGs
- normalized_counts.csv: DESeq2 normalized counts
- meta_agent1.json: Execution metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..utils.base_agent import BaseAgent

# rpy2 imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


class DEGAgent(BaseAgent):
    """Agent for DESeq2-based differential expression analysis."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "contrast": ["tumor", "normal"],  # [treatment, control]
            "padj_cutoff": 0.05,
            "log2fc_cutoff": 1.0,
            "condition_column": "condition",
            "paired_column": None,  # Column for paired samples (e.g., "donor")
            "min_count_filter": 10,
            "use_synthetic_fallback": True,  # Use synthetic DEG if DESeq2 fails
            "use_apeglm_shrinkage": True,  # Apply apeglm LFC shrinkage
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent1_deg", input_dir, output_dir, merged_config)

        self.count_matrix: Optional[pd.DataFrame] = None
        self.metadata: Optional[pd.DataFrame] = None

    def validate_inputs(self) -> bool:
        """Validate count matrix and metadata."""
        # Load count matrix
        self.count_matrix = self.load_csv("count_matrix.csv")
        if self.count_matrix is None:
            return False

        # Load metadata
        self.metadata = self.load_csv("metadata.csv")
        if self.metadata is None:
            return False

        # Check condition column exists
        condition_col = self.config["condition_column"]
        if condition_col not in self.metadata.columns:
            self.logger.error(f"Condition column '{condition_col}' not in metadata")
            return False

        # Check sample IDs match
        count_samples = set(self.count_matrix.columns[1:])  # First col is gene_id
        meta_samples = set(self.metadata.iloc[:, 0])  # First col is sample_id

        if not count_samples.issubset(meta_samples):
            missing = count_samples - meta_samples
            self.logger.error(f"Samples missing from metadata: {missing}")
            return False

        # Check contrast groups exist
        conditions = set(self.metadata[condition_col])
        contrast = self.config["contrast"]
        if not all(c in conditions for c in contrast):
            self.logger.error(f"Contrast {contrast} not all in conditions {conditions}")
            return False

        self.logger.info(f"Count matrix: {self.count_matrix.shape[0]} genes, {self.count_matrix.shape[1]-1} samples")
        self.logger.info(f"Conditions: {conditions}")

        return True

    def _run_deseq2(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run DESeq2 analysis via rpy2."""
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed. Install with: pip install rpy2")

        self.logger.info("Initializing R environment...")

        # Activate rpy2 converter context for multithreading support
        # This fixes "Conversion rules for rpy2.robjects appear to be missing" error
        from rpy2.robjects import conversion
        try:
            # Try to use the context-aware conversion (rpy2 >= 3.5)
            pandas2ri.activate()
            self.logger.info("Activated pandas2ri converter")
        except Exception as e:
            self.logger.warning(f"Could not activate pandas2ri: {e}")

        # Import R packages
        base = importr('base')
        deseq2 = importr('DESeq2')

        # Prepare count matrix (genes as rows, samples as columns)
        count_df = self.count_matrix.set_index(self.count_matrix.columns[0])

        # Filter low counts
        min_count = self.config["min_count_filter"]
        count_df = count_df[count_df.sum(axis=1) >= min_count]
        self.logger.info(f"After filtering (min_count={min_count}): {len(count_df)} genes")

        # Prepare metadata
        condition_col = self.config["condition_column"]
        meta_df = self.metadata.set_index(self.metadata.columns[0])

        # Ensure sample order matches
        sample_order = count_df.columns.tolist()
        meta_df = meta_df.loc[sample_order]

        self.logger.info("Converting to R objects...")

        with localconverter(ro.default_converter + pandas2ri.converter):
            counts_r = ro.conversion.py2rpy(count_df.astype(int))
            meta_r = ro.conversion.py2rpy(meta_df)

        # Create DESeqDataSet
        self.logger.info("Creating DESeqDataSet...")

        # Check for paired design
        paired_col = self.config.get("paired_column")
        if paired_col and paired_col in meta_df.columns:
            self.logger.info(f"Using PAIRED design: ~ {paired_col} + {condition_col}")
            design_formula = ro.Formula(f"~ {paired_col} + {condition_col}")
        else:
            self.logger.info(f"Using UNPAIRED design: ~ {condition_col}")
            design_formula = ro.Formula(f"~ {condition_col}")

        dds = deseq2.DESeqDataSetFromMatrix(
            countData=counts_r,
            colData=meta_r,
            design=design_formula
        )

        # Run DESeq2
        self.logger.info("Running DESeq2 (this may take a while)...")
        dds = deseq2.DESeq(dds)

        # Extract results
        contrast = self.config["contrast"]
        self.logger.info(f"Extracting results for contrast: {contrast[0]} vs {contrast[1]}")

        res = deseq2.results(
            dds,
            contrast=ro.StrVector([condition_col, contrast[0], contrast[1]])
        )

        # Apply apeglm LFC shrinkage if enabled
        if self.config.get("use_apeglm_shrinkage", True):
            try:
                self.logger.info("Applying apeglm LFC shrinkage...")
                # apeglm requires coef name instead of contrast
                coef_name = f"{condition_col}_{contrast[0]}_vs_{contrast[1]}"

                # Check if apeglm is available
                ro.r('if (!requireNamespace("apeglm", quietly = TRUE)) stop("apeglm not installed")')

                # Get result names to find correct coefficient
                result_names = list(deseq2.resultsNames(dds))
                self.logger.info(f"Available coefficients: {result_names}")

                # Find the coefficient that matches our contrast
                matching_coef = None
                for rn in result_names:
                    if contrast[0] in rn and contrast[1] in rn:
                        matching_coef = rn
                        break
                    elif condition_col in rn and contrast[0] in rn:
                        matching_coef = rn
                        break

                if matching_coef:
                    self.logger.info(f"Using coefficient: {matching_coef}")
                    res_shrunk = deseq2.lfcShrink(
                        dds,
                        coef=matching_coef,
                        type="apeglm"
                    )
                    res = res_shrunk
                    self.logger.info("apeglm shrinkage applied successfully")
                else:
                    self.logger.warning(f"Could not find matching coefficient for {contrast}, using unshrunk LFC")
            except Exception as e:
                self.logger.warning(f"apeglm shrinkage failed: {e}. Using unshrunk LFC.")

        # Get normalized counts using BiocGenerics
        self.logger.info("Getting normalized counts...")
        bioc_generics = importr('BiocGenerics')
        norm_counts = bioc_generics.counts(dds, normalized=True)

        # Convert back to pandas
        with localconverter(ro.default_converter + pandas2ri.converter):
            results_df = ro.conversion.rpy2py(base.as_data_frame(res))
            norm_counts_df = ro.conversion.rpy2py(base.as_data_frame(norm_counts))

        # Log original columns for debugging
        self.logger.info(f"DESeq2 result columns: {list(results_df.columns)}")
        self.logger.info(f"DESeq2 result shape: {results_df.shape}")

        # Dynamically map columns based on what DESeq2 returns
        # apeglm shrinkage returns: baseMean, log2FoldChange, lfcSE, pvalue, padj (5 cols, no stat)
        # Normal results return: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj (6 cols)
        col_mapping = {
            'baseMean': 'baseMean',
            'log2FoldChange': 'log2FC',
            'lfcSE': 'lfcSE',
            'stat': 'stat',
            'pvalue': 'pvalue',
            'padj': 'padj'
        }

        # Rename existing columns
        results_df = results_df.rename(columns=col_mapping)

        # Add stat column if missing (apeglm doesn't return it)
        if 'stat' not in results_df.columns:
            results_df['stat'] = results_df['log2FC'] / results_df['lfcSE'].replace(0, np.nan)

        # Add gene_id from index
        results_df.insert(0, 'gene_id', count_df.index)
        norm_counts_df.insert(0, 'gene_id', count_df.index)

        # Ensure consistent column order
        expected_cols = ['gene_id', 'baseMean', 'log2FC', 'lfcSE', 'stat', 'pvalue', 'padj']
        results_df = results_df[expected_cols]

        self.logger.info(f"Final result shape: {results_df.shape}")

        return results_df, norm_counts_df

    def _run_synthetic_deg(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic DEG results (fallback when DESeq2 fails)."""
        self.logger.warning("Using synthetic DEG analysis (DESeq2 unavailable)")

        count_df = self.count_matrix.set_index(self.count_matrix.columns[0])
        condition_col = self.config["condition_column"]
        contrast = self.config["contrast"]

        # Get samples per group
        group1_samples = self.metadata[
            self.metadata[condition_col] == contrast[0]
        ].iloc[:, 0].tolist()
        group2_samples = self.metadata[
            self.metadata[condition_col] == contrast[1]
        ].iloc[:, 0].tolist()

        self.logger.info(f"Group 1 ({contrast[0]}): {len(group1_samples)} samples")
        self.logger.info(f"Group 2 ({contrast[1]}): {len(group2_samples)} samples")

        # Verify samples exist in count_df columns
        g1_valid = [s for s in group1_samples if s in count_df.columns]
        g2_valid = [s for s in group2_samples if s in count_df.columns]

        if len(g1_valid) == 0 or len(g2_valid) == 0:
            self.logger.error(f"Sample mismatch! g1_valid={len(g1_valid)}, g2_valid={len(g2_valid)}")
            self.logger.error(f"Count matrix columns (first 5): {list(count_df.columns[:5])}")
            self.logger.error(f"Group1 samples (first 5): {group1_samples[:5]}")
            raise ValueError("No valid samples found for DEG analysis")

        if len(g1_valid) != len(group1_samples) or len(g2_valid) != len(group2_samples):
            self.logger.warning(f"Some samples missing from count matrix: g1={len(g1_valid)}/{len(group1_samples)}, g2={len(g2_valid)}/{len(group2_samples)}")
            group1_samples = g1_valid
            group2_samples = g2_valid

        # Calculate log2 fold change
        mean1 = count_df[group1_samples].mean(axis=1) + 1
        mean2 = count_df[group2_samples].mean(axis=1) + 1
        log2fc = np.log2(mean1 / mean2)

        self.logger.info(f"Calculated log2FC for {len(log2fc)} genes")
        self.logger.info(f"log2FC range: {log2fc.min():.2f} to {log2fc.max():.2f}")

        # Calculate p-values (t-test approximation)
        from scipy import stats
        pvalues = []
        for i, gene in enumerate(count_df.index):
            g1 = count_df.loc[gene, group1_samples].values.astype(float)
            g2 = count_df.loc[gene, group2_samples].values.astype(float)
            try:
                _, pval = stats.ttest_ind(g1, g2)
                if np.isnan(pval):
                    pval = 1.0  # Replace NaN with non-significant
                pvalues.append(pval)
            except Exception:
                pvalues.append(1.0)  # Assign non-significant p-value on error

            if (i + 1) % 10000 == 0:
                self.logger.info(f"Processed {i + 1}/{len(count_df)} genes for t-test...")

        pvalues = np.array(pvalues)

        # Replace any remaining NaN with 1.0 BEFORE multipletests
        nan_count = np.isnan(pvalues).sum()
        if nan_count > 0:
            self.logger.warning(f"Replacing {nan_count} NaN p-values with 1.0")
            pvalues = np.where(np.isnan(pvalues), 1.0, pvalues)

        self.logger.info(f"P-values < 0.05: {(pvalues < 0.05).sum()}")

        # Adjust p-values (Benjamini-Hochberg)
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(pvalues, method='fdr_bh')

        self.logger.info(f"Adjusted p-values < 0.05: {(padj < 0.05).sum()}")

        # Create results DataFrame
        results_df = pd.DataFrame({
            'gene_id': count_df.index,
            'baseMean': count_df.mean(axis=1).values,
            'log2FC': log2fc.values,
            'lfcSE': np.abs(log2fc.values) * 0.1,  # Approximate
            'stat': log2fc.values / 0.5,  # Approximate
            'pvalue': pvalues,
            'padj': padj
        })

        self.logger.info(f"Results DataFrame created: {len(results_df)} rows")

        # Normalized counts (simple CPM)
        total_counts = count_df.sum(axis=0)
        norm_counts_df = count_df.div(total_counts, axis=1) * 1e6
        norm_counts_df.insert(0, 'gene_id', count_df.index)

        return results_df, norm_counts_df

    def run(self) -> Dict[str, Any]:
        """Execute DEG analysis."""
        # Try DESeq2 first, fallback to synthetic
        try:
            if HAS_RPY2:
                results_df, norm_counts_df = self._run_deseq2()
                method_used = "DESeq2"
            else:
                results_df, norm_counts_df = self._run_synthetic_deg()
                method_used = "synthetic_ttest"
        except Exception as e:
            self.logger.warning(f"DESeq2 failed: {e}")
            if self.config["use_synthetic_fallback"]:
                results_df, norm_counts_df = self._run_synthetic_deg()
                method_used = "synthetic_ttest_fallback"
            else:
                raise

        # Debug: Log results before dropna
        self.logger.info(f"Results before dropna: {len(results_df)} rows")
        na_count = results_df['padj'].isna().sum()
        self.logger.info(f"NA padj values: {na_count}")

        # Remove NA padj values
        results_df = results_df.dropna(subset=['padj'])
        self.logger.info(f"Results after dropna: {len(results_df)} rows")

        # Save all results
        self.save_csv(results_df, "deg_all_results.csv")

        # Filter significant DEGs
        padj_cutoff = self.config["padj_cutoff"]
        log2fc_cutoff = self.config["log2fc_cutoff"]

        significant = results_df[
            (results_df['padj'] < padj_cutoff) &
            (np.abs(results_df['log2FC']) > log2fc_cutoff)
        ].copy()

        # Add direction column
        significant['direction'] = np.where(significant['log2FC'] > 0, 'up', 'down')

        # Sort by padj
        significant = significant.sort_values('padj')

        self.save_csv(significant[['gene_id', 'log2FC', 'padj', 'direction']], "deg_significant.csv")

        # Save normalized counts
        self.save_csv(norm_counts_df, "normalized_counts.csv")

        # Calculate statistics
        up_count = (significant['direction'] == 'up').sum()
        down_count = (significant['direction'] == 'down').sum()

        self.logger.info(f"DEG Analysis Complete:")
        self.logger.info(f"  Total genes analyzed: {len(results_df)}")
        self.logger.info(f"  Significant DEGs: {len(significant)}")
        self.logger.info(f"  Upregulated: {up_count}")
        self.logger.info(f"  Downregulated: {down_count}")

        return {
            "method_used": method_used,
            "total_genes": len(results_df),
            "deg_count": len(significant),
            "up_count": int(up_count),
            "down_count": int(down_count),
            "padj_cutoff": padj_cutoff,
            "log2fc_cutoff": log2fc_cutoff
        }

    def validate_outputs(self) -> bool:
        """Validate DEG outputs."""
        # Check files exist
        required_files = [
            "deg_all_results.csv",
            "deg_significant.csv",
            "normalized_counts.csv"
        ]

        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing output file: {filename}")
                return False

        # Check significant DEGs file is valid
        sig_df = pd.read_csv(self.output_dir / "deg_significant.csv")

        if len(sig_df) == 0:
            self.logger.warning("No significant DEGs found (this may be expected)")
            # Still valid, just a warning

        # Check no NA in padj
        if sig_df['padj'].isna().any():
            self.logger.error("NA values found in padj column")
            return False

        return True
