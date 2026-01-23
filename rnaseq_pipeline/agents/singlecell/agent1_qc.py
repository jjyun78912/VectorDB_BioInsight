"""
Single-Cell Agent 1: QC & Preprocessing

Quality control, filtering, normalization, and feature selection.

Input:
- count_matrix.csv or *.h5ad file
- metadata.csv (optional)

Output:
- adata_qc.h5ad: QC-filtered and normalized data
- qc_statistics.json: QC metrics and filtering summary
- cell_cycle_info.json: Cell cycle phase distribution
- figures/qc_*.png: QC visualization plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import json

warnings.filterwarnings('ignore')

from ...utils.base_agent import BaseAgent

# Import scanpy
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

# Import scrublet for doublet detection
try:
    import scrublet as scr
    HAS_SCRUBLET = True
except ImportError:
    HAS_SCRUBLET = False

# Import matplotlib for QC visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Cell cycle genes (Tirosh et al. 2016) # cell cycle을 분류할 수 있는 표준, 핵심 마커 유전자
S_GENES = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG',
    'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP',
    'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76',
    'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51',
    'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
    'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
]

G2M_GENES = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A',
    'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF',
    'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB',
    'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP',
    'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1',
    'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR',
    'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF',
    'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
]
#G1_genes은 휴지기이기 때문에 특벽할 활동이 없어서 특이적 마커 없음. -> 소거법을 표준으로 사용함.  

class SingleCellQCAgent(BaseAgent):
    """Agent 1: Quality Control & Preprocessing for single-cell RNA-seq."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Basic QC thresholds
            "min_genes_per_cell": 200,
            "max_genes_per_cell": 5000,
            "min_cells_per_gene": 3,
            "max_mito_percent": 20,
            "max_ribo_percent": 50,

            # Empty droplet filter
            "enable_empty_droplet_filter": True,
            "min_counts_per_cell": 500,

            # Doublet detection
            "enable_doublet_detection": True,
            "doublet_rate": 0.06,
            "doublet_score_threshold": 0.25,

            # Ambient RNA correction
            "enable_ambient_rna_correction": False,
            "ambient_rna_contamination": 0.1,

            # Cell cycle scoring
            "enable_cell_cycle_scoring": True,
            "regress_cell_cycle": False,

            # Normalization
            "target_sum": 10000,
            "log_transform": True,

            # HVG selection
            "n_top_genes": 2000,
            "hvg_flavor": "seurat_v3",

            # Scaling
            "scale_max_value": 10,

            # PCA
            "n_pcs": 50,

            # Visualization
            "enable_qc_plots": True,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent1_sc_qc", input_dir, output_dir, merged_config)

        self.adata = None
        self.qc_stats = {}
        self.cell_cycle_info = {}

    def validate_inputs(self) -> bool:
        """Validate input files."""
        if not HAS_SCANPY:
            self.logger.error("Scanpy not installed. Run: pip install scanpy")
            return False

        # Check for h5ad file first
        h5ad_files = list(self.input_dir.glob("*.h5ad"))
        if h5ad_files:
            self.logger.info(f"Found h5ad file: {h5ad_files[0].name}")
            self.adata = sc.read_h5ad(h5ad_files[0])
            self.logger.info(f"Loaded: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")

            # CRITICAL: Auto-detect and convert feature_name to var_names if needed
            # CellXGene Census data uses numeric indices as var_names, with gene symbols in feature_name
            self._fix_var_names_if_needed()

            return True

        # Check for count matrix
        count_matrix = self.load_csv("count_matrix.csv", required=False)
        if count_matrix is None:
            for name in ["matrix.csv", "expression.csv", "counts.csv"]:
                count_matrix = self.load_csv(name, required=False)
                if count_matrix is not None:
                    break

        if count_matrix is None:
            self.logger.error("No count matrix found")
            return False

        # Create AnnData
        gene_ids = count_matrix.iloc[:, 0].values
        cell_ids = count_matrix.columns[1:].tolist()
        X = count_matrix.iloc[:, 1:].values.T

        self.adata = sc.AnnData(
            X=X.astype(np.float32),
            obs=pd.DataFrame(index=cell_ids),
            var=pd.DataFrame(index=gene_ids)
        )
        self.adata.var_names_make_unique()

        # Load metadata if available
        metadata = self.load_csv("metadata.csv", required=False)
        if metadata is not None:
            metadata = metadata.set_index(metadata.columns[0])
            for col in metadata.columns:
                if col not in self.adata.obs.columns:
                    self.adata.obs[col] = metadata.reindex(self.adata.obs_names)[col].values

        self.logger.info(f"Created AnnData: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        return True

    def _fix_var_names_if_needed(self):
        """
        Fix var_names if they are numeric indices (e.g., from CellXGene Census).
        Gene symbols should be in var_names for downstream analysis (CellTypist, Pathway, CNV).
        """
        from collections import Counter

        # Check if var_names are numeric (common in CellXGene Census data)
        sample_names = self.adata.var_names[:10].tolist()
        looks_numeric = all(str(n).isdigit() for n in sample_names)

        # Check for common gene symbol columns
        gene_symbol_columns = ['feature_name', 'gene_symbol', 'gene_name', 'symbol', 'gene']
        gene_col = None

        for col in gene_symbol_columns:
            if col in self.adata.var.columns:
                gene_col = col
                break

        # If var_names look numeric and we have a gene symbol column, convert
        if looks_numeric and gene_col:
            self.logger.info(f"Detected numeric var_names, converting from '{gene_col}' column...")

            # Store original index
            self.adata.var['original_index'] = self.adata.var_names.copy()

            # Get gene symbols
            new_var_names = self.adata.var[gene_col].values.astype(str)

            # Handle duplicates by making them unique
            name_counts = Counter(new_var_names)
            seen = {}
            unique_names = []

            for name in new_var_names:
                if name_counts[name] > 1:
                    if name not in seen:
                        seen[name] = 0
                    seen[name] += 1
                    unique_names.append(f"{name}_{seen[name]}")
                else:
                    unique_names.append(name)

            self.adata.var_names = unique_names
            self.adata.var_names_make_unique()

            self.logger.info(f"  Converted var_names: {self.adata.var_names[:5].tolist()}...")
            self.logger.info(f"  Gene symbols ready for CellTypist and pathway analysis")

        elif looks_numeric:
            self.logger.warning(f"var_names are numeric but no gene symbol column found!")
            self.logger.warning(f"  Available columns: {list(self.adata.var.columns)}")
            self.logger.warning(f"  CellTypist and pathway analysis may fail")

    def run(self) -> Dict[str, Any]:
        """Execute QC pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Agent 1: QC & Preprocessing")
        self.logger.info("=" * 60)

        # Step 1: Calculate QC metrics
        self._calculate_qc_metrics()

        # Step 2: Filter cells and genes
        self._filter_cells()

        # Step 3: Detect doublets
        if self.config.get("enable_doublet_detection", True):
            self._detect_doublets()

        # Step 4: Filter genes
        self._filter_genes()

        # Step 5: Normalize
        self._normalize()

        # Step 6: Cell cycle scoring
        if self.config.get("enable_cell_cycle_scoring", True):
            self._score_cell_cycle()

        # Step 7: HVG selection
        self._select_hvg()

        # Step 8: Scale and PCA
        self._scale_and_pca()

        # Step 9: Generate QC plots
        if self.config.get("enable_qc_plots", True):
            self._generate_qc_plots()

        # Save outputs
        self._save_outputs()

        return {
            "status": "success",
            "n_cells": self.adata.n_obs,
            "n_genes": self.adata.n_vars,
            "n_hvg": self.adata.var['highly_variable'].sum() if 'highly_variable' in self.adata.var else 0,
            "qc_stats": self.qc_stats,
            "cell_cycle_info": self.cell_cycle_info,
            "output_file": str(self.output_dir / "adata_qc.h5ad")
        }

    def _calculate_qc_metrics(self):
        """Calculate QC metrics."""
        self.logger.info("Calculating QC metrics...")

        # Mitochondrial genes
        self.adata.var['mt'] = self.adata.var_names.str.startswith(('MT-', 'mt-'))
        # Ribosomal genes
        self.adata.var['ribo'] = self.adata.var_names.str.startswith(('RPS', 'RPL', 'Rps', 'Rpl'))
        # Hemoglobin genes
        self.adata.var['hb'] = self.adata.var_names.str.contains('^HB[^(P)]', regex=True)

        sc.pp.calculate_qc_metrics(
            self.adata,
            qc_vars=['mt', 'ribo', 'hb'],
            percent_top=None,
            log1p=False,
            inplace=True
        )

        self.qc_stats = {
            "initial_cells": self.adata.n_obs,
            "initial_genes": self.adata.n_vars,
            "filters_applied": []
        }

        self.logger.info(f"  Initial: {self.adata.n_obs} cells x {self.adata.n_vars} genes")

    def _filter_cells(self):
        """Filter cells based on QC metrics."""
        self.logger.info("Filtering cells...")

        n_before = self.adata.n_obs

        # Empty droplet filter
        if self.config.get("enable_empty_droplet_filter", True):
            min_counts = self.config.get("min_counts_per_cell", 500)
            sc.pp.filter_cells(self.adata, min_counts=min_counts)
            n_empty = n_before - self.adata.n_obs
            self.logger.info(f"  Empty droplet filter: {n_empty} cells removed")
            self.qc_stats["filters_applied"].append({
                "filter": "empty_droplet",
                "threshold": min_counts,
                "cells_removed": n_empty
            })
            n_before = self.adata.n_obs

        # Min genes filter
        sc.pp.filter_cells(self.adata, min_genes=self.config["min_genes_per_cell"])
        n_low = n_before - self.adata.n_obs
        n_before = self.adata.n_obs

        # Max genes filter
        self.adata = self.adata[
            self.adata.obs['n_genes_by_counts'] < self.config["max_genes_per_cell"]
        ].copy()
        n_high = n_before - self.adata.n_obs

        self.logger.info(f"  Gene count filter: {n_low + n_high} cells removed")
        self.qc_stats["filters_applied"].append({
            "filter": "gene_count",
            "min_genes": self.config["min_genes_per_cell"],
            "max_genes": self.config["max_genes_per_cell"],
            "cells_removed": n_low + n_high
        })

        # Mito filter
        n_before = self.adata.n_obs
        self.adata = self.adata[
            self.adata.obs['pct_counts_mt'] < self.config["max_mito_percent"]
        ].copy()
        n_mito = n_before - self.adata.n_obs

        self.logger.info(f"  Mito filter (<{self.config['max_mito_percent']}%): {n_mito} cells removed")
        self.qc_stats["filters_applied"].append({
            "filter": "mitochondrial",
            "threshold_percent": self.config["max_mito_percent"],
            "cells_removed": n_mito
        })

    def _detect_doublets(self):
        """Detect doublets using Scrublet."""
        if not HAS_SCRUBLET:
            self.logger.warning("Scrublet not installed. Skipping doublet detection.")
            return

        self.logger.info("Detecting doublets with Scrublet...")

        try:
            counts_matrix = self.adata.X
            if hasattr(counts_matrix, 'toarray'):
                counts_matrix = counts_matrix.toarray()

            scrub = scr.Scrublet(
                counts_matrix,
                expected_doublet_rate=self.config.get("doublet_rate", 0.06)
            )

            doublet_scores, _ = scrub.scrub_doublets(
                min_counts=2,
                min_cells=3,
                min_gene_variability_pctl=85,
                n_prin_comps=30
            )

            self.adata.obs['doublet_score'] = doublet_scores

            threshold = self.config.get("doublet_score_threshold", 0.25)
            predicted_doublets = doublet_scores > threshold
            self.adata.obs['predicted_doublet'] = predicted_doublets

            n_doublets = predicted_doublets.sum()
            n_before = self.adata.n_obs

            self.adata = self.adata[~self.adata.obs['predicted_doublet']].copy()

            self.logger.info(f"  Doublets detected: {n_doublets} ({n_doublets/n_before*100:.1f}%)")
            self.qc_stats["filters_applied"].append({
                "filter": "doublet_scrublet",
                "threshold": threshold,
                "doublets_detected": int(n_doublets),
                "doublet_rate": float(n_doublets/n_before)
            })

        except Exception as e:
            self.logger.warning(f"Doublet detection failed: {e}")

    def _filter_genes(self):
        """Filter genes."""
        n_before = self.adata.n_vars
        sc.pp.filter_genes(self.adata, min_cells=self.config["min_cells_per_gene"])
        n_removed = n_before - self.adata.n_vars
        self.logger.info(f"Gene filter (min_cells={self.config['min_cells_per_gene']}): {n_removed} genes removed")

    def _normalize(self):
        """Normalize counts."""
        self.logger.info("Normalizing...")

        # Store raw counts
        self.adata.layers['counts'] = self.adata.X.copy()

        # Normalize
        sc.pp.normalize_total(self.adata, target_sum=self.config["target_sum"])

        # Log transform
        if self.config.get("log_transform", True):
            sc.pp.log1p(self.adata)

        self.logger.info(f"  Target sum: {self.config['target_sum']}, log1p: {self.config.get('log_transform', True)}")

    def _score_cell_cycle(self):
        """Score cell cycle phases."""
        self.logger.info("Scoring cell cycle...")

        try:
            # Get genes present in data
            s_genes_present = [g for g in S_GENES if g in self.adata.var_names]
            g2m_genes_present = [g for g in G2M_GENES if g in self.adata.var_names]

            if len(s_genes_present) < 5 or len(g2m_genes_present) < 5:
                self.logger.warning("  Not enough cell cycle genes found. Skipping.")
                return

            sc.tl.score_genes_cell_cycle(
                self.adata,
                s_genes=s_genes_present,
                g2m_genes=g2m_genes_present
            )

            phase_counts = self.adata.obs['phase'].value_counts().to_dict()

            self.cell_cycle_info = {
                "s_genes_used": s_genes_present,
                "g2m_genes_used": g2m_genes_present,
                "phase_distribution": {k: int(v) for k, v in phase_counts.items()}
            }

            self.logger.info(f"  Phase distribution: {phase_counts}")

            # Optionally regress out cell cycle
            if self.config.get("regress_cell_cycle", False):
                self.logger.info("  Regressing out cell cycle effects...")
                sc.pp.regress_out(self.adata, ['S_score', 'G2M_score'])

        except Exception as e:
            self.logger.warning(f"Cell cycle scoring failed: {e}")

    def _select_hvg(self):
        """Select highly variable genes."""
        self.logger.info("Selecting HVGs...")

        try:
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=self.config["n_top_genes"],
                flavor=self.config.get("hvg_flavor", "seurat_v3"),
                batch_key=self.config.get("batch_key"),
                subset=False
            )

            n_hvg = self.adata.var['highly_variable'].sum()
            self.logger.info(f"  Selected {n_hvg} HVGs")

        except Exception as e:
            self.logger.warning(f"HVG selection failed: {e}")
            self.adata.var['highly_variable'] = True

    def _scale_and_pca(self):
        """Scale data and compute PCA."""
        self.logger.info("Scaling and computing PCA...")

        # Store normalized data
        self.adata.raw = self.adata.copy()

        # Subset to HVGs for scaling
        adata_hvg = self.adata[:, self.adata.var['highly_variable']].copy()

        # Scale
        sc.pp.scale(adata_hvg, max_value=self.config.get("scale_max_value", 10))

        # PCA
        n_pcs = min(self.config["n_pcs"], adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
        sc.tl.pca(adata_hvg, n_comps=n_pcs)

        # Transfer PCA results back
        self.adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
        self.adata.varm['PCs'] = np.zeros((self.adata.n_vars, n_pcs))
        hvg_idx = self.adata.var['highly_variable'].values
        self.adata.varm['PCs'][hvg_idx] = adata_hvg.varm['PCs']
        self.adata.uns['pca'] = adata_hvg.uns['pca']

        self.logger.info(f"  Computed {n_pcs} PCs")

    def _generate_qc_plots(self):
        """Generate QC visualization plots."""
        if not HAS_MATPLOTLIB:
            return

        self.logger.info("Generating QC plots...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # Violin plots
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            if 'n_genes_by_counts' in self.adata.obs:
                axes[0].violinplot([self.adata.obs['n_genes_by_counts'].values])
                axes[0].set_title('Genes per Cell')
                axes[0].set_ylabel('Count')

            if 'total_counts' in self.adata.obs:
                axes[1].violinplot([self.adata.obs['total_counts'].values])
                axes[1].set_title('UMI Counts per Cell')
                axes[1].set_ylabel('Count')

            if 'pct_counts_mt' in self.adata.obs:
                axes[2].violinplot([self.adata.obs['pct_counts_mt'].values])
                axes[2].set_title('Mitochondrial %')
                axes[2].set_ylabel('Percent')

            plt.tight_layout()
            plt.savefig(figures_dir / "qc_violin.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Scatter plot: genes vs counts
            fig, ax = plt.subplots(figsize=(6, 5))
            if 'n_genes_by_counts' in self.adata.obs and 'total_counts' in self.adata.obs:
                scatter = ax.scatter(
                    self.adata.obs['total_counts'],
                    self.adata.obs['n_genes_by_counts'],
                    c=self.adata.obs.get('pct_counts_mt', 0),
                    cmap='viridis',
                    s=1,
                    alpha=0.5
                )
                ax.set_xlabel('Total Counts')
                ax.set_ylabel('Genes Detected')
                ax.set_title('QC Scatter')
                plt.colorbar(scatter, label='MT %')
            plt.tight_layout()
            plt.savefig(figures_dir / "qc_scatter.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Doublet score histogram
            if 'doublet_score' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(self.adata.obs['doublet_score'], bins=50, edgecolor='black')
                ax.axvline(
                    self.config.get("doublet_score_threshold", 0.25),
                    color='red', linestyle='--', label='Threshold'
                )
                ax.set_xlabel('Doublet Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Doublet Score Distribution')
                ax.legend()
                plt.tight_layout()
                plt.savefig(figures_dir / "qc_doublet_histogram.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Cell cycle pie chart
            if 'phase' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(6, 6))
                phase_counts = self.adata.obs['phase'].value_counts()
                colors = {'G1': '#2ecc71', 'S': '#3498db', 'G2M': '#e74c3c'}
                ax.pie(
                    phase_counts.values,
                    labels=phase_counts.index,
                    autopct='%1.1f%%',
                    colors=[colors.get(p, '#95a5a6') for p in phase_counts.index]
                )
                ax.set_title('Cell Cycle Distribution')
                plt.tight_layout()
                plt.savefig(figures_dir / "qc_cell_cycle.png", dpi=150, bbox_inches='tight')
                plt.close()

            self.logger.info(f"  Saved QC plots to {figures_dir}")

        except Exception as e:
            self.logger.warning(f"QC plot generation failed: {e}")

    def _save_outputs(self):
        """Save output files."""
        self.logger.info("Saving outputs...")

        # Update QC stats
        self.qc_stats["final_cells"] = self.adata.n_obs
        self.qc_stats["final_genes"] = self.adata.n_vars
        self.qc_stats["cells_removed_total"] = self.qc_stats["initial_cells"] - self.adata.n_obs
        self.qc_stats["genes_removed_total"] = self.qc_stats["initial_genes"] - self.adata.n_vars

        # Save h5ad
        output_h5ad = self.output_dir / "adata_qc.h5ad"
        self.adata.write_h5ad(output_h5ad)
        self.logger.info(f"  Saved: {output_h5ad}")

        # Save QC stats
        qc_json = self.output_dir / "qc_statistics.json"
        with open(qc_json, 'w') as f:
            json.dump(self.qc_stats, f, indent=2)
        self.logger.info(f"  Saved: {qc_json}")

        # Save cell cycle info
        if self.cell_cycle_info:
            cc_json = self.output_dir / "cell_cycle_info.json"
            with open(cc_json, 'w') as f:
                json.dump(self.cell_cycle_info, f, indent=2)
            self.logger.info(f"  Saved: {cc_json}")

        self.logger.info(f"QC Complete: {self.qc_stats['initial_cells']} -> {self.adata.n_obs} cells")

    def validate_outputs(self) -> bool:
        """Validate output files were generated correctly."""
        required_files = [
            self.output_dir / "adata_qc.h5ad",
            self.output_dir / "qc_statistics.json"
        ]
        for f in required_files:
            if not f.exists():
                self.logger.error(f"Required output missing: {f}")
                return False
        return True
