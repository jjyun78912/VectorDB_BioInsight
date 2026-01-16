"""
Single-Cell RNA-seq Analysis Agent

Unified pipeline for scRNA-seq analysis using Scanpy.

Input:
- count_matrix.csv: Gene expression count matrix (genes × cells)
  OR h5ad file directly
- metadata.csv (optional): Cell metadata with annotations

Output:
- adata.h5ad: Annotated data object
- cluster_markers.csv: Marker genes per cluster
- cell_composition.csv: Cell type composition
- umap_coordinates.csv: UMAP embeddings
- figures/: UMAP, Violin, Dot plots
- meta_singlecell.json: Execution metadata

Pipeline Steps:
1. QC & Filtering
2. Normalization (log1p)
3. HVG Selection
4. PCA & Batch Correction (Harmony)
5. UMAP/t-SNE
6. Clustering (Leiden)
7. Cell Type Annotation (CellTypist/Marker-based)
8. DEG Analysis (Wilcoxon)
9. Visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import json

warnings.filterwarnings('ignore')

from ..utils.base_agent import BaseAgent

# Import scanpy
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

# Import harmony for batch correction
try:
    import harmonypy
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False

# Import celltypist for annotation
try:
    import celltypist
    HAS_CELLTYPIST = True
except ImportError:
    HAS_CELLTYPIST = False


class SingleCellAgent(BaseAgent):
    """Agent for single-cell RNA-seq analysis using Scanpy."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # QC parameters
            "min_genes_per_cell": 200,
            "max_genes_per_cell": 5000,
            "min_cells_per_gene": 3,
            "max_mito_percent": 20,
            "max_ribo_percent": 50,

            # Normalization
            "target_sum": 10000,

            # HVG selection
            "n_top_genes": 2000,
            "hvg_flavor": "seurat_v3",  # seurat, seurat_v3, cell_ranger

            # Dimensionality reduction
            "n_pcs": 50,
            "n_neighbors": 15,
            "umap_min_dist": 0.3,

            # Clustering
            "clustering_resolution": 0.5,
            "clustering_method": "leiden",  # leiden, louvain

            # Batch correction
            "batch_key": None,  # Column name for batch correction
            "use_harmony": True,

            # Cell type annotation
            "annotation_method": "marker",  # marker, celltypist, none
            "marker_genes": None,  # Dict of cell_type: [genes]
            "celltypist_model": "Immune_All_Low.pkl",

            # DEG analysis
            "deg_method": "wilcoxon",  # wilcoxon, t-test, logreg
            "deg_n_genes": 100,  # Top N DEGs per cluster

            # Output
            "save_raw": True,
            "output_format": "h5ad",

            # Cancer context
            "cancer_type": "unknown",
            "tissue_type": "unknown",
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent_singlecell", input_dir, output_dir, merged_config)

        self.adata = None
        self.markers_df = None

    def validate_inputs(self) -> bool:
        """Validate input files for single-cell analysis."""
        if not HAS_SCANPY:
            self.logger.error("Scanpy not installed. Run: pip install scanpy")
            return False

        # Check for h5ad file first
        h5ad_files = list(self.input_dir.glob("*.h5ad"))
        if h5ad_files:
            self.logger.info(f"Found h5ad file: {h5ad_files[0].name}")
            self.adata = sc.read_h5ad(h5ad_files[0])
            self.logger.info(f"Loaded AnnData: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
            return True

        # Check for count matrix
        count_matrix = self.load_csv("count_matrix.csv", required=False)
        if count_matrix is None:
            # Try alternative names
            for name in ["matrix.csv", "expression.csv", "counts.csv"]:
                count_matrix = self.load_csv(name, required=False)
                if count_matrix is not None:
                    break

        if count_matrix is None:
            self.logger.error("No count matrix found (count_matrix.csv, matrix.csv, or *.h5ad)")
            return False

        # Create AnnData from count matrix
        # Assume first column is gene_id, rest are cells
        gene_ids = count_matrix.iloc[:, 0].values
        cell_ids = count_matrix.columns[1:].tolist()
        X = count_matrix.iloc[:, 1:].values.T  # Transpose: cells × genes

        self.adata = sc.AnnData(
            X=X.astype(np.float32),
            obs=pd.DataFrame(index=cell_ids),
            var=pd.DataFrame(index=gene_ids)
        )
        self.adata.var_names_make_unique()

        # Load metadata if available
        metadata = self.load_csv("metadata.csv", required=False)
        if metadata is not None:
            # Merge with adata.obs
            meta_index = metadata.iloc[:, 0].astype(str)
            metadata = metadata.set_index(metadata.columns[0])
            for col in metadata.columns:
                if col in self.adata.obs.columns:
                    continue
                self.adata.obs[col] = metadata.loc[self.adata.obs_names, col].values

        self.logger.info(f"Created AnnData: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        return True

    def _qc_filtering(self):
        """Step 1: Quality control and filtering."""
        self.logger.info("Step 1: QC & Filtering")

        # Calculate QC metrics
        # Mitochondrial genes
        self.adata.var['mt'] = self.adata.var_names.str.startswith(('MT-', 'mt-'))
        # Ribosomal genes
        self.adata.var['ribo'] = self.adata.var_names.str.startswith(('RPS', 'RPL', 'Rps', 'Rpl'))

        sc.pp.calculate_qc_metrics(
            self.adata,
            qc_vars=['mt', 'ribo'],
            percent_top=None,
            log1p=False,
            inplace=True
        )

        n_cells_before = self.adata.n_obs
        n_genes_before = self.adata.n_vars

        # Filter cells
        sc.pp.filter_cells(self.adata, min_genes=self.config["min_genes_per_cell"])

        # Filter by max genes
        self.adata = self.adata[self.adata.obs['n_genes_by_counts'] < self.config["max_genes_per_cell"], :]

        # Filter by mito percent
        self.adata = self.adata[self.adata.obs['pct_counts_mt'] < self.config["max_mito_percent"], :]

        # Filter genes
        sc.pp.filter_genes(self.adata, min_cells=self.config["min_cells_per_gene"])

        self.logger.info(f"  Cells: {n_cells_before} -> {self.adata.n_obs} ({n_cells_before - self.adata.n_obs} removed)")
        self.logger.info(f"  Genes: {n_genes_before} -> {self.adata.n_vars} ({n_genes_before - self.adata.n_vars} removed)")

    def _normalize(self):
        """Step 2: Normalization."""
        self.logger.info("Step 2: Normalization")

        # Save raw counts
        if self.config["save_raw"]:
            self.adata.raw = self.adata.copy()

        # Total count normalization
        sc.pp.normalize_total(self.adata, target_sum=self.config["target_sum"])

        # Log transform
        sc.pp.log1p(self.adata)

        self.logger.info(f"  Normalized to {self.config['target_sum']} counts per cell + log1p")

    def _select_hvg(self):
        """Step 3: Highly variable gene selection."""
        self.logger.info("Step 3: HVG Selection")

        flavor = self.config["hvg_flavor"]
        n_top = self.config["n_top_genes"]

        if flavor == "seurat_v3":
            # For seurat_v3, need to use raw counts
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top,
                flavor=flavor,
                layer=None,
                subset=False
            )
        else:
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top,
                flavor=flavor,
                subset=False
            )

        n_hvg = self.adata.var['highly_variable'].sum()
        self.logger.info(f"  Selected {n_hvg} highly variable genes")

    def _dimensionality_reduction(self):
        """Step 4: PCA and batch correction."""
        self.logger.info("Step 4: Dimensionality Reduction")

        # Scale (only on HVG for efficiency)
        sc.pp.scale(self.adata, max_value=10)

        # PCA
        n_pcs = self.config["n_pcs"]
        sc.tl.pca(self.adata, n_comps=n_pcs, use_highly_variable=True)
        self.logger.info(f"  PCA: {n_pcs} components")

        # Batch correction with Harmony
        batch_key = self.config.get("batch_key")
        if batch_key and batch_key in self.adata.obs.columns and self.config["use_harmony"]:
            if HAS_HARMONY:
                self.logger.info(f"  Applying Harmony batch correction on '{batch_key}'")
                sc.external.pp.harmony_integrate(self.adata, key=batch_key)
                self.adata.obsm['X_pca_corrected'] = self.adata.obsm['X_pca_harmony']
            else:
                self.logger.warning("  Harmony not installed, skipping batch correction")

    def _compute_neighbors_umap(self):
        """Step 5: Compute neighbors and UMAP."""
        self.logger.info("Step 5: UMAP/t-SNE")

        # Use corrected PCA if available
        use_rep = 'X_pca_harmony' if 'X_pca_harmony' in self.adata.obsm else 'X_pca'

        # Neighbors
        sc.pp.neighbors(
            self.adata,
            n_neighbors=self.config["n_neighbors"],
            n_pcs=self.config["n_pcs"],
            use_rep=use_rep
        )

        # UMAP
        sc.tl.umap(self.adata, min_dist=self.config["umap_min_dist"])
        self.logger.info(f"  UMAP computed (min_dist={self.config['umap_min_dist']})")

        # t-SNE (optional, slower)
        # sc.tl.tsne(self.adata, n_pcs=self.config["n_pcs"])

    def _clustering(self):
        """Step 6: Clustering."""
        self.logger.info("Step 6: Clustering")

        resolution = self.config["clustering_resolution"]
        method = self.config["clustering_method"]

        if method == "leiden":
            try:
                sc.tl.leiden(self.adata, resolution=resolution, key_added='cluster')
            except ImportError:
                self.logger.warning("  leidenalg not installed, falling back to louvain")
                method = "louvain"
                sc.tl.louvain(self.adata, resolution=resolution, key_added='cluster')
        else:
            sc.tl.louvain(self.adata, resolution=resolution, key_added='cluster')

        n_clusters = len(self.adata.obs['cluster'].unique())
        self.logger.info(f"  {method.capitalize()} clustering: {n_clusters} clusters (resolution={resolution})")

    def _annotate_cells(self):
        """Step 7: Cell type annotation."""
        self.logger.info("Step 7: Cell Type Annotation")

        method = self.config["annotation_method"]

        if method == "celltypist" and HAS_CELLTYPIST:
            try:
                model = celltypist.models.Model.load(model=self.config["celltypist_model"])
                predictions = celltypist.annotate(self.adata, model=model, majority_voting=True)
                self.adata.obs['cell_type'] = predictions.predicted_labels['majority_voting']
                self.logger.info(f"  CellTypist annotation complete")
            except Exception as e:
                self.logger.warning(f"  CellTypist failed: {e}. Using cluster IDs.")
                self.adata.obs['cell_type'] = 'Cluster_' + self.adata.obs['cluster'].astype(str)

        elif method == "marker" and self.config.get("marker_genes"):
            # Marker-based annotation
            marker_genes = self.config["marker_genes"]
            self._marker_based_annotation(marker_genes)

        else:
            # Default: use cluster IDs
            self.adata.obs['cell_type'] = 'Cluster_' + self.adata.obs['cluster'].astype(str)
            self.logger.info(f"  Using cluster IDs as cell type labels")

        n_types = len(self.adata.obs['cell_type'].unique())
        self.logger.info(f"  Identified {n_types} cell types")

    def _marker_based_annotation(self, marker_genes: Dict[str, List[str]]):
        """Annotate cells based on marker gene expression."""
        scores = {}
        for cell_type, genes in marker_genes.items():
            # Get genes that exist in the data
            valid_genes = [g for g in genes if g in self.adata.var_names]
            if valid_genes:
                sc.tl.score_genes(self.adata, valid_genes, score_name=f'score_{cell_type}')
                scores[cell_type] = f'score_{cell_type}'

        if scores:
            # Assign cell type based on highest score
            score_cols = list(scores.values())
            score_df = self.adata.obs[score_cols]
            self.adata.obs['cell_type'] = score_df.idxmax(axis=1).str.replace('score_', '')
        else:
            self.adata.obs['cell_type'] = 'Unknown'

    def _find_markers(self):
        """Step 8: Find marker genes (DEG per cluster)."""
        self.logger.info("Step 8: DEG Analysis (Marker Genes)")

        # Find marker genes
        sc.tl.rank_genes_groups(
            self.adata,
            groupby='cluster',
            method=self.config["deg_method"],
            n_genes=self.config["deg_n_genes"]
        )

        # Extract to DataFrame
        result = self.adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names

        markers_list = []
        for group in groups:
            for i in range(self.config["deg_n_genes"]):
                try:
                    markers_list.append({
                        'cluster': group,
                        'gene': result['names'][group][i],
                        'score': result['scores'][group][i],
                        'logfoldchange': result['logfoldchanges'][group][i],
                        'pval': result['pvals'][group][i],
                        'pval_adj': result['pvals_adj'][group][i],
                    })
                except IndexError:
                    break

        self.markers_df = pd.DataFrame(markers_list)
        self.logger.info(f"  Found {len(self.markers_df)} marker genes across {len(groups)} clusters")

    def _generate_visualizations(self):
        """Step 9: Generate visualizations."""
        self.logger.info("Step 9: Visualization")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        sc.settings.figdir = str(figures_dir)
        sc.settings.set_figure_params(dpi=150, facecolor='white')

        # UMAP by cluster
        sc.pl.umap(
            self.adata,
            color='cluster',
            title='Clusters (UMAP)',
            save='_clusters.png',
            show=False
        )
        self.logger.info("  Saved umap_clusters.png")

        # UMAP by cell type
        sc.pl.umap(
            self.adata,
            color='cell_type',
            title='Cell Types (UMAP)',
            save='_celltypes.png',
            show=False
        )
        self.logger.info("  Saved umap_celltypes.png")

        # QC metrics violin
        sc.pl.violin(
            self.adata,
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
            jitter=0.4,
            multi_panel=True,
            save='_qc.png',
            show=False
        )
        self.logger.info("  Saved violin_qc.png")

        # Top marker genes heatmap
        if self.markers_df is not None and len(self.markers_df) > 0:
            top_markers = self.markers_df.groupby('cluster').head(5)['gene'].unique().tolist()
            if len(top_markers) > 0:
                sc.pl.heatmap(
                    self.adata,
                    var_names=top_markers[:50],  # Limit to 50
                    groupby='cluster',
                    save='_markers.png',
                    show=False
                )
                self.logger.info("  Saved heatmap_markers.png")

        # Dot plot for top markers
        if self.markers_df is not None and len(self.markers_df) > 0:
            top_per_cluster = self.markers_df.groupby('cluster').head(3)['gene'].unique().tolist()
            if len(top_per_cluster) > 0:
                sc.pl.dotplot(
                    self.adata,
                    var_names=top_per_cluster[:30],
                    groupby='cluster',
                    save='_markers.png',
                    show=False
                )
                self.logger.info("  Saved dotplot_markers.png")

    def run(self) -> Dict[str, Any]:
        """Execute the full single-cell analysis pipeline."""
        # Step 1: QC & Filtering
        self._qc_filtering()

        # Step 2: Normalization
        self._normalize()

        # Step 3: HVG Selection
        self._select_hvg()

        # Step 4: Dimensionality Reduction
        self._dimensionality_reduction()

        # Step 5: UMAP
        self._compute_neighbors_umap()

        # Step 6: Clustering
        self._clustering()

        # Step 7: Cell Type Annotation
        self._annotate_cells()

        # Step 8: Marker Genes
        self._find_markers()

        # Step 9: Visualization
        self._generate_visualizations()

        # Save outputs
        self._save_outputs()

        # Compile results
        n_clusters = len(self.adata.obs['cluster'].unique())
        n_celltypes = len(self.adata.obs['cell_type'].unique())

        results = {
            "n_cells": self.adata.n_obs,
            "n_genes": self.adata.n_vars,
            "n_hvg": int(self.adata.var['highly_variable'].sum()),
            "n_clusters": n_clusters,
            "n_celltypes": n_celltypes,
            "n_markers": len(self.markers_df) if self.markers_df is not None else 0,
            "clustering_method": self.config["clustering_method"],
            "clustering_resolution": self.config["clustering_resolution"],
        }

        self.logger.info("=" * 60)
        self.logger.info("Single-Cell Analysis Complete:")
        self.logger.info(f"  Cells: {results['n_cells']}")
        self.logger.info(f"  Genes: {results['n_genes']}")
        self.logger.info(f"  HVGs: {results['n_hvg']}")
        self.logger.info(f"  Clusters: {results['n_clusters']}")
        self.logger.info(f"  Cell Types: {results['n_celltypes']}")
        self.logger.info(f"  Marker Genes: {results['n_markers']}")

        return results

    def _save_outputs(self):
        """Save all outputs."""
        self.logger.info("Saving outputs...")

        # Save h5ad
        h5ad_path = self.output_dir / "adata.h5ad"
        self.adata.write_h5ad(h5ad_path)
        self.logger.info(f"  Saved adata.h5ad")

        # Save marker genes
        if self.markers_df is not None:
            self.save_csv(self.markers_df, "cluster_markers.csv")

        # Save cell composition
        composition = self.adata.obs.groupby(['cluster', 'cell_type']).size().reset_index(name='count')
        self.save_csv(composition, "cell_composition.csv")

        # Save UMAP coordinates
        umap_df = pd.DataFrame(
            self.adata.obsm['X_umap'],
            index=self.adata.obs_names,
            columns=['UMAP1', 'UMAP2']
        )
        umap_df['cluster'] = self.adata.obs['cluster'].values
        umap_df['cell_type'] = self.adata.obs['cell_type'].values
        self.save_csv(umap_df.reset_index(), "umap_coordinates.csv")

        # Save cell metadata
        self.save_csv(self.adata.obs.reset_index(), "cell_metadata.csv")

        # Save top markers per cluster (summary)
        if self.markers_df is not None:
            top_markers = self.markers_df.groupby('cluster').head(10)
            self.save_csv(top_markers, "top_markers_summary.csv")

    def validate_outputs(self) -> bool:
        """Validate that required outputs were generated."""
        required_files = [
            "adata.h5ad",
            "cluster_markers.csv",
            "cell_composition.csv",
            "umap_coordinates.csv",
        ]

        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing output file: {filename}")
                return False

        return True
