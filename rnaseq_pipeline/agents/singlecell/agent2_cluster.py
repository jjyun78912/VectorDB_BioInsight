"""
Single-Cell Agent 2: Clustering & Cell Type Annotation

Dimensionality reduction, clustering, and cell type annotation with ML prediction.

Input:
- adata_qc.h5ad: QC-filtered data from Agent 1

Output:
- adata_clustered.h5ad: Clustered data with annotations
- cluster_markers.csv: Top marker genes per cluster
- cell_composition.csv: Cell type composition
- celltype_predictions.json: CellTypist ML predictions
- figures/umap_*.png: UMAP visualizations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

# Import harmony for batch correction
try:
    import harmonypy
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False

# Import celltypist for ML cell type prediction
try:
    import celltypist
    from celltypist import models
    HAS_CELLTYPIST = True
except ImportError:
    HAS_CELLTYPIST = False

# Import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Default cell type markers for manual annotation
DEFAULT_CELL_MARKERS = {
    # Immune cells
    "T_cells": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "CD4_T": ["CD4", "IL7R", "CCR7", "LEF1"],
    "CD8_T": ["CD8A", "CD8B", "GZMK", "GZMA"],
    "NK_cells": ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1"],
    "B_cells": ["CD79A", "CD79B", "MS4A1", "CD19"],
    "Plasma_cells": ["JCHAIN", "MZB1", "SDC1", "IGHA1"],
    "Classical_Mono": ["CD14", "LYZ", "S100A8", "S100A9"],
    "Non_classical_Mono": ["FCGR3A", "MS4A7", "CX3CR1"],
    "Dendritic_cells": ["ITGAX", "CD1C", "CLEC9A", "FCER1A"],
    "Macrophage": ["CD68", "CD163", "MSR1", "MRC1"],
    "Mast_cells": ["TPSAB1", "CPA3", "KIT"],
    "Neutrophils": ["FCGR3B", "CSF3R", "CXCR2"],

    # Non-immune cells
    "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "Smooth_muscle": ["ACTA2", "MYH11", "TAGLN"],
    "Pericyte": ["RGS5", "PDGFRB", "KCNJ8"],

    # Neural cells
    "Neurons": ["SNAP25", "SYT1", "RBFOX3", "MAP2"],
    "Astrocytes": ["GFAP", "AQP4", "S100B", "ALDH1L1"],
    "Oligodendrocytes": ["MBP", "MOG", "OLIG1", "OLIG2"],
    "Microglia": ["AIF1", "ITGAM", "CSF1R", "CX3CR1"],
}


class SingleCellClusterAgent(BaseAgent):
    """Agent 2: Clustering and Cell Type Annotation with ML."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Dimensionality reduction
            "n_neighbors": 15,
            "umap_min_dist": 0.3,
            "compute_tsne": False,

            # Batch correction
            "batch_key": None,
            "use_harmony": True,

            # Clustering
            "clustering_method": "leiden",  # leiden or louvain
            "clustering_resolution": 0.5,
            "n_clusters_min": 3,
            "n_clusters_max": 30,

            # Cell type annotation method
            "annotation_method": "hybrid",  # marker, celltypist, hybrid
            "marker_genes": None,  # Custom markers dict
            "celltypist_model": "Immune_All_Low.pkl",

            # DEG for markers
            "deg_method": "wilcoxon",
            "deg_n_genes": 100,
            "deg_min_logfc": 0.25,

            # Visualization
            "enable_plots": True,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent2_sc_cluster", input_dir, output_dir, merged_config)

        self.adata = None
        self.markers_df = None
        self.cell_composition = None
        self.celltypist_results = None

    def validate_inputs(self) -> bool:
        """Validate input files."""
        if not HAS_SCANPY:
            self.logger.error("Scanpy not installed")
            return False

        # Look for QC output
        h5ad_file = self.input_dir / "adata_qc.h5ad"
        if not h5ad_file.exists():
            # Try alternative names
            h5ad_files = list(self.input_dir.glob("*.h5ad"))
            if h5ad_files:
                h5ad_file = h5ad_files[0]
            else:
                self.logger.error("No h5ad file found from Agent 1")
                return False

        self.adata = sc.read_h5ad(h5ad_file)
        self.logger.info(f"Loaded: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")

        # Check for PCA
        if 'X_pca' not in self.adata.obsm:
            self.logger.warning("PCA not found. Will compute.")

        return True

    def run(self) -> Dict[str, Any]:
        """Execute clustering and annotation pipeline.

        Pipeline order:
        1. PCA (from Agent 1)
        2. Batch correction (Harmony on PCA)
        3. Build k-NN graph (on PCA space)
        4. Clustering (Leiden/Louvain on graph) <- PCA 기반!
        5. UMAP (시각화용, 클러스터링 이후)
        6. Marker finding
        7. Cell type annotation

        Note: Clustering is performed on PCA-based neighbor graph,
        NOT on UMAP! UMAP is only for visualization.
        """
        self.logger.info("=" * 60)
        self.logger.info("Agent 2: Clustering & Cell Type Annotation")
        self.logger.info("=" * 60)

        # Step 1: Ensure PCA exists
        self._ensure_pca()

        # Step 2: Batch correction (optional) - on PCA space
        if self.config.get("batch_key") and self.config.get("use_harmony", True):
            self._batch_correction()

        # Step 3: Build neighborhood graph (on PCA space)
        self._build_graph()

        # Step 4: Clustering (on PCA-based graph) <- UMAP 전에 수행!
        self._cluster()

        # Step 5: UMAP/t-SNE (시각화용, 클러스터링 결과 확인용)
        self._compute_embeddings()

        # Step 6: Find marker genes
        self._find_markers()

        # Step 7: Cell type annotation (ML + marker-based hybrid)
        self._annotate_cell_types()

        # Step 8: Generate visualizations
        if self.config.get("enable_plots", True):
            self._generate_plots()

        # Save outputs
        self._save_outputs()

        return {
            "status": "success",
            "n_cells": self.adata.n_obs,
            "n_clusters": len(self.adata.obs['cluster'].unique()),
            "n_cell_types": len(self.adata.obs['cell_type'].unique()) if 'cell_type' in self.adata.obs else 0,
            "markers_per_cluster": len(self.markers_df) if self.markers_df is not None else 0,
            "output_file": str(self.output_dir / "adata_clustered.h5ad")
        }

    def _ensure_pca(self):
        """Ensure PCA is computed."""
        if 'X_pca' not in self.adata.obsm:
            self.logger.info("Computing PCA...")
            if 'highly_variable' in self.adata.var:
                sc.tl.pca(self.adata, n_comps=50, use_highly_variable=True)
            else:
                sc.tl.pca(self.adata, n_comps=50)

    def _batch_correction(self):
        """Apply Harmony batch correction."""
        batch_key = self.config.get("batch_key")

        if batch_key not in self.adata.obs.columns:
            self.logger.warning(f"Batch key '{batch_key}' not found. Skipping correction.")
            return

        if not HAS_HARMONY:
            self.logger.warning("harmonypy not installed. Skipping batch correction.")
            return

        self.logger.info(f"Applying Harmony batch correction on '{batch_key}'...")

        try:
            # Run Harmony
            ho = harmonypy.run_harmony(
                self.adata.obsm['X_pca'],
                self.adata.obs,
                batch_key,
                max_iter_harmony=20
            )

            self.adata.obsm['X_pca_harmony'] = ho.Z_corr.T
            self.adata.obsm['X_pca_original'] = self.adata.obsm['X_pca'].copy()
            self.adata.obsm['X_pca'] = self.adata.obsm['X_pca_harmony']

            self.logger.info("  Harmony correction complete")

        except Exception as e:
            self.logger.warning(f"Harmony failed: {e}")

    def _build_graph(self):
        """Build k-NN graph on PCA space.

        This is the basis for clustering (Leiden/Louvain).
        Clustering operates on this graph, NOT on UMAP.
        """
        self.logger.info("Building neighborhood graph (on PCA space)...")

        sc.pp.neighbors(
            self.adata,
            n_neighbors=self.config["n_neighbors"],
            n_pcs=min(50, self.adata.obsm['X_pca'].shape[1])
        )

    def _compute_embeddings(self):
        """Compute UMAP and optionally t-SNE for visualization.

        IMPORTANT: UMAP is computed AFTER clustering.
        UMAP is only for visualization, not for clustering.
        Clustering is based on PCA neighbor graph.
        """
        self.logger.info("Computing UMAP (for visualization)...")

        sc.tl.umap(self.adata, min_dist=self.config["umap_min_dist"])

        if self.config.get("compute_tsne", False):
            self.logger.info("Computing t-SNE...")
            sc.tl.tsne(self.adata)

    def _cluster(self):
        """Perform clustering on PCA-based neighbor graph.

        Leiden/Louvain clustering operates on the k-NN graph
        built from PCA space, NOT from UMAP.
        """
        method = self.config.get("clustering_method", "leiden")
        resolution = self.config.get("clustering_resolution", 0.5)

        self.logger.info(f"Clustering on PCA graph ({method}, resolution={resolution})...")

        if method == "leiden":
            sc.tl.leiden(self.adata, resolution=resolution, key_added='cluster')
        else:
            sc.tl.louvain(self.adata, resolution=resolution, key_added='cluster')

        n_clusters = len(self.adata.obs['cluster'].unique())
        self.logger.info(f"  Found {n_clusters} clusters")

        # Cluster sizes
        cluster_sizes = self.adata.obs['cluster'].value_counts().sort_index()
        for c, n in cluster_sizes.items():
            self.logger.info(f"    Cluster {c}: {n} cells")

    def _find_markers(self):
        """Find marker genes for each cluster."""
        self.logger.info("Finding marker genes...")

        try:
            sc.tl.rank_genes_groups(
                self.adata,
                groupby='cluster',
                method=self.config.get("deg_method", "wilcoxon"),
                n_genes=self.config.get("deg_n_genes", 100)
            )

            # Extract markers to DataFrame
            result = self.adata.uns['rank_genes_groups']
            groups = result['names'].dtype.names

            markers_list = []
            for group in groups:
                for i in range(min(self.config.get("deg_n_genes", 100), len(result['names'][group]))):
                    markers_list.append({
                        'cluster': group,
                        'gene': result['names'][group][i],
                        'score': float(result['scores'][group][i]),
                        'logfoldchange': float(result['logfoldchanges'][group][i]),
                        'pval': float(result['pvals'][group][i]),
                        'pval_adj': float(result['pvals_adj'][group][i])
                    })

            self.markers_df = pd.DataFrame(markers_list)

            # Filter by logFC
            min_logfc = self.config.get("deg_min_logfc", 0.25)
            self.markers_df = self.markers_df[
                self.markers_df['logfoldchange'].abs() >= min_logfc
            ]

            self.logger.info(f"  Found {len(self.markers_df)} marker genes")

        except Exception as e:
            self.logger.warning(f"Marker finding failed: {e}")
            self.markers_df = pd.DataFrame()

    def _annotate_cell_types(self):
        """Annotate cell types using hybrid approach."""
        method = self.config.get("annotation_method", "hybrid")
        self.logger.info(f"Annotating cell types (method={method})...")

        # Initialize with cluster as cell type
        self.adata.obs['cell_type'] = 'Unknown_' + self.adata.obs['cluster'].astype(str)

        # 1. CellTypist ML prediction
        if method in ["celltypist", "hybrid"] and HAS_CELLTYPIST:
            self._run_celltypist()

        # 2. Marker-based annotation
        if method in ["marker", "hybrid"]:
            self._run_marker_annotation()

        # 3. Final hybrid decision
        if method == "hybrid":
            self._hybrid_annotation()

        # Generate cell composition
        self._calculate_composition()

    def _run_celltypist(self):
        """Run CellTypist ML cell type prediction."""
        self.logger.info("Running CellTypist ML prediction...")

        try:
            # Download model if needed
            model_name = self.config.get("celltypist_model", "Immune_All_Low.pkl")

            try:
                model = models.Model.load(model_name)
            except:
                self.logger.info(f"  Downloading model: {model_name}")
                models.download_models(model=model_name)
                model = models.Model.load(model_name)

            # Prepare data (CellTypist expects log-normalized data)
            adata_for_ct = self.adata.copy()
            if 'counts' in adata_for_ct.layers:
                adata_for_ct.X = adata_for_ct.layers['counts'].copy()
                sc.pp.normalize_total(adata_for_ct, target_sum=10000)
                sc.pp.log1p(adata_for_ct)

            # Run prediction
            predictions = celltypist.annotate(
                adata_for_ct,
                model=model,
                majority_voting=True
            )

            # Store results
            self.adata.obs['celltypist_prediction'] = predictions.predicted_labels['predicted_labels']
            self.adata.obs['celltypist_confidence'] = predictions.probability_matrix.max(axis=1)

            if hasattr(predictions, 'majority_voting'):
                self.adata.obs['celltypist_majority'] = predictions.predicted_labels.get(
                    'majority_voting', predictions.predicted_labels['predicted_labels']
                )

            self.celltypist_results = {
                "model": model_name,
                "n_cell_types": len(self.adata.obs['celltypist_prediction'].unique()),
                "cell_types": self.adata.obs['celltypist_prediction'].value_counts().to_dict()
            }

            self.logger.info(f"  CellTypist found {self.celltypist_results['n_cell_types']} cell types")

        except Exception as e:
            self.logger.warning(f"CellTypist failed: {e}")
            self.celltypist_results = None

    def _run_marker_annotation(self):
        """Run marker-based cell type annotation."""
        self.logger.info("Running marker-based annotation...")

        # Get markers
        markers = self.config.get("marker_genes") or DEFAULT_CELL_MARKERS

        cluster_annotations = {}

        for cluster in self.adata.obs['cluster'].unique():
            # Get top markers for this cluster
            if self.markers_df is not None and len(self.markers_df) > 0:
                cluster_markers = self.markers_df[
                    self.markers_df['cluster'] == cluster
                ].head(50)['gene'].tolist()
            else:
                cluster_markers = []

            # Score each cell type
            best_type = "Unknown"
            best_score = 0

            for cell_type, type_markers in markers.items():
                # Count marker overlap
                overlap = len(set(cluster_markers) & set(type_markers))
                score = overlap / len(type_markers) if type_markers else 0

                if score > best_score:
                    best_score = score
                    best_type = cell_type

            if best_score >= 0.1:  # Minimum threshold
                cluster_annotations[cluster] = best_type
            else:
                cluster_annotations[cluster] = f"Unknown_{cluster}"

        # Apply annotations
        self.adata.obs['marker_annotation'] = self.adata.obs['cluster'].map(cluster_annotations)

        self.logger.info(f"  Marker-based: {len(set(cluster_annotations.values()))} cell types")

    def _hybrid_annotation(self):
        """Combine CellTypist and marker-based annotations."""
        self.logger.info("Combining annotations (hybrid)...")

        # Priority: CellTypist with high confidence > Marker-based > Unknown

        def get_final_type(row):
            # If CellTypist has high confidence, use it
            if 'celltypist_prediction' in row.index and 'celltypist_confidence' in row.index:
                if row['celltypist_confidence'] >= 0.7:
                    return row['celltypist_prediction']

            # Otherwise use marker-based
            if 'marker_annotation' in row.index:
                if not str(row['marker_annotation']).startswith('Unknown'):
                    return row['marker_annotation']

            # Fall back to CellTypist even with low confidence
            if 'celltypist_prediction' in row.index:
                return row['celltypist_prediction']

            return row.get('marker_annotation', f"Unknown_{row['cluster']}")

        self.adata.obs['cell_type'] = self.adata.obs.apply(get_final_type, axis=1)

        n_types = len(self.adata.obs['cell_type'].unique())
        self.logger.info(f"  Final: {n_types} cell types")

    def _calculate_composition(self):
        """Calculate cell type composition per cluster."""
        composition_data = []

        for cluster in self.adata.obs['cluster'].unique():
            cluster_cells = self.adata.obs[self.adata.obs['cluster'] == cluster]
            cell_type_counts = cluster_cells['cell_type'].value_counts()

            for cell_type, count in cell_type_counts.items():
                composition_data.append({
                    'cluster': cluster,
                    'cell_type': cell_type,
                    'count': count
                })

        self.cell_composition = pd.DataFrame(composition_data)

    def _generate_plots(self):
        """Generate visualization plots."""
        if not HAS_MATPLOTLIB:
            return

        self.logger.info("Generating plots...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # UMAP by cluster
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(self.adata, color='cluster', ax=ax, show=False, legend_loc='on data')
            plt.tight_layout()
            plt.savefig(figures_dir / "umap_clusters.png", dpi=150, bbox_inches='tight')
            plt.close()

            # UMAP by cell type
            if 'cell_type' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(12, 8))
                sc.pl.umap(self.adata, color='cell_type', ax=ax, show=False)
                plt.tight_layout()
                plt.savefig(figures_dir / "umap_celltypes.png", dpi=150, bbox_inches='tight')
                plt.close()

            # UMAP by CellTypist
            if 'celltypist_prediction' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(12, 8))
                sc.pl.umap(self.adata, color='celltypist_prediction', ax=ax, show=False)
                plt.tight_layout()
                plt.savefig(figures_dir / "umap_celltypist.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Top markers dotplot
            if self.markers_df is not None and len(self.markers_df) > 0:
                top_markers = self.markers_df.groupby('cluster').head(3)['gene'].unique().tolist()[:30]
                if top_markers:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    sc.pl.dotplot(self.adata, top_markers, groupby='cluster', ax=ax, show=False)
                    plt.tight_layout()
                    plt.savefig(figures_dir / "dotplot_markers.png", dpi=150, bbox_inches='tight')
                    plt.close()

            self.logger.info(f"  Saved plots to {figures_dir}")

        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")

    def _save_outputs(self):
        """Save output files."""
        self.logger.info("Saving outputs...")

        # Save h5ad
        output_h5ad = self.output_dir / "adata_clustered.h5ad"
        self.adata.write_h5ad(output_h5ad)
        self.logger.info(f"  Saved: {output_h5ad}")

        # Save markers
        if self.markers_df is not None and len(self.markers_df) > 0:
            markers_csv = self.output_dir / "cluster_markers.csv"
            self.markers_df.to_csv(markers_csv, index=False)
            self.logger.info(f"  Saved: {markers_csv}")

            # Top markers summary
            top_markers = self.markers_df.groupby('cluster').head(10)
            top_csv = self.output_dir / "top_markers_summary.csv"
            top_markers.to_csv(top_csv, index=False)

        # Save cell composition
        if self.cell_composition is not None:
            comp_csv = self.output_dir / "cell_composition.csv"
            self.cell_composition.to_csv(comp_csv, index=False)
            self.logger.info(f"  Saved: {comp_csv}")

        # Save CellTypist results
        if self.celltypist_results:
            ct_json = self.output_dir / "celltype_predictions.json"
            with open(ct_json, 'w') as f:
                json.dump(self.celltypist_results, f, indent=2)
            self.logger.info(f"  Saved: {ct_json}")

        # Save UMAP coordinates
        umap_df = pd.DataFrame(
            self.adata.obsm['X_umap'],
            index=self.adata.obs_names,
            columns=['UMAP1', 'UMAP2']
        )
        umap_df['cluster'] = self.adata.obs['cluster'].values
        if 'cell_type' in self.adata.obs:
            umap_df['cell_type'] = self.adata.obs['cell_type'].values
        umap_df.to_csv(self.output_dir / "umap_coordinates.csv")

        self.logger.info("Clustering complete!")

    def validate_outputs(self) -> bool:
        """Validate output files were generated correctly."""
        required_files = [
            self.output_dir / "adata_clustered.h5ad"
        ]
        for f in required_files:
            if not f.exists():
                self.logger.error(f"Required output missing: {f}")
                return False
        return True
