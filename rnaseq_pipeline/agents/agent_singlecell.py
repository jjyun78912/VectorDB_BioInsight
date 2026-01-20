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
- pseudobulk_prediction.json: Cancer type prediction from pseudo-bulk
- driver_genes.csv: Marker genes matched to COSMIC/OncoKB drivers
- cluster_pathways.csv: Pathway enrichment per cluster
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
10. Pseudo-bulk Cancer Prediction (NEW)
11. Driver Gene Matching (NEW)
12. Cluster Pathway Enrichment (NEW)
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

# Import gseapy for pathway enrichment
try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

# Import ML predictor
try:
    from ..ml.pancancer_classifier import PanCancerClassifier
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False

# Import liana for cell-cell interaction (optional)
try:
    import liana as li
    HAS_LIANA = True
except ImportError:
    HAS_LIANA = False

# Import decoupler for pathway activity (optional)
try:
    import decoupler as dc
    HAS_DECOUPLER = True
except ImportError:
    HAS_DECOUPLER = False


# ═══════════════════════════════════════════════════════════════════════════════
# Known Cancer Driver Gene Databases (for marker matching)
# ═══════════════════════════════════════════════════════════════════════════════

COSMIC_TIER1_GENES = {
    'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'RB1',
    'BRCA1', 'BRCA2', 'MYC', 'ERBB2', 'CDK4', 'MDM2', 'CCND1', 'CDKN2A',
    'ATM', 'AKT1', 'NRAS', 'HRAS', 'FGFR1', 'FGFR2', 'FGFR3', 'MET',
    'ALK', 'ROS1', 'RET', 'KIT', 'PDGFRA', 'ABL1', 'JAK2', 'BCL2',
    'VHL', 'NF1', 'NF2', 'WT1', 'SMAD4', 'CTNNB1', 'IDH1', 'IDH2',
    'ARID1A', 'ARID2', 'TERT', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'FLT3',
    'NPM1', 'DNMT3A', 'TET2', 'ASXL1', 'EZH2', 'KMT2D', 'KMT2C',
    'CREBBP', 'EP300', 'SETD2', 'BAP1', 'STAG2', 'SF3B1', 'U2AF1',
    'SRSF2', 'ZRSR2', 'PHF6', 'WT1', 'RUNX1', 'CEBPA', 'GATA1', 'GATA2'
}

ONCOKB_ACTIONABLE_GENES = {
    'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'ERBB2', 'MYC',
    'BRCA1', 'BRCA2', 'ALK', 'ROS1', 'RET', 'MET', 'NTRK1', 'NTRK2',
    'NTRK3', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'CDKN2A', 'AKT1',
    'MTOR', 'STK11', 'ESR1', 'AR', 'BTK', 'BCL2', 'IDH1', 'IDH2',
    'KIT', 'PDGFRA', 'FLT3', 'JAK2', 'ABL1', 'ERBB3', 'RAF1', 'MAP2K1'
}

# Tumor Microenvironment markers
TME_GENES = {
    'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT',  # Immune checkpoints
    'CD8A', 'CD8B', 'CD4', 'FOXP3', 'CD68', 'CD163', 'CD14',  # Immune markers
    'VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR',  # Angiogenesis
    'TGFB1', 'TGFB2', 'IL6', 'IL10', 'CXCL8', 'CCL2',  # Cytokines
    'COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'FAP', 'ACTA2'  # Stromal markers
}


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

            # NEW: Enhanced analysis options
            "enable_cancer_prediction": True,  # Pseudo-bulk → ML prediction
            "enable_driver_matching": True,    # Marker → COSMIC/OncoKB matching
            "enable_pathway_analysis": True,   # Per-cluster pathway enrichment
            "pathway_databases": ["GO_Biological_Process_2023", "KEGG_2021_Human"],
            "pathway_top_genes": 100,          # Top N markers per cluster for pathway
            "model_dir": None,                 # Path to pre-trained ML model

            # NEW: Advanced analysis options
            "enable_trajectory": True,         # Trajectory/Pseudotime analysis
            "trajectory_root": None,           # Root cluster for trajectory (auto-detect if None)
            "enable_cell_interaction": True,   # Cell-cell interaction analysis
            "interaction_method": "cellphonedb",  # cellphonedb, natmi, connectome, etc.
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent_singlecell", input_dir, output_dir, merged_config)

        self.adata = None
        self.markers_df = None
        self.driver_genes_df = None
        self.cluster_pathways_df = None
        self.cancer_prediction = None
        self.trajectory_results = None
        self.interaction_results = None
        self._is_preprocessed = False  # Flag to skip QC/norm if h5ad already processed

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

            # Check if already preprocessed (has cluster, UMAP, HVG etc.)
            if 'cluster' in self.adata.obs.columns and 'X_umap' in self.adata.obsm:
                self._is_preprocessed = True
                self.logger.info("  → Detected pre-processed h5ad (skipping QC/Norm/UMAP)")
                # Ensure required columns exist
                if 'highly_variable' not in self.adata.var.columns:
                    self.adata.var['highly_variable'] = True
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
                model_name = self.config.get("celltypist_model", "Immune_All_Low.pkl")
                self.logger.info(f"  Loading CellTypist model: {model_name}")
                model = celltypist.models.Model.load(model=model_name)
                predictions = celltypist.annotate(self.adata, model=model, majority_voting=True)
                self.adata.obs['cell_type'] = predictions.predicted_labels['majority_voting']
                self.adata.obs['cell_type_confidence'] = predictions.probability_matrix.max(axis=1).values
                self.logger.info(f"  CellTypist annotation complete")
            except Exception as e:
                self.logger.warning(f"  CellTypist failed: {e}. Falling back to marker-based.")
                self._annotate_clusters_by_markers()

        elif method == "marker" and self.config.get("marker_genes"):
            # User-provided marker-based annotation
            marker_genes = self.config["marker_genes"]
            self._marker_based_annotation(marker_genes)

        elif method == "auto" or method == "marker":
            # Auto marker-based annotation using cluster-level marker analysis
            self._annotate_clusters_by_markers()

        else:
            # Default: use cluster IDs
            self.adata.obs['cell_type'] = 'Cluster_' + self.adata.obs['cluster'].astype(str)
            self.logger.info(f"  Using cluster IDs as cell type labels")

        n_types = len(self.adata.obs['cell_type'].unique())
        self.logger.info(f"  Identified {n_types} cell types")

    def _auto_marker_annotation(self):
        """Automatic cell type annotation using canonical markers."""
        self.logger.info("  Running auto marker-based annotation...")

        # Canonical marker genes for common cell types
        canonical_markers = {
            # Immune cells
            'T cells': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B'],
            'CD4+ T cells': ['CD4', 'IL7R', 'CCR7', 'CD3D'],
            'CD8+ T cells': ['CD8A', 'CD8B', 'GZMK', 'GZMB', 'CD3D'],
            'NK cells': ['GNLY', 'NKG7', 'KLRD1', 'KLRF1', 'NCR1'],
            'B cells': ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'PAX5'],
            'Plasma cells': ['JCHAIN', 'MZB1', 'SDC1', 'IGHG1'],
            'Monocytes': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCN1'],
            'Macrophages': ['CD68', 'CD163', 'MARCO', 'MSR1', 'MRC1'],
            'Dendritic cells': ['FCER1A', 'CD1C', 'CLEC10A', 'ITGAX'],
            'Mast cells': ['TPSAB1', 'TPSB2', 'CPA3', 'KIT'],
            'Neutrophils': ['FCGR3B', 'CSF3R', 'S100A8', 'S100A9'],

            # Stromal cells
            'Fibroblasts': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM'],
            'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'ERG', 'FLT1'],
            'Smooth muscle': ['ACTA2', 'MYH11', 'TAGLN', 'CNN1'],
            'Pericytes': ['RGS5', 'PDGFRB', 'CSPG4', 'NOTCH3'],

            # Epithelial cells (organ-specific)
            'Hepatocytes': ['ALB', 'APOB', 'SERPINA1', 'HP', 'TF'],
            'Cholangiocytes': ['KRT19', 'KRT7', 'EPCAM', 'SOX9'],
            'Enterocytes': ['FABP2', 'APOA1', 'APOC3', 'RBP2'],
            'Pneumocytes': ['SFTPC', 'SFTPB', 'AGER', 'AQP5'],
            'Keratinocytes': ['KRT1', 'KRT5', 'KRT14', 'KRT10'],

            # Cancer-related
            'Tumor cells': ['EPCAM', 'KRT8', 'KRT18', 'MKI67', 'TOP2A'],
            'CAFs': ['FAP', 'PDPN', 'ACTA2', 'COL1A1', 'COL3A1'],  # Cancer-associated fibroblasts
        }

        # Score each cell type
        scores = {}
        for cell_type, genes in canonical_markers.items():
            valid_genes = [g for g in genes if g in self.adata.var_names]
            if len(valid_genes) >= 2:  # Need at least 2 markers
                try:
                    sc.tl.score_genes(self.adata, valid_genes, score_name=f'score_{cell_type}')
                    scores[cell_type] = f'score_{cell_type}'
                except Exception as e:
                    self.logger.debug(f"    Failed to score {cell_type}: {e}")

        if scores:
            # Get best cell type per cell
            score_cols = list(scores.values())
            score_df = self.adata.obs[score_cols]

            # Assign based on highest score
            best_type = score_df.idxmax(axis=1).str.replace('score_', '')
            best_score = score_df.max(axis=1)

            # Mark low-confidence as "Unknown"
            threshold = 0.1  # Minimum score threshold
            best_type[best_score < threshold] = 'Unknown'

            self.adata.obs['cell_type'] = best_type
            self.adata.obs['cell_type_score'] = best_score

            # Log summary
            type_counts = self.adata.obs['cell_type'].value_counts()
            self.logger.info(f"  Auto-annotation found {len(type_counts)} cell types:")
            for ct, cnt in type_counts.head(10).items():
                self.logger.info(f"    {ct}: {cnt} cells ({cnt/len(self.adata)*100:.1f}%)")
        else:
            self.logger.warning("  No valid markers found, using cluster IDs")
            self.adata.obs['cell_type'] = 'Cluster_' + self.adata.obs['cluster'].astype(str)

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

    def _annotate_clusters_by_markers(self):
        """
        Annotate clusters by their top marker genes.

        This method:
        1. Finds top markers for each cluster
        2. Matches markers to canonical cell type signatures
        3. Assigns cell type names to clusters
        4. Updates cell_type column with meaningful names
        """
        self.logger.info("  Annotating clusters by marker genes...")

        # Extended canonical markers with priority weights
        canonical_signatures = {
            # Immune cells - High specificity markers first
            'T_cells': {
                'markers': ['CD3D', 'CD3E', 'CD3G', 'TRAC', 'CD2'],
                'subtype_markers': {
                    'CD4_T': ['CD4', 'IL7R', 'CCR7', 'LEF1'],
                    'CD8_T': ['CD8A', 'CD8B', 'GZMK', 'GZMB', 'PRF1'],
                    'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2'],
                    'NKT': ['KLRB1', 'ZBTB16', 'CD3D'],
                }
            },
            'NK_cells': {
                'markers': ['NKG7', 'GNLY', 'KLRD1', 'KLRF1', 'NCAM1', 'NCR1'],
            },
            'B_cells': {
                'markers': ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'PAX5', 'BANK1'],
                'subtype_markers': {
                    'Plasma': ['JCHAIN', 'MZB1', 'SDC1', 'IGHG1', 'XBP1'],
                    'Memory_B': ['CD27', 'IGHM', 'IGHD'],
                }
            },
            'Monocytes': {
                'markers': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCN1', 'VCAN'],
                'subtype_markers': {
                    'Classical_Mono': ['CD14', 'LYZ', 'S100A8'],
                    'Non_classical_Mono': ['FCGR3A', 'CX3CR1', 'CDKN1C'],
                }
            },
            'Macrophages': {
                'markers': ['CD68', 'CD163', 'MARCO', 'MSR1', 'MRC1', 'APOE'],
                'subtype_markers': {
                    'M1_Macro': ['CD80', 'CD86', 'IL1B', 'TNF'],
                    'M2_Macro': ['CD163', 'MRC1', 'CD206', 'IL10'],
                }
            },
            'Dendritic_cells': {
                'markers': ['FCER1A', 'CD1C', 'CLEC10A', 'ITGAX', 'HLA-DRA'],
                'subtype_markers': {
                    'cDC1': ['CLEC9A', 'XCR1', 'BATF3'],
                    'cDC2': ['CD1C', 'FCER1A', 'CLEC10A'],
                    'pDC': ['LILRA4', 'IL3RA', 'CLEC4C'],
                }
            },
            'Mast_cells': {
                'markers': ['TPSAB1', 'TPSB2', 'CPA3', 'KIT', 'MS4A2'],
            },
            'Neutrophils': {
                'markers': ['FCGR3B', 'CSF3R', 'CXCR2', 'S100A8', 'S100A9'],
            },

            # Stromal cells
            'Fibroblasts': {
                'markers': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'PDGFRA'],
                'subtype_markers': {
                    'CAF': ['FAP', 'ACTA2', 'PDPN', 'POSTN'],
                    'Myofibroblast': ['ACTA2', 'MYH11', 'TAGLN'],
                }
            },
            'Endothelial': {
                'markers': ['PECAM1', 'VWF', 'CDH5', 'ERG', 'FLT1', 'KDR'],
                'subtype_markers': {
                    'Lymphatic_EC': ['LYVE1', 'PROX1', 'PDPN'],
                    'Vascular_EC': ['VWF', 'ACKR1', 'PLVAP'],
                }
            },
            'Pericytes': {
                'markers': ['RGS5', 'PDGFRB', 'CSPG4', 'NOTCH3', 'ACTA2'],
            },
            'Smooth_muscle': {
                'markers': ['ACTA2', 'MYH11', 'TAGLN', 'CNN1', 'DES'],
            },

            # =============================================
            # LIVER (LIHC)
            # =============================================
            'Hepatocytes': {
                'markers': ['ALB', 'APOB', 'SERPINA1', 'HP', 'TF', 'ASGR1', 'CYP3A4',
                           'TTR', 'TAT', 'ADH1B', 'RBP4', 'CYP2C9', 'CYP2E1', 'CYP2A6',
                           'CYP2B6', 'CFHR3', 'CFHR1', 'A2M', 'CFB', 'IGFBP2', 'LEAP2',
                           'DDT', 'ANG', 'GATM', 'PCK1', 'G6PC', 'FGB', 'FGA'],
            },
            'Cholangiocytes': {
                'markers': ['KRT19', 'KRT7', 'EPCAM', 'SOX9', 'CFTR', 'MUC1', 'SPP1'],
            },
            'LSECs': {  # Liver sinusoidal endothelial cells
                'markers': ['CLEC4G', 'CLEC4M', 'STAB2', 'LYVE1', 'FCN3', 'CLEC1B',
                           'CLDN5', 'GNG11', 'IGFBP7', 'ID1', 'ID3', 'TIMP1', 'TM4SF1',
                           'CCL14', 'LIFR', 'HSPG2', 'PLVAP', 'ACKR1'],
            },
            'Kupffer_cells': {
                'markers': ['MARCO', 'CD163', 'TIMD4', 'CLEC4F', 'CD68', 'VSIG4', 'C1QA', 'C1QB'],
            },
            'Stellate_cells': {
                'markers': ['LRAT', 'RGS5', 'ACTA2', 'TAGLN', 'COL1A1', 'PDGFRB', 'DES'],
            },

            # =============================================
            # BREAST (BRCA)
            # =============================================
            'Luminal_epithelial': {
                'markers': ['KRT8', 'KRT18', 'KRT19', 'EPCAM', 'ESR1', 'PGR', 'GATA3', 'FOXA1', 'MUC1'],
            },
            'Basal_epithelial': {
                'markers': ['KRT5', 'KRT14', 'KRT17', 'TP63', 'EGFR', 'ACTA2'],
            },
            'Myoepithelial': {
                'markers': ['ACTA2', 'MYH11', 'KRT5', 'KRT14', 'TAGLN', 'CNN1', 'TP63'],
            },

            # =============================================
            # LUNG (LUAD/LUSC)
            # =============================================
            'Alveolar_type1': {  # AT1
                'markers': ['AGER', 'PDPN', 'CAV1', 'AQP5', 'CLDN18', 'HOPX'],
            },
            'Alveolar_type2': {  # AT2
                'markers': ['SFTPC', 'SFTPB', 'SFTPA1', 'SFTPA2', 'SFTPD', 'ABCA3', 'NAPSA', 'NKX2-1'],
            },
            'Club_cells': {
                'markers': ['SCGB1A1', 'SCGB3A1', 'SCGB3A2', 'CYP2F1', 'BPIFB1'],
            },
            'Ciliated_cells': {
                'markers': ['FOXJ1', 'TPPP3', 'PIFO', 'DNAH5', 'DNAI1', 'RSPH1', 'CFAP299'],
            },
            'Goblet_cells': {
                'markers': ['MUC5AC', 'MUC5B', 'SPDEF', 'TFF3', 'AGR2'],
            },
            'Basal_airway': {
                'markers': ['KRT5', 'KRT17', 'TP63', 'NGFR', 'KRT14'],
            },

            # =============================================
            # COLON/COLORECTAL (COAD)
            # =============================================
            'Colonocytes': {
                'markers': ['CA1', 'CA2', 'FABP1', 'CEACAM7', 'AQP8', 'SLC26A3', 'GUCA2A'],
            },
            'Goblet_intestinal': {
                'markers': ['MUC2', 'TFF3', 'SPDEF', 'FCGBP', 'ZG16', 'CLCA1'],
            },
            'Enterocytes': {
                'markers': ['FABP2', 'FABP1', 'SLC5A1', 'ANPEP', 'SI', 'APOA1', 'APOA4'],
            },
            'Paneth_cells': {
                'markers': ['DEFA5', 'DEFA6', 'LYZ', 'REG3A', 'REG3G'],
            },
            'Enteroendocrine': {
                'markers': ['CHGA', 'CHGB', 'SYP', 'NEUROD1', 'GCG', 'PYY'],
            },
            'Stem_intestinal': {
                'markers': ['LGR5', 'ASCL2', 'OLFM4', 'SOX9', 'RGMB'],
            },

            # =============================================
            # STOMACH (STAD)
            # =============================================
            'Chief_cells': {
                'markers': ['PGA3', 'PGA4', 'PGA5', 'PGC', 'LIPF', 'GKN2'],
            },
            'Parietal_cells': {
                'markers': ['ATP4A', 'ATP4B', 'GIF', 'KCNQ1'],
            },
            'Pit_cells': {  # Foveolar/mucous cells
                'markers': ['MUC5AC', 'TFF1', 'TFF2', 'GKN1', 'MUC6'],
            },

            # =============================================
            # KIDNEY (KIRC)
            # =============================================
            'Proximal_tubule': {
                'markers': ['SLC34A1', 'LRP2', 'CUBN', 'SLC5A12', 'ALDOB', 'GPX3'],
            },
            'Distal_tubule': {
                'markers': ['SLC12A3', 'CALB1', 'TRPM6'],
            },
            'Collecting_duct': {
                'markers': ['AQP2', 'AQP3', 'HSD11B2', 'SCNN1A', 'SCNN1B', 'SCNN1G'],
                'subtype_markers': {
                    'Principal_cell': ['AQP2', 'AQP3', 'FXYD4'],
                    'Intercalated_A': ['SLC4A1', 'ATP6V1G3', 'ATP6V0D2'],
                    'Intercalated_B': ['SLC26A4', 'SLC4A9'],
                }
            },
            'Loop_of_Henle': {
                'markers': ['SLC12A1', 'UMOD', 'CLDN16', 'CLDN19'],
            },
            'Podocytes': {
                'markers': ['NPHS1', 'NPHS2', 'WT1', 'PODXL', 'SYNPO'],
            },
            'Mesangial_cells': {
                'markers': ['PDGFRB', 'DES', 'ACTA2', 'TAGLN', 'ITGA8'],
            },

            # =============================================
            # PANCREAS (PAAD)
            # =============================================
            'Acinar_cells': {
                'markers': ['PRSS1', 'PRSS2', 'CPA1', 'CPA2', 'CELA3A', 'CELA3B', 'PNLIP', 'AMY2A'],
            },
            'Ductal_cells': {
                'markers': ['KRT19', 'KRT7', 'CFTR', 'MUC1', 'SOX9', 'SPP1', 'CA2'],
            },
            'Alpha_cells': {
                'markers': ['GCG', 'ARX', 'IRX2', 'TTR'],
            },
            'Beta_cells': {
                'markers': ['INS', 'IAPP', 'MAFA', 'NKX6-1', 'PDX1', 'SLC2A2'],
            },
            'Delta_cells': {
                'markers': ['SST', 'RBP4', 'HHEX'],
            },
            'PP_cells': {
                'markers': ['PPY', 'SERTM1'],
            },

            # =============================================
            # PROSTATE (PRAD)
            # =============================================
            'Luminal_prostate': {
                'markers': ['KRT8', 'KRT18', 'AR', 'NKX3-1', 'KLK3', 'ACPP', 'MSMB'],
            },
            'Basal_prostate': {
                'markers': ['KRT5', 'KRT14', 'TP63', 'KRT15'],
            },
            'Neuroendocrine_prostate': {
                'markers': ['CHGA', 'SYP', 'NCAM1', 'ENO2'],
            },

            # =============================================
            # OVARY (OV)
            # =============================================
            'Ovarian_epithelial': {
                'markers': ['PAX8', 'WT1', 'KRT7', 'MUC16', 'EPCAM', 'CA125'],
            },
            'Granulosa_cells': {
                'markers': ['FSHR', 'CYP19A1', 'FOXL2', 'AMH', 'INHA', 'INHBA'],
            },
            'Theca_cells': {
                'markers': ['CYP17A1', 'STAR', 'LHCGR'],
            },

            # =============================================
            # UTERUS (UCEC)
            # =============================================
            'Endometrial_epithelial': {
                'markers': ['EPCAM', 'KRT8', 'KRT18', 'MUC1', 'PAX8', 'ESR1', 'PGR'],
            },
            'Endometrial_stromal': {
                'markers': ['VIM', 'CD10', 'MME', 'IGFBP1', 'PRL'],
            },

            # =============================================
            # THYROID (THCA)
            # =============================================
            'Thyrocytes': {
                'markers': ['TG', 'TPO', 'TSHR', 'NKX2-1', 'PAX8', 'SLC5A5', 'DIO1', 'DIO2'],
            },
            'Parafollicular_C_cells': {
                'markers': ['CALCA', 'CHGA', 'SYP'],
            },

            # =============================================
            # BRAIN (GBM/LGG)
            # =============================================
            'Astrocytes': {
                'markers': ['GFAP', 'AQP4', 'SLC1A2', 'SLC1A3', 'ALDH1L1', 'S100B', 'GJA1'],
            },
            'Oligodendrocytes': {
                'markers': ['MBP', 'PLP1', 'MOG', 'MAG', 'OLIG1', 'OLIG2', 'SOX10', 'CNP'],
            },
            'OPC': {  # Oligodendrocyte precursor cells
                'markers': ['PDGFRA', 'CSPG4', 'OLIG1', 'OLIG2', 'SOX10'],
            },
            'Neurons': {
                'markers': ['RBFOX3', 'SYT1', 'SNAP25', 'SLC17A7', 'GAD1', 'GAD2', 'MAP2'],
            },
            'Microglia': {
                'markers': ['CX3CR1', 'P2RY12', 'TMEM119', 'AIF1', 'CSF1R', 'ITGAM'],
            },
            'Ependymal': {
                'markers': ['FOXJ1', 'PIFO', 'DNAH9', 'CCDC153'],
            },

            # =============================================
            # MELANOMA (SKCM)
            # =============================================
            'Melanocytes': {
                'markers': ['MLANA', 'PMEL', 'TYR', 'TYRP1', 'DCT', 'MITF', 'SOX10'],
            },
            'Keratinocytes': {
                'markers': ['KRT1', 'KRT5', 'KRT10', 'KRT14', 'LOR', 'IVL', 'FLG'],
            },

            # =============================================
            # HEAD & NECK (HNSC)
            # =============================================
            'Squamous_epithelial': {
                'markers': ['KRT5', 'KRT6A', 'KRT13', 'KRT14', 'TP63', 'DSG3'],
            },

            # =============================================
            # BLADDER (BLCA)
            # =============================================
            'Urothelial': {
                'markers': ['UPK1A', 'UPK1B', 'UPK2', 'UPK3A', 'KRT20', 'GATA3'],
            },
            'Umbrella_cells': {
                'markers': ['UPK1A', 'UPK2', 'UPK3A', 'KRT20'],
            },
            'Intermediate_urothelial': {
                'markers': ['KRT13', 'KRT7', 'UPK1B'],
            },
            'Basal_urothelial': {
                'markers': ['KRT5', 'KRT17', 'CD44', 'TP63'],
            },

            # =============================================
            # COMMON / GENERAL
            # =============================================
            # Plasma cells (B cell derived)
            'Plasma_cells': {
                'markers': ['MZB1', 'JCHAIN', 'IGKC', 'IGHG1', 'IGHG3', 'XBP1', 'SDC1',
                           'FKBP11', 'SEC11C', 'SSR3', 'HSP90B1', 'PRDM1', 'IRF4'],
            },

            # Erythroid cells
            'Erythrocytes': {
                'markers': ['HBB', 'HBA1', 'HBA2', 'HBD', 'ALAS2', 'SLC25A37', 'AHSP',
                           'CA1', 'BLVRB', 'SNCA', 'GYPA', 'SLC4A1', 'ANK1'],
            },

            # Generic epithelial (fallback)
            'Epithelial': {
                'markers': ['EPCAM', 'KRT8', 'KRT18', 'CDH1'],
            },

            # Proliferating cells
            'Proliferating': {
                'markers': ['MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1'],
            },

            # Cancer stem cells markers
            'Cancer_stem_cells': {
                'markers': ['CD44', 'ALDH1A1', 'PROM1', 'SOX2', 'NANOG', 'POU5F1'],
            },
        }

        # First run marker finding if not done
        if not hasattr(self.adata.uns, 'rank_genes_groups') or 'rank_genes_groups' not in self.adata.uns:
            self.logger.info("    Running marker gene analysis...")
            sc.tl.rank_genes_groups(
                self.adata,
                groupby='cluster',
                method='wilcoxon',
                n_genes=50
            )

        result = self.adata.uns['rank_genes_groups']
        clusters = result['names'].dtype.names

        cluster_to_celltype = {}
        cluster_scores = {}

        for cluster in clusters:
            # Get top 50 markers for this cluster
            top_markers = list(result['names'][cluster][:50])
            top_scores = list(result['scores'][cluster][:50])
            marker_set = set(top_markers[:30])  # Focus on top 30 for matching

            best_match = None
            best_score = 0
            subtype_match = None

            for cell_type, info in canonical_signatures.items():
                markers = info['markers'] if isinstance(info, dict) else info
                if isinstance(info, dict):
                    markers = info.get('markers', [])
                    subtypes = info.get('subtype_markers', {})
                else:
                    markers = info
                    subtypes = {}

                # Calculate overlap score
                overlap = marker_set.intersection(set(markers))
                if overlap:
                    # Weight by marker position (earlier = better)
                    score = 0
                    for marker in overlap:
                        if marker in top_markers:
                            idx = top_markers.index(marker)
                            score += (50 - idx) / 50  # Higher score for earlier markers
                        else:
                            score += 0.5

                    # Bonus for multiple markers
                    score *= (1 + 0.2 * len(overlap))

                    if score > best_score:
                        best_score = score
                        best_match = cell_type

                        # Check subtypes
                        for subtype, sub_markers in subtypes.items():
                            sub_overlap = marker_set.intersection(set(sub_markers))
                            if len(sub_overlap) >= 2:
                                subtype_match = subtype
                                break

            if best_match and best_score > 1.0:  # Minimum threshold
                if subtype_match:
                    final_type = subtype_match
                else:
                    final_type = best_match
                cluster_to_celltype[cluster] = final_type
                cluster_scores[cluster] = best_score
            else:
                cluster_to_celltype[cluster] = f'Unknown_{cluster}'
                cluster_scores[cluster] = 0

        # Handle duplicate cell types (add numbering)
        type_counts = {}
        final_mapping = {}
        for cluster in sorted(clusters):
            cell_type = cluster_to_celltype[cluster]
            if cell_type in type_counts:
                type_counts[cell_type] += 1
                final_mapping[cluster] = f'{cell_type}_{type_counts[cell_type]}'
            else:
                type_counts[cell_type] = 1
                final_mapping[cluster] = cell_type

        # Update cell_type column
        self.adata.obs['cell_type'] = self.adata.obs['cluster'].map(final_mapping)

        # Log results
        self.logger.info("    Cluster → Cell Type mapping:")
        for cluster in sorted(clusters):
            ct = final_mapping[cluster]
            score = cluster_scores[cluster]
            n_cells = (self.adata.obs['cluster'] == cluster).sum()
            self.logger.info(f"      {cluster} → {ct} (score: {score:.2f}, n={n_cells})")

        return final_mapping

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

        # UMAP by cluster (no title - will be added in HTML report)
        sc.pl.umap(
            self.adata,
            color='cluster',
            title='',
            save='_clusters.png',
            show=False
        )
        self.logger.info("  Saved umap_clusters.png")

        # UMAP by cell type (no title - will be added in HTML report)
        sc.pl.umap(
            self.adata,
            color='cell_type',
            title='',
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

        # Top marker genes - use same genes for both plots
        if self.markers_df is not None and len(self.markers_df) > 0:
            # Get top 3 markers per cluster for consistent visualization
            top_markers = self.markers_df.groupby('cluster').head(3)['gene'].unique().tolist()[:30]
            n_clusters = len(self.adata.obs['cluster'].unique())

            if len(top_markers) > 0:
                # Calculate figure height based on number of clusters (same for both plots)
                fig_height = max(8, n_clusters * 0.5)
                fig_width = 10

                # Matrixplot (cluster mean heatmap) - same Y-axis as Dotplot
                sc.pl.matrixplot(
                    self.adata,
                    var_names=top_markers,
                    groupby='cluster',
                    figsize=(fig_width, fig_height),
                    cmap='viridis',
                    standard_scale='var',  # Standardize per gene
                    colorbar_title='Mean\nexpression',
                    save='_markers.png',
                    show=False
                )
                # Scanpy saves as 'matrixplot__markers.png', rename to expected name
                old_heatmap = figures_dir / 'matrixplot__markers.png'
                new_heatmap = figures_dir / 'heatmap_markers.png'
                if old_heatmap.exists():
                    old_heatmap.rename(new_heatmap)
                self.logger.info("  Saved heatmap_markers.png (matrixplot)")

                # Dotplot with same dimensions for Y-axis alignment
                sc.pl.dotplot(
                    self.adata,
                    var_names=top_markers,
                    groupby='cluster',
                    figsize=(fig_width, fig_height),
                    save='_markers.png',
                    show=False
                )
                # Scanpy saves as 'dotplot__markers.png', rename to expected name
                old_path = figures_dir / 'dotplot__markers.png'
                new_path = figures_dir / 'dotplot_markers.png'
                if old_path.exists():
                    old_path.rename(new_path)
                self.logger.info("  Saved dotplot_markers.png")

        # Combined UMAP + Bar chart visualization
        self._generate_celltype_composition_plot(figures_dir)

    def _generate_celltype_composition_plot(self, figures_dir: Path):
        """Generate cell type composition bar chart only (UMAP is separate)."""
        import matplotlib.pyplot as plt

        try:
            # Get cell type counts and sort by frequency
            cell_type_counts = self.adata.obs['cell_type'].value_counts()
            cell_types = cell_type_counts.index.tolist()
            counts = cell_type_counts.values
            percentages = counts / counts.sum() * 100

            # Create a consistent color palette (same as UMAP for consistency)
            n_types = len(cell_types)
            cmap = plt.cm.get_cmap('tab20' if n_types <= 20 else 'tab20b')
            colors = [cmap(i % 20) for i in range(n_types)]

            # Create figure with single bar chart
            fig, ax = plt.subplots(figsize=(10, max(6, n_types * 0.4)))

            y_pos = range(len(cell_types))
            bars = ax.barh(y_pos, percentages, color=colors, edgecolor='white', linewidth=0.5)

            # Add percentage labels
            for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts)):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}% ({cnt:,})',
                    va='center',
                    ha='left',
                    fontsize=10
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(cell_types, fontsize=11)
            ax.set_xlabel('Percentage (%)', fontsize=12)
            # No title - will be added in HTML report
            ax.set_xlim(0, max(percentages) * 1.35)  # Leave room for labels
            ax.invert_yaxis()  # Largest on top
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            save_path = figures_dir / 'celltype_barchart.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            self.logger.info("  Saved celltype_barchart.png")

        except Exception as e:
            self.logger.warning(f"  Failed to generate composition plot: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 10: Pseudo-bulk Cancer Prediction
    # ═══════════════════════════════════════════════════════════════════════════════

    def _generate_pseudobulk(self) -> pd.DataFrame:
        """Generate pseudo-bulk expression by aggregating cells.

        Returns:
            DataFrame with genes as rows, clusters as columns (TPM-like values)
        """
        self.logger.info("Step 10: Generating Pseudo-bulk for Cancer Prediction")

        # Use raw counts if available, otherwise use normalized
        if self.adata.raw is not None:
            expr_matrix = self.adata.raw.X
            gene_names = self.adata.raw.var_names
        else:
            expr_matrix = self.adata.X
            gene_names = self.adata.var_names

        # Convert to dense if sparse
        if hasattr(expr_matrix, 'toarray'):
            expr_matrix = expr_matrix.toarray()

        # Aggregate by cluster (sum of counts)
        clusters = self.adata.obs['cluster'].values
        unique_clusters = np.unique(clusters)

        pseudobulk_data = {}
        for cluster in unique_clusters:
            mask = clusters == cluster
            # Sum counts across cells in this cluster
            cluster_sum = expr_matrix[mask, :].sum(axis=0)
            # Flatten if needed
            if hasattr(cluster_sum, 'A1'):
                cluster_sum = cluster_sum.A1
            pseudobulk_data[f'Cluster_{cluster}'] = cluster_sum

        pseudobulk_df = pd.DataFrame(pseudobulk_data, index=gene_names)

        # Convert to TPM-like (counts per million)
        for col in pseudobulk_df.columns:
            total = pseudobulk_df[col].sum()
            if total > 0:
                pseudobulk_df[col] = pseudobulk_df[col] / total * 1e6

        # Also create a "bulk" sample (all cells aggregated)
        all_sum = expr_matrix.sum(axis=0)
        if hasattr(all_sum, 'A1'):
            all_sum = all_sum.A1
        total = all_sum.sum()
        if total > 0:
            all_sum = all_sum / total * 1e6
        pseudobulk_df['Bulk_All'] = all_sum

        self.logger.info(f"  Generated pseudo-bulk: {len(pseudobulk_df)} genes × {len(pseudobulk_df.columns)} samples")
        return pseudobulk_df

    def _predict_cancer_type(self):
        """Predict cancer type from pseudo-bulk expression using ML model."""
        if not self.config.get("enable_cancer_prediction", True):
            self.logger.info("  Cancer prediction disabled, skipping...")
            return

        if not HAS_ML_PREDICTOR:
            self.logger.warning("  ML predictor not available, skipping cancer prediction")
            return

        try:
            # Generate pseudo-bulk
            pseudobulk_df = self._generate_pseudobulk()

            # Find model directory
            model_dir = self.config.get("model_dir")
            if model_dir is None:
                # Try default location
                default_model = Path(__file__).parent.parent.parent / "models" / "rnaseq" / "pancancer"
                if default_model.exists():
                    model_dir = default_model
                else:
                    self.logger.warning("  No ML model found, skipping cancer prediction")
                    return

            # Load classifier
            classifier = PanCancerClassifier(model_dir=model_dir)
            classifier.load()
            self.logger.info(f"  Loaded Pan-Cancer classifier from {model_dir}")

            # Predict (pseudobulk_df is already genes × samples format)
            # classifier.predict expects Gene x Sample matrix
            results = classifier.predict(pseudobulk_df)

            # Process results
            predictions = []
            for result in results:
                predictions.append({
                    'sample_id': result.sample_id,
                    'predicted_cancer': result.predicted_cancer,
                    'predicted_cancer_korean': result.predicted_cancer_korean,
                    'confidence': result.confidence,
                    'confidence_level': result.confidence_level,
                    'is_unknown': result.is_unknown,
                    'top_3_predictions': result.top_k_predictions[:3] if result.top_k_predictions else [],
                })

            # Get bulk prediction as main result
            bulk_pred = next((p for p in predictions if p['sample_id'] == 'Bulk_All'), None)

            self.cancer_prediction = {
                'predicted_cancer': bulk_pred['predicted_cancer'] if bulk_pred else 'Unknown',
                'predicted_cancer_korean': bulk_pred['predicted_cancer_korean'] if bulk_pred else '알 수 없음',
                'confidence': bulk_pred['confidence'] if bulk_pred else 0.0,
                'confidence_level': bulk_pred['confidence_level'] if bulk_pred else 'low',
                'all_predictions': predictions,
                'cluster_predictions': [p for p in predictions if p['sample_id'] != 'Bulk_All'],
            }

            # Check cluster agreement
            cluster_preds = [p['predicted_cancer'] for p in predictions if p['sample_id'] != 'Bulk_All']
            if cluster_preds:
                most_common = max(set(cluster_preds), key=cluster_preds.count)
                agreement = cluster_preds.count(most_common) / len(cluster_preds)
                self.cancer_prediction['cluster_agreement'] = agreement
                self.cancer_prediction['most_common_prediction'] = most_common

            self.logger.info(f"  Predicted cancer: {self.cancer_prediction['predicted_cancer']} "
                           f"(confidence: {self.cancer_prediction['confidence']:.1%})")

            # Save prediction
            pred_path = self.output_dir / "pseudobulk_prediction.json"
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(self.cancer_prediction, f, indent=2, ensure_ascii=False)
            self.logger.info(f"  Saved prediction to {pred_path.name}")

        except Exception as e:
            self.logger.error(f"  Cancer prediction failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 11: Driver Gene Matching
    # ═══════════════════════════════════════════════════════════════════════════════

    def _match_driver_genes(self):
        """Match marker genes to known cancer driver databases."""
        if not self.config.get("enable_driver_matching", True):
            self.logger.info("  Driver matching disabled, skipping...")
            return

        if self.markers_df is None or len(self.markers_df) == 0:
            self.logger.warning("  No marker genes found, skipping driver matching")
            return

        self.logger.info("Step 11: Matching Markers to Driver Databases")

        # Get unique marker genes
        marker_genes = set(self.markers_df['gene'].unique())
        self.logger.info(f"  Total unique markers: {len(marker_genes)}")

        # Match against databases
        cosmic_matches = marker_genes & COSMIC_TIER1_GENES
        oncokb_matches = marker_genes & ONCOKB_ACTIONABLE_GENES
        tme_matches = marker_genes & TME_GENES
        all_driver_genes = cosmic_matches | oncokb_matches

        self.logger.info(f"  COSMIC Tier1 matches: {len(cosmic_matches)}")
        self.logger.info(f"  OncoKB actionable matches: {len(oncokb_matches)}")
        self.logger.info(f"  TME markers: {len(tme_matches)}")

        # Build driver genes DataFrame
        driver_records = []
        for _, row in self.markers_df.iterrows():
            gene = row['gene']
            if gene in all_driver_genes or gene in tme_matches:
                record = {
                    'gene': gene,
                    'cluster': row['cluster'],
                    'score': row.get('score', 0),
                    'logfoldchange': row.get('logfoldchange', 0),
                    'pval_adj': row.get('pval_adj', row.get('pval', 1)),
                    'is_cosmic_tier1': gene in COSMIC_TIER1_GENES,
                    'is_oncokb_actionable': gene in ONCOKB_ACTIONABLE_GENES,
                    'is_tme_marker': gene in TME_GENES,
                    'driver_type': self._classify_driver_type(gene),
                }
                driver_records.append(record)

        self.driver_genes_df = pd.DataFrame(driver_records)

        if len(self.driver_genes_df) > 0:
            # Sort by score
            self.driver_genes_df = self.driver_genes_df.sort_values('score', ascending=False)
            # Remove duplicates, keeping highest score
            self.driver_genes_df = self.driver_genes_df.drop_duplicates(subset='gene', keep='first')

            # Save
            self.save_csv(self.driver_genes_df, "driver_genes.csv")
            self.logger.info(f"  Saved {len(self.driver_genes_df)} driver genes to driver_genes.csv")

            # Summary by type
            summary = {
                'total_drivers': len(self.driver_genes_df),
                'cosmic_tier1': len(self.driver_genes_df[self.driver_genes_df['is_cosmic_tier1']]),
                'oncokb_actionable': len(self.driver_genes_df[self.driver_genes_df['is_oncokb_actionable']]),
                'tme_markers': len(self.driver_genes_df[self.driver_genes_df['is_tme_marker']]),
                'top_drivers': self.driver_genes_df.head(10)['gene'].tolist(),
            }
            self.logger.info(f"  Top 10 drivers: {', '.join(summary['top_drivers'])}")
        else:
            self.logger.info("  No driver genes found in markers")

    def _classify_driver_type(self, gene: str) -> str:
        """Classify driver gene type."""
        types = []
        if gene in {'TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL', 'NF1', 'NF2', 'CDKN2A', 'SMAD4', 'BAP1'}:
            types.append('tumor_suppressor')
        if gene in {'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'MYC', 'ERBB2', 'CDK4', 'MDM2', 'CCND1', 'AKT1', 'NRAS', 'HRAS'}:
            types.append('oncogene')
        if gene in TME_GENES:
            types.append('tme_marker')
        if gene in {'ALK', 'ROS1', 'RET', 'NTRK1', 'NTRK2', 'NTRK3', 'FGFR1', 'FGFR2', 'FGFR3', 'MET'}:
            types.append('targetable_kinase')
        return ';'.join(types) if types else 'other'

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 12: Cluster Pathway Enrichment
    # ═══════════════════════════════════════════════════════════════════════════════

    def _analyze_cluster_pathways(self):
        """Run pathway enrichment analysis per cluster."""
        if not self.config.get("enable_pathway_analysis", True):
            self.logger.info("  Pathway analysis disabled, skipping...")
            return

        if not HAS_GSEAPY:
            self.logger.warning("  gseapy not available, skipping pathway analysis")
            return

        if self.markers_df is None or len(self.markers_df) == 0:
            self.logger.warning("  No marker genes found, skipping pathway analysis")
            return

        self.logger.info("Step 12: Cluster Pathway Enrichment")

        databases = self.config.get("pathway_databases", ["GO_Biological_Process_2023", "KEGG_2021_Human"])
        top_n = self.config.get("pathway_top_genes", 100)

        all_pathways = []
        clusters = self.markers_df['cluster'].unique()

        for cluster in clusters:
            # Get top markers for this cluster
            cluster_markers = self.markers_df[self.markers_df['cluster'] == cluster]
            top_genes = cluster_markers.nlargest(top_n, 'score')['gene'].tolist()

            if len(top_genes) < 5:
                self.logger.warning(f"    Cluster {cluster}: too few genes ({len(top_genes)}), skipping")
                continue

            self.logger.info(f"    Cluster {cluster}: {len(top_genes)} genes")

            for db in databases:
                try:
                    enr = gp.enrichr(
                        gene_list=top_genes,
                        gene_sets=db,
                        organism='human',
                        outdir=None,
                        no_plot=True,
                        verbose=False
                    )

                    if enr.results is not None and len(enr.results) > 0:
                        # Get top 5 pathways per database
                        top_pathways = enr.results.nsmallest(5, 'Adjusted P-value')
                        for _, row in top_pathways.iterrows():
                            all_pathways.append({
                                'cluster': cluster,
                                'database': db,
                                'term': row['Term'],
                                'padj': row['Adjusted P-value'],
                                'odds_ratio': row.get('Odds Ratio', 0),
                                'combined_score': row.get('Combined Score', 0),
                                'genes': row.get('Genes', ''),
                                'gene_count': len(row.get('Genes', '').split(';')) if row.get('Genes') else 0,
                            })
                except Exception as e:
                    self.logger.warning(f"    Enrichr failed for cluster {cluster}, {db}: {e}")
                    continue

        if all_pathways:
            self.cluster_pathways_df = pd.DataFrame(all_pathways)
            self.save_csv(self.cluster_pathways_df, "cluster_pathways.csv")
            self.logger.info(f"  Saved {len(self.cluster_pathways_df)} pathway terms to cluster_pathways.csv")

            # Summary
            n_clusters_with_pathways = self.cluster_pathways_df['cluster'].nunique()
            self.logger.info(f"  Pathways found for {n_clusters_with_pathways}/{len(clusters)} clusters")
        else:
            self.logger.info("  No significant pathways found")

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 13: Trajectory Analysis (PAGA + Diffusion Pseudotime)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _analyze_trajectory(self):
        """Perform trajectory analysis using PAGA and diffusion pseudotime."""
        if not self.config.get("enable_trajectory", True):
            self.logger.info("  Trajectory analysis disabled, skipping...")
            return

        self.logger.info("Step 13: Trajectory Analysis (PAGA + Pseudotime)")

        try:
            # PAGA (Partition-based Graph Abstraction)
            self.logger.info("  Computing PAGA...")
            sc.tl.paga(self.adata, groups='cluster')

            # Find root cluster (if not specified)
            root_cluster = self.config.get("trajectory_root")
            if root_cluster is None:
                # Auto-detect: use cluster with highest stem cell marker score or most progenitor-like
                root_cluster = self._find_root_cluster()
                self.logger.info(f"  Auto-detected root cluster: {root_cluster}")

            # Diffusion pseudotime
            self.logger.info("  Computing diffusion pseudotime...")
            # Set root cell (random cell from root cluster)
            root_cells = self.adata.obs[self.adata.obs['cluster'] == str(root_cluster)].index
            if len(root_cells) > 0:
                self.adata.uns['iroot'] = np.where(self.adata.obs_names == root_cells[0])[0][0]
                sc.tl.diffmap(self.adata)
                sc.tl.dpt(self.adata)
                self.logger.info(f"  Diffusion pseudotime computed (root: cluster {root_cluster})")
            else:
                self.logger.warning(f"  Root cluster {root_cluster} not found, using first cell")
                self.adata.uns['iroot'] = 0
                sc.tl.diffmap(self.adata)
                sc.tl.dpt(self.adata)

            # Save trajectory results
            self.trajectory_results = {
                'root_cluster': str(root_cluster),
                'has_paga': 'paga' in self.adata.uns,
                'has_dpt': 'dpt_pseudotime' in self.adata.obs.columns,
                'pseudotime_range': [
                    float(self.adata.obs['dpt_pseudotime'].min()),
                    float(self.adata.obs['dpt_pseudotime'].max())
                ] if 'dpt_pseudotime' in self.adata.obs.columns else None,
            }

            # Save pseudotime to CSV
            if 'dpt_pseudotime' in self.adata.obs.columns:
                pseudotime_df = self.adata.obs[['cluster', 'cell_type', 'dpt_pseudotime']].copy()
                pseudotime_df['cell_id'] = self.adata.obs_names
                self.save_csv(pseudotime_df.reset_index(drop=True), "trajectory_pseudotime.csv")

            # Generate trajectory plots
            self._generate_trajectory_plots()

            self.logger.info(f"  Trajectory analysis complete")

        except Exception as e:
            self.logger.error(f"  Trajectory analysis failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _find_root_cluster(self) -> str:
        """Auto-detect root cluster based on stem/progenitor markers."""
        # Stem/progenitor markers
        stem_markers = ['CD34', 'KIT', 'THY1', 'PROM1', 'NANOG', 'POU5F1', 'SOX2']
        progenitor_markers = ['CD38', 'PTPRC', 'CD44']

        # Check which markers exist
        valid_markers = [g for g in stem_markers + progenitor_markers if g in self.adata.var_names]

        if valid_markers:
            # Score cells for stemness
            try:
                sc.tl.score_genes(self.adata, valid_markers, score_name='stemness_score')
                # Find cluster with highest average stemness
                cluster_scores = self.adata.obs.groupby('cluster')['stemness_score'].mean()
                root_cluster = cluster_scores.idxmax()
                return root_cluster
            except:
                pass

        # Fallback: use cluster with fewest differentiation markers
        # Or simply use cluster "0"
        return '0'

    def _generate_trajectory_plots(self):
        """Generate trajectory visualization plots."""
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # PAGA graph
            if 'paga' in self.adata.uns:
                sc.pl.paga(
                    self.adata,
                    color='cluster',
                    title='PAGA Graph',
                    save='_paga.png',
                    show=False
                )
                self.logger.info("  Saved paga_paga.png")

            # UMAP with pseudotime
            if 'dpt_pseudotime' in self.adata.obs.columns:
                sc.pl.umap(
                    self.adata,
                    color='dpt_pseudotime',
                    title='Diffusion Pseudotime',
                    save='_pseudotime.png',
                    show=False
                )
                self.logger.info("  Saved umap_pseudotime.png")

        except Exception as e:
            self.logger.warning(f"  Failed to generate trajectory plots: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 14: Cell-Cell Interaction Analysis
    # ═══════════════════════════════════════════════════════════════════════════════

    def _analyze_cell_interactions(self):
        """Analyze cell-cell interactions using ligand-receptor databases."""
        if not self.config.get("enable_cell_interaction", True):
            self.logger.info("  Cell-cell interaction analysis disabled, skipping...")
            return

        self.logger.info("Step 14: Cell-Cell Interaction Analysis")

        # Check for required packages
        if HAS_LIANA:
            self._run_liana_interaction()
        else:
            # Fallback: simple ligand-receptor analysis
            self._run_simple_lr_analysis()

    def _run_liana_interaction(self):
        """Run cell-cell interaction analysis using LIANA."""
        try:
            self.logger.info("  Running LIANA cell-cell interaction analysis...")

            # Run LIANA with multiple methods
            li.mt.rank_aggregate(
                self.adata,
                groupby='cell_type',
                resource_name='consensus',
                verbose=False
            )

            # Get results
            if 'liana_res' in self.adata.uns:
                results = self.adata.uns['liana_res']

                # Save top interactions
                top_interactions = results.nsmallest(100, 'magnitude_rank')
                self.save_csv(top_interactions, "cell_interactions.csv")
                self.logger.info(f"  Saved {len(top_interactions)} top interactions")

                # Summarize
                self.interaction_results = {
                    'method': 'liana',
                    'n_interactions': len(results),
                    'n_significant': len(results[results['magnitude_rank'] < 0.05]),
                    'top_pairs': top_interactions[['source', 'target', 'ligand_complex', 'receptor_complex']].head(10).to_dict('records')
                }

                self.logger.info(f"  Found {self.interaction_results['n_significant']} significant interactions")

        except Exception as e:
            self.logger.error(f"  LIANA analysis failed: {e}")
            self._run_simple_lr_analysis()

    def _run_simple_lr_analysis(self):
        """Enhanced ligand-receptor analysis with CellChat-style approach."""
        self.logger.info("  Running enhanced ligand-receptor analysis...")

        # Comprehensive ligand-receptor database (CellChat/CellPhoneDB style)
        lr_database = {
            # =====================================================
            # IMMUNE CHECKPOINTS (Critical for Immunotherapy)
            # =====================================================
            'immune_checkpoint': [
                ('CD274', 'PDCD1', 'PD-L1_PD-1', 'inhibitory'),      # PD-L1 - PD-1
                ('PDCD1LG2', 'PDCD1', 'PD-L2_PD-1', 'inhibitory'),   # PD-L2 - PD-1
                ('CD80', 'CTLA4', 'CD80_CTLA4', 'inhibitory'),       # CD80 - CTLA4
                ('CD86', 'CTLA4', 'CD86_CTLA4', 'inhibitory'),       # CD86 - CTLA4
                ('CD80', 'CD28', 'CD80_CD28', 'stimulatory'),        # CD80 - CD28
                ('CD86', 'CD28', 'CD86_CD28', 'stimulatory'),        # CD86 - CD28
                ('LGALS9', 'HAVCR2', 'Galectin9_TIM3', 'inhibitory'),# Galectin-9 - TIM-3
                ('TNFRSF14', 'BTLA', 'HVEM_BTLA', 'inhibitory'),     # HVEM - BTLA
                ('CD274', 'CD80', 'PD-L1_CD80', 'inhibitory'),       # PD-L1 - CD80 (cis)
                ('TIGIT', 'PVR', 'TIGIT_PVR', 'inhibitory'),         # TIGIT - PVR
                ('CD96', 'PVR', 'CD96_PVR', 'inhibitory'),           # CD96 - PVR
                ('LAG3', 'HLA-DRA', 'LAG3_MHCII', 'inhibitory'),     # LAG-3 - MHC-II
                ('SIGLEC10', 'CD24', 'SIGLEC10_CD24', 'inhibitory'), # Don't eat me signal
            ],

            # =====================================================
            # GROWTH FACTORS & RECEPTORS
            # =====================================================
            'growth_factor': [
                ('VEGFA', 'KDR', 'VEGFA_VEGFR2', 'angiogenesis'),
                ('VEGFA', 'FLT1', 'VEGFA_VEGFR1', 'angiogenesis'),
                ('VEGFB', 'FLT1', 'VEGFB_VEGFR1', 'angiogenesis'),
                ('VEGFC', 'FLT4', 'VEGFC_VEGFR3', 'lymphangiogenesis'),
                ('EGF', 'EGFR', 'EGF_EGFR', 'proliferation'),
                ('TGFA', 'EGFR', 'TGFA_EGFR', 'proliferation'),
                ('AREG', 'EGFR', 'AREG_EGFR', 'proliferation'),
                ('HGF', 'MET', 'HGF_MET', 'invasion'),
                ('TGFB1', 'TGFBR1', 'TGFB1_TGFBR1', 'immunosuppression'),
                ('TGFB1', 'TGFBR2', 'TGFB1_TGFBR2', 'immunosuppression'),
                ('TGFB2', 'TGFBR1', 'TGFB2_TGFBR1', 'immunosuppression'),
                ('PDGFA', 'PDGFRA', 'PDGFA_PDGFRA', 'fibrosis'),
                ('PDGFB', 'PDGFRB', 'PDGFB_PDGFRB', 'fibrosis'),
                ('FGF2', 'FGFR1', 'FGF2_FGFR1', 'angiogenesis'),
                ('IGF1', 'IGF1R', 'IGF1_IGF1R', 'survival'),
                ('IGF2', 'IGF1R', 'IGF2_IGF1R', 'survival'),
                ('WNT5A', 'FZD5', 'WNT5A_FZD5', 'invasion'),
            ],

            # =====================================================
            # CHEMOKINES & RECEPTORS (TME recruitment)
            # =====================================================
            'chemokine': [
                ('CXCL12', 'CXCR4', 'CXCL12_CXCR4', 'homing'),       # SDF-1 - CXCR4
                ('CXCL12', 'CXCR7', 'CXCL12_CXCR7', 'homing'),       # SDF-1 - ACKR3
                ('CCL2', 'CCR2', 'CCL2_CCR2', 'monocyte_recruitment'),
                ('CCL5', 'CCR5', 'CCL5_CCR5', 'T_cell_recruitment'),
                ('CXCL8', 'CXCR1', 'CXCL8_CXCR1', 'neutrophil_recruitment'),
                ('CXCL8', 'CXCR2', 'CXCL8_CXCR2', 'neutrophil_recruitment'),
                ('CXCL9', 'CXCR3', 'CXCL9_CXCR3', 'T_cell_recruitment'),
                ('CXCL10', 'CXCR3', 'CXCL10_CXCR3', 'T_cell_recruitment'),
                ('CXCL11', 'CXCR3', 'CXCL11_CXCR3', 'T_cell_recruitment'),
                ('CCL19', 'CCR7', 'CCL19_CCR7', 'DC_migration'),
                ('CCL21', 'CCR7', 'CCL21_CCR7', 'DC_migration'),
                ('CCL3', 'CCR1', 'CCL3_CCR1', 'macrophage_recruitment'),
                ('CCL4', 'CCR5', 'CCL4_CCR5', 'macrophage_recruitment'),
                ('CCL17', 'CCR4', 'CCL17_CCR4', 'Treg_recruitment'),
                ('CCL22', 'CCR4', 'CCL22_CCR4', 'Treg_recruitment'),
                ('CX3CL1', 'CX3CR1', 'CX3CL1_CX3CR1', 'NK_recruitment'),
            ],

            # =====================================================
            # CYTOKINES & INTERLEUKINS
            # =====================================================
            'cytokine': [
                ('IL6', 'IL6R', 'IL6_IL6R', 'inflammation'),
                ('IL6', 'IL6ST', 'IL6_gp130', 'inflammation'),
                ('IL10', 'IL10RA', 'IL10_IL10R', 'immunosuppression'),
                ('IL1B', 'IL1R1', 'IL1B_IL1R', 'inflammation'),
                ('IL1A', 'IL1R1', 'IL1A_IL1R', 'inflammation'),
                ('IFNG', 'IFNGR1', 'IFNG_IFNGR', 'antitumor'),
                ('TNF', 'TNFRSF1A', 'TNF_TNFR1', 'apoptosis'),
                ('TNF', 'TNFRSF1B', 'TNF_TNFR2', 'survival'),
                ('IL2', 'IL2RA', 'IL2_IL2R', 'T_cell_activation'),
                ('IL15', 'IL15RA', 'IL15_IL15R', 'NK_activation'),
                ('IL4', 'IL4R', 'IL4_IL4R', 'Th2_polarization'),
                ('IL13', 'IL13RA1', 'IL13_IL13R', 'Th2_polarization'),
                ('IL17A', 'IL17RA', 'IL17_IL17R', 'inflammation'),
                ('IL23A', 'IL23R', 'IL23_IL23R', 'Th17_polarization'),
                ('CSF1', 'CSF1R', 'CSF1_CSF1R', 'macrophage_polarization'),
                ('CSF2', 'CSF2RA', 'GMCSF_GMCSFR', 'DC_maturation'),
            ],

            # =====================================================
            # DEATH RECEPTOR SIGNALING
            # =====================================================
            'death_receptor': [
                ('TNFSF10', 'TNFRSF10A', 'TRAIL_DR4', 'apoptosis'),
                ('TNFSF10', 'TNFRSF10B', 'TRAIL_DR5', 'apoptosis'),
                ('FASLG', 'FAS', 'FASL_FAS', 'apoptosis'),
                ('TNFSF14', 'TNFRSF14', 'LIGHT_HVEM', 'apoptosis'),
            ],

            # =====================================================
            # NOTCH SIGNALING
            # =====================================================
            'notch': [
                ('DLL1', 'NOTCH1', 'DLL1_NOTCH1', 'differentiation'),
                ('DLL4', 'NOTCH1', 'DLL4_NOTCH1', 'angiogenesis'),
                ('JAG1', 'NOTCH1', 'JAG1_NOTCH1', 'stemness'),
                ('JAG1', 'NOTCH2', 'JAG1_NOTCH2', 'differentiation'),
                ('JAG2', 'NOTCH1', 'JAG2_NOTCH1', 'differentiation'),
            ],

            # =====================================================
            # CELL ADHESION & MIGRATION
            # =====================================================
            'adhesion': [
                ('ICAM1', 'ITGAL', 'ICAM1_LFA1', 'adhesion'),        # LFA-1 = CD11a/CD18
                ('VCAM1', 'ITGA4', 'VCAM1_VLA4', 'adhesion'),        # VLA-4 = α4β1
                ('SELE', 'SELPLG', 'ESELECTIN_PSGL1', 'rolling'),
                ('SELP', 'SELPLG', 'PSELECTIN_PSGL1', 'rolling'),
                ('CD44', 'SPP1', 'CD44_OPN', 'invasion'),
                ('CD44', 'HAS1', 'CD44_HA', 'invasion'),
                ('ITGAV', 'THBS1', 'INTEGRIN_TSP1', 'adhesion'),
                ('CDH1', 'CDH1', 'ECAD_ECAD', 'homophilic'),         # E-cadherin
            ],

            # =====================================================
            # MHC & ANTIGEN PRESENTATION
            # =====================================================
            'antigen_presentation': [
                ('HLA-A', 'CD8A', 'MHCI_CD8', 'T_cell_recognition'),
                ('HLA-B', 'CD8A', 'MHCI_CD8', 'T_cell_recognition'),
                ('HLA-C', 'CD8A', 'MHCI_CD8', 'T_cell_recognition'),
                ('HLA-DRA', 'CD4', 'MHCII_CD4', 'T_cell_recognition'),
                ('HLA-A', 'KIR2DL1', 'MHCI_KIR', 'NK_inhibition'),
                ('HLA-A', 'KIR3DL1', 'MHCI_KIR', 'NK_inhibition'),
                ('B2M', 'LILRB1', 'B2M_LILRB1', 'NK_inhibition'),
            ],

            # =====================================================
            # COSTIMULATORY MOLECULES
            # =====================================================
            'costimulatory': [
                ('CD40LG', 'CD40', 'CD40L_CD40', 'APC_activation'),
                ('TNFSF4', 'TNFRSF4', 'OX40L_OX40', 'T_cell_survival'),
                ('TNFSF9', 'TNFRSF9', '4-1BBL_4-1BB', 'T_cell_survival'),
                ('CD70', 'CD27', 'CD70_CD27', 'T_cell_activation'),
                ('ICOS', 'ICOSLG', 'ICOS_ICOSL', 'T_cell_activation'),
            ],
        }

        # Convert to simple list format
        lr_pairs = []
        lr_categories = {}
        for category, pairs in lr_database.items():
            for pair_info in pairs:
                ligand, receptor, name, function = pair_info
                lr_pairs.append((ligand, receptor))
                lr_categories[(ligand, receptor)] = {
                    'category': category,
                    'name': name,
                    'function': function
                }

        # Calculate mean expression per cell type for all genes at once
        cell_types = self.adata.obs['cell_type'].unique().tolist()
        self.logger.info(f"  Analyzing {len(lr_pairs)} ligand-receptor pairs across {len(cell_types)} cell types...")

        # Pre-compute mean expression per cell type
        expr_by_celltype = {}
        for ct in cell_types:
            mask = self.adata.obs['cell_type'] == ct
            subset = self.adata[mask]
            if hasattr(subset.X, 'toarray'):
                expr_by_celltype[ct] = pd.Series(
                    np.asarray(subset.X.mean(axis=0)).flatten(),
                    index=self.adata.var_names
                )
            else:
                expr_by_celltype[ct] = pd.Series(
                    subset.X.mean(axis=0).flatten(),
                    index=self.adata.var_names
                )

        # Calculate interactions
        interactions = []
        for ligand, receptor in lr_pairs:
            if ligand not in self.adata.var_names or receptor not in self.adata.var_names:
                continue

            pair_info = lr_categories.get((ligand, receptor), {})
            category = pair_info.get('category', 'unknown')
            name = pair_info.get('name', f'{ligand}_{receptor}')
            function = pair_info.get('function', 'unknown')

            for source_type in cell_types:
                ligand_expr = float(expr_by_celltype[source_type].get(ligand, 0))
                if ligand_expr < 0.1:  # Minimum ligand expression threshold
                    continue

                for target_type in cell_types:
                    receptor_expr = float(expr_by_celltype[target_type].get(receptor, 0))
                    if receptor_expr < 0.1:  # Minimum receptor expression threshold
                        continue

                    # CellChat-style score: geometric mean
                    score = np.sqrt(ligand_expr * receptor_expr)

                    # Specificity: how specific is this interaction?
                    # (expression in this pair vs overall mean)
                    all_ligand_expr = np.mean([expr_by_celltype[ct].get(ligand, 0) for ct in cell_types])
                    all_receptor_expr = np.mean([expr_by_celltype[ct].get(receptor, 0) for ct in cell_types])
                    specificity = score / (np.sqrt(all_ligand_expr * all_receptor_expr) + 0.01)

                    interactions.append({
                        'source': source_type,
                        'target': target_type,
                        'ligand': ligand,
                        'receptor': receptor,
                        'interaction_name': name,
                        'category': category,
                        'function': function,
                        'ligand_expr': ligand_expr,
                        'receptor_expr': receptor_expr,
                        'interaction_score': score,
                        'specificity': specificity,
                    })

        if interactions:
            interaction_df = pd.DataFrame(interactions)
            interaction_df = interaction_df.sort_values('interaction_score', ascending=False)

            # Save full results
            self.save_csv(interaction_df, "cell_interactions.csv")

            # Summarize by category
            category_summary = interaction_df.groupby('category').agg({
                'interaction_score': ['count', 'mean', 'max']
            }).round(3)
            category_summary.columns = ['n_interactions', 'mean_score', 'max_score']
            category_summary = category_summary.sort_values('n_interactions', ascending=False)

            # Top interactions per category
            top_by_category = {}
            for cat in interaction_df['category'].unique():
                cat_df = interaction_df[interaction_df['category'] == cat]
                top_by_category[cat] = cat_df.nlargest(5, 'interaction_score')[
                    ['source', 'target', 'interaction_name', 'interaction_score']
                ].to_dict('records')

            self.interaction_results = {
                'method': 'cellchat_style',
                'n_interactions': len(interaction_df),
                'n_pairs_checked': len(lr_pairs),
                'n_pairs_expressed': len(interaction_df[['ligand', 'receptor']].drop_duplicates()),
                'category_summary': category_summary.to_dict(),
                'top_by_category': top_by_category,
                'top_interactions': interaction_df.head(30).to_dict('records'),
            }

            self.logger.info(f"  Found {len(interaction_df)} interactions from {self.interaction_results['n_pairs_expressed']} L-R pairs")
            self.logger.info(f"  Categories: {list(category_summary.index)}")
        else:
            self.logger.info("  No significant interactions found")
            self.interaction_results = {'method': 'cellchat_style', 'n_interactions': 0}

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 15: TME (Tumor Microenvironment) Analysis
    # ═══════════════════════════════════════════════════════════════════════════════

    def _analyze_tme(self):
        """Analyze tumor microenvironment composition and immune infiltration scores."""
        if not self.config.get("enable_tme_analysis", True):
            self.logger.info("  TME analysis disabled, skipping...")
            return

        self.logger.info("Step 15: TME (Tumor Microenvironment) Analysis")

        # TME cell type categories
        tme_categories = {
            # Immune cells - Anti-tumor
            'cytotoxic': ['CD8_T', 'CD8A', 'NK_cells', 'NKT'],
            'helper': ['CD4_T', 'T_cells', 'Th1', 'Th2'],
            'antigen_presenting': ['Dendritic_cells', 'cDC1', 'cDC2', 'pDC', 'B_cells'],

            # Immune cells - Pro-tumor / Immunosuppressive
            'immunosuppressive': ['Treg', 'MDSC', 'M2_Macro'],
            'myeloid': ['Monocytes', 'Macrophages', 'Classical_Mono', 'Non_classical_Mono', 'Kupffer_cells'],

            # Stromal cells
            'stromal': ['Fibroblasts', 'CAF', 'Myofibroblast', 'Stellate_cells', 'Pericytes'],
            'endothelial': ['Endothelial', 'Vascular_EC', 'Lymphatic_EC', 'LSECs'],

            # Tumor/Epithelial cells (varies by cancer type)
            'epithelial': ['Epithelial', 'Hepatocytes', 'Colonocytes', 'Alveolar_type2',
                          'Luminal_epithelial', 'Ductal_cells', 'Acinar_cells'],
        }

        # Immune signature genes (CIBERSORT-style markers)
        immune_signatures = {
            'T_cell_activation': ['CD3D', 'CD3E', 'CD28', 'LCK', 'ZAP70', 'ITK'],
            'Cytotoxic_activity': ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'GNLY', 'NKG7', 'IFNG'],
            'Exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX'],
            'Treg_signature': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'IL10'],
            'M1_macrophage': ['CD80', 'CD86', 'IL1B', 'TNF', 'NOS2', 'CXCL9', 'CXCL10'],
            'M2_macrophage': ['CD163', 'MRC1', 'MSR1', 'IL10', 'TGFB1', 'CCL18'],
            'Inflammatory': ['IL1B', 'IL6', 'TNF', 'CXCL8', 'CCL2', 'PTGS2'],
            'Angiogenesis': ['VEGFA', 'VEGFB', 'VEGFC', 'FGF2', 'ANGPT1', 'ANGPT2'],
            'Hypoxia': ['HIF1A', 'LDHA', 'SLC2A1', 'PGK1', 'VEGFA', 'CA9'],
            'Antigen_presentation': ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 'B2M', 'TAP1', 'TAP2'],
            'IFN_response': ['STAT1', 'IRF1', 'IRF7', 'MX1', 'OAS1', 'ISG15', 'IFIT1'],
        }

        # Calculate TME composition based on cell type annotations
        tme_composition = {}
        total_cells = len(self.adata.obs)

        for category, cell_types in tme_categories.items():
            count = 0
            for ct in cell_types:
                # Match cell types (handling suffixes like _2, _3)
                matches = self.adata.obs['cell_type'].str.startswith(ct).sum()
                count += matches
            tme_composition[category] = {
                'count': int(count),
                'percentage': count / total_cells * 100
            }

        # Calculate immune signature scores per cell type
        signature_scores = {}
        cell_types = self.adata.obs['cell_type'].unique()

        for sig_name, genes in immune_signatures.items():
            valid_genes = [g for g in genes if g in self.adata.var_names]
            if len(valid_genes) >= 2:  # Need at least 2 genes
                # Score using scanpy
                score_name = f'score_{sig_name}'
                sc.tl.score_genes(self.adata, valid_genes, score_name=score_name)

                # Calculate mean score per cell type
                scores_by_ct = {}
                for ct in cell_types:
                    mask = self.adata.obs['cell_type'] == ct
                    mean_score = float(self.adata.obs.loc[mask, score_name].mean())
                    scores_by_ct[ct] = mean_score
                signature_scores[sig_name] = scores_by_ct

        # Calculate immune infiltration score (overall)
        immune_cell_types = ['T_cells', 'NK_cells', 'B_cells', 'Monocytes', 'Macrophages',
                            'Dendritic_cells', 'Plasma_cells', 'Mast_cells', 'Neutrophils']
        immune_count = 0
        for ct in immune_cell_types:
            immune_count += self.adata.obs['cell_type'].str.startswith(ct).sum()

        # Calculate stromal score
        stromal_types = ['Fibroblasts', 'CAF', 'Endothelial', 'Pericytes', 'Stellate']
        stromal_count = 0
        for ct in stromal_types:
            stromal_count += self.adata.obs['cell_type'].str.startswith(ct).sum()

        # Calculate tumor purity (non-immune, non-stromal)
        tumor_purity = (total_cells - immune_count - stromal_count) / total_cells * 100

        # Immune score: ratio of immune cells
        immune_score = immune_count / total_cells * 100

        # Stromal score: ratio of stromal cells
        stromal_score = stromal_count / total_cells * 100

        # Hot vs Cold tumor classification
        # Hot: high immune infiltration (>30%) with cytotoxic markers
        # Cold: low immune infiltration (<10%)
        cytotoxic_present = any(self.adata.obs['cell_type'].str.contains('CD8|NK|cytotoxic', case=False))
        if immune_score > 30 and cytotoxic_present:
            tumor_phenotype = 'Hot (Inflamed)'
        elif immune_score > 15:
            tumor_phenotype = 'Immune-Altered'
        else:
            tumor_phenotype = 'Cold (Desert)'

        self.tme_results = {
            'composition': tme_composition,
            'signature_scores': signature_scores,
            'immune_score': immune_score,
            'stromal_score': stromal_score,
            'tumor_purity': tumor_purity,
            'tumor_phenotype': tumor_phenotype,
            'total_cells': total_cells,
            'immune_cells': immune_count,
            'stromal_cells': stromal_count,
        }

        # Save TME results
        tme_df = pd.DataFrame([
            {'category': k, 'count': v['count'], 'percentage': v['percentage']}
            for k, v in tme_composition.items()
        ])
        self.save_csv(tme_df, "tme_composition.csv")

        # Save signature scores
        if signature_scores:
            sig_df = pd.DataFrame(signature_scores).T
            sig_df.index.name = 'signature'
            self.save_csv(sig_df.reset_index(), "tme_signature_scores.csv")

        self.logger.info(f"  Tumor phenotype: {tumor_phenotype}")
        self.logger.info(f"  Immune score: {immune_score:.1f}%")
        self.logger.info(f"  Stromal score: {stromal_score:.1f}%")
        self.logger.info(f"  Tumor purity: {tumor_purity:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 16: Gene Regulatory Network Analysis (Cell Type-specific)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _analyze_grn(self):
        """Infer gene regulatory networks per cell type (SCENIC-style approach)."""
        if not self.config.get("enable_grn_analysis", True):
            self.logger.info("  GRN analysis disabled, skipping...")
            return

        self.logger.info("Step 16: Gene Regulatory Network Analysis")

        try:
            # Define key transcription factors for cancer biology
            key_tfs = {
                # Master regulators
                'TP53', 'MYC', 'MYCN', 'JUN', 'FOS', 'STAT3', 'STAT1',
                'NFE2L2', 'HIF1A', 'NFKB1', 'RELA', 'SP1', 'E2F1',
                # Lineage TFs
                'GATA3', 'FOXA1', 'FOXA2', 'HNF4A', 'SOX2', 'SOX9',
                'PAX8', 'CDX2', 'NKX2-1', 'RUNX1', 'PU.1',
                # Immune TFs
                'FOXP3', 'TBX21', 'GATA3', 'RORC', 'BCL6', 'IRF4',
                'BATF', 'IRF8', 'SPI1', 'TCF7', 'LEF1',
                # EMT TFs
                'SNAI1', 'SNAI2', 'TWIST1', 'TWIST2', 'ZEB1', 'ZEB2',
                # Stemness TFs
                'NANOG', 'POU5F1', 'SOX2', 'KLF4', 'BMI1',
            }

            # Get available TFs in the data
            available_tfs = [tf for tf in key_tfs if tf in self.adata.var_names]
            self.logger.info(f"  Found {len(available_tfs)}/{len(key_tfs)} key TFs in data")

            if len(available_tfs) < 5:
                self.logger.warning("  Too few TFs found, skipping GRN analysis")
                self.grn_results = None
                return

            # Get expression data
            if hasattr(self.adata.X, 'toarray'):
                expr_matrix = pd.DataFrame(
                    self.adata.X.toarray(),
                    index=self.adata.obs_names,
                    columns=self.adata.var_names
                )
            else:
                expr_matrix = pd.DataFrame(
                    self.adata.X,
                    index=self.adata.obs_names,
                    columns=self.adata.var_names
                )

            # Get top variable genes for target genes (exclude TFs)
            hvg_mask = self.adata.var.get('highly_variable', pd.Series(True, index=self.adata.var_names))
            target_genes = [g for g in self.adata.var_names[hvg_mask] if g not in available_tfs][:500]

            grn_results = {}
            tf_activity_scores = {}

            # Analyze GRN per cell type
            cell_types = self.adata.obs['cell_type'].unique()

            for cell_type in cell_types:
                cell_mask = self.adata.obs['cell_type'] == cell_type
                n_cells = cell_mask.sum()

                if n_cells < 20:  # Need enough cells for correlation
                    continue

                cell_expr = expr_matrix.loc[cell_mask]

                # Calculate TF-target correlations
                tf_target_corrs = []

                for tf in available_tfs:
                    if tf not in cell_expr.columns:
                        continue

                    tf_expr = cell_expr[tf]
                    if tf_expr.std() < 0.1:  # Skip low-variance TFs
                        continue

                    # Calculate Spearman correlation with top targets
                    for target in target_genes[:100]:  # Top 100 targets per TF
                        if target not in cell_expr.columns or target == tf:
                            continue

                        target_expr = cell_expr[target]
                        if target_expr.std() < 0.1:
                            continue

                        try:
                            from scipy.stats import spearmanr
                            corr, pval = spearmanr(tf_expr, target_expr)

                            if abs(corr) > 0.3 and pval < 0.05:
                                tf_target_corrs.append({
                                    'TF': tf,
                                    'target': target,
                                    'correlation': corr,
                                    'p_value': pval,
                                    'regulation': 'activation' if corr > 0 else 'repression'
                                })
                        except:
                            continue

                if tf_target_corrs:
                    grn_df = pd.DataFrame(tf_target_corrs)
                    grn_df = grn_df.sort_values('correlation', key=abs, ascending=False)
                    grn_results[cell_type] = grn_df.head(50).to_dict('records')

                # Calculate TF activity score (mean expression z-score)
                tf_scores = {}
                for tf in available_tfs:
                    if tf in cell_expr.columns:
                        tf_scores[tf] = float(cell_expr[tf].mean())
                tf_activity_scores[cell_type] = tf_scores

            # Identify master regulators (TFs with most targets)
            master_regulators = {}
            for cell_type, edges in grn_results.items():
                tf_counts = {}
                for edge in edges:
                    tf = edge['TF']
                    tf_counts[tf] = tf_counts.get(tf, 0) + 1

                if tf_counts:
                    sorted_tfs = sorted(tf_counts.items(), key=lambda x: x[1], reverse=True)
                    master_regulators[cell_type] = sorted_tfs[:5]

            self.grn_results = {
                'n_tfs': len(available_tfs),
                'tfs_found': available_tfs,
                'n_cell_types': len(grn_results),
                'grn_edges': grn_results,
                'tf_activity': tf_activity_scores,
                'master_regulators': master_regulators,
            }

            # Save results
            all_edges = []
            for cell_type, edges in grn_results.items():
                for edge in edges:
                    edge['cell_type'] = cell_type
                    all_edges.append(edge)

            if all_edges:
                self.save_csv(pd.DataFrame(all_edges), "grn_edges.csv")

            # Save TF activity
            tf_activity_df = pd.DataFrame(tf_activity_scores).T
            tf_activity_df.index.name = 'cell_type'
            self.save_csv(tf_activity_df.reset_index(), "tf_activity_scores.csv")

            # Save master regulators
            mr_rows = []
            for cell_type, regulators in master_regulators.items():
                for rank, (tf, count) in enumerate(regulators, 1):
                    mr_rows.append({
                        'cell_type': cell_type,
                        'rank': rank,
                        'TF': tf,
                        'n_targets': count
                    })
            if mr_rows:
                self.save_csv(pd.DataFrame(mr_rows), "master_regulators.csv")

            self.logger.info(f"  Analyzed {len(grn_results)} cell types")
            self.logger.info(f"  Total GRN edges: {len(all_edges)}")

            # Log top master regulators per cell type
            for cell_type, regulators in list(master_regulators.items())[:3]:
                top_tf = regulators[0] if regulators else ('N/A', 0)
                self.logger.info(f"  {cell_type}: Top TF = {top_tf[0]} ({top_tf[1]} targets)")

        except Exception as e:
            self.logger.error(f"  GRN analysis failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.grn_results = None

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW STEP 17: CNV Inference (Malignant Cell Detection)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _infer_cnv(self):
        """Infer copy number variations to distinguish malignant from normal cells."""
        if not self.config.get("enable_cnv_inference", True):
            self.logger.info("  CNV inference disabled, skipping...")
            return

        self.logger.info("Step 16: CNV Inference (Malignant Cell Detection)")

        try:
            # Simple CNV inference using chromosome arm expression
            # Based on inferCNV approach but simplified

            # Load gene chromosome positions (simplified - using gene name patterns)
            # In real implementation, would use GTF file
            # For now, score based on known amplified/deleted regions in cancer

            # Common cancer CNV signatures by gene
            cnv_signatures = {
                # Commonly amplified oncogenes
                'amplified': {
                    'MYC': 8,      # 8q24 - amplified in many cancers
                    'ERBB2': 17,   # 17q12 - breast cancer
                    'EGFR': 7,     # 7p11 - lung, GBM
                    'CCND1': 11,   # 11q13 - various
                    'CDK4': 12,    # 12q14 - various
                    'MDM2': 12,    # 12q15 - various
                    'FGFR1': 8,    # 8p11 - lung, breast
                    'MET': 7,      # 7q31 - various
                    'PIK3CA': 3,   # 3q26 - various
                },
                # Commonly deleted tumor suppressors
                'deleted': {
                    'TP53': 17,    # 17p13
                    'RB1': 13,     # 13q14
                    'CDKN2A': 9,   # 9p21
                    'CDKN2B': 9,   # 9p21
                    'PTEN': 10,    # 10q23
                    'BRCA1': 17,   # 17q21
                    'BRCA2': 13,   # 13q13
                    'APC': 5,      # 5q22
                    'SMAD4': 18,   # 18q21
                }
            }

            # Calculate CNV score per cell based on expression of these genes
            cell_cnv_scores = []

            # Get expression data
            if hasattr(self.adata.X, 'toarray'):
                expr_matrix = self.adata.X.toarray()
            else:
                expr_matrix = self.adata.X

            # Calculate reference (normal cells) - use immune cells as reference
            immune_mask = self.adata.obs['cell_type'].str.contains(
                'T_cells|NK_cells|B_cells|Monocytes|Macrophages|Dendritic',
                case=False, na=False
            )

            if immune_mask.sum() > 50:  # Need enough reference cells
                reference_expr = expr_matrix[immune_mask].mean(axis=0)
            else:
                # Use all cells as reference if not enough immune cells
                reference_expr = expr_matrix.mean(axis=0)

            # Calculate amplification and deletion scores
            amp_genes = [g for g in cnv_signatures['amplified'].keys() if g in self.adata.var_names]
            del_genes = [g for g in cnv_signatures['deleted'].keys() if g in self.adata.var_names]

            amp_scores = np.zeros(len(self.adata))
            del_scores = np.zeros(len(self.adata))

            for gene in amp_genes:
                gene_idx = self.adata.var_names.get_loc(gene)
                ref_val = reference_expr[gene_idx] if reference_expr[gene_idx] > 0 else 0.01
                cell_expr = expr_matrix[:, gene_idx]
                # Log ratio compared to reference
                amp_scores += np.log2((cell_expr + 0.01) / (ref_val + 0.01))

            for gene in del_genes:
                gene_idx = self.adata.var_names.get_loc(gene)
                ref_val = reference_expr[gene_idx] if reference_expr[gene_idx] > 0 else 0.01
                cell_expr = expr_matrix[:, gene_idx]
                # Negative log ratio for deletions (lower expression = higher score)
                del_scores -= np.log2((cell_expr + 0.01) / (ref_val + 0.01))

            # Normalize scores
            if len(amp_genes) > 0:
                amp_scores = amp_scores / len(amp_genes)
            if len(del_genes) > 0:
                del_scores = del_scores / len(del_genes)

            # Combined CNV score (higher = more likely malignant)
            cnv_scores = amp_scores + del_scores

            # Add to adata
            self.adata.obs['cnv_score'] = cnv_scores
            self.adata.obs['amp_score'] = amp_scores
            self.adata.obs['del_score'] = del_scores

            # Classify cells as likely malignant or normal
            # Use threshold based on distribution
            threshold = np.percentile(cnv_scores, 75)  # Top 25% as likely malignant
            self.adata.obs['malignancy'] = np.where(
                cnv_scores > threshold, 'Likely_Malignant', 'Likely_Normal'
            )

            # Calculate per cell type
            cnv_by_celltype = self.adata.obs.groupby('cell_type').agg({
                'cnv_score': ['mean', 'std'],
                'malignancy': lambda x: (x == 'Likely_Malignant').sum()
            })
            cnv_by_celltype.columns = ['cnv_mean', 'cnv_std', 'n_malignant']
            cnv_by_celltype['n_total'] = self.adata.obs.groupby('cell_type').size()
            cnv_by_celltype['pct_malignant'] = cnv_by_celltype['n_malignant'] / cnv_by_celltype['n_total'] * 100

            self.cnv_results = {
                'n_likely_malignant': int((self.adata.obs['malignancy'] == 'Likely_Malignant').sum()),
                'n_likely_normal': int((self.adata.obs['malignancy'] == 'Likely_Normal').sum()),
                'threshold': float(threshold),
                'amp_genes_found': amp_genes,
                'del_genes_found': del_genes,
                'cnv_by_celltype': cnv_by_celltype.to_dict(),
            }

            # Save results
            self.save_csv(cnv_by_celltype.reset_index(), "cnv_by_celltype.csv")

            self.logger.info(f"  Likely malignant cells: {self.cnv_results['n_likely_malignant']}")
            self.logger.info(f"  Likely normal cells: {self.cnv_results['n_likely_normal']}")
            self.logger.info(f"  CNV threshold: {threshold:.2f}")

        except Exception as e:
            self.logger.error(f"  CNV inference failed: {e}")
            self.cnv_results = None

    def run(self) -> Dict[str, Any]:
        """Execute the full single-cell analysis pipeline."""
        # Validate inputs first
        if not self.validate_inputs():
            return {"status": "error", "error": "Input validation failed"}

        # Skip preprocessing if already done (pre-processed h5ad)
        if self._is_preprocessed:
            self.logger.info("Using pre-processed h5ad - skipping Steps 1-6")
            # Check if cell_type annotation is needed
            # Re-annotate if cell_type is missing or if all values start with "Cluster_"
            needs_annotation = 'cell_type' not in self.adata.obs.columns
            if not needs_annotation and 'cell_type' in self.adata.obs.columns:
                # Check if cell types are just cluster IDs (Cluster_0, Cluster_1, etc.)
                cell_types = self.adata.obs['cell_type'].unique()
                if all(str(ct).startswith('Cluster_') for ct in cell_types):
                    needs_annotation = True
                    self.logger.info("  Cell types are cluster IDs, running annotation...")

            if needs_annotation or self.config.get("force_reannotate", False):
                self._annotate_cells()
            else:
                self.logger.info("Step 7: Cell Type Annotation (already present)")
        else:
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

        # Step 8: Marker Genes (always run to get fresh markers)
        self._find_markers()

        # Step 9: Visualization
        self._generate_visualizations()

        # NEW Step 10: Pseudo-bulk Cancer Prediction
        self._predict_cancer_type()

        # NEW Step 11: Driver Gene Matching
        self._match_driver_genes()

        # NEW Step 12: Cluster Pathway Enrichment
        self._analyze_cluster_pathways()

        # NEW Step 13: Trajectory Analysis
        self._analyze_trajectory()

        # NEW Step 14: Cell-Cell Interaction
        self._analyze_cell_interactions()

        # NEW Step 15: Tumor Microenvironment Analysis
        self._analyze_tme()

        # NEW Step 16: Gene Regulatory Network Analysis
        self._analyze_grn()

        # NEW Step 17: CNV Inference (Malignant Cell Detection)
        self._infer_cnv()

        # Save outputs
        self._save_outputs()

        # Compile results
        n_clusters = len(self.adata.obs['cluster'].unique())
        n_celltypes = len(self.adata.obs['cell_type'].unique())

        results = {
            "status": "success",
            "n_cells": self.adata.n_obs,
            "n_genes": self.adata.n_vars,
            "n_hvg": int(self.adata.var['highly_variable'].sum()),
            "n_clusters": n_clusters,
            "n_celltypes": n_celltypes,
            "n_markers": len(self.markers_df) if self.markers_df is not None else 0,
            "clustering_method": self.config["clustering_method"],
            "clustering_resolution": self.config["clustering_resolution"],
            # NEW results
            "cancer_prediction": self.cancer_prediction,
            "n_driver_genes": len(self.driver_genes_df) if self.driver_genes_df is not None else 0,
            "n_pathway_terms": len(self.cluster_pathways_df) if self.cluster_pathways_df is not None else 0,
            "trajectory": self.trajectory_results,
            "cell_interactions": self.interaction_results,
            # Advanced analysis results
            "tme_analysis": self.tme_results if hasattr(self, 'tme_results') else None,
            "grn_analysis": self.grn_results if hasattr(self, 'grn_results') else None,
            "cnv_analysis": self.cnv_results if hasattr(self, 'cnv_results') else None,
        }

        self.logger.info("=" * 60)
        self.logger.info("Single-Cell Analysis Complete:")
        self.logger.info(f"  Cells: {results['n_cells']}")
        self.logger.info(f"  Genes: {results['n_genes']}")
        self.logger.info(f"  HVGs: {results['n_hvg']}")
        self.logger.info(f"  Clusters: {results['n_clusters']}")
        self.logger.info(f"  Cell Types: {results['n_celltypes']}")
        self.logger.info(f"  Marker Genes: {results['n_markers']}")
        # NEW
        if self.cancer_prediction:
            self.logger.info(f"  Predicted Cancer: {self.cancer_prediction.get('predicted_cancer', 'N/A')} "
                           f"({self.cancer_prediction.get('confidence', 0):.1%})")
        self.logger.info(f"  Driver Genes: {results['n_driver_genes']}")
        self.logger.info(f"  Pathway Terms: {results['n_pathway_terms']}")

        # Advanced analysis summary
        if hasattr(self, 'tme_results') and self.tme_results:
            self.logger.info(f"  TME Phenotype: {self.tme_results.get('tumor_phenotype', 'N/A')}")
        if hasattr(self, 'grn_results') and self.grn_results:
            self.logger.info(f"  GRN: {self.grn_results.get('n_tfs', 0)} TFs, "
                           f"{self.grn_results.get('n_cell_types', 0)} cell types analyzed")
        if hasattr(self, 'cnv_results') and self.cnv_results:
            self.logger.info(f"  CNV: {self.cnv_results.get('n_likely_malignant', 0)} likely malignant cells")

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
