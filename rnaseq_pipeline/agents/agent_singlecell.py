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

        # Combined UMAP + Bar chart visualization
        self._generate_celltype_composition_plot(figures_dir)

    def _generate_celltype_composition_plot(self, figures_dir: Path):
        """Generate combined UMAP and cell type composition bar chart."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        try:
            # Get cell type counts and sort by frequency
            cell_type_counts = self.adata.obs['cell_type'].value_counts()
            cell_types = cell_type_counts.index.tolist()
            counts = cell_type_counts.values
            percentages = counts / counts.sum() * 100

            # Create a consistent color palette
            n_types = len(cell_types)
            cmap = plt.cm.get_cmap('tab20' if n_types <= 20 else 'tab20b')
            colors = {ct: cmap(i % 20) for i, ct in enumerate(cell_types)}

            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.2, 1]})

            # Left: UMAP with cell types
            ax_umap = axes[0]
            umap_coords = self.adata.obsm['X_umap']

            for ct in cell_types:
                mask = self.adata.obs['cell_type'] == ct
                ax_umap.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=[colors[ct]],
                    label=ct,
                    s=3,
                    alpha=0.7,
                    rasterized=True
                )

            ax_umap.set_xlabel('UMAP1', fontsize=12)
            ax_umap.set_ylabel('UMAP2', fontsize=12)
            ax_umap.set_title('Cell Types (UMAP)', fontsize=14, fontweight='bold')
            ax_umap.set_xticks([])
            ax_umap.set_yticks([])

            # Right: Horizontal bar chart
            ax_bar = axes[1]
            y_pos = range(len(cell_types))
            bar_colors = [colors[ct] for ct in cell_types]

            bars = ax_bar.barh(y_pos, percentages, color=bar_colors, edgecolor='white', linewidth=0.5)

            # Add percentage labels
            for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts)):
                ax_bar.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}% ({cnt:,})',
                    va='center',
                    ha='left',
                    fontsize=9
                )

            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(cell_types, fontsize=10)
            ax_bar.set_xlabel('Percentage (%)', fontsize=12)
            ax_bar.set_title('Cell Type Composition', fontsize=14, fontweight='bold')
            ax_bar.set_xlim(0, max(percentages) * 1.3)  # Leave room for labels
            ax_bar.invert_yaxis()  # Largest on top
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)

            plt.tight_layout()
            save_path = figures_dir / 'celltype_composition.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            self.logger.info("  Saved celltype_composition.png")

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
        """Simple ligand-receptor analysis without LIANA."""
        self.logger.info("  Running simple ligand-receptor analysis...")

        # Common ligand-receptor pairs (curated list)
        lr_pairs = [
            # Immune checkpoints
            ('CD274', 'PDCD1'),      # PD-L1 - PD-1
            ('CD80', 'CD28'),        # CD80 - CD28
            ('CD80', 'CTLA4'),       # CD80 - CTLA4
            ('CD86', 'CD28'),        # CD86 - CD28
            ('LGALS9', 'HAVCR2'),    # Galectin-9 - TIM-3

            # Growth factors
            ('VEGFA', 'KDR'),        # VEGF - VEGFR2
            ('VEGFA', 'FLT1'),       # VEGF - VEGFR1
            ('EGF', 'EGFR'),         # EGF - EGFR
            ('HGF', 'MET'),          # HGF - MET
            ('TGFB1', 'TGFBR1'),     # TGF-β - TGF-βR

            # Chemokines
            ('CXCL12', 'CXCR4'),     # SDF-1 - CXCR4
            ('CCL2', 'CCR2'),        # MCP-1 - CCR2
            ('CCL5', 'CCR5'),        # RANTES - CCR5
            ('CXCL8', 'CXCR1'),      # IL-8 - CXCR1

            # TNF family
            ('TNFSF10', 'TNFRSF10A'),  # TRAIL - DR4
            ('FASLG', 'FAS'),        # FasL - Fas

            # Interleukins
            ('IL6', 'IL6R'),         # IL-6 - IL-6R
            ('IL10', 'IL10RA'),      # IL-10 - IL-10R
            ('IL1B', 'IL1R1'),       # IL-1β - IL-1R

            # Notch signaling
            ('DLL1', 'NOTCH1'),      # DLL1 - Notch1
            ('JAG1', 'NOTCH1'),      # Jagged1 - Notch1

            # Adhesion
            ('ICAM1', 'ITGAL'),      # ICAM-1 - LFA-1
            ('VCAM1', 'ITGA4'),      # VCAM-1 - VLA-4
        ]

        # Check which pairs are expressed
        interactions = []
        for ligand, receptor in lr_pairs:
            if ligand in self.adata.var_names and receptor in self.adata.var_names:
                # Calculate expression per cell type
                for source_type in self.adata.obs['cell_type'].unique():
                    source_mask = self.adata.obs['cell_type'] == source_type
                    ligand_expr = self.adata[source_mask, ligand].X.mean() if hasattr(self.adata[source_mask, ligand].X, 'mean') else np.mean(self.adata[source_mask, ligand].X.toarray())

                    for target_type in self.adata.obs['cell_type'].unique():
                        target_mask = self.adata.obs['cell_type'] == target_type
                        receptor_expr = self.adata[target_mask, receptor].X.mean() if hasattr(self.adata[target_mask, receptor].X, 'mean') else np.mean(self.adata[target_mask, receptor].X.toarray())

                        # Calculate interaction score (product of expressions)
                        score = float(ligand_expr) * float(receptor_expr)

                        if score > 0.01:  # Threshold for minimum expression
                            interactions.append({
                                'source': source_type,
                                'target': target_type,
                                'ligand': ligand,
                                'receptor': receptor,
                                'ligand_expr': float(ligand_expr),
                                'receptor_expr': float(receptor_expr),
                                'interaction_score': score,
                            })

        if interactions:
            interaction_df = pd.DataFrame(interactions)
            interaction_df = interaction_df.sort_values('interaction_score', ascending=False)
            self.save_csv(interaction_df.head(200), "cell_interactions.csv")

            self.interaction_results = {
                'method': 'simple_lr',
                'n_interactions': len(interaction_df),
                'n_pairs_checked': len(lr_pairs),
                'top_interactions': interaction_df.head(20).to_dict('records'),
            }

            self.logger.info(f"  Found {len(interaction_df)} ligand-receptor interactions")
        else:
            self.logger.info("  No significant interactions found")
            self.interaction_results = {'method': 'simple_lr', 'n_interactions': 0}

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
