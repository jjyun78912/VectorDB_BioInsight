"""
Single-Cell Agent 5: CNV Inference & ML Prediction

Three ML predictions:
1. Cell Type Prediction (CellTypist) - Done in Agent 2
2. Cancer Type Prediction (Pseudo-bulk → CatBoost)
3. Malignant Cell Detection (CNV inference + classifier)

Input:
- adata_trajectory.h5ad or adata_clustered.h5ad

Output:
- adata_cnv.h5ad: Data with CNV scores and predictions
- cancer_prediction.json: Cancer type prediction results
- malignant_cells.csv: Malignant cell classification
- cnv_scores.csv: CNV scores per cell
- figures/cnv_*.png: CNV visualizations
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

# Import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import infercnvpy for CNV inference
try:
    import infercnvpy as cnv
    HAS_INFERCNV = True
except ImportError:
    HAS_INFERCNV = False

# Import ML predictor
try:
    from ...ml.pancancer_classifier import PanCancerClassifier
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False


# Chromosome information for CNV
CHROMOSOME_ORDER = [str(i) for i in range(1, 23)] + ['X', 'Y']


class SingleCellCNVMLAgent(BaseAgent):
    """Agent 5: CNV Inference & ML Prediction."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Cancer type prediction (ML #2)
            "enable_cancer_prediction": True,
            "model_dir": None,  # Path to pre-trained CatBoost model
            "pseudobulk_method": "mean",  # mean, median, sum

            # CNV inference
            "enable_cnv_inference": True,
            "reference_key": None,  # Column in obs for reference cells (e.g., 'cell_type')
            "reference_cat": None,  # Reference cell type (e.g., 'T_cells', 'Fibroblast')
            "cnv_window_size": 100,  # Window size for CNV smoothing

            # Malignant detection (ML #3)
            "enable_malignant_detection": True,
            "cnv_threshold": 0.1,  # Threshold for CNV-based malignancy

            # Visualization
            "enable_plots": True,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent5_sc_cnv_ml", input_dir, output_dir, merged_config)

        self.adata = None
        self.cancer_prediction = None
        self.cnv_results = None
        self.malignant_results = None

    def validate_inputs(self) -> bool:
        """Validate input files."""
        if not HAS_SCANPY:
            self.logger.error("Scanpy not installed")
            return False

        # Load data (prefer trajectory output)
        for name in ["adata_trajectory.h5ad", "adata_clustered.h5ad", "adata_qc.h5ad"]:
            h5ad_file = self.input_dir / name
            if h5ad_file.exists():
                self.adata = sc.read_h5ad(h5ad_file)
                self.logger.info(f"Loaded: {h5ad_file.name}")
                break

        if self.adata is None:
            h5ad_files = list(self.input_dir.glob("*.h5ad"))
            if h5ad_files:
                self.adata = sc.read_h5ad(h5ad_files[0])
            else:
                self.logger.error("No h5ad file found")
                return False

        self.logger.info(f"Data: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        return True

    def run(self) -> Dict[str, Any]:
        """Execute CNV and ML prediction pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Agent 5: CNV Inference & ML Prediction")
        self.logger.info("=" * 60)

        # ML #2: Cancer Type Prediction (Pseudo-bulk → CatBoost)
        if self.config.get("enable_cancer_prediction", True):
            self._predict_cancer_type()

        # CNV Inference
        if self.config.get("enable_cnv_inference", True):
            self._infer_cnv()

        # ML #3: Malignant Cell Detection
        if self.config.get("enable_malignant_detection", True):
            self._detect_malignant_cells()

        # Generate visualizations
        if self.config.get("enable_plots", True):
            self._generate_plots()

        # Save outputs
        self._save_outputs()

        return {
            "status": "success",
            "cancer_prediction": self.cancer_prediction,
            "cnv_computed": self.cnv_results is not None,
            "n_malignant_cells": self.malignant_results['n_malignant'] if self.malignant_results else 0,
            "output_file": str(self.output_dir / "adata_cnv.h5ad")
        }

    def _predict_cancer_type(self):
        """ML #2: Predict cancer type from pseudo-bulk expression."""
        self.logger.info("=" * 50)
        self.logger.info("ML #2: Cancer Type Prediction (Pseudo-bulk)")
        self.logger.info("=" * 50)

        try:
            # Create pseudo-bulk expression
            pseudobulk = self._create_pseudobulk()

            if pseudobulk is None:
                self.logger.warning("Could not create pseudo-bulk. Skipping.")
                return

            # Try to load pre-trained model
            if HAS_ML_PREDICTOR:
                self._predict_with_catboost(pseudobulk)
            else:
                self.logger.warning("ML predictor not available. Using signature-based prediction.")
                self._predict_with_signatures(pseudobulk)

        except Exception as e:
            self.logger.error(f"Cancer prediction failed: {e}")
            self.cancer_prediction = {"error": str(e)}

    def _create_pseudobulk(self) -> Optional[pd.Series]:
        """Create pseudo-bulk expression by aggregating cells."""
        self.logger.info("Creating pseudo-bulk expression...")

        method = self.config.get("pseudobulk_method", "mean")

        # Get expression matrix
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Aggregate
        if method == "mean":
            pseudobulk = np.mean(X, axis=0)
        elif method == "median":
            pseudobulk = np.median(X, axis=0)
        elif method == "sum":
            pseudobulk = np.sum(X, axis=0)
        else:
            pseudobulk = np.mean(X, axis=0)

        # Create Series with gene names
        pseudobulk_series = pd.Series(pseudobulk, index=self.adata.var_names)

        self.logger.info(f"  Created pseudo-bulk ({method}): {len(pseudobulk_series)} genes")

        return pseudobulk_series

    def _predict_with_catboost(self, pseudobulk: pd.Series):
        """Predict cancer type using pre-trained CatBoost model."""
        self.logger.info("Predicting with CatBoost model...")

        try:
            # Initialize classifier
            model_dir = self.config.get("model_dir")

            if model_dir:
                classifier = PanCancerClassifier(model_dir=Path(model_dir))
            else:
                # Try default model locations
                possible_paths = [
                    Path(__file__).parents[3] / "models" / "rnaseq" / "pancancer",  # VectorDB_BioInsight/models/rnaseq/pancancer
                    Path("/Users/admin/VectorDB_BioInsight/models/rnaseq/pancancer"),  # Absolute path
                ]

                default_model = None
                for path in possible_paths:
                    # Check for various model file patterns
                    model_files = [
                        path / "ensemble" / "catboost.cbm",
                        path / "model.cbm",
                        path / "catboost_model.cbm",
                    ]
                    if path.exists() and any(f.exists() for f in model_files):
                        default_model = path
                        break

                if default_model:
                    self.logger.info(f"  Found model at: {default_model}")
                    classifier = PanCancerClassifier(model_dir=default_model)
                else:
                    self.logger.warning("No model found. Using signature-based prediction.")
                    self._predict_with_signatures(pseudobulk)
                    return

            # Prepare input - predict() expects Gene x Sample format
            input_df = pd.DataFrame(pseudobulk.values, index=pseudobulk.index, columns=["pseudobulk"])

            # Predict - returns List[ClassificationResult]
            results = classifier.predict(input_df, sample_ids=["pseudobulk"])

            if results and len(results) > 0:
                result = results[0]  # Get first result

                # ClassificationResult attributes (check the actual dataclass fields)
                predicted_cancer = getattr(result, "predicted_cancer", None)
                confidence = getattr(result, "confidence", 0)
                top_k = getattr(result, "top_k_predictions", [])
                is_unknown = getattr(result, "is_unknown", False)

                # Convert top_k predictions to simple format
                top_predictions = []
                if top_k:
                    for pred in top_k:
                        if isinstance(pred, dict):
                            top_predictions.append(pred)
                        elif hasattr(pred, 'cancer_type'):
                            top_predictions.append({
                                'cancer_type': pred.cancer_type,
                                'probability': pred.probability
                            })

                self.cancer_prediction = {
                    "method": "catboost",
                    "predicted_type": predicted_cancer if predicted_cancer and not is_unknown else "Unknown",
                    "confidence": float(confidence) if confidence else 0.0,
                    "top_predictions": top_predictions,
                    "is_unknown": is_unknown,
                    "warning": "예측 결과는 진단이 아니며 참고용입니다."
                }

                self.logger.info(f"  Predicted: {self.cancer_prediction['predicted_type']} "
                               f"({self.cancer_prediction['confidence']*100:.1f}%)")
            else:
                self.logger.warning("No prediction results returned")
                self._predict_with_signatures(pseudobulk)

        except Exception as e:
            self.logger.warning(f"CatBoost prediction failed: {e}")
            self._predict_with_signatures(pseudobulk)

    def _predict_with_signatures(self, pseudobulk: pd.Series):
        """Fallback: Predict cancer type using gene signatures."""
        self.logger.info("Using signature-based prediction...")

        # Cancer-specific marker genes
        cancer_signatures = {
            "BRCA": ["ESR1", "PGR", "ERBB2", "GATA3", "FOXA1"],
            "LUAD": ["TTF1", "NKX2-1", "NAPSA", "SFTPC", "KRT7"],
            "LUSC": ["TP63", "KRT5", "KRT6A", "SOX2", "SERPINB3"],
            "COAD": ["CDX2", "VILLIN", "CK20", "MUC2", "CDH17"],
            "LIHC": ["AFP", "GPC3", "HNF4A", "ALB", "APOA1"],
            "PAAD": ["PDX1", "MUC1", "CA19-9", "CEA", "KRAS"],
            "PRAD": ["KLK3", "NKX3-1", "AR", "FOLH1", "TMPRSS2"],
            "KIRC": ["CA9", "PAX8", "PAX2", "VIM", "VEGFA"],
            "GBM": ["GFAP", "OLIG2", "SOX2", "EGFR", "IDH1"],
            "OV": ["PAX8", "WT1", "CA125", "MUC16", "HE4"],
            "SKCM": ["MLANA", "MITF", "TYR", "S100B", "SOX10"],
        }

        scores = {}
        for cancer, markers in cancer_signatures.items():
            markers_present = [m for m in markers if m in pseudobulk.index]
            if markers_present:
                score = pseudobulk[markers_present].mean()
                scores[cancer] = float(score)

        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_cancer = sorted_scores[0][0]
            top_score = sorted_scores[0][1]

            self.cancer_prediction = {
                "method": "signature",
                "predicted_type": top_cancer,
                "confidence": min(top_score / 10, 1.0),  # Normalize
                "all_scores": dict(sorted_scores[:5]),
                "warning": "시그니처 기반 예측은 정확도가 낮을 수 있습니다."
            }

            self.logger.info(f"  Predicted (signature): {top_cancer}")
        else:
            self.cancer_prediction = {
                "method": "signature",
                "predicted_type": "Unknown",
                "confidence": 0,
                "warning": "시그니처 유전자가 충분하지 않습니다."
            }

    def _infer_cnv(self):
        """Infer copy number variations from expression."""
        self.logger.info("=" * 50)
        self.logger.info("CNV Inference")
        self.logger.info("=" * 50)

        if not HAS_INFERCNV:
            self.logger.warning("infercnvpy not installed. Using simplified CNV inference.")
            self._simplified_cnv_inference()
            return

        try:
            # Determine reference cells
            ref_key = self.config.get("reference_key")
            ref_cat = self.config.get("reference_cat")

            if ref_key and ref_cat and ref_key in self.adata.obs:
                reference_cells = self.adata.obs[ref_key] == ref_cat
                self.logger.info(f"  Using {reference_cells.sum()} reference cells ({ref_cat})")
            else:
                # Auto-detect: use immune cells as reference
                reference_cells = self._auto_detect_reference()

            # Add chromosome information to genes
            self._add_chromosome_info()

            # Run inferCNVpy
            cnv.tl.infercnv(
                self.adata,
                reference_key=ref_key,
                reference_cat=ref_cat,
                window_size=self.config.get("cnv_window_size", 100)
            )

            # Calculate CNV scores
            cnv.tl.cnv_score(self.adata)

            self.cnv_results = {
                "method": "infercnvpy",
                "reference_type": ref_cat or "auto",
                "n_reference_cells": int(reference_cells.sum()) if reference_cells is not None else 0
            }

            self.logger.info("  CNV inference complete")

        except Exception as e:
            self.logger.warning(f"InferCNV failed: {e}")
            self._simplified_cnv_inference()

    def _simplified_cnv_inference(self):
        """Simplified CNV inference without infercnvpy."""
        self.logger.info("Running simplified CNV inference...")

        try:
            # Add chromosome info
            self._add_chromosome_info()

            # Get genes with chromosome info
            if 'chromosome' not in self.adata.var:
                self.logger.warning("No chromosome info. Skipping CNV.")
                return

            # Filter to only mapped genes (exclude 'unknown')
            mapped_mask = self.adata.var['chromosome'] != 'unknown'
            n_mapped = mapped_mask.sum()

            if n_mapped < 10:
                self.logger.warning(f"  Only {n_mapped} genes mapped to chromosomes. Skipping CNV.")
                # Set default scores
                self.adata.obs['cnv_score'] = 0.0
                self.cnv_results = {
                    "method": "simplified",
                    "mean_score": 0.0,
                    "std_score": 0.0,
                    "note": "Insufficient chromosome-mapped genes"
                }
                return

            # Group genes by chromosome (excluding unknown)
            mapped_var = self.adata.var[mapped_mask]
            chr_genes = mapped_var.groupby('chromosome').groups

            # Calculate per-chromosome expression deviation
            X = self.adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()

            # Global mean per gene
            gene_means = np.mean(X, axis=0)

            # CNV score: deviation from expected
            cnv_scores = np.zeros(self.adata.n_obs)
            n_chrom_used = 0

            for chrom in CHROMOSOME_ORDER:
                if chrom not in chr_genes:
                    continue

                gene_idx = [self.adata.var_names.get_loc(g) for g in chr_genes[chrom]
                           if g in self.adata.var_names]

                if not gene_idx:
                    continue

                # Chromosome expression
                chr_expr = X[:, gene_idx]
                chr_mean = gene_means[gene_idx]

                # Deviation (amplification or deletion)
                deviation = np.abs(np.mean(chr_expr, axis=1) - np.mean(chr_mean))
                cnv_scores += deviation
                n_chrom_used += 1

            # Normalize
            if n_chrom_used > 0:
                cnv_scores = cnv_scores / n_chrom_used
            else:
                cnv_scores = np.zeros(self.adata.n_obs)

            self.adata.obs['cnv_score'] = cnv_scores

            self.cnv_results = {
                "method": "simplified",
                "mean_score": float(np.nanmean(cnv_scores)),
                "std_score": float(np.nanstd(cnv_scores)),
                "n_chromosomes_used": n_chrom_used,
                "n_genes_mapped": int(n_mapped)
            }

            self.logger.info(f"  CNV inference: {n_chrom_used} chromosomes, {n_mapped} genes")
            self.logger.info(f"  CNV scores: mean={np.nanmean(cnv_scores):.4f}, std={np.nanstd(cnv_scores):.4f}")

        except Exception as e:
            self.logger.warning(f"Simplified CNV failed: {e}")
            import traceback
            traceback.print_exc()

    def _add_chromosome_info(self):
        """Add chromosome information to genes using built-in mapping."""
        if 'chromosome' in self.adata.var:
            return

        # Common human gene-chromosome mapping for cancer-related genes
        # This covers key genes for CNV analysis
        gene_chr_map = {
            # Chromosome 1
            'TP73': '1', 'NRAS': '1', 'BCL9': '1', 'MCL1': '1', 'NOTCH2': '1',
            # Chromosome 2
            'ALK': '2', 'MYCN': '2', 'MSH2': '2', 'MSH6': '2', 'IDH1': '2',
            # Chromosome 3
            'VHL': '3', 'BAP1': '3', 'SETD2': '3', 'PBRM1': '3', 'MLH1': '3',
            # Chromosome 4
            'FGFR3': '4', 'KIT': '4', 'PDGFRA': '4', 'KDR': '4',
            # Chromosome 5
            'APC': '5', 'TERT': '5', 'PIK3R1': '5', 'NPM1': '5',
            # Chromosome 6
            'VEGFA': '6', 'CDKN1A': '6', 'ESR1': '6', 'HLA-A': '6', 'HLA-B': '6',
            # Chromosome 7
            'EGFR': '7', 'MET': '7', 'BRAF': '7', 'CDK6': '7', 'EZH2': '7',
            # Chromosome 8
            'MYC': '8', 'FGFR1': '8', 'RECQL4': '8',
            # Chromosome 9
            'CDKN2A': '9', 'CDKN2B': '9', 'JAK2': '9', 'NOTCH1': '9', 'ABL1': '9',
            # Chromosome 10
            'PTEN': '10', 'FGFR2': '10', 'RET': '10', 'MGMT': '10',
            # Chromosome 11
            'ATM': '11', 'CCND1': '11', 'WT1': '11', 'FGF3': '11', 'FGF4': '11',
            # Chromosome 12
            'KRAS': '12', 'CDK4': '12', 'MDM2': '12', 'ERBB3': '12',
            # Chromosome 13
            'RB1': '13', 'BRCA2': '13', 'FLT3': '13',
            # Chromosome 14
            'AKT1': '14', 'TSC2': '14', 'MAX': '14',
            # Chromosome 15
            'BLM': '15', 'IDH2': '15', 'B2M': '15',
            # Chromosome 16
            'TSC1': '16', 'PALB2': '16', 'CREBBP': '16', 'CBFB': '16',
            # Chromosome 17
            'TP53': '17', 'ERBB2': '17', 'BRCA1': '17', 'NF1': '17', 'RNF43': '17', 'MAP2K4': '17',
            # Chromosome 18
            'SMAD4': '18', 'BCL2': '18', 'DCC': '18',
            # Chromosome 19
            'STK11': '19', 'AKT2': '19', 'CEBPA': '19', 'JAK3': '19',
            # Chromosome 20
            'AURKA': '20', 'SRC': '20', 'TOP1': '20', 'GNAS': '20',
            # Chromosome 21
            'RUNX1': '21', 'ERG': '21', 'TMPRSS2': '21',
            # Chromosome 22
            'NF2': '22', 'CHEK2': '22', 'SMARCB1': '22', 'BCR': '22',
            # Chromosome X
            'AR': 'X', 'ATRX': 'X', 'KDM6A': 'X', 'PHF6': 'X',
        }

        # Map genes
        self.adata.var['chromosome'] = self.adata.var_names.map(
            lambda x: gene_chr_map.get(x, 'unknown')
        )

        # Count mapped genes
        n_mapped = (self.adata.var['chromosome'] != 'unknown').sum()
        self.logger.info(f"  Chromosome mapping: {n_mapped}/{self.adata.n_vars} genes mapped")

    def _auto_detect_reference(self) -> Optional[pd.Series]:
        """Auto-detect reference (non-malignant) cells."""
        if 'cell_type' not in self.adata.obs:
            return None

        # Common reference cell types (immune/stromal)
        reference_types = ['T_cells', 'T_cell', 'B_cells', 'B_cell', 'NK_cells',
                          'Fibroblast', 'Endothelial', 'Immune']

        for ref_type in reference_types:
            if ref_type in self.adata.obs['cell_type'].values:
                return self.adata.obs['cell_type'] == ref_type

        return None

    def _detect_malignant_cells(self):
        """ML #3: Detect malignant cells based on CNV and markers."""
        self.logger.info("=" * 50)
        self.logger.info("ML #3: Malignant Cell Detection")
        self.logger.info("=" * 50)

        try:
            # Initialize malignant score
            malignant_score = np.zeros(self.adata.n_obs)

            # 1. CNV-based score
            if 'cnv_score' in self.adata.obs:
                cnv_score = self.adata.obs['cnv_score'].values
                # Normalize
                cnv_score_norm = (cnv_score - cnv_score.min()) / (cnv_score.max() - cnv_score.min() + 1e-10)
                malignant_score += cnv_score_norm * 0.5  # 50% weight

            # 2. Cancer marker expression
            cancer_markers = ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'CCNB1',  # Proliferation
                            'EPCAM', 'KRT18', 'KRT19',  # Epithelial
                            'CD44', 'ALDH1A1', 'PROM1']  # Stemness

            markers_present = [m for m in cancer_markers if m in self.adata.var_names]
            if markers_present:
                marker_expr = self.adata[:, markers_present].X
                if hasattr(marker_expr, 'toarray'):
                    marker_expr = marker_expr.toarray()
                marker_score = np.mean(marker_expr, axis=1)
                marker_score_norm = (marker_score - marker_score.min()) / (marker_score.max() - marker_score.min() + 1e-10)
                malignant_score += marker_score_norm * 0.3  # 30% weight

            # 3. Cell type-based score
            if 'cell_type' in self.adata.obs:
                # Non-immune epithelial cells are more likely malignant in tumors
                immune_types = ['T_cells', 'B_cells', 'NK_cells', 'Monocyte', 'Macrophage',
                               'Dendritic', 'Mast', 'Neutrophil']
                is_immune = self.adata.obs['cell_type'].str.contains('|'.join(immune_types), case=False, na=False)
                malignant_score[~is_immune] += 0.2  # 20% weight for non-immune

            # Store scores
            self.adata.obs['malignant_score'] = malignant_score

            # Classify
            threshold = self.config.get("cnv_threshold", 0.1)
            # Use higher threshold for combined score
            classification_threshold = 0.5

            self.adata.obs['is_malignant'] = malignant_score > classification_threshold

            n_malignant = self.adata.obs['is_malignant'].sum()
            pct_malignant = n_malignant / self.adata.n_obs * 100

            self.malignant_results = {
                "n_malignant": int(n_malignant),
                "n_normal": int(self.adata.n_obs - n_malignant),
                "pct_malignant": float(pct_malignant),
                "threshold": classification_threshold,
                "score_components": ["cnv", "markers", "cell_type"]
            }

            self.logger.info(f"  Malignant cells: {n_malignant} ({pct_malignant:.1f}%)")

            # Per-cluster breakdown
            if 'cluster' in self.adata.obs:
                cluster_malignant = self.adata.obs.groupby('cluster')['is_malignant'].mean()
                for cluster, pct in cluster_malignant.items():
                    self.logger.info(f"    Cluster {cluster}: {pct*100:.1f}% malignant")

        except Exception as e:
            self.logger.error(f"Malignant detection failed: {e}")
            self.malignant_results = {"error": str(e)}

    def _generate_plots(self):
        """Generate CNV and ML visualization plots."""
        if not HAS_MATPLOTLIB:
            return

        self.logger.info("Generating plots...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # CNV score UMAP
            if 'cnv_score' in self.adata.obs and 'X_umap' in self.adata.obsm:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(self.adata, color='cnv_score', ax=ax, show=False, cmap='Reds')
                plt.title('CNV Score')
                plt.tight_layout()
                plt.savefig(figures_dir / "cnv_score_umap.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Malignant cells UMAP
            if 'is_malignant' in self.adata.obs and 'X_umap' in self.adata.obsm:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(self.adata, color='is_malignant', ax=ax, show=False)
                plt.title('Malignant Cell Classification')
                plt.tight_layout()
                plt.savefig(figures_dir / "malignant_umap.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Malignant score distribution
            if 'malignant_score' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(self.adata.obs['malignant_score'], bins=50, edgecolor='black')
                ax.axvline(0.5, color='red', linestyle='--', label='Threshold')
                ax.set_xlabel('Malignant Score')
                ax.set_ylabel('Number of Cells')
                ax.set_title('Malignant Score Distribution')
                ax.legend()
                plt.tight_layout()
                plt.savefig(figures_dir / "malignant_score_dist.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Cancer prediction summary (if available)
            if self.cancer_prediction and 'all_scores' in self.cancer_prediction:
                scores = self.cancer_prediction['all_scores']
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(list(scores.keys()), list(scores.values()), color='steelblue')
                ax.set_xlabel('Score')
                ax.set_title('Cancer Type Prediction Scores')
                plt.tight_layout()
                plt.savefig(figures_dir / "cancer_prediction.png", dpi=150, bbox_inches='tight')
                plt.close()

            self.logger.info(f"  Saved plots to {figures_dir}")

        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")

    def _save_outputs(self):
        """Save output files."""
        self.logger.info("Saving outputs...")

        # Save h5ad
        output_h5ad = self.output_dir / "adata_cnv.h5ad"
        self.adata.write_h5ad(output_h5ad)
        self.logger.info(f"  Saved: {output_h5ad}")

        # Save cancer prediction
        if self.cancer_prediction:
            pred_json = self.output_dir / "cancer_prediction.json"
            with open(pred_json, 'w') as f:
                json.dump(self.cancer_prediction, f, indent=2)
            self.logger.info(f"  Saved: {pred_json}")

        # Save CNV results
        if self.cnv_results:
            cnv_json = self.output_dir / "cnv_results.json"
            with open(cnv_json, 'w') as f:
                json.dump(self.cnv_results, f, indent=2)

        # Save CNV scores
        if 'cnv_score' in self.adata.obs:
            cnv_df = pd.DataFrame({
                'cell_id': self.adata.obs_names,
                'cnv_score': self.adata.obs['cnv_score'].values,
                'cluster': self.adata.obs.get('cluster', 'NA')
            })
            cnv_df.to_csv(self.output_dir / "cnv_scores.csv", index=False)

        # Save malignant classification
        if 'is_malignant' in self.adata.obs:
            malignant_df = pd.DataFrame({
                'cell_id': self.adata.obs_names,
                'malignant_score': self.adata.obs['malignant_score'].values,
                'is_malignant': self.adata.obs['is_malignant'].values,
                'cluster': self.adata.obs.get('cluster', 'NA'),
                'cell_type': self.adata.obs.get('cell_type', 'NA')
            })
            malignant_df.to_csv(self.output_dir / "malignant_cells.csv", index=False)
            self.logger.info(f"  Saved: malignant_cells.csv")

        # Save malignant results summary
        if self.malignant_results:
            mal_json = self.output_dir / "malignant_results.json"
            with open(mal_json, 'w') as f:
                json.dump(self.malignant_results, f, indent=2)

        self.logger.info("CNV & ML prediction complete!")

    def validate_outputs(self) -> bool:
        """Validate output files were generated correctly."""
        required_files = [
            self.output_dir / "adata_cnv.h5ad"
        ]
        for f in required_files:
            if not f.exists():
                self.logger.error(f"Required output missing: {f}")
                return False
        return True
