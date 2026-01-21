"""
Single-Cell Agent 4: Trajectory & Dynamics Analysis

Pseudotime inference, trajectory analysis, and RNA velocity.

Input:
- adata_clustered.h5ad: Clustered data from Agent 2/3

Output:
- adata_trajectory.h5ad: Data with trajectory/pseudotime
- trajectory_results.json: Trajectory analysis summary
- figures/trajectory_*.png: Trajectory visualizations
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

# Import scvelo for RNA velocity (optional)
try:
    import scvelo as scv
    HAS_SCVELO = True
except ImportError:
    HAS_SCVELO = False

# Import paga
try:
    from scanpy.tools import paga
    HAS_PAGA = True
except ImportError:
    HAS_PAGA = True  # PAGA is part of scanpy


class SingleCellTrajectoryAgent(BaseAgent):
    """Agent 4: Trajectory & Dynamics Analysis."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Trajectory method
            "trajectory_method": "paga",  # paga, diffmap, or velocity
            "root_cluster": None,  # Auto-detect if None

            # PAGA parameters
            "paga_threshold": 0.05,

            # Diffusion map parameters
            "n_dcs": 15,

            # Pseudotime
            "compute_pseudotime": True,
            "pseudotime_method": "dpt",  # dpt (diffusion pseudotime)

            # RNA velocity (requires scvelo)
            "compute_velocity": False,  # Requires spliced/unspliced layers
            "velocity_mode": "stochastic",

            # Gene dynamics
            "compute_gene_dynamics": True,
            "n_top_genes_dynamics": 50,

            # Visualization
            "enable_plots": True,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent4_sc_trajectory", input_dir, output_dir, merged_config)

        self.adata = None
        self.trajectory_results = {}

    def validate_inputs(self) -> bool:
        """Validate input files."""
        if not HAS_SCANPY:
            self.logger.error("Scanpy not installed")
            return False

        # Load clustered data
        h5ad_file = self.input_dir / "adata_clustered.h5ad"
        if not h5ad_file.exists():
            h5ad_files = list(self.input_dir.glob("*.h5ad"))
            if h5ad_files:
                h5ad_file = h5ad_files[0]
            else:
                self.logger.error("No h5ad file found")
                return False

        self.adata = sc.read_h5ad(h5ad_file)
        self.logger.info(f"Loaded: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")

        # Check for UMAP
        if 'X_umap' not in self.adata.obsm:
            self.logger.warning("UMAP not found. Will compute.")

        return True

    def run(self) -> Dict[str, Any]:
        """Execute trajectory analysis pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Agent 4: Trajectory & Dynamics Analysis")
        self.logger.info("=" * 60)

        # Step 1: Ensure embeddings
        self._ensure_embeddings()

        # Step 2: Run PAGA (trajectory connectivity)
        self._run_paga()

        # Step 3: Compute diffusion map
        self._compute_diffusion_map()

        # Step 4: Compute pseudotime
        if self.config.get("compute_pseudotime", True):
            self._compute_pseudotime()

        # Step 5: RNA velocity (if available)
        if self.config.get("compute_velocity", False):
            self._compute_velocity()

        # Step 6: Gene dynamics along trajectory
        if self.config.get("compute_gene_dynamics", True):
            self._compute_gene_dynamics()

        # Step 7: Generate visualizations
        if self.config.get("enable_plots", True):
            self._generate_plots()

        # Save outputs
        self._save_outputs()

        return {
            "status": "success",
            "has_pseudotime": 'dpt_pseudotime' in self.adata.obs,
            "has_paga": 'paga' in self.adata.uns,
            "has_velocity": 'velocity' in self.adata.layers if hasattr(self.adata, 'layers') else False,
            "trajectory_results": self.trajectory_results,
            "output_file": str(self.output_dir / "adata_trajectory.h5ad")
        }

    def _ensure_embeddings(self):
        """Ensure necessary embeddings exist."""
        # Check neighbors
        if 'neighbors' not in self.adata.uns:
            self.logger.info("Computing neighbors...")
            if 'X_pca' not in self.adata.obsm:
                sc.tl.pca(self.adata)
            sc.pp.neighbors(self.adata)

        # Check UMAP
        if 'X_umap' not in self.adata.obsm:
            self.logger.info("Computing UMAP...")
            sc.tl.umap(self.adata)

    def _run_paga(self):
        """Run PAGA for trajectory connectivity."""
        self.logger.info("Running PAGA trajectory analysis...")

        try:
            # PAGA requires cluster information
            if 'cluster' not in self.adata.obs:
                self.logger.warning("No cluster column found. Skipping PAGA.")
                return

            # Run PAGA
            sc.tl.paga(self.adata, groups='cluster')

            # Get connectivity
            paga_connectivities = self.adata.uns['paga']['connectivities'].toarray()
            n_clusters = len(self.adata.obs['cluster'].unique())

            # Find strong connections
            threshold = self.config.get("paga_threshold", 0.05)
            connections = []

            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    if paga_connectivities[i, j] > threshold:
                        connections.append({
                            'source': str(i),
                            'target': str(j),
                            'weight': float(paga_connectivities[i, j])
                        })

            self.trajectory_results['paga'] = {
                'n_clusters': n_clusters,
                'n_connections': len(connections),
                'connections': connections,
                'threshold': threshold
            }

            self.logger.info(f"  PAGA found {len(connections)} cluster connections")

        except Exception as e:
            self.logger.warning(f"PAGA failed: {e}")

    def _compute_diffusion_map(self):
        """Compute diffusion map for trajectory."""
        self.logger.info("Computing diffusion map...")

        try:
            n_dcs = self.config.get("n_dcs", 15)
            sc.tl.diffmap(self.adata, n_comps=n_dcs)

            self.trajectory_results['diffmap'] = {
                'n_components': n_dcs,
                'computed': True
            }

            self.logger.info(f"  Computed {n_dcs} diffusion components")

        except Exception as e:
            self.logger.warning(f"Diffusion map failed: {e}")

    def _compute_pseudotime(self):
        """Compute pseudotime using diffusion pseudotime."""
        self.logger.info("Computing pseudotime...")

        try:
            # Determine root cell/cluster
            root_cluster = self.config.get("root_cluster")

            if root_cluster is None:
                # Auto-detect: use cluster with most stem-like signature
                root_cluster = self._detect_root_cluster()

            if root_cluster is not None:
                # Find a root cell in the root cluster
                root_cells = self.adata.obs[self.adata.obs['cluster'] == str(root_cluster)].index
                if len(root_cells) > 0:
                    # Use cell closest to cluster centroid
                    root_cell_idx = self._find_centroid_cell(root_cells)
                    self.adata.uns['iroot'] = root_cell_idx
                else:
                    # Use first cell
                    self.adata.uns['iroot'] = 0
            else:
                # Use first cell as root
                self.adata.uns['iroot'] = 0

            # Compute DPT
            sc.tl.dpt(self.adata)

            # Store results
            self.trajectory_results['pseudotime'] = {
                'method': 'dpt',
                'root_cluster': str(root_cluster) if root_cluster is not None else 'auto',
                'root_cell_idx': int(self.adata.uns.get('iroot', 0)),
                'pseudotime_range': [
                    float(self.adata.obs['dpt_pseudotime'].min()),
                    float(self.adata.obs['dpt_pseudotime'].max())
                ]
            }

            self.logger.info(f"  Pseudotime computed (root cluster: {root_cluster})")

        except Exception as e:
            self.logger.warning(f"Pseudotime computation failed: {e}")

    def _detect_root_cluster(self) -> Optional[str]:
        """Auto-detect root cluster based on stem/progenitor markers."""
        stem_markers = ['CD34', 'KIT', 'PROM1', 'THY1', 'ALDH1A1', 'SOX2', 'NANOG', 'POU5F1']

        best_cluster = None
        best_score = -1

        for cluster in self.adata.obs['cluster'].unique():
            cluster_cells = self.adata[self.adata.obs['cluster'] == cluster]

            # Score stem markers
            markers_present = [m for m in stem_markers if m in self.adata.var_names]
            if not markers_present:
                continue

            expr = cluster_cells[:, markers_present].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            score = np.mean(expr)

            if score > best_score:
                best_score = score
                best_cluster = cluster

        return best_cluster

    def _find_centroid_cell(self, cell_indices) -> int:
        """Find cell closest to cluster centroid."""
        if 'X_umap' in self.adata.obsm:
            coords = self.adata[cell_indices].obsm['X_umap']
        elif 'X_pca' in self.adata.obsm:
            coords = self.adata[cell_indices].obsm['X_pca'][:, :2]
        else:
            return 0

        centroid = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        local_idx = np.argmin(distances)

        # Convert to global index
        global_idx = np.where(self.adata.obs_names == cell_indices[local_idx])[0][0]
        return int(global_idx)

    def _compute_velocity(self):
        """Compute RNA velocity if scvelo is available."""
        if not HAS_SCVELO:
            self.logger.warning("scvelo not installed. Skipping RNA velocity.")
            return

        # Check for spliced/unspliced layers
        if 'spliced' not in self.adata.layers or 'unspliced' not in self.adata.layers:
            self.logger.warning("No spliced/unspliced layers. Skipping RNA velocity.")
            return

        self.logger.info("Computing RNA velocity...")

        try:
            # Filter and normalize for velocity
            scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=2000)
            scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)

            # Compute velocity
            mode = self.config.get("velocity_mode", "stochastic")
            scv.tl.velocity(self.adata, mode=mode)
            scv.tl.velocity_graph(self.adata)

            self.trajectory_results['velocity'] = {
                'mode': mode,
                'computed': True
            }

            self.logger.info(f"  RNA velocity computed (mode: {mode})")

        except Exception as e:
            self.logger.warning(f"RNA velocity failed: {e}")

    def _compute_gene_dynamics(self):
        """Identify genes with dynamic expression along trajectory."""
        if 'dpt_pseudotime' not in self.adata.obs:
            self.logger.warning("No pseudotime. Skipping gene dynamics.")
            return

        self.logger.info("Computing gene dynamics along trajectory...")

        try:
            # Get pseudotime
            pseudotime = self.adata.obs['dpt_pseudotime'].values

            # Remove cells with infinite pseudotime
            valid_mask = np.isfinite(pseudotime)
            if valid_mask.sum() < 100:
                self.logger.warning("Too few valid cells for gene dynamics.")
                return

            # Compute correlation with pseudotime for each gene
            n_genes = min(self.config.get("n_top_genes_dynamics", 50), self.adata.n_vars)

            correlations = []
            for gene in self.adata.var_names:
                expr = self.adata[:, gene].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                else:
                    expr = np.array(expr).flatten()

                # Compute Spearman correlation
                from scipy.stats import spearmanr
                corr, pval = spearmanr(pseudotime[valid_mask], expr[valid_mask])

                if np.isfinite(corr):
                    correlations.append({
                        'gene': gene,
                        'correlation': corr,
                        'pval': pval,
                        'direction': 'increasing' if corr > 0 else 'decreasing'
                    })

            # Sort by absolute correlation
            correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

            # Get top genes
            top_increasing = [g for g in correlations if g['correlation'] > 0][:n_genes // 2]
            top_decreasing = [g for g in correlations if g['correlation'] < 0][:n_genes // 2]

            self.trajectory_results['gene_dynamics'] = {
                'top_increasing': top_increasing,
                'top_decreasing': top_decreasing,
                'n_genes_analyzed': len(correlations)
            }

            self.logger.info(f"  Found {len(top_increasing)} increasing, {len(top_decreasing)} decreasing genes")

        except Exception as e:
            self.logger.warning(f"Gene dynamics failed: {e}")

    def _generate_plots(self):
        """Generate trajectory visualization plots."""
        if not HAS_MATPLOTLIB:
            return

        self.logger.info("Generating trajectory plots...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # PAGA graph
            if 'paga' in self.adata.uns:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.paga(
                    self.adata,
                    color='cluster',
                    ax=ax,
                    show=False,
                    threshold=self.config.get("paga_threshold", 0.05)
                )
                plt.tight_layout()
                plt.savefig(figures_dir / "trajectory_paga.png", dpi=150, bbox_inches='tight')
                plt.close()

            # UMAP with pseudotime
            if 'dpt_pseudotime' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(
                    self.adata,
                    color='dpt_pseudotime',
                    ax=ax,
                    show=False,
                    cmap='viridis'
                )
                plt.tight_layout()
                plt.savefig(figures_dir / "trajectory_pseudotime.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Diffusion map
            if 'X_diffmap' in self.adata.obsm:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.diffmap(
                    self.adata,
                    color='cluster',
                    ax=ax,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(figures_dir / "trajectory_diffmap.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Gene dynamics heatmap
            if 'gene_dynamics' in self.trajectory_results:
                dynamics = self.trajectory_results['gene_dynamics']
                top_genes = [g['gene'] for g in dynamics.get('top_increasing', [])[:10]]
                top_genes += [g['gene'] for g in dynamics.get('top_decreasing', [])[:10]]

                if top_genes and 'dpt_pseudotime' in self.adata.obs:
                    # Sort cells by pseudotime
                    valid_cells = self.adata.obs['dpt_pseudotime'].replace([np.inf, -np.inf], np.nan).dropna()
                    sorted_cells = valid_cells.sort_values().index

                    # Get expression matrix
                    genes_present = [g for g in top_genes if g in self.adata.var_names]
                    if genes_present:
                        expr = self.adata[sorted_cells, genes_present].X
                        if hasattr(expr, 'toarray'):
                            expr = expr.toarray()

                        fig, ax = plt.subplots(figsize=(12, 8))
                        im = ax.imshow(expr.T, aspect='auto', cmap='viridis')
                        ax.set_yticks(range(len(genes_present)))
                        ax.set_yticklabels(genes_present)
                        ax.set_xlabel('Cells (ordered by pseudotime)')
                        ax.set_ylabel('Genes')
                        ax.set_title('Gene Expression Along Trajectory')
                        plt.colorbar(im, label='Expression')
                        plt.tight_layout()
                        plt.savefig(figures_dir / "trajectory_gene_dynamics.png", dpi=150, bbox_inches='tight')
                        plt.close()

            # RNA velocity
            if HAS_SCVELO and 'velocity' in self.adata.layers:
                fig, ax = plt.subplots(figsize=(10, 8))
                scv.pl.velocity_embedding_stream(
                    self.adata,
                    basis='umap',
                    color='cluster',
                    ax=ax,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(figures_dir / "trajectory_velocity.png", dpi=150, bbox_inches='tight')
                plt.close()

            self.logger.info(f"  Saved plots to {figures_dir}")

        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")

    def _save_outputs(self):
        """Save output files."""
        self.logger.info("Saving outputs...")

        # Save h5ad with trajectory info
        output_h5ad = self.output_dir / "adata_trajectory.h5ad"
        self.adata.write_h5ad(output_h5ad)
        self.logger.info(f"  Saved: {output_h5ad}")

        # Save trajectory results
        results_json = self.output_dir / "trajectory_results.json"
        with open(results_json, 'w') as f:
            json.dump(self.trajectory_results, f, indent=2, default=str)
        self.logger.info(f"  Saved: {results_json}")

        # Save pseudotime values
        if 'dpt_pseudotime' in self.adata.obs:
            pseudotime_df = pd.DataFrame({
                'cell_id': self.adata.obs_names,
                'pseudotime': self.adata.obs['dpt_pseudotime'].values,
                'cluster': self.adata.obs['cluster'].values
            })
            pseudotime_df.to_csv(self.output_dir / "pseudotime_values.csv", index=False)

        self.logger.info("Trajectory analysis complete!")

    def validate_outputs(self) -> bool:
        """Validate output files were generated correctly."""
        required_files = [
            self.output_dir / "adata_trajectory.h5ad"
        ]
        for f in required_files:
            if not f.exists():
                self.logger.error(f"Required output missing: {f}")
                return False
        return True
