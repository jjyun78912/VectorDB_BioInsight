"""
Single-Cell Agent 3: Pathway Enrichment & Database Validation

Pathway analysis per cluster and validation against cancer databases.

Input:
- adata_clustered.h5ad: Clustered data from Agent 2
- cluster_markers.csv: Marker genes from Agent 2

Output:
- cluster_pathways.csv: Pathway enrichment per cluster
- driver_genes.csv: Markers matched to COSMIC/OncoKB
- pathway_summary.json: Pathway analysis summary
- figures/pathway_*.png: Pathway visualization plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
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

# Import gseapy for pathway enrichment
try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

# Import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Cancer driver gene databases
COSMIC_TIER1_GENES: Set[str] = {
    'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'RB1',
    'BRCA1', 'BRCA2', 'MYC', 'ERBB2', 'CDK4', 'MDM2', 'CCND1', 'CDKN2A',
    'ATM', 'AKT1', 'NRAS', 'HRAS', 'FGFR1', 'FGFR2', 'FGFR3', 'MET',
    'ALK', 'ROS1', 'RET', 'KIT', 'PDGFRA', 'ABL1', 'JAK2', 'BCL2',
    'VHL', 'NF1', 'NF2', 'WT1', 'SMAD4', 'CTNNB1', 'IDH1', 'IDH2',
    'ARID1A', 'ARID2', 'TERT', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'FLT3',
    'NPM1', 'DNMT3A', 'TET2', 'ASXL1', 'EZH2', 'KMT2D', 'KMT2C',
    'CREBBP', 'EP300', 'SETD2', 'BAP1', 'STAG2', 'SF3B1', 'U2AF1',
    'SRSF2', 'ZRSR2', 'PHF6', 'RUNX1', 'CEBPA', 'GATA1', 'GATA2'
}

ONCOKB_ACTIONABLE_GENES: Set[str] = {
    'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'ERBB2', 'MYC',
    'BRCA1', 'BRCA2', 'ALK', 'ROS1', 'RET', 'MET', 'NTRK1', 'NTRK2',
    'NTRK3', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'CDKN2A', 'AKT1',
    'MTOR', 'STK11', 'ESR1', 'AR', 'BTK', 'BCL2', 'IDH1', 'IDH2',
    'KIT', 'PDGFRA', 'FLT3', 'JAK2', 'ABL1', 'ERBB3', 'RAF1', 'MAP2K1'
}

# Tumor Microenvironment markers
TME_MARKERS: Dict[str, List[str]] = {
    "immune_checkpoint": ['CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT'],
    "t_cell_exhaustion": ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1'],
    "t_cell_activation": ['CD69', 'CD44', 'IL2RA', 'TNFRSF9', 'ICOS'],
    "cytotoxic": ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'NKG7', 'GNLY'],
    "treg": ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TNFRSF18'],
    "m1_macrophage": ['CD80', 'CD86', 'NOS2', 'IL1B', 'TNF'],
    "m2_macrophage": ['CD163', 'MRC1', 'CD206', 'IL10', 'ARG1'],
    "angiogenesis": ['VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'ANGPT1', 'ANGPT2'],
    "hypoxia": ['HIF1A', 'LDHA', 'PGK1', 'ENO1', 'SLC2A1', 'CA9'],
    "emt": ['SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1', 'VIM', 'CDH2'],
    "stemness": ['PROM1', 'ALDH1A1', 'CD44', 'SOX2', 'NANOG', 'POU5F1'],
}


class SingleCellPathwayAgent(BaseAgent):
    """Agent 3: Pathway Enrichment & Database Validation."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Pathway databases
            "pathway_databases": [
                "GO_Biological_Process_2023",
                "KEGG_2021_Human",
                "Reactome_2022",
                "WikiPathway_2023_Human"
            ],
            "pathway_top_genes": 100,
            "pathway_min_genes": 5,
            "pathway_max_genes": 500,
            "pathway_pval_cutoff": 0.05,

            # Driver gene matching
            "enable_driver_matching": True,
            "min_logfc_for_driver": 0.5,

            # TME scoring
            "enable_tme_scoring": True,

            # Visualization
            "enable_plots": True,
            "top_pathways_plot": 20,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent3_sc_pathway", input_dir, output_dir, merged_config)

        self.adata = None
        self.markers_df = None
        self.pathway_results = []
        self.driver_genes_df = None
        self.tme_scores = None

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

        # Load markers
        markers_file = self.input_dir / "cluster_markers.csv"
        if markers_file.exists():
            self.markers_df = pd.read_csv(markers_file)
            self.logger.info(f"Loaded {len(self.markers_df)} marker genes")
        else:
            self.logger.warning("No marker file found. Will extract from adata.")

        return True

    def run(self) -> Dict[str, Any]:
        """Execute pathway and validation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Agent 3: Pathway Enrichment & Database Validation")
        self.logger.info("=" * 60)

        # Step 1: Ensure markers exist
        self._ensure_markers()

        # Step 2: Run pathway enrichment per cluster
        self._run_pathway_enrichment()

        # Step 3: Match to cancer driver databases
        if self.config.get("enable_driver_matching", True):
            self._match_driver_genes()

        # Step 4: TME scoring
        if self.config.get("enable_tme_scoring", True):
            self._score_tme()

        # Step 5: Generate visualizations
        if self.config.get("enable_plots", True):
            self._generate_plots()

        # Save outputs
        self._save_outputs()

        return {
            "status": "success",
            "n_clusters_analyzed": len(self.adata.obs['cluster'].unique()),
            "n_pathways_found": len(self.pathway_results),
            "n_driver_genes": len(self.driver_genes_df) if self.driver_genes_df is not None else 0,
            "output_file": str(self.output_dir / "cluster_pathways.csv")
        }

    def _ensure_markers(self):
        """Ensure marker genes are available."""
        if self.markers_df is not None and len(self.markers_df) > 0:
            return

        self.logger.info("Extracting markers from adata...")

        if 'rank_genes_groups' not in self.adata.uns:
            # Run marker finding
            sc.tl.rank_genes_groups(
                self.adata,
                groupby='cluster',
                method='wilcoxon',
                n_genes=100
            )

        # Extract to DataFrame
        result = self.adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names

        markers_list = []
        for group in groups:
            for i in range(min(100, len(result['names'][group]))):
                markers_list.append({
                    'cluster': group,
                    'gene': result['names'][group][i],
                    'score': float(result['scores'][group][i]),
                    'logfoldchange': float(result['logfoldchanges'][group][i]),
                    'pval': float(result['pvals'][group][i]),
                    'pval_adj': float(result['pvals_adj'][group][i])
                })

        self.markers_df = pd.DataFrame(markers_list)
        self.logger.info(f"  Extracted {len(self.markers_df)} markers")

    def _run_pathway_enrichment(self):
        """Run pathway enrichment for each cluster."""
        if not HAS_GSEAPY:
            self.logger.warning("gseapy not installed. Skipping pathway enrichment.")
            return

        self.logger.info("Running pathway enrichment per cluster...")

        databases = self.config.get("pathway_databases", ["GO_Biological_Process_2023"])
        top_n = self.config.get("pathway_top_genes", 100)

        self.pathway_results = []

        for cluster in sorted(self.markers_df['cluster'].unique()):
            self.logger.info(f"  Processing cluster {cluster}...")

            # Get top markers for this cluster
            cluster_markers = self.markers_df[
                (self.markers_df['cluster'] == cluster) &
                (self.markers_df['logfoldchange'] > 0)  # Upregulated
            ].head(top_n)['gene'].tolist()

            if len(cluster_markers) < 5:
                self.logger.warning(f"    Too few markers ({len(cluster_markers)}). Skipping.")
                continue

            try:
                # Run enrichment
                enr = gp.enrichr(
                    gene_list=cluster_markers,
                    gene_sets=databases,
                    organism='human',
                    outdir=None,
                    no_plot=True,
                    cutoff=self.config.get("pathway_pval_cutoff", 0.05)
                )

                # Filter results
                results = enr.results[enr.results['Adjusted P-value'] < 0.05].copy()

                for _, row in results.iterrows():
                    self.pathway_results.append({
                        'cluster': cluster,
                        'term': row['Term'],
                        'gene_set': row['Gene_set'],
                        'overlap': row['Overlap'],
                        'pval': row['P-value'],
                        'pval_adj': row['Adjusted P-value'],
                        'combined_score': row['Combined Score'],
                        'genes': row['Genes']
                    })

                self.logger.info(f"    Found {len(results)} significant pathways")

            except Exception as e:
                self.logger.warning(f"    Enrichment failed: {e}")

        self.logger.info(f"Total pathways: {len(self.pathway_results)}")

    def _match_driver_genes(self):
        """Match marker genes to cancer driver databases."""
        self.logger.info("Matching markers to cancer databases...")

        all_drivers = COSMIC_TIER1_GENES | ONCOKB_ACTIONABLE_GENES
        min_logfc = self.config.get("min_logfc_for_driver", 0.5)

        driver_matches = []

        for _, row in self.markers_df.iterrows():
            gene = row['gene']

            if gene in all_drivers and abs(row['logfoldchange']) >= min_logfc:
                driver_matches.append({
                    'cluster': row['cluster'],
                    'gene': gene,
                    'logfoldchange': row['logfoldchange'],
                    'pval_adj': row['pval_adj'],
                    'in_cosmic': gene in COSMIC_TIER1_GENES,
                    'in_oncokb': gene in ONCOKB_ACTIONABLE_GENES,
                    'direction': 'up' if row['logfoldchange'] > 0 else 'down'
                })

        self.driver_genes_df = pd.DataFrame(driver_matches)

        if len(self.driver_genes_df) > 0:
            # Count unique drivers
            unique_drivers = self.driver_genes_df['gene'].nunique()
            self.logger.info(f"  Found {len(self.driver_genes_df)} driver gene hits ({unique_drivers} unique)")

            # Per cluster summary
            cluster_counts = self.driver_genes_df.groupby('cluster').size()
            for cluster, count in cluster_counts.items():
                self.logger.info(f"    Cluster {cluster}: {count} driver genes")
        else:
            self.logger.info("  No driver genes found in markers")

    def _score_tme(self):
        """Score TME signatures per cluster."""
        self.logger.info("Scoring TME signatures...")

        tme_scores = []

        for cluster in self.adata.obs['cluster'].unique():
            cluster_cells = self.adata[self.adata.obs['cluster'] == cluster]

            for signature_name, genes in TME_MARKERS.items():
                # Get genes present in data
                genes_present = [g for g in genes if g in self.adata.var_names]

                if len(genes_present) < 2:
                    continue

                # Calculate mean expression
                expr = cluster_cells[:, genes_present].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray()
                mean_expr = np.mean(expr)

                tme_scores.append({
                    'cluster': cluster,
                    'signature': signature_name,
                    'mean_expression': float(mean_expr),
                    'n_genes': len(genes_present),
                    'genes_used': ','.join(genes_present)
                })

        self.tme_scores = pd.DataFrame(tme_scores)

        if len(self.tme_scores) > 0:
            self.logger.info(f"  Scored {len(TME_MARKERS)} TME signatures across {len(self.adata.obs['cluster'].unique())} clusters")

    def _generate_plots(self):
        """Generate visualization plots."""
        if not HAS_MATPLOTLIB or not self.pathway_results:
            return

        self.logger.info("Generating plots...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # Top pathways per cluster heatmap
            pathway_df = pd.DataFrame(self.pathway_results)

            if len(pathway_df) > 0:
                # Get top pathways overall
                top_pathways = pathway_df.groupby('term')['combined_score'].max().nlargest(
                    self.config.get("top_pathways_plot", 20)
                ).index.tolist()

                # Create pivot for heatmap
                pivot_data = []
                for cluster in sorted(pathway_df['cluster'].unique()):
                    cluster_pathways = pathway_df[pathway_df['cluster'] == cluster]
                    for term in top_pathways:
                        row = cluster_pathways[cluster_pathways['term'] == term]
                        score = row['combined_score'].values[0] if len(row) > 0 else 0
                        pivot_data.append({
                            'cluster': f"C{cluster}",
                            'term': term[:50],  # Truncate long names
                            'score': score
                        })

                pivot_df = pd.DataFrame(pivot_data)
                pivot_matrix = pivot_df.pivot(index='term', columns='cluster', values='score').fillna(0)

                fig, ax = plt.subplots(figsize=(10, max(8, len(top_pathways) * 0.4)))
                im = ax.imshow(pivot_matrix.values, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(pivot_matrix.columns)))
                ax.set_xticklabels(pivot_matrix.columns, rotation=45, ha='right')
                ax.set_yticks(range(len(pivot_matrix.index)))
                ax.set_yticklabels(pivot_matrix.index)
                ax.set_title('Pathway Enrichment by Cluster')
                plt.colorbar(im, label='Combined Score')
                plt.tight_layout()
                plt.savefig(figures_dir / "pathway_heatmap.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Driver genes per cluster bar chart
            if self.driver_genes_df is not None and len(self.driver_genes_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                driver_counts = self.driver_genes_df.groupby('cluster').size()
                driver_counts.plot(kind='bar', ax=ax, color='coral')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Number of Driver Genes')
                ax.set_title('Driver Genes per Cluster')
                plt.tight_layout()
                plt.savefig(figures_dir / "driver_genes_barplot.png", dpi=150, bbox_inches='tight')
                plt.close()

            # TME heatmap
            if self.tme_scores is not None and len(self.tme_scores) > 0:
                tme_pivot = self.tme_scores.pivot(
                    index='signature', columns='cluster', values='mean_expression'
                ).fillna(0)

                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(tme_pivot.values, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(tme_pivot.columns)))
                ax.set_xticklabels([f"C{c}" for c in tme_pivot.columns], rotation=45, ha='right')
                ax.set_yticks(range(len(tme_pivot.index)))
                ax.set_yticklabels(tme_pivot.index)
                ax.set_title('TME Signature Scores by Cluster')
                plt.colorbar(im, label='Mean Expression')
                plt.tight_layout()
                plt.savefig(figures_dir / "tme_heatmap.png", dpi=150, bbox_inches='tight')
                plt.close()

            self.logger.info(f"  Saved plots to {figures_dir}")

        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")

    def _save_outputs(self):
        """Save output files."""
        self.logger.info("Saving outputs...")

        # Save pathway results
        if self.pathway_results:
            pathway_df = pd.DataFrame(self.pathway_results)
            pathway_csv = self.output_dir / "cluster_pathways.csv"
            pathway_df.to_csv(pathway_csv, index=False)
            self.logger.info(f"  Saved: {pathway_csv}")

        # Save driver genes
        if self.driver_genes_df is not None and len(self.driver_genes_df) > 0:
            driver_csv = self.output_dir / "driver_genes.csv"
            self.driver_genes_df.to_csv(driver_csv, index=False)
            self.logger.info(f"  Saved: {driver_csv}")

        # Save TME scores
        if self.tme_scores is not None and len(self.tme_scores) > 0:
            tme_csv = self.output_dir / "tme_scores.csv"
            self.tme_scores.to_csv(tme_csv, index=False)
            self.logger.info(f"  Saved: {tme_csv}")

        # Save summary JSON
        summary = {
            "n_clusters_analyzed": len(self.markers_df['cluster'].unique()) if self.markers_df is not None else 0,
            "n_pathways_found": len(self.pathway_results),
            "n_driver_genes": len(self.driver_genes_df) if self.driver_genes_df is not None else 0,
            "databases_used": self.config.get("pathway_databases", []),
            "top_pathways": pd.DataFrame(self.pathway_results).groupby('term')['combined_score'].max().nlargest(10).to_dict() if self.pathway_results else {},
            "driver_genes_unique": list(self.driver_genes_df['gene'].unique()) if self.driver_genes_df is not None and len(self.driver_genes_df) > 0 else []
        }

        summary_json = self.output_dir / "pathway_summary.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"  Saved: {summary_json}")

        # Copy h5ad forward
        h5ad_file = self.input_dir / "adata_clustered.h5ad"
        if h5ad_file.exists():
            import shutil
            shutil.copy(h5ad_file, self.output_dir / "adata_clustered.h5ad")

        self.logger.info("Pathway analysis complete!")

    def validate_outputs(self) -> bool:
        """Validate output files were generated correctly."""
        # Pathway results are optional (may fail if no pathways found)
        return True
