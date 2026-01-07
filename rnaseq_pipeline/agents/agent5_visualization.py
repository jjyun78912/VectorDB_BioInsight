"""
Agent 5: Visualization

Generates publication-quality figures from analysis results.

Input:
- deg_all_results.csv: From Agent 1
- deg_significant.csv: From Agent 1
- normalized_counts.csv: From Agent 1
- hub_genes.csv: From Agent 2
- network_edges.csv: From Agent 2
- pathway_summary.csv: From Agent 3
- integrated_gene_table.csv: From Agent 4

Output:
- figures/volcano_plot.png
- figures/heatmap_top50.png
- figures/pca_plot.png
- figures/network_graph.png
- figures/pathway_barplot.png
- figures/interpretation_summary.png
- meta_agent5.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.base_agent import BaseAgent


class VisualizationAgent(BaseAgent):
    """Agent for generating publication-quality visualizations."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "figure_format": ["png", "svg"],
            "dpi": 300,
            "style": "whitegrid",
            "color_palette": "RdBu_r",
            "figsize": {
                "volcano": (10, 8),
                "heatmap": (12, 10),
                "pca": (10, 8),
                "network": (12, 12),
                "pathway": (10, 8),
                "summary": (14, 10)
            },
            "top_genes_heatmap": 50,
            "top_pathways": 15,
            "label_top_genes": 10
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent5_visualization", input_dir, output_dir, merged_config)

        # Create figures subdirectory
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style(self.config["style"])
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def validate_inputs(self) -> bool:
        """Validate input files."""
        # Required files
        self.deg_all = self.load_csv("deg_all_results.csv", required=False)
        self.deg_sig = self.load_csv("deg_significant.csv", required=False)
        self.norm_counts = self.load_csv("normalized_counts.csv", required=False)

        # Optional files (from later agents)
        self.hub_genes = self.load_csv("hub_genes.csv", required=False)
        self.network_edges = self.load_csv("network_edges.csv", required=False)
        self.pathway_summary = self.load_csv("pathway_summary.csv", required=False)
        self.integrated_table = self.load_csv("integrated_gene_table.csv", required=False)

        # Need at least DEG results
        if self.deg_all is None and self.deg_sig is None:
            self.logger.error("No DEG results found")
            return False

        return True

    def _save_figure(self, fig: plt.Figure, name: str) -> List[str]:
        """Save figure in multiple formats."""
        saved_files = []
        for fmt in self.config["figure_format"]:
            filepath = self.figures_dir / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=self.config["dpi"], bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            saved_files.append(str(filepath))
            self.logger.info(f"Saved {filepath.name}")
        plt.close(fig)
        return saved_files

    def _plot_volcano(self) -> Optional[List[str]]:
        """Generate volcano plot."""
        if self.deg_all is None:
            self.logger.warning("Skipping volcano plot - no DEG results")
            return None

        self.logger.info("Generating volcano plot...")

        fig, ax = plt.subplots(figsize=self.config["figsize"]["volcano"])

        df = self.deg_all.copy()
        df['neg_log10_padj'] = -np.log10(df['padj'].clip(lower=1e-300))

        # Determine significance
        padj_cutoff = 0.05
        log2fc_cutoff = 1.0

        df['significance'] = 'Not Significant'
        df.loc[(df['padj'] < padj_cutoff) & (df['log2FC'] > log2fc_cutoff), 'significance'] = 'Up'
        df.loc[(df['padj'] < padj_cutoff) & (df['log2FC'] < -log2fc_cutoff), 'significance'] = 'Down'

        colors = {'Not Significant': 'lightgray', 'Up': '#E74C3C', 'Down': '#3498DB'}

        for sig, color in colors.items():
            subset = df[df['significance'] == sig]
            ax.scatter(subset['log2FC'], subset['neg_log10_padj'],
                      c=color, alpha=0.6, s=20, label=sig)

        # Add significance lines
        ax.axhline(y=-np.log10(padj_cutoff), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=log2fc_cutoff, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-log2fc_cutoff, color='gray', linestyle='--', alpha=0.5)

        # Label top genes
        if self.deg_sig is not None and len(self.deg_sig) > 0:
            top_genes = self.deg_sig.head(self.config["label_top_genes"])
            for _, row in top_genes.iterrows():
                gene_row = df[df['gene_id'] == row['gene_id']]
                if len(gene_row) > 0:
                    x = gene_row['log2FC'].values[0]
                    y = gene_row['neg_log10_padj'].values[0]
                    ax.annotate(row['gene_id'], (x, y), fontsize=8,
                               ha='center', va='bottom')

        ax.set_xlabel('log2 Fold Change')
        ax.set_ylabel('-log10 Adjusted P-value')
        ax.set_title('Volcano Plot: Differential Expression')
        ax.legend(loc='upper right')

        # Add counts
        n_up = (df['significance'] == 'Up').sum()
        n_down = (df['significance'] == 'Down').sum()
        ax.text(0.02, 0.98, f'Up: {n_up}\nDown: {n_down}',
               transform=ax.transAxes, verticalalignment='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return self._save_figure(fig, "volcano_plot")

    def _plot_heatmap(self) -> Optional[List[str]]:
        """Generate heatmap of top DEGs."""
        if self.norm_counts is None or self.deg_sig is None:
            self.logger.warning("Skipping heatmap - missing data")
            return None

        self.logger.info("Generating heatmap...")

        # Get top genes
        n_genes = min(self.config["top_genes_heatmap"], len(self.deg_sig))
        top_genes = self.deg_sig.head(n_genes)['gene_id'].tolist()

        # Filter expression data
        gene_col = self.norm_counts.columns[0]
        expr_df = self.norm_counts[self.norm_counts[gene_col].isin(top_genes)]
        expr_df = expr_df.set_index(gene_col)

        if len(expr_df) == 0:
            self.logger.warning("No matching genes for heatmap")
            return None

        # Z-score normalize
        expr_zscore = expr_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

        # Plot
        fig, ax = plt.subplots(figsize=self.config["figsize"]["heatmap"])

        sns.heatmap(expr_zscore, cmap=self.config["color_palette"],
                   center=0, ax=ax, xticklabels=True,
                   yticklabels=True if n_genes <= 30 else False,
                   cbar_kws={'label': 'Z-score'})

        ax.set_title(f'Heatmap: Top {n_genes} DEGs')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Genes')

        plt.tight_layout()

        return self._save_figure(fig, "heatmap_top50")

    def _plot_pca(self) -> Optional[List[str]]:
        """Generate PCA plot."""
        if self.norm_counts is None:
            self.logger.warning("Skipping PCA - no normalized counts")
            return None

        self.logger.info("Generating PCA plot...")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Prepare data
        gene_col = self.norm_counts.columns[0]
        expr_df = self.norm_counts.set_index(gene_col)

        # Transpose (samples x genes)
        expr_t = expr_df.T

        # Scale and PCA
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_t)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(expr_scaled)

        # Plot
        fig, ax = plt.subplots(figsize=self.config["figsize"]["pca"])

        ax.scatter(pca_result[:, 0], pca_result[:, 1], s=100, alpha=0.7)

        # Label points
        for i, sample in enumerate(expr_t.index):
            ax.annotate(sample, (pca_result[i, 0], pca_result[i, 1]),
                       fontsize=8, ha='center', va='bottom')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA: Sample Distribution')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        return self._save_figure(fig, "pca_plot")

    def _plot_network(self) -> Optional[List[str]]:
        """Generate network visualization."""
        if self.network_edges is None or len(self.network_edges) == 0:
            self.logger.warning("Skipping network plot - no edges")
            return None

        if self.hub_genes is None or len(self.hub_genes) == 0:
            self.logger.warning("Skipping network plot - no hub genes")
            return None

        self.logger.info("Generating network plot...")

        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("networkx not installed - skipping network plot")
            return None

        # Build graph from edges
        G = nx.Graph()
        for _, row in self.network_edges.iterrows():
            G.add_edge(row['gene1'], row['gene2'], weight=row['abs_correlation'])

        if G.number_of_nodes() == 0:
            return None

        # Get hub genes for highlighting
        hub_set = set(self.hub_genes['gene_id'].tolist())

        # Limit to subgraph around hubs for visualization
        if G.number_of_nodes() > 100:
            # Keep only hub genes and their neighbors
            nodes_to_keep = set()
            for hub in hub_set:
                if hub in G:
                    nodes_to_keep.add(hub)
                    nodes_to_keep.update(list(G.neighbors(hub))[:10])
            G = G.subgraph(nodes_to_keep).copy()

        # Plot
        fig, ax = plt.subplots(figsize=self.config["figsize"]["network"])

        pos = nx.spring_layout(G, k=2, iterations=50)

        # Node colors based on hub status
        node_colors = ['#E74C3C' if n in hub_set else '#3498DB' for n in G.nodes()]
        node_sizes = [300 if n in hub_set else 100 for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8, ax=ax)

        # Label hub genes
        hub_labels = {n: n for n in G.nodes() if n in hub_set}
        nx.draw_networkx_labels(G, pos, hub_labels, font_size=8, ax=ax)

        ax.set_title(f'Co-expression Network\n({G.number_of_nodes()} genes, {G.number_of_edges()} edges)')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', label='Hub Gene'),
            Patch(facecolor='#3498DB', label='Other Gene')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        return self._save_figure(fig, "network_graph")

    def _plot_pathway_barplot(self) -> Optional[List[str]]:
        """Generate pathway enrichment barplot."""
        if self.pathway_summary is None or len(self.pathway_summary) == 0:
            self.logger.warning("Skipping pathway plot - no pathway data")
            return None

        self.logger.info("Generating pathway barplot...")

        # Get top pathways
        n_pathways = min(self.config["top_pathways"], len(self.pathway_summary))
        top_pathways = self.pathway_summary.head(n_pathways).copy()

        if len(top_pathways) == 0:
            return None

        # -log10 transform p-values
        top_pathways['neg_log10_padj'] = -np.log10(top_pathways['padj'].clip(lower=1e-300))

        # Truncate long names
        top_pathways['term_short'] = top_pathways['term_name'].apply(
            lambda x: x[:40] + '...' if len(str(x)) > 40 else x
        )

        # Plot
        fig, ax = plt.subplots(figsize=self.config["figsize"]["pathway"])

        colors = sns.color_palette("viridis", n_pathways)

        bars = ax.barh(range(n_pathways), top_pathways['neg_log10_padj'],
                      color=colors, alpha=0.8)

        ax.set_yticks(range(n_pathways))
        ax.set_yticklabels(top_pathways['term_short'])
        ax.set_xlabel('-log10 Adjusted P-value')
        ax.set_title('Top Enriched Pathways')

        # Add gene counts
        for i, (_, row) in enumerate(top_pathways.iterrows()):
            ax.text(row['neg_log10_padj'] + 0.1, i,
                   f"({row['gene_count']})",
                   va='center', fontsize=8)

        ax.invert_yaxis()
        plt.tight_layout()

        return self._save_figure(fig, "pathway_barplot")

    def _plot_interpretation_summary(self) -> Optional[List[str]]:
        """Generate interpretation summary visualization."""
        if self.integrated_table is None or len(self.integrated_table) == 0:
            self.logger.warning("Skipping interpretation summary - no data")
            return None

        self.logger.info("Generating interpretation summary...")

        fig, axes = plt.subplots(2, 2, figsize=self.config["figsize"]["summary"])

        # 1. Confidence distribution (pie)
        ax1 = axes[0, 0]
        conf_counts = self.integrated_table['confidence'].value_counts()
        colors = {'high': '#27AE60', 'medium': '#F39C12', 'low': '#E74C3C',
                 'novel_candidate': '#9B59B6', 'requires_validation': '#95A5A6'}
        pie_colors = [colors.get(c, 'gray') for c in conf_counts.index]
        ax1.pie(conf_counts.values, labels=conf_counts.index, colors=pie_colors,
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Confidence Distribution')

        # 2. DB match vs Hub status (bar)
        ax2 = axes[0, 1]
        categories = ['DB Matched\n& Hub', 'DB Matched\nOnly', 'Hub Only\n(Novel)', 'Neither']
        counts = [
            ((self.integrated_table['db_matched']) & (self.integrated_table['is_hub'])).sum(),
            ((self.integrated_table['db_matched']) & (~self.integrated_table['is_hub'])).sum(),
            ((~self.integrated_table['db_matched']) & (self.integrated_table['is_hub'])).sum(),
            ((~self.integrated_table['db_matched']) & (~self.integrated_table['is_hub'])).sum()
        ]
        bar_colors = ['#27AE60', '#3498DB', '#9B59B6', '#95A5A6']
        ax2.bar(categories, counts, color=bar_colors)
        ax2.set_ylabel('Number of Genes')
        ax2.set_title('DB Match vs Hub Status')

        # 3. Score distribution (histogram)
        ax3 = axes[1, 0]
        ax3.hist(self.integrated_table['interpretation_score'], bins=20,
                color='#3498DB', edgecolor='white', alpha=0.7)
        ax3.axvline(x=5, color='#27AE60', linestyle='--', label='High threshold')
        ax3.axvline(x=3, color='#F39C12', linestyle='--', label='Medium threshold')
        ax3.set_xlabel('Interpretation Score')
        ax3.set_ylabel('Number of Genes')
        ax3.set_title('Score Distribution')
        ax3.legend()

        # 4. Top genes table
        ax4 = axes[1, 1]
        ax4.axis('off')
        top_10 = self.integrated_table.head(10)[['gene_id', 'log2FC', 'is_hub', 'db_matched', 'confidence']]
        top_10 = top_10.copy()
        top_10['is_hub'] = top_10['is_hub'].map({True: 'Yes', False: 'No'})
        top_10['db_matched'] = top_10['db_matched'].map({True: 'Yes', False: 'No'})
        top_10['log2FC'] = top_10['log2FC'].round(2)

        table = ax4.table(cellText=top_10.values,
                         colLabels=['Gene', 'log2FC', 'Hub', 'DB', 'Confidence'],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Top 10 Genes by Interpretation Score', y=0.95)

        plt.tight_layout()

        return self._save_figure(fig, "interpretation_summary")

    def run(self) -> Dict[str, Any]:
        """Generate all visualizations."""
        generated_figures = []
        failed_figures = []

        # Generate each figure
        figure_functions = [
            ("volcano_plot", self._plot_volcano),
            ("heatmap", self._plot_heatmap),
            ("pca_plot", self._plot_pca),
            ("network_graph", self._plot_network),
            ("pathway_barplot", self._plot_pathway_barplot),
            ("interpretation_summary", self._plot_interpretation_summary)
        ]

        for name, func in figure_functions:
            try:
                result = func()
                if result:
                    generated_figures.extend(result)
                else:
                    failed_figures.append(name)
            except Exception as e:
                self.logger.error(f"Error generating {name}: {e}")
                failed_figures.append(name)

        self.logger.info(f"Visualization Complete:")
        self.logger.info(f"  Generated: {len(generated_figures)} files")
        self.logger.info(f"  Failed/Skipped: {len(failed_figures)}")

        return {
            "figures_generated": generated_figures,
            "failed_figures": failed_figures,
            "total_generated": len(generated_figures)
        }

    def validate_outputs(self) -> bool:
        """Validate visualization outputs."""
        # Check figures directory exists
        if not self.figures_dir.exists():
            self.logger.error("Figures directory not created")
            return False

        # Check at least some figures were generated
        png_files = list(self.figures_dir.glob("*.png"))
        if len(png_files) == 0:
            self.logger.warning("No PNG figures generated")
            # Still valid if input data was limited

        return True
