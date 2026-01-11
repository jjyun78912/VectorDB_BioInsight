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
- figures/volcano_interactive.html  # NEW: Plotly interactive volcano
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

# Plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

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
                "volcano": (12, 10),  # Increased size for better readability
                "heatmap": (12, 10),
                "pca": (10, 8),
                "network": (12, 12),
                "pathway": (10, 8),
                "summary": (14, 10)
            },
            "top_genes_heatmap": 50,
            "top_pathways": 15,
            "label_top_genes": 20,  # Show more gene labels
            "generate_interactive": True,  # Generate Plotly interactive volcano
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

        # Load metadata for sample labels (optional)
        self.metadata = self.load_csv("metadata.csv", required=False)
        self.sample_to_condition = {}
        if self.metadata is not None:
            # Build sample -> condition mapping
            sample_col = self.metadata.columns[0]  # First col is sample_id
            condition_col = self.config.get("condition_column", "condition")
            if condition_col in self.metadata.columns:
                self.sample_to_condition = dict(
                    zip(self.metadata[sample_col], self.metadata[condition_col])
                )
                self.logger.info(f"Loaded metadata: {len(self.sample_to_condition)} samples")

        # Need at least DEG results
        if self.deg_all is None and self.deg_sig is None:
            self.logger.error("No DEG results found")
            return False

        return True

    def _save_figure(self, fig: plt.Figure, name: str, preserve_facecolor: bool = False) -> List[str]:
        """Save figure in multiple formats."""
        saved_files = []
        # Use figure's facecolor if preserve_facecolor=True, otherwise white
        facecolor = fig.get_facecolor() if preserve_facecolor else 'white'
        for fmt in self.config["figure_format"]:
            filepath = self.figures_dir / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=self.config["dpi"], bbox_inches='tight',
                       facecolor=facecolor, edgecolor='none')
            saved_files.append(str(filepath))
            self.logger.info(f"Saved {filepath.name}")
        plt.close(fig)
        return saved_files

    def _get_gene_symbol(self, gene_id: str) -> str:
        """Extract gene symbol from gene_id or integrated table."""
        # Try to get symbol from integrated table
        if self.integrated_table is not None and 'gene_symbol' in self.integrated_table.columns:
            match = self.integrated_table[self.integrated_table['gene_id'] == gene_id]
            if len(match) > 0 and pd.notna(match['gene_symbol'].values[0]):
                return match['gene_symbol'].values[0]

        # Fallback: extract symbol from ENSG ID if possible
        # e.g., "ENSG00000141736.14" -> just use the ID
        return gene_id.split('.')[0] if '.' in gene_id else gene_id

    def _plot_volcano(self) -> Optional[List[str]]:
        """Generate volcano plot with improved gene labels."""
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

        colors = {'Not Significant': '#D3D3D3', 'Up': '#E74C3C', 'Down': '#3498DB'}

        # Plot each significance category with different sizes
        for sig, color in colors.items():
            subset = df[df['significance'] == sig]
            size = 60 if sig != 'Not Significant' else 30  # Larger size for significant genes
            alpha = 0.8 if sig != 'Not Significant' else 0.4
            ax.scatter(subset['log2FC'], subset['neg_log10_padj'],
                      c=color, alpha=alpha, s=size, label=f'{sig} ({len(subset)})',
                      edgecolors='white', linewidth=0.5)

        # Add significance lines
        ax.axhline(y=-np.log10(padj_cutoff), color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=log2fc_cutoff, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=-log2fc_cutoff, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Label top genes with gene symbols - SORT BY |log2FC| to match Top DEGs bar chart
        if self.deg_sig is not None and len(self.deg_sig) > 0:
            # Sort by absolute log2FC to match the Top DEGs bar chart
            deg_sorted = self.deg_sig.copy()
            deg_sorted['abs_log2FC'] = deg_sorted['log2FC'].abs()
            deg_sorted = deg_sorted.sort_values('abs_log2FC', ascending=False)

            # Get top upregulated and downregulated separately (by |log2FC|)
            top_up = deg_sorted[deg_sorted['direction'] == 'up'].head(self.config["label_top_genes"] // 2)
            top_down = deg_sorted[deg_sorted['direction'] == 'down'].head(self.config["label_top_genes"] // 2)
            top_genes = pd.concat([top_up, top_down])

            from adjustText import adjust_text
            texts = []

            for _, row in top_genes.iterrows():
                gene_row = df[df['gene_id'] == row['gene_id']]
                if len(gene_row) > 0:
                    x = gene_row['log2FC'].values[0]
                    y = gene_row['neg_log10_padj'].values[0]

                    # Get gene symbol for display
                    if 'gene_symbol' in row and pd.notna(row['gene_symbol']):
                        label = row['gene_symbol']
                    else:
                        label = self._get_gene_symbol(row['gene_id'])

                    text = ax.annotate(label, (x, y), fontsize=9, fontweight='bold',
                                       ha='center', va='bottom',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                                alpha=0.8, edgecolor='none'))
                    texts.append(text)

            # Try to adjust text positions to avoid overlap
            try:
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
            except Exception:
                pass  # adjustText may fail silently

        ax.set_xlabel('log2 Fold Change', fontsize=14, fontweight='bold')
        ax.set_ylabel('-log10 Adjusted P-value', fontsize=14, fontweight='bold')
        ax.set_title('Volcano Plot: Differential Expression Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        # Add counts in a box
        n_up = (df['significance'] == 'Up').sum()
        n_down = (df['significance'] == 'Down').sum()
        n_total = len(df)
        ax.text(0.02, 0.98, f'Total genes: {n_total:,}\n↑ Upregulated: {n_up}\n↓ Downregulated: {n_down}',
               transform=ax.transAxes, verticalalignment='top',
               fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        # Improve grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        return self._save_figure(fig, "volcano_plot")

    def _plot_volcano_interactive(self) -> Optional[str]:
        """Generate interactive Plotly volcano plot with hover/zoom."""
        if not HAS_PLOTLY:
            self.logger.warning("Plotly not installed - skipping interactive volcano")
            return None

        if self.deg_all is None:
            self.logger.warning("Skipping interactive volcano - no DEG results")
            return None

        self.logger.info("Generating interactive volcano plot (Plotly)...")

        df = self.deg_all.copy()
        df['neg_log10_padj'] = -np.log10(df['padj'].clip(lower=1e-300))

        # Determine significance
        padj_cutoff = 0.05
        log2fc_cutoff = 1.0

        df['significance'] = 'Not Significant'
        df.loc[(df['padj'] < padj_cutoff) & (df['log2FC'] > log2fc_cutoff), 'significance'] = 'Upregulated'
        df.loc[(df['padj'] < padj_cutoff) & (df['log2FC'] < -log2fc_cutoff), 'significance'] = 'Downregulated'

        # Get gene symbols for hover text
        df['gene_symbol'] = df['gene_id'].apply(self._get_gene_symbol)

        # Create hover text with detailed information
        df['hover_text'] = df.apply(
            lambda row: f"<b>{row['gene_symbol']}</b><br>" +
                       f"Gene ID: {row['gene_id']}<br>" +
                       f"log2FC: {row['log2FC']:.3f}<br>" +
                       f"padj: {row['padj']:.2e}<br>" +
                       f"-log10(padj): {row['neg_log10_padj']:.2f}<br>" +
                       f"Status: {row['significance']}",
            axis=1
        )

        # Color mapping
        color_map = {
            'Not Significant': 'rgba(180, 180, 180, 0.4)',
            'Upregulated': 'rgba(231, 76, 60, 0.8)',
            'Downregulated': 'rgba(52, 152, 219, 0.8)'
        }

        fig = go.Figure()

        # Add traces for each significance category
        for sig in ['Not Significant', 'Upregulated', 'Downregulated']:
            subset = df[df['significance'] == sig]
            marker_size = 6 if sig == 'Not Significant' else 10

            fig.add_trace(go.Scatter(
                x=subset['log2FC'],
                y=subset['neg_log10_padj'],
                mode='markers',
                name=f'{sig} ({len(subset)})',
                marker=dict(
                    size=marker_size,
                    color=color_map[sig],
                    line=dict(width=0.5, color='white')
                ),
                text=subset['hover_text'],
                hoverinfo='text',
                hovertemplate='%{text}<extra></extra>'
            ))

        # Add significance threshold lines
        max_y = df['neg_log10_padj'].max() * 1.1
        max_x = max(abs(df['log2FC'].min()), abs(df['log2FC'].max())) * 1.1

        # Horizontal line (padj threshold)
        fig.add_hline(y=-np.log10(padj_cutoff), line_dash="dash", line_color="gray",
                     annotation_text=f"padj = {padj_cutoff}", annotation_position="right")

        # Vertical lines (log2FC thresholds)
        fig.add_vline(x=log2fc_cutoff, line_dash="dash", line_color="gray")
        fig.add_vline(x=-log2fc_cutoff, line_dash="dash", line_color="gray")

        # Add annotations for top genes - SORT BY |log2FC| to match Top DEGs bar chart
        if self.deg_sig is not None and len(self.deg_sig) > 0:
            # Sort by absolute log2FC to match the Top DEGs bar chart
            deg_sorted = self.deg_sig.copy()
            deg_sorted['abs_log2FC'] = deg_sorted['log2FC'].abs()
            deg_sorted = deg_sorted.sort_values('abs_log2FC', ascending=False)

            top_up = deg_sorted[deg_sorted['direction'] == 'up'].head(10)
            top_down = deg_sorted[deg_sorted['direction'] == 'down'].head(10)
            top_genes = pd.concat([top_up, top_down])

            for _, row in top_genes.iterrows():
                gene_row = df[df['gene_id'] == row['gene_id']]
                if len(gene_row) > 0:
                    x = gene_row['log2FC'].values[0]
                    y = gene_row['neg_log10_padj'].values[0]
                    symbol = gene_row['gene_symbol'].values[0]

                    fig.add_annotation(
                        x=x, y=y,
                        text=f"<b>{symbol}</b>",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor='gray',
                        ax=20 if x > 0 else -20,
                        ay=-20,
                        font=dict(size=10, color='black'),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='gray',
                        borderwidth=1
                    )

        # Update layout
        n_up = (df['significance'] == 'Upregulated').sum()
        n_down = (df['significance'] == 'Downregulated').sum()

        fig.update_layout(
            title=dict(
                text=f'<b>Interactive Volcano Plot</b><br><sup>↑ Up: {n_up} | ↓ Down: {n_down} | Total: {len(df):,}</sup>',
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title='<b>log2 Fold Change</b>',
            yaxis_title='<b>-log10 Adjusted P-value</b>',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1
            ),
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=800,
            # Add zoom and pan tools
            dragmode='zoom',
            xaxis=dict(
                range=[-max_x, max_x],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='lightgray'
            ),
            yaxis=dict(
                range=[0, max_y],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='lightgray'
            )
        )

        # Add modebar buttons for interaction
        config = {
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'volcano_plot_interactive',
                'height': 800,
                'width': 1000,
                'scale': 2
            }
        }

        # Save as HTML
        html_path = self.figures_dir / "volcano_interactive.html"
        fig.write_html(str(html_path), config=config, include_plotlyjs='cdn')
        self.logger.info(f"Saved interactive volcano: {html_path.name}")

        return str(html_path)

    def _get_sample_label(self, sample_id: str) -> str:
        """Get formatted sample label with condition."""
        condition = self.sample_to_condition.get(sample_id, None)
        if condition:
            # Shorten condition name
            cond_short = condition[:1].upper()  # T for tumor, N for normal
            # Shorten sample ID (take last part or abbreviate)
            if '-' in sample_id:
                parts = sample_id.split('-')
                # For TCGA: TCGA-BH-A18H-01A -> BH-A18H (T)
                if len(parts) >= 3:
                    short_id = f"{parts[1]}-{parts[2]}"
                else:
                    short_id = parts[-1]
            else:
                short_id = sample_id[:10]
            return f"{short_id} ({cond_short})"
        return sample_id[:15] if len(sample_id) > 15 else sample_id

    def _plot_heatmap(self) -> Optional[List[str]]:
        """Generate heatmap of top DEGs with sample condition labels."""
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

        # Sort samples by condition if metadata available
        sample_order = list(expr_df.columns)
        if self.sample_to_condition:
            # Sort: Tumor first, then Normal (or alphabetically by condition)
            sample_order = sorted(
                sample_order,
                key=lambda x: (self.sample_to_condition.get(x, 'zzz'), x)
            )
            expr_df = expr_df[sample_order]

        # Z-score normalize
        expr_zscore = expr_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

        # Create formatted sample labels
        sample_labels = [self._get_sample_label(s) for s in expr_df.columns]

        # Get gene symbols for y-axis if available
        gene_labels = []
        for gene_id in expr_zscore.index:
            symbol = self._get_gene_symbol(gene_id)
            gene_labels.append(symbol if symbol != gene_id else gene_id.split('.')[0])

        # Create figure with condition color bar
        if self.sample_to_condition:
            # Use clustermap for better visualization with color annotation
            fig = plt.figure(figsize=(14, 12))

            # Create condition colors
            conditions = [self.sample_to_condition.get(s, 'Unknown') for s in expr_df.columns]
            unique_conditions = list(set(conditions))
            condition_palette = {'tumor': '#E74C3C', 'normal': '#3498DB',
                               'Tumor': '#E74C3C', 'Normal': '#3498DB'}
            # Add fallback colors for other conditions
            for i, cond in enumerate(unique_conditions):
                if cond not in condition_palette:
                    condition_palette[cond] = plt.cm.Set2(i % 8)

            col_colors = [condition_palette.get(c, '#95A5A6') for c in conditions]

            # Main heatmap axis
            ax_heatmap = fig.add_axes([0.15, 0.1, 0.7, 0.75])
            ax_colorbar = fig.add_axes([0.15, 0.88, 0.7, 0.02])
            ax_legend = fig.add_axes([0.88, 0.1, 0.02, 0.75])

            # Draw condition color bar
            for i, color in enumerate(col_colors):
                ax_colorbar.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none'))
            ax_colorbar.set_xlim(0, len(col_colors))
            ax_colorbar.set_ylim(0, 1)
            ax_colorbar.axis('off')
            ax_colorbar.set_title('Sample Condition', fontsize=10, pad=5)

            # Draw heatmap
            im = ax_heatmap.imshow(expr_zscore.values, cmap=self.config["color_palette"],
                                   aspect='auto', vmin=-2, vmax=2)

            # Set labels
            ax_heatmap.set_xticks(range(len(sample_labels)))
            ax_heatmap.set_xticklabels(sample_labels, rotation=90, fontsize=7, ha='center')
            ax_heatmap.set_yticks(range(len(gene_labels)))
            ax_heatmap.set_yticklabels(gene_labels if n_genes <= 50 else [], fontsize=8)

            ax_heatmap.set_xlabel('Samples', fontsize=12, fontweight='bold')
            ax_heatmap.set_ylabel('Genes', fontsize=12, fontweight='bold')

            # Colorbar for z-score
            cbar = plt.colorbar(im, cax=ax_legend)
            cbar.set_label('Z-score', fontsize=10)

            # Add legend for conditions
            from matplotlib.patches import Patch
            legend_patches = [Patch(facecolor=condition_palette.get(c, '#95A5A6'),
                                   label=c.capitalize()) for c in unique_conditions]
            fig.legend(handles=legend_patches, loc='upper right',
                      bbox_to_anchor=(0.99, 0.99), fontsize=9, title='Condition')

            fig.suptitle(f'Heatmap: Top {n_genes} DEGs (sorted by condition)',
                        fontsize=14, fontweight='bold', y=0.98)

        else:
            # Simple heatmap without condition info
            fig, ax = plt.subplots(figsize=self.config["figsize"]["heatmap"])

            sns.heatmap(expr_zscore, cmap=self.config["color_palette"],
                       center=0, ax=ax, xticklabels=sample_labels,
                       yticklabels=gene_labels if n_genes <= 50 else False,
                       cbar_kws={'label': 'Z-score'})

            ax.set_title(f'Heatmap: Top {n_genes} DEGs', fontsize=14, fontweight='bold')
            ax.set_xlabel('Samples', fontsize=12)
            ax.set_ylabel('Genes', fontsize=12)
            plt.xticks(rotation=90, fontsize=7)

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
        """Generate galaxy-style network visualization with gene symbols."""
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

        # Build gene_id to gene_symbol mapping from integrated_gene_table
        gene_id_to_symbol = {}
        if self.integrated_table is not None:
            for _, row in self.integrated_table.iterrows():
                if pd.notna(row.get('gene_symbol')) and row['gene_symbol']:
                    gene_id_to_symbol[row['gene_id']] = row['gene_symbol']

        def get_gene_label(gene_id: str) -> str:
            """Convert ENSG ID to gene symbol, fallback to short ID."""
            if gene_id in gene_id_to_symbol:
                return gene_id_to_symbol[gene_id]
            # Fallback: shorten ENSG ID (ENSG00000122952.17 -> ENSG...2952)
            if gene_id.startswith('ENSG'):
                parts = gene_id.split('.')
                return f"...{parts[0][-4:]}"
            return gene_id

        # Build graph from edges with gene symbols
        G = nx.Graph()
        for _, row in self.network_edges.iterrows():
            label1 = get_gene_label(row['gene1'])
            label2 = get_gene_label(row['gene2'])
            G.add_edge(label1, label2, weight=row['abs_correlation'])

        if G.number_of_nodes() == 0:
            return None

        # Get hub genes for highlighting (convert to symbols)
        hub_set_ids = set(self.hub_genes['gene_id'].tolist())
        hub_set_labels = {get_gene_label(h) for h in hub_set_ids}

        # Get hub scores for sizing
        hub_scores = {}
        for _, row in self.hub_genes.iterrows():
            label = get_gene_label(row['gene_id'])
            hub_scores[label] = row.get('hub_score', 0.5)

        # Limit to subgraph around hubs for visualization
        if G.number_of_nodes() > 100:
            nodes_to_keep = set()
            for hub_label in hub_set_labels:
                if hub_label in G:
                    nodes_to_keep.add(hub_label)
                    nodes_to_keep.update(list(G.neighbors(hub_label))[:10])
            G = G.subgraph(nodes_to_keep).copy()

        # === Galaxy/Space Theme ===
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 14), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')

        # Use Kamada-Kawai layout for more organic feel
        pos = nx.kamada_kawai_layout(G)

        # Calculate node degrees for sizing non-hub nodes
        degrees = dict(G.degree())

        # Node styling - Galaxy theme
        node_colors = []
        node_sizes = []
        node_alphas = []

        for n in G.nodes():
            if n in hub_set_labels:
                # Hub genes: bright stars (gold/orange gradient based on hub_score)
                score = hub_scores.get(n, 0.5)
                node_colors.append(plt.cm.YlOrRd(0.5 + score * 0.5))  # Yellow-Orange-Red
                node_sizes.append(400 + score * 600)  # Size based on hub_score
                node_alphas.append(0.95)
            else:
                # Other genes: distant stars (cyan/blue)
                deg = degrees.get(n, 1)
                node_colors.append('#4fc3f7')  # Cyan blue
                node_sizes.append(80 + deg * 15)  # Size based on degree
                node_alphas.append(0.7)

        # Draw edges with gradient opacity based on weight
        edge_weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1

        for (u, v), w in zip(G.edges(), edge_weights):
            alpha = 0.1 + (w / max_weight) * 0.3
            # Purple-ish edges for galaxy feel
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                   color='#7c4dff', alpha=alpha, linewidth=0.5, zorder=1)

        # Draw nodes with glow effect
        # First pass: glow (larger, more transparent)
        for i, n in enumerate(G.nodes()):
            if n in hub_set_labels:
                ax.scatter(pos[n][0], pos[n][1],
                          s=node_sizes[i] * 2,
                          c=[node_colors[i]],
                          alpha=0.2, zorder=2)

        # Second pass: actual nodes
        for i, n in enumerate(G.nodes()):
            ax.scatter(pos[n][0], pos[n][1],
                      s=node_sizes[i],
                      c=[node_colors[i]],
                      alpha=node_alphas[i],
                      edgecolors='white' if n in hub_set_labels else 'none',
                      linewidths=1.5 if n in hub_set_labels else 0,
                      zorder=3)

        # Labels for ALL genes (not just hubs)
        for n in G.nodes():
            x, y = pos[n]
            if n in hub_set_labels:
                # Hub genes: larger, white text with slight offset
                ax.annotate(n, (x, y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='white',
                           ha='left', va='bottom', zorder=4)
            else:
                # Other genes: smaller, cyan text
                ax.annotate(n, (x, y),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=6, color='#80deea', alpha=0.8,
                           ha='left', va='bottom', zorder=4)

        # Title with galaxy theme (without special characters for font compatibility)
        ax.set_title('Gene Co-expression Network\nGalaxy View',
                    fontsize=16, fontweight='bold', color='white', pad=20)

        # Subtitle with stats
        ax.text(0.5, 0.02, f'{G.number_of_nodes()} genes · {G.number_of_edges()} connections · {len(hub_set_labels)} hub genes',
               transform=ax.transAxes, ha='center', fontsize=10, color='#b0bec5')

        ax.axis('off')
        ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)
        ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)

        # Legend with custom styling
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='#0d1117', markerfacecolor='#ff9800',
                   markersize=12, markeredgecolor='white', markeredgewidth=1.5, label='Hub Gene'),
            Line2D([0], [0], marker='o', color='#0d1117', markerfacecolor='#4fc3f7',
                   markersize=8, label='Connected Gene'),
            Line2D([0], [0], color='#7c4dff', linewidth=2, alpha=0.5, label='Co-expression')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right',
                          facecolor='#161b22', edgecolor='#30363d',
                          fontsize=9, labelcolor='white')

        plt.tight_layout()

        # Reset style to default after this plot
        result = self._save_figure(fig, "network_graph", preserve_facecolor=True)
        plt.style.use('default')
        return result

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

    def _plot_network_3d_interactive(self) -> Optional[str]:
        """Generate interactive 3D network visualization with Three.js + 3d-force-graph.

        Inspired by Obsidian 3D Graph plugin style with:
        - Bloom/glow effects
        - Particle animations on links
        - Smooth force-directed physics
        - Dark space theme
        """
        if self.network_edges is None or len(self.network_edges) == 0:
            self.logger.warning("Skipping 3D network - no edges")
            return None

        if self.hub_genes is None or len(self.hub_genes) == 0:
            self.logger.warning("Skipping 3D network - no hub genes")
            return None

        self.logger.info("Generating interactive 3D network (Obsidian-style)...")

        try:
            import networkx as nx
            import json
        except ImportError:
            self.logger.warning("networkx not installed - skipping 3D network")
            return None

        # Build gene_id to gene_symbol mapping
        gene_id_to_symbol = {}
        if self.integrated_table is not None:
            for _, row in self.integrated_table.iterrows():
                if pd.notna(row.get('gene_symbol')) and row['gene_symbol']:
                    gene_id_to_symbol[row['gene_id']] = row['gene_symbol']

        # Get log2FC and direction for coloring
        gene_info = {}
        if self.integrated_table is not None:
            for _, row in self.integrated_table.iterrows():
                gene_id = row['gene_id']
                gene_info[gene_id] = {
                    'log2FC': row.get('log2FC', 0),
                    'direction': row.get('direction', 'none'),
                    'padj': row.get('padj', 1)
                }

        def get_gene_label(gene_id: str) -> str:
            if gene_id in gene_id_to_symbol:
                return gene_id_to_symbol[gene_id]
            if gene_id.startswith('ENSG'):
                parts = gene_id.split('.')
                return f"...{parts[0][-4:]}"
            return gene_id

        # Build graph
        G = nx.Graph()
        for _, row in self.network_edges.iterrows():
            gene1, gene2 = row['gene1'], row['gene2']
            label1 = get_gene_label(gene1)
            label2 = get_gene_label(gene2)
            G.add_edge(label1, label2, weight=row['abs_correlation'])
            G.nodes[label1]['gene_id'] = gene1
            G.nodes[label2]['gene_id'] = gene2

        if G.number_of_nodes() == 0:
            return None

        # Get hub info
        hub_set_ids = set(self.hub_genes['gene_id'].tolist())
        hub_set_labels = {get_gene_label(h) for h in hub_set_ids}
        hub_scores = {}
        for _, row in self.hub_genes.iterrows():
            label = get_gene_label(row['gene_id'])
            hub_scores[label] = row.get('hub_score', 0.5)

        # Limit graph size
        if G.number_of_nodes() > 150:
            nodes_to_keep = set()
            for hub_label in hub_set_labels:
                if hub_label in G:
                    nodes_to_keep.add(hub_label)
                    nodes_to_keep.update(list(G.neighbors(hub_label))[:8])
            G = G.subgraph(nodes_to_keep).copy()

        # Prepare data for 3d-force-graph
        nodes_data = []
        for node in G.nodes():
            is_hub = node in hub_set_labels
            gene_id = G.nodes[node].get('gene_id', '')
            info = gene_info.get(gene_id, {})
            log2fc = info.get('log2FC', 0)
            direction = info.get('direction', 'none')
            degree = G.degree(node)

            # Color based on direction and hub status - White background palette
            if is_hub:
                if direction == 'up':
                    color = '#dc2626'  # Red (upregulated hub)
                elif direction == 'down':
                    color = '#2563eb'  # Blue (downregulated hub)
                else:
                    color = '#7c3aed'  # Purple (unknown hub)
            else:
                if direction == 'up':
                    color = '#f87171'  # Light red (upregulated)
                elif direction == 'down':
                    color = '#60a5fa'  # Light blue (downregulated)
                else:
                    color = '#9ca3af'  # Gray

            nodes_data.append({
                'id': node,
                'name': node,
                'isHub': is_hub,
                'hubScore': hub_scores.get(node, 0),
                'log2FC': round(log2fc, 2),
                'direction': direction,
                'degree': degree,
                'color': color,
                'size': 8 + hub_scores.get(node, 0) * 15 if is_hub else 4 + degree * 0.5
            })

        links_data = []
        for edge in G.edges(data=True):
            links_data.append({
                'source': edge[0],
                'target': edge[1],
                'weight': round(edge[2].get('weight', 0.5), 3)
            })

        graph_data = json.dumps({'nodes': nodes_data, 'links': links_data})

        hub_count = len([n for n in nodes_data if n['isHub']])
        stats_text = f"{len(nodes_data)} genes · {len(links_data)} connections · {hub_count} hub genes"

        # HTML template with 3d-force-graph - Premium Scientific Style
        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Gene Co-expression Network</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three-spritetext@1"></script>
    <script src="https://unpkg.com/3d-force-graph@1"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #ffffff;
            overflow: hidden;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        #container {{ width: 100vw; height: 100vh; }}

        /* Subtle gradient accent */
        .ambient-glow {{
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(180px);
            opacity: 0.08;
            pointer-events: none;
            z-index: 0;
        }}
        .glow-1 {{ background: #3b82f6; top: -200px; left: -200px; }}
        .glow-2 {{ background: #8b5cf6; bottom: -200px; right: -200px; }}

        /* Header */
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 24px 32px;
            background: linear-gradient(to bottom, rgba(255,255,255,0.95) 0%, transparent 100%);
            z-index: 100;
            pointer-events: none;
        }}
        .header h1 {{
            color: #1f2937;
            font-size: 18px;
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        .header .subtitle {{
            color: #6b7280;
            font-size: 12px;
            margin-top: 6px;
            font-weight: 400;
            letter-spacing: 0.02em;
        }}

        /* Info panel */
        .info-panel {{
            position: fixed;
            top: 100px;
            right: 24px;
            width: 200px;
            z-index: 100;
        }}
        .panel-section {{
            background: rgba(255,255,255,0.9);
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}
        .panel-title {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #6b7280;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 6px 0;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            box-shadow: 0 2px 6px currentColor;
        }}
        .legend-label {{
            font-size: 12px;
            color: #374151;
            font-weight: 400;
        }}
        .legend-dot.hub-up {{ background: #dc2626; color: rgba(220,38,38,0.4); }}
        .legend-dot.hub-down {{ background: #2563eb; color: rgba(37,99,235,0.4); }}
        .legend-dot.gene-up {{ background: #f87171; color: rgba(248,113,113,0.4); }}
        .legend-dot.gene-down {{ background: #60a5fa; color: rgba(96,165,250,0.4); }}

        /* Stats */
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f3f4f6;
        }}
        .stat-row:last-child {{ border: none; }}
        .stat-label {{ font-size: 11px; color: #6b7280; }}
        .stat-value {{ font-size: 12px; color: #1f2937; font-weight: 600; }}

        /* Controls */
        .controls {{
            display: flex;
            gap: 8px;
            margin-top: 4px;
        }}
        .control-btn {{
            flex: 1;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 0;
            color: #374151;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s ease;
            font-family: inherit;
        }}
        .control-btn:hover {{
            background: #f3f4f6;
            color: #111827;
            border-color: #d1d5db;
        }}
        .control-btn.active {{
            background: #eff6ff;
            border-color: #3b82f6;
            color: #2563eb;
        }}

        /* Tooltip */
        .node-tooltip {{
            position: fixed;
            background: rgba(255,255,255,0.98);
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            color: #1f2937;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transform: translateY(4px);
            transition: all 0.15s ease;
            min-width: 180px;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            z-index: 200;
        }}
        .node-tooltip.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        .tooltip-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .tooltip-gene {{
            font-size: 15px;
            font-weight: 600;
            color: #111827;
        }}
        .tooltip-badge {{
            background: linear-gradient(135deg, #3b82f6, #6366f1);
            color: #fff;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 9px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
        }}
        .tooltip-label {{ color: #6b7280; font-size: 11px; }}
        .tooltip-value {{ color: #1f2937; font-weight: 500; }}
        .tooltip-up {{ color: #dc2626; }}
        .tooltip-down {{ color: #2563eb; }}

        /* Bottom bar */
        .bottom-bar {{
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 24px;
            background: rgba(255,255,255,0.95);
            border: 1px solid #e5e7eb;
            border-radius: 100px;
            padding: 12px 24px;
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 100;
        }}
        .bottom-stat {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .bottom-stat-icon {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #3b82f6;
        }}
        .bottom-stat-text {{
            font-size: 12px;
            color: #6b7280;
        }}
        .bottom-stat-text span {{
            color: #1f2937;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="ambient-glow glow-1"></div>
    <div class="ambient-glow glow-2"></div>

    <div id="container"></div>

    <div class="header">
        <h1>Gene Co-expression Network</h1>
        <div class="subtitle">Interactive 3D visualization · Rotate · Zoom · Click to focus</div>
    </div>

    <div class="info-panel">
        <div class="panel-section">
            <div class="panel-title">Legend</div>
            <div class="legend-item"><div class="legend-dot hub-up"></div><span class="legend-label">Hub gene (up)</span></div>
            <div class="legend-item"><div class="legend-dot hub-down"></div><span class="legend-label">Hub gene (down)</span></div>
            <div class="legend-item"><div class="legend-dot gene-up"></div><span class="legend-label">Gene (up)</span></div>
            <div class="legend-item"><div class="legend-dot gene-down"></div><span class="legend-label">Gene (down)</span></div>
        </div>

        <div class="panel-section">
            <div class="panel-title">Controls</div>
            <div class="controls">
                <button class="control-btn active" id="labelBtn" onclick="toggleLabels()">Labels</button>
                <button class="control-btn" onclick="resetCamera()">Reset</button>
            </div>
        </div>
    </div>

    <div class="bottom-bar">
        <div class="bottom-stat">
            <div class="bottom-stat-icon"></div>
            <div class="bottom-stat-text"><span>{len(nodes_data)}</span> genes</div>
        </div>
        <div class="bottom-stat">
            <div class="bottom-stat-icon" style="background: #6366f1;"></div>
            <div class="bottom-stat-text"><span>{len(links_data)}</span> connections</div>
        </div>
        <div class="bottom-stat">
            <div class="bottom-stat-icon" style="background: #dc2626;"></div>
            <div class="bottom-stat-text"><span>{hub_count}</span> hub genes</div>
        </div>
    </div>

    <div class="node-tooltip" id="tooltip"></div>

    <script>
        const graphData = {graph_data};

        let showLabels = true;

        const Graph = ForceGraph3D()
            (document.getElementById('container'))
            .graphData(graphData)
            .backgroundColor('#ffffff')
            .showNavInfo(false)

            // Large, visible node styling
            .nodeVal(node => node.isHub ? 25 : 10)
            .nodeColor(node => node.color)
            .nodeOpacity(1.0)
            .nodeResolution(32)

            // Large, readable labels on white background
            .nodeLabel('')
            .nodeThreeObject(node => {{
                if (!showLabels && !node.isHub) return null;

                const sprite = new SpriteText(node.name);
                sprite.color = node.isHub ? '#000000' : '#374151';
                sprite.textHeight = node.isHub ? 8 : 5;
                sprite.fontWeight = node.isHub ? 'bold' : '500';
                sprite.fontFace = 'Arial, sans-serif';
                sprite.backgroundColor = node.isHub ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.7)';
                sprite.padding = 2;
                sprite.borderRadius = 3;
                return sprite;
            }})
            .nodeThreeObjectExtend(true)

            // Bold link styling - clearly visible on white
            .linkWidth(link => 1 + link.weight * 2)
            .linkOpacity(0.6)
            .linkColor(() => '#6b7280')

            .linkDirectionalParticles(0)

            // Optimized physics
            .d3AlphaDecay(0.008)
            .d3VelocityDecay(0.15)
            .d3Force('charge', d3.forceManyBody().strength(-400).distanceMax(500))
            .d3Force('link', d3.forceLink().distance(100).strength(0.2))
            .d3Force('center', d3.forceCenter())
            .warmupTicks(400)
            .cooldownTicks(500)

            // Smooth interactions
            .onNodeHover(node => {{
                document.body.style.cursor = node ? 'pointer' : 'default';
                const tooltip = document.getElementById('tooltip');
                if (node) {{
                    const dir = node.direction === 'up' ?
                        '<span class="tooltip-up">↑ Upregulated</span>' :
                        node.direction === 'down' ?
                        '<span class="tooltip-down">↓ Downregulated</span>' : '—';

                    tooltip.innerHTML = `
                        <div class="tooltip-header">
                            <span class="tooltip-gene">${{node.name}}</span>
                            ${{node.isHub ? '<span class="tooltip-badge">Hub</span>' : ''}}
                        </div>
                        <div class="tooltip-row"><span class="tooltip-label">Expression</span><span class="tooltip-value">${{dir}}</span></div>
                        <div class="tooltip-row"><span class="tooltip-label">log₂FC</span><span class="tooltip-value">${{node.log2FC.toFixed(2)}}</span></div>
                        <div class="tooltip-row"><span class="tooltip-label">Connections</span><span class="tooltip-value">${{node.degree}}</span></div>
                        ${{node.isHub ? `<div class="tooltip-row"><span class="tooltip-label">Hub score</span><span class="tooltip-value">${{node.hubScore.toFixed(3)}}</span></div>` : ''}}
                    `;
                    tooltip.classList.add('visible');
                }} else {{
                    tooltip.classList.remove('visible');
                }}
            }})
            .onNodeClick(node => {{
                const distance = 100;
                const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
                Graph.cameraPosition(
                    {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
                    node,
                    1200
                );
            }});

        // Tooltip tracking
        document.addEventListener('mousemove', e => {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.left = (e.clientX + 16) + 'px';
            tooltip.style.top = (e.clientY + 16) + 'px';
        }});

        // Controls
        function toggleLabels() {{
            showLabels = !showLabels;
            document.getElementById('labelBtn').classList.toggle('active', showLabels);
            Graph.nodeThreeObject(Graph.nodeThreeObject());
        }}

        function resetCamera() {{
            Graph.cameraPosition({{ x: 0, y: 0, z: 350 }}, {{ x: 0, y: 0, z: 0 }}, 1200);
        }}

        // Smooth initial camera
        setTimeout(() => {{
            Graph.cameraPosition({{ x: 180, y: 120, z: 220 }});
        }}, 100);
    </script>
</body>
</html>'''

        # Save HTML
        filepath = self.figures_dir / "network_3d_interactive.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_template)
        self.logger.info(f"Saved {filepath.name}")

        return str(filepath)

    def run(self) -> Dict[str, Any]:
        """Generate all visualizations."""
        generated_figures = []
        failed_figures = []
        interactive_files = []

        # Generate each static figure
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

        # Generate interactive plots (Plotly)
        if self.config.get("generate_interactive", True):
            # Interactive volcano plot
            try:
                interactive_result = self._plot_volcano_interactive()
                if interactive_result:
                    interactive_files.append(interactive_result)
                    self.logger.info("Interactive volcano plot generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating interactive volcano: {e}")

            # Interactive 3D network
            try:
                network_3d_result = self._plot_network_3d_interactive()
                if network_3d_result:
                    interactive_files.append(network_3d_result)
                    self.logger.info("Interactive 3D network generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating 3D network: {e}")

        self.logger.info(f"Visualization Complete:")
        self.logger.info(f"  Static figures: {len(generated_figures)} files")
        self.logger.info(f"  Interactive: {len(interactive_files)} files")
        self.logger.info(f"  Failed/Skipped: {len(failed_figures)}")

        return {
            "figures_generated": generated_figures,
            "interactive_files": interactive_files,
            "failed_figures": failed_figures,
            "total_generated": len(generated_figures) + len(interactive_files)
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
