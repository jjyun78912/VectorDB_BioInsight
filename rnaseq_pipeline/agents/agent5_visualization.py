"""
Agent 5: Visualization

Generates publication-quality figures from analysis results.

Input:
- deg_all_results.csv: From Agent 1
- deg_significant.csv: From Agent 1
- normalized_counts.csv: From Agent 1
- counts_for_pca.csv: From Agent 1 (variance-stabilized for PCA)
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
    """Agent for generating publication-quality visualizations.

    Style: npj Systems Biology and Applications journal style
    - Clean white backgrounds
    - Professional color palette (blue/red/gray)
    - Helvetica/Arial fonts
    - Minimal gridlines
    - Clear labels and legends
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "figure_format": ["png", "svg"],
            "dpi": 300,
            "style": "white",  # Clean white background
            "color_palette": "RdBu_r",
            "figsize": {
                "volcano": (10, 8),
                "heatmap": (12, 10),
                "pca": (8, 6),
                "network": (10, 10),
                "pathway": (10, 6),
                "summary": (12, 8),
                "boxplot": (10, 6)
            },
            "top_genes_heatmap": 50,
            "top_pathways": 15,
            "label_top_genes": 15,
            "generate_interactive": True,
            # npj color scheme
            "colors": {
                "up": "#c62828",        # Red for upregulated
                "down": "#1565c0",      # Blue for downregulated
                "ns": "#bdbdbd",        # Gray for non-significant
                "primary": "#0056b9",   # npj blue
                "accent": "#e87722",    # npj orange
                "text": "#212121",      # Dark gray text
                "grid": "#e0e0e0",      # Light grid
            }
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent5_visualization", input_dir, output_dir, merged_config)

        # Create figures subdirectory
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Set npj journal style
        self._set_npj_style()

    def _set_npj_style(self):
        """Set matplotlib style to match npj Systems Biology journal."""
        # Reset to default first
        plt.rcdefaults()

        # Set seaborn style
        sns.set_style("white")

        # npj-style parameters
        plt.rcParams.update({
            # Font settings (Helvetica-like)
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 11,

            # Axes
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'normal',
            'axes.linewidth': 1.0,
            'axes.edgecolor': '#424242',
            'axes.labelcolor': '#212121',
            'axes.spines.top': False,
            'axes.spines.right': False,

            # Ticks
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.color': '#424242',
            'ytick.color': '#424242',
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,

            # Legend
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 1.0,
            'legend.edgecolor': '#e0e0e0',

            # Figure
            'figure.facecolor': 'white',
            'figure.edgecolor': 'white',
            'figure.titlesize': 14,
            'figure.titleweight': 'bold',

            # Savefig
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'white',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,

            # Grid (minimal)
            'axes.grid': False,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
        })

    def validate_inputs(self) -> bool:
        """Validate input files."""
        # Required files
        self.deg_all = self.load_csv("deg_all_results.csv", required=False)
        self.deg_sig = self.load_csv("deg_significant.csv", required=False)
        self.norm_counts = self.load_csv("normalized_counts.csv", required=False)

        # Variance-stabilized counts for PCA (preferred over raw normalized counts)
        self.pca_counts = self.load_csv("counts_for_pca.csv", required=False)
        if self.pca_counts is not None:
            self.logger.info("Loaded variance-stabilized counts for PCA (counts_for_pca.csv)")
        else:
            self.logger.info("counts_for_pca.csv not found, will use normalized_counts.csv with log2 transform")

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
        """Generate npj-style volcano plot.

        Style based on npj Systems Biology and Applications:
        - Clean white background with minimal borders
        - Red (up) / Blue (down) / Gray (NS) color scheme
        - Thin dashed threshold lines
        - Clear gene labels with subtle styling
        """
        if self.deg_all is None:
            self.logger.warning("Skipping volcano plot - no DEG results")
            return None

        self.logger.info("Generating npj-style volcano plot...")

        colors = self.config["colors"]
        fig, ax = plt.subplots(figsize=self.config["figsize"]["volcano"])

        df = self.deg_all.copy()
        df['neg_log10_padj'] = -np.log10(df['padj'].clip(lower=1e-300))

        # Thresholds
        padj_cutoff = 0.05
        log2fc_cutoff = 1.0

        # Categorize genes
        df['significance'] = 'NS'
        df.loc[(df['padj'] < padj_cutoff) & (df['log2FC'] > log2fc_cutoff), 'significance'] = 'Up'
        df.loc[(df['padj'] < padj_cutoff) & (df['log2FC'] < -log2fc_cutoff), 'significance'] = 'Down'

        # npj color scheme
        color_map = {'NS': colors['ns'], 'Up': colors['up'], 'Down': colors['down']}

        # Plot in order: NS first (background), then significant
        for sig in ['NS', 'Down', 'Up']:
            subset = df[df['significance'] == sig]
            if sig == 'NS':
                ax.scatter(subset['log2FC'], subset['neg_log10_padj'],
                          c=color_map[sig], alpha=0.3, s=15, label=f'NS ({len(subset):,})',
                          edgecolors='none', rasterized=True)
            else:
                ax.scatter(subset['log2FC'], subset['neg_log10_padj'],
                          c=color_map[sig], alpha=0.7, s=25,
                          label=f'{sig} ({len(subset):,})',
                          edgecolors='white', linewidth=0.3)

        # Threshold lines (subtle dashed)
        ax.axhline(y=-np.log10(padj_cutoff), color='#757575', linestyle='--',
                  alpha=0.6, linewidth=0.8, zorder=0)
        ax.axvline(x=log2fc_cutoff, color='#757575', linestyle='--',
                  alpha=0.6, linewidth=0.8, zorder=0)
        ax.axvline(x=-log2fc_cutoff, color='#757575', linestyle='--',
                  alpha=0.6, linewidth=0.8, zorder=0)

        # Label top genes
        if self.deg_sig is not None and len(self.deg_sig) > 0:
            deg_sorted = self.deg_sig.copy()
            deg_sorted['abs_log2FC'] = deg_sorted['log2FC'].abs()
            deg_sorted = deg_sorted.sort_values('abs_log2FC', ascending=False)

            n_labels = self.config["label_top_genes"]
            top_up = deg_sorted[deg_sorted['direction'] == 'up'].head(n_labels // 2)
            top_down = deg_sorted[deg_sorted['direction'] == 'down'].head(n_labels // 2)
            top_genes = pd.concat([top_up, top_down])

            try:
                from adjustText import adjust_text
                texts = []

                for _, row in top_genes.iterrows():
                    gene_row = df[df['gene_id'] == row['gene_id']]
                    if len(gene_row) > 0:
                        x = gene_row['log2FC'].values[0]
                        y = gene_row['neg_log10_padj'].values[0]

                        if 'gene_symbol' in row and pd.notna(row['gene_symbol']):
                            label = row['gene_symbol']
                        else:
                            label = self._get_gene_symbol(row['gene_id'])

                        # npj style: italic gene names
                        text = ax.annotate(label, (x, y), fontsize=8,
                                          fontstyle='italic', fontweight='medium',
                                          ha='center', va='bottom',
                                          color=colors['text'])
                        texts.append(text)

                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='#9e9e9e',
                                                   alpha=0.5, lw=0.5),
                           expand_points=(1.5, 1.5))
            except ImportError:
                self.logger.warning("adjustText not installed - skipping labels")
            except Exception as e:
                self.logger.warning(f"Label adjustment failed: {e}")

        # Axis labels (npj style)
        ax.set_xlabel('log$_2$ Fold Change', fontsize=12)
        ax.set_ylabel('-log$_{10}$ Adjusted P-value', fontsize=12)

        # Title with panel label style
        ax.set_title('Differential Expression Analysis', fontsize=12, fontweight='bold',
                    loc='left', pad=10)

        # Legend (compact, top-right)
        legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.95,
                          edgecolor='#e0e0e0', fancybox=False)
        legend.get_frame().set_linewidth(0.5)

        # Summary stats (bottom-left, subtle)
        n_up = (df['significance'] == 'Up').sum()
        n_down = (df['significance'] == 'Down').sum()
        stats_text = f'n = {len(df):,} genes\nUp: {n_up:,} | Down: {n_down:,}'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=9, color='#616161', va='bottom',
               bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                        edgecolor='#e0e0e0', linewidth=0.5))

        # Clean up spines
        sns.despine(ax=ax)

        plt.tight_layout()
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

        # Note: Labels removed to avoid overlap - gene names shown on hover only

        # Update layout
        n_up = (df['significance'] == 'Upregulated').sum()
        n_down = (df['significance'] == 'Downregulated').sum()

        fig.update_layout(
            title=dict(
                text=f'<b>Interactive Volcano Plot</b><br><sup>↑ Up: {n_up} | ↓ Down: {n_down} | Total: {len(df):,}</sup>',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='log2 Fold Change',
            yaxis_title='-log10(padj)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd',
                borderwidth=1,
                font=dict(size=10)
            ),
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=500,
            margin=dict(l=60, r=30, t=80, b=60),
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
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'drawline', 'drawopenpath', 'eraseshape', 'select2d'],
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'volcano_plot_interactive',
                'height': 600,
                'width': 800,
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
        """Generate npj-style heatmap of top DEGs.

        Style based on npj Systems Biology:
        - Blue-White-Red diverging colormap
        - Clean condition annotation bar
        - Gene symbols on y-axis
        - Sample names with condition on x-axis
        - Clustered by expression pattern

        Generates two heatmaps:
        1. Full heatmap (Top 50 DEGs)
        2. Key genes heatmap (Hub genes + Known Drivers)
        """
        if self.norm_counts is None or self.deg_sig is None:
            self.logger.warning("Skipping heatmap - missing data")
            return None

        self.logger.info("Generating npj-style heatmaps...")

        colors = self.config["colors"]
        saved_files = []

        # Create custom colormap (Blue-White-Red)
        from matplotlib.colors import LinearSegmentedColormap
        npj_cmap = LinearSegmentedColormap.from_list('npj_heatmap',
            [colors['down'], '#ffffff', colors['up']])

        # ===== FULL HEATMAP (Top 50 DEGs) =====
        n_genes = min(self.config["top_genes_heatmap"], len(self.deg_sig))
        top_genes = self.deg_sig.head(n_genes)['gene_id'].tolist()

        full_files = self._create_heatmap(
            gene_list=top_genes,
            title=f'Expression Heatmap (Top {n_genes} DEGs)',
            filename='heatmap_top50',
            cmap=npj_cmap,
            colors=colors,
            show_sample_names=True
        )
        if full_files:
            saved_files.extend(full_files)

        # ===== KEY GENES HEATMAP (Hub + Drivers) =====
        key_genes = self._get_key_genes_for_heatmap()
        if key_genes and len(key_genes) >= 5:
            key_files = self._create_heatmap(
                gene_list=key_genes,
                title=f'Key Genes Heatmap (Hub Genes + Known Drivers, n={len(key_genes)})',
                filename='heatmap_key_genes',
                cmap=npj_cmap,
                colors=colors,
                show_sample_names=True,
                figsize=(14, 8)  # Wider for sample names
            )
            if key_files:
                saved_files.extend(key_files)

        return saved_files if saved_files else None

    def _get_key_genes_for_heatmap(self) -> List[str]:
        """Get key genes (Hub genes + Known Drivers) for focused heatmap."""
        key_genes = set()

        # Add Hub genes
        if self.hub_genes is not None and len(self.hub_genes) > 0:
            hub_ids = self.hub_genes['gene_id'].head(20).tolist()
            key_genes.update(hub_ids)
            self.logger.info(f"Added {len(hub_ids)} hub genes to key heatmap")

        # Add Known Drivers from db_matched_genes
        db_matched_path = self.input_dir / "db_matched_genes.csv"
        if db_matched_path.exists():
            try:
                db_df = pd.read_csv(db_matched_path)
                if 'gene_id' in db_df.columns:
                    driver_ids = db_df['gene_id'].head(15).tolist()
                    key_genes.update(driver_ids)
                    self.logger.info(f"Added {len(driver_ids)} driver genes to key heatmap")
            except Exception as e:
                self.logger.warning(f"Could not load db_matched_genes: {e}")

        # Add integrated genes with high scores
        integrated_path = self.input_dir / "integrated_gene_table.csv"
        if integrated_path.exists():
            try:
                int_df = pd.read_csv(integrated_path)
                if 'interpretation_score' in int_df.columns and 'gene_id' in int_df.columns:
                    high_score = int_df[int_df['interpretation_score'] >= 3.0]
                    high_score_ids = high_score['gene_id'].head(10).tolist()
                    key_genes.update(high_score_ids)
                    self.logger.info(f"Added {len(high_score_ids)} high-score genes to key heatmap")
            except Exception as e:
                self.logger.warning(f"Could not load integrated_gene_table: {e}")

        return list(key_genes)

    def _create_heatmap(self, gene_list: List[str], title: str, filename: str,
                        cmap, colors: dict, show_sample_names: bool = False,
                        figsize: tuple = None) -> Optional[List[str]]:
        """Create a single heatmap with given gene list."""
        # Filter expression data
        gene_col = self.norm_counts.columns[0]
        expr_df = self.norm_counts[self.norm_counts[gene_col].isin(gene_list)]
        expr_df = expr_df.set_index(gene_col)

        if len(expr_df) == 0:
            self.logger.warning(f"No matching genes for {filename}")
            return None

        # Sort samples by condition
        sample_order = list(expr_df.columns)
        if self.sample_to_condition:
            sample_order = sorted(
                sample_order,
                key=lambda x: (self.sample_to_condition.get(x, 'zzz'), x)
            )
            expr_df = expr_df[sample_order]

        # Z-score normalize (row-wise)
        expr_zscore = expr_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)

        # Get gene symbols
        gene_labels = []
        for gene_id in expr_zscore.index:
            symbol = self._get_gene_symbol(gene_id)
            gene_labels.append(symbol if symbol != gene_id else gene_id.split('.')[0][-8:])

        # Create sample labels with condition
        sample_labels = []
        for sample in expr_df.columns:
            cond = self.sample_to_condition.get(sample, 'Unknown') if self.sample_to_condition else ''
            # Shorten sample name if too long
            short_sample = sample[:15] + '...' if len(sample) > 18 else sample
            # Add condition indicator
            if cond.lower() in ['tumor', 'tumour', 'cancer']:
                sample_labels.append(f"{short_sample} (T)")
            elif cond.lower() == 'normal':
                sample_labels.append(f"{short_sample} (N)")
            else:
                sample_labels.append(f"{short_sample} ({cond[:3]})")

        # Prepare condition colors
        col_colors = None
        unique_conditions = []
        cond_palette = {}
        if self.sample_to_condition:
            conditions = [self.sample_to_condition.get(s, 'Unknown') for s in expr_df.columns]
            unique_conditions = sorted(set(conditions))
            cond_palette = {'tumor': colors['up'], 'normal': colors['down'],
                          'Tumor': colors['up'], 'Normal': colors['down']}
            for i, c in enumerate(unique_conditions):
                if c not in cond_palette:
                    cond_palette[c] = plt.cm.Set2(i)
            col_colors = pd.Series([cond_palette.get(c, '#9e9e9e') for c in conditions],
                                   index=expr_df.columns)

        # Determine figure size
        n_samples = len(expr_df.columns)
        n_genes = len(expr_df)
        if figsize is None:
            figsize = self.config["figsize"]["heatmap"]

        # Adjust figure size based on sample count for readability
        if show_sample_names and n_samples > 30:
            figsize = (max(figsize[0], n_samples * 0.25), figsize[1])

        # Create clustermap
        g = sns.clustermap(
            expr_zscore,
            cmap=cmap,
            center=0,
            vmin=-2, vmax=2,
            col_cluster=False,  # Don't cluster samples (keep condition order)
            row_cluster=True,   # Cluster genes
            col_colors=col_colors,
            yticklabels=gene_labels if n_genes <= 60 else False,
            xticklabels=sample_labels if show_sample_names else False,
            figsize=figsize,
            dendrogram_ratio=(0.1, 0.05),
            colors_ratio=0.02,
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            tree_kws={'linewidths': 0.5, 'colors': '#757575'}
        )

        # Style adjustments
        g.ax_heatmap.set_xlabel('Samples', fontsize=11)
        g.ax_heatmap.set_ylabel('Genes', fontsize=11)

        # Rotate x-axis labels for readability
        if show_sample_names:
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        # Colorbar label
        g.cax.set_ylabel('Z-score', fontsize=10, rotation=90)
        g.cax.yaxis.set_label_position('left')

        # Title
        g.fig.suptitle(title, fontsize=12, fontweight='bold', x=0.5, y=1.02)

        # Add condition legend if applicable
        if self.sample_to_condition and unique_conditions:
            from matplotlib.patches import Patch
            legend_patches = [Patch(facecolor=cond_palette.get(c, '#9e9e9e'),
                                   edgecolor='#424242', linewidth=0.5,
                                   label=c.capitalize()) for c in unique_conditions]
            g.ax_heatmap.legend(handles=legend_patches, loc='upper left',
                               bbox_to_anchor=(1.15, 1), fontsize=9,
                               title='Condition', title_fontsize=10,
                               frameon=True, edgecolor='#e0e0e0')

        plt.tight_layout()

        # Save using clustermap's figure
        saved_files = []
        for fmt in self.config["figure_format"]:
            filepath = self.figures_dir / f"{filename}.{fmt}"
            g.savefig(filepath, dpi=self.config["dpi"], bbox_inches='tight',
                     facecolor='white', edgecolor='none')
            saved_files.append(str(filepath))
            self.logger.info(f"Saved {filepath.name}")
        plt.close(g.fig)
        return saved_files

    def _plot_pca(self) -> Optional[List[str]]:
        """Generate PCA plot with condition-based coloring.

        Uses variance-stabilized counts (vst) for better PCA results.
        VST/rlog transformation stabilizes variance across mean expression levels,
        preventing high-expression genes from dominating PCA.
        """
        # Use variance-stabilized counts if available, otherwise fall back to normalized
        if self.pca_counts is not None:
            counts_for_pca = self.pca_counts
            self.logger.info("Using variance-stabilized counts for PCA")
        elif self.norm_counts is not None:
            counts_for_pca = self.norm_counts
            self.logger.info("Using normalized counts with log2 transform for PCA")
        else:
            self.logger.warning("Skipping PCA - no counts data available")
            return None

        self.logger.info("Generating PCA plot...")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Prepare data
        gene_col = counts_for_pca.columns[0]
        expr_df = counts_for_pca.set_index(gene_col)

        # If using raw normalized counts (not vst), apply log2 transform
        if self.pca_counts is None:
            self.logger.info("Applying log2(x+1) transformation for PCA")
            expr_df = np.log2(expr_df + 1)

        # Transpose (samples x genes)
        expr_t = expr_df.T

        # StandardScaler centers and scales: z = (x - mean) / std
        # This is appropriate after vst/log transform
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_t)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(expr_scaled)

        # Load metadata for condition coloring
        # Try multiple possible locations for metadata.csv
        possible_metadata_paths = [
            self.input_dir / "metadata.csv",                    # Current input dir
            self.input_dir.parent / "metadata.csv",             # Parent (run dir)
            self.input_dir.parent.parent / "metadata.csv",      # Grandparent (pipeline output)
            self.input_dir.parent.parent.parent / "metadata.csv",  # Original input
        ]

        metadata_path = None
        for path in possible_metadata_paths:
            if path.exists():
                metadata_path = path
                break

        # First, use pre-loaded sample_to_condition from validate_inputs
        sample_conditions = {}
        if self.sample_to_condition:
            sample_conditions = self.sample_to_condition.copy()
            self.logger.info(f"Using pre-loaded conditions: {len(sample_conditions)} samples")
        elif metadata_path and metadata_path.exists():
            try:
                metadata_df = pd.read_csv(metadata_path)
                for _, row in metadata_df.iterrows():
                    sample_id = row.get('sample_id', row.get('sample', ''))
                    condition = row.get('condition', row.get('group', 'unknown'))
                    sample_conditions[sample_id] = condition
                self.logger.info(f"Loaded metadata for {len(sample_conditions)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load metadata: {e}")

        # If no metadata, try to infer condition from TCGA barcode pattern
        # TCGA barcode format: TCGA-XX-XXXX-01A (01-09: Tumor, 10-19: Normal, 11: Solid Tissue Normal)
        if not sample_conditions:
            import re
            tcga_pattern = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})')
            inferred_count = 0
            for sample in expr_t.index:
                match = tcga_pattern.search(sample)
                if match:
                    sample_type = int(match.group(1))
                    if sample_type >= 10:  # Normal tissue (10-19)
                        sample_conditions[sample] = 'normal'
                    else:  # Tumor tissue (01-09)
                        sample_conditions[sample] = 'tumor'
                    inferred_count += 1
            if inferred_count > 0:
                self.logger.info(f"Inferred condition from TCGA barcode for {inferred_count} samples")

        # If still no conditions, try to infer from cancer_prediction.json
        # Samples in prediction are tumor, others are normal
        if not sample_conditions:
            import json
            prediction_paths = [
                self.input_dir / "cancer_prediction.json",
                self.input_dir.parent / "cancer_prediction.json",
                self.input_dir.parent.parent / "cancer_prediction.json",
            ]
            for pred_path in prediction_paths:
                if pred_path.exists():
                    try:
                        with open(pred_path) as f:
                            prediction = json.load(f)
                        # Get tumor samples from all_results or sample_count
                        tumor_samples = set()
                        if 'all_results' in prediction:
                            tumor_samples = {r['sample_id'] for r in prediction['all_results']}

                        # If all_results is incomplete, use sample_count heuristic
                        # Samples used in prediction are tumor, rest are normal
                        if not tumor_samples and 'sample_count' in prediction:
                            # Can't determine specific samples, skip
                            pass

                        if tumor_samples:
                            for sample in expr_t.index:
                                if sample in tumor_samples:
                                    sample_conditions[sample] = 'tumor'
                                else:
                                    sample_conditions[sample] = 'normal'
                            self.logger.info(f"Inferred condition from cancer_prediction.json: {len(tumor_samples)} tumor samples")
                            break
                    except Exception as e:
                        self.logger.warning(f"Could not load cancer_prediction.json: {e}")

        # Define colors for conditions
        condition_colors = {
            'tumor': '#dc2626',      # Red
            'cancer': '#dc2626',     # Red
            'case': '#dc2626',       # Red
            'treatment': '#dc2626',  # Red
            'normal': '#2563eb',     # Blue
            'control': '#2563eb',    # Blue
            'healthy': '#2563eb',    # Blue
        }
        default_color = '#6b7280'  # Gray for unknown

        # Plot
        fig, ax = plt.subplots(figsize=self.config["figsize"]["pca"])

        # Get colors for each sample
        colors = []
        for sample in expr_t.index:
            condition = sample_conditions.get(sample, 'unknown').lower()
            color = condition_colors.get(condition, default_color)
            colors.append(color)

        # Scatter plot with colors
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                            c=colors, s=120, alpha=0.8, edgecolors='white', linewidths=1.5)

        # Label points - only for small datasets, otherwise use interactive version
        n_samples = len(expr_t.index)
        if n_samples <= 20:
            # Show all labels for small datasets
            for i, sample in enumerate(expr_t.index):
                ax.annotate(sample, (pca_result[i, 0], pca_result[i, 1]),
                           fontsize=8, ha='center', va='bottom')
        elif n_samples <= 50:
            # Show shortened labels for medium datasets
            for i, sample in enumerate(expr_t.index):
                # Shorten TCGA sample IDs: TCGA-XX-XXXX-01A-... -> XX-XXXX
                if sample.startswith('TCGA-'):
                    parts = sample.split('-')
                    short_label = f"{parts[1]}-{parts[2]}" if len(parts) >= 3 else sample[:12]
                else:
                    short_label = sample[:12]
                ax.annotate(short_label, (pca_result[i, 0], pca_result[i, 1]),
                           fontsize=7, ha='center', va='bottom', alpha=0.7)
        # For large datasets (>50), no labels - use interactive HTML version for hover

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA: Sample Distribution')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        # Add legend if we have conditions
        if sample_conditions:
            unique_conditions = list(set(sample_conditions.values()))
            legend_elements = []
            from matplotlib.patches import Patch
            for cond in sorted(unique_conditions):
                color = condition_colors.get(cond.lower(), default_color)
                legend_elements.append(Patch(facecolor=color, edgecolor='white',
                                            label=cond.capitalize()))
            ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

        saved_files = self._save_figure(fig, "pca_plot")

        # Generate interactive PCA plot (Plotly) for hover functionality
        if HAS_PLOTLY:
            try:
                interactive_path = self._plot_pca_interactive(
                    pca_result, expr_t.index, sample_conditions,
                    condition_colors, default_color, pca
                )
                if interactive_path:
                    saved_files.append(interactive_path)
            except Exception as e:
                self.logger.warning(f"Interactive PCA failed: {e}")

        return saved_files

    def _plot_pca_interactive(self, pca_result, sample_names, sample_conditions,
                               condition_colors, default_color, pca) -> Optional[str]:
        """Generate interactive PCA plot with Plotly for hover sample identification."""
        self.logger.info("Generating interactive PCA plot (Plotly)...")

        # Prepare data
        conditions = []
        colors_list = []
        for sample in sample_names:
            cond = sample_conditions.get(sample, 'unknown')
            conditions.append(cond.capitalize())
            colors_list.append(condition_colors.get(cond.lower(), default_color))

        # Create figure
        fig = go.Figure()

        # Add traces for each condition
        unique_conds = list(set(conditions))
        plotly_colors = {
            'Tumor': '#dc2626', 'Normal': '#2563eb', 'Cancer': '#dc2626',
            'Control': '#2563eb', 'Unknown': '#6b7280'
        }

        for cond in sorted(unique_conds):
            mask = [c == cond for c in conditions]
            indices = [i for i, m in enumerate(mask) if m]

            fig.add_trace(go.Scatter(
                x=[pca_result[i, 0] for i in indices],
                y=[pca_result[i, 1] for i in indices],
                mode='markers',
                name=f'{cond} ({len(indices)})',
                marker=dict(
                    size=12,
                    color=plotly_colors.get(cond, '#6b7280'),
                    line=dict(width=1, color='white')
                ),
                text=[sample_names[i] for i in indices],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))

        # Layout - compact size to fit report width
        fig.update_layout(
            title=dict(
                text=f'<b>PCA: 샘플 분포</b><br><sup>마우스를 올리면 샘플 ID 확인</sup>',
                x=0.5,
                font=dict(size=14)
            ),
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd', borderwidth=1,
                font=dict(size=11)
            ),
            hovermode='closest',
            template='plotly_white',
            width=700, height=400,
            margin=dict(l=50, r=30, t=60, b=50)
        )

        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)

        # Save
        html_path = self.figures_dir / "pca_interactive.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        self.logger.info(f"Saved interactive PCA: {html_path.name}")

        return str(html_path)

    def _plot_network(self) -> Optional[List[str]]:
        """Generate npj-style network visualization.

        Style based on npj Systems Biology:
        - Clean white background
        - Simple node colors (red/blue by direction)
        - Hub genes highlighted with larger size and bold labels
        - Minimal, professional styling
        """
        if self.network_edges is None or len(self.network_edges) == 0:
            self.logger.warning("Skipping network plot - no edges")
            return None

        if self.hub_genes is None or len(self.hub_genes) == 0:
            self.logger.warning("Skipping network plot - no hub genes")
            return None

        self.logger.info("Generating npj-style network plot...")

        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("networkx not installed - skipping network plot")
            return None

        colors = self.config["colors"]

        # Build gene mappings
        gene_id_to_symbol = {}
        gene_id_to_direction = {}
        if self.integrated_table is not None:
            for _, row in self.integrated_table.iterrows():
                gid = row['gene_id']
                if pd.notna(row.get('gene_symbol')) and row['gene_symbol']:
                    gene_id_to_symbol[gid] = row['gene_symbol']
                gene_id_to_direction[gid] = row.get('direction', 'none')

        def get_gene_label(gene_id: str) -> str:
            if gene_id in gene_id_to_symbol:
                return gene_id_to_symbol[gene_id]
            if gene_id.startswith('ENSG'):
                return gene_id.split('.')[0][-6:]
            return gene_id

        # Build graph
        G = nx.Graph()
        for _, row in self.network_edges.iterrows():
            label1 = get_gene_label(row['gene1'])
            label2 = get_gene_label(row['gene2'])
            G.add_edge(label1, label2, weight=row['abs_correlation'])
            G.nodes[label1]['gene_id'] = row['gene1']
            G.nodes[label2]['gene_id'] = row['gene2']

        if G.number_of_nodes() == 0:
            return None

        # Hub gene info
        hub_set_ids = set(self.hub_genes['gene_id'].tolist())
        hub_set_labels = {get_gene_label(h) for h in hub_set_ids}

        # Limit to hub-centered subgraph
        if G.number_of_nodes() > 80:
            nodes_to_keep = set()
            for hub_label in hub_set_labels:
                if hub_label in G:
                    nodes_to_keep.add(hub_label)
                    nodes_to_keep.update(list(G.neighbors(hub_label))[:8])
            G = G.subgraph(nodes_to_keep).copy()

        # npj-style figure
        self._set_npj_style()
        fig, ax = plt.subplots(figsize=self.config["figsize"]["network"])

        # Layout
        pos = nx.kamada_kawai_layout(G)

        # Node attributes
        node_colors_list = []
        node_sizes = []
        degrees = dict(G.degree())

        for n in G.nodes():
            gene_id = G.nodes[n].get('gene_id', '')
            direction = gene_id_to_direction.get(gene_id, 'none')
            is_hub = n in hub_set_labels

            # Color by direction
            if direction == 'up':
                node_colors_list.append(colors['up'])
            elif direction == 'down':
                node_colors_list.append(colors['down'])
            else:
                node_colors_list.append(colors['ns'])

            # Size by hub status
            if is_hub:
                node_sizes.append(600)
            else:
                node_sizes.append(100 + degrees.get(n, 1) * 20)

        # Draw edges (subtle gray)
        edge_weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edge_color='#bdbdbd',
                              width=[0.5 + w for w in edge_weights],
                              alpha=0.5)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_color=node_colors_list,
                              node_size=node_sizes,
                              edgecolors='white',
                              linewidths=1.5)

        # Labels
        hub_labels = {n: n for n in G.nodes() if n in hub_set_labels}
        other_labels = {n: n for n in G.nodes() if n not in hub_set_labels}

        # Hub labels: bold, larger
        nx.draw_networkx_labels(G, pos, hub_labels, ax=ax,
                               font_size=10, font_weight='bold',
                               font_color=colors['text'])

        # Other labels: smaller, lighter
        nx.draw_networkx_labels(G, pos, other_labels, ax=ax,
                               font_size=7, font_weight='normal',
                               font_color='#757575')

        # Title
        ax.set_title('Gene Co-expression Network', fontsize=12, fontweight='bold',
                    loc='left', pad=15)

        # Stats annotation
        stats_text = f'{G.number_of_nodes()} genes | {G.number_of_edges()} edges | {len(hub_set_labels)} hubs'
        ax.text(0.5, -0.02, stats_text, transform=ax.transAxes,
               ha='center', fontsize=9, color='#616161')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='white', markerfacecolor=colors['up'],
                   markersize=10, markeredgecolor='white', label='Upregulated'),
            Line2D([0], [0], marker='o', color='white', markerfacecolor=colors['down'],
                   markersize=10, markeredgecolor='white', label='Downregulated'),
            Line2D([0], [0], marker='o', color='white', markerfacecolor=colors['ns'],
                   markersize=6, markeredgecolor='white', label='Other'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                 frameon=True, edgecolor='#e0e0e0', fancybox=False)

        ax.axis('off')
        ax.margins(0.1)

        plt.tight_layout()
        return self._save_figure(fig, "network_graph")

    def _plot_network_2d_interactive(self) -> Optional[str]:
        """Generate 2D interactive network visualization with Plotly (hover for gene names)."""
        if not HAS_PLOTLY:
            self.logger.warning("Plotly not installed - skipping interactive 2D network")
            return None

        if self.network_edges is None or len(self.network_edges) == 0:
            self.logger.warning("Skipping interactive 2D network - no edges")
            return None

        if self.hub_genes is None or len(self.hub_genes) == 0:
            self.logger.warning("Skipping interactive 2D network - no hub genes")
            return None

        self.logger.info("Generating interactive 2D network (Plotly)...")

        try:
            import networkx as nx
        except ImportError:
            return None

        colors = self.config["colors"]

        # Build gene mappings
        gene_id_to_symbol = {}
        gene_id_to_direction = {}
        gene_id_to_log2fc = {}
        if self.integrated_table is not None:
            for _, row in self.integrated_table.iterrows():
                gid = row['gene_id']
                if pd.notna(row.get('gene_symbol')) and row['gene_symbol']:
                    gene_id_to_symbol[gid] = row['gene_symbol']
                gene_id_to_direction[gid] = row.get('direction', 'none')
                gene_id_to_log2fc[gid] = row.get('log2FC', 0)

        def get_gene_label(gene_id: str) -> str:
            if gene_id in gene_id_to_symbol:
                return gene_id_to_symbol[gene_id]
            if gene_id.startswith('ENSG'):
                return gene_id.split('.')[0][-6:]
            return gene_id

        # Build graph
        G = nx.Graph()
        for _, row in self.network_edges.iterrows():
            label1 = get_gene_label(row['gene1'])
            label2 = get_gene_label(row['gene2'])
            G.add_edge(label1, label2, weight=row['abs_correlation'])
            G.nodes[label1]['gene_id'] = row['gene1']
            G.nodes[label2]['gene_id'] = row['gene2']

        if G.number_of_nodes() == 0:
            return None

        # Hub gene info
        hub_set_ids = set(self.hub_genes['gene_id'].tolist())
        hub_set_labels = {get_gene_label(h) for h in hub_set_ids}

        # Limit to hub-centered subgraph
        if G.number_of_nodes() > 80:
            nodes_to_keep = set()
            for hub_label in hub_set_labels:
                if hub_label in G:
                    nodes_to_keep.add(hub_label)
                    nodes_to_keep.update(list(G.neighbors(hub_label))[:8])
            G = G.subgraph(nodes_to_keep).copy()

        # Layout
        pos = nx.kamada_kawai_layout(G)

        # Create edge traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#bdbdbd'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces by direction
        fig = go.Figure()
        fig.add_trace(edge_trace)

        # Group nodes by direction
        nodes_by_direction = {'up': [], 'down': [], 'other': []}
        for node in G.nodes():
            gene_id = G.nodes[node].get('gene_id', '')
            direction = gene_id_to_direction.get(gene_id, 'none')
            if direction == 'up':
                nodes_by_direction['up'].append(node)
            elif direction == 'down':
                nodes_by_direction['down'].append(node)
            else:
                nodes_by_direction['other'].append(node)

        # Add traces for each direction
        direction_config = {
            'up': {'color': colors['up'], 'name': 'Upregulated'},
            'down': {'color': colors['down'], 'name': 'Downregulated'},
            'other': {'color': colors['ns'], 'name': 'Other'}
        }

        for direction, nodes in nodes_by_direction.items():
            if not nodes:
                continue

            node_x, node_y, node_text, node_sizes = [], [], [], []
            for node in nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                gene_id = G.nodes[node].get('gene_id', '')
                log2fc = gene_id_to_log2fc.get(gene_id, 0)
                degree = G.degree(node)
                is_hub = node in hub_set_labels

                hover_text = f"<b>{node}</b><br>log2FC: {log2fc:.2f}<br>Connections: {degree}"
                if is_hub:
                    hover_text += "<br><b>★ Hub Gene</b>"
                node_text.append(hover_text)

                # Size by hub status
                if is_hub:
                    node_sizes.append(25)
                else:
                    node_sizes.append(12)

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                name=direction_config[direction]['name'],
                marker=dict(
                    size=node_sizes,
                    color=direction_config[direction]['color'],
                    line=dict(width=1, color='white')
                ),
                text=node_text,
                hoverinfo='text',
                hovertemplate='%{text}<extra></extra>'
            ))

        # Layout
        fig.update_layout(
            title=dict(
                text=f'<b>Gene Co-expression Network</b><br><sup>{G.number_of_nodes()} genes | {G.number_of_edges()} edges | Hover for details</sup>',
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor='rgba(255,255,255,0.9)'
            ),
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=750,
            height=550,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # Save
        html_path = self.figures_dir / "network_2d_interactive.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        self.logger.info(f"Saved interactive 2D network: {html_path.name}")

        return str(html_path)

    def _plot_pathway_barplot(self) -> Optional[List[str]]:
        """Generate npj-style pathway enrichment barplot.

        Style based on npj Systems Biology:
        - Horizontal bars with gradient colors
        - Gene count annotations
        - Clean axis styling
        - Color by significance level
        """
        if self.pathway_summary is None or len(self.pathway_summary) == 0:
            self.logger.warning("Skipping pathway plot - no pathway data")
            return None

        self.logger.info("Generating npj-style pathway barplot...")

        colors = self.config["colors"]

        # Get top pathways
        n_pathways = min(self.config["top_pathways"], len(self.pathway_summary))
        top_pathways = self.pathway_summary.head(n_pathways).copy()

        if len(top_pathways) == 0:
            return None

        # -log10 transform p-values
        top_pathways['neg_log10_padj'] = -np.log10(top_pathways['padj'].clip(lower=1e-300))

        # Truncate long pathway names (npj style: max 50 chars)
        top_pathways['term_short'] = top_pathways['term_name'].apply(
            lambda x: x[:50] + '...' if len(str(x)) > 50 else x
        )

        # Reverse for plotting (most significant at top)
        top_pathways = top_pathways.iloc[::-1]

        # Create figure
        fig, ax = plt.subplots(figsize=self.config["figsize"]["pathway"])

        # Color gradient based on significance (blue scale)
        n = len(top_pathways)
        bar_colors = [plt.cm.Blues(0.4 + 0.5 * i / n) for i in range(n)]

        # Plot horizontal bars
        y_pos = np.arange(n)
        bars = ax.barh(y_pos, top_pathways['neg_log10_padj'],
                      color=bar_colors, edgecolor='white', linewidth=0.5,
                      height=0.7)

        # Y-axis labels (pathway names)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_pathways['term_short'], fontsize=9)

        # X-axis
        ax.set_xlabel('-log$_{10}$ Adjusted P-value', fontsize=11)

        # Title (npj style: left-aligned, bold)
        ax.set_title('Pathway Enrichment Analysis', fontsize=12, fontweight='bold',
                    loc='left', pad=10)

        # Add gene count annotations at end of bars
        for i, (_, row) in enumerate(top_pathways.iterrows()):
            gene_count = int(row['gene_count'])
            x_pos = row['neg_log10_padj']

            # Gene count label
            ax.text(x_pos + 0.2, i, f'n={gene_count}',
                   va='center', ha='left', fontsize=8, color='#616161')

        # Add vertical line at significance threshold
        sig_line = -np.log10(0.05)
        ax.axvline(x=sig_line, color='#757575', linestyle='--',
                  linewidth=0.8, alpha=0.6, zorder=0)
        ax.text(sig_line, n - 0.5, 'p=0.05', fontsize=8, color='#757575',
               ha='right', va='bottom')

        # Clean up
        ax.set_xlim(0, top_pathways['neg_log10_padj'].max() * 1.15)
        sns.despine(ax=ax)

        # Add subtle grid on x-axis only
        ax.xaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return self._save_figure(fig, "pathway_barplot")

    def _plot_interpretation_summary(self) -> Optional[List[str]]:
        """Generate npj-style interpretation summary.

        Style: Clean 2x2 panel layout with:
        - Bar chart for confidence distribution (not pie)
        - Grouped bar for DB match vs Hub status
        - Histogram for score distribution
        - Clean table for top genes
        """
        if self.integrated_table is None or len(self.integrated_table) == 0:
            self.logger.warning("Skipping interpretation summary - no data")
            return None

        self.logger.info("Generating npj-style interpretation summary...")

        colors = self.config["colors"]

        fig, axes = plt.subplots(2, 2, figsize=self.config["figsize"]["summary"])
        fig.suptitle('Analysis Summary', fontsize=14, fontweight='bold', y=1.02)

        # === Panel A: Confidence Distribution (bar chart, not pie) ===
        ax1 = axes[0, 0]
        conf_counts = self.integrated_table['confidence'].value_counts()

        conf_colors = {
            'high': '#2e7d32',
            'medium': '#f57c00',
            'low': '#c62828',
            'novel_candidate': '#7b1fa2',
            'requires_validation': '#757575'
        }

        bar_cols = [conf_colors.get(c, '#9e9e9e') for c in conf_counts.index]
        bars = ax1.bar(range(len(conf_counts)), conf_counts.values, color=bar_cols,
                      edgecolor='white', linewidth=0.5)
        ax1.set_xticks(range(len(conf_counts)))
        ax1.set_xticklabels([c.replace('_', '\n') for c in conf_counts.index],
                           fontsize=8, rotation=0)
        ax1.set_ylabel('Number of Genes', fontsize=10)
        ax1.set_title('a) Confidence Distribution', fontsize=11, fontweight='bold',
                     loc='left', pad=8)

        # Add count labels on bars
        for bar, count in zip(bars, conf_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontsize=8)

        sns.despine(ax=ax1)

        # === Panel B: DB Match vs Hub Status ===
        ax2 = axes[0, 1]
        categories = ['DB+Hub', 'DB Only', 'Hub Only', 'Neither']
        counts = [
            ((self.integrated_table['db_matched']) & (self.integrated_table['is_hub'])).sum(),
            ((self.integrated_table['db_matched']) & (~self.integrated_table['is_hub'])).sum(),
            ((~self.integrated_table['db_matched']) & (self.integrated_table['is_hub'])).sum(),
            ((~self.integrated_table['db_matched']) & (~self.integrated_table['is_hub'])).sum()
        ]
        bar_colors_b = [colors['primary'], colors['down'], colors['accent'], colors['ns']]
        bars2 = ax2.bar(categories, counts, color=bar_colors_b, edgecolor='white', linewidth=0.5)
        ax2.set_ylabel('Number of Genes', fontsize=10)
        ax2.set_title('b) Validation Status', fontsize=11, fontweight='bold',
                     loc='left', pad=8)
        ax2.tick_params(axis='x', labelsize=9)

        for bar, count in zip(bars2, counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        str(count), ha='center', va='bottom', fontsize=8)

        sns.despine(ax=ax2)

        # === Panel C: Score Distribution (histogram) ===
        ax3 = axes[1, 0]
        scores = self.integrated_table['interpretation_score']
        ax3.hist(scores, bins=20, color=colors['primary'], edgecolor='white',
                linewidth=0.5, alpha=0.8)
        ax3.axvline(x=5, color='#2e7d32', linestyle='--', linewidth=1.5,
                   label='High (≥5)')
        ax3.axvline(x=3, color='#f57c00', linestyle='--', linewidth=1.5,
                   label='Medium (≥3)')
        ax3.set_xlabel('Interpretation Score', fontsize=10)
        ax3.set_ylabel('Number of Genes', fontsize=10)
        ax3.set_title('c) Score Distribution', fontsize=11, fontweight='bold',
                     loc='left', pad=8)
        ax3.legend(fontsize=8, frameon=True, edgecolor='#e0e0e0')
        sns.despine(ax=ax3)

        # === Panel D: Top Genes Table ===
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Prepare data
        top_10 = self.integrated_table.head(10).copy()

        # Get gene symbols if available
        gene_names = []
        for _, row in top_10.iterrows():
            if 'gene_symbol' in row and pd.notna(row['gene_symbol']):
                gene_names.append(row['gene_symbol'])
            else:
                gene_names.append(self._get_gene_symbol(row['gene_id']))

        table_data = []
        for i, (_, row) in enumerate(top_10.iterrows()):
            table_data.append([
                gene_names[i],
                f"{row['log2FC']:.2f}",
                '●' if row['is_hub'] else '',
                '●' if row['db_matched'] else '',
                row['confidence'][:3].upper()
            ])

        # Create table
        table = ax4.table(
            cellText=table_data,
            colLabels=['Gene', 'log₂FC', 'Hub', 'DB', 'Conf'],
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.15, 0.12, 0.12, 0.15]
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        # Header styling
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#f5f5f5')
                cell.set_text_props(fontweight='bold', fontsize=9)
            cell.set_edgecolor('#e0e0e0')
            cell.set_linewidth(0.5)

        ax4.set_title('d) Top 10 Genes', fontsize=11, fontweight='bold',
                     loc='left', pad=8, x=0.1)

        plt.tight_layout()
        return self._save_figure(fig, "interpretation_summary")

    def _plot_expression_boxplot(self) -> Optional[List[str]]:
        """Generate npj-style box plot for top DEGs expression.

        Shows expression distribution across conditions for top genes.
        Style: Clean box plots with individual points overlay.
        """
        if self.norm_counts is None or self.deg_sig is None:
            self.logger.warning("Skipping boxplot - missing data")
            return None

        if not self.sample_to_condition:
            self.logger.warning("Skipping boxplot - no condition metadata")
            return None

        self.logger.info("Generating npj-style expression boxplot...")

        colors = self.config["colors"]

        # Get top 8 genes
        n_genes = min(8, len(self.deg_sig))
        top_genes = self.deg_sig.head(n_genes)

        # Prepare data
        gene_col = self.norm_counts.columns[0]
        plot_data = []

        for _, row in top_genes.iterrows():
            gene_id = row['gene_id']
            if 'gene_symbol' in row and pd.notna(row['gene_symbol']):
                gene_name = row['gene_symbol']
            else:
                gene_name = self._get_gene_symbol(gene_id)

            expr_row = self.norm_counts[self.norm_counts[gene_col] == gene_id]
            if len(expr_row) == 0:
                continue

            for sample in expr_row.columns[1:]:
                condition = self.sample_to_condition.get(sample, 'Unknown')
                expr_val = expr_row[sample].values[0]
                plot_data.append({
                    'Gene': gene_name,
                    'Expression': np.log2(expr_val + 1),  # log2 transform
                    'Condition': condition.capitalize()
                })

        if not plot_data:
            return None

        df_plot = pd.DataFrame(plot_data)

        # Create figure
        fig, ax = plt.subplots(figsize=self.config["figsize"]["boxplot"])

        # Condition colors
        conditions = df_plot['Condition'].unique()
        cond_palette = {}
        for cond in conditions:
            if 'tumor' in cond.lower():
                cond_palette[cond] = colors['up']
            elif 'normal' in cond.lower():
                cond_palette[cond] = colors['down']
            else:
                cond_palette[cond] = colors['ns']

        # Box plot with strip plot overlay
        sns.boxplot(data=df_plot, x='Gene', y='Expression', hue='Condition',
                   ax=ax, palette=cond_palette, width=0.6,
                   linewidth=1, fliersize=0)

        # Add individual points
        sns.stripplot(data=df_plot, x='Gene', y='Expression', hue='Condition',
                     ax=ax, palette=cond_palette, dodge=True,
                     size=3, alpha=0.5, legend=False)

        # Styling
        ax.set_xlabel('')
        ax.set_ylabel('Expression (log$_2$ normalized counts)', fontsize=11)
        ax.set_title('Expression of Top DEGs by Condition', fontsize=12,
                    fontweight='bold', loc='left', pad=10)

        # Rotate x labels
        plt.xticks(rotation=45, ha='right', fontsize=10, style='italic')

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(conditions)], labels[:len(conditions)],
                 loc='upper right', fontsize=9, frameon=True,
                 edgecolor='#e0e0e0', title='Condition', title_fontsize=9)

        sns.despine(ax=ax)
        plt.tight_layout()

        return self._save_figure(fig, "expression_boxplot")

    def _plot_network_3d_interactive(self) -> Optional[str]:
        """Generate CSN (Cell-Specific Network) paper style network visualization.

        Features (based on Fig.11 CSN paper):
        - Clean white background
        - Hub genes: Large purple/magenta circles
        - Non-hub genes: Small rounded rectangles with labels
        - Blue edges: Activation (positive correlation)
        - Purple/pink edges: Repression (negative correlation)
        - All nodes have gene name labels
        """
        if self.network_edges is None or len(self.network_edges) == 0:
            self.logger.warning("Skipping network - no edges")
            return None

        if self.hub_genes is None or len(self.hub_genes) == 0:
            self.logger.warning("Skipping network - no hub genes")
            return None

        self.logger.info("Generating CSN paper-style network...")

        try:
            import networkx as nx
            import json
        except ImportError:
            self.logger.warning("networkx not installed - skipping network")
            return None

        # Build gene_id to gene_symbol mapping
        gene_id_to_symbol = {}
        if self.integrated_table is not None:
            for _, row in self.integrated_table.iterrows():
                if pd.notna(row.get('gene_symbol')) and row['gene_symbol']:
                    gene_id_to_symbol[row['gene_id']] = row['gene_symbol']

        # Get log2FC and direction for edge coloring
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

        # Build graph - Only hub genes and their direct neighbors
        G = nx.Graph()
        hub_set_ids = set(self.hub_genes['gene_id'].tolist())

        for _, row in self.network_edges.iterrows():
            gene1, gene2 = row['gene1'], row['gene2']
            # Only include edges connected to hub genes
            if gene1 in hub_set_ids or gene2 in hub_set_ids:
                label1 = get_gene_label(gene1)
                label2 = get_gene_label(gene2)
                # Store correlation sign for edge color (positive=activation, negative=repression)
                correlation = row.get('correlation', row.get('abs_correlation', 0.5))
                G.add_edge(label1, label2,
                          weight=row['abs_correlation'],
                          correlation=correlation)
                G.nodes[label1]['gene_id'] = gene1
                G.nodes[label2]['gene_id'] = gene2

        if G.number_of_nodes() == 0:
            return None

        # Get hub labels
        hub_set_labels = {get_gene_label(h) for h in hub_set_ids}
        hub_scores = {}
        for _, row in self.hub_genes.iterrows():
            label = get_gene_label(row['gene_id'])
            hub_scores[label] = row.get('hub_score', 0.5)

        # Limit to top hub genes and their neighbors for clarity
        if G.number_of_nodes() > 100:
            nodes_to_keep = set()
            for hub_label in list(hub_set_labels)[:20]:  # Top 20 hubs
                if hub_label in G:
                    nodes_to_keep.add(hub_label)
                    nodes_to_keep.update(list(G.neighbors(hub_label))[:8])  # 8 neighbors each
            G = G.subgraph(nodes_to_keep).copy()

        # Prepare data for visualization
        nodes_data = []
        for node in G.nodes():
            is_hub = node in hub_set_labels
            gene_id = G.nodes[node].get('gene_id', '')
            info = gene_info.get(gene_id, {})
            log2fc = info.get('log2FC', 0)
            direction = info.get('direction', 'none')
            degree = G.degree(node)

            nodes_data.append({
                'id': node,
                'isHub': is_hub,
                'log2FC': round(log2fc, 2),
                'direction': direction,
                'degree': degree
            })

        links_data = []
        for edge in G.edges(data=True):
            correlation = edge[2].get('correlation', 0.5)
            # Positive correlation = activation (blue), Negative = repression (purple)
            edge_type = 'activation' if correlation > 0 else 'repression'
            links_data.append({
                'source': edge[0],
                'target': edge[1],
                'weight': round(edge[2].get('weight', 0.5), 3),
                'type': edge_type
            })

        graph_data = json.dumps({'nodes': nodes_data, 'links': links_data})
        hub_count = len([n for n in nodes_data if n['isHub']])

        # CSN paper style - white background, purple hubs, rounded rect non-hubs
        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Gene Network - CSN Paper Style</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #ffffff;
            font-family: Arial, Helvetica, sans-serif;
            overflow: hidden;
        }}

        svg#network {{
            display: block;
            background: #ffffff;
        }}

        /* Legend - CSN paper style */
        .legend {{
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 10px 25px;
            font-size: 12px;
            color: #333333;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 30px;
        }}
        .legend-title {{
            font-weight: bold;
            color: #000000;
            margin-right: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-line {{
            width: 30px;
            height: 3px;
        }}
        .legend-line.activation {{
            background: #4A90D9;
        }}
        .legend-line.repression {{
            background: #9B59B6;
        }}

        /* Stats */
        .stats {{
            position: fixed;
            top: 20px;
            left: 20px;
            background: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 12px 18px;
            font-size: 12px;
            color: #666666;
            z-index: 100;
        }}
        .stats-title {{
            font-weight: bold;
            color: #333333;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .stats-row {{
            margin: 4px 0;
        }}
        .stats span {{
            color: #000000;
            font-weight: bold;
        }}

        /* Tooltip */
        .tooltip {{
            position: absolute;
            background: #ffffff;
            border: 1px solid #999999;
            border-radius: 4px;
            padding: 10px 14px;
            font-size: 11px;
            color: #333333;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s ease;
            z-index: 200;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip-title {{
            font-weight: bold;
            font-size: 12px;
            color: #000000;
            margin-bottom: 5px;
        }}
        .tooltip-row {{
            margin: 2px 0;
        }}
        .tooltip-up {{ color: #c0392b; }}
        .tooltip-down {{ color: #2980b9; }}
    </style>
</head>
<body>
    <svg id="network"></svg>

    <div class="stats">
        <div class="stats-title">Gene Co-expression Network</div>
        <div class="stats-row"><span>{len(nodes_data)}</span> nodes</div>
        <div class="stats-row"><span>{len(links_data)}</span> edges</div>
        <div class="stats-row"><span>{hub_count}</span> hub genes</div>
    </div>

    <div class="legend">
        <span class="legend-title">Interaction</span>
        <div class="legend-item">
            <div class="legend-line activation"></div>
            <span>Activation</span>
        </div>
        <div class="legend-item">
            <div class="legend-line repression"></div>
            <span>Repression</span>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        const data = {graph_data};
        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#network")
            .attr("width", width)
            .attr("height", height);

        const container = svg.append("g");

        // Enable zoom and pan
        svg.call(d3.zoom()
            .scaleExtent([0.2, 5])
            .on("zoom", (event) => {{
                container.attr("transform", event.transform);
            }}));

        // Force simulation - spread out like CSN paper
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(80).strength(0.4))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.isHub ? 35 : 45));

        // Draw links - CSN style: blue for activation, purple for repression
        const link = container.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("stroke", d => d.type === 'activation' ? '#4A90D9' : '#9B59B6')
            .attr("stroke-opacity", 0.7)
            .attr("stroke-width", 1.5);

        // Create node groups
        const nodeGroup = container.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .style("cursor", "pointer")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);

        // Draw nodes - CSN style
        // Hub genes: Large purple/magenta circles
        // Non-hub genes: Small rounded rectangles
        nodeGroup.each(function(d) {{
            const g = d3.select(this);

            if (d.isHub) {{
                // Hub gene: Large purple circle
                g.append("circle")
                    .attr("r", 22)
                    .attr("fill", "#8E44AD")  // Purple for hub genes
                    .attr("stroke", "#5B2C6F")
                    .attr("stroke-width", 2);

                // Hub label inside circle (white text)
                g.append("text")
                    .attr("text-anchor", "middle")
                    .attr("dy", "0.35em")
                    .attr("fill", "#ffffff")
                    .attr("font-size", "9px")
                    .attr("font-weight", "bold")
                    .text(d.id.length > 8 ? d.id.substring(0, 7) + '...' : d.id);
            }} else {{
                // Non-hub gene: Small rounded rectangle with label
                const labelWidth = Math.max(d.id.length * 6 + 10, 40);
                const labelHeight = 18;

                g.append("rect")
                    .attr("x", -labelWidth / 2)
                    .attr("y", -labelHeight / 2)
                    .attr("width", labelWidth)
                    .attr("height", labelHeight)
                    .attr("rx", 3)  // Rounded corners
                    .attr("ry", 3)
                    .attr("fill", "#ffffff")
                    .attr("stroke", "#7FB3D5")  // Light blue border
                    .attr("stroke-width", 1);

                // Non-hub label (blue text)
                g.append("text")
                    .attr("text-anchor", "middle")
                    .attr("dy", "0.35em")
                    .attr("fill", "#2980B9")  // Blue text
                    .attr("font-size", "8px")
                    .text(d.id.length > 10 ? d.id.substring(0, 9) + '...' : d.id);
            }}
        }});

        // Tooltip
        const tooltip = document.getElementById('tooltip');

        function showTooltip(event, d) {{
            const dirClass = d.direction === 'up' ? 'tooltip-up' : 'tooltip-down';
            const dirText = d.direction === 'up' ? '↑ Up' : '↓ Down';

            tooltip.innerHTML = `
                <div class="tooltip-title">${{d.id}}</div>
                <div class="tooltip-row">${{d.isHub ? '● Hub Gene' : '○ Connected Gene'}}</div>
                <div class="tooltip-row">log2FC: <span class="${{dirClass}}">${{d.log2FC > 0 ? '+' : ''}}${{d.log2FC}}</span></div>
                <div class="tooltip-row">Direction: <span class="${{dirClass}}">${{dirText}}</span></div>
                <div class="tooltip-row">Connections: ${{d.degree}}</div>
            `;

            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
            tooltip.classList.add('visible');
        }}

        function hideTooltip() {{
            tooltip.classList.remove('visible');
        }}

        // Update positions
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodeGroup
                .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        // Drag functions
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
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

        # Generate each static figure (npj style)
        figure_functions = [
            ("volcano_plot", self._plot_volcano),
            ("heatmap", self._plot_heatmap),
            ("pca_plot", self._plot_pca),
            ("network_graph", self._plot_network),
            ("pathway_barplot", self._plot_pathway_barplot),
            ("expression_boxplot", self._plot_expression_boxplot),  # NEW: npj-style boxplot
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

            # Interactive 2D network (hover-only labels)
            try:
                network_2d_result = self._plot_network_2d_interactive()
                if network_2d_result:
                    interactive_files.append(network_2d_result)
                    self.logger.info("Interactive 2D network generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating 2D network: {e}")

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
