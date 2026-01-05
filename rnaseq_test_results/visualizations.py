#!/usr/bin/env python3
"""
RNA-seq Visualization Module

Provides publication-quality visualizations:
- Volcano Plot
- MA Plot
- Heatmap
- PCA Plot
- Network Graph

Author: BioInsight AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class RNAseqVisualizer:
    """RNA-seq 시각화 클래스"""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'up': '#E74C3C',      # Red
            'down': '#3498DB',    # Blue
            'ns': '#95A5A6',      # Gray
            'highlight': '#F39C12' # Orange
        }

    def volcano_plot(
        self,
        deg_results: pd.DataFrame,
        log2fc_col: str = 'log2FoldChange',
        padj_col: str = 'padj',
        gene_col: str = 'gene',
        log2fc_threshold: float = 1.0,
        padj_threshold: float = 0.05,
        highlight_genes: Optional[List[str]] = None,
        title: str = 'Volcano Plot',
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Volcano Plot: log2FC vs -log10(padj)

        Args:
            deg_results: DEG 결과 DataFrame
            log2fc_col: log2 fold change 컬럼명
            padj_col: adjusted p-value 컬럼명
            gene_col: 유전자명 컬럼
            log2fc_threshold: log2FC 기준값
            padj_threshold: p-value 기준값
            highlight_genes: 강조할 유전자 목록
            title: 그래프 제목
            figsize: 그래프 크기
            save: 파일 저장 여부

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate -log10(padj)
        df = deg_results.copy()
        df['neg_log10_padj'] = -np.log10(df[padj_col].clip(lower=1e-300))

        # Classify genes
        conditions = [
            (df[padj_col] < padj_threshold) & (df[log2fc_col] > log2fc_threshold),
            (df[padj_col] < padj_threshold) & (df[log2fc_col] < -log2fc_threshold),
        ]
        choices = ['up', 'down']
        df['regulation'] = np.select(conditions, choices, default='ns')

        # Plot non-significant
        ns_data = df[df['regulation'] == 'ns']
        ax.scatter(
            ns_data[log2fc_col], ns_data['neg_log10_padj'],
            c=self.colors['ns'], alpha=0.5, s=20, label='Not Significant'
        )

        # Plot upregulated
        up_data = df[df['regulation'] == 'up']
        ax.scatter(
            up_data[log2fc_col], up_data['neg_log10_padj'],
            c=self.colors['up'], alpha=0.7, s=30, label=f'Up ({len(up_data)})'
        )

        # Plot downregulated
        down_data = df[df['regulation'] == 'down']
        ax.scatter(
            down_data[log2fc_col], down_data['neg_log10_padj'],
            c=self.colors['down'], alpha=0.7, s=30, label=f'Down ({len(down_data)})'
        )

        # Threshold lines
        ax.axhline(y=-np.log10(padj_threshold), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=log2fc_threshold, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-log2fc_threshold, color='gray', linestyle='--', alpha=0.5)

        # Highlight specific genes
        if highlight_genes:
            highlight_df = df[df[gene_col].isin(highlight_genes)]
            ax.scatter(
                highlight_df[log2fc_col], highlight_df['neg_log10_padj'],
                c=self.colors['highlight'], s=100, marker='*',
                edgecolors='black', linewidths=0.5, label='Highlighted', zorder=5
            )

            # Add labels for highlighted genes
            for _, row in highlight_df.iterrows():
                ax.annotate(
                    row[gene_col],
                    (row[log2fc_col], row['neg_log10_padj']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold'
                )

        # Top genes annotation
        sig_df = df[df['regulation'] != 'ns'].nlargest(10, 'neg_log10_padj')
        for _, row in sig_df.iterrows():
            if highlight_genes is None or row[gene_col] not in highlight_genes:
                ax.annotate(
                    row[gene_col],
                    (row[log2fc_col], row['neg_log10_padj']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8
                )

        ax.set_xlabel('log₂ Fold Change', fontsize=12)
        ax.set_ylabel('-log₁₀ (Adjusted P-value)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'volcano_plot.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        return fig

    def ma_plot(
        self,
        deg_results: pd.DataFrame,
        basemean_col: str = 'baseMean',
        log2fc_col: str = 'log2FoldChange',
        padj_col: str = 'padj',
        gene_col: str = 'gene',
        log2fc_threshold: float = 1.0,
        padj_threshold: float = 0.05,
        title: str = 'MA Plot',
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        MA Plot: Mean expression vs log2FC
        """
        fig, ax = plt.subplots(figsize=figsize)

        df = deg_results.copy()
        df['log_basemean'] = np.log10(df[basemean_col] + 1)

        # Classify
        conditions = [
            (df[padj_col] < padj_threshold) & (df[log2fc_col] > log2fc_threshold),
            (df[padj_col] < padj_threshold) & (df[log2fc_col] < -log2fc_threshold),
        ]
        choices = ['up', 'down']
        df['regulation'] = np.select(conditions, choices, default='ns')

        # Plot
        for reg, color in [('ns', self.colors['ns']), ('up', self.colors['up']), ('down', self.colors['down'])]:
            data = df[df['regulation'] == reg]
            ax.scatter(
                data['log_basemean'], data[log2fc_col],
                c=color, alpha=0.6, s=20,
                label=f'{reg.capitalize()} ({len(data)})'
            )

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=log2fc_threshold, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-log2fc_threshold, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('log₁₀ (Mean Expression)', fontsize=12)
        ax.set_ylabel('log₂ Fold Change', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'ma_plot.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        return fig

    def heatmap(
        self,
        expression_matrix: pd.DataFrame,
        genes: Optional[List[str]] = None,
        metadata: Optional[pd.DataFrame] = None,
        condition_col: str = 'condition',
        n_top_genes: int = 50,
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        z_score: bool = True,
        title: str = 'Expression Heatmap',
        figsize: Tuple[int, int] = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Expression Heatmap with clustering
        """
        df = expression_matrix.copy()

        # Select genes
        if genes:
            genes_in_data = [g for g in genes if g in df.index]
            df = df.loc[genes_in_data]
        elif n_top_genes:
            # Select top variable genes
            gene_var = df.var(axis=1).nlargest(n_top_genes)
            df = df.loc[gene_var.index]

        # Z-score normalization
        if z_score:
            df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            df = df.fillna(0)

        # Prepare annotation colors
        col_colors = None
        if metadata is not None and condition_col in metadata.columns:
            conditions = metadata.set_index('sample_id')[condition_col]
            conditions = conditions[conditions.index.isin(df.columns)]

            unique_conditions = conditions.unique()
            color_palette = sns.color_palette("Set2", len(unique_conditions))
            condition_colors = dict(zip(unique_conditions, color_palette))
            col_colors = conditions.map(condition_colors)

        # Create clustermap
        g = sns.clustermap(
            df,
            cmap='RdBu_r',
            center=0,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            col_colors=col_colors,
            figsize=figsize,
            xticklabels=True,
            yticklabels=True,
            dendrogram_ratio=(0.1, 0.1),
            cbar_pos=(0.02, 0.8, 0.03, 0.15)
        )

        g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        # Adjust label sizes
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=45, ha='right')
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)

        if save:
            filepath = self.output_dir / 'heatmap.png'
            g.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        return g.fig

    def pca_plot(
        self,
        expression_matrix: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        condition_col: str = 'condition',
        n_components: int = 2,
        title: str = 'PCA Plot',
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        PCA Plot for sample clustering visualization
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data (samples x genes)
        X = expression_matrix.T.values
        X = np.log2(X + 1)  # Log transform
        X = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)

        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'sample': expression_matrix.columns
        })

        # Add condition info
        if metadata is not None and condition_col in metadata.columns:
            condition_map = metadata.set_index('sample_id')[condition_col].to_dict()
            pca_df['condition'] = pca_df['sample'].map(condition_map)

            # Plot with colors
            conditions = pca_df['condition'].unique()
            colors = sns.color_palette("Set1", len(conditions))

            for cond, color in zip(conditions, colors):
                mask = pca_df['condition'] == cond
                ax.scatter(
                    pca_df.loc[mask, 'PC1'],
                    pca_df.loc[mask, 'PC2'],
                    c=[color], s=100, label=cond, alpha=0.8,
                    edgecolors='white', linewidths=1
                )
        else:
            ax.scatter(
                pca_df['PC1'], pca_df['PC2'],
                c=self.colors['up'], s=100, alpha=0.8,
                edgecolors='white', linewidths=1
            )

        # Add sample labels
        for _, row in pca_df.iterrows():
            ax.annotate(
                row['sample'],
                (row['PC1'], row['PC2']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, alpha=0.7
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if metadata is not None:
            ax.legend(loc='best')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'pca_plot.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        return fig

    def network_plot(
        self,
        hub_genes: pd.DataFrame,
        network_graph=None,
        top_n: int = 30,
        gene_col: str = 'gene',
        score_col: str = 'composite_score',
        log2fc_col: str = 'log2FoldChange',
        title: str = 'Hub Gene Network',
        figsize: Tuple[int, int] = (12, 12),
        save: bool = True
    ) -> plt.Figure:
        """
        Network visualization of hub genes
        """
        import networkx as nx

        fig, ax = plt.subplots(figsize=figsize)

        # Get top hub genes
        top_hubs = hub_genes.nlargest(top_n, score_col)

        # Create network if not provided
        if network_graph is None:
            G = nx.Graph()
            genes = top_hubs[gene_col].tolist()
            G.add_nodes_from(genes)

            # Add edges based on score similarity
            for i, g1 in enumerate(genes):
                for j, g2 in enumerate(genes):
                    if i < j:
                        score1 = top_hubs[top_hubs[gene_col] == g1][score_col].values[0]
                        score2 = top_hubs[top_hubs[gene_col] == g2][score_col].values[0]
                        if abs(score1 - score2) < 0.3:
                            G.add_edge(g1, g2, weight=1 - abs(score1 - score2))
        else:
            # Subgraph of top genes
            genes = top_hubs[gene_col].tolist()
            G = network_graph.subgraph(genes).copy()

        # Node sizes based on score
        node_sizes = []
        for node in G.nodes():
            score = top_hubs[top_hubs[gene_col] == node][score_col].values
            if len(score) > 0:
                node_sizes.append(score[0] * 2000 + 200)
            else:
                node_sizes.append(200)

        # Node colors based on regulation
        node_colors = []
        for node in G.nodes():
            log2fc = top_hubs[top_hubs[gene_col] == node][log2fc_col].values
            if len(log2fc) > 0:
                if log2fc[0] > 0:
                    node_colors.append(self.colors['up'])
                else:
                    node_colors.append(self.colors['down'])
            else:
                node_colors.append(self.colors['ns'])

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, ax=ax)
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='white',
            linewidths=2,
            ax=ax
        )
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )

        # Legend
        up_patch = mpatches.Patch(color=self.colors['up'], label='Upregulated')
        down_patch = mpatches.Patch(color=self.colors['down'], label='Downregulated')
        ax.legend(handles=[up_patch, down_patch], loc='upper left')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'network_plot.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        return fig

    def summary_dashboard(
        self,
        deg_results: pd.DataFrame,
        hub_genes: pd.DataFrame,
        expression_matrix: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        title: str = 'RNA-seq Analysis Dashboard',
        save: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots
        """
        fig = plt.figure(figsize=(20, 16))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Volcano Plot (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._mini_volcano(ax1, deg_results)

        # 2. MA Plot (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        self._mini_ma(ax2, deg_results)

        # 3. DEG Summary (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._deg_summary_bar(ax3, deg_results)

        # 4. PCA Plot (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._mini_pca(ax4, expression_matrix, metadata)

        # 5. Top Hub Genes (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        self._hub_genes_bar(ax5, hub_genes)

        # 6. Network (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._mini_network(ax6, hub_genes)

        # 7. Heatmap (bottom - full width)
        ax7 = fig.add_subplot(gs[2, :])
        self._mini_heatmap(ax7, expression_matrix, deg_results, metadata)

        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

        if save:
            filepath = self.output_dir / 'dashboard.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        return fig

    def _mini_volcano(self, ax, deg_results):
        """Mini volcano plot for dashboard"""
        df = deg_results.copy()
        df['neg_log10_padj'] = -np.log10(df['padj'].clip(lower=1e-300))

        sig_up = (df['padj'] < 0.05) & (df['log2FoldChange'] > 1)
        sig_down = (df['padj'] < 0.05) & (df['log2FoldChange'] < -1)

        ax.scatter(df.loc[~sig_up & ~sig_down, 'log2FoldChange'],
                   df.loc[~sig_up & ~sig_down, 'neg_log10_padj'],
                   c=self.colors['ns'], alpha=0.3, s=10)
        ax.scatter(df.loc[sig_up, 'log2FoldChange'],
                   df.loc[sig_up, 'neg_log10_padj'],
                   c=self.colors['up'], alpha=0.6, s=15)
        ax.scatter(df.loc[sig_down, 'log2FoldChange'],
                   df.loc[sig_down, 'neg_log10_padj'],
                   c=self.colors['down'], alpha=0.6, s=15)

        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-1, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('log₂FC')
        ax.set_ylabel('-log₁₀(padj)')
        ax.set_title('Volcano Plot', fontweight='bold')

    def _mini_ma(self, ax, deg_results):
        """Mini MA plot for dashboard"""
        df = deg_results.copy()
        df['log_basemean'] = np.log10(df['baseMean'] + 1)

        sig = (df['padj'] < 0.05) & (abs(df['log2FoldChange']) > 1)

        ax.scatter(df.loc[~sig, 'log_basemean'], df.loc[~sig, 'log2FoldChange'],
                   c=self.colors['ns'], alpha=0.3, s=10)
        ax.scatter(df.loc[sig, 'log_basemean'], df.loc[sig, 'log2FoldChange'],
                   c=self.colors['up'], alpha=0.6, s=15)

        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('log₁₀(Mean Expression)')
        ax.set_ylabel('log₂FC')
        ax.set_title('MA Plot', fontweight='bold')

    def _deg_summary_bar(self, ax, deg_results):
        """DEG summary bar chart"""
        sig = deg_results[(deg_results['padj'] < 0.05) & (abs(deg_results['log2FoldChange']) > 1)]
        up = len(sig[sig['log2FoldChange'] > 0])
        down = len(sig[sig['log2FoldChange'] < 0])

        bars = ax.bar(['Upregulated', 'Downregulated'], [up, down],
                      color=[self.colors['up'], self.colors['down']])

        for bar, val in zip(bars, [up, down]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Number of DEGs')
        ax.set_title('DEG Summary', fontweight='bold')

    def _mini_pca(self, ax, expression_matrix, metadata):
        """Mini PCA plot for dashboard"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = np.log2(expression_matrix.T.values + 1)
        X = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        if metadata is not None and 'condition' in metadata.columns:
            conditions = metadata.set_index('sample_id')['condition']
            conditions = conditions[conditions.index.isin(expression_matrix.columns)]

            for cond in conditions.unique():
                mask = [conditions.get(s, '') == cond for s in expression_matrix.columns]
                ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                          label=cond, s=50, alpha=0.8)
            ax.legend(fontsize=8)
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], s=50, alpha=0.8)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA', fontweight='bold')

    def _hub_genes_bar(self, ax, hub_genes):
        """Top hub genes bar chart"""
        top = hub_genes.nlargest(15, 'composite_score')

        colors = [self.colors['up'] if x > 0 else self.colors['down']
                  for x in top['log2FoldChange']]

        bars = ax.barh(top['gene'], top['composite_score'], color=colors)
        ax.set_xlabel('Composite Score')
        ax.set_title('Top 15 Hub Genes', fontweight='bold')
        ax.invert_yaxis()

    def _mini_network(self, ax, hub_genes):
        """Mini network plot for dashboard"""
        import networkx as nx

        top = hub_genes.nlargest(20, 'composite_score')
        G = nx.Graph()
        genes = top['gene'].tolist()
        G.add_nodes_from(genes)

        # Add edges
        for i, g1 in enumerate(genes):
            for j, g2 in enumerate(genes):
                if i < j:
                    s1 = top[top['gene'] == g1]['composite_score'].values[0]
                    s2 = top[top['gene'] == g2]['composite_score'].values[0]
                    if abs(s1 - s2) < 0.2:
                        G.add_edge(g1, g2)

        pos = nx.spring_layout(G, k=1.5, seed=42)

        node_colors = [self.colors['up'] if top[top['gene'] == n]['log2FoldChange'].values[0] > 0
                       else self.colors['down'] for n in G.nodes()]

        nx.draw_networkx(G, pos, ax=ax, node_color=node_colors,
                        node_size=300, font_size=6, alpha=0.8)
        ax.set_title('Hub Gene Network', fontweight='bold')
        ax.axis('off')

    def _mini_heatmap(self, ax, expression_matrix, deg_results, metadata):
        """Mini heatmap for dashboard"""
        # Get top DEGs
        sig = deg_results[(deg_results['padj'] < 0.05) &
                          (abs(deg_results['log2FoldChange']) > 1)]
        top_genes = sig.nlargest(30, 'log2FoldChange')['gene'].tolist()

        genes_in_data = [g for g in top_genes if g in expression_matrix.index]
        if len(genes_in_data) < 10:
            genes_in_data = expression_matrix.index[:30].tolist()

        df = expression_matrix.loc[genes_in_data]

        # Z-score
        df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        df = df.fillna(0)

        sns.heatmap(df, cmap='RdBu_r', center=0, ax=ax,
                   xticklabels=True, yticklabels=True,
                   cbar_kws={'shrink': 0.5})

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        ax.set_title('Top DEGs Expression Heatmap', fontweight='bold')


def create_all_visualizations(
    deg_results: pd.DataFrame,
    hub_genes: pd.DataFrame,
    expression_matrix: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    output_dir: str = "figures"
) -> Dict[str, plt.Figure]:
    """
    Create all visualizations at once

    Returns:
        Dictionary of figure names to Figure objects
    """
    viz = RNAseqVisualizer(output_dir=output_dir)

    figures = {}

    logger.info("Creating visualizations...")

    # 1. Volcano Plot
    logger.info("  1/6 Volcano Plot...")
    figures['volcano'] = viz.volcano_plot(deg_results)

    # 2. MA Plot
    logger.info("  2/6 MA Plot...")
    figures['ma'] = viz.ma_plot(deg_results)

    # 3. PCA Plot
    logger.info("  3/6 PCA Plot...")
    figures['pca'] = viz.pca_plot(expression_matrix, metadata)

    # 4. Heatmap
    logger.info("  4/6 Heatmap...")
    sig_genes = deg_results[
        (deg_results['padj'] < 0.05) &
        (abs(deg_results['log2FoldChange']) > 1)
    ]['gene'].tolist()[:50]
    figures['heatmap'] = viz.heatmap(expression_matrix, genes=sig_genes, metadata=metadata)

    # 5. Network Plot
    logger.info("  5/6 Network Plot...")
    figures['network'] = viz.network_plot(hub_genes)

    # 6. Dashboard
    logger.info("  6/6 Dashboard...")
    figures['dashboard'] = viz.summary_dashboard(
        deg_results, hub_genes, expression_matrix, metadata
    )

    logger.info(f"✓ All visualizations saved to {output_dir}/")

    plt.close('all')

    return figures


if __name__ == "__main__":
    # Test with existing results
    import sys
    sys.path.insert(0, '/Users/admin/VectorDB_BioInsight')

    results_dir = Path('/Users/admin/VectorDB_BioInsight/rnaseq_test_results/test_run')

    deg_results = pd.read_csv(results_dir / 'deseq2_all_results.csv')
    hub_genes = pd.read_csv(results_dir / 'hub_genes.csv')
    expression_matrix = pd.read_csv(results_dir / 'normalized_counts.csv', index_col=0)
    metadata = pd.read_csv(results_dir / 'metadata.csv')

    figures = create_all_visualizations(
        deg_results, hub_genes, expression_matrix, metadata,
        output_dir=str(results_dir / 'figures')
    )

    print(f"Created {len(figures)} visualizations")
