"""
Bulk RNA-seq Specific Section Components

Sections unique to Bulk RNA-seq analysis:
- DEG Analysis (Volcano, Heatmap)
- Pathway Enrichment
- Network Analysis (Hub genes)
"""

from typing import Optional
import pandas as pd

from .common import BaseSection
from ..base_report import ReportConfig, ReportData


class DEGSection(BaseSection):
    """Differential Expression Gene analysis section."""

    section_id = "deg"
    section_number = 3
    section_icon = "🧬"
    section_title = "차등발현 분석 (DEG)"
    section_title_en = "Differential Expression Analysis"

    def is_available(self) -> bool:
        return self.data.deg_significant is not None and len(self.data.deg_significant) > 0

    def render(self) -> str:
        if not self.is_available():
            return ""

        deg_df = self.data.deg_significant
        fc_col = 'log2FC' if 'log2FC' in deg_df.columns else 'log2FoldChange'

        n_total = len(deg_df)
        n_up = len(deg_df[deg_df[fc_col] > 0]) if fc_col in deg_df.columns else 0
        n_down = n_total - n_up

        # Summary stats
        stats_html = f'''
        <div class="summary-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 1.5rem;">
            <div class="summary-card highlight">
                <div class="value">{n_total:,}</div>
                <div class="label">Total DEGs</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #dc2626;">{n_up:,}</div>
                <div class="label">Upregulated ↑</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #2563eb;">{n_down:,}</div>
                <div class="label">Downregulated ↓</div>
            </div>
        </div>
        '''

        # Figures
        figures_html = '<div class="figure-grid">'

        # Volcano plot - Prioritize interactive version
        volcano_fig = None
        volcano_interactive_key = None

        # First check for interactive versions
        for key in ['volcano_interactive', 'volcano_plot', 'volcano']:
            if key in self.data.interactive_figures:
                volcano_interactive_key = key
                break

        # Fallback to static
        if not volcano_interactive_key:
            for key in ['volcano_plot', 'volcano']:
                if key in self.data.figures:
                    volcano_fig = self.data.figures[key]
                    break

        if volcano_interactive_key:
            figures_html += f'''
            <div class="figure-panel">
                <div class="figure-title">Volcano Plot (Interactive - 호버로 유전자 확인)</div>
                <div class="interactive-container" style="min-height: 500px;">
                    {self.data.interactive_figures[volcano_interactive_key]}
                </div>
                <div class="figure-caption">
                    X축: log2 Fold Change, Y축: -log10(adjusted p-value).
                    빨간색: 상향 발현, 파란색: 하향 발현. <strong>마우스 호버로 유전자 정보 확인</strong>
                </div>
            </div>
            '''
        elif volcano_fig:
            figures_html += f'''
            <div class="figure-panel">
                <div class="figure-title">Volcano Plot</div>
                <div class="figure-content">
                    <img src="{volcano_fig}" alt="Volcano Plot">
                </div>
                <div class="figure-caption">
                    빨간색: 상향 발현 (log2FC > 1), 파란색: 하향 발현 (log2FC < -1)
                </div>
            </div>
            '''

        # Heatmap - Prioritize interactive version
        heatmap_fig = None
        heatmap_interactive_key = None

        # Check for interactive versions
        for key in ['heatmap_interactive', 'heatmap_top50_interactive', 'deg_heatmap_interactive']:
            if key in self.data.interactive_figures:
                heatmap_interactive_key = key
                break

        # Fallback to static
        if not heatmap_interactive_key:
            for key in ['heatmap', 'heatmap_top50', 'deg_heatmap']:
                if key in self.data.figures:
                    heatmap_fig = self.data.figures[key]
                    break

        if heatmap_interactive_key:
            figures_html += f'''
            <div class="figure-panel">
                <div class="figure-title">Expression Heatmap (Interactive - 호버로 확인)</div>
                <div class="interactive-container" style="min-height: 600px;">
                    {self.data.interactive_figures[heatmap_interactive_key]}
                </div>
                <div class="figure-caption">
                    상위 50개 DEG의 샘플별 발현 패턴. Z-score 정규화 적용. <strong>마우스 호버로 유전자/샘플 정보 확인</strong>
                </div>
            </div>
            '''
        elif heatmap_fig:
            figures_html += f'''
            <div class="figure-panel">
                <div class="figure-title">Expression Heatmap (Top 50 DEGs)</div>
                <div class="figure-content">
                    <img src="{heatmap_fig}" alt="Heatmap">
                </div>
                <div class="figure-caption">
                    상위 50개 DEG의 샘플별 발현 패턴. Z-score 정규화 적용.
                </div>
            </div>
            '''

        figures_html += '</div>'

        # Top DEGs table
        gene_col = 'gene_symbol' if 'gene_symbol' in deg_df.columns else 'gene_id'
        top_df = deg_df.nlargest(10, fc_col) if fc_col in deg_df.columns else deg_df.head(10)

        table_rows = []
        for _, row in top_df.iterrows():
            gene = row.get(gene_col, 'N/A')
            fc = row.get(fc_col, 0)
            pval = row.get('padj', row.get('pvalue', 0))
            direction = '<span class="badge badge-up">↑ Up</span>' if fc > 0 else '<span class="badge badge-down">↓ Down</span>'

            table_rows.append(f'''
            <tr>
                <td><span class="gene-symbol">{gene}</span></td>
                <td>{fc:.2f}</td>
                <td>{pval:.2e}</td>
                <td>{direction}</td>
            </tr>
            ''')

        table_html = f'''
        <h3>Top 10 Upregulated Genes</h3>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>Gene</th><th>log2FC</th><th>Adj. P-value</th><th>Direction</th></tr>
                </thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
        </div>
        '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {stats_html}
            {figures_html}
            {table_html}
        </section>
        '''


class VolcanoSection(BaseSection):
    """Standalone Volcano plot section (if needed separately)."""

    section_id = "volcano"
    section_icon = "🌋"
    section_title = "Volcano Plot"
    section_title_en = "Volcano Plot"

    def is_available(self) -> bool:
        return ('volcano_plot' in self.data.figures or
                'volcano' in self.data.figures or
                'volcano_interactive' in self.data.interactive_figures)

    def render(self) -> str:
        # Usually included in DEGSection, so this is optional
        return ""


class HeatmapSection(BaseSection):
    """Standalone Heatmap section (if needed separately)."""

    section_id = "heatmap"
    section_icon = "🔥"
    section_title = "Expression Heatmap"
    section_title_en = "Expression Heatmap"

    def is_available(self) -> bool:
        return 'heatmap' in self.data.figures or 'heatmap_top50' in self.data.figures

    def render(self) -> str:
        # Usually included in DEGSection, so this is optional
        return ""


class PathwaySection(BaseSection):
    """Pathway enrichment analysis section."""

    section_id = "pathway"
    section_number = 4
    section_icon = "🛤️"
    section_title = "경로 및 기능 분석"
    section_title_en = "Pathway & Functional Analysis"

    def is_available(self) -> bool:
        return self.data.pathway_summary is not None and len(self.data.pathway_summary) > 0

    def render(self) -> str:
        if not self.is_available():
            return ""

        pathway_df = self.data.pathway_summary

        # Figures
        figures_html = '<div class="figure-grid">'

        for key in ['pathway_barplot', 'go_barplot', 'kegg_barplot', 'enrichment_plot']:
            if key in self.data.figures:
                figures_html += f'''
                <div class="figure-panel">
                    <div class="figure-title">Pathway Enrichment</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="Pathway Enrichment">
                    </div>
                </div>
                '''
                break

        figures_html += '</div>'

        # Pathway table
        table_rows = []
        for _, row in pathway_df.head(15).iterrows():
            term = row.get('Term', row.get('term', row.get('pathway', 'N/A')))
            pval = row.get('Adjusted P-value', row.get('padj', row.get('pvalue', 0)))
            genes = row.get('Genes', row.get('genes', row.get('overlap_genes', '')))
            source = row.get('Gene_set', row.get('source', row.get('database', 'N/A')))

            # Truncate term if too long
            if len(str(term)) > 60:
                term = str(term)[:57] + '...'

            # Truncate genes
            if isinstance(genes, str) and len(genes) > 50:
                genes = genes[:47] + '...'

            table_rows.append(f'''
            <tr>
                <td>{term}</td>
                <td>{source}</td>
                <td>{pval:.2e}</td>
                <td style="font-size: 0.8rem;">{genes}</td>
            </tr>
            ''')

        table_html = f'''
        <h3>Top Enriched Pathways</h3>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>Term</th><th>Source</th><th>Adj. P-value</th><th>Genes</th></tr>
                </thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
        </div>
        '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {figures_html}
            {table_html}
        </section>
        '''


class NetworkSection(BaseSection):
    """Network analysis and hub genes section."""

    section_id = "network"
    section_number = 6
    section_icon = "🕸️"
    section_title = "네트워크 분석"
    section_title_en = "Network Analysis"

    def is_available(self) -> bool:
        return self.data.hub_genes is not None and len(self.data.hub_genes) > 0

    def render(self) -> str:
        if not self.is_available():
            return ""

        hub_df = self.data.hub_genes

        # Hub genes table
        gene_col = 'gene_symbol' if 'gene_symbol' in hub_df.columns else 'gene'
        table_rows = []

        for _, row in hub_df.head(20).iterrows():
            gene = row.get(gene_col, row.get('gene_id', 'N/A'))
            degree = row.get('degree', row.get('Degree', 'N/A'))
            betweenness = row.get('betweenness', row.get('Betweenness', 'N/A'))
            hub_score = row.get('hub_score', row.get('Hub_Score', 'N/A'))

            # Format numbers
            if isinstance(betweenness, float):
                betweenness = f"{betweenness:.4f}"
            if isinstance(hub_score, float):
                hub_score = f"{hub_score:.2f}"

            table_rows.append(f'''
            <tr>
                <td><span class="gene-symbol">{gene}</span></td>
                <td>{degree}</td>
                <td>{betweenness}</td>
                <td>{hub_score}</td>
            </tr>
            ''')

        table_html = f'''
        <h3>Hub Genes (Top 20)</h3>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>Gene</th><th>Degree</th><th>Betweenness</th><th>Hub Score</th></tr>
                </thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
        </div>
        '''

        # Network figure
        network_html = ""

        # Check for interactive 3D network
        for key in ['network_3d_interactive', 'network_3d', 'ppi_network']:
            if key in self.data.interactive_figures:
                network_html = f'''
                <h3>Interactive Network Visualization</h3>
                <div class="interactive-container" style="min-height: 600px;">
                    {self.data.interactive_figures[key]}
                </div>
                <div class="figure-caption">
                    3D 네트워크 시각화. 노드 크기: Hub score, 엣지: 상관관계 강도
                </div>
                '''
                break

        if not network_html:
            for key in ['network_plot', 'network', 'ppi_network']:
                if key in self.data.figures:
                    network_html = f'''
                    <div class="figure-panel">
                        <div class="figure-title">Gene Co-expression Network</div>
                        <div class="figure-content">
                            <img src="{self.data.figures[key]}" alt="Network">
                        </div>
                    </div>
                    '''
                    break

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {table_html}
            {network_html}
            <div class="note-box info">
                <span class="note-box-icon">ℹ️</span>
                <div class="note-box-content">
                    Hub genes는 네트워크에서 중심성이 높은 유전자로, 조절자 역할을 할 가능성이 있습니다.
                    그러나 이는 발현 기반 예측이며, driver gene과는 구별됩니다.
                </div>
            </div>
        </section>
        '''
