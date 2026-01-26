"""
Single-cell RNA-seq Specific Section Components

Sections unique to Single-cell analysis:
- Cell Type Analysis (UMAP, Composition)
- Marker Genes
- Trajectory Analysis
- TME Analysis
- GRN Analysis
- Ploidy Inference
- Cell-Cell Interaction
"""

from typing import Optional
import pandas as pd

from .common import BaseSection
from ..base_report import ReportConfig, ReportData


class CellTypeSection(BaseSection):
    """Cell type analysis section (UMAP + Composition)."""

    section_id = "celltype"
    section_number = 3
    section_icon = "🔬"
    section_title = "세포 유형 분석"
    section_title_en = "Cell Type Analysis"

    def is_available(self) -> bool:
        return (self.data.cell_composition is not None or
                'umap_celltypes' in self.data.figures or
                'umap_clusters' in self.data.figures)

    def render(self) -> str:
        if not self.is_available():
            return ""

        figures_html = '<div class="figure-grid">'

        # UMAP plots
        for key, title in [
            ('umap_celltypes', 'Cell Types (UMAP)'),
            ('umap_clusters', 'Clusters (UMAP)'),
            ('umap', 'UMAP Visualization'),
        ]:
            if key in self.data.figures:
                figures_html += f'''
                <div class="figure-panel">
                    <div class="figure-title">{title}</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="{title}">
                    </div>
                </div>
                '''

        # Cell type bar chart
        if 'celltype_barchart' in self.data.figures:
            figures_html += f'''
            <div class="figure-panel">
                <div class="figure-title">Cell Type Composition</div>
                <div class="figure-content">
                    <img src="{self.data.figures['celltype_barchart']}" alt="Cell Type Composition">
                </div>
            </div>
            '''

        figures_html += '</div>'

        # Composition table
        table_html = ""
        if self.data.cell_composition is not None and len(self.data.cell_composition) > 0:
            df = self.data.cell_composition

            # Determine column names
            celltype_col = next((c for c in ['cell_type', 'celltype', 'predicted_labels'] if c in df.columns), None)
            count_col = next((c for c in ['count', 'n_cells', 'cell_count'] if c in df.columns), None)
            pct_col = next((c for c in ['percentage', 'percent', 'proportion'] if c in df.columns), None)

            if celltype_col:
                rows = []
                for _, row in df.head(15).iterrows():
                    celltype = row.get(celltype_col, 'N/A')
                    count = row.get(count_col, 'N/A') if count_col else 'N/A'
                    pct = row.get(pct_col, 'N/A') if pct_col else 'N/A'

                    if isinstance(pct, float):
                        pct = f"{pct:.1f}%"
                    if isinstance(count, float):
                        count = f"{int(count):,}"

                    rows.append(f'''
                    <tr>
                        <td>{celltype}</td>
                        <td>{count}</td>
                        <td>{pct}</td>
                    </tr>
                    ''')

                table_html = f'''
                <h3>Cell Type Composition</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Cell Type</th><th>Count</th><th>Percentage</th></tr>
                        </thead>
                        <tbody>{''.join(rows)}</tbody>
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


class MarkerSection(BaseSection):
    """Cluster marker genes section."""

    section_id = "markers"
    section_number = 4
    section_icon = "🧬"
    section_title = "클러스터별 마커 유전자"
    section_title_en = "Cluster Marker Genes"

    def is_available(self) -> bool:
        return self.data.cluster_markers is not None and len(self.data.cluster_markers) > 0

    def render(self) -> str:
        if not self.is_available():
            return ""

        # Figures
        figures_html = '<div class="figure-grid">'

        for key, title in [
            ('heatmap_markers', 'Marker Heatmap'),
            ('dotplot_markers', 'Marker Dotplot'),
            ('violin_markers', 'Marker Violin Plot'),
        ]:
            if key in self.data.figures:
                figures_html += f'''
                <div class="figure-panel">
                    <div class="figure-title">{title}</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="{title}">
                    </div>
                </div>
                '''

        figures_html += '</div>'

        # Marker genes table
        df = self.data.cluster_markers

        # Detect column names
        gene_col = next((c for c in ['names', 'gene', 'gene_symbol'] if c in df.columns), 'names')
        cluster_col = next((c for c in ['group', 'cluster', 'leiden'] if c in df.columns), 'group')
        score_col = next((c for c in ['scores', 'logfoldchanges', 'avg_log2FC'] if c in df.columns), None)
        pval_col = next((c for c in ['pvals_adj', 'p_val_adj', 'padj'] if c in df.columns), None)

        # Get top markers per cluster
        clusters = df[cluster_col].unique() if cluster_col in df.columns else []
        cluster_tables = []

        for cluster in list(clusters)[:6]:  # Max 6 clusters to show
            cluster_df = df[df[cluster_col] == cluster].head(5)

            rows = []
            for _, row in cluster_df.iterrows():
                gene = row.get(gene_col, 'N/A')
                score = row.get(score_col, 'N/A') if score_col else 'N/A'
                pval = row.get(pval_col, 'N/A') if pval_col else 'N/A'

                if isinstance(score, float):
                    score = f"{score:.2f}"
                if isinstance(pval, float):
                    pval = f"{pval:.2e}"

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{gene}</span></td>
                    <td>{score}</td>
                    <td>{pval}</td>
                </tr>
                ''')

            cluster_tables.append(f'''
            <div class="marker-cluster-card">
                <h4>Cluster {cluster}</h4>
                <table>
                    <thead><tr><th>Gene</th><th>Score</th><th>P-adj</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            ''')

        table_html = f'''
        <h3>Top Markers per Cluster</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
            {''.join(cluster_tables)}
        </div>
        '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {figures_html}
            {table_html}
        </section>
        '''


class TrajectorySection(BaseSection):
    """Trajectory / Pseudotime analysis section."""

    section_id = "trajectory"
    section_icon = "🔄"
    section_title = "Trajectory 분석 (Pseudotime)"
    section_title_en = "Trajectory Analysis"

    def is_available(self) -> bool:
        return (self.data.trajectory_pseudotime is not None or
                'trajectory' in self.data.figures or
                'pseudotime' in self.data.figures)

    def render(self) -> str:
        if not self.is_available():
            return ""

        figures_html = ""
        for key in ['trajectory', 'pseudotime', 'paga', 'trajectory_umap']:
            if key in self.data.figures:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">Trajectory Analysis</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="Trajectory">
                    </div>
                    <div class="figure-caption">
                        세포 분화 경로 및 pseudotime 분석 결과
                    </div>
                </div>
                '''
                break

        if not figures_html:
            figures_html = '<p class="text-muted">Trajectory 시각화를 사용할 수 없습니다.</p>'

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {figures_html}
        </section>
        '''


class TMESection(BaseSection):
    """Tumor Microenvironment analysis section."""

    section_id = "tme"
    section_icon = "🏔️"
    section_title = "종양 미세환경 (TME) 분석"
    section_title_en = "Tumor Microenvironment Analysis"

    def is_available(self) -> bool:
        return self.data.tme_composition is not None

    def render(self) -> str:
        if not self.is_available():
            return ""

        df = self.data.tme_composition

        # Figures
        figures_html = ""
        for key in ['tme_composition', 'tme_umap', 'immune_infiltration']:
            if key in self.data.figures:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">TME Composition</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="TME">
                    </div>
                </div>
                '''
                break

        # TME composition table
        table_rows = []
        for _, row in df.head(10).iterrows():
            celltype = row.get('cell_type', row.get('component', 'N/A'))
            fraction = row.get('fraction', row.get('proportion', row.get('percentage', 0)))

            if isinstance(fraction, float):
                fraction = f"{fraction:.1%}" if fraction < 1 else f"{fraction:.1f}%"

            table_rows.append(f'<tr><td>{celltype}</td><td>{fraction}</td></tr>')

        table_html = f'''
        <h3>TME Cell Composition</h3>
        <div class="table-wrapper">
            <table>
                <thead><tr><th>Cell Type</th><th>Fraction</th></tr></thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
        </div>
        '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {figures_html}
            {table_html}
            <div class="note-box info">
                <span class="note-box-icon">ℹ️</span>
                <div class="note-box-content">
                    TME 구성은 면역치료 반응 예측에 중요한 정보를 제공합니다.
                </div>
            </div>
        </section>
        '''


class GRNSection(BaseSection):
    """Gene Regulatory Network section."""

    section_id = "grn"
    section_icon = "🧬"
    section_title = "유전자 조절 네트워크 (GRN)"
    section_title_en = "Gene Regulatory Network"

    def is_available(self) -> bool:
        return self.data.grn_edges is not None or 'grn_network' in self.data.figures

    def render(self) -> str:
        if not self.is_available():
            return ""

        figures_html = ""
        for key in ['grn_network', 'tf_activity', 'regulon_heatmap']:
            if key in self.data.figures:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">Gene Regulatory Network</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="GRN">
                    </div>
                </div>
                '''
                break

        table_html = ""
        if self.data.grn_edges is not None and len(self.data.grn_edges) > 0:
            df = self.data.grn_edges.head(15)

            rows = []
            for _, row in df.iterrows():
                tf = row.get('TF', row.get('source', 'N/A'))
                target = row.get('target', row.get('gene', 'N/A'))
                weight = row.get('weight', row.get('importance', 'N/A'))

                if isinstance(weight, float):
                    weight = f"{weight:.3f}"

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{tf}</span></td>
                    <td><span class="gene-symbol">{target}</span></td>
                    <td>{weight}</td>
                </tr>
                ''')

            table_html = f'''
            <h3>Top Regulatory Interactions</h3>
            <div class="table-wrapper">
                <table>
                    <thead><tr><th>TF</th><th>Target</th><th>Weight</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
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


class PloidySection(BaseSection):
    """Ploidy inference / CNV analysis section."""

    section_id = "ploidy"
    section_icon = "📊"
    section_title = "Ploidy 추론 (악성세포 감별)"
    section_title_en = "Ploidy Inference"

    def is_available(self) -> bool:
        return (self.data.ploidy_by_celltype is not None or
                'cnv_heatmap' in self.data.figures or
                'ploidy' in self.data.figures)

    def render(self) -> str:
        if not self.is_available():
            return ""

        figures_html = ""
        for key in ['cnv_heatmap', 'ploidy', 'infercnv']:
            if key in self.data.figures:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">CNV Inference</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="CNV">
                    </div>
                    <div class="figure-caption">
                        Copy Number Variation 추론 결과. 악성세포는 이수성(aneuploidy)을 보입니다.
                    </div>
                </div>
                '''
                break

        table_html = ""
        if self.data.ploidy_by_celltype is not None and len(self.data.ploidy_by_celltype) > 0:
            df = self.data.ploidy_by_celltype

            rows = []
            for _, row in df.head(10).iterrows():
                celltype = row.get('cell_type', row.get('cluster', 'N/A'))
                ploidy = row.get('ploidy', row.get('cnv_score', 'N/A'))
                is_malignant = row.get('is_malignant', row.get('malignant', None))

                malignant_badge = ""
                if is_malignant is True:
                    malignant_badge = '<span class="badge badge-up">Malignant</span>'
                elif is_malignant is False:
                    malignant_badge = '<span class="badge badge-down">Normal</span>'

                if isinstance(ploidy, float):
                    ploidy = f"{ploidy:.2f}"

                rows.append(f'''
                <tr>
                    <td>{celltype}</td>
                    <td>{ploidy}</td>
                    <td>{malignant_badge}</td>
                </tr>
                ''')

            table_html = f'''
            <h3>Ploidy by Cell Type</h3>
            <div class="table-wrapper">
                <table>
                    <thead><tr><th>Cell Type</th><th>Ploidy Score</th><th>Classification</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
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


class InteractionSection(BaseSection):
    """Cell-Cell Interaction analysis section."""

    section_id = "interactions"
    section_icon = "🔗"
    section_title = "Cell-Cell Interaction"
    section_title_en = "Cell-Cell Interaction Analysis"

    def is_available(self) -> bool:
        return (self.data.cell_interactions is not None or
                'cellchat' in self.data.figures or
                'interaction_heatmap' in self.data.figures)

    def render(self) -> str:
        if not self.is_available():
            return ""

        figures_html = ""
        for key in ['cellchat', 'interaction_heatmap', 'ligand_receptor', 'interaction_network']:
            if key in self.data.figures:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">Cell-Cell Interaction</div>
                    <div class="figure-content">
                        <img src="{self.data.figures[key]}" alt="Interactions">
                    </div>
                </div>
                '''
                break

        table_html = ""
        if self.data.cell_interactions is not None and len(self.data.cell_interactions) > 0:
            df = self.data.cell_interactions.head(15)

            rows = []
            for _, row in df.iterrows():
                source = row.get('source', row.get('cell_type_1', 'N/A'))
                target = row.get('target', row.get('cell_type_2', 'N/A'))
                ligand = row.get('ligand', row.get('ligand_gene', 'N/A'))
                receptor = row.get('receptor', row.get('receptor_gene', 'N/A'))
                score = row.get('score', row.get('prob', 'N/A'))

                if isinstance(score, float):
                    score = f"{score:.3f}"

                rows.append(f'''
                <tr>
                    <td>{source}</td>
                    <td>{target}</td>
                    <td><span class="gene-symbol">{ligand}</span></td>
                    <td><span class="gene-symbol">{receptor}</span></td>
                    <td>{score}</td>
                </tr>
                ''')

            table_html = f'''
            <h3>Top Cell-Cell Interactions</h3>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>Source</th><th>Target</th><th>Ligand</th><th>Receptor</th><th>Score</th></tr>
                    </thead>
                    <tbody>{''.join(rows)}</tbody>
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
