"""
Common Section Components

Shared sections used by both Bulk and Single-cell reports.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from ..base_report import ReportConfig, ReportData


class BaseSection(ABC):
    """Base class for report sections."""

    section_id: str = ""
    section_number: Optional[int] = None
    section_icon: str = ""
    section_title: str = ""
    section_title_en: str = ""

    def __init__(self, config: ReportConfig, data: ReportData):
        self.config = config
        self.data = data

    def get_title(self) -> str:
        """Get localized title."""
        if self.config.language == "en":
            return self.section_title_en or self.section_title
        return self.section_title

    def section_header(self) -> str:
        """Generate section header HTML."""
        number_html = f'<span class="section-number">{self.section_number}</span>' if self.section_number else ''
        icon_html = f'<span class="section-icon">{self.section_icon}</span>' if self.section_icon else ''

        return f'''
        <div class="section-header">
            {number_html}
            {icon_html}
            <h2 class="section-title">{self.get_title()}</h2>
        </div>
        '''

    @abstractmethod
    def render(self) -> str:
        """Render the section HTML."""
        pass

    def is_available(self) -> bool:
        """Check if section has data to display."""
        return True


class CoverSection(BaseSection):
    """Cover page section."""

    section_id = "cover"

    def render(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Data type badge
        data_type_labels = {
            "bulk": ("Bulk RNA-seq", "bulk"),
            "singlecell": ("Single-cell RNA-seq", "singlecell"),
            "multiomic": ("Multi-omic Analysis", "multiomic"),
        }
        label, badge_class = data_type_labels.get(self.config.data_type, ("RNA-seq", "bulk"))

        # Get stats based on data type
        if self.config.data_type == "singlecell":
            summary = self.data.adata_summary or {}
            stats = [
                ("세포 수", f"{summary.get('n_cells', 0):,}"),
                ("유전자 수", f"{summary.get('n_genes', 0):,}"),
                ("클러스터", f"{summary.get('n_clusters', 0)}"),
                ("세포 유형", f"{summary.get('n_celltypes', 0)}"),
            ]
        else:
            deg_df = self.data.deg_significant
            n_deg = len(deg_df) if deg_df is not None else 0
            hub_df = self.data.hub_genes
            n_hub = len(hub_df) if hub_df is not None else 0

            # Get up/down counts
            n_up, n_down = 0, 0
            if deg_df is not None and len(deg_df) > 0:
                fc_col = 'log2FC' if 'log2FC' in deg_df.columns else 'log2FoldChange'
                if fc_col in deg_df.columns:
                    n_up = len(deg_df[deg_df[fc_col] > 0])
                    n_down = n_deg - n_up

            stats = [
                ("DEG 수", f"{n_deg:,}"),
                ("상향 발현", f"{n_up:,}"),
                ("하향 발현", f"{n_down:,}"),
                ("Hub Genes", f"{n_hub}"),
            ]

        stats_html = ''.join([
            f'''
            <div class="meta-item">
                <div class="meta-label">{label}</div>
                <div class="meta-value">{value}</div>
            </div>
            '''
            for label, value in stats
        ])

        cancer_type = self.config.cancer_type.upper() if self.config.cancer_type != "unknown" else "Cancer"

        return f'''
        <div class="cover-page">
            <span class="data-type-badge {badge_class}">{label}</span>
            <h1>{self.config.report_title}</h1>
            <p class="subtitle">{cancer_type} 분석 결과 보고서</p>
            <div class="meta-info">
                <div class="meta-item">
                    <div class="meta-label">생성일</div>
                    <div class="meta-value">{timestamp}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">암종</div>
                    <div class="meta-value">{cancer_type}</div>
                </div>
                {stats_html}
            </div>
        </div>
        '''


class SummarySection(BaseSection):
    """Summary statistics section."""

    section_id = "summary"
    section_number = 1
    section_icon = "📊"
    section_title = "분석 요약"
    section_title_en = "Analysis Summary"

    def render(self) -> str:
        cards = []

        if self.config.data_type == "singlecell":
            summary = self.data.adata_summary or {}
            cards = [
                ("n_cells", "Total Cells", summary.get('n_cells', 0), False),
                ("n_genes", "Total Genes", summary.get('n_genes', 0), False),
                ("n_clusters", "Clusters", summary.get('n_clusters', 0), False),
                ("n_celltypes", "Cell Types", summary.get('n_celltypes', 0), True),
                ("n_markers", "Marker Genes", len(self.data.cluster_markers) if self.data.cluster_markers is not None else 0, False),
                ("n_drivers", "Driver Genes", len(self.data.driver_genes) if self.data.driver_genes is not None else 0, False),
            ]
        else:
            deg_df = self.data.deg_significant
            hub_df = self.data.hub_genes
            pathway_df = self.data.pathway_summary

            n_deg = len(deg_df) if deg_df is not None else 0
            n_hub = len(hub_df) if hub_df is not None else 0
            n_pathway = len(pathway_df) if pathway_df is not None else 0

            # Up/Down counts
            n_up, n_down = 0, 0
            if deg_df is not None and len(deg_df) > 0:
                fc_col = 'log2FC' if 'log2FC' in deg_df.columns else 'log2FoldChange'
                if fc_col in deg_df.columns:
                    n_up = len(deg_df[deg_df[fc_col] > 0])
                    n_down = n_deg - n_up

            cards = [
                ("deg", "Total DEGs", n_deg, True),
                ("up", "Upregulated", n_up, False),
                ("down", "Downregulated", n_down, False),
                ("hub", "Hub Genes", n_hub, False),
                ("pathway", "Pathways", n_pathway, False),
            ]

            # Add driver count if available
            driver_known = self.data.driver_known or []
            if driver_known:
                cards.append(("driver", "Known Drivers", len(driver_known), False))

        cards_html = ''.join([
            f'''
            <div class="summary-card{' highlight' if highlight else ''}">
                <div class="value">{value:,}</div>
                <div class="label">{label}</div>
            </div>
            '''
            for _, label, value, highlight in cards
        ])

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            <div class="summary-grid">
                {cards_html}
            </div>
        </section>
        '''


class AbstractSection(BaseSection):
    """Extended abstract section (LLM-generated)."""

    section_id = "abstract"
    section_icon = "📝"
    section_title = "연구 요약 (Abstract)"
    section_title_en = "Extended Abstract"

    def is_available(self) -> bool:
        return self.data.abstract_extended is not None

    def render(self) -> str:
        if not self.is_available():
            return ""

        abstract = self.data.abstract_extended
        content = abstract.get('content', '') if isinstance(abstract, dict) else str(abstract)

        # Handle structured abstract
        if isinstance(abstract, dict):
            sections = []
            for key in ['background', 'methods', 'results', 'conclusions']:
                if key in abstract:
                    title = key.capitalize()
                    sections.append(f"<p><strong>{title}:</strong> {abstract[key]}</p>")
            if sections:
                content = ''.join(sections)

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            <div class="abstract-content">
                {content}
            </div>
            <div class="note-box info">
                <span class="note-box-icon">ℹ️</span>
                <div class="note-box-content">
                    본 요약은 AI 모델에 의해 자동 생성되었습니다. 핵심 발견 사항을 검토하시기 바랍니다.
                </div>
            </div>
        </section>
        '''


class QCSection(BaseSection):
    """Quality Control section."""

    section_id = "qc"
    section_number = 2
    section_icon = "🔍"
    section_title = "데이터 품질 관리 (QC)"
    section_title_en = "Quality Control"

    def render(self) -> str:
        figures_html = ""

        if self.config.data_type == "singlecell":
            # Single-cell QC: violin plot
            if 'violin_qc' in self.data.figures:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">QC Metrics Distribution</div>
                    <div class="figure-content">
                        <img src="{self.data.figures['violin_qc']}" alt="QC Violin Plot">
                    </div>
                    <div class="figure-caption">
                        세포별 유전자 수, UMI counts, 미토콘드리아 비율 분포
                    </div>
                </div>
                '''
        else:
            # Bulk QC: PCA plot - Prioritize interactive version
            pca_interactive_key = None
            pca_fig = None

            # First check for interactive versions
            for key in ['pca_interactive', 'pca_plot', 'pca']:
                if key in self.data.interactive_figures:
                    pca_interactive_key = key
                    break

            # Fallback to static
            if not pca_interactive_key:
                for key in ['pca_plot', 'pca']:
                    if key in self.data.figures:
                        pca_fig = self.data.figures[key]
                        break

            if pca_interactive_key:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">PCA Plot (Interactive - 호버로 샘플 확인)</div>
                    <div class="interactive-container" style="min-height: 500px;">
                        {self.data.interactive_figures[pca_interactive_key]}
                    </div>
                    <div class="figure-caption">
                        주성분 분석 결과. <strong>마우스 호버로 샘플 정보 확인</strong>
                    </div>
                </div>
                '''
            elif pca_fig:
                figures_html = f'''
                <div class="figure-panel">
                    <div class="figure-title">PCA Plot</div>
                    <div class="figure-content">
                        <img src="{pca_fig}" alt="PCA Plot">
                    </div>
                    <div class="figure-caption">
                        주성분 분석 결과. 샘플 간 전사체 유사성을 보여줍니다.
                    </div>
                </div>
                '''

        if not figures_html:
            figures_html = '<p class="text-muted">QC 시각화를 사용할 수 없습니다.</p>'

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            <div class="figure-grid">
                {figures_html}
            </div>
        </section>
        '''


class DriverSection(BaseSection):
    """Driver gene analysis section with RAG + External API integration."""

    section_id = "driver"
    section_number = 5
    section_icon = "🎯"
    section_title = "Driver 유전자 분석"
    section_title_en = "Driver Gene Analysis"

    def is_available(self) -> bool:
        return bool(self.data.driver_known or self.data.driver_novel or
                    self.data.driver_genes is not None or
                    self.data.rag_interpretations is not None)

    def render(self) -> str:
        if not self.is_available():
            return ""

        # Known drivers
        known_html = ""
        if self.data.driver_known:
            rows = []
            for driver in self.data.driver_known[:15]:
                gene = driver.get('gene_symbol', driver.get('gene', 'N/A'))
                source = driver.get('source', driver.get('evidence_sources', 'N/A'))
                score = driver.get('score', driver.get('confidence_score', 0))
                role = driver.get('role', driver.get('oncogenic_role', 'N/A'))

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{gene}</span></td>
                    <td>{source}</td>
                    <td>{role}</td>
                    <td>{score:.2f}</td>
                </tr>
                ''')

            known_html = f'''
            <div class="driver-panel">
                <h3>검증된 Driver Genes (COSMIC/OncoKB)</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Gene</th>
                                <th>Source</th>
                                <th>Role</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </div>
            </div>
            '''

        # Candidate drivers (novel/regulators)
        novel_html = ""
        if self.data.driver_novel:
            rows = []
            for driver in self.data.driver_novel[:10]:
                gene = driver.get('gene_symbol', driver.get('gene', 'N/A'))
                score = driver.get('score', driver.get('confidence_score', 0))
                evidence = driver.get('evidence', driver.get('reasoning', 'N/A'))

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{gene}</span></td>
                    <td>{score:.2f}</td>
                    <td>{evidence[:100]}...</td>
                </tr>
                ''')

            novel_html = f'''
            <div class="driver-panel">
                <h3>Candidate Regulators (발현 기반 예측)</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Gene</th>
                                <th>Score</th>
                                <th>Evidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </div>
                <div class="note-box warning">
                    <span class="note-box-icon">⚠️</span>
                    <div class="note-box-content">
                        Candidate regulators는 발현 기반 예측 결과입니다. 실험적 검증이 필요합니다.
                    </div>
                </div>
            </div>
            '''

        # Single-cell driver genes
        sc_driver_html = ""
        if self.data.driver_genes is not None and len(self.data.driver_genes) > 0:
            df = self.data.driver_genes.head(15)
            rows = []
            for _, row in df.iterrows():
                gene = row.get('gene', row.get('gene_symbol', 'N/A'))
                cluster = row.get('cluster', 'N/A')
                cosmic = '✅' if row.get('is_cosmic_tier1', False) else '❌'
                oncokb = '✅' if row.get('is_oncokb_actionable', False) else '❌'

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{gene}</span></td>
                    <td>{cluster}</td>
                    <td>{cosmic}</td>
                    <td>{oncokb}</td>
                </tr>
                ''')

            sc_driver_html = f'''
            <div class="driver-panel">
                <h3>클러스터별 Driver Genes</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Gene</th>
                                <th>Cluster</th>
                                <th>COSMIC Tier1</th>
                                <th>OncoKB</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </div>
            </div>
            '''

        # RAG + External API interpretation results
        rag_html = self._render_rag_interpretations()

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {known_html}
            {novel_html}
            {sc_driver_html}
            {rag_html}
        </section>
        '''

    def _render_rag_interpretations(self) -> str:
        """Render RAG + External API gene interpretations."""
        if not self.data.rag_interpretations:
            return ""

        rag_data = self.data.rag_interpretations
        interpretations = rag_data.get('interpretations', {})

        if not interpretations:
            return ""

        # Build gene interpretation cards
        cards_html = []
        for gene_symbol, interp in list(interpretations.items())[:10]:  # Top 10 genes
            log2fc = interp.get('log2fc', 0)
            direction = interp.get('direction', 'up')
            direction_icon = '🔺' if direction == 'up' else '🔻'
            direction_class = 'up' if direction == 'up' else 'down'

            interpretation_text = interp.get('interpretation', 'No interpretation available.')
            confidence = interp.get('confidence', 'medium')
            confidence_badge = {
                'high': '<span class="badge badge-success">High Confidence</span>',
                'medium': '<span class="badge badge-warning">Medium</span>',
                'low': '<span class="badge badge-danger">Low</span>',
            }.get(confidence, '<span class="badge">Unknown</span>')

            # Citations (from RAG)
            citations = interp.get('citations', [])
            pmids = interp.get('pmids', [])
            citations_html = ""
            if citations:
                cite_items = []
                for i, cite in enumerate(citations[:3], 1):
                    title = cite.get('paper_title', 'Unknown')[:60]
                    doi = cite.get('doi', '')
                    year = cite.get('year', '')
                    if title == 'Unknown' and doi:
                        title = f"DOI: {doi}"
                    cite_items.append(f'<li>[{i}] {title} ({year})</li>')
                citations_html = f'''
                <div class="citations">
                    <strong>📚 문헌 근거:</strong>
                    <ul class="citation-list">{''.join(cite_items)}</ul>
                </div>
                '''

            # External API annotations
            api_annotations = []
            cancer_role = interp.get('cancer_role', '')
            if cancer_role and cancer_role != 'Unknown':
                api_annotations.append(f'<span class="api-tag oncokb">🎯 {cancer_role}</span>')

            if interp.get('is_oncogene'):
                api_annotations.append('<span class="api-tag oncogene">Oncogene</span>')
            if interp.get('is_tsg'):
                api_annotations.append('<span class="api-tag tsg">TSG</span>')
            if interp.get('actionable'):
                api_annotations.append('<span class="api-tag actionable">💊 Actionable</span>')

            protein_func = interp.get('protein_function', '')
            if protein_func:
                api_annotations.append(f'<div class="protein-func"><strong>Protein:</strong> {protein_func[:150]}...</div>')

            pathways = interp.get('pathways', [])
            if pathways:
                pathway_names = [p.get('name', p) if isinstance(p, dict) else str(p) for p in pathways[:3]]
                api_annotations.append(f'<div class="pathways"><strong>Pathways:</strong> {", ".join(pathway_names)}</div>')

            api_html = f'<div class="api-annotations">{"".join(api_annotations)}</div>' if api_annotations else ""

            # Sources used
            sources = interp.get('sources_used', [])
            sources_html = ""
            if sources:
                sources_html = f'<div class="sources-used"><strong>Data Sources:</strong> {", ".join(sources)}</div>'

            cards_html.append(f'''
            <div class="gene-interpretation-card">
                <div class="card-header">
                    <span class="gene-symbol-large">{gene_symbol}</span>
                    <span class="direction-badge {direction_class}">{direction_icon} log2FC: {log2fc:.2f}</span>
                    {confidence_badge}
                </div>
                <div class="card-body">
                    <p class="interpretation-text">{interpretation_text}</p>
                    {api_html}
                    {citations_html}
                    {sources_html}
                </div>
            </div>
            ''')

        # Summary statistics
        n_genes = rag_data.get('genes_interpreted', len(interpretations))
        cancer_type = rag_data.get('cancer_type', 'unknown')

        return f'''
        <div class="rag-interpretations-panel">
            <h3>🔬 Multi-Source 유전자 해석 (RAG + External API)</h3>
            <div class="interpretation-summary">
                <p>총 <strong>{n_genes}개</strong> 유전자에 대해 문헌 기반 RAG와 외부 API (OncoKB, STRING, UniProt, KEGG)를 통합하여 해석을 수행했습니다.</p>
                <p>암종: <strong>{cancer_type.upper()}</strong></p>
            </div>
            <div class="gene-cards-container">
                {''.join(cards_html)}
            </div>
            <div class="note-box info">
                <span class="note-box-icon">ℹ️</span>
                <div class="note-box-content">
                    <strong>데이터 소스 설명:</strong><br>
                    • <strong>RAG (문헌 검색)</strong>: VectorDB에 인덱싱된 암 연구 논문에서 관련 문헌을 검색하여 해석<br>
                    • <strong>OncoKB/CIViC</strong>: 임상적으로 검증된 암 유전자 annotation<br>
                    • <strong>STRING</strong>: 단백질-단백질 상호작용 네트워크<br>
                    • <strong>UniProt</strong>: 단백질 기능 및 세포 내 위치<br>
                    • <strong>KEGG/Reactome</strong>: 생물학적 경로 정보
                </div>
            </div>
        </div>
        '''


class ClinicalSection(BaseSection):
    """Clinical implications section."""

    section_id = "clinical"
    section_number = 8
    section_icon = "💊"
    section_title = "임상적 시사점"
    section_title_en = "Clinical Implications"

    def render(self) -> str:
        # Extract clinical info from research recommendations if available
        rec = self.data.research_recommendations or {}

        therapeutic_html = ""
        if 'therapeutic_targets' in rec:
            targets = rec['therapeutic_targets']
            if isinstance(targets, list) and targets:
                rows = []
                for t in targets[:10]:
                    if isinstance(t, dict):
                        rows.append(f'''
                        <tr>
                            <td><span class="gene-symbol">{t.get('gene', 'N/A')}</span></td>
                            <td>{t.get('priority', 'N/A')}</td>
                            <td>{t.get('drugs', t.get('existing_drugs', 'N/A'))}</td>
                            <td>{t.get('rationale', '')[:80]}...</td>
                        </tr>
                        ''')

                if rows:
                    therapeutic_html = f'''
                    <h3>치료 표적 후보</h3>
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr><th>Gene</th><th>Priority</th><th>Drugs</th><th>Rationale</th></tr>
                            </thead>
                            <tbody>{''.join(rows)}</tbody>
                        </table>
                    </div>
                    '''

        # Biomarkers
        biomarker_html = ""
        if 'biomarker_development' in rec:
            markers = rec['biomarker_development']
            if isinstance(markers, list) and markers:
                items = []
                for m in markers[:5]:
                    if isinstance(m, dict):
                        items.append(f"<li><strong>{m.get('gene', 'N/A')}</strong>: {m.get('type', '')} - {m.get('rationale', '')}</li>")

                if items:
                    biomarker_html = f'''
                    <h3>바이오마커 후보</h3>
                    <ul>{''.join(items)}</ul>
                    '''

        if not therapeutic_html and not biomarker_html:
            content = '''
            <p>분석 결과를 바탕으로 한 임상적 시사점:</p>
            <ul>
                <li>발현 변화가 큰 유전자들에 대한 기능 분석이 필요합니다.</li>
                <li>DB 검증된 driver genes는 치료 표적 후보로 고려할 수 있습니다.</li>
                <li>추가적인 실험적 검증이 임상 적용 전 필요합니다.</li>
            </ul>
            '''
        else:
            content = f"{therapeutic_html}{biomarker_html}"

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {content}
            <div class="note-box warning">
                <span class="note-box-icon">⚠️</span>
                <div class="note-box-content">
                    <strong>면책조항:</strong> 본 분석은 연구 목적으로만 제공되며, 임상적 결정에 직접 사용해서는 안 됩니다.
                </div>
            </div>
        </section>
        '''


class FollowUpSection(BaseSection):
    """Follow-up experiments section."""

    section_id = "followup"
    section_number = 9
    section_icon = "🔬"
    section_title = "검증 실험 제안"
    section_title_en = "Suggested Follow-up Experiments"

    def render(self) -> str:
        rec = self.data.research_recommendations or {}

        validation_html = ""
        if 'experimental_validation' in rec:
            validations = rec['experimental_validation']
            if isinstance(validations, list):
                items = []
                for v in validations:
                    if isinstance(v, dict):
                        items.append(f'''
                        <div class="experiment-card">
                            <h4>{v.get('method', v.get('type', 'N/A'))}</h4>
                            <p><strong>대상:</strong> {v.get('targets', v.get('genes', 'N/A'))}</p>
                            <p><strong>목적:</strong> {v.get('purpose', v.get('rationale', 'N/A'))}</p>
                        </div>
                        ''')
                    elif isinstance(v, str):
                        items.append(f'<li>{v}</li>')

                if items:
                    if '<div' in items[0]:
                        validation_html = f'<div class="experiment-grid">{"".join(items)}</div>'
                    else:
                        validation_html = f'<ul>{"".join(items)}</ul>'

        if not validation_html:
            validation_html = '''
            <div class="interpretation-grid">
                <div class="interpretation-card">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">🧬</span>
                        <span class="interpretation-title">발현 검증</span>
                    </div>
                    <div class="interpretation-content">
                        <ul>
                            <li><strong>qRT-PCR:</strong> 주요 DEG 발현 검증</li>
                            <li><strong>Western Blot:</strong> 단백질 수준 확인</li>
                            <li><strong>IHC:</strong> 조직 내 발현 패턴 확인</li>
                        </ul>
                    </div>
                </div>
                <div class="interpretation-card">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">🔧</span>
                        <span class="interpretation-title">기능 연구</span>
                    </div>
                    <div class="interpretation-content">
                        <ul>
                            <li><strong>Knockdown:</strong> siRNA/shRNA를 이용한 기능 상실 연구</li>
                            <li><strong>Overexpression:</strong> 유전자 과발현 효과 분석</li>
                            <li><strong>CRISPR:</strong> 유전자 편집을 통한 기능 검증</li>
                        </ul>
                    </div>
                </div>
            </div>
            '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {validation_html}
        </section>
        '''


class MethodsSection(BaseSection):
    """Methods and parameters section."""

    section_id = "methods"
    section_number = 10
    section_icon = "⚙️"
    section_title = "분석 방법 및 파라미터"
    section_title_en = "Methods & Parameters"

    def render(self) -> str:
        if self.config.data_type == "singlecell":
            methods_content = '''
            <h3>Single-cell 분석 파이프라인</h3>
            <ul>
                <li><strong>QC 필터링:</strong> Scanpy (min_genes=200, min_cells=3, max_mito=20%)</li>
                <li><strong>정규화:</strong> Total count normalization + log1p</li>
                <li><strong>HVG 선택:</strong> Highly variable genes (flavor='seurat_v3')</li>
                <li><strong>차원축소:</strong> PCA (n_pcs=50) + UMAP</li>
                <li><strong>클러스터링:</strong> Leiden algorithm (resolution=0.5-1.0)</li>
                <li><strong>세포 유형 주석:</strong> CellTypist (auto-detected model)</li>
                <li><strong>마커 유전자:</strong> Wilcoxon rank-sum test (log2FC>0.5, padj<0.05)</li>
            </ul>
            '''
        else:
            methods_content = '''
            <h3>Bulk RNA-seq 분석 파이프라인</h3>
            <ul>
                <li><strong>차등발현 분석:</strong> DESeq2 with apeglm shrinkage</li>
                <li><strong>DEG 기준:</strong> |log2FC| > 1.0, adjusted p-value < 0.05</li>
                <li><strong>네트워크 분석:</strong> Spearman correlation + NetworkX</li>
                <li><strong>Hub gene 선정:</strong> Betweenness centrality + Degree</li>
                <li><strong>Pathway 분석:</strong> Enrichr (GO BP, KEGG, Reactome)</li>
                <li><strong>DB 검증:</strong> COSMIC, OncoKB, IntOGen</li>
                <li><strong>ML 예측:</strong> CatBoost + SHAP explainer</li>
            </ul>
            '''

        # Add execution time from metadata if available
        exec_time_html = ""
        if self.data.meta_agents:
            times = []
            for agent, meta in self.data.meta_agents.items():
                if 'execution_time_seconds' in meta:
                    times.append(f"{agent}: {meta['execution_time_seconds']:.1f}s")
            if times:
                exec_time_html = f'''
                <h3>실행 시간</h3>
                <p>{', '.join(times)}</p>
                '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {methods_content}
            {exec_time_html}
            <h3>소프트웨어 버전</h3>
            <ul>
                <li>BioInsight Pipeline: v3.0</li>
                <li>Python: 3.11+</li>
                <li>DESeq2: 1.40+</li>
                <li>Scanpy: 1.9+</li>
            </ul>
        </section>
        '''


class ResearchSection(BaseSection):
    """Research recommendations section."""

    section_id = "research"
    section_icon = "🔭"
    section_title = "후속 연구 추천"
    section_title_en = "Research Recommendations"

    def is_available(self) -> bool:
        return self.data.research_recommendations is not None

    def render(self) -> str:
        if not self.is_available():
            return ""

        rec = self.data.research_recommendations

        directions_html = ""
        if 'future_research_directions' in rec:
            directions = rec['future_research_directions']
            if isinstance(directions, dict):
                items = []
                for term, content in directions.items():
                    term_label = {"short_term": "단기", "mid_term": "중기", "long_term": "장기"}.get(term, term)
                    if isinstance(content, list):
                        content = ', '.join(content)
                    items.append(f"<li><strong>{term_label}:</strong> {content}</li>")
                if items:
                    directions_html = f'<ul>{"".join(items)}</ul>'
            elif isinstance(directions, list):
                items = [f"<li>{d}</li>" for d in directions]
                directions_html = f'<ul>{"".join(items)}</ul>'

        if not directions_html:
            directions_html = '<p>연구 추천 정보가 없습니다.</p>'

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {directions_html}
        </section>
        '''


class ReferencesSection(BaseSection):
    """Literature references section (RAG-based)."""

    section_id = "references"
    section_number = 11
    section_icon = "📚"
    section_title = "문헌 참조 (RAG)"
    section_title_en = "Literature References"

    def is_available(self) -> bool:
        return self.data.interpretation_report is not None

    def render(self) -> str:
        if not self.is_available():
            return ""

        interp = self.data.interpretation_report
        interpretations = interp.get('gene_interpretations', interp.get('interpretations', []))

        if not interpretations:
            return ""

        items = []
        for item in interpretations[:10]:
            if isinstance(item, dict):
                gene = item.get('gene_symbol', item.get('gene', 'N/A'))
                text = item.get('interpretation', item.get('summary', ''))
                pmids = item.get('pmids', [])

                pmid_links = ""
                if pmids:
                    links = [f'<a href="https://pubmed.ncbi.nlm.nih.gov/{p}" target="_blank">PMID:{p}</a>' for p in pmids[:3]]
                    pmid_links = f" [{', '.join(links)}]"

                items.append(f'''
                <div class="reference-item">
                    <span class="gene-symbol">{gene}</span>: {text}{pmid_links}
                </div>
                ''')

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            <div class="references-list">
                {''.join(items)}
            </div>
        </section>
        '''


class AppendixSection(BaseSection):
    """Appendix with supplementary data."""

    section_id = "appendix"
    section_number = 12
    section_icon = "📎"
    section_title = "부록: 보충 데이터"
    section_title_en = "Appendix: Supplementary Data"

    def render(self) -> str:
        tables_html = ""

        # DEG table for bulk
        if self.data.deg_significant is not None and len(self.data.deg_significant) > 0:
            df = self.data.deg_significant.head(50)
            gene_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'gene_id'
            fc_col = 'log2FC' if 'log2FC' in df.columns else 'log2FoldChange'

            rows = []
            for _, row in df.iterrows():
                gene = row.get(gene_col, 'N/A')
                fc = row.get(fc_col, 0)
                pval = row.get('padj', row.get('pvalue', 0))

                direction = '<span class="badge badge-up">↑</span>' if fc > 0 else '<span class="badge badge-down">↓</span>'

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{gene}</span></td>
                    <td>{fc:.2f}</td>
                    <td>{pval:.2e}</td>
                    <td>{direction}</td>
                </tr>
                ''')

            tables_html = f'''
            <h3>Top 50 Differentially Expressed Genes</h3>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>Gene</th><th>log2FC</th><th>Adj. P-value</th><th>Direction</th></tr>
                    </thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            '''

        # Marker genes table for single-cell
        if self.data.cluster_markers is not None and len(self.data.cluster_markers) > 0:
            df = self.data.cluster_markers.head(50)

            rows = []
            for _, row in df.iterrows():
                gene = row.get('gene', row.get('names', 'N/A'))
                cluster = row.get('cluster', row.get('group', 'N/A'))
                score = row.get('scores', row.get('logfoldchanges', 0))

                rows.append(f'''
                <tr>
                    <td><span class="gene-symbol">{gene}</span></td>
                    <td>{cluster}</td>
                    <td>{score:.2f}</td>
                </tr>
                ''')

            tables_html += f'''
            <h3>Top Cluster Marker Genes</h3>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>Gene</th><th>Cluster</th><th>Score</th></tr>
                    </thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            '''

        if not tables_html:
            tables_html = '<p>보충 데이터가 없습니다.</p>'

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {tables_html}
        </section>
        '''
