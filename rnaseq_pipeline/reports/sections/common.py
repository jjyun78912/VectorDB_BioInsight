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
    """Study overview section with summary statistics."""

    section_id = "study-overview"
    section_number = 1
    section_icon = "📊"
    section_title = "연구 개요"
    section_title_en = "Study Overview"

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

    section_id = "brief-abstract"
    section_icon = "📄"
    section_title = "연구 요약 (Extended Abstract)"
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
        <section class="brief-abstract-section" id="{self.section_id}">
            {self.section_header()}
            <p class="section-subtitle">LLM 기반 종합 분석 요약</p>
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
    section_title = "데이터 품질 관리 (Data QC)"
    section_title_en = "Data Quality Control"

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
    """Cancer-related gene validation section with RAG + External API integration.

    Note: This section validates Hub genes against cancer databases (COSMIC/OncoKB).
    Hub genes are expression-based network centrality genes, NOT true driver genes.
    True driver genes require mutation data (WGS/WES) for identification.
    """

    section_id = "driver-analysis"
    section_number = 5
    section_icon = "🎯"
    section_title = "유전자 후보 검증 (Gene Candidate Validation)"
    section_title_en = "Gene Candidate Validation"

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
                <h3>COSMIC/OncoKB 매칭 유전자</h3>
                <p class="panel-description" style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
                    DEG/Hub genes 중 암 데이터베이스에서 검증된 유전자입니다.
                    <strong>발현 변화가 있으나, 실제 driver 여부는 변이(mutation) 분석이 필요합니다.</strong>
                </p>
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


class MLPredictionSection(BaseSection):
    """ML Cancer Type Prediction section.

    Shows results from:
    - Bulk RNA-seq: Direct ML prediction
    - Single-cell RNA-seq: Pseudo-bulk aggregation → ML prediction

    Activation condition:
    - data.cancer_prediction is not None (from cancer_prediction.json or pseudobulk_prediction.json)
    """

    section_id = "ml_prediction"
    section_icon = "🤖"
    section_title = "ML 암종 예측"
    section_title_en = "ML Cancer Type Prediction"

    def is_available(self) -> bool:
        return self.data.cancer_prediction is not None

    def render(self) -> str:
        if not self.is_available():
            return ""

        pred = self.data.cancer_prediction

        # Main prediction
        predicted = pred.get('predicted_cancer', pred.get('prediction', 'Unknown'))
        predicted_kr = pred.get('predicted_cancer_korean', predicted)
        confidence = pred.get('confidence', pred.get('probability', 0))
        confidence_level = pred.get('confidence_level', 'medium')

        # Confidence styling
        conf_color = {
            'high': '#059669',
            'medium': '#d97706',
            'low': '#dc2626'
        }.get(confidence_level, '#64748b')

        # Main prediction card
        main_pred_html = f'''
        <div class="ml-prediction-main">
            <div class="prediction-result">
                <div class="prediction-label">예측된 암종</div>
                <div class="prediction-value">{predicted}</div>
                <div class="prediction-korean">{predicted_kr}</div>
            </div>
            <div class="confidence-meter">
                <div class="confidence-label">신뢰도</div>
                <div class="confidence-value" style="color: {conf_color};">{confidence:.1%}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: {confidence*100}%; background: {conf_color};"></div>
                </div>
                <div class="confidence-level">{confidence_level.upper()}</div>
            </div>
        </div>
        '''

        # Per-cluster predictions (Single-cell pseudo-bulk)
        cluster_html = ""
        all_preds = pred.get('all_predictions', [])
        if all_preds and len(all_preds) > 1:
            cluster_rows = []
            for p in all_preds:
                sample_id = p.get('sample_id', 'N/A')
                cancer = p.get('predicted_cancer', 'N/A')
                conf = p.get('confidence', 0)
                conf_lvl = p.get('confidence_level', 'low')

                badge_class = {'high': 'badge-high', 'medium': 'badge-medium', 'low': 'badge-low'}.get(conf_lvl, '')

                cluster_rows.append(f'''
                <tr>
                    <td>{sample_id}</td>
                    <td><strong>{cancer}</strong></td>
                    <td>{conf:.1%}</td>
                    <td><span class="badge {badge_class}">{conf_lvl}</span></td>
                </tr>
                ''')

            agreement = pred.get('cluster_agreement', None)
            agreement_html = ""
            if agreement is not None:
                agreement_html = f'''
                <div class="cluster-agreement">
                    클러스터 간 일치도: <strong>{agreement:.1%}</strong>
                </div>
                '''

            cluster_html = f'''
            <div class="cluster-predictions">
                <h4>클러스터별 예측 (Pseudo-bulk)</h4>
                {agreement_html}
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Sample/Cluster</th><th>예측 암종</th><th>신뢰도</th><th>Level</th></tr>
                        </thead>
                        <tbody>{''.join(cluster_rows)}</tbody>
                    </table>
                </div>
            </div>
            '''

        # Top features (SHAP)
        features_html = ""
        top_features = pred.get('top_features', pred.get('shap_features', []))
        if top_features:
            feature_items = []
            for f in top_features[:10]:
                gene = f.get('gene', f.get('feature', 'N/A'))
                importance = f.get('importance', f.get('shap_value', 0))
                direction = '↑' if importance > 0 else '↓'
                feature_items.append(f'''
                <div class="feature-item">
                    <span class="gene-symbol">{gene}</span>
                    <span class="feature-importance">{direction} {abs(importance):.3f}</span>
                </div>
                ''')

            features_html = f'''
            <div class="top-features">
                <h4>주요 예측 근거 (SHAP)</h4>
                <div class="features-grid">
                    {''.join(feature_items)}
                </div>
            </div>
            '''

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {main_pred_html}
            {cluster_html}
            {features_html}
            <div class="note-box warning">
                <span class="note-box-icon">⚠️</span>
                <div class="note-box-content">
                    <strong>면책조항:</strong> ML 예측은 참고용이며 진단 목적으로 사용할 수 없습니다.
                    예측 결과는 반드시 조직병리학적 검사로 확인해야 합니다.
                </div>
            </div>
            <style>
                .ml-prediction-main {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 2rem;
                    margin-bottom: 2rem;
                }}
                .prediction-result, .confidence-meter {{
                    background: var(--bg-tertiary);
                    padding: 1.5rem;
                    border-radius: 12px;
                    text-align: center;
                }}
                .prediction-label, .confidence-label {{
                    font-size: 0.85rem;
                    color: var(--text-muted);
                    margin-bottom: 0.5rem;
                }}
                .prediction-value {{
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--primary);
                    font-family: var(--font-mono);
                }}
                .prediction-korean {{
                    font-size: 1rem;
                    color: var(--text-secondary);
                    margin-top: 0.25rem;
                }}
                .confidence-value {{
                    font-size: 2rem;
                    font-weight: 700;
                    font-family: var(--font-mono);
                }}
                .confidence-bar-container {{
                    height: 8px;
                    background: var(--bg-secondary);
                    border-radius: 4px;
                    margin: 0.75rem 0;
                    overflow: hidden;
                }}
                .confidence-bar-fill {{
                    height: 100%;
                    border-radius: 4px;
                    transition: width 0.3s;
                }}
                .confidence-level {{
                    font-size: 0.75rem;
                    font-weight: 600;
                }}
                .cluster-predictions {{
                    margin-top: 1.5rem;
                }}
                .cluster-agreement {{
                    margin-bottom: 1rem;
                    padding: 0.75rem;
                    background: var(--bg-tertiary);
                    border-radius: 8px;
                }}
                .top-features {{
                    margin-top: 1.5rem;
                }}
                .features-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                    gap: 0.5rem;
                }}
                .feature-item {{
                    display: flex;
                    justify-content: space-between;
                    padding: 0.5rem 0.75rem;
                    background: var(--bg-tertiary);
                    border-radius: 6px;
                    font-size: 0.85rem;
                }}
                .feature-importance {{
                    font-family: var(--font-mono);
                    color: var(--text-muted);
                }}
                @media (max-width: 768px) {{
                    .ml-prediction-main {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </section>
        '''


class ClinicalSection(BaseSection):
    """Clinical implications section."""

    section_id = "clinical"
    section_number = 7
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
    """Research recommendations section including therapeutic targets, drug repurposing, and validation experiments."""

    section_id = "research-recommendations"
    section_number = 8
    section_icon = "🔬"
    section_title = "후속 연구 제안"
    section_title_en = "Research Recommendations"

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
    section_number = 9
    section_icon = "⚙️"
    section_title = "분석 방법"
    section_title_en = "Methods"

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
    """Literature-based interpretation section (RAG-based)."""

    section_id = "rag-summary"
    section_number = 10
    section_icon = "📚"
    section_title = "문헌 기반 해석"
    section_title_en = "Literature-based Interpretation"

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
    """Appendix with supplementary data and download links."""

    section_id = "detailed-table"
    section_number = 11
    section_icon = "📎"
    section_title = "부록"
    section_title_en = "Appendix"

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

        # Download section for full data
        download_html = self._render_download_section()

        return f'''
        <section class="section" id="{self.section_id}">
            {self.section_header()}
            {tables_html}
            {download_html}
        </section>
        '''

    def _render_download_section(self) -> str:
        """Render download links for full data files."""
        download_items = []

        # Check available files for download
        if self.data.deg_significant is not None and len(self.data.deg_significant) > 0:
            n_deg = len(self.data.deg_significant)
            download_items.append({
                "name": "deg_significant.csv",
                "label": f"전체 DEG 목록 ({n_deg:,}개)",
                "icon": "📊",
                "description": "모든 유의한 차등발현 유전자 (log2FC, p-value 포함)"
            })

        if self.data.deg_all is not None and len(self.data.deg_all) > 0:
            n_all = len(self.data.deg_all)
            download_items.append({
                "name": "deg_all_results.csv",
                "label": f"전체 분석 결과 ({n_all:,}개)",
                "icon": "📋",
                "description": "필터링 전 모든 유전자의 DEG 분석 결과"
            })

        if self.data.hub_genes is not None and len(self.data.hub_genes) > 0:
            n_hub = len(self.data.hub_genes)
            download_items.append({
                "name": "hub_genes.csv",
                "label": f"Hub 유전자 ({n_hub}개)",
                "icon": "🕸️",
                "description": "네트워크 중심성 기반 Hub 유전자 목록"
            })

        if self.data.pathway_summary is not None and len(self.data.pathway_summary) > 0:
            n_pathway = len(self.data.pathway_summary)
            download_items.append({
                "name": "pathway_summary.csv",
                "label": f"Pathway 분석 ({n_pathway}개)",
                "icon": "🛤️",
                "description": "GO/KEGG enrichment 분석 결과"
            })

        if self.data.integrated_gene_table is not None and len(self.data.integrated_gene_table) > 0:
            n_int = len(self.data.integrated_gene_table)
            download_items.append({
                "name": "integrated_gene_table.csv",
                "label": f"통합 유전자 테이블 ({n_int:,}개)",
                "icon": "🔗",
                "description": "DEG + Hub + DB 검증 통합 테이블"
            })

        if self.data.cluster_markers is not None and len(self.data.cluster_markers) > 0:
            n_markers = len(self.data.cluster_markers)
            download_items.append({
                "name": "cluster_markers.csv",
                "label": f"클러스터 마커 ({n_markers:,}개)",
                "icon": "🧬",
                "description": "클러스터별 마커 유전자 전체 목록"
            })

        if not download_items:
            return ""

        items_html = ""
        for item in download_items:
            items_html += f'''
            <div class="download-item">
                <span class="download-icon">{item['icon']}</span>
                <div class="download-info">
                    <span class="download-label">{item['label']}</span>
                    <span class="download-desc">{item['description']}</span>
                </div>
                <a href="./{item['name']}" download class="download-btn">다운로드</a>
            </div>
            '''

        return f'''
        <div class="download-section">
            <h3>📥 전체 데이터 다운로드</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                위 테이블은 상위 50개만 표시됩니다. 전체 데이터는 아래에서 다운로드하세요.
            </p>
            <div class="download-grid">
                {items_html}
            </div>
            <style>
                .download-section {{
                    margin-top: 2rem;
                    padding: 1.5rem;
                    background: var(--bg-tertiary);
                    border-radius: 12px;
                    border: 1px solid var(--border-light);
                }}
                .download-grid {{
                    display: flex;
                    flex-direction: column;
                    gap: 0.75rem;
                }}
                .download-item {{
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding: 1rem;
                    background: var(--bg-card);
                    border-radius: 8px;
                    border: 1px solid var(--border-light);
                    transition: box-shadow 0.2s;
                }}
                .download-item:hover {{
                    box-shadow: var(--shadow-md);
                }}
                .download-icon {{
                    font-size: 1.5rem;
                }}
                .download-info {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 0.25rem;
                }}
                .download-label {{
                    font-weight: 600;
                    color: var(--text-primary);
                }}
                .download-desc {{
                    font-size: 0.8rem;
                    color: var(--text-muted);
                }}
                .download-btn {{
                    padding: 0.5rem 1rem;
                    background: var(--primary);
                    color: white;
                    border-radius: 6px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    text-decoration: none;
                    transition: background 0.2s;
                }}
                .download-btn:hover {{
                    background: var(--primary-dark);
                    text-decoration: none;
                }}
            </style>
        </div>
        '''
