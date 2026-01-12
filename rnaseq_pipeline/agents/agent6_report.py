"""
Agent 6: Interactive HTML Report Generation v2.0

Generates a comprehensive, interactive HTML report with:
- Executive Summary (10초 파악)
- Visual Dashboard (30초 파악)
- Detailed Findings (5분 분석)
- Methods & Appendix (참조용)

Design Principles:
1. Information Hierarchy - 4-level structure
2. Visual-first approach - Gene Status Cards, confidence badges
3. Interactive elements - DataTables, search, filter
4. Clear confidence scoring - 5-point system with visual indicators
"""

import json
import base64
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ..utils.base_agent import BaseAgent

# Claude API for Extended Abstract generation
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ReportAgent(BaseAgent):
    """Agent for generating interactive HTML reports with modern design."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "report_title": "RNA-seq Analysis Report",
            "author": "BioInsight AI Pipeline",
            "include_methods": True,
            "include_downloads": True,
            "max_table_rows": 100,
            "embed_figures": True,
            "cancer_type": "unknown"
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent6_report", input_dir, output_dir, merged_config)

    def validate_inputs(self) -> bool:
        """Validate that required data files exist."""
        return True  # Allow report generation with partial data

    def _load_all_data(self) -> Dict[str, Any]:
        """Load all available data from previous agents."""
        data = {}

        search_paths = [
            self.input_dir,
            self.input_dir / "agent1_deg",
            self.input_dir / "agent2_network",
            self.input_dir / "agent3_pathway",
            self.input_dir / "agent4_validation",
            self.input_dir / "agent5_visualization"
        ]

        csv_files = [
            "deg_significant.csv",
            "deg_all_results.csv",
            "hub_genes.csv",
            "network_nodes.csv",
            "pathway_summary.csv",
            "integrated_gene_table.csv",
            "db_matched_genes.csv"
        ]

        json_files = [
            "interpretation_report.json",
            "meta_agent1.json",
            "meta_agent2.json",
            "meta_agent3.json",
            "meta_agent4.json",
            "meta_agent5.json"
        ]

        # Load CSVs
        for filename in csv_files:
            for path in search_paths:
                filepath = path / filename
                if filepath.exists():
                    try:
                        df = pd.read_csv(filepath)
                        key = filename.replace(".csv", "")
                        data[key] = df.to_dict(orient='records')
                        data[key + '_df'] = df
                        self.logger.info(f"Loaded {filename}: {len(df)} rows")
                    except Exception as e:
                        self.logger.warning(f"Error loading {filename}: {e}")
                    break

        # Load JSONs
        for filename in json_files:
            for path in search_paths:
                filepath = path / filename
                if filepath.exists():
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            key = filename.replace(".json", "")
                            data[key] = json.load(f)
                        self.logger.info(f"Loaded {filename}")
                    except Exception as e:
                        self.logger.warning(f"Error loading {filename}: {e}")
                    break

        # Load figures
        data['figures'] = {}
        figures_dir = self.input_dir / "figures"
        if not figures_dir.exists():
            figures_dir = self.input_dir / "agent5_visualization" / "figures"

        if figures_dir.exists():
            for img_path in figures_dir.glob("*.png"):
                if self.config["embed_figures"]:
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        data['figures'][img_path.stem] = f"data:image/png;base64,{img_data}"
                else:
                    data['figures'][img_path.stem] = str(img_path)
                self.logger.info(f"Loaded figure: {img_path.name}")

            # Load interactive HTML files (Plotly)
            data['interactive_figures'] = {}
            for html_path in figures_dir.glob("*.html"):
                try:
                    with open(html_path, 'r', encoding='utf-8') as f:
                        data['interactive_figures'][html_path.stem] = f.read()
                    self.logger.info(f"Loaded interactive figure: {html_path.name}")
                except Exception as e:
                    self.logger.warning(f"Error loading {html_path.name}: {e}")

        # Load extended abstract if exists (from run_dir parent)
        run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
        extended_abstract_path = run_dir / "abstract_extended.json"
        if extended_abstract_path.exists():
            try:
                with open(extended_abstract_path, 'r', encoding='utf-8') as f:
                    data['abstract_extended'] = json.load(f)
                self.logger.info("Loaded extended abstract")
            except Exception as e:
                self.logger.warning(f"Error loading extended abstract: {e}")

        return data

    def _calculate_overall_confidence(self, data: Dict) -> tuple:
        """Calculate overall analysis confidence score."""
        score = 0
        reasons = []

        # Check DEG count
        deg_count = len(data.get('deg_significant', []))
        if 100 <= deg_count <= 5000:
            score += 1
            reasons.append("적절한 DEG 수")
        elif deg_count > 0:
            reasons.append("DEG 수 확인 필요")

        # Check DB matches
        db_matched = len(data.get('db_matched_genes', []))
        if db_matched > 0:
            score += 1
            reasons.append(f"{db_matched}개 DB 매칭")

        # Check hub genes
        hub_count = len(data.get('hub_genes', []))
        if hub_count > 0:
            score += 1
            reasons.append(f"{hub_count}개 Hub 유전자")

        # Check pathway enrichment
        pathway_count = len(data.get('pathway_summary', []))
        if pathway_count >= 5:
            score += 1
            reasons.append("유의한 Pathway 발견")

        # Check high confidence genes
        interpretation = data.get('interpretation_report', {})
        high_conf = interpretation.get('summary', {}).get('high_confidence_count', 0)
        if high_conf > 0:
            score += 1
            reasons.append(f"{high_conf}개 High confidence")

        # Determine level
        if score >= 4:
            level = "high"
            emoji = "🟢"
            label = "높음"
        elif score >= 2:
            level = "medium"
            emoji = "🟡"
            label = "중간"
        else:
            level = "low"
            emoji = "🔴"
            label = "낮음"

        return level, emoji, label, score, reasons

    def _get_top_gene_info(self, data: Dict) -> Dict:
        """Get information about the top gene."""
        integrated = data.get('integrated_gene_table', [])
        if not integrated:
            return {"symbol": "N/A", "log2fc": 0, "direction": ""}

        # Sort by interpretation score or log2FC
        sorted_genes = sorted(
            integrated,
            key=lambda x: abs(x.get('log2FC', 0)),
            reverse=True
        )

        if sorted_genes:
            top = sorted_genes[0]
            return {
                "symbol": top.get('gene_symbol', top.get('gene_id', 'Unknown')),
                "log2fc": top.get('log2FC', 0),
                "direction": "↑" if top.get('log2FC', 0) > 0 else "↓"
            }
        return {"symbol": "N/A", "log2fc": 0, "direction": ""}

    def _generate_rag_summary_html(self, data: Dict) -> str:
        """Generate RAG-based Literature Summary section."""
        integrated_df = data.get('integrated_gene_table_df')

        if integrated_df is None or 'rag_interpretation' not in integrated_df.columns:
            return ""

        # Get genes with RAG interpretations
        rag_genes = integrated_df[integrated_df['rag_interpretation'].notna() &
                                   (integrated_df['rag_interpretation'] != '')]

        if len(rag_genes) == 0:
            return ""

        # Sort by interpretation score
        rag_genes = rag_genes.sort_values('interpretation_score', ascending=False)

        # Build gene interpretation cards
        gene_cards_html = ""
        for _, gene in rag_genes.head(10).iterrows():
            symbol = gene.get('gene_symbol', gene.get('gene_id', 'Unknown'))
            log2fc = gene.get('log2FC', 0)
            direction = "↑" if log2fc > 0 else "↓"
            dir_class = "up" if log2fc > 0 else "down"
            interpretation = gene.get('rag_interpretation', '')
            pmids = str(gene.get('rag_pmids', ''))
            confidence = gene.get('confidence', 'low')
            is_hub = gene.get('is_hub', False)

            # Parse PMIDs
            pmid_list = [p.strip() for p in pmids.split(',') if p.strip() and p.strip() != 'nan']
            pmid_links = ' '.join([
                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" target="_blank" class="pmid-chip">PMID:{pmid}</a>'
                for pmid in pmid_list[:3]
            ])

            # Truncate interpretation
            interp_preview = interpretation[:300] + "..." if len(str(interpretation)) > 300 else interpretation

            gene_cards_html += f'''
            <div class="rag-gene-card">
                <div class="rag-gene-header">
                    <div class="rag-gene-title">
                        <span class="rag-gene-symbol">{symbol}</span>
                        <span class="rag-gene-fc {dir_class}">{direction} {abs(log2fc):.2f}</span>
                        {'<span class="hub-indicator">HUB</span>' if is_hub else ''}
                    </div>
                    <span class="rag-confidence {confidence}">{confidence.upper()}</span>
                </div>
                <div class="rag-gene-body">
                    <p class="rag-interpretation-text">{interp_preview}</p>
                    <div class="rag-pmids">{pmid_links if pmid_links else '<span class="no-pmid">문헌 검색 결과 없음</span>'}</div>
                </div>
            </div>
            '''

        # Summary stats
        total_with_rag = len(rag_genes)
        high_conf = len(rag_genes[rag_genes['confidence'] == 'high'])
        with_pmids = len(rag_genes[rag_genes['rag_pmids'].notna() & (rag_genes['rag_pmids'] != '')])

        return f'''
        <section class="rag-summary" id="rag-summary">
            <div class="rag-summary-header">
                <div class="rag-title-section">
                    <h2>📚 Literature-Based Gene Interpretation (RAG + LLM)</h2>
                    <p class="rag-subtitle">Vector DB 검색 + Claude API 기반 문헌 해석</p>
                </div>
                <div class="rag-stats">
                    <div class="rag-stat">
                        <span class="rag-stat-value">{total_with_rag}</span>
                        <span class="rag-stat-label">Genes Analyzed</span>
                    </div>
                    <div class="rag-stat">
                        <span class="rag-stat-value">{with_pmids}</span>
                        <span class="rag-stat-label">With Citations</span>
                    </div>
                    <div class="rag-stat">
                        <span class="rag-stat-value">{high_conf}</span>
                        <span class="rag-stat-label">High Confidence</span>
                    </div>
                </div>
            </div>

            <div class="rag-method-note">
                <span class="method-icon">🔬</span>
                <div class="method-text">
                    <strong>분석 방법:</strong> PubMedBERT 임베딩 기반 Vector Search로 관련 논문을 검색하고,
                    Claude API를 통해 유전자별 문헌 기반 해석을 생성했습니다.
                    각 해석에는 근거 논문의 PMID가 첨부됩니다.
                </div>
            </div>

            <div class="rag-genes-grid">
                {gene_cards_html}
            </div>

            <div class="rag-disclaimer">
                <span class="disclaimer-icon">⚠️</span>
                AI 생성 해석입니다. 모든 내용은 원문 논문을 통해 검증이 필요합니다.
            </div>
        </section>
        '''

    def _generate_executive_summary_html(self, data: Dict) -> str:
        """Generate Level 1: Executive Summary (10초 파악)."""
        deg_count = len(data.get('deg_significant', []))
        top_gene = self._get_top_gene_info(data)
        conf_level, conf_emoji, conf_label, conf_score, conf_reasons = self._calculate_overall_confidence(data)

        # Get interpretation summary
        interpretation = data.get('interpretation_report', {})
        v2_interp = interpretation.get('v2_interpretation', {})
        summary_text = v2_interp.get('interpretation', '분석 결과를 확인하세요.')[:200]

        # Count up/down
        integrated_df = data.get('integrated_gene_table_df')
        up_count = down_count = 0
        if integrated_df is not None and 'direction' in integrated_df.columns:
            up_count = (integrated_df['direction'] == 'up').sum()
            down_count = (integrated_df['direction'] == 'down').sum()

        return f'''
        <section class="executive-summary" id="executive-summary">
            <div class="summary-header">
                <div class="summary-title">
                    <h2>Executive Summary</h2>
                    <span class="confidence-badge {conf_level}">{conf_emoji} {conf_label} Confidence</span>
                </div>
            </div>

            <div class="key-metrics">
                <div class="metric-card primary">
                    <div class="metric-value">{deg_count:,}</div>
                    <div class="metric-label">DEGs</div>
                    <div class="metric-detail">↑{up_count:,} / ↓{down_count:,}</div>
                </div>
                <div class="metric-card highlight">
                    <div class="metric-value">{top_gene['symbol']}</div>
                    <div class="metric-label">Top Gene</div>
                    <div class="metric-detail">{top_gene['direction']} {abs(top_gene['log2fc']):.1f}x</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{conf_score}/5</div>
                    <div class="metric-label">신뢰도 점수</div>
                    <div class="metric-detail">{', '.join(conf_reasons[:2])}</div>
                </div>
            </div>

            <div class="one-line-summary">
                <h4>한 줄 요약</h4>
                <p>{summary_text}...</p>
            </div>

            <div class="warning-box">
                <span class="warning-icon">⚠️</span>
                <span>이 분석 결과는 연구 참고용이며, 임상 진단 목적으로 사용할 수 없습니다. 모든 결론은 실험적 검증이 필요합니다.</span>
            </div>
        </section>
        '''

    def _build_network_toggle(self, network_interactive: str) -> str:
        """Build network view toggle buttons (Python 3.11 compatible)."""
        if network_interactive:
            return (
                '<div class="view-toggle">'
                '<button class="toggle-btn active" onclick="showNetworkView(\'interactive\')">3D Interactive</button>'
                '<button class="toggle-btn" onclick="showNetworkView(\'static\')">Static</button>'
                '</div>'
            )
        return ''

    def _build_network_content(self, network_interactive: str, network_src: str) -> str:
        """Build network content HTML (Python 3.11 compatible)."""
        if network_interactive:
            escaped_html = network_interactive.replace('"', '&quot;')
            return (
                '<div id="network-interactive" class="network-view active">'
                f'<iframe id="network-iframe" srcdoc="{escaped_html}" style="width:100%; height:500px; border:none; border-radius:8px;"></iframe>'
                '<p class="panel-note">💡 마우스 드래그로 회전, 스크롤로 확대/축소, 유전자 클릭으로 포커스</p>'
                '</div>'
                '<div id="network-static" class="network-view" style="display:none;">'
                f'<img src="{network_src}" alt="Network" />'
                '</div>'
            )
        elif network_src:
            return f'<img src="{network_src}" alt="Network" />'
        else:
            return '<p class="no-data">No plot available</p>'

    def _build_network_ai_interpretation(self, interp: Dict) -> str:
        """Build AI interpretation section for network visualization."""
        if not interp:
            return ''

        return f'''
        <div class="ai-interpretation">
            <div class="ai-header">🤖 AI 해석</div>
            <p class="ai-summary">{interp.get('summary', '')}</p>
            <p><strong>Hub 유전자 분석:</strong> {interp.get('hub_gene_analysis', '')}</p>
            <p><strong>네트워크 구조:</strong> {interp.get('network_topology', '')}</p>
            <p><strong>생물학적 의미:</strong> {interp.get('biological_implications', '')}</p>
        </div>
        '''

    def _build_heatmap_ai_interpretation(self, interp: Dict) -> str:
        """Build AI interpretation section for heatmap visualization."""
        if not interp:
            return ''

        observations = interp.get('key_observations', [])
        observations_html = "".join([f"<li>{obs}</li>" for obs in observations[:3]])

        return f'''
        <div class="ai-interpretation">
            <div class="ai-header">🤖 AI 해석</div>
            <p class="ai-summary">{interp.get('summary', '')}</p>
            <ul class="ai-observations">{observations_html}</ul>
            <p><strong>발현 패턴:</strong> {interp.get('pattern_analysis', '')}</p>
        </div>
        '''

    def _generate_visual_dashboard_html(self, data: Dict) -> str:
        """Generate Level 2: Visual Dashboard (30초 파악)."""
        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})
        viz_interpretations = data.get('visualization_interpretations', {})

        # Get key figures
        volcano_src = figures.get('volcano_plot', '')
        pathway_src = figures.get('pathway_barplot', '')
        network_src = figures.get('network_graph', '')
        heatmap_src = figures.get('heatmap_top50', '')
        volcano_interactive = interactive_figures.get('volcano_interactive', '')
        network_interactive = interactive_figures.get('network_3d_interactive', '')

        # Generate SHAP-like top genes bar
        integrated = data.get('integrated_gene_table', [])
        top_genes_html = ""
        if integrated:
            sorted_genes = sorted(
                integrated[:50],  # Top 50
                key=lambda x: abs(x.get('log2FC', 0)),
                reverse=True
            )[:10]

            max_fc = max(abs(g.get('log2FC', 1)) for g in sorted_genes) if sorted_genes else 1

            for gene in sorted_genes:
                symbol = gene.get('gene_symbol', gene.get('gene_id', 'Unknown'))
                log2fc = gene.get('log2FC', 0)
                width = int(abs(log2fc) / max_fc * 100)
                direction = "up" if log2fc > 0 else "down"
                arrow = "↑" if log2fc > 0 else "↓"

                top_genes_html += f'''
                <div class="gene-bar-item">
                    <span class="gene-name">{symbol}</span>
                    <div class="gene-bar-container">
                        <div class="gene-bar {direction}" style="width: {width}%"></div>
                    </div>
                    <span class="gene-value">{arrow}{abs(log2fc):.2f}</span>
                </div>
                '''

        # Pathway summary
        pathways = data.get('pathway_summary', [])[:8]
        pathway_dots_html = ""
        for pw in pathways:
            term = pw.get('term_name', 'Unknown')[:35]
            padj = pw.get('padj', 1)
            gene_count = pw.get('gene_count', 0)
            dots = min(5, max(1, int(-math.log10(padj if padj > 0 else 1e-10) / 2)))
            dots_html = "●" * dots + "○" * (5 - dots)
            pathway_dots_html += f'''
            <div class="pathway-item">
                <span class="pathway-name">{term}</span>
                <span class="pathway-dots">{dots_html}</span>
                <span class="pathway-genes">{gene_count}</span>
            </div>
            '''

        # Volcano plot section with toggle for static/interactive
        volcano_interp = viz_interpretations.get('volcano_plot', {})
        volcano_llm_section = ""
        if volcano_interp:
            observations = volcano_interp.get('key_observations', [])
            observations_html = "".join([f"<li>{obs}</li>" for obs in observations[:3]])
            volcano_llm_section = f'''
            <div class="ai-interpretation">
                <div class="ai-header">🤖 AI 해석</div>
                <p class="ai-summary">{volcano_interp.get('summary', '')}</p>
                <ul class="ai-observations">{observations_html}</ul>
                <p class="ai-significance"><strong>생물학적 의미:</strong> {volcano_interp.get('biological_significance', '')}</p>
            </div>
            '''
        volcano_desc = f'''<p class="panel-desc"><strong>X축:</strong> log2 Fold Change (발현 변화량) | <strong>Y축:</strong> -log10(padj) (통계적 유의성)<br>
        <span style="color:#dc2626;">●빨간점</span> = 상향조절 (암에서 증가) | <span style="color:#2563eb;">●파란점</span> = 하향조절 (암에서 감소) | 점선 = 유의성 기준선</p>
        {volcano_llm_section}'''

        if volcano_interactive:
            volcano_section = f'''
                <div class="dashboard-panel main-plot volcano-container">
                    <div class="volcano-header">
                        <h4>Volcano Plot - 차등발현 유전자 분포</h4>
                        <div class="view-toggle">
                            <button class="toggle-btn active" onclick="showVolcanoView('interactive')">Interactive</button>
                            <button class="toggle-btn" onclick="showVolcanoView('static')">Static</button>
                        </div>
                    </div>
                    {volcano_desc}
                    <div id="volcano-interactive" class="volcano-view active">
                        <iframe id="volcano-iframe" srcdoc="{volcano_interactive.replace('"', '&quot;')}" style="width:100%; height:450px; border:none; border-radius:8px;"></iframe>
                        <p class="panel-note">💡 마우스 드래그로 확대, 유전자 위에 마우스를 올리면 상세 정보 표시</p>
                    </div>
                    <div id="volcano-static" class="volcano-view" style="display:none;">
                        {f'<img src="{volcano_src}" alt="Volcano Plot" />' if volcano_src else '<p class="no-data">No plot available</p>'}
                    </div>
                </div>
            '''
        else:
            volcano_section = f'''
                <div class="dashboard-panel main-plot">
                    <h4>Volcano Plot - 차등발현 유전자 분포</h4>
                    {volcano_desc}
                    {f'<img src="{volcano_src}" alt="Volcano Plot" />' if volcano_src else '<p class="no-data">No plot available</p>'}
                </div>
            '''

        return f'''
        <section class="visual-dashboard" id="visual-dashboard">
            <h2>Visual Dashboard</h2>
            <p class="section-intro">RNA-seq 분석 결과의 핵심 시각화입니다. 각 그래프가 의미하는 바를 확인하세요.</p>

            <div class="dashboard-grid">
                {volcano_section}

                <div class="dashboard-panel">
                    <h4>Top 10 DEGs (|log2FC|)</h4>
                    <p class="panel-desc">발현 변화량이 가장 큰 상위 10개 유전자입니다. 빨간색은 상향조절(암에서 증가), 파란색은 하향조절(암에서 감소)을 의미합니다.</p>
                    <div class="gene-bars">
                        {top_genes_html if top_genes_html else '<p class="no-data">No data</p>'}
                    </div>
                    <p class="panel-note">⚠️ 발현 변화량 기준 정렬 (생물학적 중요도와 다를 수 있음)</p>
                </div>

                <div class="dashboard-panel">
                    <h4>Pathway Enrichment</h4>
                    <p class="panel-desc">DEG들이 어떤 생물학적 경로에 집중되어 있는지 보여줍니다. 점이 많을수록 통계적으로 유의미합니다.</p>
                    <div class="pathway-list">
                        {pathway_dots_html if pathway_dots_html else '<p class="no-data">No pathways</p>'}
                    </div>
                    <p class="panel-note">●●●●● = 매우 유의미 (padj < 0.00001), 숫자 = 해당 경로의 유전자 수</p>
                </div>

                <div class="dashboard-panel network-container">
                    <div class="network-header">
                        <h4>Network Hub Genes</h4>
                        {self._build_network_toggle(network_interactive)}
                    </div>
                    <p class="panel-desc">유전자 간 공발현(co-expression) 네트워크에서 중심적 역할을 하는 Hub 유전자입니다. Hub는 많은 유전자와 연결되어 있어 핵심 조절자일 가능성이 높습니다.</p>
                    {self._build_network_content(network_interactive, network_src)}
                    {self._build_network_ai_interpretation(viz_interpretations.get('network_graph', {}))}
                </div>

                <div class="dashboard-panel full-width">
                    <h4>Expression Heatmap (Top 50 DEGs)</h4>
                    <p class="panel-desc">상위 50개 DEG의 샘플별 발현 패턴입니다. 빨간색은 높은 발현, 파란색은 낮은 발현을 의미합니다. 샘플들이 조건(Tumor vs Normal)에 따라 구분되는지 확인하세요.</p>
                    {f'<img src="{heatmap_src}" alt="Heatmap" />' if heatmap_src else '<p class="no-data">No heatmap available</p>'}
                    {self._build_heatmap_ai_interpretation(viz_interpretations.get('heatmap', {}))}
                </div>
            </div>
        </section>
        '''

    def _generate_gene_status_cards_html(self, data: Dict) -> str:
        """Generate improved Gene Status Cards."""
        db_matched = data.get('db_matched_genes', [])
        interpretation = data.get('interpretation_report', {})
        matched_genes = interpretation.get('matched_genes', [])

        cards_html = ""

        # Combine DB matched info with interpretation
        gene_details = {}
        for gene in db_matched:
            gene_id = gene.get('gene_id', '')
            gene_details[gene_id] = gene

        # Get RAG interpretations
        rag_interps = interpretation.get('rag_interpretation', {}).get('interpretations', {})

        for idx, gene_info in enumerate(matched_genes[:10]):
            gene_id = gene_info.get('gene', '')
            details = gene_details.get(gene_id, {})

            symbol = details.get('gene_symbol', gene_id)
            log2fc = details.get('log2FC', 0)
            padj = details.get('padj', 1)
            direction = "up" if log2fc > 0 else "down"
            direction_text = "상향조절" if log2fc > 0 else "하향조절"
            fold_change = 2 ** abs(log2fc)
            db_sources = details.get('db_sources', '')
            cancer_match = details.get('cancer_type_match', False)

            checklist = gene_info.get('checklist', {})
            confidence = checklist.get('confidence', 'low')
            tags = checklist.get('tags', [])
            score = checklist.get('interpretation_score', 0)

            # Get RAG interpretation for this gene
            rag_info = rag_interps.get(symbol, {})
            rag_text = rag_info.get('interpretation', '')
            rag_citations = rag_info.get('citations', [])
            rag_pmids = rag_info.get('pmids', [])
            has_rag = bool(rag_text)

            # Calculate confidence dots (1-5)
            conf_dots = min(5, max(1, int(score)))
            conf_dots_html = "🟢" * conf_dots + "⚪" * (5 - conf_dots)

            # Expression bar (relative to max)
            expr_width = min(100, int(abs(log2fc) / 5 * 100))

            # RAG section HTML
            rag_section = ""
            if has_rag:
                rag_preview = rag_text[:200] + "..." if len(rag_text) > 200 else rag_text
                pmid_links = ' '.join([f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" target="_blank" class="pmid-link">PMID:{pmid}</a>' for pmid in rag_pmids[:3]])
                rag_section = f'''
                    <div class="rag-interpretation">
                        <span class="rag-label">📚 Literature Insight</span>
                        <p class="rag-text">{rag_preview}</p>
                        <div class="rag-citations">{pmid_links if pmid_links else f'<span class="citation-count">{len(rag_citations)} citations</span>'}</div>
                    </div>
                '''

            cards_html += f'''
            <div class="gene-status-card {'has-rag' if has_rag else ''}">
                <div class="card-header">
                    <div class="gene-info">
                        <span class="gene-symbol">{symbol}</span>
                        <span class="gene-rank">Rank #{idx + 1}</span>
                    </div>
                    <span class="confidence-badge {confidence}">{confidence.upper()}</span>
                </div>

                <div class="card-body">
                    <div class="stat-row">
                        <span class="stat-label">Expression</span>
                        <div class="stat-bar-container">
                            <div class="stat-bar {direction}" style="width: {expr_width}%"></div>
                        </div>
                        <span class="stat-value">{direction_text} {fold_change:.1f}x (p={padj:.1e})</span>
                    </div>

                    <div class="stat-row">
                        <span class="stat-label">DB Source</span>
                        <span class="stat-value db-tags">
                            {' '.join([f'<span class="db-tag">{db}</span>' for db in db_sources.split(';') if db])}
                            {'<span class="cancer-match">✓ Cancer Match</span>' if cancer_match else ''}
                        </span>
                    </div>

                    <div class="stat-row">
                        <span class="stat-label">Confidence</span>
                        <span class="stat-value confidence-dots">{conf_dots_html} {conf_dots}/5</span>
                    </div>

                    {rag_section}
                </div>

                <div class="card-footer">
                    <div class="tags">
                        {' '.join([f'<span class="tag">{tag}</span>' for tag in tags[:3]])}
                        {'<span class="tag rag-tag">📚 RAG</span>' if has_rag else ''}
                    </div>
                </div>
            </div>
            '''

        return f'''
        <section class="gene-status-cards" id="gene-cards">
            <h2>Gene Status Cards (DB-Matched)</h2>
            <div class="cards-grid">
                {cards_html if cards_html else '<p class="no-data">No DB-matched genes found</p>'}
            </div>
        </section>
        '''

    def _generate_detailed_table_html(self, data: Dict) -> str:
        """Generate Level 3: Detailed Findings with DataTables."""
        integrated = data.get('integrated_gene_table', [])[:self.config['max_table_rows']]

        rows_html = ""
        for gene in integrated:
            gene_id = gene.get('gene_id', '')
            symbol = gene.get('gene_symbol', gene_id)
            log2fc = gene.get('log2FC', 0)
            padj = gene.get('padj', 1)
            direction = gene.get('direction', '')
            is_hub = "Yes" if gene.get('is_hub', False) else "No"
            db_matched = "Yes" if gene.get('db_matched', False) else "No"
            confidence = gene.get('confidence', 'requires_validation')
            score = gene.get('interpretation_score', 0)

            rows_html += f'''
            <tr>
                <td>{symbol}</td>
                <td class="{direction}">{log2fc:.3f}</td>
                <td>{padj:.2e}</td>
                <td>{is_hub}</td>
                <td>{db_matched}</td>
                <td><span class="badge {confidence}">{confidence}</span></td>
                <td>{score:.1f}</td>
            </tr>
            '''

        return f'''
        <section class="detailed-findings" id="detailed-table">
            <h2>Detailed Gene Analysis</h2>

            <div class="table-controls">
                <input type="text" id="gene-search" class="search-input"
                       placeholder="🔍 유전자 검색..." onkeyup="filterTable()">
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="filterByConfidence('all')">All</button>
                    <button class="filter-btn" onclick="filterByConfidence('high')">High</button>
                    <button class="filter-btn" onclick="filterByConfidence('medium')">Medium</button>
                    <button class="filter-btn" onclick="filterByConfidence('novel_candidate')">Novel</button>
                </div>
            </div>

            <div class="table-container">
                <table id="gene-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Gene ↕</th>
                            <th onclick="sortTable(1)">Log2FC ↕</th>
                            <th onclick="sortTable(2)">P-adj ↕</th>
                            <th onclick="sortTable(3)">Hub ↕</th>
                            <th onclick="sortTable(4)">DB Match ↕</th>
                            <th onclick="sortTable(5)">Confidence</th>
                            <th onclick="sortTable(6)">Score ↕</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>

            <div class="table-footer">
                <span>총 {len(integrated):,}개 유전자 표시 (상위 {self.config['max_table_rows']}개)</span>
                <button class="download-btn" onclick="downloadCSV()">📥 CSV 다운로드</button>
            </div>
        </section>
        '''

    def _generate_methods_html(self) -> str:
        """Generate Level 4: Methods & Appendix."""
        return '''
        <section class="methods-section" id="methods">
            <h2>Methods & Parameters</h2>

            <div class="methods-grid">
                <div class="method-card">
                    <h4>🧬 DEG Analysis</h4>
                    <ul>
                        <li>Tool: DESeq2</li>
                        <li>Cutoff: |log2FC| > 1, padj < 0.05</li>
                        <li>Normalization: Median of ratios</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>🕸️ Network Analysis</h4>
                    <ul>
                        <li>Tool: NetworkX</li>
                        <li>Correlation: Spearman > 0.7</li>
                        <li>Hub: Top 20 by centrality</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>📊 Pathway Enrichment</h4>
                    <ul>
                        <li>Tool: gseapy (Enrichr)</li>
                        <li>DB: GO (BP/MF/CC), KEGG</li>
                        <li>Cutoff: padj < 0.05</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>✅ DB Validation</h4>
                    <ul>
                        <li>COSMIC Tier 1 genes</li>
                        <li>OncoKB annotated</li>
                        <li>Cancer-type specific</li>
                    </ul>
                </div>
            </div>

            <div class="confidence-explanation">
                <h4>신뢰도 점수 계산</h4>
                <table class="score-table">
                    <tr><td>DEG 통계 유의성 (padj < 0.05)</td><td>+1점</td></tr>
                    <tr><td>TCGA 패턴 일치</td><td>+1점</td></tr>
                    <tr><td>문헌 검증 (DB match)</td><td>+1점</td></tr>
                    <tr><td>Hub gene status</td><td>+1점</td></tr>
                    <tr><td>Cancer-type specific</td><td>+1점</td></tr>
                </table>

                <div class="confidence-legend">
                    <span>🟢🟢🟢🟢🟢 5/5 매우 높음</span>
                    <span>🟢🟢🟢🟢⚪ 4/5 높음</span>
                    <span>🟢🟢🟢⚪⚪ 3/5 중간</span>
                    <span>🟢🟢⚪⚪⚪ 2/5 낮음</span>
                    <span>🟢⚪⚪⚪⚪ 1/5 검증 필요</span>
                </div>
            </div>
        </section>
        '''

    def _generate_css(self) -> str:
        """Generate npj Systems Biology and Applications journal-style CSS."""
        return '''
        <style>
            /* ========== npj SYSTEMS BIOLOGY STYLE ========== */
            /* Based on Nature/Springer journal design guidelines */

            :root {
                /* npj Brand Colors */
                --npj-blue: #0056b9;
                --npj-blue-dark: #003d82;
                --npj-blue-light: #e8f4fc;
                --npj-orange: #e87722;

                /* Neutral Palette */
                --gray-50: #fafafa;
                --gray-100: #f5f5f5;
                --gray-200: #eeeeee;
                --gray-300: #e0e0e0;
                --gray-400: #bdbdbd;
                --gray-500: #9e9e9e;
                --gray-600: #757575;
                --gray-700: #616161;
                --gray-800: #424242;
                --gray-900: #212121;

                /* Semantic Colors */
                --success: #2e7d32;
                --warning: #f57c00;
                --danger: #c62828;
                --info: #1565c0;

                /* Typography */
                --font-sans: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                --font-serif: Georgia, 'Times New Roman', Times, serif;
                --font-mono: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;

                /* Spacing */
                --spacing-xs: 4px;
                --spacing-sm: 8px;
                --spacing-md: 16px;
                --spacing-lg: 24px;
                --spacing-xl: 32px;
                --spacing-2xl: 48px;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: var(--font-sans);
                font-size: 15px;
                line-height: 1.6;
                color: var(--gray-900);
                background: #ffffff;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }

            /* ========== HEADER / COVER PAGE ========== */
            .cover-page {
                background: linear-gradient(135deg, var(--npj-blue) 0%, var(--npj-blue-dark) 100%);
                color: white;
                padding: 80px 40px;
                text-align: center;
                position: relative;
            }

            .cover-page::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--npj-orange);
            }

            .cover-content {
                max-width: 900px;
                margin: 0 auto;
            }

            .cover-badge {
                display: inline-block;
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.3);
                padding: 6px 16px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-bottom: 24px;
            }

            .cover-title {
                font-family: var(--font-sans);
                font-size: 32px;
                font-weight: 700;
                line-height: 1.3;
                margin-bottom: 16px;
                letter-spacing: -0.5px;
            }

            .cover-subtitle {
                font-size: 16px;
                font-weight: 400;
                opacity: 0.9;
                margin-bottom: 40px;
            }

            .cover-stats {
                display: flex;
                justify-content: center;
                gap: 48px;
                margin-bottom: 40px;
            }

            .cover-stat {
                text-align: center;
            }

            .cover-stat .stat-number {
                display: block;
                font-size: 36px;
                font-weight: 700;
                letter-spacing: -1px;
            }

            .cover-stat .stat-label {
                display: block;
                font-size: 12px;
                font-weight: 500;
                opacity: 0.8;
                margin-top: 4px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .cover-meta {
                background: rgba(0,0,0,0.2);
                border-radius: 4px;
                padding: 20px 32px;
                text-align: left;
                display: inline-block;
            }

            .cover-meta p {
                margin: 4px 0;
                font-size: 13px;
            }

            .cover-footer {
                position: absolute;
                bottom: 20px;
                left: 0;
                right: 0;
                font-size: 11px;
                opacity: 0.7;
            }

            /* ========== NAVIGATION BAR ========== */
            .nav-bar {
                background: white;
                border-bottom: 1px solid var(--gray-200);
                position: sticky;
                top: 0;
                z-index: 100;
            }

            .nav-container {
                max-width: 1100px;
                margin: 0 auto;
                padding: 0 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                height: 56px;
            }

            .nav-brand {
                font-weight: 700;
                font-size: 14px;
                color: var(--npj-blue);
                letter-spacing: -0.3px;
            }

            .nav-links {
                display: flex;
                gap: 24px;
            }

            .nav-links a {
                color: var(--gray-700);
                text-decoration: none;
                font-size: 13px;
                font-weight: 500;
                padding: 4px 0;
                border-bottom: 2px solid transparent;
                transition: all 0.2s;
            }

            .nav-links a:hover {
                color: var(--npj-blue);
                border-bottom-color: var(--npj-blue);
            }

            /* ========== MAIN CONTENT ========== */
            .paper-content {
                max-width: 900px;
                margin: 0 auto;
                padding: 48px 24px;
            }

            .paper-content section {
                margin-bottom: 48px;
            }

            .paper-content h2 {
                font-family: var(--font-sans);
                font-size: 20px;
                font-weight: 700;
                color: var(--gray-900);
                margin-bottom: 20px;
                padding-bottom: 12px;
                border-bottom: 2px solid var(--npj-blue);
                letter-spacing: -0.3px;
            }

            /* ========== ABSTRACT SECTION ========== */
            .abstract-section { margin-top: 0; }

            .abstract-box {
                background: var(--gray-50);
                border-left: 4px solid var(--npj-blue);
                padding: 24px 28px;
            }

            .abstract-content p {
                margin-bottom: 12px;
                font-size: 14px;
                line-height: 1.7;
                text-align: justify;
                color: var(--gray-800);
            }

            .abstract-content p strong {
                color: var(--gray-900);
                font-weight: 600;
            }

            .abstract-keywords {
                margin-top: 16px;
                padding-top: 12px;
                border-top: 1px solid var(--gray-200);
                font-size: 13px;
                color: var(--gray-700);
            }

            /* Extended Abstract Styles */
            .abstract-box.extended {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-left: 4px solid var(--npj-blue);
                padding: 28px 32px;
            }

            .key-findings {
                margin-top: 24px;
                padding: 20px;
                background: white;
                border-radius: 8px;
                border: 1px solid var(--gray-200);
            }

            .key-findings h4 {
                color: var(--npj-blue);
                font-size: 15px;
                margin-bottom: 12px;
            }

            .key-findings ul {
                margin: 0;
                padding-left: 24px;
            }

            .key-findings li {
                font-size: 13px;
                color: var(--gray-700);
                margin-bottom: 8px;
                line-height: 1.5;
            }

            .validation-priorities {
                margin-top: 20px;
                padding: 20px;
                background: #f0fdf4;
                border-radius: 8px;
                border: 1px solid #86efac;
            }

            .validation-priorities h4 {
                color: #166534;
                font-size: 15px;
                margin-bottom: 12px;
            }

            .validation-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }

            .validation-item {
                font-size: 13px;
                color: var(--gray-700);
                background: white;
                padding: 10px 14px;
                border-radius: 6px;
            }

            .validation-item strong {
                color: #166534;
            }

            .ml-interpretation {
                margin-top: 20px;
                padding: 20px;
                background: #fef3c7;
                border-radius: 8px;
                border: 1px solid #fcd34d;
            }

            .ml-interpretation h4 {
                color: #92400e;
                font-size: 15px;
                margin-bottom: 12px;
            }

            .ml-interpretation p {
                font-size: 13px;
                color: var(--gray-700);
                line-height: 1.6;
                margin: 0;
            }

            /* ========== FIGURE PANELS (npj Style) ========== */
            .visual-dashboard {
                background: white;
                margin-bottom: 32px;
            }

            .visual-dashboard h2 {
                font-size: 20px;
                margin-bottom: 24px;
            }

            .section-intro {
                font-size: 14px;
                color: var(--gray-600);
                margin-bottom: 24px;
                text-align: left;
            }

            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }

            .dashboard-panel {
                background: white;
                border: 1px solid var(--gray-200);
                padding: 20px;
            }

            .dashboard-panel.main-plot {
                grid-row: span 2;
            }

            .dashboard-panel.full-width {
                grid-column: 1 / -1;
            }

            .dashboard-panel h4 {
                font-size: 13px;
                font-weight: 700;
                color: var(--gray-900);
                margin-bottom: 8px;
                text-transform: none;
                letter-spacing: 0;
            }

            .dashboard-panel h4::before {
                content: attr(data-label);
                display: inline-block;
                font-weight: 700;
                margin-right: 8px;
                color: var(--gray-900);
            }

            .panel-desc {
                font-size: 12px;
                color: var(--gray-600);
                line-height: 1.5;
                margin-bottom: 16px;
                padding: 10px 12px;
                background: var(--npj-blue-light);
                border-left: 3px solid var(--npj-blue);
            }

            .panel-note {
                font-size: 11px;
                color: var(--gray-600);
                margin-top: 12px;
                padding: 8px 10px;
                background: #fff8e1;
                border-left: 3px solid var(--npj-orange);
            }

            /* ========== AI INTERPRETATION STYLES ========== */
            .ai-interpretation {
                margin-top: 16px;
                padding: 14px 16px;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-radius: 8px;
                border-left: 4px solid #0ea5e9;
            }

            .ai-header {
                font-weight: 700;
                color: #0369a1;
                margin-bottom: 8px;
                font-size: 13px;
            }

            .ai-summary {
                font-size: 13px;
                color: var(--gray-700);
                margin-bottom: 10px;
                line-height: 1.6;
            }

            .ai-observations {
                margin: 10px 0;
                padding-left: 20px;
                font-size: 12px;
                color: var(--gray-700);
            }

            .ai-observations li {
                margin-bottom: 4px;
                line-height: 1.5;
            }

            .ai-significance {
                font-size: 12px;
                color: var(--gray-700);
                margin-top: 8px;
            }

            .dashboard-panel img {
                width: 100%;
                display: block;
            }

            /* ========== FIGURE TOGGLE BUTTONS ========== */
            .volcano-container, .network-container { position: relative; }

            .volcano-header, .network-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }

            .view-toggle {
                display: flex;
                gap: 2px;
                background: var(--gray-100);
                padding: 2px;
                border-radius: 3px;
            }

            .toggle-btn {
                padding: 4px 10px;
                border: none;
                border-radius: 2px;
                background: transparent;
                color: var(--gray-600);
                font-size: 11px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.15s;
            }

            .toggle-btn.active {
                background: white;
                color: var(--npj-blue);
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }

            .volcano-view, .network-view { display: none; }
            .volcano-view.active, .network-view.active { display: block; }

            /* ========== GENE BARS (Bar Chart Style) ========== */
            .gene-bars {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .gene-bar-item {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .gene-name {
                width: 70px;
                font-size: 11px;
                font-weight: 600;
                color: var(--gray-800);
                font-family: var(--font-mono);
            }

            .gene-bar-container {
                flex: 1;
                height: 16px;
                background: var(--gray-100);
                border-radius: 2px;
                overflow: hidden;
            }

            .gene-bar {
                height: 100%;
                border-radius: 2px;
                transition: width 0.4s ease;
            }

            .gene-bar.up { background: #c62828; }
            .gene-bar.down { background: var(--npj-blue); }

            .gene-value {
                width: 50px;
                font-size: 11px;
                font-weight: 600;
                color: var(--gray-700);
                text-align: right;
                font-family: var(--font-mono);
            }

            /* ========== PATHWAY LIST ========== */
            .pathway-list {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }

            .pathway-item {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 6px 0;
                border-bottom: 1px solid var(--gray-100);
            }

            .pathway-item:last-child { border-bottom: none; }

            .pathway-name {
                flex: 1;
                font-size: 11px;
                font-weight: 500;
                color: var(--gray-800);
                line-height: 1.3;
            }

            .pathway-dots {
                font-size: 10px;
                color: var(--success);
                letter-spacing: 1px;
            }

            .pathway-genes {
                min-width: 28px;
                font-size: 10px;
                font-weight: 600;
                color: var(--npj-blue);
                text-align: center;
                background: var(--npj-blue-light);
                padding: 2px 6px;
                border-radius: 2px;
            }

            /* ========== TABLES (npj Journal Style) ========== */
            .detailed-findings {
                background: white;
                margin-bottom: 32px;
            }

            .detailed-findings h2 {
                font-size: 20px;
                margin-bottom: 20px;
            }

            .table-controls {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                gap: 16px;
            }

            .search-input {
                flex: 1;
                max-width: 280px;
                padding: 8px 12px;
                border: 1px solid var(--gray-300);
                border-radius: 3px;
                font-size: 13px;
                transition: border-color 0.2s;
            }

            .search-input:focus {
                outline: none;
                border-color: var(--npj-blue);
            }

            .filter-buttons { display: flex; gap: 4px; }

            .filter-btn {
                padding: 6px 12px;
                border: 1px solid var(--gray-300);
                border-radius: 3px;
                background: white;
                color: var(--gray-700);
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.15s;
            }

            .filter-btn:hover {
                background: var(--gray-50);
                border-color: var(--gray-400);
            }

            .filter-btn.active {
                background: var(--npj-blue);
                border-color: var(--npj-blue);
                color: white;
            }

            .table-container {
                max-height: 450px;
                overflow-y: auto;
                border: 1px solid var(--gray-200);
            }

            #gene-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }

            #gene-table th {
                background: var(--gray-50);
                padding: 10px 12px;
                text-align: left;
                font-size: 11px;
                font-weight: 700;
                color: var(--gray-700);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 2px solid var(--gray-300);
                position: sticky;
                top: 0;
                cursor: pointer;
            }

            #gene-table th:hover { background: var(--gray-100); }

            #gene-table td {
                padding: 8px 12px;
                border-bottom: 1px solid var(--gray-100);
                font-size: 12px;
            }

            #gene-table tbody tr:hover { background: var(--npj-blue-light); }

            #gene-table td.up { color: #c62828; font-weight: 600; }
            #gene-table td.down { color: var(--npj-blue); font-weight: 600; }

            .badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 2px;
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            .badge.high { background: #c8e6c9; color: #1b5e20; }
            .badge.medium { background: #fff3e0; color: #e65100; }
            .badge.low { background: #ffebee; color: #b71c1c; }
            .badge.novel_candidate { background: #e8eaf6; color: #283593; }
            .badge.requires_validation { background: var(--gray-100); color: var(--gray-600); }

            .table-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid var(--gray-200);
            }

            .table-footer span {
                font-size: 12px;
                color: var(--gray-600);
            }

            .download-btn {
                padding: 6px 14px;
                background: var(--npj-blue);
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s;
            }

            .download-btn:hover { background: var(--npj-blue-dark); }

            /* ========== GENE STATUS CARDS ========== */
            .gene-status-cards {
                background: white;
                margin-bottom: 32px;
            }

            .gene-status-cards h2 {
                font-size: 20px;
                margin-bottom: 20px;
            }

            .cards-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 16px;
            }

            .gene-status-card {
                background: var(--gray-50);
                border: 1px solid var(--gray-200);
                overflow: hidden;
                transition: box-shadow 0.2s;
            }

            .gene-status-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                background: white;
                border-bottom: 1px solid var(--gray-200);
            }

            .gene-info { display: flex; align-items: center; gap: 8px; }

            .gene-symbol {
                font-size: 14px;
                font-weight: 700;
                color: var(--npj-blue);
                font-family: var(--font-mono);
            }

            .gene-rank {
                font-size: 10px;
                color: var(--gray-500);
                font-weight: 500;
            }

            .card-body { padding: 16px; }

            .stat-row {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }

            .stat-label {
                width: 72px;
                font-size: 11px;
                color: var(--gray-600);
                font-weight: 500;
            }

            .stat-bar-container {
                flex: 1;
                height: 6px;
                background: var(--gray-200);
                border-radius: 3px;
                margin: 0 10px;
                overflow: hidden;
            }

            .stat-bar {
                height: 100%;
                border-radius: 3px;
            }

            .stat-bar.up { background: #c62828; }
            .stat-bar.down { background: var(--npj-blue); }

            .stat-value {
                font-size: 11px;
                color: var(--gray-700);
            }

            .db-tags { display: flex; gap: 4px; flex-wrap: wrap; }

            .db-tag {
                background: var(--npj-blue);
                color: white;
                padding: 1px 6px;
                border-radius: 2px;
                font-size: 9px;
                font-weight: 600;
                text-transform: uppercase;
            }

            .cancer-match {
                background: var(--success);
                color: white;
                padding: 1px 6px;
                border-radius: 2px;
                font-size: 9px;
                font-weight: 600;
            }

            .confidence-dots { font-size: 12px; }

            /* ========== RAG INTERPRETATION ========== */
            .rag-interpretation {
                margin-top: 12px;
                padding: 10px;
                background: var(--npj-blue-light);
                border-left: 3px solid var(--npj-blue);
            }

            .rag-label {
                font-size: 10px;
                font-weight: 700;
                color: var(--npj-blue);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }

            .rag-text {
                font-size: 11px;
                color: var(--gray-700);
                line-height: 1.5;
                margin: 0 0 6px 0;
            }

            .rag-citations { display: flex; gap: 6px; flex-wrap: wrap; }

            .pmid-link {
                font-size: 10px;
                color: var(--npj-blue);
                background: white;
                padding: 2px 6px;
                border-radius: 2px;
                text-decoration: none;
                border: 1px solid var(--npj-blue);
                font-weight: 500;
            }

            .pmid-link:hover {
                background: var(--npj-blue);
                color: white;
            }

            .card-footer {
                padding: 10px 16px;
                background: white;
                border-top: 1px solid var(--gray-200);
            }

            .tags { display: flex; gap: 4px; flex-wrap: wrap; }

            .tag {
                background: var(--gray-200);
                color: var(--gray-700);
                padding: 2px 6px;
                border-radius: 2px;
                font-size: 9px;
                font-weight: 600;
            }

            .tag.rag-tag {
                background: var(--npj-blue);
                color: white;
            }

            .gene-status-card.has-rag {
                border-color: var(--npj-blue);
                border-width: 2px;
            }

            /* ========== RAG SUMMARY SECTION ========== */
            .rag-summary {
                background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
                padding: 28px;
                margin-bottom: 32px;
                color: white;
            }

            .rag-summary-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 20px;
                flex-wrap: wrap;
                gap: 16px;
            }

            .rag-title-section h2 {
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 4px;
                color: white;
                border-bottom: none;
                padding-bottom: 0;
            }

            .rag-subtitle {
                font-size: 12px;
                color: rgba(255,255,255,0.7);
            }

            .rag-stats { display: flex; gap: 20px; }

            .rag-stat { text-align: center; }

            .rag-stat-value {
                display: block;
                font-size: 24px;
                font-weight: 700;
                color: #90caf9;
            }

            .rag-stat-label {
                font-size: 10px;
                color: rgba(255,255,255,0.6);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .rag-method-note {
                display: flex;
                align-items: flex-start;
                gap: 10px;
                background: rgba(255,255,255,0.1);
                padding: 12px;
                margin-bottom: 20px;
            }

            .method-icon { font-size: 20px; }

            .method-text {
                font-size: 12px;
                color: rgba(255,255,255,0.85);
                line-height: 1.5;
            }

            .method-text strong { color: #90caf9; }

            .rag-genes-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
                gap: 12px;
                margin-bottom: 16px;
            }

            .rag-gene-card {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.1);
                overflow: hidden;
            }

            .rag-gene-card:hover {
                background: rgba(255,255,255,0.12);
            }

            .rag-gene-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 14px;
                background: rgba(0,0,0,0.2);
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }

            .rag-gene-title { display: flex; align-items: center; gap: 8px; }

            .rag-gene-symbol {
                font-size: 13px;
                font-weight: 700;
                color: #90caf9;
                font-family: var(--font-mono);
            }

            .rag-gene-fc {
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 2px;
            }

            .rag-gene-fc.up { background: rgba(198,40,40,0.4); color: #ef9a9a; }
            .rag-gene-fc.down { background: rgba(0,86,185,0.4); color: #90caf9; }

            .hub-indicator {
                background: var(--npj-orange);
                color: white;
                font-size: 9px;
                padding: 2px 5px;
                border-radius: 2px;
                font-weight: 700;
                text-transform: uppercase;
            }

            .rag-confidence {
                font-size: 9px;
                padding: 3px 8px;
                border-radius: 2px;
                font-weight: 600;
                text-transform: uppercase;
            }

            .rag-confidence.high { background: rgba(46,125,50,0.4); color: #a5d6a7; }
            .rag-confidence.medium { background: rgba(245,158,11,0.4); color: #ffcc80; }
            .rag-confidence.low { background: rgba(117,117,117,0.4); color: #bdbdbd; }

            .rag-gene-body { padding: 14px; }

            .rag-interpretation-text {
                font-size: 12px;
                color: rgba(255,255,255,0.85);
                line-height: 1.5;
                margin: 0 0 10px 0;
            }

            .rag-pmids { display: flex; gap: 6px; flex-wrap: wrap; }

            .pmid-chip {
                font-size: 10px;
                padding: 3px 8px;
                background: rgba(63,81,181,0.5);
                color: #c5cae9;
                border-radius: 2px;
                text-decoration: none;
                font-weight: 500;
            }

            .pmid-chip:hover {
                background: rgba(63,81,181,0.8);
                color: white;
            }

            .no-pmid {
                font-size: 11px;
                color: rgba(255,255,255,0.4);
                font-style: italic;
            }

            .rag-disclaimer {
                display: flex;
                align-items: center;
                gap: 8px;
                background: rgba(255,152,0,0.15);
                border: 1px solid rgba(255,152,0,0.3);
                padding: 10px 14px;
                font-size: 11px;
                color: #ffcc80;
            }

            /* ========== METHODS SECTION ========== */
            .methods-section {
                background: white;
                margin-bottom: 32px;
            }

            .methods-section h2 {
                font-size: 20px;
                margin-bottom: 20px;
            }

            .methods-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 20px;
            }

            .method-card {
                background: var(--gray-50);
                padding: 16px;
                border-left: 3px solid var(--npj-blue);
            }

            .method-card h4 {
                font-size: 13px;
                font-weight: 700;
                margin-bottom: 10px;
                color: var(--gray-900);
            }

            .method-card ul {
                list-style: none;
                font-size: 12px;
                color: var(--gray-700);
            }

            .method-card li {
                padding: 3px 0;
                padding-left: 12px;
                position: relative;
            }

            .method-card li::before {
                content: '•';
                position: absolute;
                left: 0;
                color: var(--npj-blue);
            }

            .confidence-explanation {
                background: var(--gray-50);
                padding: 16px;
            }

            .confidence-explanation h4 {
                font-size: 13px;
                font-weight: 700;
                margin-bottom: 12px;
            }

            .score-table {
                width: 100%;
                margin-bottom: 12px;
                font-size: 12px;
            }

            .score-table td {
                padding: 6px 8px;
                border-bottom: 1px solid var(--gray-200);
            }

            .score-table td:last-child {
                text-align: right;
                font-weight: 700;
                color: var(--npj-blue);
            }

            .confidence-legend {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                font-size: 11px;
                color: var(--gray-600);
            }

            /* ========== EXECUTIVE SUMMARY ========== */
            .executive-summary {
                background: white;
                border: 1px solid var(--gray-200);
                padding: 24px;
                margin-bottom: 24px;
            }

            .summary-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }

            .summary-title { display: flex; align-items: center; gap: 12px; }

            .summary-title h2 {
                font-size: 18px;
                color: var(--gray-900);
                border-bottom: none;
                padding-bottom: 0;
                margin-bottom: 0;
            }

            .confidence-badge {
                padding: 4px 12px;
                border-radius: 2px;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .confidence-badge.high { background: #c8e6c9; color: #1b5e20; }
            .confidence-badge.medium { background: #fff3e0; color: #e65100; }
            .confidence-badge.low { background: #ffebee; color: #b71c1c; }

            .key-metrics {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-bottom: 20px;
            }

            .metric-card {
                background: var(--gray-50);
                padding: 20px;
                text-align: center;
                border: 1px solid var(--gray-200);
            }

            .metric-card.primary {
                background: var(--npj-blue-light);
                border-color: var(--npj-blue);
            }

            .metric-card.highlight {
                background: #fff8e1;
                border-color: var(--npj-orange);
            }

            .metric-value {
                font-size: 28px;
                font-weight: 700;
                color: var(--npj-blue);
            }

            .metric-card.highlight .metric-value { color: var(--npj-orange); }

            .metric-label {
                color: var(--gray-600);
                font-size: 12px;
                font-weight: 500;
                margin-top: 4px;
            }

            .metric-detail {
                font-size: 11px;
                color: var(--gray-500);
                margin-top: 6px;
            }

            .one-line-summary {
                background: var(--gray-50);
                padding: 14px 16px;
                margin-bottom: 14px;
            }

            .one-line-summary h4 {
                font-size: 11px;
                color: var(--gray-600);
                margin-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .one-line-summary p {
                font-size: 13px;
                color: var(--gray-800);
                line-height: 1.5;
            }

            .warning-box {
                display: flex;
                align-items: center;
                gap: 10px;
                background: #fff8e1;
                border: 1px solid #ffe082;
                padding: 10px 14px;
                font-size: 12px;
                color: #f57c00;
            }

            .warning-icon { font-size: 16px; }

            /* ========== FOOTER ========== */
            .paper-footer {
                background: var(--gray-900);
                color: white;
                padding: 32px 24px;
                text-align: center;
            }

            .footer-content {
                max-width: 700px;
                margin: 0 auto;
            }

            .paper-footer p {
                font-size: 12px;
                line-height: 1.6;
                color: rgba(255,255,255,0.8);
            }

            .footer-credit {
                margin-top: 16px;
                font-size: 11px;
                color: rgba(255,255,255,0.5);
            }

            .no-data {
                color: var(--gray-400);
                font-style: italic;
                padding: 16px;
                text-align: center;
                font-size: 12px;
            }

            /* ========== RESPONSIVE ========== */
            @media (max-width: 768px) {
                .cover-title { font-size: 24px; }
                .cover-stats { gap: 24px; }
                .cover-stat .stat-number { font-size: 28px; }
                .key-metrics { grid-template-columns: 1fr; }
                .dashboard-grid { grid-template-columns: 1fr; }
                .dashboard-panel.main-plot { grid-row: auto; }
                .cards-grid { grid-template-columns: 1fr; }
                .rag-genes-grid { grid-template-columns: 1fr; }
                .table-controls { flex-direction: column; align-items: stretch; }
                .nav-links { display: none; }
            }

            @media print {
                .nav-bar { display: none; }
                .cover-page { min-height: auto; padding: 40px; }
                .paper-content { padding: 20px; }
                .toggle-btn, .filter-btn, .download-btn { display: none; }
            }
        </style>
        '''

    def _generate_javascript(self, data: Dict) -> str:
        """Generate interactive JavaScript."""
        # Prepare data for CSV download
        integrated = data.get('integrated_gene_table', [])
        csv_data = "gene_id,gene_symbol,log2FC,padj,direction,is_hub,db_matched,confidence,score\\n"
        for g in integrated[:500]:
            csv_data += f"{g.get('gene_id','')},{g.get('gene_symbol','')},{g.get('log2FC',0):.4f},{g.get('padj',1):.2e},{g.get('direction','')},{g.get('is_hub',False)},{g.get('db_matched',False)},{g.get('confidence','')},{g.get('interpretation_score',0):.2f}\\n"

        return f'''
        <script>
            // Table filtering
            function filterTable() {{
                const input = document.getElementById('gene-search');
                const filter = input.value.toLowerCase();
                const table = document.getElementById('gene-table');
                const rows = table.getElementsByTagName('tr');

                for (let i = 1; i < rows.length; i++) {{
                    const cells = rows[i].getElementsByTagName('td');
                    let found = false;
                    for (let j = 0; j < cells.length; j++) {{
                        if (cells[j].textContent.toLowerCase().includes(filter)) {{
                            found = true;
                            break;
                        }}
                    }}
                    rows[i].style.display = found ? '' : 'none';
                }}
            }}

            // Filter by confidence
            let activeFilter = 'all';
            function filterByConfidence(level) {{
                activeFilter = level;
                const buttons = document.querySelectorAll('.filter-btn');
                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                const table = document.getElementById('gene-table');
                const rows = table.getElementsByTagName('tr');

                for (let i = 1; i < rows.length; i++) {{
                    const badge = rows[i].querySelector('.badge');
                    if (level === 'all' || (badge && badge.classList.contains(level))) {{
                        rows[i].style.display = '';
                    }} else {{
                        rows[i].style.display = 'none';
                    }}
                }}
            }}

            // Table sorting
            let sortDirection = {{}};
            function sortTable(colIndex) {{
                const table = document.getElementById('gene-table');
                const rows = Array.from(table.rows).slice(1);
                const dir = sortDirection[colIndex] = !sortDirection[colIndex];

                rows.sort((a, b) => {{
                    let aVal = a.cells[colIndex].textContent;
                    let bVal = b.cells[colIndex].textContent;

                    // Try numeric sort
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return dir ? aNum - bNum : bNum - aNum;
                    }}

                    // String sort
                    return dir ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }});

                const tbody = table.querySelector('tbody');
                rows.forEach(row => tbody.appendChild(row));
            }}

            // CSV download
            function downloadCSV() {{
                const csvContent = "{csv_data}";
                const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = 'rnaseq_analysis_results.csv';
                link.click();
            }}

            // Smooth scroll for navigation
            document.querySelectorAll('.nav-pill').forEach(link => {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{ behavior: 'smooth' }});
                    }}
                }});
            }});

            // Volcano view toggle
            function showVolcanoView(view) {{
                const interactiveView = document.getElementById('volcano-interactive');
                const staticView = document.getElementById('volcano-static');
                const buttons = document.querySelectorAll('.volcano-container .view-toggle .toggle-btn');

                if (view === 'interactive') {{
                    interactiveView.classList.add('active');
                    staticView.classList.remove('active');
                    interactiveView.style.display = 'block';
                    staticView.style.display = 'none';
                    buttons[0].classList.add('active');
                    buttons[1].classList.remove('active');
                }} else {{
                    interactiveView.classList.remove('active');
                    staticView.classList.add('active');
                    interactiveView.style.display = 'none';
                    staticView.style.display = 'block';
                    buttons[0].classList.remove('active');
                    buttons[1].classList.add('active');
                }}
            }}

            function showNetworkView(view) {{
                const interactiveView = document.getElementById('network-interactive');
                const staticView = document.getElementById('network-static');
                const buttons = document.querySelectorAll('.network-container .view-toggle .toggle-btn');

                if (view === 'interactive') {{
                    interactiveView.classList.add('active');
                    staticView.classList.remove('active');
                    interactiveView.style.display = 'block';
                    staticView.style.display = 'none';
                    buttons[0].classList.add('active');
                    buttons[1].classList.remove('active');
                }} else {{
                    interactiveView.classList.remove('active');
                    staticView.classList.add('active');
                    interactiveView.style.display = 'none';
                    staticView.style.display = 'block';
                    buttons[0].classList.remove('active');
                    buttons[1].classList.add('active');
                }}
            }}
        </script>
        '''

    def _generate_cover_page_html(self, data: Dict) -> str:
        """Generate Cell-style cover page."""
        interpretation = data.get('interpretation_report', {})
        cancer_type = interpretation.get('cancer_type', self.config.get('cancer_type', 'Unknown'))
        cancer_type_kr = {
            'breast_cancer': '유방암',
            'lung_cancer': '폐암',
            'pancreatic_cancer': '췌장암',
            'colorectal_cancer': '대장암'
        }.get(cancer_type, cancer_type)

        # Get stats
        deg_count = len(data.get('deg_significant', []))
        hub_count = len(data.get('hub_genes', []))
        pathway_count = len(data.get('pathway_summary', []))

        return f'''
        <section class="cover-page">
            <div class="cover-content">
                <div class="cover-badge">RNA-seq Differential Expression Analysis</div>
                <h1 class="cover-title">{cancer_type_kr} 전사체 분석 보고서</h1>
                <p class="cover-subtitle">Comprehensive Transcriptomic Profiling and Pathway Analysis</p>

                <div class="cover-stats">
                    <div class="cover-stat">
                        <span class="stat-number">{deg_count:,}</span>
                        <span class="stat-label">Differentially Expressed Genes</span>
                    </div>
                    <div class="cover-stat">
                        <span class="stat-number">{hub_count}</span>
                        <span class="stat-label">Hub Genes Identified</span>
                    </div>
                    <div class="cover-stat">
                        <span class="stat-number">{pathway_count}</span>
                        <span class="stat-label">Enriched Pathways</span>
                    </div>
                </div>

                <div class="cover-meta">
                    <p><strong>Analysis Date:</strong> {datetime.now().strftime("%B %d, %Y")}</p>
                    <p><strong>Pipeline:</strong> BioInsight AI RNA-seq Pipeline v2.0</p>
                    <p><strong>Methods:</strong> DESeq2, WGCNA Network Analysis, GO/KEGG Enrichment</p>
                </div>
            </div>
            <div class="cover-footer">
                <p>This report was generated using AI-assisted analysis. All findings require experimental validation.</p>
            </div>
        </section>
        '''

    def _generate_abstract_html(self, data: Dict) -> str:
        """Generate paper-style extended abstract/summary section."""
        # Try to load extended abstract first
        extended_abstract = data.get('abstract_extended', {})

        if extended_abstract and extended_abstract.get('abstract_extended'):
            abstract_text = extended_abstract['abstract_extended']
            key_findings = extended_abstract.get('key_findings', [])
            validation = extended_abstract.get('validation_priorities', {})
            ml_interp = extended_abstract.get('ml_interpretation', '')

            # Format abstract with paragraphs
            paragraphs = abstract_text.split('\n\n')
            formatted_paragraphs = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])

            # Key findings list
            findings_html = ''
            if key_findings:
                findings_html = '<div class="key-findings"><h4>📌 주요 발견</h4><ul>'
                for finding in key_findings[:6]:
                    findings_html += f'<li>{finding}</li>'
                findings_html += '</ul></div>'

            # Validation priorities
            validation_html = ''
            if validation:
                validation_html = '<div class="validation-priorities"><h4>🧬 실험적 검증 제안</h4><div class="validation-grid">'
                if validation.get('qPCR'):
                    validation_html += f'<div class="validation-item"><strong>qRT-PCR:</strong> {", ".join(validation["qPCR"][:5])}</div>'
                if validation.get('western_blot'):
                    validation_html += f'<div class="validation-item"><strong>Western Blot:</strong> {", ".join(validation["western_blot"][:3])}</div>'
                if validation.get('functional_study'):
                    validation_html += f'<div class="validation-item"><strong>Functional Study:</strong> {", ".join(validation["functional_study"][:3])}</div>'
                if validation.get('biomarker_candidates'):
                    validation_html += f'<div class="validation-item"><strong>Biomarker 후보:</strong> {", ".join(validation["biomarker_candidates"][:3])}</div>'
                validation_html += '</div></div>'

            # ML interpretation
            ml_html = ''
            if ml_interp:
                ml_html = f'<div class="ml-interpretation"><h4>🤖 ML 예측 해석</h4><p>{ml_interp}</p></div>'

            return f'''
        <section class="abstract-section" id="abstract">
            <h2>Extended Abstract</h2>
            <div class="abstract-box extended">
                <div class="abstract-content">
                    {formatted_paragraphs}
                </div>
                {findings_html}
                {validation_html}
                {ml_html}
            </div>
        </section>
            '''

        # Fallback to basic abstract
        interpretation = data.get('interpretation_report', {})
        rag_data = interpretation.get('rag_interpretation', {})
        summary = rag_data.get('summary', '')

        deg_count = len(data.get('deg_significant', []))
        hub_genes = data.get('hub_genes', [])
        hub_names = [g.get('gene_symbol', g.get('gene_id', '')) for g in hub_genes[:5]]

        pathways = data.get('pathway_summary', [])[:3]
        pathway_names = [p.get('term_name', '')[:50] for p in pathways]

        if not summary:
            summary = f'''
본 분석에서는 {deg_count:,}개의 차등발현 유전자(DEGs)를 식별하였습니다.
네트워크 분석을 통해 {len(hub_genes)}개의 Hub 유전자({", ".join(hub_names[:3])} 등)가
핵심 조절자로 확인되었습니다. Pathway enrichment 분석 결과,
{", ".join(pathway_names[:2])} 등의 경로가 유의하게 농축되었습니다.
            '''

        return f'''
        <section class="abstract-section" id="abstract">
            <h2>Abstract</h2>
            <div class="abstract-box">
                <div class="abstract-content">
                    <p><strong>Background:</strong> RNA-seq 기반 전사체 분석을 통해 질환 관련 유전자 발현 변화를 체계적으로 분석하였습니다.</p>
                    <p><strong>Methods:</strong> DESeq2를 이용한 차등발현 분석, WGCNA 기반 네트워크 분석, GO/KEGG pathway enrichment 분석을 수행하였습니다.</p>
                    <p><strong>Results:</strong> {summary.strip()}</p>
                    <p><strong>Conclusions:</strong> 본 분석에서 확인된 Hub 유전자와 enriched pathway는 후속 기능 연구 및 바이오마커 개발의 유망한 후보입니다.</p>
                </div>
                <div class="abstract-keywords">
                    <strong>Keywords:</strong> RNA-seq, Differential Expression, Network Analysis, {", ".join(hub_names[:3])}
                </div>
            </div>
        </section>
        '''

    def _generate_html(self, data: Dict) -> str:
        """Generate complete HTML report in Cell journal style."""
        interpretation = data.get('interpretation_report', {})
        cancer_type = interpretation.get('cancer_type', self.config.get('cancer_type', 'Unknown'))

        return f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config["report_title"]}</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    {self._generate_css()}
</head>
<body>
    <!-- Cover Page -->
    {self._generate_cover_page_html(data)}

    <!-- Navigation -->
    <nav class="nav-bar">
        <div class="nav-container">
            <span class="nav-brand">BioInsight Report</span>
            <div class="nav-links">
                <a href="#abstract">Abstract</a>
                <a href="#figures">Figures</a>
                <a href="#rag-summary">Literature Analysis</a>
                <a href="#gene-cards">Key Genes</a>
                <a href="#methods">Methods</a>
            </div>
        </div>
    </nav>

    <main class="paper-content">
        <!-- Abstract -->
        {self._generate_abstract_html(data)}

        <!-- Figures Section -->
        <section class="figures-section" id="figures">
            <h2>Figures</h2>
            {self._generate_visual_dashboard_html(data)}
        </section>

        <!-- RAG Literature Analysis -->
        {self._generate_rag_summary_html(data)}

        <!-- Key Genes -->
        <section class="genes-section" id="gene-cards">
            <h2>Key Gene Analysis</h2>
            {self._generate_gene_status_cards_html(data)}
        </section>

        <!-- Detailed Data Table -->
        <section class="data-section" id="detailed-table">
            <h2>Supplementary Data</h2>
            {self._generate_detailed_table_html(data)}
        </section>

        <!-- Methods -->
        {self._generate_methods_html() if self.config["include_methods"] else ""}
    </main>

    <footer class="paper-footer">
        <div class="footer-content">
            <p><strong>Disclaimer:</strong> This report is generated by AI-assisted analysis pipeline.
            All findings are preliminary and require experimental validation before clinical application.</p>
            <p class="footer-credit">Generated by BioInsight AI RNA-seq Pipeline v2.0 | {datetime.now().strftime("%Y-%m-%d")}</p>
        </div>
    </footer>

    {self._generate_javascript(data)}
</body>
</html>
'''

    def run(self) -> Dict[str, Any]:
        """Generate the HTML report."""
        data = self._load_all_data()

        # Generate extended abstract if not already present
        if 'abstract_extended' not in data:
            self.logger.info("Generating extended abstract with Claude API...")
            extended_abstract = self._generate_extended_abstract(data)
            if extended_abstract:
                data['abstract_extended'] = extended_abstract

        # Generate visualization interpretations
        run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
        viz_interp_path = run_dir / "visualization_interpretations.json"
        if viz_interp_path.exists():
            try:
                with open(viz_interp_path, 'r', encoding='utf-8') as f:
                    data['visualization_interpretations'] = json.load(f)
                self.logger.info("Loaded existing visualization interpretations")
            except Exception as e:
                self.logger.warning(f"Error loading visualization interpretations: {e}")
        else:
            self.logger.info("Generating visualization interpretations with Claude API...")
            viz_interpretations = self._generate_visualization_interpretations(data)
            if viz_interpretations:
                data['visualization_interpretations'] = viz_interpretations

        self.save_json(data, "report_data.json")

        html_content = self._generate_html(data)

        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Report generated: {report_path}")

        return {
            "report_path": str(report_path),
            "data_sources_loaded": list(data.keys()),
            "figures_embedded": len(data.get('figures', {}))
        }

    def _generate_extended_abstract(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate extended abstract using Claude API.

        Creates a comprehensive abstract with:
        - Background, Methods, Results, Conclusions (Korean)
        - Key findings list
        - Validation priorities (qPCR, Western blot, Functional study, Biomarker candidates)
        - ML prediction interpretation
        - RAG-based literature interpretation
        """
        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("anthropic package not available, skipping extended abstract generation")
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set, skipping extended abstract generation")
            return None

        # Prepare analysis summary for Claude
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        integrated_df = data.get('integrated_gene_table_df')
        interpretation = data.get('interpretation_report', {})

        # Basic stats - handle both 'log2FC' and 'log2FoldChange' column names
        n_deg = len(deg_df) if deg_df is not None else 0
        log2fc_col = 'log2FC' if deg_df is not None and 'log2FC' in deg_df.columns else 'log2FoldChange'
        n_up = len(deg_df[deg_df[log2fc_col] > 0]) if deg_df is not None and log2fc_col in deg_df.columns else 0
        n_down = n_deg - n_up

        # Hub genes info - handle both 'gene_id' and 'gene_symbol' column names
        hub_genes_info = []
        if hub_df is not None and len(hub_df) > 0:
            hub_log2fc_col = 'log2FC' if 'log2FC' in hub_df.columns else 'log2FoldChange'
            for _, row in hub_df.head(10).iterrows():
                gene_name = row.get('gene_id', row.get('gene_symbol', row.get('gene_name', 'Unknown')))
                degree = row.get('degree', 0)
                log2fc = row.get(hub_log2fc_col, 0)
                hub_genes_info.append(f"- {gene_name} (degree={degree}, log2FC={log2fc:.2f})")

        # Pathway info
        pathway_info = []
        if pathway_df is not None and len(pathway_df) > 0:
            for _, row in pathway_df.head(5).iterrows():
                term = row.get('Term', row.get('term', 'Unknown'))
                pval = row.get('P-value', row.get('pvalue', 0))
                pathway_info.append(f"- {term} (p={pval:.2e})")

        # ML prediction info (check for prediction files)
        ml_info = ""
        run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
        ml_prediction_path = run_dir / "ml_prediction" / "prediction_summary.json"
        if ml_prediction_path.exists():
            try:
                with open(ml_prediction_path, 'r') as f:
                    ml_data = json.load(f)
                ml_info = f"""
ML 예측 결과:
- 총 샘플 수: {ml_data.get('total_samples', 0)}
- 예측 분포: {ml_data.get('prediction_distribution', {})}
- 평균 신뢰도: {ml_data.get('average_confidence', 0):.2f}
- 예상 암종: {ml_data.get('expected_cancer', 'Unknown')}
- 직접 예측율: {ml_data.get('brca_hit_rate', 0) * 100:.1f}%
- Top-3 예측율: {ml_data.get('brca_in_top3_rate', 0) * 100:.1f}%
- 유전자 매칭율: {ml_data.get('gene_matching_rate', 0) * 100:.1f}%
"""
            except Exception as e:
                self.logger.warning(f"Error loading ML prediction: {e}")

        # RAG interpretation info (load from rag_interpretations.json)
        rag_info = ""
        rag_summary = {"genes_analyzed": 0, "key_findings": [], "pmids": []}
        rag_path = self.input_dir / "rag_interpretations.json"
        if rag_path.exists():
            try:
                with open(rag_path, 'r', encoding='utf-8') as f:
                    rag_data = json.load(f)
                rag_summary["genes_analyzed"] = rag_data.get('genes_interpreted', 0)

                # Extract key interpretations and PMIDs
                interpretations = rag_data.get('interpretations', {})
                literature_supported = []
                novel_candidates = []
                all_pmids = set()

                for gene, gene_data in interpretations.items():
                    interp = gene_data.get('interpretation', '')
                    pmids = gene_data.get('pmids', [])
                    log2fc = gene_data.get('log2fc', 0)
                    direction = gene_data.get('direction', '')

                    all_pmids.update(pmids)

                    # Check if literature supports this gene
                    if 'cannot' not in interp.lower() and 'not directly' not in interp.lower():
                        literature_supported.append({
                            'gene': gene, 'log2fc': log2fc,
                            'interpretation': interp[:200], 'pmids': pmids
                        })
                    else:
                        novel_candidates.append(gene)

                rag_summary["pmids"] = list(all_pmids)
                rag_summary["literature_supported"] = literature_supported[:5]
                rag_summary["novel_candidates"] = novel_candidates[:10]

                rag_info = f"""
RAG 기반 문헌 해석 결과:
- 분석된 유전자 수: {rag_summary['genes_analyzed']}개
- 참조된 PMID 수: {len(all_pmids)}개
- 문헌 지원 유전자: {', '.join([g['gene'] for g in literature_supported[:5]]) if literature_supported else '없음'}
- 신규 바이오마커 후보 (기존 문헌 미기재): {', '.join(novel_candidates[:5]) if novel_candidates else '없음'}
"""
                self.logger.info(f"Loaded RAG interpretations: {rag_summary['genes_analyzed']} genes")
            except Exception as e:
                self.logger.warning(f"Error loading RAG interpretations: {e}")

        # Study info from config
        study_name = self.config.get('study_name', 'RNA-seq Analysis')
        cancer_type = self.config.get('cancer_type', 'cancer')

        # Build prompt
        prompt = f"""당신은 바이오인포매틱스 전문가입니다. 아래 RNA-seq 분석 결과를 바탕으로 학술 논문 스타일의 확장된 초록(Extended Abstract)을 작성해주세요.

## 분석 정보
- 연구명: {study_name}
- 암종: {cancer_type}
- 총 DEG 수: {n_deg}개 (상향조절: {n_up}개, 하향조절: {n_down}개)

## Hub 유전자 (Top 10)
{chr(10).join(hub_genes_info) if hub_genes_info else '정보 없음'}

## 주요 Pathway
{chr(10).join(pathway_info) if pathway_info else '정보 없음'}

{ml_info}

{rag_info}

## 요청 사항
다음 JSON 형식으로 응답해주세요:

```json
{{
  "title": "한국어 제목",
  "title_en": "English Title",
  "abstract_extended": "**배경(Background)**: ...\\n\\n**방법(Methods)**: ...\\n\\n**결과(Results)**: ...\\n\\n**ML 예측 분석(Predictive Analysis)**: ...\\n\\n**RAG 문헌 해석(Literature-based Interpretation)**: ...\\n\\n**실험적 검증 제안(Suggested Validations)**: ...\\n\\n**결론 및 의의(Conclusions)**: ...",
  "key_findings": [
    "주요 발견 1",
    "주요 발견 2",
    ...
  ],
  "validation_priorities": {{
    "qPCR": ["gene1", "gene2", ...],
    "western_blot": ["gene1", "gene2", ...],
    "functional_study": ["gene1", "gene2", ...],
    "biomarker_candidates": ["gene1", "gene2", ...]
  }},
  "ml_interpretation": "ML 예측 결과에 대한 해석 (Top-3 예측율, 유전자 매칭율 등 포함)",
  "rag_interpretation": "RAG 문헌 해석 결과 요약 (문헌 지원 유전자 vs 신규 바이오마커 후보 구분)",
  "literature_sources": {{
    "pmid_count": {len(rag_summary.get('pmids', []))},
    "key_pmids": {rag_summary.get('pmids', [])[:5]}
  }}
}}
```

중요:
1. 한국어로 작성 (영문 제목만 영어)
2. 발견된 Hub 유전자를 validation_priorities에 실제 유전자명으로 포함
3. ML 예측의 Top-3 예측율과 유전자 매칭율을 해석에 포함
4. RAG 문헌 해석 섹션 필수 포함 - 문헌 지원 유전자와 신규 바이오마커 후보를 구분하여 설명
5. PMID 인용 형식 사용 (예: [PMID: 35409110])
6. 실험적 검증 방법을 구체적으로 제안
"""

        try:
            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                extended_abstract = json.loads(json_str)

                # Save to file
                output_path = run_dir / "abstract_extended.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(extended_abstract, f, ensure_ascii=False, indent=2)

                self.logger.info(f"Extended abstract generated: {output_path}")
                return extended_abstract
            else:
                self.logger.warning("Could not extract JSON from Claude response")
                return None

        except Exception as e:
            self.logger.error(f"Error generating extended abstract: {e}")
            return None

    def _generate_visualization_interpretations(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate LLM-based interpretations for each visualization.

        Creates structured interpretations for:
        - Volcano Plot: DEG 분포 해석
        - Heatmap: 발현 패턴 해석
        - Network Graph: Hub 유전자 및 상호작용 해석
        - PCA Plot: 샘플 분리도 해석
        - Pathway Bar Plot: 경로 분석 해석
        """
        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("anthropic package not available, skipping visualization interpretations")
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set, skipping visualization interpretations")
            return None

        # Prepare data summaries for each visualization
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        network_nodes_df = data.get('network_nodes_df')

        # DEG stats for volcano
        n_deg = len(deg_df) if deg_df is not None else 0
        log2fc_col = 'log2FC' if deg_df is not None and 'log2FC' in deg_df.columns else 'log2FoldChange'
        n_up = len(deg_df[deg_df[log2fc_col] > 0]) if deg_df is not None and log2fc_col in deg_df.columns else 0
        n_down = n_deg - n_up

        # Top DEGs by |log2FC|
        top_up_genes = []
        top_down_genes = []
        if deg_df is not None and log2fc_col in deg_df.columns:
            deg_sorted = deg_df.sort_values(log2fc_col, ascending=False)
            gene_col = 'gene_id' if 'gene_id' in deg_df.columns else 'gene_symbol'
            top_up_genes = deg_sorted.head(5)[gene_col].tolist()
            top_down_genes = deg_sorted.tail(5)[gene_col].tolist()

        # Hub genes for network
        hub_genes_list = []
        if hub_df is not None:
            hub_log2fc_col = 'log2FC' if 'log2FC' in hub_df.columns else 'log2FoldChange'
            for _, row in hub_df.head(10).iterrows():
                gene_name = row.get('gene_id', row.get('gene_symbol', 'Unknown'))
                degree = row.get('degree', 0)
                log2fc = row.get(hub_log2fc_col, 0)
                hub_genes_list.append(f"{gene_name}(degree={degree}, log2FC={log2fc:.2f})")

        # Network stats
        total_edges = len(data.get('network_edges', []))

        # Pathway info
        pathway_list = []
        if pathway_df is not None:
            for _, row in pathway_df.head(10).iterrows():
                term = row.get('Term', row.get('term', 'Unknown'))
                pval = row.get('P-value', row.get('pvalue', 0))
                genes = row.get('Genes', row.get('genes', ''))
                pathway_list.append(f"- {term}: p={pval:.2e}, genes=[{genes[:50]}...]")

        # Study info
        study_name = self.config.get('study_name', 'RNA-seq Analysis')
        cancer_type = self.config.get('cancer_type', 'cancer')

        prompt = f"""당신은 바이오인포매틱스 시각화 전문가입니다. 아래 RNA-seq 분석 결과의 각 시각화에 대한 해석을 제공해주세요.

## 분석 정보
- 연구명: {study_name}
- 암종: {cancer_type}
- 총 DEG 수: {n_deg}개 (상향조절: {n_up}개, 하향조절: {n_down}개)
- 상위 상향조절 유전자: {', '.join(top_up_genes)}
- 상위 하향조절 유전자: {', '.join(top_down_genes)}
- Hub 유전자: {', '.join(hub_genes_list[:5])}
- 총 네트워크 edge 수: {total_edges}

## Pathway 정보
{chr(10).join(pathway_list) if pathway_list else '정보 없음'}

다음 JSON 형식으로 각 시각화에 대한 해석을 제공해주세요:

```json
{{
  "volcano_plot": {{
    "title": "Volcano Plot 해석",
    "summary": "1-2문장 요약",
    "key_observations": [
      "관찰 1",
      "관찰 2",
      "관찰 3"
    ],
    "biological_significance": "생물학적 의미 설명",
    "interpretation_guide": "이 플롯을 해석하는 방법 안내"
  }},
  "heatmap": {{
    "title": "발현 히트맵 해석",
    "summary": "1-2문장 요약",
    "key_observations": [
      "관찰 1",
      "관찰 2"
    ],
    "pattern_analysis": "발현 패턴 분석",
    "interpretation_guide": "이 플롯을 해석하는 방법 안내"
  }},
  "network_graph": {{
    "title": "유전자 상호작용 네트워크 해석",
    "summary": "1-2문장 요약",
    "hub_gene_analysis": "Hub 유전자 분석",
    "network_topology": "네트워크 구조 특성",
    "biological_implications": "생물학적 의미",
    "interpretation_guide": "이 플롯을 해석하는 방법 안내"
  }},
  "pca_plot": {{
    "title": "PCA 분석 해석",
    "summary": "1-2문장 요약",
    "separation_analysis": "샘플 분리도 분석",
    "variance_explanation": "분산 설명",
    "interpretation_guide": "이 플롯을 해석하는 방법 안내"
  }},
  "pathway_barplot": {{
    "title": "Pathway 분석 해석",
    "summary": "1-2문장 요약",
    "top_pathways": [
      "주요 pathway 1 설명",
      "주요 pathway 2 설명"
    ],
    "functional_theme": "전체적인 기능적 테마",
    "therapeutic_implications": "치료적 함의",
    "interpretation_guide": "이 플롯을 해석하는 방법 안내"
  }},
  "expression_boxplot": {{
    "title": "유전자 발현 분포 해석",
    "summary": "1-2문장 요약",
    "key_observations": [
      "관찰 1",
      "관찰 2"
    ],
    "interpretation_guide": "이 플롯을 해석하는 방법 안내"
  }}
}}
```

중요:
1. 한국어로 작성
2. 각 시각화의 특성에 맞는 구체적인 해석 제공
3. 생물학적/의학적 의미를 포함
4. 연구자가 플롯을 이해할 수 있도록 해석 가이드 포함
"""

        try:
            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                viz_interpretations = json.loads(json_str)

                # Save to file
                run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
                output_path = run_dir / "visualization_interpretations.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(viz_interpretations, f, ensure_ascii=False, indent=2)

                self.logger.info(f"Visualization interpretations generated: {output_path}")
                return viz_interpretations
            else:
                self.logger.warning("Could not extract JSON from Claude response")
                return None

        except Exception as e:
            self.logger.error(f"Error generating visualization interpretations: {e}")
            return None

    def validate_outputs(self) -> bool:
        """Validate report outputs."""
        report_path = self.output_dir / "report.html"
        if not report_path.exists():
            self.logger.error("Report HTML not generated")
            return False

        if report_path.stat().st_size < 1000:
            self.logger.error("Report HTML seems too small")
            return False

        return True
