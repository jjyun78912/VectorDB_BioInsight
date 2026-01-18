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

# LLM API for Extended Abstract generation
# Priority: OpenAI (gpt-4o-mini, cheaper) > Anthropic (Claude, backup)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

        # Load driver analysis data if exists
        driver_dirs = [
            self.input_dir / "driver_analysis",
            run_dir / "driver_analysis",
            run_dir / "agent6_report" / "driver_analysis"
        ]

        for driver_dir in driver_dirs:
            if driver_dir.exists():
                # Load known drivers
                known_path = driver_dir / "driver_known.csv"
                if known_path.exists():
                    try:
                        known_df = pd.read_csv(known_path)
                        data['driver_known'] = known_df.to_dict(orient='records')
                        self.logger.info(f"Loaded driver_known.csv: {len(known_df)} rows")
                    except Exception as e:
                        self.logger.warning(f"Error loading driver_known.csv: {e}")

                # Load candidate regulators (or novel drivers for backward compat)
                novel_path = driver_dir / "driver_candidate_regulators.csv"
                if not novel_path.exists():
                    novel_path = driver_dir / "driver_novel.csv"
                if novel_path.exists():
                    try:
                        novel_df = pd.read_csv(novel_path)
                        data['driver_novel'] = novel_df.to_dict(orient='records')
                        self.logger.info(f"Loaded {novel_path.name}: {len(novel_df)} rows")
                    except Exception as e:
                        self.logger.warning(f"Error loading candidate regulators: {e}")

                # Load driver summary
                summary_path = driver_dir / "driver_summary.json"
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            data['driver_summary'] = json.load(f)
                        self.logger.info("Loaded driver_summary.json")
                    except Exception as e:
                        self.logger.warning(f"Error loading driver_summary.json: {e}")

                break  # Found driver data, stop searching

        # Load cancer type prediction if exists
        cancer_pred_paths = [
            run_dir / "cancer_prediction.json",
            self.input_dir / "cancer_prediction.json",
            self.input_dir.parent / "cancer_prediction.json"
        ]

        for pred_path in cancer_pred_paths:
            if pred_path.exists():
                try:
                    with open(pred_path, 'r', encoding='utf-8') as f:
                        data['cancer_prediction'] = json.load(f)
                    self.logger.info(f"Loaded cancer_prediction.json: {data['cancer_prediction'].get('predicted_cancer', 'unknown')}")
                except Exception as e:
                    self.logger.warning(f"Error loading cancer_prediction.json: {e}")
                break

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
                    <button class="filter-btn" onclick="filterByConfidence('novel_candidate')">Candidate</button>
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

    def _generate_driver_analysis_html(self, data: Dict) -> str:
        """Generate Driver Gene Analysis section (Known + Candidate Regulator tracks)."""
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        driver_summary = data.get('driver_summary', {})

        if not driver_known and not driver_novel:
            return ""

        # Known drivers cards
        known_cards_html = ""
        for idx, driver in enumerate(driver_known[:10]):
            gene = driver.get('gene_symbol', 'Unknown')
            score = driver.get('score', 0)
            log2fc = driver.get('log2fc', 0)
            direction = "↑" if log2fc > 0 else "↓"
            dir_class = "up" if log2fc > 0 else "down"
            cosmic_tier = driver.get('cosmic_tier', '')
            cosmic_role = driver.get('cosmic_role', '')
            tcga_freq = driver.get('tcga_mutation_freq', 0) * 100
            tcga_count = driver.get('tcga_sample_count', 0)
            hotspots = driver.get('hotspots', [])
            val_method = driver.get('validation_method', '')
            val_detail = driver.get('validation_detail', '')
            is_hub = driver.get('is_hub', False)

            # Score badge color
            if score >= 70:
                score_class = "high"
                score_label = "High"
            elif score >= 50:
                score_class = "medium"
                score_label = "Medium"
            else:
                score_class = "low"
                score_label = "Low"

            hotspot_chips = ""
            if hotspots:
                hotspot_chips = "".join([f'<span class="hotspot-chip">{h}</span>' for h in hotspots[:3]])

            known_cards_html += f'''
            <div class="driver-card known">
                <div class="driver-header">
                    <div class="driver-title">
                        <span class="driver-rank">#{idx + 1}</span>
                        <span class="driver-gene">{gene}</span>
                        {'<span class="hub-badge">HUB</span>' if is_hub else ''}
                    </div>
                    <span class="driver-score {score_class}">{score:.0f}/100</span>
                </div>
                <div class="driver-body">
                    <div class="driver-evidence">
                        <div class="evidence-row">
                            <span class="evidence-label">Expression</span>
                            <span class="evidence-value {dir_class}">{direction} {abs(log2fc):.2f}</span>
                        </div>
                        <div class="evidence-row">
                            <span class="evidence-label">COSMIC</span>
                            <span class="evidence-value">{cosmic_tier} · {cosmic_role}</span>
                        </div>
                        <div class="evidence-row">
                            <span class="evidence-label">TCGA Freq</span>
                            <span class="evidence-value">{tcga_freq:.1f}% ({tcga_count} samples)</span>
                        </div>
                        {f'<div class="evidence-row"><span class="evidence-label">Hotspots</span><span class="evidence-value">{hotspot_chips}</span></div>' if hotspot_chips else ''}
                    </div>
                    <div class="driver-validation">
                        <span class="validation-icon">🧪</span>
                        <div class="validation-text">
                            <strong>{val_method}</strong>
                            <span>{val_detail}</span>
                        </div>
                    </div>
                </div>
            </div>
            '''

        # Candidate regulator cards
        novel_cards_html = ""
        for idx, driver in enumerate(driver_novel[:10]):
            gene = driver.get('gene_symbol', 'Unknown')
            score = driver.get('score', 0)
            log2fc = driver.get('log2fc', 0)
            direction = "↑" if log2fc > 0 else "↓"
            dir_class = "up" if log2fc > 0 else "down"
            hub_score = driver.get('hub_score', 0)
            pathway_impact = driver.get('pathway_impact', 0)
            val_method = driver.get('validation_method', '')
            val_detail = driver.get('validation_detail', '')
            is_hub = driver.get('is_hub', False)

            # Score badge color
            if score >= 70:
                score_class = "high"
            elif score >= 50:
                score_class = "medium"
            else:
                score_class = "low"

            novel_cards_html += f'''
            <div class="driver-card novel">
                <div class="driver-header">
                    <div class="driver-title">
                        <span class="driver-rank">#{idx + 1}</span>
                        <span class="driver-gene">{gene}</span>
                        <span class="novel-badge">NEW</span>
                        {'<span class="hub-badge">HUB</span>' if is_hub else ''}
                    </div>
                    <span class="driver-score {score_class}">{score:.0f}/100</span>
                </div>
                <div class="driver-body">
                    <div class="driver-evidence">
                        <div class="evidence-row">
                            <span class="evidence-label">Expression</span>
                            <span class="evidence-value {dir_class}">{direction} {abs(log2fc):.2f}</span>
                        </div>
                        <div class="evidence-row">
                            <span class="evidence-label">Hub Score</span>
                            <span class="evidence-value">{'●' * min(5, int(hub_score * 5))}{'○' * (5 - min(5, int(hub_score * 5)))}</span>
                        </div>
                        <div class="evidence-row">
                            <span class="evidence-label">Pathway Impact</span>
                            <span class="evidence-value">{'●' * min(5, int(pathway_impact * 5))}{'○' * (5 - min(5, int(pathway_impact * 5)))}</span>
                        </div>
                    </div>
                    <div class="driver-validation novel-validation">
                        <span class="validation-icon">🔬</span>
                        <div class="validation-text">
                            <strong>{val_method}</strong>
                            <span>{val_detail}</span>
                        </div>
                    </div>
                </div>
            </div>
            '''

        # Summary stats
        total_known = driver_summary.get('total_known_candidates', len(driver_known))
        total_novel = driver_summary.get('total_novel_candidates', len(driver_novel))
        high_conf_known = driver_summary.get('high_confidence_known', 0)
        high_conf_novel = driver_summary.get('high_confidence_novel', 0)
        actionable = driver_summary.get('actionable_targets', [])
        research = driver_summary.get('research_targets', [])

        return f'''
        <section class="driver-analysis" id="driver-analysis">
            <div class="driver-header-section">
                <h2>🎯 Driver Gene Analysis</h2>
                <p class="driver-subtitle">RNA-seq 발현 패턴 + TCGA 돌연변이 데이터 기반 Driver 예측</p>
            </div>

            <div class="driver-summary-stats">
                <div class="driver-stat known-stat">
                    <span class="stat-value">{total_known}</span>
                    <span class="stat-label">Known Drivers</span>
                    <span class="stat-detail">{high_conf_known} high confidence</span>
                </div>
                <div class="driver-stat novel-stat">
                    <span class="stat-value">{total_novel}</span>
                    <span class="stat-label">Candidate Regulators</span>
                    <span class="stat-detail">{high_conf_novel} high confidence</span>
                </div>
                <div class="driver-stat actionable-stat">
                    <span class="stat-value">{len(actionable)}</span>
                    <span class="stat-label">Actionable</span>
                    <span class="stat-detail">{', '.join(actionable[:3]) if actionable else 'None'}</span>
                </div>
            </div>

            <div class="driver-method-note">
                <span class="method-icon">📊</span>
                <div class="method-text">
                    <strong>Known Driver Track:</strong> COSMIC Cancer Gene Census + TCGA 돌연변이 빈도 + 발현 변화량 기반 scoring<br>
                    <strong>Candidate Regulator Track:</strong> Hub Gene 점수 + 발현 변화량 + Pathway 영향력 + 문헌 지지도 기반 scoring
                </div>
            </div>

            <div class="driver-tracks">
                <div class="driver-track known-track">
                    <h3>🏆 Known Driver Track</h3>
                    <p class="track-desc">COSMIC/OncoKB에서 검증된 암 드라이버 유전자. 타겟 치료제 개발 후보.</p>
                    <div class="driver-cards-grid">
                        {known_cards_html if known_cards_html else '<p class="no-data">No known drivers found in DEG list</p>'}
                    </div>
                </div>

                <div class="driver-track novel-track">
                    <h3>🔬 Candidate Regulator Track</h3>
                    <p class="track-desc">네트워크 분석 기반 핵심 조절인자 후보. 문헌 검증 및 기능 연구가 필요한 유전자.</p>
                    <div class="driver-cards-grid">
                        {novel_cards_html if novel_cards_html else '<p class="no-data">No candidate regulators found</p>'}
                    </div>
                </div>
            </div>

            <div class="driver-disclaimer">
                <span class="disclaimer-icon">⚠️</span>
                <div class="disclaimer-text">
                    <strong>주의사항:</strong> RNA-seq 데이터만으로는 Driver 유전자를 확정할 수 없습니다.
                    Candidate Regulator는 "확정된 driver"가 아닌 "추가 검증이 필요한 후보"입니다.
                    실제 돌연변이 확인을 위해서는 WES/WGS 또는 Targeted NGS가 필요합니다.
                </div>
            </div>
        </section>
        '''

    def _generate_study_overview_html(self, data: Dict) -> str:
        """Generate Study Overview section with ML-predicted cancer type."""
        deg_df = data.get('deg_significant_df')

        # Get cancer prediction from ML model
        cancer_prediction = data.get('cancer_prediction', {})

        # Determine cancer type display
        if cancer_prediction:
            predicted_cancer = cancer_prediction.get('predicted_cancer', 'Unknown')
            cancer_korean = cancer_prediction.get('cancer_korean', '')
            confidence = cancer_prediction.get('confidence', 0)
            agreement_ratio = cancer_prediction.get('agreement_ratio', 0)

            # Format cancer type display with Korean name and confidence
            if cancer_korean:
                cancer_type_display = f"{predicted_cancer} ({cancer_korean})"
            else:
                cancer_type_display = predicted_cancer

            # Confidence badge styling
            if confidence >= 0.8:
                confidence_badge = f'<span class="confidence-badge high">신뢰도: {confidence:.1%}</span>'
            elif confidence >= 0.6:
                confidence_badge = f'<span class="confidence-badge medium">신뢰도: {confidence:.1%}</span>'
            else:
                confidence_badge = f'<span class="confidence-badge low">신뢰도: {confidence:.1%}</span>'

            prediction_method = "🤖 ML 예측 (Pan-Cancer Classifier)"
            prediction_note = f"<small>샘플 일치율: {agreement_ratio:.1%}</small>"
        else:
            # Fallback to config-specified cancer type
            cancer_type_display = self.config.get('cancer_type_korean', self.config.get('cancer_type', 'Unknown'))
            if cancer_type_display.lower() == 'unknown':
                cancer_type_display = '암종 미확인'
                confidence_badge = '<span class="confidence-badge low">예측 불가</span>'
                prediction_method = "⚠️ ML 예측 실패"
                prediction_note = "<small>count matrix 확인 필요</small>"
            else:
                confidence_badge = '<span class="confidence-badge medium">사용자 지정</span>'
                prediction_method = "사용자 지정"
                prediction_note = ""

        # Get sample info from config
        original_files = self.config.get('original_files', {})
        dataset_id = original_files.get('count_matrix', 'Unknown Dataset')

        # Get counts
        total_deg = len(deg_df) if deg_df is not None else 0
        up_count = len(deg_df[deg_df['log2FC'] > 0]) if deg_df is not None and 'log2FC' in deg_df.columns else 0
        down_count = len(deg_df[deg_df['log2FC'] < 0]) if deg_df is not None and 'log2FC' in deg_df.columns else 0

        contrast = self.config.get('contrast', ['tumor', 'normal'])

        return f'''
        <section class="study-overview-section" id="study-overview">
            <h2>1. Study Overview</h2>

            <div class="overview-grid">
                <div class="overview-table">
                    <table class="info-table">
                        <tr><td><strong>Dataset ID</strong></td><td>{dataset_id}</td></tr>
                        <tr>
                            <td><strong>예측 암종 (Cancer Type)</strong></td>
                            <td>
                                <span class="cancer-type-predicted">{cancer_type_display}</span> {confidence_badge}
                                <br/>{prediction_note}
                            </td>
                        </tr>
                        <tr><td><strong>예측 방법</strong></td><td>{prediction_method}</td></tr>
                        <tr><td><strong>Comparison</strong></td><td>{contrast[0]} vs {contrast[1]}</td></tr>
                        <tr><td><strong>Analysis Pipeline</strong></td><td>BioInsight AI v2.0</td></tr>
                        <tr><td><strong>Analysis Date</strong></td><td>{datetime.now().strftime("%Y-%m-%d")}</td></tr>
                    </table>
                </div>

                <div class="deg-summary-box">
                    <h4>DEG Summary</h4>
                    <table class="info-table">
                        <tr><td>Total DEGs</td><td><strong>{total_deg:,}</strong></td></tr>
                        <tr><td>Upregulated</td><td class="up-text">{up_count:,}</td></tr>
                        <tr><td>Downregulated</td><td class="down-text">{down_count:,}</td></tr>
                        <tr><td>Cutoff (|log2FC|)</td><td>> 1.0</td></tr>
                        <tr><td>Cutoff (padj)</td><td>< 0.05</td></tr>
                    </table>
                </div>
            </div>
        </section>
        '''

    def _generate_qc_section_html(self, data: Dict) -> str:
        """Generate Data Quality Control section with PCA and correlation heatmap."""
        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})

        # Look for PCA and correlation heatmap
        pca_fig = figures.get('pca_plot', '')
        heatmap_fig = figures.get('sample_correlation', figures.get('heatmap', ''))

        pca_html = ''
        if pca_fig:
            pca_html = f'<img src="{pca_fig}" alt="PCA Plot" class="figure-img">'
        else:
            pca_html = '<p class="no-data">PCA plot not available</p>'

        heatmap_html = ''
        if heatmap_fig:
            heatmap_html = f'<img src="{heatmap_fig}" alt="Sample Correlation" class="figure-img">'
        else:
            heatmap_html = '<p class="no-data">Sample correlation heatmap not available</p>'

        return f'''
        <section class="qc-section" id="qc">
            <h2>2. Data Quality Control</h2>

            <div class="qc-grid">
                <div class="qc-panel">
                    <h4>2.1 Sample Clustering (PCA)</h4>
                    <div class="figure-container">
                        {pca_html}
                    </div>
                    <p class="figure-caption">Principal Component Analysis showing sample clustering between conditions.</p>
                </div>

                <div class="qc-panel">
                    <h4>2.2 Sample Correlation</h4>
                    <div class="figure-container">
                        {heatmap_html}
                    </div>
                    <p class="figure-caption">Correlation heatmap showing similarity between samples.</p>
                </div>
            </div>
        </section>
        '''

    def _generate_deg_analysis_html(self, data: Dict) -> str:
        """Generate Differential Expression Analysis section."""
        deg_df = data.get('deg_significant_df')
        figures = data.get('figures', {})

        # Get figures
        volcano_fig = figures.get('volcano_plot', '')
        heatmap_fig = figures.get('top_genes_heatmap', figures.get('heatmap', ''))

        volcano_html = f'<img src="{volcano_fig}" alt="Volcano Plot" class="figure-img">' if volcano_fig else '<p class="no-data">Volcano plot not available</p>'
        heatmap_html = f'<img src="{heatmap_fig}" alt="Heatmap" class="figure-img">' if heatmap_fig else '<p class="no-data">Heatmap not available</p>'

        # Top upregulated genes table
        up_table = ''
        down_table = ''

        if deg_df is not None and len(deg_df) > 0:
            # Sort by log2FC and get top 20 up/down
            if 'log2FC' in deg_df.columns:
                up_genes = deg_df[deg_df['log2FC'] > 0].nlargest(20, 'log2FC')
                down_genes = deg_df[deg_df['log2FC'] < 0].nsmallest(20, 'log2FC')

                # Build upregulated table
                up_rows = ''
                for _, row in up_genes.iterrows():
                    gene_id = row.get('gene_id', row.get('gene_symbol', 'N/A'))
                    log2fc = row.get('log2FC', 0)
                    padj = row.get('padj', 1)
                    up_rows += f'<tr><td>{gene_id}</td><td class="up-text">{log2fc:.2f}</td><td>{padj:.2e}</td></tr>'

                up_table = f'''
                <div class="deg-table-panel">
                    <h4>Top 20 Upregulated Genes</h4>
                    <table class="deg-table">
                        <thead><tr><th>Gene</th><th>log2FC</th><th>adj.p-value</th></tr></thead>
                        <tbody>{up_rows}</tbody>
                    </table>
                </div>
                '''

                # Build downregulated table
                down_rows = ''
                for _, row in down_genes.iterrows():
                    gene_id = row.get('gene_id', row.get('gene_symbol', 'N/A'))
                    log2fc = row.get('log2FC', 0)
                    padj = row.get('padj', 1)
                    down_rows += f'<tr><td>{gene_id}</td><td class="down-text">{log2fc:.2f}</td><td>{padj:.2e}</td></tr>'

                down_table = f'''
                <div class="deg-table-panel">
                    <h4>Top 20 Downregulated Genes</h4>
                    <table class="deg-table">
                        <thead><tr><th>Gene</th><th>log2FC</th><th>adj.p-value</th></tr></thead>
                        <tbody>{down_rows}</tbody>
                    </table>
                </div>
                '''

        return f'''
        <section class="deg-section" id="deg-analysis">
            <h2>3. Differential Expression Analysis</h2>

            <div class="deg-figures-grid">
                <div class="figure-panel">
                    <h4>3.1 Volcano Plot</h4>
                    <div class="figure-container">
                        {volcano_html}
                    </div>
                    <p class="figure-caption">Volcano plot showing differentially expressed genes. Red: upregulated, Blue: downregulated.</p>
                </div>

                <div class="figure-panel">
                    <h4>3.2 Top DEGs Heatmap</h4>
                    <div class="figure-container">
                        {heatmap_html}
                    </div>
                    <p class="figure-caption">Expression heatmap of top differentially expressed genes.</p>
                </div>
            </div>

            <div class="deg-tables-grid">
                {up_table}
                {down_table}
            </div>
        </section>
        '''

    def _generate_pathway_section_html(self, data: Dict) -> str:
        """Generate Pathway & Functional Analysis section with GO subcategories."""
        pathway_df = data.get('pathway_summary_df')
        figures = data.get('figures', {})

        pathway_fig = figures.get('pathway_enrichment', figures.get('go_enrichment', ''))
        pathway_html = f'<img src="{pathway_fig}" alt="Pathway Enrichment" class="figure-img">' if pathway_fig else ''

        # Separate pathways by category
        go_bp_rows = ''
        go_mf_rows = ''
        go_cc_rows = ''
        kegg_rows = ''

        if pathway_df is not None and len(pathway_df) > 0:
            for _, row in pathway_df.head(50).iterrows():
                term = row.get('term', row.get('Term', 'N/A'))
                gene_count = row.get('gene_count', row.get('Overlap', 'N/A'))
                pval = row.get('pvalue', row.get('P-value', row.get('Adjusted P-value', 1)))
                db = row.get('database', row.get('Gene_set', ''))

                if isinstance(gene_count, str) and '/' in gene_count:
                    gene_count = gene_count.split('/')[0]

                row_html = f'<tr><td>{term[:60]}{"..." if len(str(term)) > 60 else ""}</td><td>{gene_count}</td><td>{pval:.2e}</td></tr>'

                if 'Biological_Process' in str(db) or 'BP' in str(db):
                    go_bp_rows += row_html
                elif 'Molecular_Function' in str(db) or 'MF' in str(db):
                    go_mf_rows += row_html
                elif 'Cellular_Component' in str(db) or 'CC' in str(db):
                    go_cc_rows += row_html
                elif 'KEGG' in str(db):
                    kegg_rows += row_html
                else:
                    go_bp_rows += row_html  # Default to BP

        return f'''
        <section class="pathway-section" id="pathway-analysis">
            <h2>4. Pathway & Functional Analysis</h2>

            <div class="pathway-figure">
                {pathway_html}
            </div>

            <div class="pathway-subsections">
                <div class="pathway-panel">
                    <h4>4.1 GO Biological Process (BP)</h4>
                    <table class="pathway-table">
                        <thead><tr><th>Term</th><th>Genes</th><th>p-value</th></tr></thead>
                        <tbody>{go_bp_rows if go_bp_rows else "<tr><td colspan='3'>No significant BP terms</td></tr>"}</tbody>
                    </table>
                </div>

                <div class="pathway-panel">
                    <h4>4.2 GO Molecular Function (MF)</h4>
                    <table class="pathway-table">
                        <thead><tr><th>Term</th><th>Genes</th><th>p-value</th></tr></thead>
                        <tbody>{go_mf_rows if go_mf_rows else "<tr><td colspan='3'>No significant MF terms</td></tr>"}</tbody>
                    </table>
                </div>

                <div class="pathway-panel">
                    <h4>4.3 GO Cellular Component (CC)</h4>
                    <table class="pathway-table">
                        <thead><tr><th>Term</th><th>Genes</th><th>p-value</th></tr></thead>
                        <tbody>{go_cc_rows if go_cc_rows else "<tr><td colspan='3'>No significant CC terms</td></tr>"}</tbody>
                    </table>
                </div>

                <div class="pathway-panel">
                    <h4>4.4 KEGG Pathway</h4>
                    <table class="pathway-table">
                        <thead><tr><th>Pathway</th><th>Genes</th><th>p-value</th></tr></thead>
                        <tbody>{kegg_rows if kegg_rows else "<tr><td colspan='3'>No significant KEGG pathways</td></tr>"}</tbody>
                    </table>
                </div>
            </div>
        </section>
        '''

    def _generate_network_section_html(self, data: Dict) -> str:
        """Generate Network Analysis section."""
        hub_df = data.get('hub_genes_df')
        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})

        network_fig = figures.get('network_plot', figures.get('network_2d', ''))
        network_html = f'<img src="{network_fig}" alt="Network" class="figure-img">' if network_fig else ''

        # Hub genes table
        hub_table = ''
        if hub_df is not None and len(hub_df) > 0:
            hub_rows = ''
            for _, row in hub_df.head(20).iterrows():
                gene = row.get('gene_id', row.get('gene', 'N/A'))
                degree = row.get('degree', row.get('enhanced_score', 'N/A'))
                betweenness = row.get('betweenness', row.get('betweenness_centrality', 'N/A'))
                betweenness_str = f'{betweenness:.4f}' if isinstance(betweenness, (int, float)) else str(betweenness)
                hub_rows += f'<tr><td>{gene}</td><td>{degree}</td><td>{betweenness_str}</td></tr>'

            hub_table = f'''
            <table class="hub-table">
                <thead><tr><th>Gene</th><th>Degree/Score</th><th>Betweenness</th></tr></thead>
                <tbody>{hub_rows}</tbody>
            </table>
            '''

        return f'''
        <section class="network-section" id="network-analysis">
            <h2>6. Network Analysis</h2>

            <div class="network-grid">
                <div class="network-figure-panel">
                    <h4>6.1 Co-expression Network</h4>
                    <div class="figure-container">
                        {network_html if network_html else '<p class="no-data">Network visualization not available</p>'}
                    </div>
                    <p class="figure-caption">Gene co-expression network constructed from DEGs.</p>
                </div>

                <div class="hub-genes-panel">
                    <h4>6.2 Hub Genes</h4>
                    {hub_table if hub_table else '<p class="no-data">No hub genes identified</p>'}
                    <p class="panel-note">Hub genes are highly connected nodes in the network that may play key regulatory roles.</p>
                </div>
            </div>
        </section>
        '''

    def _generate_clinical_implications_html(self, data: Dict) -> str:
        """Generate Clinical Implications section."""
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        interpretation = data.get('interpretation_report', {})

        # Biomarker potential
        biomarkers = []
        therapeutic_targets = []

        for d in driver_known[:5]:
            gene = d.get('gene_symbol', d.get('gene', 'Unknown'))
            evidence = d.get('evidence_summary', '')
            biomarkers.append(f'<li><strong>{gene}</strong>: {evidence[:100]}...</li>')

        for d in driver_novel[:5]:
            gene = d.get('gene_symbol', d.get('gene', 'Unknown'))
            evidence = d.get('regulatory_evidence', d.get('evidence_summary', ''))
            therapeutic_targets.append(f'<li><strong>{gene}</strong>: {evidence[:100] if evidence else "Candidate regulatory gene"}...</li>')

        return f'''
        <section class="clinical-section" id="clinical-implications">
            <h2>8. Clinical Implications</h2>

            <div class="clinical-grid">
                <div class="clinical-panel">
                    <h4>8.1 Biomarker Potential</h4>
                    <ul class="clinical-list">
                        {chr(10).join(biomarkers) if biomarkers else '<li>No established biomarkers identified</li>'}
                    </ul>
                    <p class="panel-note">These genes show potential as diagnostic or prognostic biomarkers based on database validation.</p>
                </div>

                <div class="clinical-panel">
                    <h4>8.2 Therapeutic Targets</h4>
                    <ul class="clinical-list">
                        {chr(10).join(therapeutic_targets) if therapeutic_targets else '<li>No therapeutic targets identified</li>'}
                    </ul>
                    <p class="panel-note">Candidate regulatory genes that may serve as potential therapeutic targets pending validation.</p>
                </div>
            </div>

            <div class="disclaimer-box">
                <strong>⚠️ Important Note:</strong> All clinical implications are computational predictions and require
                experimental and clinical validation before any diagnostic or therapeutic application.
            </div>
        </section>
        '''

    def _generate_followup_experiments_html(self, data: Dict) -> str:
        """Generate Suggested Follow-up Experiments section."""
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        hub_df = data.get('hub_genes_df')

        # Get top genes to validate
        top_genes = []
        if driver_known:
            top_genes.extend([str(d.get('gene_symbol', d.get('gene', ''))) for d in driver_known[:3]])
        if driver_novel:
            top_genes.extend([str(d.get('gene_symbol', d.get('gene', ''))) for d in driver_novel[:3]])
        if hub_df is not None and len(hub_df) > 0:
            hub_genes = [str(g) for g in hub_df.head(3)['gene_id'].tolist()] if 'gene_id' in hub_df.columns else []
            top_genes.extend(hub_genes)

        top_genes = [str(g) for g in list(set(top_genes))[:5] if g]  # Unique top 5, ensure strings
        genes_str = ', '.join(top_genes) if top_genes else 'identified candidate genes'

        return f'''
        <section class="followup-section" id="followup-experiments">
            <h2>9. Suggested Follow-up Experiments</h2>

            <div class="experiment-grid">
                <div class="experiment-panel">
                    <h4>9.1 Validation Experiments</h4>
                    <ul>
                        <li><strong>qRT-PCR</strong>: Validate expression of {genes_str}</li>
                        <li><strong>Western Blot</strong>: Confirm protein-level changes</li>
                        <li><strong>IHC</strong>: Verify tissue localization</li>
                    </ul>
                </div>

                <div class="experiment-panel">
                    <h4>9.2 Functional Studies</h4>
                    <ul>
                        <li><strong>siRNA/shRNA Knockdown</strong>: Assess phenotypic changes</li>
                        <li><strong>CRISPR-Cas9 Knockout</strong>: Generate knockout models</li>
                        <li><strong>Overexpression</strong>: Study gain-of-function effects</li>
                        <li><strong>Proliferation Assay</strong>: MTT/CCK-8 assay</li>
                        <li><strong>Migration/Invasion</strong>: Transwell assay</li>
                    </ul>
                </div>

                <div class="experiment-panel">
                    <h4>9.3 In Vivo Studies</h4>
                    <ul>
                        <li><strong>Xenograft Model</strong>: Tumor growth in mice</li>
                        <li><strong>PDX Model</strong>: Patient-derived xenografts</li>
                        <li><strong>Drug Treatment</strong>: Therapeutic efficacy testing</li>
                    </ul>
                </div>
            </div>

            <div class="priority-note">
                <h5>Recommended Priority</h5>
                <p>Based on analysis results, we recommend prioritizing validation of: <strong>{genes_str}</strong></p>
            </div>
        </section>
        '''

    def _generate_research_recommendations_html(self, data: Dict) -> str:
        """Generate comprehensive Research Recommendations section."""
        recommendations = data.get('research_recommendations', {})

        if not recommendations:
            return '''
            <section class="research-recommendations-section" id="research-recommendations">
                <h2>9.5 후속 연구 추천 (Research Recommendations)</h2>
                <p class="no-data">연구 추천 데이터가 없습니다. LLM API를 통해 생성됩니다.</p>
            </section>
            '''

        # Extract sections
        therapeutic = recommendations.get('therapeutic_targets', {})
        drug_repurposing = recommendations.get('drug_repurposing', {})
        experimental = recommendations.get('experimental_validation', {})
        biomarker = recommendations.get('biomarker_development', {})
        future = recommendations.get('future_research_directions', {})
        collaboration = recommendations.get('collaboration_suggestions', {})
        funding = recommendations.get('funding_opportunities', {})
        cautions = recommendations.get('cautions_and_limitations', {})

        # Build HTML for therapeutic targets
        therapeutic_html = self._build_therapeutic_targets_html(therapeutic)

        # Build HTML for drug repurposing
        drug_html = self._build_drug_repurposing_html(drug_repurposing)

        # Build HTML for experimental validation
        experimental_html = self._build_experimental_validation_html(experimental)

        # Build HTML for biomarker development
        biomarker_html = self._build_biomarker_html(biomarker)

        # Build HTML for future research directions
        future_html = self._build_future_research_html(future)

        # Build HTML for collaboration and funding
        collab_funding_html = self._build_collab_funding_html(collaboration, funding)

        # Build HTML for cautions
        cautions_html = self._build_cautions_html(cautions)

        return f'''
        <section class="research-recommendations-section" id="research-recommendations">
            <h2>9.5 후속 연구 추천 (Research Recommendations)</h2>

            <div class="rec-intro">
                <p>본 섹션은 RNA-seq 분석 결과를 바탕으로 AI가 생성한 후속 연구 추천입니다.
                치료 타겟 후보, 약물 재목적화 가능성, 실험 검증 전략, 바이오마커 개발 방향을 제시합니다.</p>
            </div>

            {therapeutic_html}
            {drug_html}
            {experimental_html}
            {biomarker_html}
            {future_html}
            {collab_funding_html}
            {cautions_html}
        </section>
        '''

    def _build_therapeutic_targets_html(self, therapeutic: Dict) -> str:
        """Build HTML for therapeutic targets section."""
        if not therapeutic:
            return ''

        high_priority = therapeutic.get('high_priority', [])
        medium_priority = therapeutic.get('medium_priority', [])
        description = therapeutic.get('description', '')

        high_rows = ''
        for t in high_priority[:5]:
            gene = t.get('gene', 'N/A')
            rationale = t.get('rationale', 'N/A')
            drugs = ', '.join(t.get('existing_drugs', [])) or '-'
            target_class = t.get('target_class', 'N/A')
            high_rows += f'''
                <tr>
                    <td><strong>{gene}</strong></td>
                    <td>{rationale}</td>
                    <td>{drugs}</td>
                    <td>{target_class}</td>
                </tr>
            '''

        medium_rows = ''
        for t in medium_priority[:5]:
            gene = t.get('gene', 'N/A')
            rationale = t.get('rationale', 'N/A')
            drugs = ', '.join(t.get('existing_drugs', [])) or '-'
            target_class = t.get('target_class', 'N/A')
            medium_rows += f'''
                <tr>
                    <td>{gene}</td>
                    <td>{rationale}</td>
                    <td>{drugs}</td>
                    <td>{target_class}</td>
                </tr>
            '''

        return f'''
        <div class="rec-subsection">
            <h3>🎯 치료 타겟 후보 (Therapeutic Targets)</h3>
            <p class="rec-description">{description}</p>

            <h4>High Priority</h4>
            <table class="rec-table">
                <thead>
                    <tr><th>유전자</th><th>추천 근거</th><th>기존 약물</th><th>타겟 분류</th></tr>
                </thead>
                <tbody>{high_rows if high_rows else '<tr><td colspan="4">데이터 없음</td></tr>'}</tbody>
            </table>

            <h4>Medium Priority</h4>
            <table class="rec-table">
                <thead>
                    <tr><th>유전자</th><th>추천 근거</th><th>기존 약물</th><th>타겟 분류</th></tr>
                </thead>
                <tbody>{medium_rows if medium_rows else '<tr><td colspan="4">데이터 없음</td></tr>'}</tbody>
            </table>
        </div>
        '''

    def _build_drug_repurposing_html(self, drug_repurposing: Dict) -> str:
        """Build HTML for drug repurposing section."""
        if not drug_repurposing:
            return ''

        candidates = drug_repurposing.get('candidates', [])
        description = drug_repurposing.get('description', '')

        rows = ''
        for c in candidates[:5]:
            drug = c.get('drug', 'N/A')
            target = c.get('target_gene', 'N/A')
            original = c.get('original_indication', 'N/A')
            rationale = c.get('repurposing_rationale', 'N/A')
            status = c.get('clinical_status', 'N/A')
            rows += f'''
                <tr>
                    <td><strong>{drug}</strong></td>
                    <td>{target}</td>
                    <td>{original}</td>
                    <td>{rationale}</td>
                    <td>{status}</td>
                </tr>
            '''

        return f'''
        <div class="rec-subsection">
            <h3>💊 약물 재목적화 후보 (Drug Repurposing)</h3>
            <p class="rec-description">{description}</p>

            <table class="rec-table">
                <thead>
                    <tr><th>약물</th><th>타겟 유전자</th><th>기존 적응증</th><th>재목적화 근거</th><th>임상 상태</th></tr>
                </thead>
                <tbody>{rows if rows else '<tr><td colspan="5">데이터 없음</td></tr>'}</tbody>
            </table>
        </div>
        '''

    def _build_experimental_validation_html(self, experimental: Dict) -> str:
        """Build HTML for experimental validation section."""
        if not experimental:
            return ''

        description = experimental.get('description', '')
        immediate = experimental.get('immediate_validation', {})
        functional = experimental.get('functional_studies', {})
        clinical = experimental.get('clinical_validation', {})

        # Immediate validation
        qpcr = immediate.get('qPCR', {})
        wb = immediate.get('western_blot', {})
        qpcr_genes = ', '.join(qpcr.get('genes', [])) or 'N/A'
        qpcr_purpose = qpcr.get('purpose', '')
        wb_genes = ', '.join(wb.get('genes', [])) or 'N/A'
        wb_purpose = wb.get('purpose', '')

        # Functional studies
        knockdown = functional.get('knockdown_knockout', {})
        overexp = functional.get('overexpression', {})
        kd_genes = ', '.join(knockdown.get('genes', [])) or 'N/A'
        kd_method = knockdown.get('method', '')
        kd_readout = knockdown.get('readout', '')
        oe_genes = ', '.join(overexp.get('genes', [])) or 'N/A'
        oe_method = overexp.get('method', '')
        oe_readout = overexp.get('readout', '')

        # Clinical validation
        tissue = clinical.get('tissue_analysis', {})
        liquid = clinical.get('liquid_biopsy', {})
        tissue_genes = ', '.join(tissue.get('genes', [])) or 'N/A'
        tissue_method = tissue.get('method', '')
        liquid_biomarkers = ', '.join(liquid.get('biomarkers', [])) or 'N/A'
        liquid_method = liquid.get('method', '')

        return f'''
        <div class="rec-subsection">
            <h3>🔬 실험 검증 전략 (Experimental Validation)</h3>
            <p class="rec-description">{description}</p>

            <div class="validation-grid">
                <div class="validation-panel">
                    <h4>1차 검증 (Immediate)</h4>
                    <ul>
                        <li><strong>qPCR</strong>: {qpcr_genes}<br><em>{qpcr_purpose}</em></li>
                        <li><strong>Western Blot</strong>: {wb_genes}<br><em>{wb_purpose}</em></li>
                    </ul>
                </div>

                <div class="validation-panel">
                    <h4>기능 연구 (Functional)</h4>
                    <ul>
                        <li><strong>Knockdown/Knockout</strong>: {kd_genes}<br>방법: {kd_method}<br>측정: {kd_readout}</li>
                        <li><strong>Overexpression</strong>: {oe_genes}<br>방법: {oe_method}<br>측정: {oe_readout}</li>
                    </ul>
                </div>

                <div class="validation-panel">
                    <h4>임상 검증 (Clinical)</h4>
                    <ul>
                        <li><strong>조직 분석</strong>: {tissue_genes}<br>방법: {tissue_method}</li>
                        <li><strong>액체 생검</strong>: {liquid_biomarkers}<br>방법: {liquid_method}</li>
                    </ul>
                </div>
            </div>
        </div>
        '''

    def _build_biomarker_html(self, biomarker: Dict) -> str:
        """Build HTML for biomarker development section."""
        if not biomarker:
            return ''

        description = biomarker.get('description', '')
        diagnostic = biomarker.get('diagnostic_candidates', [])
        prognostic = biomarker.get('prognostic_candidates', [])

        diag_rows = ''
        for b in diagnostic[:5]:
            gene = b.get('gene', 'N/A')
            marker_type = b.get('marker_type', 'N/A')
            evidence = b.get('evidence_level', 'N/A')
            rationale = b.get('rationale', 'N/A')
            diag_rows += f'<tr><td><strong>{gene}</strong></td><td>{marker_type}</td><td>{evidence}</td><td>{rationale}</td></tr>'

        prog_rows = ''
        for b in prognostic[:5]:
            gene = b.get('gene', 'N/A')
            association = b.get('association', 'N/A')
            validation = b.get('validation_needed', 'N/A')
            prog_rows += f'<tr><td><strong>{gene}</strong></td><td>{association}</td><td>{validation}</td></tr>'

        return f'''
        <div class="rec-subsection">
            <h3>🧬 바이오마커 개발 (Biomarker Development)</h3>
            <p class="rec-description">{description}</p>

            <h4>진단 바이오마커 후보</h4>
            <table class="rec-table">
                <thead><tr><th>유전자</th><th>마커 유형</th><th>근거 수준</th><th>추천 근거</th></tr></thead>
                <tbody>{diag_rows if diag_rows else '<tr><td colspan="4">데이터 없음</td></tr>'}</tbody>
            </table>

            <h4>예후 바이오마커 후보</h4>
            <table class="rec-table">
                <thead><tr><th>유전자</th><th>예후 연관성</th><th>필요 검증</th></tr></thead>
                <tbody>{prog_rows if prog_rows else '<tr><td colspan="3">데이터 없음</td></tr>'}</tbody>
            </table>
        </div>
        '''

    def _build_future_research_html(self, future: Dict) -> str:
        """Build HTML for future research directions section."""
        if not future:
            return ''

        description = future.get('description', '')
        short_term = future.get('short_term', [])
        medium_term = future.get('medium_term', [])
        long_term = future.get('long_term', [])

        def build_timeline_items(items):
            html = ''
            for item in items[:3]:
                direction = item.get('direction', 'N/A')
                timeline = item.get('timeline', 'N/A')
                resources = item.get('resources_needed', 'N/A')
                outcome = item.get('expected_outcome', 'N/A')
                html += f'''
                    <div class="timeline-item">
                        <h5>{direction}</h5>
                        <p><strong>기간:</strong> {timeline}</p>
                        <p><strong>필요 자원:</strong> {resources}</p>
                        <p><strong>예상 결과:</strong> {outcome}</p>
                    </div>
                '''
            return html if html else '<p>데이터 없음</p>'

        return f'''
        <div class="rec-subsection">
            <h3>🔮 향후 연구 방향 (Future Research Directions)</h3>
            <p class="rec-description">{description}</p>

            <div class="timeline-grid">
                <div class="timeline-column">
                    <h4>단기 (6개월 이내)</h4>
                    {build_timeline_items(short_term)}
                </div>
                <div class="timeline-column">
                    <h4>중기 (1-2년)</h4>
                    {build_timeline_items(medium_term)}
                </div>
                <div class="timeline-column">
                    <h4>장기 (3-5년)</h4>
                    {build_timeline_items(long_term)}
                </div>
            </div>
        </div>
        '''

    def _build_collab_funding_html(self, collaboration: Dict, funding: Dict) -> str:
        """Build HTML for collaboration and funding section."""
        collab_desc = collaboration.get('description', '') if collaboration else ''
        expertise = collaboration.get('expertise_needed', []) if collaboration else []
        partnerships = collaboration.get('potential_partnerships', []) if collaboration else []

        funding_desc = funding.get('description', '') if funding else ''
        grant_types = funding.get('suitable_grant_types', []) if funding else []
        selling_points = funding.get('key_selling_points', []) if funding else []

        expertise_list = ''.join([f'<li>{e}</li>' for e in expertise]) or '<li>데이터 없음</li>'
        partnerships_list = ''.join([f'<li>{p}</li>' for p in partnerships]) or '<li>데이터 없음</li>'
        grants_list = ''.join([f'<li>{g}</li>' for g in grant_types]) or '<li>데이터 없음</li>'
        selling_list = ''.join([f'<li>{s}</li>' for s in selling_points]) or '<li>데이터 없음</li>'

        return f'''
        <div class="rec-subsection">
            <h3>🤝 협력 및 연구비 (Collaboration & Funding)</h3>

            <div class="collab-grid">
                <div class="collab-panel">
                    <h4>협력 연구 제안</h4>
                    <p>{collab_desc}</p>
                    <h5>필요 전문성</h5>
                    <ul>{expertise_list}</ul>
                    <h5>잠재적 협력 파트너</h5>
                    <ul>{partnerships_list}</ul>
                </div>

                <div class="collab-panel">
                    <h4>연구비 지원 기회</h4>
                    <p>{funding_desc}</p>
                    <h5>적합한 연구비 유형</h5>
                    <ul>{grants_list}</ul>
                    <h5>연구의 강점</h5>
                    <ul>{selling_list}</ul>
                </div>
            </div>
        </div>
        '''

    def _build_cautions_html(self, cautions: Dict) -> str:
        """Build HTML for cautions and limitations section."""
        if not cautions:
            return ''

        description = cautions.get('description', '')
        technical = cautions.get('technical_limitations', [])
        interpretation = cautions.get('interpretation_caveats', [])
        validation = cautions.get('validation_requirements', [])

        tech_list = ''.join([f'<li>{t}</li>' for t in technical]) or '<li>없음</li>'
        interp_list = ''.join([f'<li>{i}</li>' for i in interpretation]) or '<li>없음</li>'
        valid_list = ''.join([f'<li>{v}</li>' for v in validation]) or '<li>없음</li>'

        return f'''
        <div class="rec-subsection cautions-section">
            <h3>⚠️ 주의사항 및 한계점 (Cautions & Limitations)</h3>
            <p class="rec-description">{description}</p>

            <div class="cautions-grid">
                <div class="caution-panel">
                    <h4>기술적 한계</h4>
                    <ul>{tech_list}</ul>
                </div>
                <div class="caution-panel">
                    <h4>해석상 주의점</h4>
                    <ul>{interp_list}</ul>
                </div>
                <div class="caution-panel">
                    <h4>필수 검증 사항</h4>
                    <ul>{valid_list}</ul>
                </div>
            </div>
        </div>
        '''

    def _generate_methods_html(self) -> str:
        """Generate Level 4: Methods & Appendix."""
        return '''
        <section class="methods-section" id="methods">
            <h2>10. Methods & Parameters</h2>

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
        """Generate Claude AI-inspired modern CSS design."""
        # Load CSS from external file
        css_path = Path(__file__).parent.parent / "assets" / "claude_report_style.css"
        if css_path.exists():
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            return f'<style>\n{css_content}\n</style>'

        # Fallback to embedded CSS if file not found
        return '''
        <style>
            /* ========== CLAUDE AI MODERN STYLE (FALLBACK) ========== */

            :root {
                /* Claude AI Inspired Color Palette */
                --claude-bg: #faf9f7;
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

            /* Abstract Title */
            .abstract-title {
                margin-bottom: 24px;
                padding-bottom: 16px;
                border-bottom: 1px solid var(--gray-200);
            }

            .abstract-title h3 {
                font-size: 18px;
                font-weight: 600;
                color: var(--npj-blue);
                margin: 0 0 8px 0;
                line-height: 1.4;
            }

            .abstract-title .title-en {
                font-size: 14px;
                color: var(--gray-600);
                font-style: italic;
                margin: 0;
            }

            /* Driver Gene Interpretation */
            .driver-interpretation {
                margin-top: 20px;
                padding: 20px;
                background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
                border-radius: 8px;
                border: 1px solid #fca5a5;
            }

            .driver-interpretation h4 {
                color: #991b1b;
                font-size: 15px;
                margin-bottom: 12px;
            }

            .driver-interpretation p {
                font-size: 13px;
                color: var(--gray-700);
                line-height: 1.6;
                margin: 0;
            }

            /* RAG Literature Interpretation */
            .rag-interpretation {
                margin-top: 20px;
                padding: 20px;
                background: linear-gradient(135deg, #eff6ff 0%, #e0f2fe 100%);
                border-radius: 8px;
                border: 1px solid #93c5fd;
            }

            .rag-interpretation h4 {
                color: #1e40af;
                font-size: 15px;
                margin-bottom: 12px;
            }

            .rag-interpretation p {
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

            .cancer-type-predicted {
                font-size: 1.2em;
                font-weight: 700;
                color: var(--primary);
                margin-right: 8px;
            }

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

            /* ========== DRIVER ANALYSIS SECTION ========== */
            .driver-analysis {
                background: var(--gray-50);
                border-radius: 12px;
                padding: var(--spacing-xl);
                margin: var(--spacing-xl) 0;
            }

            .driver-header-section {
                text-align: center;
                margin-bottom: var(--spacing-xl);
            }

            .driver-header-section h2 {
                font-size: 24px;
                color: var(--npj-blue);
                margin-bottom: 8px;
            }

            .driver-subtitle {
                color: var(--gray-600);
                font-size: 14px;
            }

            .driver-summary-stats {
                display: flex;
                justify-content: center;
                gap: var(--spacing-xl);
                margin-bottom: var(--spacing-xl);
            }

            .driver-stat {
                text-align: center;
                background: white;
                padding: var(--spacing-lg);
                border-radius: 12px;
                min-width: 140px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }

            .driver-stat .stat-value {
                display: block;
                font-size: 32px;
                font-weight: 700;
            }

            .known-stat .stat-value { color: var(--npj-blue); }
            .novel-stat .stat-value { color: var(--npj-orange); }
            .actionable-stat .stat-value { color: var(--success); }

            .driver-stat .stat-label {
                display: block;
                font-size: 12px;
                color: var(--gray-600);
                margin-top: 4px;
            }

            .driver-stat .stat-detail {
                display: block;
                font-size: 11px;
                color: var(--gray-400);
                margin-top: 4px;
            }

            .driver-method-note {
                display: flex;
                gap: 12px;
                background: var(--npj-blue-light);
                padding: var(--spacing-md);
                border-radius: 8px;
                margin-bottom: var(--spacing-xl);
                font-size: 13px;
            }

            .driver-method-note .method-icon {
                font-size: 20px;
            }

            .driver-method-note .method-text {
                line-height: 1.6;
                color: var(--gray-700);
            }

            .driver-tracks {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: var(--spacing-xl);
            }

            .driver-track {
                background: white;
                border-radius: 12px;
                padding: var(--spacing-lg);
            }

            .known-track {
                border-left: 4px solid var(--npj-blue);
            }

            .novel-track {
                border-left: 4px solid var(--npj-orange);
            }

            .driver-track h3 {
                font-size: 18px;
                margin-bottom: 8px;
            }

            .track-desc {
                font-size: 13px;
                color: var(--gray-600);
                margin-bottom: var(--spacing-md);
            }

            .driver-cards-grid {
                display: flex;
                flex-direction: column;
                gap: var(--spacing-md);
            }

            .driver-card {
                background: var(--gray-50);
                border-radius: 8px;
                overflow: hidden;
            }

            .driver-card.known {
                border: 1px solid var(--npj-blue-light);
            }

            .driver-card.novel {
                border: 1px solid #fff3e0;
            }

            .driver-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                background: white;
            }

            .driver-title {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .driver-rank {
                font-size: 12px;
                color: var(--gray-400);
            }

            .driver-gene {
                font-size: 16px;
                font-weight: 600;
                color: var(--gray-900);
            }

            .hub-badge {
                background: var(--info);
                color: white;
                font-size: 9px;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: 600;
            }

            .novel-badge {
                background: var(--npj-orange);
                color: white;
                font-size: 9px;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: 600;
            }

            .driver-score {
                font-size: 14px;
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 4px;
            }

            .driver-score.high {
                background: #dcfce7;
                color: #166534;
            }

            .driver-score.medium {
                background: #fef3c7;
                color: #92400e;
            }

            .driver-score.low {
                background: #fef2f2;
                color: #991b1b;
            }

            .driver-body {
                padding: 12px 16px;
            }

            .driver-evidence {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin-bottom: 12px;
            }

            .evidence-row {
                display: flex;
                flex-direction: column;
                gap: 2px;
            }

            .evidence-label {
                font-size: 10px;
                color: var(--gray-500);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .evidence-value {
                font-size: 13px;
                color: var(--gray-800);
            }

            .evidence-value.up { color: #dc2626; }
            .evidence-value.down { color: #2563eb; }

            .hotspot-chip {
                display: inline-block;
                background: var(--gray-200);
                color: var(--gray-700);
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 3px;
                margin-right: 4px;
            }

            .driver-validation {
                display: flex;
                gap: 10px;
                padding: 10px;
                background: white;
                border-radius: 6px;
                border: 1px solid var(--gray-200);
            }

            .driver-validation .validation-icon {
                font-size: 18px;
            }

            .driver-validation .validation-text {
                font-size: 12px;
                line-height: 1.5;
            }

            .driver-validation .validation-text strong {
                display: block;
                color: var(--gray-800);
            }

            .driver-validation .validation-text span {
                color: var(--gray-600);
            }

            .novel-validation {
                background: #fffbeb;
                border-color: #fef3c7;
            }

            .driver-disclaimer {
                display: flex;
                gap: 12px;
                background: #fef2f2;
                padding: var(--spacing-md);
                border-radius: 8px;
                margin-top: var(--spacing-xl);
                border: 1px solid #fecaca;
            }

            .driver-disclaimer .disclaimer-icon {
                font-size: 20px;
            }

            .driver-disclaimer .disclaimer-text {
                font-size: 13px;
                line-height: 1.6;
                color: #991b1b;
            }

            /* ========== NEW TEMPLATE SECTIONS ========== */

            /* Study Overview Section */
            .study-overview-section {
                background: var(--gray-50);
                border-radius: 8px;
                padding: var(--spacing-xl);
                margin-bottom: var(--spacing-xl);
            }

            .study-overview-section h2 {
                margin-top: 0;
            }

            .overview-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: var(--spacing-lg);
            }

            .overview-card {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                padding: var(--spacing-lg);
            }

            .overview-card h4 {
                color: var(--npj-blue);
                font-size: 14px;
                font-weight: 600;
                margin-bottom: var(--spacing-md);
                padding-bottom: var(--spacing-sm);
                border-bottom: 1px solid var(--gray-200);
            }

            .overview-card table {
                width: 100%;
                font-size: 13px;
            }

            .overview-card td {
                padding: 6px 0;
                border-bottom: 1px solid var(--gray-100);
            }

            .overview-card td:first-child {
                font-weight: 500;
                color: var(--gray-600);
                width: 40%;
            }

            .overview-card td:last-child {
                color: var(--gray-900);
            }

            /* QC Section */
            .qc-section {
                margin-bottom: var(--spacing-xl);
            }

            .qc-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: var(--spacing-lg);
            }

            .qc-panel {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                padding: var(--spacing-lg);
            }

            .qc-panel h4 {
                font-size: 14px;
                font-weight: 600;
                margin-bottom: var(--spacing-md);
                color: var(--gray-800);
            }

            .qc-panel img {
                width: 100%;
                border-radius: 4px;
            }

            /* DEG Analysis Section */
            .deg-analysis-section {
                margin-bottom: var(--spacing-xl);
            }

            .deg-subsection {
                margin-bottom: var(--spacing-xl);
            }

            .deg-subsection h3 {
                font-size: 16px;
                font-weight: 600;
                color: var(--gray-800);
                margin-bottom: var(--spacing-md);
                padding-left: var(--spacing-md);
                border-left: 3px solid var(--npj-blue);
            }

            .deg-figure-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: var(--spacing-lg);
                margin-bottom: var(--spacing-lg);
            }

            .deg-figure {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                padding: var(--spacing-md);
            }

            .deg-figure.full-width {
                grid-column: 1 / -1;
            }

            .deg-figure img {
                width: 100%;
                border-radius: 4px;
            }

            .deg-tables-container {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: var(--spacing-lg);
            }

            .deg-table-panel {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                overflow: hidden;
            }

            .deg-table-panel h4 {
                font-size: 13px;
                font-weight: 600;
                padding: var(--spacing-md);
                margin: 0;
                background: var(--gray-50);
                border-bottom: 1px solid var(--gray-200);
            }

            .deg-table-panel h4.up-regulated {
                color: #c62828;
                border-left: 3px solid #c62828;
            }

            .deg-table-panel h4.down-regulated {
                color: var(--npj-blue);
                border-left: 3px solid var(--npj-blue);
            }

            .deg-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }

            .deg-table th {
                background: var(--gray-50);
                padding: 8px 12px;
                text-align: left;
                font-weight: 600;
                color: var(--gray-700);
                border-bottom: 1px solid var(--gray-200);
            }

            .deg-table td {
                padding: 8px 12px;
                border-bottom: 1px solid var(--gray-100);
            }

            .deg-table tr:hover {
                background: var(--npj-blue-light);
            }

            /* Pathway Section */
            .pathway-section {
                margin-bottom: var(--spacing-xl);
            }

            .pathway-category {
                margin-bottom: var(--spacing-xl);
            }

            .pathway-category h3 {
                font-size: 16px;
                font-weight: 600;
                color: var(--gray-800);
                margin-bottom: var(--spacing-md);
                padding-left: var(--spacing-md);
                border-left: 3px solid var(--success);
            }

            .pathway-table-container {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                overflow: hidden;
            }

            .pathway-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }

            .pathway-table th {
                background: var(--gray-50);
                padding: 10px 12px;
                text-align: left;
                font-weight: 600;
                color: var(--gray-700);
                border-bottom: 2px solid var(--gray-300);
            }

            .pathway-table td {
                padding: 10px 12px;
                border-bottom: 1px solid var(--gray-100);
            }

            .pathway-table tr:hover {
                background: var(--npj-blue-light);
            }

            .significance-bar {
                height: 8px;
                background: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
            }

            .significance-bar-fill {
                height: 100%;
                background: var(--success);
                border-radius: 4px;
            }

            /* Network Section */
            .network-section {
                margin-bottom: var(--spacing-xl);
            }

            .network-figure-container {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                padding: var(--spacing-lg);
                margin-bottom: var(--spacing-lg);
            }

            .hub-genes-table-container {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                overflow: hidden;
            }

            .hub-genes-table-container h4 {
                font-size: 14px;
                font-weight: 600;
                padding: var(--spacing-md);
                margin: 0;
                background: var(--gray-50);
                border-bottom: 1px solid var(--gray-200);
            }

            /* Clinical Implications Section */
            .clinical-implications-section {
                margin-bottom: var(--spacing-xl);
            }

            .clinical-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: var(--spacing-lg);
            }

            .clinical-card {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                padding: var(--spacing-lg);
            }

            .clinical-card h4 {
                font-size: 14px;
                font-weight: 600;
                color: var(--npj-blue);
                margin-bottom: var(--spacing-md);
                padding-bottom: var(--spacing-sm);
                border-bottom: 1px solid var(--gray-200);
            }

            .clinical-card h4.biomarkers::before {
                content: "🎯 ";
            }

            .clinical-card h4.therapeutic::before {
                content: "💊 ";
            }

            .clinical-item {
                display: flex;
                gap: var(--spacing-md);
                padding: var(--spacing-sm) 0;
                border-bottom: 1px solid var(--gray-100);
            }

            .clinical-item:last-child {
                border-bottom: none;
            }

            .clinical-gene {
                font-weight: 600;
                font-family: var(--font-mono);
                color: var(--gray-900);
                min-width: 80px;
            }

            .clinical-desc {
                font-size: 13px;
                color: var(--gray-600);
                line-height: 1.5;
            }

            /* Follow-up Experiments Section */
            .followup-section {
                margin-bottom: var(--spacing-xl);
            }

            .followup-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: var(--spacing-lg);
            }

            .followup-card {
                background: white;
                border: 1px solid var(--gray-200);
                border-radius: 6px;
                padding: var(--spacing-lg);
            }

            .followup-card h4 {
                font-size: 14px;
                font-weight: 600;
                color: var(--gray-800);
                margin-bottom: var(--spacing-md);
                padding-bottom: var(--spacing-sm);
                border-bottom: 1px solid var(--gray-200);
            }

            .followup-card h4::before {
                margin-right: 8px;
            }

            .followup-card.validation h4::before {
                content: "🧪";
            }

            .followup-card.functional h4::before {
                content: "🔬";
            }

            .followup-card.invivo h4::before {
                content: "🐭";
            }

            .followup-list {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .followup-list li {
                font-size: 13px;
                color: var(--gray-700);
                padding: 8px 0;
                padding-left: 20px;
                border-bottom: 1px solid var(--gray-100);
                position: relative;
            }

            .followup-list li::before {
                content: "•";
                position: absolute;
                left: 0;
                color: var(--npj-blue);
                font-weight: bold;
            }

            .followup-list li:last-child {
                border-bottom: none;
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
                .driver-tracks { grid-template-columns: 1fr; }
                .driver-summary-stats { flex-direction: column; }
                .driver-evidence { grid-template-columns: 1fr; }
                .overview-grid { grid-template-columns: 1fr; }
                .qc-grid { grid-template-columns: 1fr; }
                .deg-figure-grid { grid-template-columns: 1fr; }
                .deg-tables-container { grid-template-columns: 1fr; }
                .clinical-grid { grid-template-columns: 1fr; }
                .followup-grid { grid-template-columns: 1fr; }
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
            driver_interp = extended_abstract.get('driver_interpretation', '')
            rag_interp = extended_abstract.get('rag_interpretation', '')
            title = extended_abstract.get('title', '')
            title_en = extended_abstract.get('title_en', '')

            # Format abstract with paragraphs
            paragraphs = abstract_text.split('\n\n')
            formatted_paragraphs = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])

            # Title section
            title_html = ''
            if title or title_en:
                title_html = f'''
                <div class="abstract-title">
                    <h3>{title}</h3>
                    <p class="title-en">{title_en}</p>
                </div>
                '''

            # Key findings list
            findings_html = ''
            if key_findings:
                findings_html = '<div class="key-findings"><h4>📌 주요 발견</h4><ul>'
                for finding in key_findings[:8]:
                    findings_html += f'<li>{finding}</li>'
                findings_html += '</ul></div>'

            # Driver Gene interpretation
            driver_html = ''
            if driver_interp:
                driver_html = f'<div class="driver-interpretation"><h4>🧬 Driver Gene Analysis 해석</h4><p>{driver_interp}</p></div>'

            # RAG Literature interpretation
            rag_html = ''
            if rag_interp:
                rag_html = f'<div class="rag-interpretation"><h4>📚 문헌 기반 해석</h4><p>{rag_interp}</p></div>'

            # Validation priorities
            validation_html = ''
            if validation:
                validation_html = '<div class="validation-priorities"><h4>🔬 실험적 검증 제안</h4><div class="validation-grid">'
                if validation.get('qPCR'):
                    validation_html += f'<div class="validation-item"><strong>qRT-PCR:</strong> {", ".join(validation["qPCR"][:5])}</div>'
                if validation.get('western_blot'):
                    validation_html += f'<div class="validation-item"><strong>Western Blot:</strong> {", ".join(validation["western_blot"][:3])}</div>'
                if validation.get('functional_study'):
                    validation_html += f'<div class="validation-item"><strong>Functional Study:</strong> {", ".join(validation["functional_study"][:3])}</div>'
                if validation.get('targeted_sequencing'):
                    validation_html += f'<div class="validation-item"><strong>Targeted Sequencing:</strong> {", ".join(validation["targeted_sequencing"][:3])}</div>'
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
                {title_html}
                <div class="abstract-content">
                    {formatted_paragraphs}
                </div>
                {findings_html}
                {driver_html}
                {rag_html}
                {validation_html}
                {ml_html}
            </div>
        </section>
            '''

        # Fallback to comprehensive abstract (when Claude API unavailable)
        interpretation = data.get('interpretation_report', {})
        cancer_type = self.config.get('cancer_type', 'cancer')
        contrast = self.config.get('contrast', ['Tumor', 'Normal'])

        # DEG stats
        deg_df = data.get('deg_significant_df')
        deg_count = len(deg_df) if deg_df is not None else len(data.get('deg_significant', []))
        log2fc_col = 'log2FC' if deg_df is not None and 'log2FC' in deg_df.columns else 'log2FoldChange'
        n_up = len(deg_df[deg_df[log2fc_col] > 0]) if deg_df is not None and log2fc_col in deg_df.columns else 0
        n_down = deg_count - n_up

        # Hub genes - map Ensembl IDs to gene symbols using integrated_gene_table
        hub_df = data.get('hub_genes_df')
        integrated_df = data.get('integrated_gene_table_df')
        hub_genes = data.get('hub_genes', [])

        # Create gene_id to gene_symbol mapping from integrated_gene_table
        gene_id_to_symbol = {}
        if integrated_df is not None and len(integrated_df) > 0:
            if 'gene_id' in integrated_df.columns and 'gene_symbol' in integrated_df.columns:
                for _, row in integrated_df.iterrows():
                    gene_id = str(row.get('gene_id', ''))
                    gene_symbol = str(row.get('gene_symbol', ''))
                    if gene_id and gene_symbol and gene_symbol != 'nan':
                        gene_id_to_symbol[gene_id] = gene_symbol

        hub_names = []
        if hub_df is not None and len(hub_df) > 0:
            for _, row in hub_df.head(5).iterrows():
                gene_id = str(row.get('gene_id', ''))
                # Try to get symbol from mapping, fallback to gene_id
                gene_symbol = gene_id_to_symbol.get(gene_id, '')
                if gene_symbol and not gene_symbol.startswith('ENSG'):
                    hub_names.append(gene_symbol)
                elif gene_id:
                    # If no symbol found, use gene_id but try to clean it
                    hub_names.append(gene_id.split('.')[0] if '.' in gene_id else gene_id)
        else:
            hub_names = [str(g.get('gene_symbol', g.get('gene_id', ''))) for g in hub_genes[:5]]

        # Pathways
        pathway_df = data.get('pathway_summary_df')
        pathway_names = []
        if pathway_df is not None and len(pathway_df) > 0:
            # Check for different column names
            term_col = None
            for col in ['term_name', 'Term', 'term', 'pathway']:
                if col in pathway_df.columns:
                    term_col = col
                    break
            if term_col:
                pathway_names = pathway_df[term_col].head(3).tolist()
        else:
            pathways = data.get('pathway_summary', [])[:3]
            pathway_names = [p.get('term_name', p.get('Term', ''))[:50] for p in pathways]

        # Clean pathway names (remove GO IDs for readability)
        pathway_names = [name.split(' (GO:')[0] if ' (GO:' in str(name) else str(name) for name in pathway_names]

        # Driver info
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        known_count = len(driver_known) if driver_known else 0
        novel_count = len(driver_novel) if driver_novel else 0
        known_names = [d.get('gene_symbol', '') for d in driver_known[:3]] if driver_known else []
        novel_names = [d.get('gene_symbol', '') for d in driver_novel[:3]] if driver_novel else []

        # DB matched genes
        db_matched_df = data.get('db_matched_genes_df')
        db_count = len(db_matched_df) if db_matched_df is not None else 0

        # RAG interpretation
        rag_genes_count = 0
        rag_path = self.input_dir / "rag_interpretations.json"
        if rag_path.exists():
            try:
                import json
                with open(rag_path, 'r') as f:
                    rag_genes_count = json.load(f).get('genes_interpreted', 0)
            except:
                pass

        # Build comprehensive abstract sections
        background = f"""본 연구는 {cancer_type.replace('_', ' ').title()} 환자의 RNA-seq 데이터를 이용하여
{contrast[0]} 대비 {contrast[1]} 그룹 간의 유전자 발현 차이를 분석하고,
잠재적 Driver 유전자 및 치료 타겟을 발굴하고자 수행되었습니다."""

        methods = f"""차등발현 분석은 DESeq2를 이용하였으며 (|log2FC| > 1, padj < 0.05),
상관관계 기반 네트워크 분석으로 Hub 유전자를 도출하였습니다.
GO/KEGG pathway enrichment 분석(Enrichr)과
COSMIC/OncoKB/IntOGen 데이터베이스 검증을 수행하였습니다.
Driver 유전자 예측은 Two-Track 시스템(Known Driver + Candidate Regulator)을 적용하였습니다."""

        results_deg = f"""총 {deg_count:,}개의 DEGs를 식별하였으며 (상향조절 {n_up:,}개, 하향조절 {n_down:,}개)"""
        results_hub = f"""네트워크 분석 결과 {len(hub_names) if hub_names else 0}개의 Hub 유전자({', '.join(hub_names[:3]) if hub_names else 'N/A'} 등)가 확인되었습니다."""
        results_pathway = f"""Pathway 분석에서 {', '.join(pathway_names[:2]) if pathway_names else 'N/A'} 등이 유의하게 농축되었습니다."""

        results_driver = ""
        if known_count > 0 or novel_count > 0:
            results_driver = f"""Driver 분석 결과, Known Driver 후보 {known_count}개({', '.join(known_names) if known_names else 'N/A'} 등)와
Candidate Regulator 후보 {novel_count}개({', '.join(novel_names) if novel_names else 'N/A'} 등)를 도출하였습니다."""

        results_db = ""
        if db_count > 0:
            results_db = f"""COSMIC/OncoKB 데이터베이스에서 {db_count}개의 알려진 암 유전자가 매칭되었습니다."""

        results_rag = ""
        if rag_genes_count > 0:
            results_rag = f"""문헌 기반 RAG 해석을 통해 {rag_genes_count}개 핵심 유전자의 암종 특이적 역할을 분석하였습니다."""

        conclusions = f"""본 분석에서 확인된 Hub 유전자와 Driver 후보는
{cancer_type.replace('_', ' ').title()}의 바이오마커 및 치료 타겟 개발에 유망한 후보입니다.
특히 Known Driver 유전자들은 Targeted NGS를 통해,
Candidate Regulator 후보들은 문헌 검토 후 기능적 검증 실험을 통해 추가 검증이 권장됩니다."""

        # Build key findings
        key_findings = []
        if deg_count > 0:
            key_findings.append(f"총 {deg_count:,}개 DEGs 식별 (상향 {n_up:,}개, 하향 {n_down:,}개)")
        if hub_names:
            key_findings.append(f"핵심 Hub 유전자: {', '.join(hub_names[:3])}")
        if pathway_names:
            key_findings.append(f"주요 Pathway: {pathway_names[0][:40] if pathway_names else 'N/A'}")
        if known_count > 0:
            key_findings.append(f"Known Driver 후보 {known_count}개 (COSMIC/OncoKB/IntOGen 검증)")
        if novel_count > 0:
            key_findings.append(f"Candidate Regulator 후보 {novel_count}개 (문헌 검토 필요)")
        if db_count > 0:
            key_findings.append(f"암 유전자 DB 매칭 {db_count}개")

        findings_html = ''
        if key_findings:
            findings_html = '<div class="key-findings"><h4>📌 주요 발견</h4><ul>'
            for finding in key_findings[:8]:
                findings_html += f'<li>{finding}</li>'
            findings_html += '</ul></div>'

        # Driver interpretation
        driver_html = ''
        if known_count > 0 or novel_count > 0:
            driver_interp = f"""Known Driver Track에서 {known_count}개의 후보가 COSMIC, OncoKB, IntOGen 데이터베이스에서 검증되었습니다.
이들은 기존에 알려진 암 유전자로서 Targeted NGS 패널을 통한 변이 확인이 권장됩니다.
Candidate Regulator Track에서는 {novel_count}개의 조절인자 후보가 Hub gene 특성과 발현 패턴 분석을 통해 도출되었으며,
이들은 "확정된 driver"가 아닌 "추가 검증이 필요한 후보"로, 문헌 검토 후 기능적 검증 실험이 필요합니다."""
            driver_html = f'<div class="driver-interpretation"><h4>🧬 Driver Gene Analysis 해석</h4><p>{driver_interp}</p></div>'

        # Validation suggestions
        validation_html = '<div class="validation-priorities"><h4>🔬 실험적 검증 제안</h4><div class="validation-grid">'
        if hub_names:
            validation_html += f'<div class="validation-item"><strong>qRT-PCR:</strong> {", ".join(hub_names[:5])}</div>'
        if known_names:
            validation_html += f'<div class="validation-item"><strong>Targeted Sequencing:</strong> {", ".join(known_names[:3])}</div>'
        if novel_names:
            validation_html += f'<div class="validation-item"><strong>Functional Study:</strong> {", ".join(novel_names[:3])}</div>'
        validation_html += '</div></div>'

        return f'''
        <section class="abstract-section" id="abstract">
            <h2>Extended Abstract</h2>
            <div class="abstract-box extended">
                <div class="abstract-content">
                    <p><strong>배경:</strong> {background.strip()}</p>
                    <p><strong>방법:</strong> {methods.strip()}</p>
                    <p><strong>결과:</strong> {results_deg} {results_hub} {results_pathway} {results_driver} {results_db} {results_rag}</p>
                    <p><strong>결론:</strong> {conclusions.strip()}</p>
                </div>
                {findings_html}
                {driver_html}
                {validation_html}
                <div class="abstract-keywords">
                    <strong>Keywords:</strong> RNA-seq, Differential Expression, Network Analysis, Driver Gene, {cancer_type.replace('_', ' ').title()}, {", ".join(hub_names[:3]) if hub_names else ""}
                </div>
            </div>
        </section>
        '''

    def _generate_html(self, data: Dict) -> str:
        """Generate complete HTML report following the new template structure.

        Sections:
        1. Study Overview
        2. Data Quality Control (QC)
        3. Differential Expression Analysis
        4. Pathway & Functional Analysis
        5. Driver Gene Analysis
        6. Network Analysis
        7. (Survival Analysis - Optional, not implemented yet)
        8. Clinical Implications
        9. Suggested Follow-up Experiments
        10. Methods Summary
        11. References (via RAG)
        12. Appendix (Supplementary Data)
        """
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
                <a href="#study-overview">Overview</a>
                <a href="#qc-section">QC</a>
                <a href="#deg-analysis">DEG</a>
                <a href="#pathway-section">Pathway</a>
                <a href="#driver-analysis">Driver</a>
                <a href="#network-section">Network</a>
                <a href="#clinical-implications">Clinical</a>
                <a href="#research-recommendations">Research</a>
                <a href="#methods">Methods</a>
            </div>
        </div>
    </nav>

    <main class="paper-content">
        <!-- 1. Study Overview -->
        {self._generate_study_overview_html(data)}

        <!-- 2. Data Quality Control -->
        {self._generate_qc_section_html(data)}

        <!-- 3. Differential Expression Analysis -->
        {self._generate_deg_analysis_html(data)}

        <!-- 4. Pathway & Functional Analysis -->
        {self._generate_pathway_section_html(data)}

        <!-- 5. Driver Gene Analysis -->
        {self._generate_driver_analysis_html(data)}

        <!-- 6. Network Analysis -->
        {self._generate_network_section_html(data)}

        <!-- 8. Clinical Implications -->
        {self._generate_clinical_implications_html(data)}

        <!-- 9. Suggested Follow-up Experiments -->
        {self._generate_followup_experiments_html(data)}

        <!-- 9.5 Research Recommendations -->
        {self._generate_research_recommendations_html(data)}

        <!-- 10. Methods Summary -->
        {self._generate_methods_html() if self.config["include_methods"] else ""}

        <!-- 11. References (Literature via RAG) -->
        {self._generate_rag_summary_html(data)}

        <!-- 12. Appendix / Supplementary Data -->
        <section class="data-section" id="detailed-table">
            <h2>12. Appendix: Supplementary Data</h2>
            {self._generate_detailed_table_html(data)}
        </section>
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

    def _run_driver_prediction(self, data: Dict[str, Any]) -> None:
        """Run Driver Gene Prediction and add results to data dict."""
        try:
            from ..ml.driver_predictor import DriverPredictor
        except ImportError:
            self.logger.warning("Driver predictor module not available")
            return

        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        integrated_df = data.get('integrated_gene_table_df')

        if deg_df is None or len(deg_df) == 0:
            self.logger.warning("No DEG data available for driver prediction")
            return

        # Get cancer type from config or interpretation
        interpretation = data.get('interpretation_report', {})
        cancer_type = interpretation.get('cancer_type', self.config.get('cancer_type', 'unknown'))

        self.logger.info(f"Running Driver Gene Prediction for {cancer_type}...")

        try:
            predictor = DriverPredictor(cancer_type)
            results = predictor.predict(deg_df, hub_df, integrated_df)

            # Convert DriverCandidate objects to dicts
            data['driver_known'] = [d.to_dict() for d in results.get('known_drivers', [])]
            data['driver_novel'] = [d.to_dict() for d in results.get('candidate_regulators', results.get('novel_drivers', []))]
            data['driver_summary'] = results.get('summary', {})

            # Save results to files
            output_dir = self.output_dir / "driver_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            predictor.save_results(output_dir)

            self.logger.info(f"Driver prediction complete: {len(data['driver_known'])} known, {len(data['driver_novel'])} candidate regulators")

        except Exception as e:
            self.logger.warning(f"Driver prediction failed: {e}")
            import traceback
            traceback.print_exc()

    def run(self) -> Dict[str, Any]:
        """Generate the HTML report."""
        data = self._load_all_data()

        # Generate extended abstract if not already present
        if 'abstract_extended' not in data:
            self.logger.info("Generating extended abstract with Claude API...")
            extended_abstract = self._generate_extended_abstract(data)
            if extended_abstract:
                data['abstract_extended'] = extended_abstract

        # Run Driver Gene Prediction
        self._run_driver_prediction(data)

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

        # Generate research recommendations
        run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
        research_rec_path = run_dir / "research_recommendations.json"
        if research_rec_path.exists():
            try:
                with open(research_rec_path, 'r', encoding='utf-8') as f:
                    data['research_recommendations'] = json.load(f)
                self.logger.info("Loaded existing research recommendations")
            except Exception as e:
                self.logger.warning(f"Error loading research recommendations: {e}")
        else:
            self.logger.info("Generating research recommendations with LLM API...")
            research_recommendations = self._generate_research_recommendations(data)
            if research_recommendations:
                data['research_recommendations'] = research_recommendations

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

        Creates a comprehensive abstract that summarizes ALL report sections:
        - DEG Analysis (Volcano, Heatmap)
        - Network Analysis (Hub genes, PPI)
        - Pathway Enrichment (GO/KEGG)
        - Driver Gene Analysis (Known/Candidate Regulator)
        - Database Validation (COSMIC, OncoKB, IntOGen)
        - ML Prediction (if available)
        - RAG Literature Interpretation
        - Validation Recommendations
        """
        # Try OpenAI first (cheaper: gpt-4o-mini), then Anthropic as fallback
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        use_openai = OPENAI_AVAILABLE and openai_key
        use_anthropic = ANTHROPIC_AVAILABLE and anthropic_key and not use_openai

        if not use_openai and not use_anthropic:
            self.logger.warning("No LLM API available (need OPENAI_API_KEY or ANTHROPIC_API_KEY)")
            return None

        llm_provider = "OpenAI" if use_openai else "Anthropic"
        self.logger.info(f"Using {llm_provider} for extended abstract generation")

        # Prepare analysis summary
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

        # Top DEGs by fold change
        top_up_genes = []
        top_down_genes = []
        if deg_df is not None and len(deg_df) > 0:
            gene_col = 'gene_symbol' if 'gene_symbol' in deg_df.columns else 'gene_id'
            sorted_df = deg_df.sort_values(log2fc_col, ascending=False)
            for _, row in sorted_df.head(5).iterrows():
                gene = str(row.get(gene_col, 'Unknown'))
                fc = row.get(log2fc_col, 0)
                if not gene.startswith('ENSG'):
                    top_up_genes.append(f"{gene} (log2FC={fc:.2f})")
            for _, row in sorted_df.tail(5).iterrows():
                gene = str(row.get(gene_col, 'Unknown'))
                fc = row.get(log2fc_col, 0)
                if not gene.startswith('ENSG'):
                    top_down_genes.append(f"{gene} (log2FC={fc:.2f})")

        # Hub genes info - handle both 'gene_id' and 'gene_symbol' column names
        hub_genes_info = []
        hub_gene_names = []
        if hub_df is not None and len(hub_df) > 0:
            hub_log2fc_col = 'log2FC' if 'log2FC' in hub_df.columns else 'log2FoldChange'
            for _, row in hub_df.head(10).iterrows():
                gene_name = str(row.get('gene_id', row.get('gene_symbol', row.get('gene_name', 'Unknown'))))
                degree = row.get('degree', 0)
                log2fc = row.get(hub_log2fc_col, 0)
                hub_genes_info.append(f"- {gene_name} (degree={degree}, log2FC={log2fc:.2f})")
                hub_gene_names.append(gene_name)

        # Pathway info - categorize by GO/KEGG
        pathway_info = []
        go_terms = []
        kegg_terms = []
        if pathway_df is not None and len(pathway_df) > 0:
            for _, row in pathway_df.head(10).iterrows():
                term = row.get('Term', row.get('term', 'Unknown'))
                pval = row.get('P-value', row.get('pvalue', 0))
                gene_count = row.get('Overlap', row.get('gene_count', ''))
                pathway_info.append(f"- {term} (p={pval:.2e}, genes={gene_count})")
                if 'GO:' in str(term) or 'biological_process' in str(row.get('Gene_set', '')).lower():
                    go_terms.append(term)
                else:
                    kegg_terms.append(term)

        # Driver Gene Analysis info
        run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
        driver_info = ""
        known_drivers = []
        candidate_regulators = []

        # Check agent6_report folder first
        driver_dir = run_dir / "agent6_report" / "driver_analysis"
        if not driver_dir.exists():
            driver_dir = self.output_dir / "driver_analysis"

        if driver_dir.exists():
            # Load known drivers
            known_path = driver_dir / "driver_known.csv"
            if known_path.exists():
                try:
                    known_df = pd.read_csv(known_path)
                    for _, row in known_df.head(10).iterrows():
                        gene = row.get('gene_symbol', '')
                        score = row.get('score', 0)
                        tier = row.get('cosmic_tier', '')
                        role = row.get('cosmic_role', '')
                        direction = row.get('direction', '')
                        known_drivers.append(f"- {gene} (score={score:.1f}, {tier}, {role}, {direction})")
                except Exception as e:
                    self.logger.warning(f"Error loading known drivers: {e}")

            # Load candidate regulators
            novel_path = driver_dir / "driver_candidate_regulators.csv"
            if not novel_path.exists():
                novel_path = driver_dir / "driver_novel.csv"  # fallback to old name
            if novel_path.exists():
                try:
                    novel_df = pd.read_csv(novel_path)
                    for _, row in novel_df.head(10).iterrows():
                        gene = row.get('gene_symbol', '')
                        score = row.get('score', 0)
                        hub_score = row.get('hub_score', 0)
                        direction = row.get('direction', '')
                        lit_support = row.get('literature_support', 'unknown')
                        candidate_regulators.append(f"- {gene} (score={score:.1f}, hub={hub_score:.2f}, {direction}, lit={lit_support})")
                except Exception as e:
                    self.logger.warning(f"Error loading candidate regulators: {e}")

            # Load summary
            summary_path = driver_dir / "driver_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        driver_summary = json.load(f)
                    lit_breakdown = driver_summary.get('literature_support_breakdown', {})
                    driver_info = f"""
## Driver Gene Analysis 결과
- Known Driver 후보: {driver_summary.get('total_known_candidates', 0)}개
- Candidate Regulator 후보: {driver_summary.get('total_candidate_regulators', driver_summary.get('total_novel_candidates', 0))}개
- High Confidence Known: {driver_summary.get('high_confidence_known', 0)}개
- High Confidence Regulators: {driver_summary.get('high_confidence_regulators', driver_summary.get('high_confidence_novel', 0))}개
- Literature Support: emerging={lit_breakdown.get('emerging', 0)}, uncharacterized={lit_breakdown.get('uncharacterized', 0)}
- 연구 타겟 추천: {', '.join(driver_summary.get('research_targets', [])[:5])}

### Top Known Drivers (COSMIC/OncoKB/IntOGen 검증됨)
{chr(10).join(known_drivers[:5]) if known_drivers else '없음'}

### Top Candidate Regulators (문헌 검토 + 기능 검증 필요)
{chr(10).join(candidate_regulators[:5]) if candidate_regulators else '없음'}
"""
                except Exception as e:
                    self.logger.warning(f"Error loading driver summary: {e}")

        # Database validation info
        db_matched_df = data.get('db_matched_genes_df')
        db_info = ""
        if db_matched_df is not None and len(db_matched_df) > 0:
            db_genes = []
            for _, row in db_matched_df.head(10).iterrows():
                gene = row.get('gene_symbol', '')
                sources = row.get('db_sources', '')
                cancer_match = row.get('cancer_type_match', False)
                db_genes.append(f"- {gene} ({sources}, cancer_specific={cancer_match})")
            db_info = f"""
## 데이터베이스 검증 결과
- COSMIC/OncoKB 매칭 유전자: {len(db_matched_df)}개
{chr(10).join(db_genes[:5])}
"""

        # ML prediction info (check for prediction files)
        ml_info = ""
        ml_prediction_path = run_dir / "ml_prediction" / "prediction_summary.json"
        if ml_prediction_path.exists():
            try:
                with open(ml_prediction_path, 'r') as f:
                    ml_data = json.load(f)
                ml_info = f"""
## ML 예측 결과
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
                interpretation_samples = []

                for gene, gene_data in interpretations.items():
                    interp = gene_data.get('interpretation', '')
                    pmids = gene_data.get('pmids', [])
                    log2fc = gene_data.get('log2fc', 0)
                    direction = gene_data.get('direction', '')

                    all_pmids.update(pmids)

                    # Store sample interpretations for abstract
                    if len(interpretation_samples) < 3 and interp:
                        interpretation_samples.append({
                            'gene': gene, 'interpretation': interp[:300], 'pmids': pmids[:2]
                        })

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
                rag_summary["interpretation_samples"] = interpretation_samples

                rag_info = f"""
## RAG 기반 문헌 해석 결과
- 분석된 유전자 수: {rag_summary['genes_analyzed']}개
- 참조된 PMID 수: {len(all_pmids)}개
- 문헌 지원 유전자: {', '.join([g['gene'] for g in literature_supported[:5]]) if literature_supported else '없음'}
- 신규 바이오마커 후보 (기존 문헌 미기재): {', '.join(novel_candidates[:5]) if novel_candidates else '없음'}

### 주요 유전자 해석 샘플
{chr(10).join([f"- {s['gene']}: {s['interpretation'][:150]}... (PMID: {', '.join(s['pmids'][:2])})" for s in interpretation_samples[:3]]) if interpretation_samples else '없음'}
"""
                self.logger.info(f"Loaded RAG interpretations: {rag_summary['genes_analyzed']} genes")
            except Exception as e:
                self.logger.warning(f"Error loading RAG interpretations: {e}")

        # Figures generated
        figures_info = ""
        figures = data.get('figures', [])
        interactive_figures = data.get('interactive_figures', [])
        if figures or interactive_figures:
            figure_types = []
            for fig in figures:
                if 'volcano' in fig.lower():
                    figure_types.append('Volcano Plot (DEG 분포)')
                elif 'heatmap' in fig.lower():
                    figure_types.append('Heatmap (발현 패턴)')
                elif 'pca' in fig.lower():
                    figure_types.append('PCA Plot (샘플 분리)')
                elif 'network' in fig.lower():
                    figure_types.append('Network Graph (유전자 상호작용)')
                elif 'pathway' in fig.lower():
                    figure_types.append('Pathway Bar Plot (기능 분석)')
                elif 'boxplot' in fig.lower():
                    figure_types.append('Expression Boxplot (발현 비교)')

            figures_info = f"""
## 생성된 시각화
- 정적 Figure: {len(figures)}개
- 인터랙티브 Figure: {len(interactive_figures)}개
- Figure 종류: {', '.join(figure_types[:6])}
"""

        # Study info from config
        study_name = self.config.get('report_title', self.config.get('study_name', 'RNA-seq Analysis'))
        cancer_type = self.config.get('cancer_type', 'cancer')
        contrast = self.config.get('contrast', ['Tumor', 'Normal'])

        # Build comprehensive prompt
        prompt = f"""당신은 바이오인포매틱스 전문가입니다. 아래 RNA-seq 분석 결과를 바탕으로 학술 논문 스타일의 포괄적인 초록(Extended Abstract)을 작성해주세요.

이 초록은 전체 리포트의 모든 섹션을 요약해야 합니다.

## 연구 개요
- 연구명: {study_name}
- 암종: {cancer_type}
- 비교 그룹: {contrast[0]} vs {contrast[1]}

## 1. 차등발현 분석 (DEG Analysis)
- 총 DEG 수: {n_deg:,}개
- 상향조절 유전자: {n_up:,}개
- 하향조절 유전자: {n_down:,}개

### 가장 크게 상향조절된 유전자 (Top 5)
{chr(10).join(top_up_genes) if top_up_genes else '정보 없음'}

### 가장 크게 하향조절된 유전자 (Top 5)
{chr(10).join(top_down_genes) if top_down_genes else '정보 없음'}

## 2. 네트워크 분석 (Hub Genes)
- 총 Hub 유전자: {len(hub_gene_names)}개
{chr(10).join(hub_genes_info[:10]) if hub_genes_info else '정보 없음'}

## 3. Pathway Enrichment 분석
{chr(10).join(pathway_info[:10]) if pathway_info else '정보 없음'}

{driver_info}

{db_info}

{ml_info}

{rag_info}

{figures_info}

## 요청 사항
위 분석 결과를 종합하여 학술 논문 수준의 Extended Abstract를 JSON 형식으로 작성해주세요.

반드시 아래 모든 섹션을 포함해야 합니다:
1. 배경 (Background) - 연구의 필요성과 목적
2. 방법 (Methods) - DESeq2, Network analysis, Pathway enrichment, Driver prediction 등
3. 결과 (Results) - DEG 수, Hub 유전자, 주요 Pathway, Driver 후보 등 핵심 수치 포함
4. Driver Gene Analysis - Known Driver와 Candidate Regulator 후보 구분하여 설명
5. 문헌 기반 해석 - RAG 분석 결과 요약
6. 검증 제안 - 실험적 검증 방법 제안
7. 결론 (Conclusions) - 연구의 의의와 향후 방향

```json
{{
  "title": "한국어 제목 (암종, DEG 수, 주요 발견 포함)",
  "title_en": "English Title",
  "abstract_extended": "배경: ...\\n\\n방법: ...\\n\\n결과: ...\\n\\nDriver Gene Analysis: ...\\n\\n문헌 기반 해석: ...\\n\\n검증 제안: ...\\n\\n결론: ...",
  "key_findings": [
    "주요 발견 1 (DEG 관련)",
    "주요 발견 2 (Hub 유전자 관련)",
    "주요 발견 3 (Driver 관련)",
    "주요 발견 4 (Pathway 관련)",
    "주요 발견 5 (문헌 해석 관련)",
    "주요 발견 6 (검증 제안)"
  ],
  "validation_priorities": {{
    "qPCR": ["gene1", "gene2", ...],
    "western_blot": ["gene1", "gene2", ...],
    "functional_study": ["gene1", "gene2", ...],
    "targeted_sequencing": ["driver1", "driver2", ...],
    "biomarker_candidates": ["gene1", "gene2", ...]
  }},
  "driver_interpretation": "Known Driver와 Candidate Regulator 후보에 대한 종합 해석",
  "ml_interpretation": "ML 예측 결과에 대한 해석 (있는 경우)",
  "rag_interpretation": "RAG 문헌 해석 결과 요약",
  "literature_sources": {{
    "pmid_count": {len(rag_summary.get('pmids', []))},
    "key_pmids": {rag_summary.get('pmids', [])[:5]}
  }}
}}
```

중요 지침:
1. 한국어로 작성 (영문 제목만 영어)
2. 모든 수치는 실제 분석 결과에서 가져올 것 (DEG 수: {n_deg:,}개, Hub 유전자: {len(hub_gene_names)}개 등)
3. Driver Gene Analysis 섹션 필수 - Known Driver/Candidate Regulator 구분하여 상위 유전자 명시
4. Hub 유전자와 Driver 후보를 validation_priorities에 실제 유전자명으로 포함
5. PMID 인용 형식 사용 (예: PMID 35409110)
6. abstract_extended는 최소 800자 이상으로 상세하게 작성
7. key_findings는 6개 이상, 각 섹션에서 핵심 발견 포함

문체 지침:
- 마크다운 특수기호 사용 금지 (**, __, ##, [], () 등)
- 학술 논문처럼 자연스럽고 격식있는 문체 사용
- 괄호 안의 영문 병기는 최소화하고 필요시 한글로 풀어 설명
- 불필요한 강조나 꾸밈 없이 명료하게 기술
"""

        try:
            # Call LLM API (OpenAI or Anthropic)
            if use_openai:
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
            else:
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
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

                self.logger.info(f"Extended abstract generated via {llm_provider}: {output_path}")
                return extended_abstract
            else:
                self.logger.warning(f"Could not extract JSON from {llm_provider} response")
                return None

        except Exception as e:
            self.logger.error(f"Error generating extended abstract via {llm_provider}: {e}")
            # Return fallback extended abstract when API fails
            return self._generate_fallback_extended_abstract(data)

    def _generate_fallback_extended_abstract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template-based extended abstract when LLM API is unavailable."""
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        integrated_df = data.get('integrated_gene_table_df')
        cancer_type = self.config.get('cancer_type', 'cancer').replace('_', ' ').title()
        contrast = self.config.get('contrast', ['Tumor', 'Normal'])

        # DEG stats
        n_deg = len(deg_df) if deg_df is not None else 0
        log2fc_col = 'log2FC' if deg_df is not None and 'log2FC' in deg_df.columns else 'log2FoldChange'
        n_up = len(deg_df[deg_df[log2fc_col] > 0]) if deg_df is not None and log2fc_col in deg_df.columns else 0
        n_down = n_deg - n_up

        # Top genes
        top_up_genes = []
        top_down_genes = []
        if deg_df is not None and log2fc_col in deg_df.columns:
            deg_sorted = deg_df.sort_values(log2fc_col, ascending=False)
            gene_col = 'gene_symbol' if 'gene_symbol' in deg_df.columns else 'gene_id'
            top_up_genes = [str(g) for g in deg_sorted.head(5)[gene_col].tolist() if not str(g).startswith('ENSG')][:5]
            top_down_genes = [str(g) for g in deg_sorted.tail(5)[gene_col].tolist() if not str(g).startswith('ENSG')][:5]

        # Hub genes
        hub_gene_names = []
        if hub_df is not None:
            for _, row in hub_df.head(10).iterrows():
                gene_name = str(row.get('gene_id', row.get('gene_symbol', '')))
                if gene_name and not gene_name.startswith('ENSG'):
                    hub_gene_names.append(gene_name)

        # Pathway names
        pathway_names = []
        if pathway_df is not None and len(pathway_df) > 0:
            term_col = None
            for col in ['term_name', 'Term', 'term']:
                if col in pathway_df.columns:
                    term_col = col
                    break
            if term_col:
                pathway_names = [str(t).split(' (GO:')[0][:60] for t in pathway_df[term_col].head(5).tolist()]

        # Driver info
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        known_count = len(driver_known) if driver_known else 0
        novel_count = len(driver_novel) if driver_novel else 0
        known_names = [d.get('gene_symbol', '') for d in driver_known[:5]] if driver_known else []
        novel_names = [d.get('gene_symbol', '') for d in driver_novel[:5]] if driver_novel else []

        # DB matched
        db_matched_df = data.get('db_matched_genes_df')
        db_count = len(db_matched_df) if db_matched_df is not None else 0

        # Build comprehensive abstract text
        abstract_text = f"""배경: 본 연구는 {cancer_type} 환자의 RNA-seq 데이터를 분석하여 {contrast[0]}과 {contrast[1]} 그룹 간의 유전자 발현 차이를 규명하고, 잠재적 바이오마커 및 치료 타겟을 발굴하고자 수행되었습니다.

방법: 차등발현 분석은 DESeq2를 이용하였으며 (|log2FC| > 1, padj < 0.05), Spearman 상관관계 기반 네트워크 분석으로 Hub 유전자를 도출하였습니다. GO/KEGG pathway enrichment 분석(Enrichr)과 COSMIC/OncoKB/IntOGen 데이터베이스 검증을 수행하였고, Driver 유전자 예측에는 Two-Track 시스템(Known Driver + Candidate Regulator)을 적용하였습니다.

결과: 총 {n_deg:,}개의 차등발현 유전자(DEGs)를 식별하였으며, 이 중 상향조절 유전자 {n_up:,}개, 하향조절 유전자 {n_down:,}개가 포함됩니다. 상향조절 상위 유전자는 {', '.join(top_up_genes[:3]) if top_up_genes else 'N/A'}이며, 하향조절 상위 유전자는 {', '.join(top_down_genes[:3]) if top_down_genes else 'N/A'}입니다.

네트워크 분석 결과 {len(hub_gene_names)}개의 Hub 유전자({', '.join(hub_gene_names[:5]) if hub_gene_names else 'N/A'} 등)가 확인되었습니다. Pathway 분석에서는 {', '.join(pathway_names[:2]) if pathway_names else 'N/A'} 등이 유의하게 농축되었습니다.

Driver Gene Analysis: Known Driver Track에서 {known_count}개의 후보({', '.join(known_names[:3]) if known_names else 'N/A'} 등)가 COSMIC, OncoKB, IntOGen 데이터베이스에서 검증되었습니다. Candidate Regulator Track에서는 {novel_count}개의 조절인자 후보({', '.join(novel_names[:3]) if novel_names else 'N/A'} 등)가 발현 패턴과 네트워크 특성 분석을 통해 도출되었습니다.

결론: 본 분석에서 확인된 Hub 유전자와 Driver 후보는 {cancer_type}의 바이오마커 및 치료 타겟 개발에 유망한 후보입니다. Known Driver 유전자들은 Targeted NGS를 통해, Candidate Regulator 후보들은 문헌 검토 후 기능적 검증 실험을 통해 추가 검증이 권장됩니다."""

        # Key findings
        key_findings = []
        if n_deg > 0:
            key_findings.append(f"총 {n_deg:,}개 DEGs 식별 (상향 {n_up:,}개, 하향 {n_down:,}개)")
        if top_up_genes:
            key_findings.append(f"상향조절 상위 유전자: {', '.join(top_up_genes[:3])}")
        if top_down_genes:
            key_findings.append(f"하향조절 상위 유전자: {', '.join(top_down_genes[:3])}")
        if hub_gene_names:
            key_findings.append(f"핵심 Hub 유전자: {', '.join(hub_gene_names[:3])}")
        if pathway_names:
            key_findings.append(f"주요 Pathway: {pathway_names[0][:50]}")
        if known_count > 0:
            key_findings.append(f"Known Driver 후보 {known_count}개 (COSMIC/OncoKB/IntOGen 검증)")
        if novel_count > 0:
            key_findings.append(f"Candidate Regulator 후보 {novel_count}개 (추가 검증 필요)")
        if db_count > 0:
            key_findings.append(f"암 유전자 DB 매칭 {db_count}개")

        # Driver interpretation
        driver_interp = ""
        if known_count > 0 or novel_count > 0:
            driver_interp = f"Known Driver Track에서 {known_count}개의 후보가 암 유전자 데이터베이스에서 검증되었습니다. 이들은 기존에 알려진 암 유전자로서 Targeted NGS 패널을 통한 변이 확인이 권장됩니다. Candidate Regulator Track에서는 {novel_count}개의 조절인자 후보가 도출되었으며, 이들은 '확정된 driver'가 아닌 '추가 검증이 필요한 후보'입니다."

        # Validation priorities
        validation_priorities = {
            "qPCR": hub_gene_names[:5] if hub_gene_names else [],
            "western_blot": hub_gene_names[:3] if hub_gene_names else [],
            "targeted_sequencing": known_names[:3] if known_names else [],
            "functional_study": novel_names[:3] if novel_names else [],
            "biomarker_candidates": (top_up_genes[:2] + top_down_genes[:2])[:4] if top_up_genes or top_down_genes else []
        }

        return {
            "abstract_extended": abstract_text,
            "title": f"{cancer_type} RNA-seq 차등발현 분석 및 Driver 유전자 예측 연구",
            "title_en": f"Differential Expression Analysis and Driver Gene Prediction in {cancer_type}",
            "key_findings": key_findings,
            "driver_interpretation": driver_interp,
            "rag_interpretation": f"{len(hub_gene_names)}개 핵심 유전자에 대한 문헌 기반 해석이 수행되었습니다. 상세 내용은 Literature-Based Interpretation 섹션을 참조하세요.",
            "validation_priorities": validation_priorities,
            "ml_interpretation": ""
        }

    def _generate_visualization_interpretations(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate LLM-based interpretations for each visualization.

        Creates structured interpretations for:
        - Volcano Plot: DEG 분포 해석
        - Heatmap: 발현 패턴 해석
        - Network Graph: Hub 유전자 및 상호작용 해석
        - PCA Plot: 샘플 분리도 해석
        - Pathway Bar Plot: 경로 분석 해석
        """
        # Try OpenAI first (cheaper), then Anthropic as fallback
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        use_openai = OPENAI_AVAILABLE and openai_key
        use_anthropic = ANTHROPIC_AVAILABLE and anthropic_key and not use_openai

        if not use_openai and not use_anthropic:
            self.logger.warning("No LLM API available for visualization interpretations")
            return self._generate_fallback_viz_interpretations(data)

        llm_provider = "OpenAI" if use_openai else "Anthropic"

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
            top_up_genes = [str(g) for g in deg_sorted.head(5)[gene_col].tolist()]
            top_down_genes = [str(g) for g in deg_sorted.tail(5)[gene_col].tolist()]

        # Hub genes for network
        hub_genes_list = []
        if hub_df is not None:
            hub_log2fc_col = 'log2FC' if 'log2FC' in hub_df.columns else 'log2FoldChange'
            for _, row in hub_df.head(10).iterrows():
                gene_name = str(row.get('gene_id', row.get('gene_symbol', 'Unknown')))
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
            # Call LLM API (OpenAI or Anthropic)
            if use_openai:
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
            else:
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
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

                self.logger.info(f"Visualization interpretations generated via {llm_provider}: {output_path}")
                return viz_interpretations
            else:
                self.logger.warning(f"Could not extract JSON from {llm_provider} response")
                return None

        except Exception as e:
            self.logger.error(f"Error generating visualization interpretations via {llm_provider}: {e}")
            # Return fallback interpretations when API fails
            return self._generate_fallback_viz_interpretations(data)

    def _generate_fallback_viz_interpretations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template-based visualization interpretations when LLM API is unavailable."""
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        cancer_type = self.config.get('cancer_type', 'cancer').replace('_', ' ').title()

        # DEG stats
        n_deg = len(deg_df) if deg_df is not None else 0
        log2fc_col = 'log2FC' if deg_df is not None and 'log2FC' in deg_df.columns else 'log2FoldChange'
        n_up = len(deg_df[deg_df[log2fc_col] > 0]) if deg_df is not None and log2fc_col in deg_df.columns else 0
        n_down = n_deg - n_up

        # Top genes
        top_up_genes = []
        top_down_genes = []
        if deg_df is not None and log2fc_col in deg_df.columns:
            deg_sorted = deg_df.sort_values(log2fc_col, ascending=False)
            gene_col = 'gene_symbol' if 'gene_symbol' in deg_df.columns else 'gene_id'
            top_up_genes = [str(g) for g in deg_sorted.head(5)[gene_col].tolist() if not str(g).startswith('ENSG')][:3]
            top_down_genes = [str(g) for g in deg_sorted.tail(5)[gene_col].tolist() if not str(g).startswith('ENSG')][:3]

        # Hub genes
        hub_gene_names = []
        if hub_df is not None:
            for _, row in hub_df.head(5).iterrows():
                gene_name = str(row.get('gene_id', row.get('gene_symbol', '')))
                if gene_name and not gene_name.startswith('ENSG'):
                    hub_gene_names.append(gene_name)

        # Pathway names
        pathway_names = []
        if pathway_df is not None and len(pathway_df) > 0:
            term_col = None
            for col in ['term_name', 'Term', 'term']:
                if col in pathway_df.columns:
                    term_col = col
                    break
            if term_col:
                pathway_names = [str(t).split(' (GO:')[0][:50] for t in pathway_df[term_col].head(3).tolist()]

        return {
            "volcano_plot": {
                "title": "Volcano Plot 해석",
                "summary": f"총 {n_deg:,}개의 차등발현 유전자(DEGs)가 식별되었습니다. 상향조절 유전자 {n_up:,}개(빨간점), 하향조절 유전자 {n_down:,}개(파란점)가 유의하게 변화했습니다.",
                "key_observations": [
                    f"상향조절 상위 유전자: {', '.join(top_up_genes) if top_up_genes else 'N/A'}",
                    f"하향조절 상위 유전자: {', '.join(top_down_genes) if top_down_genes else 'N/A'}",
                    f"상향/하향 비율: {n_up}/{n_down} ({n_up/(n_deg)*100:.1f}% 상향)" if n_deg > 0 else "데이터 없음"
                ],
                "biological_significance": f"{cancer_type}에서 발현 변화가 큰 유전자들은 암의 발생, 진행, 또는 전이에 관여할 가능성이 있습니다. 특히 상향조절된 유전자는 oncogene 역할을, 하향조절된 유전자는 tumor suppressor 역할을 할 수 있습니다.",
                "interpretation_guide": "X축(log2FC)은 발현 변화량을, Y축(-log10 p-value)은 통계적 유의성을 나타냅니다. 그래프의 오른쪽 상단에 위치한 점일수록 유의하게 상향조절된 유전자입니다."
            },
            "heatmap": {
                "title": "Heatmap 해석",
                "summary": f"상위 DEGs의 발현 패턴을 시각화했습니다. 암 조직과 정상 조직 간의 명확한 발현 차이가 관찰됩니다.",
                "key_observations": [
                    "암 샘플과 정상 샘플이 hierarchical clustering에서 분리되어 있습니다.",
                    f"Hub 유전자({', '.join(hub_gene_names[:3]) if hub_gene_names else 'N/A'})에서 일관된 발현 패턴이 관찰됩니다.",
                    "발현 패턴의 일관성은 분석 결과의 신뢰도를 높여줍니다."
                ],
                "pattern_analysis": "색상이 빨간색일수록 높은 발현, 파란색일수록 낮은 발현을 나타냅니다. 샘플 간 유사한 발현 패턴을 보이는 유전자들은 같은 생물학적 경로에 관여할 가능성이 높습니다.",
                "interpretation_guide": "각 열은 샘플, 각 행은 유전자를 나타냅니다. Dendrograms은 유사한 발현 패턴을 가진 유전자/샘플의 군집을 보여줍니다."
            },
            "network_graph": {
                "title": "Network Analysis 해석",
                "summary": f"유전자 상관관계 네트워크에서 {len(hub_gene_names)}개의 Hub 유전자가 식별되었습니다.",
                "hub_gene_analysis": f"핵심 Hub 유전자: {', '.join(hub_gene_names[:5]) if hub_gene_names else 'N/A'}. 이들은 네트워크에서 많은 연결을 가지며, 핵심 조절 역할을 할 가능성이 높습니다.",
                "network_topology": "네트워크는 scale-free 특성을 보이며, 소수의 Hub 유전자가 다수의 유전자와 연결되어 있습니다. 이는 생물학적 네트워크의 전형적인 특성입니다.",
                "biological_implications": f"Hub 유전자는 {cancer_type}의 핵심 조절자로 작용할 수 있으며, 치료 타겟이나 바이오마커 후보로서 추가 연구가 필요합니다.",
                "interpretation_guide": "노드 크기는 연결 수(degree)에 비례합니다. 큰 노드가 Hub 유전자입니다. 간선(edge)은 유전자 간 상관관계를 나타냅니다."
            },
            "pca_plot": {
                "title": "PCA 분석 해석",
                "summary": "Principal Component Analysis를 통해 샘플 간 전체적인 발현 패턴 차이를 시각화했습니다.",
                "separation_analysis": "암 조직과 정상 조직 샘플이 PCA 공간에서 분리되어 있으면, 두 그룹 간 유의한 발현 차이가 있음을 나타냅니다.",
                "variance_explanation": "PC1(x축)이 가장 많은 분산을 설명하며, 주로 암/정상 간의 차이를 반영합니다. PC2(y축)는 그 다음으로 많은 분산을 설명합니다.",
                "interpretation_guide": "각 점은 하나의 샘플을 나타냅니다. 가까이 위치한 샘플들은 비슷한 발현 프로파일을 가집니다."
            },
            "pathway_barplot": {
                "title": "Pathway Enrichment 해석",
                "summary": f"DEGs가 농축된 상위 생물학적 경로: {', '.join(pathway_names[:2]) if pathway_names else 'N/A'}",
                "top_pathways": [
                    f"{pathway_names[0]}: 가장 유의하게 농축된 경로" if pathway_names else "데이터 없음",
                    f"{pathway_names[1] if len(pathway_names) > 1 else 'N/A'}: 두 번째로 유의한 경로",
                    "이들 경로는 암의 발생 및 진행과 관련된 핵심 생물학적 프로세스를 나타냅니다."
                ],
                "functional_theme": f"식별된 경로들은 {cancer_type}의 주요 특성(세포 증식, 대사 변화, 면역 반응 등)을 반영합니다.",
                "therapeutic_implications": "농축된 경로들 중 약물 타겟이 존재하는 경로는 치료 전략 개발에 활용될 수 있습니다.",
                "interpretation_guide": "막대 길이는 -log10(p-value)를 나타내며, 길수록 통계적으로 더 유의합니다. 막대 위 숫자는 해당 경로에 포함된 DEG 수입니다."
            }
        }

    def _generate_research_recommendations(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate comprehensive research recommendations using LLM.

        Creates actionable next-step recommendations including:
        - Therapeutic target candidates (druggable genes)
        - Drug repurposing suggestions (DGIdb-based)
        - Experimental validation priorities
        - Future research directions
        - Biomarker development opportunities
        """
        # Try OpenAI first (cheaper), then Anthropic as fallback
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        use_openai = OPENAI_AVAILABLE and openai_key
        use_anthropic = ANTHROPIC_AVAILABLE and anthropic_key and not use_openai

        if not use_openai and not use_anthropic:
            self.logger.warning("No LLM API available for research recommendations")
            return self._generate_fallback_research_recommendations(data)

        llm_provider = "OpenAI" if use_openai else "Anthropic"
        self.logger.info(f"Generating research recommendations via {llm_provider}...")

        # Gather comprehensive data for recommendations
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        db_matched_df = data.get('db_matched_genes_df')

        # DEG stats
        n_deg = len(deg_df) if deg_df is not None else 0
        log2fc_col = 'log2FC' if deg_df is not None and 'log2FC' in deg_df.columns else 'log2FoldChange'
        n_up = len(deg_df[deg_df[log2fc_col] > 0]) if deg_df is not None and log2fc_col in deg_df.columns else 0
        n_down = n_deg - n_up

        # Top genes with gene symbols
        top_up_genes = []
        top_down_genes = []
        if deg_df is not None and log2fc_col in deg_df.columns:
            deg_sorted = deg_df.sort_values(log2fc_col, ascending=False)
            gene_col = 'gene_symbol' if 'gene_symbol' in deg_df.columns else 'gene_id'
            for _, row in deg_sorted.head(10).iterrows():
                gene = str(row.get(gene_col, 'Unknown'))
                fc = row.get(log2fc_col, 0)
                if not gene.startswith('ENSG'):
                    top_up_genes.append(f"{gene} (log2FC={fc:.2f})")
            for _, row in deg_sorted.tail(10).iterrows():
                gene = str(row.get(gene_col, 'Unknown'))
                fc = row.get(log2fc_col, 0)
                if not gene.startswith('ENSG'):
                    top_down_genes.append(f"{gene} (log2FC={fc:.2f})")

        # Hub genes
        hub_genes_info = []
        if hub_df is not None:
            hub_log2fc_col = 'log2FC' if 'log2FC' in hub_df.columns else 'log2FoldChange'
            gene_col = 'gene_symbol' if 'gene_symbol' in hub_df.columns else 'gene_id'
            for _, row in hub_df.head(15).iterrows():
                gene = str(row.get(gene_col, row.get('gene_id', 'Unknown')))
                degree = row.get('degree', 0)
                fc = row.get(hub_log2fc_col, 0)
                if not gene.startswith('ENSG'):
                    hub_genes_info.append(f"{gene} (degree={degree}, log2FC={fc:.2f})")

        # Pathway info
        pathway_info = []
        if pathway_df is not None:
            for _, row in pathway_df.head(15).iterrows():
                term = row.get('Term', row.get('term', 'Unknown'))
                pval = row.get('P-value', row.get('pvalue', 0))
                genes = row.get('Genes', row.get('genes', ''))[:100]
                pathway_info.append(f"- {term} (p={pval:.2e}): {genes}")

        # Known drivers
        known_driver_info = []
        for d in driver_known[:10]:
            gene = d.get('gene_symbol', '')
            tier = d.get('cosmic_tier', '')
            role = d.get('cosmic_role', '')
            direction = d.get('direction', '')
            known_driver_info.append(f"{gene} ({tier}, {role}, {direction})")

        # Candidate regulators
        candidate_info = []
        for d in driver_novel[:10]:
            gene = d.get('gene_symbol', '')
            hub_score = d.get('hub_score', 0)
            direction = d.get('direction', '')
            candidate_info.append(f"{gene} (hub={hub_score:.2f}, {direction})")

        # DB matched genes
        db_info = []
        if db_matched_df is not None:
            for _, row in db_matched_df.head(10).iterrows():
                gene = row.get('gene_symbol', '')
                sources = row.get('db_sources', '')
                db_info.append(f"{gene} ({sources})")

        # Study info
        cancer_type = self.config.get('cancer_type', 'cancer').replace('_', ' ')
        contrast = self.config.get('contrast', ['Tumor', 'Normal'])

        prompt = f"""당신은 암 연구 전문가이자 바이오인포매틱스 컨설턴트입니다. 아래 RNA-seq 분석 결과를 바탕으로 구체적이고 실행 가능한 후속 연구 추천을 제공해주세요.

## 분석 개요
- 암종: {cancer_type}
- 비교 그룹: {contrast[0]} vs {contrast[1]}
- 총 DEG: {n_deg:,}개 (상향 {n_up:,}개, 하향 {n_down:,}개)

## 상향조절 상위 유전자
{chr(10).join(top_up_genes[:10]) if top_up_genes else '없음'}

## 하향조절 상위 유전자
{chr(10).join(top_down_genes[:10]) if top_down_genes else '없음'}

## Hub 유전자 (네트워크 핵심 조절자)
{chr(10).join(hub_genes_info[:10]) if hub_genes_info else '없음'}

## Pathway Enrichment 결과
{chr(10).join(pathway_info[:10]) if pathway_info else '없음'}

## Known Driver 후보 (COSMIC/OncoKB 검증됨)
{chr(10).join(known_driver_info[:10]) if known_driver_info else '없음'}

## Candidate Regulator 후보 (신규 발견)
{chr(10).join(candidate_info[:10]) if candidate_info else '없음'}

## 암 DB 매칭 유전자
{chr(10).join(db_info[:10]) if db_info else '없음'}

아래 JSON 형식으로 종합적인 후속 연구 추천을 제공해주세요:

```json
{{
  "therapeutic_targets": {{
    "description": "치료 타겟 후보 설명 (2-3문장)",
    "high_priority": [
      {{"gene": "유전자명", "rationale": "추천 이유 (druggability, 발현 변화, 기능 등)", "existing_drugs": ["관련 약물 1", "약물 2"], "target_class": "kinase/receptor/transcription factor 등"}}
    ],
    "medium_priority": [
      {{"gene": "유전자명", "rationale": "추천 이유", "existing_drugs": [], "target_class": "분류"}}
    ]
  }},
  "drug_repurposing": {{
    "description": "약물 재목적화 가능성 설명 (2-3문장)",
    "candidates": [
      {{"drug": "약물명", "target_gene": "타겟 유전자", "original_indication": "기존 적응증", "repurposing_rationale": "재목적화 근거", "clinical_status": "FDA 승인/임상시험 단계"}}
    ]
  }},
  "experimental_validation": {{
    "description": "실험 검증 전략 설명 (2-3문장)",
    "immediate_validation": {{
      "qPCR": {{"genes": ["유전자1", "유전자2"], "purpose": "발현 검증 목적"}},
      "western_blot": {{"genes": ["유전자1"], "purpose": "단백질 발현 검증"}}
    }},
    "functional_studies": {{
      "knockdown_knockout": {{"genes": ["유전자1"], "method": "siRNA/CRISPR", "readout": "측정 지표"}},
      "overexpression": {{"genes": ["유전자1"], "method": "plasmid/viral", "readout": "측정 지표"}}
    }},
    "clinical_validation": {{
      "tissue_analysis": {{"method": "IHC/IF", "genes": ["유전자1"], "sample_type": "FFPE/fresh frozen"}},
      "liquid_biopsy": {{"biomarkers": ["바이오마커1"], "method": "ctDNA/CTC"}}
    }}
  }},
  "biomarker_development": {{
    "description": "바이오마커 개발 가능성 (2-3문장)",
    "diagnostic_candidates": [
      {{"gene": "유전자명", "marker_type": "진단/예후/예측", "evidence_level": "high/medium/low", "rationale": "추천 근거"}}
    ],
    "prognostic_candidates": [
      {{"gene": "유전자명", "association": "좋은/나쁜 예후", "validation_needed": "필요한 검증"}}
    ]
  }},
  "future_research_directions": {{
    "description": "향후 연구 방향 요약 (2-3문장)",
    "short_term": [
      {{"direction": "연구 방향 1", "timeline": "6개월 이내", "resources_needed": "필요 자원", "expected_outcome": "예상 결과"}}
    ],
    "medium_term": [
      {{"direction": "연구 방향 2", "timeline": "1-2년", "resources_needed": "필요 자원", "expected_outcome": "예상 결과"}}
    ],
    "long_term": [
      {{"direction": "연구 방향 3", "timeline": "3-5년", "resources_needed": "필요 자원", "expected_outcome": "예상 결과"}}
    ]
  }},
  "collaboration_suggestions": {{
    "description": "협력 연구 제안",
    "expertise_needed": ["필요 전문성 1", "필요 전문성 2"],
    "potential_partnerships": ["잠재적 협력 기관/연구실 유형"]
  }},
  "funding_opportunities": {{
    "description": "연구비 지원 가능성",
    "suitable_grant_types": ["적합한 연구비 유형 1", "유형 2"],
    "key_selling_points": ["연구의 강점 1", "강점 2"]
  }},
  "cautions_and_limitations": {{
    "description": "주의사항 및 한계점",
    "technical_limitations": ["기술적 한계 1"],
    "interpretation_caveats": ["해석상 주의점 1"],
    "validation_requirements": ["필수 검증 사항 1"]
  }}
}}
```

중요 지침:
1. 한국어로 작성 (유전자명, 약물명, 기술 용어는 영어 유지)
2. 구체적이고 실행 가능한 추천 제공
3. 분석 결과에 기반한 맞춤형 추천 (일반적인 내용 지양)
4. 실제 유전자명과 관련 약물 정보 포함
5. 우선순위와 근거를 명확히 제시
6. 현실적인 timeline과 resource 제안
7. therapeutic_targets의 high_priority는 3-5개, medium_priority는 3-5개
8. drug_repurposing candidates는 3-5개
9. 각 섹션에 최소 2개 이상의 구체적 항목 포함
"""

        try:
            # Call LLM API
            if use_openai:
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=6000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
            else:
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=6000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = message.content[0].text

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                recommendations = json.loads(json_str)

                # Save to file
                run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
                output_path = run_dir / "research_recommendations.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(recommendations, f, ensure_ascii=False, indent=2)

                self.logger.info(f"Research recommendations generated via {llm_provider}: {output_path}")
                return recommendations
            else:
                self.logger.warning(f"Could not extract JSON from {llm_provider} response")
                return self._generate_fallback_research_recommendations(data)

        except Exception as e:
            self.logger.error(f"Error generating research recommendations via {llm_provider}: {e}")
            return self._generate_fallback_research_recommendations(data)

    def _generate_fallback_research_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template-based research recommendations when LLM API is unavailable."""
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        cancer_type = self.config.get('cancer_type', 'cancer').replace('_', ' ').title()

        # Extract gene names
        hub_gene_names = []
        if hub_df is not None:
            gene_col = 'gene_symbol' if 'gene_symbol' in hub_df.columns else 'gene_id'
            for _, row in hub_df.head(10).iterrows():
                gene = str(row.get(gene_col, row.get('gene_id', '')))
                if gene and not gene.startswith('ENSG'):
                    hub_gene_names.append(gene)

        # Top DEGs
        top_up = []
        top_down = []
        if deg_df is not None:
            log2fc_col = 'log2FC' if 'log2FC' in deg_df.columns else 'log2FoldChange'
            gene_col = 'gene_symbol' if 'gene_symbol' in deg_df.columns else 'gene_id'
            if log2fc_col in deg_df.columns:
                sorted_df = deg_df.sort_values(log2fc_col, ascending=False)
                for _, row in sorted_df.head(5).iterrows():
                    gene = str(row.get(gene_col, ''))
                    if gene and not gene.startswith('ENSG'):
                        top_up.append(gene)
                for _, row in sorted_df.tail(5).iterrows():
                    gene = str(row.get(gene_col, ''))
                    if gene and not gene.startswith('ENSG'):
                        top_down.append(gene)

        # Known and novel drivers
        known_names = [d.get('gene_symbol', '') for d in driver_known[:5] if d.get('gene_symbol')]
        novel_names = [d.get('gene_symbol', '') for d in driver_novel[:5] if d.get('gene_symbol')]

        # Pathway names
        pathway_names = []
        if pathway_df is not None and len(pathway_df) > 0:
            term_col = None
            for col in ['term_name', 'Term', 'term']:
                if col in pathway_df.columns:
                    term_col = col
                    break
            if term_col:
                pathway_names = [str(t).split(' (GO:')[0][:60] for t in pathway_df[term_col].head(5).tolist()]

        return {
            "therapeutic_targets": {
                "description": f"{cancer_type}에서 식별된 Hub 유전자와 Driver 후보를 기반으로 치료 타겟을 제안합니다. 상향조절된 유전자는 억제제, 하향조절된 유전자는 활성화제 개발 대상입니다.",
                "high_priority": [
                    {"gene": gene, "rationale": "Hub 유전자로서 네트워크 중심성이 높음", "existing_drugs": [], "target_class": "추가 분석 필요"}
                    for gene in hub_gene_names[:3]
                ] if hub_gene_names else [],
                "medium_priority": [
                    {"gene": gene, "rationale": "Driver 후보로 식별됨", "existing_drugs": [], "target_class": "추가 분석 필요"}
                    for gene in known_names[:3]
                ] if known_names else []
            },
            "drug_repurposing": {
                "description": "Hub 유전자와 Driver 후보에 대한 기존 약물 재목적화 가능성을 검토하세요. DGIdb (dgidb.org)에서 약물-유전자 상호작용을 조회할 수 있습니다.",
                "candidates": [
                    {"drug": "DGIdb 조회 필요", "target_gene": gene, "original_indication": "조회 필요", "repurposing_rationale": f"{cancer_type}에서 유의한 발현 변화", "clinical_status": "확인 필요"}
                    for gene in hub_gene_names[:3]
                ] if hub_gene_names else []
            },
            "experimental_validation": {
                "description": "분석 결과를 실험적으로 검증하기 위한 단계별 전략입니다. 1차 검증(qPCR), 2차 검증(Western blot), 기능 연구 순으로 진행하세요.",
                "immediate_validation": {
                    "qPCR": {"genes": hub_gene_names[:5] if hub_gene_names else top_up[:3], "purpose": "mRNA 발현 변화 검증"},
                    "western_blot": {"genes": hub_gene_names[:3] if hub_gene_names else top_up[:2], "purpose": "단백질 수준 발현 확인"}
                },
                "functional_studies": {
                    "knockdown_knockout": {"genes": top_up[:2] if top_up else [], "method": "siRNA 또는 CRISPR-Cas9", "readout": "세포 증식, 이동, 침윤 능력"},
                    "overexpression": {"genes": top_down[:2] if top_down else [], "method": "발현 벡터 transfection", "readout": "종양 억제 효과"}
                },
                "clinical_validation": {
                    "tissue_analysis": {"method": "면역조직화학(IHC)", "genes": hub_gene_names[:3] if hub_gene_names else [], "sample_type": "FFPE 조직"},
                    "liquid_biopsy": {"biomarkers": hub_gene_names[:2] if hub_gene_names else [], "method": "ctDNA 또는 CTC 분석"}
                }
            },
            "biomarker_development": {
                "description": f"Hub 유전자와 일관된 발현 변화를 보이는 유전자는 {cancer_type}의 진단 또는 예후 바이오마커 후보입니다.",
                "diagnostic_candidates": [
                    {"gene": gene, "marker_type": "진단", "evidence_level": "medium", "rationale": "유의한 발현 변화와 네트워크 중심성"}
                    for gene in hub_gene_names[:3]
                ] if hub_gene_names else [],
                "prognostic_candidates": [
                    {"gene": gene, "association": "추가 분석 필요", "validation_needed": "생존 분석 (TCGA 또는 GEO 데이터)"}
                    for gene in hub_gene_names[:3]
                ] if hub_gene_names else []
            },
            "future_research_directions": {
                "description": f"{cancer_type} 연구를 위한 단기, 중기, 장기 연구 방향을 제안합니다.",
                "short_term": [
                    {"direction": "Hub 유전자 발현 검증 (qPCR, Western blot)", "timeline": "3-6개월", "resources_needed": "분자생물학 실험 장비, 항체", "expected_outcome": "발현 변화 확인"},
                    {"direction": "독립 코호트에서 발현 검증", "timeline": "3-6개월", "resources_needed": "GEO/TCGA 데이터", "expected_outcome": "결과 재현성 확인"}
                ],
                "medium_term": [
                    {"direction": "기능 연구 (knockdown/overexpression)", "timeline": "1-2년", "resources_needed": "세포주, transfection 시약, CRISPR 시스템", "expected_outcome": "인과관계 규명"},
                    {"direction": "약물 스크리닝", "timeline": "1-2년", "resources_needed": "약물 라이브러리, HTS 시스템", "expected_outcome": "후보 약물 발굴"}
                ],
                "long_term": [
                    {"direction": "전임상 동물 모델 연구", "timeline": "2-4년", "resources_needed": "동물 시설, PDX 모델", "expected_outcome": "치료 효과 검증"},
                    {"direction": "임상시험 설계", "timeline": "3-5년", "resources_needed": "IRB 승인, 임상 협력 네트워크", "expected_outcome": "임상 적용 가능성 평가"}
                ]
            },
            "collaboration_suggestions": {
                "description": "본 연구의 심화를 위해 다학제 협력이 권장됩니다.",
                "expertise_needed": ["약물화학 (drug design)", "임상종양학", "생물정보학", "분자생물학"],
                "potential_partnerships": ["암 센터", "제약회사 R&D", "생물정보학 코어 시설"]
            },
            "funding_opportunities": {
                "description": "연구 확장을 위한 연구비 지원 기회를 제안합니다.",
                "suitable_grant_types": ["기초연구사업 (한국연구재단)", "바이오의료기술개발사업", "암정복추진연구개발사업"],
                "key_selling_points": [
                    f"{cancer_type}에서 신규 치료 타겟 후보 발굴",
                    "빅데이터 기반 바이오마커 개발",
                    "약물 재목적화를 통한 빠른 임상 적용 가능성"
                ]
            },
            "cautions_and_limitations": {
                "description": "분석 결과 해석 및 후속 연구 수행 시 주의사항입니다.",
                "technical_limitations": [
                    "RNA 수준 변화가 단백질 수준을 반영하지 않을 수 있음",
                    "배치 효과 및 샘플 heterogeneity 가능성"
                ],
                "interpretation_caveats": [
                    "Hub 유전자 ≠ Driver 유전자 (발현 변화 vs 인과관계)",
                    "Pathway 농축이 기능적 중요성을 보장하지 않음"
                ],
                "validation_requirements": [
                    "독립 코호트에서 결과 재현 필수",
                    "기능 실험을 통한 인과관계 검증 필요"
                ]
            }
        }

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
