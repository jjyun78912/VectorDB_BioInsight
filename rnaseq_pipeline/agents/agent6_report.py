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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ..utils.base_agent import BaseAgent


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

    def _generate_visual_dashboard_html(self, data: Dict) -> str:
        """Generate Level 2: Visual Dashboard (30초 파악)."""
        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})

        # Get key figures
        volcano_src = figures.get('volcano_plot', '')
        pathway_src = figures.get('pathway_barplot', '')
        network_src = figures.get('network_graph', '')
        heatmap_src = figures.get('heatmap_top50', '')
        volcano_interactive = interactive_figures.get('volcano_interactive', '')

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
        volcano_desc = '''<p class="panel-desc"><strong>X축:</strong> log2 Fold Change (발현 변화량) | <strong>Y축:</strong> -log10(padj) (통계적 유의성)<br>
        <span style="color:#dc2626;">●빨간점</span> = 상향조절 (암에서 증가) | <span style="color:#2563eb;">●파란점</span> = 하향조절 (암에서 감소) | 점선 = 유의성 기준선</p>'''

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
                        <iframe id="volcano-iframe" srcdoc="{volcano_interactive.replace('"', '&quot;')}" style="width:100%; height:550px; border:none; border-radius:8px;"></iframe>
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

                <div class="dashboard-panel">
                    <h4>Network Hub Genes</h4>
                    <p class="panel-desc">유전자 간 공발현(co-expression) 네트워크에서 중심적 역할을 하는 Hub 유전자입니다. Hub는 많은 유전자와 연결되어 있어 핵심 조절자일 가능성이 높습니다.</p>
                    {f'<img src="{network_src}" alt="Network" />' if network_src else '<p class="no-data">No plot available</p>'}
                </div>

                <div class="dashboard-panel full-width">
                    <h4>Expression Heatmap (Top 50 DEGs)</h4>
                    <p class="panel-desc">상위 50개 DEG의 샘플별 발현 패턴입니다. 빨간색은 높은 발현, 파란색은 낮은 발현을 의미합니다. 샘플들이 조건(Tumor vs Normal)에 따라 구분되는지 확인하세요.</p>
                    {f'<img src="{heatmap_src}" alt="Heatmap" />' if heatmap_src else '<p class="no-data">No heatmap available</p>'}
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
        """Generate modern CSS styles."""
        return '''
        <style>
            :root {
                --primary: #4F46E5;
                --primary-light: #818CF8;
                --secondary: #7C3AED;
                --success: #10B981;
                --warning: #F59E0B;
                --danger: #EF4444;
                --gray-50: #F9FAFB;
                --gray-100: #F3F4F6;
                --gray-200: #E5E7EB;
                --gray-500: #6B7280;
                --gray-700: #374151;
                --gray-900: #111827;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #F0F4FF 0%, #F9FAFB 100%);
                color: var(--gray-900);
                line-height: 1.6;
                min-height: 100vh;
            }

            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

            /* Header */
            header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                padding: 40px 20px;
                position: relative;
                overflow: hidden;
            }

            header::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -10%;
                width: 400px;
                height: 400px;
                background: rgba(255,255,255,0.1);
                border-radius: 50%;
            }

            header h1 {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 8px;
            }

            header .meta {
                opacity: 0.9;
                font-size: 0.9rem;
            }

            /* Navigation */
            .nav-pills {
                display: flex;
                gap: 10px;
                padding: 15px 20px;
                background: white;
                border-bottom: 1px solid var(--gray-200);
                position: sticky;
                top: 0;
                z-index: 100;
                overflow-x: auto;
            }

            .nav-pill {
                padding: 8px 16px;
                border-radius: 20px;
                background: var(--gray-100);
                color: var(--gray-700);
                text-decoration: none;
                font-size: 0.85rem;
                font-weight: 500;
                white-space: nowrap;
                transition: all 0.2s;
            }

            .nav-pill:hover, .nav-pill.active {
                background: var(--primary);
                color: white;
            }

            /* Executive Summary */
            .executive-summary {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }

            .summary-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }

            .summary-title {
                display: flex;
                align-items: center;
                gap: 15px;
            }

            .summary-title h2 {
                font-size: 1.5rem;
                color: var(--gray-900);
            }

            .confidence-badge {
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .confidence-badge.high { background: #D1FAE5; color: #065F46; }
            .confidence-badge.medium { background: #FEF3C7; color: #92400E; }
            .confidence-badge.low { background: #FEE2E2; color: #991B1B; }

            .key-metrics {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 24px;
            }

            .metric-card {
                background: var(--gray-50);
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                border: 2px solid transparent;
                transition: all 0.2s;
            }

            .metric-card:hover {
                border-color: var(--primary-light);
                transform: translateY(-2px);
            }

            .metric-card.primary { background: linear-gradient(135deg, #EEF2FF, #E0E7FF); }
            .metric-card.highlight { background: linear-gradient(135deg, #FEF3C7, #FDE68A); }

            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--primary);
            }

            .metric-card.highlight .metric-value { color: #B45309; }

            .metric-label {
                color: var(--gray-500);
                font-size: 0.9rem;
                margin-top: 4px;
            }

            .metric-detail {
                font-size: 0.8rem;
                color: var(--gray-500);
                margin-top: 8px;
            }

            .one-line-summary {
                background: var(--gray-50);
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 16px;
            }

            .one-line-summary h4 {
                font-size: 0.85rem;
                color: var(--gray-500);
                margin-bottom: 8px;
            }

            .one-line-summary p {
                font-size: 1rem;
                color: var(--gray-700);
            }

            .warning-box {
                display: flex;
                align-items: center;
                gap: 12px;
                background: #FEF3C7;
                border: 1px solid #FCD34D;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 0.85rem;
                color: #92400E;
            }

            .warning-icon { font-size: 1.2rem; }

            /* Visual Dashboard */
            .visual-dashboard {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }

            .visual-dashboard h2 {
                font-size: 1.5rem;
                margin-bottom: 24px;
                color: var(--gray-900);
            }

            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }

            .dashboard-panel {
                background: #ffffff;
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.06);
                border: 1px solid #e5e7eb;
            }

            .dashboard-panel.main-plot {
                grid-row: span 2;
            }

            .dashboard-panel.full-width {
                grid-column: 1 / -1;
            }

            .dashboard-panel h4 {
                font-size: 1rem;
                font-weight: 600;
                color: #1f2937;
                margin-bottom: 16px;
                padding-bottom: 12px;
                border-bottom: 2px solid #f3f4f6;
            }

            .dashboard-panel img {
                width: 100%;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }

            .panel-note {
                font-size: 0.8rem;
                color: #6b7280;
                margin-top: 12px;
                padding: 8px 12px;
                background: #fef3c7;
                border-radius: 8px;
                border-left: 3px solid #f59e0b;
            }

            .panel-desc {
                font-size: 0.85rem;
                color: #4b5563;
                line-height: 1.6;
                margin-bottom: 16px;
                padding: 12px 16px;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-radius: 10px;
                border-left: 4px solid #3b82f6;
            }

            .section-intro {
                font-size: 1rem;
                color: #6b7280;
                margin-bottom: 24px;
                text-align: center;
            }

            /* Volcano Toggle */
            .volcano-container { position: relative; }

            .volcano-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }

            .view-toggle {
                display: flex;
                gap: 4px;
                background: var(--gray-200);
                padding: 3px;
                border-radius: 6px;
            }

            .toggle-btn {
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
                background: transparent;
                color: var(--gray-600);
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.2s;
            }

            .toggle-btn.active {
                background: white;
                color: var(--primary);
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .toggle-btn:hover:not(.active) {
                color: var(--gray-900);
            }

            .volcano-view {
                display: none;
            }

            .volcano-view.active {
                display: block;
            }

            /* Gene Bars - Enhanced for readability */
            .gene-bars { display: flex; flex-direction: column; gap: 10px; }

            .gene-bar-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 4px 0;
            }

            .gene-name {
                width: 80px;
                font-size: 0.9rem;
                font-weight: 600;
                color: #1f2937;
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            }

            .gene-bar-container {
                flex: 1;
                height: 20px;
                background: #f3f4f6;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            }

            .gene-bar {
                height: 100%;
                border-radius: 10px;
                transition: width 0.5s ease;
                box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            }

            .gene-bar.up { background: linear-gradient(90deg, #f87171, #dc2626); }
            .gene-bar.down { background: linear-gradient(90deg, #60a5fa, #2563eb); }

            .gene-value {
                width: 60px;
                font-size: 0.85rem;
                font-weight: 600;
                color: #374151;
                text-align: right;
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            }

            /* Pathway List - Enhanced */
            .pathway-list { display: flex; flex-direction: column; gap: 8px; }

            .pathway-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 8px 0;
                border-bottom: 1px solid #f3f4f6;
            }

            .pathway-item:last-child {
                border-bottom: none;
            }

            .pathway-name {
                flex: 1;
                font-size: 0.85rem;
                font-weight: 500;
                color: #374151;
                line-height: 1.3;
            }

            .pathway-dots {
                font-size: 0.85rem;
                color: #10b981;
                letter-spacing: 2px;
                font-weight: 600;
            }

            .pathway-genes {
                width: 35px;
                font-size: 0.8rem;
                font-weight: 600;
                color: #6366f1;
                text-align: right;
                background: #eef2ff;
                padding: 2px 6px;
                border-radius: 4px;
            }

            /* Gene Status Cards */
            .gene-status-cards {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }

            .gene-status-cards h2 {
                font-size: 1.5rem;
                margin-bottom: 24px;
            }

            .cards-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 20px;
            }

            .gene-status-card {
                background: var(--gray-50);
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--gray-200);
                transition: all 0.2s;
            }

            .gene-status-card:hover {
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }

            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 20px;
                background: white;
                border-bottom: 1px solid var(--gray-200);
            }

            .gene-info { display: flex; align-items: center; gap: 10px; }

            .gene-symbol {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--primary);
            }

            .gene-rank {
                font-size: 0.75rem;
                color: var(--gray-500);
            }

            .card-body { padding: 20px; }

            .stat-row {
                display: flex;
                align-items: center;
                margin-bottom: 12px;
            }

            .stat-label {
                width: 80px;
                font-size: 0.8rem;
                color: var(--gray-500);
            }

            .stat-bar-container {
                flex: 1;
                height: 8px;
                background: var(--gray-200);
                border-radius: 4px;
                margin: 0 10px;
                overflow: hidden;
            }

            .stat-bar {
                height: 100%;
                border-radius: 4px;
            }

            .stat-bar.up { background: var(--danger); }
            .stat-bar.down { background: var(--primary); }

            .stat-value {
                font-size: 0.8rem;
                color: var(--gray-700);
            }

            .db-tags { display: flex; gap: 6px; flex-wrap: wrap; }

            .db-tag {
                background: var(--primary);
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.7rem;
            }

            .cancer-match {
                background: var(--success);
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.7rem;
            }

            .confidence-dots {
                font-size: 0.9rem;
            }

            /* RAG Interpretation Styles */
            .rag-interpretation {
                margin-top: 15px;
                padding: 12px;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-radius: 8px;
                border-left: 3px solid var(--primary);
            }

            .rag-label {
                font-size: 0.75rem;
                font-weight: 600;
                color: var(--primary);
                display: block;
                margin-bottom: 6px;
            }

            .rag-text {
                font-size: 0.8rem;
                color: var(--gray-700);
                line-height: 1.5;
                margin: 0 0 8px 0;
            }

            .rag-citations {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .pmid-link {
                font-size: 0.7rem;
                color: var(--primary);
                background: white;
                padding: 2px 8px;
                border-radius: 4px;
                text-decoration: none;
                border: 1px solid var(--primary);
            }

            .pmid-link:hover {
                background: var(--primary);
                color: white;
            }

            .citation-count {
                font-size: 0.7rem;
                color: var(--gray-500);
            }

            .gene-status-card.has-rag {
                border-color: var(--primary);
                border-width: 2px;
            }

            .tag.rag-tag {
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
            }

            /* RAG Summary Section */
            .rag-summary {
                background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 24px;
                color: white;
            }

            .rag-summary-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 24px;
                flex-wrap: wrap;
                gap: 20px;
            }

            .rag-title-section h2 {
                font-size: 1.5rem;
                margin-bottom: 6px;
                color: white;
            }

            .rag-subtitle {
                font-size: 0.9rem;
                color: rgba(255,255,255,0.7);
            }

            .rag-stats {
                display: flex;
                gap: 24px;
            }

            .rag-stat {
                text-align: center;
            }

            .rag-stat-value {
                display: block;
                font-size: 1.8rem;
                font-weight: 700;
                color: #a5b4fc;
            }

            .rag-stat-label {
                font-size: 0.75rem;
                color: rgba(255,255,255,0.6);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .rag-method-note {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 24px;
            }

            .method-icon {
                font-size: 1.5rem;
            }

            .method-text {
                font-size: 0.85rem;
                color: rgba(255,255,255,0.85);
                line-height: 1.6;
            }

            .method-text strong {
                color: #c7d2fe;
            }

            .rag-genes-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 16px;
                margin-bottom: 20px;
            }

            .rag-gene-card {
                background: rgba(255,255,255,0.08);
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.1);
                transition: all 0.2s;
            }

            .rag-gene-card:hover {
                background: rgba(255,255,255,0.12);
                transform: translateY(-2px);
            }

            .rag-gene-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 14px 16px;
                background: rgba(0,0,0,0.2);
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }

            .rag-gene-title {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .rag-gene-symbol {
                font-size: 1.1rem;
                font-weight: 700;
                color: #c7d2fe;
            }

            .rag-gene-fc {
                font-size: 0.85rem;
                padding: 2px 8px;
                border-radius: 4px;
            }

            .rag-gene-fc.up {
                background: rgba(239,68,68,0.3);
                color: #fca5a5;
            }

            .rag-gene-fc.down {
                background: rgba(59,130,246,0.3);
                color: #93c5fd;
            }

            .hub-indicator {
                background: linear-gradient(135deg, #f472b6, #fb7185);
                color: white;
                font-size: 0.65rem;
                padding: 2px 6px;
                border-radius: 4px;
                font-weight: 600;
            }

            .rag-confidence {
                font-size: 0.7rem;
                padding: 4px 10px;
                border-radius: 12px;
                font-weight: 600;
            }

            .rag-confidence.high {
                background: rgba(16,185,129,0.3);
                color: #6ee7b7;
            }

            .rag-confidence.medium {
                background: rgba(245,158,11,0.3);
                color: #fcd34d;
            }

            .rag-confidence.low {
                background: rgba(107,114,128,0.3);
                color: #d1d5db;
            }

            .rag-gene-body {
                padding: 16px;
            }

            .rag-interpretation-text {
                font-size: 0.85rem;
                color: rgba(255,255,255,0.85);
                line-height: 1.6;
                margin: 0 0 12px 0;
            }

            .rag-pmids {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .pmid-chip {
                font-size: 0.7rem;
                padding: 4px 10px;
                background: rgba(99,102,241,0.4);
                color: #c7d2fe;
                border-radius: 12px;
                text-decoration: none;
                transition: all 0.2s;
            }

            .pmid-chip:hover {
                background: rgba(99,102,241,0.7);
                color: white;
            }

            .no-pmid {
                font-size: 0.75rem;
                color: rgba(255,255,255,0.4);
                font-style: italic;
            }

            .rag-disclaimer {
                display: flex;
                align-items: center;
                gap: 10px;
                background: rgba(245,158,11,0.15);
                border: 1px solid rgba(245,158,11,0.3);
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 0.8rem;
                color: #fcd34d;
            }

            .disclaimer-icon {
                font-size: 1rem;
            }

            .card-footer {
                padding: 12px 20px;
                background: white;
                border-top: 1px solid var(--gray-200);
            }

            .tags { display: flex; gap: 6px; flex-wrap: wrap; }

            .tag {
                background: var(--gray-200);
                color: var(--gray-700);
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.7rem;
            }

            /* Detailed Table */
            .detailed-findings {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }

            .detailed-findings h2 {
                font-size: 1.5rem;
                margin-bottom: 24px;
            }

            .table-controls {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                gap: 20px;
            }

            .search-input {
                flex: 1;
                max-width: 300px;
                padding: 10px 16px;
                border: 2px solid var(--gray-200);
                border-radius: 8px;
                font-size: 0.9rem;
                transition: border-color 0.2s;
            }

            .search-input:focus {
                outline: none;
                border-color: var(--primary);
            }

            .filter-buttons { display: flex; gap: 8px; }

            .filter-btn {
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                background: var(--gray-100);
                color: var(--gray-700);
                font-size: 0.85rem;
                cursor: pointer;
                transition: all 0.2s;
            }

            .filter-btn:hover, .filter-btn.active {
                background: var(--primary);
                color: white;
            }

            .table-container {
                max-height: 500px;
                overflow-y: auto;
                border-radius: 8px;
                border: 1px solid var(--gray-200);
            }

            #gene-table {
                width: 100%;
                border-collapse: collapse;
            }

            #gene-table th {
                background: var(--gray-50);
                padding: 14px 16px;
                text-align: left;
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--gray-700);
                position: sticky;
                top: 0;
                cursor: pointer;
                user-select: none;
            }

            #gene-table th:hover { background: var(--gray-100); }

            #gene-table td {
                padding: 12px 16px;
                font-size: 0.85rem;
                border-bottom: 1px solid var(--gray-100);
            }

            #gene-table tr:hover { background: var(--gray-50); }

            #gene-table td.up { color: var(--danger); }
            #gene-table td.down { color: var(--primary); }

            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 500;
            }

            .badge.high { background: #D1FAE5; color: #065F46; }
            .badge.medium { background: #FEF3C7; color: #92400E; }
            .badge.low { background: #FEE2E2; color: #991B1B; }
            .badge.novel_candidate { background: #EDE9FE; color: #5B21B6; }
            .badge.requires_validation { background: var(--gray-100); color: var(--gray-500); }

            .table-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 16px;
                padding-top: 16px;
                border-top: 1px solid var(--gray-200);
            }

            .table-footer span { font-size: 0.85rem; color: var(--gray-500); }

            .download-btn {
                padding: 8px 16px;
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 0.85rem;
                cursor: pointer;
                transition: background 0.2s;
            }

            .download-btn:hover { background: var(--primary-light); }

            /* Methods Section */
            .methods-section {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }

            .methods-section h2 {
                font-size: 1.5rem;
                margin-bottom: 24px;
            }

            .methods-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 24px;
            }

            .method-card {
                background: var(--gray-50);
                border-radius: 8px;
                padding: 20px;
            }

            .method-card h4 {
                font-size: 0.95rem;
                margin-bottom: 12px;
                color: var(--gray-900);
            }

            .method-card ul {
                list-style: none;
                font-size: 0.85rem;
                color: var(--gray-600);
            }

            .method-card li {
                padding: 4px 0;
            }

            .confidence-explanation {
                background: var(--gray-50);
                border-radius: 8px;
                padding: 20px;
            }

            .confidence-explanation h4 {
                margin-bottom: 12px;
            }

            .score-table {
                width: 100%;
                margin-bottom: 16px;
            }

            .score-table td {
                padding: 8px;
                font-size: 0.85rem;
            }

            .score-table td:last-child {
                text-align: right;
                font-weight: 600;
                color: var(--primary);
            }

            .confidence-legend {
                display: flex;
                flex-wrap: wrap;
                gap: 16px;
                font-size: 0.8rem;
                color: var(--gray-600);
            }

            /* Footer */
            footer {
                text-align: center;
                padding: 40px 20px;
                color: var(--gray-500);
                font-size: 0.85rem;
            }

            .no-data {
                color: var(--gray-400);
                font-style: italic;
                padding: 20px;
                text-align: center;
            }

            @media (max-width: 768px) {
                .key-metrics { grid-template-columns: 1fr; }
                .dashboard-grid { grid-template-columns: 1fr; }
                .dashboard-panel.main-plot { grid-row: auto; }
                .cards-grid { grid-template-columns: 1fr; }
                .table-controls { flex-direction: column; align-items: stretch; }
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
                const buttons = document.querySelectorAll('.view-toggle .toggle-btn');

                if (view === 'interactive') {{
                    interactiveView.classList.add('active');
                    staticView.classList.remove('active');
                    buttons[0].classList.add('active');
                    buttons[1].classList.remove('active');
                }} else {{
                    interactiveView.classList.remove('active');
                    staticView.classList.add('active');
                    buttons[0].classList.remove('active');
                    buttons[1].classList.add('active');
                }}
            }}
        </script>
        '''

    def _generate_html(self, data: Dict) -> str:
        """Generate complete HTML report."""
        interpretation = data.get('interpretation_report', {})
        cancer_type = interpretation.get('cancer_type', self.config.get('cancer_type', 'Unknown'))

        return f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config["report_title"]}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    {self._generate_css()}
</head>
<body>
    <header>
        <div class="container">
            <h1>🧬 {self.config["report_title"]}</h1>
            <p class="meta">
                Generated by {self.config["author"]} |
                {datetime.now().strftime("%Y-%m-%d %H:%M")} |
                Cancer Type: {cancer_type}
            </p>
        </div>
    </header>

    <nav class="nav-pills">
        <a href="#executive-summary" class="nav-pill active">Executive Summary</a>
        <a href="#visual-dashboard" class="nav-pill">Visual Dashboard</a>
        <a href="#rag-summary" class="nav-pill">📚 RAG Interpretation</a>
        <a href="#gene-cards" class="nav-pill">Gene Cards</a>
        <a href="#detailed-table" class="nav-pill">Detailed Analysis</a>
        <a href="#methods" class="nav-pill">Methods</a>
    </nav>

    <div class="container">
        {self._generate_executive_summary_html(data)}
        {self._generate_visual_dashboard_html(data)}
        {self._generate_rag_summary_html(data)}
        {self._generate_gene_status_cards_html(data)}
        {self._generate_detailed_table_html(data)}
        {self._generate_methods_html() if self.config["include_methods"] else ""}
    </div>

    <footer>
        <p>Generated by BioInsight AI RNA-seq Pipeline v2.0</p>
        <p>This report is for research purposes only. All findings require experimental validation.</p>
    </footer>

    {self._generate_javascript(data)}
</body>
</html>
'''

    def run(self) -> Dict[str, Any]:
        """Generate the HTML report."""
        data = self._load_all_data()
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
