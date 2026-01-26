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
# Priority: Claude (more accurate, less hallucination) > OpenAI (fallback)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_rag_context_for_report(cancer_type: str, key_genes: list = None) -> str:
    """Get RAG context from VectorDB for grounded LLM interpretation.

    This ensures LLM responses are based on actual literature,
    preventing hallucination.
    """
    try:
        from rnaseq_pipeline.rag.gene_interpreter import GeneInterpreter
        interpreter = GeneInterpreter()

        context_parts = []

        # Get cancer-specific context
        if cancer_type and cancer_type.lower() not in ["unknown", ""]:
            cancer_query = f"{cancer_type} RNA-seq transcriptomics differential expression"
            cancer_results = interpreter.search_papers(cancer_query, top_k=3)
            if cancer_results:
                context_parts.append(f"## {cancer_type} 관련 문헌 근거:")
                for r in cancer_results[:3]:
                    title = r.get('title', 'Unknown')
                    pmid = r.get('pmid', '')
                    abstract = r.get('abstract', r.get('content', ''))[:400]
                    context_parts.append(f"- {title} [PMID: {pmid}]\n  {abstract}...")

        # Get gene-specific context for top genes
        if key_genes and len(key_genes) > 0:
            for gene in key_genes[:5]:
                if gene and not gene.startswith('ENSG'):
                    gene_results = interpreter.search_papers(f"{gene} cancer expression", top_k=2)
                    if gene_results:
                        context_parts.append(f"\n## {gene} 관련 근거:")
                        for r in gene_results[:2]:
                            title = r.get('title', 'Unknown')
                            pmid = r.get('pmid', '')
                            context_parts.append(f"- {title} [PMID: {pmid}]")

        return "\n".join(context_parts) if context_parts else ""

    except Exception:
        return ""


def call_llm_with_rag(prompt: str, cancer_type: str = None, key_genes: list = None,
                      max_tokens: int = 4000, logger=None,
                      use_opus: bool = False) -> Optional[str]:
    """Call LLM (Claude preferred) with RAG context for grounded responses.

    Args:
        prompt: Main prompt to send to LLM
        cancer_type: Cancer type for RAG context retrieval
        key_genes: Key genes for RAG context retrieval
        max_tokens: Maximum tokens for response
        logger: Logger instance for logging
        use_opus: If True, use Claude Opus 4 for highest quality writing (5x cost)

    Returns:
        LLM response text or None if failed
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Runtime availability check
    anthropic_available = False
    try:
        import anthropic as anthropic_module
        anthropic_available = True
    except ImportError:
        pass

    openai_available = False
    try:
        from openai import OpenAI as OpenAIClient
        openai_available = True
    except ImportError:
        pass

    # Prefer Claude for accuracy and less hallucination
    use_anthropic = anthropic_available and anthropic_key
    use_openai = openai_available and openai_key and not use_anthropic

    if not use_anthropic and not use_openai:
        if logger:
            logger.warning("No LLM API available (need ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        return None

    # Get RAG context for grounding
    rag_context = get_rag_context_for_report(cancer_type, key_genes)

    # Build system prompt for accurate, hallucination-free responses
    system_prompt = """당신은 RNA-seq 분석 결과를 해석하는 전문 바이오인포매틱스 연구자입니다.

중요한 지침:
1. 반드시 제공된 데이터와 문헌 근거만을 기반으로 해석하세요.
2. 확실하지 않은 내용은 "~일 가능성이 있다", "추가 검증이 필요하다"로 표현하세요.
3. 가능한 경우 PMID 인용을 포함하세요.
4. 절대로 데이터에 없는 정보를 추측하거나 만들어내지 마세요.
5. 임상적 결론이나 진단적 판단은 피하세요.
6. 모든 해석은 한국어로 작성하세요."""

    # Combine RAG context with prompt
    full_prompt = prompt
    if rag_context:
        full_prompt = f"""다음은 VectorDB에서 검색된 관련 문헌 정보입니다. 이 정보를 참고하여 해석하세요:

{rag_context}

---

{prompt}"""

    llm_provider = "Claude" if use_anthropic else "OpenAI"
    if logger:
        logger.info(f"Using {llm_provider} for LLM generation (RAG context: {'Yes' if rag_context else 'No'})")

    try:
        if use_anthropic:
            import anthropic as anthropic_module
            client = anthropic_module.Anthropic(api_key=anthropic_key)

            # Select model based on use_opus flag
            if use_opus:
                model_id = "claude-opus-4-20250514"
                model_name = "Claude Opus 4"
            else:
                model_id = "claude-sonnet-4-20250514"
                model_name = "Claude Sonnet 4"

            if logger:
                logger.info(f"Using {model_name} for generation")

            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text

        elif use_openai:
            from openai import OpenAI as OpenAIClient
            client = OpenAIClient(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content

    except Exception as e:
        if logger:
            logger.error(f"LLM API error ({llm_provider}): {e}")
        return None


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
            "db_matched_genes.csv",
            # Multi-omic integration files
            "integrated_drivers.csv",
            "confirmed_drivers.csv",
            "actionable_targets.csv",
            "driver_mutations.csv",
            "annotated_variants.csv"
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
            emoji = "[G]"
            label = "높음"
        elif score >= 2:
            level = "medium"
            emoji = "[Y]"
            label = "중간"
        else:
            level = "low"
            emoji = "[R]"
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
                    <h2>10. 문헌 기반 해석</h2>
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
                <span class="method-icon">[Lab]</span>
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
                <span class="disclaimer-icon">[!]</span>
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
                    <h2>핵심 요약</h2>
                    <span class="confidence-badge {conf_level}">{conf_emoji} 신뢰도: {conf_label}</span>
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
                    <div class="metric-label">최상위 유전자</div>
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
                <span class="warning-icon">[!]</span>
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
                '<p class="panel-note">[Tip] 마우스 드래그로 회전, 스크롤로 확대/축소, 유전자 클릭으로 포커스</p>'
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
            <div class="ai-header">[AI] AI 해석</div>
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
            <div class="ai-header">[AI] AI 해석</div>
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
                <div class="ai-header">[AI] AI 해석</div>
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
                        <p class="panel-note">[Tip] 마우스 드래그로 확대, 유전자 위에 마우스를 올리면 상세 정보 표시</p>
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
                    <p class="panel-note">[!] 발현 변화량 기준 정렬 (생물학적 중요도와 다를 수 있음)</p>
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
            conf_dots_html = "[G]" * conf_dots + "[O]" * (5 - conf_dots)

            # Expression bar (relative to max)
            expr_width = min(100, int(abs(log2fc) / 5 * 100))

            # RAG section HTML
            rag_section = ""
            if has_rag:
                rag_preview = rag_text[:200] + "..." if len(rag_text) > 200 else rag_text
                pmid_links = ' '.join([f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" target="_blank" class="pmid-link">PMID:{pmid}</a>' for pmid in rag_pmids[:3]])
                rag_section = f'''
                    <div class="rag-interpretation">
                        <span class="rag-label">[Ref] Literature Insight</span>
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
                            {'<span class="cancer-match">[OK] Cancer Match</span>' if cancer_match else ''}
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
                        {'<span class="tag rag-tag">[Ref] RAG</span>' if has_rag else ''}
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
            <h2>전체 DEG 유전자 목록</h2>

            <div class="section-description">
                <p>DESeq2 분석에서 통계적으로 유의한 발현 변화를 보인 모든 유전자입니다 (padj &lt; 0.05).</p>
            </div>

            <!-- 점수 체계 설명 -->
            <div class="score-explanation">
                <div class="score-header">
                    <span class="score-icon">[Chart]</span>
                    <span class="score-title">종합 점수 산정 기준 (총 6.5점)</span>
                </div>
                <div class="score-grid">
                    <div class="score-item">
                        <span class="score-label">Hub 유전자</span>
                        <span class="score-value">+2.0</span>
                        <span class="score-desc">네트워크 중심 유전자</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">DB 매칭</span>
                        <span class="score-value">+2.0</span>
                        <span class="score-desc">COSMIC/OncoKB 등록</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">암종 일치</span>
                        <span class="score-value">+1.5</span>
                        <span class="score-desc">해당 암종 연관 기록</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">Pathway 연관</span>
                        <span class="score-value">+0.5</span>
                        <span class="score-desc">3개 이상 경로 포함</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">발현 방향</span>
                        <span class="score-value">+0.5</span>
                        <span class="score-desc">문헌과 방향 일치</span>
                    </div>
                </div>
                <div class="confidence-legend">
                    <span class="legend-title">신뢰도 등급:</span>
                    <span class="badge high">HIGH</span> DB 매칭 + 5점 이상
                    <span class="badge medium">MEDIUM</span> DB 매칭 + 3점 이상
                    <span class="badge novel_candidate">CANDIDATE</span> Hub 유전자 (DB 미등록)
                    <span class="badge low">LOW</span> 1.5점 이상
                    <span class="badge requires_validation">REQUIRES VALIDATION</span> 추가 검증 필요
                </div>
            </div>

            <div class="table-controls">
                <input type="text" id="gene-search" class="search-input"
                       placeholder="[Search] 유전자 검색..." onkeyup="filterTable()">
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="filterByConfidence('all')">전체</button>
                    <button class="filter-btn" onclick="filterByConfidence('high')">High</button>
                    <button class="filter-btn" onclick="filterByConfidence('medium')">Medium</button>
                    <button class="filter-btn" onclick="filterByConfidence('novel_candidate')">Candidate</button>
                </div>
            </div>

            <div class="table-container">
                <table id="gene-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">유전자 ↕</th>
                            <th onclick="sortTable(1)">Log2FC ↕</th>
                            <th onclick="sortTable(2)">P-adj ↕</th>
                            <th onclick="sortTable(3)">Hub ↕</th>
                            <th onclick="sortTable(4)">DB 매칭 ↕</th>
                            <th onclick="sortTable(5)">신뢰도</th>
                            <th onclick="sortTable(6)">점수 ↕</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>

            <div class="table-footer">
                <span>총 {len(integrated):,}개 유전자 (상위 {self.config['max_table_rows']}개 표시)</span>
                <button class="download-btn" onclick="downloadCSV()">[Download] CSV 다운로드</button>
            </div>
        </section>
        '''

    def _get_driver_disclaimer_html(self) -> str:
        """Return driver disclaimer HTML (extracted for Python 3.11 compatibility)."""
        return '''
            <div class="driver-disclaimer">
                <span class="disclaimer-icon">[!]</span>
                <div class="disclaimer-text">
                    <strong>주의사항:</strong> RNA-seq 데이터만으로는 Driver 유전자를 확정할 수 없습니다.
                    후보 조절자는 "확정된 driver"가 아닌 "추가 검증이 필요한 후보"입니다.
                    실제 돌연변이 확인을 위해서는 WES/WGS 또는 Targeted NGS가 필요합니다.
                </div>
            </div>
            '''

    def _generate_driver_analysis_html(self, data: Dict) -> str:
        """Generate Driver Gene Analysis section (Known + Candidate Regulator tracks).

        Now includes Multi-omic Integration section when WGS/WES data is available.
        """
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        driver_summary = data.get('driver_summary', {})

        # Multi-omic integration data (from WGS/WES pipeline)
        integrated_drivers_df = data.get('integrated_drivers_df')
        confirmed_drivers_df = data.get('confirmed_drivers_df')
        actionable_targets_df = data.get('actionable_targets_df')
        driver_mutations_df = data.get('driver_mutations_df')

        has_multiomic = integrated_drivers_df is not None and len(integrated_drivers_df) > 0

        if not driver_known and not driver_novel and not has_multiomic:
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
            gene_function = driver.get('gene_function', '')

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

            # Gene function description
            function_html = ""
            if gene_function:
                function_html = f'<div class="gene-function"><span class="function-icon">[Book]</span><span class="function-text">{gene_function}</span></div>'

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
                    {function_html}
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
                        <span class="validation-icon">[Test]</span>
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
            gene_function = driver.get('gene_function', '')

            # Score badge color
            if score >= 70:
                score_class = "high"
            elif score >= 50:
                score_class = "medium"
            else:
                score_class = "low"

            # Gene function description
            function_html = ""
            if gene_function:
                function_html = f'<div class="gene-function"><span class="function-icon">[Book]</span><span class="function-text">{gene_function}</span></div>'

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
                    {function_html}
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
                        <span class="validation-icon">[Lab]</span>
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
                <h2>5. Driver 유전자 분석</h2>
                <p class="driver-subtitle">RNA-seq 발현 패턴 + TCGA 돌연변이 데이터 기반 Driver 예측</p>
            </div>

            <div class="driver-summary-stats">
                <div class="driver-stat known-stat">
                    <span class="stat-value">{total_known}</span>
                    <span class="stat-label">알려진 Driver</span>
                    <span class="stat-detail">높은 신뢰도 {high_conf_known}개</span>
                </div>
                <div class="driver-stat novel-stat">
                    <span class="stat-value">{total_novel}</span>
                    <span class="stat-label">후보 조절자</span>
                    <span class="stat-detail">높은 신뢰도 {high_conf_novel}개</span>
                </div>
                <div class="driver-stat actionable-stat">
                    <span class="stat-value">{len(actionable)}</span>
                    <span class="stat-label">치료 표적</span>
                    <span class="stat-detail">{', '.join(actionable[:3]) if actionable else '없음'}</span>
                </div>
            </div>

            <div class="driver-method-note">
                <span class="method-icon">[Chart]</span>
                <div class="method-text">
                    <strong>알려진 Driver Track:</strong> COSMIC Cancer Gene Census + TCGA 돌연변이 빈도 + 발현 변화량 기반 scoring<br>
                    <strong>후보 조절자 Track:</strong> Hub Gene 점수 + 발현 변화량 + Pathway 영향력 + 문헌 지지도 기반 scoring
                </div>
            </div>

            <div class="driver-tracks">
                <div class="driver-track known-track">
                    <h3>[Trophy] 알려진 Driver Track</h3>
                    <p class="track-desc">COSMIC/OncoKB에서 검증된 암 드라이버 유전자. 타겟 치료제 개발 후보.</p>
                    <div class="driver-cards-grid">
                        {known_cards_html if known_cards_html else '<p class="no-data">DEG 목록에서 알려진 driver가 발견되지 않음</p>'}
                    </div>
                </div>

                <div class="driver-track novel-track">
                    <h3>[Lab] 후보 조절자 Track</h3>
                    <p class="track-desc">네트워크 분석 기반 핵심 조절인자 후보. 문헌 검증 및 기능 연구가 필요한 유전자.</p>
                    <div class="driver-cards-grid">
                        {novel_cards_html if novel_cards_html else '<p class="no-data">후보 조절자가 발견되지 않음</p>'}
                    </div>
                </div>
            </div>

            {self._generate_multiomic_section_html(data) if has_multiomic else self._get_driver_disclaimer_html()}
        </section>
        '''

    def _generate_multiomic_section_html(self, data: Dict) -> str:
        """Generate Multi-omic Integration section (WGS/WES + RNA-seq).

        This section shows CONFIRMED driver genes with both mutation AND expression evidence.
        """
        integrated_drivers_df = data.get('integrated_drivers_df')
        confirmed_drivers_df = data.get('confirmed_drivers_df')
        actionable_targets_df = data.get('actionable_targets_df')
        driver_mutations_df = data.get('driver_mutations_df')

        if integrated_drivers_df is None or len(integrated_drivers_df) == 0:
            return ""

        # Count by classification
        n_confirmed = len(integrated_drivers_df[integrated_drivers_df['classification'] == 'confirmed_driver']) if 'classification' in integrated_drivers_df.columns else 0
        n_high_conf = len(integrated_drivers_df[integrated_drivers_df['classification'] == 'high_confidence']) if 'classification' in integrated_drivers_df.columns else 0
        n_candidates = len(integrated_drivers_df[integrated_drivers_df['classification'] == 'candidate']) if 'classification' in integrated_drivers_df.columns else 0
        n_actionable = len(actionable_targets_df) if actionable_targets_df is not None else 0
        n_mutations = len(driver_mutations_df) if driver_mutations_df is not None else 0

        # Count validated genes
        n_validated = 0
        n_hotspot_validated = 0
        if 'db_validated' in integrated_drivers_df.columns:
            n_validated = int(integrated_drivers_df['db_validated'].sum())
        if 'hotspot_validated' in integrated_drivers_df.columns:
            n_hotspot_validated = int(integrated_drivers_df['hotspot_validated'].sum())

        # Confirmed drivers cards
        confirmed_cards_html = ""
        if confirmed_drivers_df is not None and len(confirmed_drivers_df) > 0:
            for idx, row in confirmed_drivers_df.head(10).iterrows():
                gene = row.get('gene_symbol', 'Unknown')
                score = row.get('confidence_score', 0)
                log2fc = row.get('log2fc', 0)
                direction = "↑" if log2fc > 0 else "↓"
                dir_class = "up" if log2fc > 0 else "down"
                mutation_score = row.get('mutation_driver_score', 0)
                is_hotspot = row.get('is_hotspot', False)
                drugs = row.get('actionable_drugs', '')

                # Validation status
                db_validated = row.get('db_validated', False)
                hotspot_validated = row.get('hotspot_validated', False)
                drug_validated = row.get('drug_validated', False)
                validation_sources = row.get('validation_sources', [])
                oncokb_level = row.get('oncokb_level', '')
                cosmic_tier = row.get('cosmic_tier', '')

                # Parse validation_sources if string
                if isinstance(validation_sources, str):
                    validation_sources = [s.strip() for s in validation_sources.split(',') if s.strip()]

                # Build validation badges
                validation_badges = ""
                if db_validated:
                    if hotspot_validated:
                        validation_badges += f'<span class="validation-badge validated">[OK] Hotspot 검증됨</span>'
                    if cosmic_tier:
                        validation_badges += f'<span class="validation-badge cosmic">COSMIC {cosmic_tier}</span>'
                    if oncokb_level:
                        validation_badges += f'<span class="validation-badge oncokb">OncoKB Lv{oncokb_level}</span>'
                    if drug_validated:
                        validation_badges += f'<span class="validation-badge dgidb">DGIdb [OK]</span>'
                else:
                    validation_badges = '<span class="validation-badge unvalidated">[!] 외부 검증 필요</span>'

                confirmed_cards_html += f'''
                <div class="driver-card confirmed-driver {'validated' if db_validated else 'needs-validation'}">
                    <div class="driver-header">
                        <div class="driver-title">
                            <span class="driver-gene">{gene}</span>
                            <span class="confirmed-badge">[OK] CONFIRMED</span>
                            {'<span class="hotspot-badge">[Hot] Hotspot</span>' if is_hotspot else ''}
                        </div>
                        <span class="driver-score high">{score:.0f}/100</span>
                    </div>
                    <div class="validation-row">
                        {validation_badges}
                    </div>
                    <div class="driver-body">
                        <div class="evidence-grid">
                            <div class="evidence-item mutation">
                                <span class="evidence-icon">[DNA]</span>
                                <span class="evidence-label">Mutation</span>
                                <span class="evidence-value">{mutation_score:.0f}점</span>
                            </div>
                            <div class="evidence-item expression">
                                <span class="evidence-icon">[Chart]</span>
                                <span class="evidence-label">Expression</span>
                                <span class="evidence-value {dir_class}">{direction} {abs(log2fc):.2f}</span>
                            </div>
                        </div>
                        {f'<div class="drug-info"><span class="drug-icon">[Drug]</span><span class="drug-list">{drugs}</span></div>' if drugs else ''}
                    </div>
                </div>
                '''

        # Actionable targets table
        actionable_table_html = ""
        if actionable_targets_df is not None and len(actionable_targets_df) > 0:
            rows_html = ""
            for idx, row in actionable_targets_df.head(10).iterrows():
                gene = row.get('gene_symbol', 'Unknown')
                classification = row.get('classification', '')
                drugs = row.get('actionable_drugs', '')
                score = row.get('confidence_score', 0)
                drug_validated = row.get('drug_validated', False)
                db_validated = row.get('db_validated', False)

                class_badge = "confirmed" if "confirmed" in classification else "candidate"
                validation_icon = "[OK]" if drug_validated else ("△" if db_validated else "[!]")
                validation_class = "validated" if drug_validated else ("partial" if db_validated else "unvalidated")

                rows_html += f'''
                <tr class="{validation_class}">
                    <td><strong>{gene}</strong></td>
                    <td><span class="class-badge {class_badge}">{classification}</span></td>
                    <td>{score:.0f}</td>
                    <td class="drug-cell">{drugs}</td>
                    <td class="validation-cell"><span class="validation-icon {validation_class}">{validation_icon}</span></td>
                </tr>
                '''

            actionable_table_html = f'''
            <div class="actionable-targets">
                <h4>[Drug] Actionable 치료 표적</h4>
                <p class="table-note">[OK] = DGIdb 검증됨, △ = DB 부분 검증, [!] = 검증 필요</p>
                <table class="actionable-table">
                    <thead>
                        <tr>
                            <th>유전자</th>
                            <th>분류</th>
                            <th>점수</th>
                            <th>표적 약물</th>
                            <th>검증</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            '''

        # Mutation summary
        mutation_summary_html = ""
        if driver_mutations_df is not None and len(driver_mutations_df) > 0:
            mutation_list = ""
            for idx, row in driver_mutations_df.head(8).iterrows():
                gene = row.get('gene', 'Unknown')
                aa_change = row.get('amino_acid_change', '')
                vaf = row.get('vaf', 0)
                is_hotspot = row.get('is_hotspot', False)

                mutation_list += f'''
                <div class="mutation-item {'hotspot' if is_hotspot else ''}">
                    <span class="mutation-gene">{gene}</span>
                    <span class="mutation-change">{aa_change}</span>
                    <span class="mutation-vaf">VAF: {vaf:.1%}</span>
                    {'<span class="hotspot-marker">[Hot]</span>' if is_hotspot else ''}
                </div>
                '''

            mutation_summary_html = f'''
            <div class="mutation-summary">
                <h4>[Lab] 검출된 Driver Mutations</h4>
                <div class="mutation-list">
                    {mutation_list}
                </div>
            </div>
            '''

        return f'''
        <div class="multiomic-integration">
            <div class="multiomic-header">
                <h3>[DNA] Multi-omic 통합 분석 (RNA-seq + WGS/WES)</h3>
                <p class="multiomic-subtitle">
                    <span class="highlight">[OK] 실제 변이 데이터 기반</span> -
                    RNA-seq 발현 변화와 WGS/WES 변이 데이터를 통합하여
                    <strong>확정된 Driver 유전자</strong>를 식별했습니다.
                    <span class="validation-highlight">외부 DB 검증 완료: {n_validated}개</span>
                </p>
            </div>

            <div class="multiomic-stats">
                <div class="stat-card confirmed">
                    <span class="stat-icon">[OK]</span>
                    <span class="stat-value">{n_confirmed}</span>
                    <span class="stat-label">Confirmed Driver</span>
                    <span class="stat-desc">변이 + 발현 + DB검증</span>
                </div>
                <div class="stat-card high-conf">
                    <span class="stat-icon">★</span>
                    <span class="stat-value">{n_high_conf}</span>
                    <span class="stat-label">High Confidence</span>
                    <span class="stat-desc">강한 증거 1개 이상</span>
                </div>
                <div class="stat-card validated">
                    <span class="stat-icon">[Lab]</span>
                    <span class="stat-value">{n_validated}</span>
                    <span class="stat-label">DB 검증됨</span>
                    <span class="stat-desc">COSMIC/OncoKB/DGIdb</span>
                </div>
                <div class="stat-card actionable">
                    <span class="stat-icon">[Drug]</span>
                    <span class="stat-value">{n_actionable}</span>
                    <span class="stat-label">Actionable</span>
                    <span class="stat-desc">표적 약물 존재</span>
                </div>
            </div>

            <div class="validation-info-box">
                <h5>[Search] 외부 데이터베이스 검증</h5>
                <p>Driver 유전자와 약물 정보는 다음 데이터베이스에서 검증됩니다:</p>
                <ul>
                    <li><strong>COSMIC</strong>: 암 체세포 변이 데이터베이스 (Tier 1 암 유전자)</li>
                    <li><strong>OncoKB</strong>: 정밀 종양학 지식 베이스 (Level 1-4 근거 수준)</li>
                    <li><strong>DGIdb</strong>: 약물-유전자 상호작용 데이터베이스 (FDA 승인 약물)</li>
                    <li><strong>ClinVar</strong>: 임상 변이 해석 데이터베이스</li>
                </ul>
                <p class="validation-note">[!] "Confirmed Driver"는 반드시 외부 DB 검증이 필요합니다. 검증되지 않은 경우 "High Confidence"로 분류됩니다.</p>
            </div>

            <div class="confirmed-drivers-section">
                <h4>✅ Confirmed Driver 유전자</h4>
                <p class="section-desc">변이(Mutation), 발현(Expression), 그리고 <strong>외부 DB 검증</strong>이 모두 확인된 Driver 유전자입니다.</p>
                <div class="confirmed-cards-grid">
                    {confirmed_cards_html if confirmed_cards_html else '<p class="no-data">외부 DB 검증이 완료된 Confirmed driver가 없습니다. High Confidence 섹션을 확인하세요.</p>'}
                </div>
            </div>

            {mutation_summary_html}

            {actionable_table_html}

            <div class="multiomic-note">
                <span class="note-icon">[Info]</span>
                <div class="note-text">
                    <strong>Multi-omic + DB 검증의 중요성:</strong>
                    <ul>
                        <li>RNA-seq만: Driver 유전자를 "예측"만 가능</li>
                        <li>RNA-seq + WGS/WES: 체세포 변이 "확인" 가능</li>
                        <li><strong>RNA-seq + WGS/WES + 외부 DB 검증</strong>: 임상적으로 의미있는 Driver "확정"</li>
                    </ul>
                    Hotspot 변이는 COSMIC/OncoKB에서, 표적 약물은 DGIdb에서 검증되어야 신뢰할 수 있습니다.
                </div>
            </div>
        </div>
        '''

    def _generate_study_overview_html(self, data: Dict) -> str:
        """Generate Study Overview section with ML-predicted cancer type."""
        deg_df = data.get('deg_significant_df')

        # Get cancer prediction from ML model
        cancer_prediction = data.get('cancer_prediction', {})

        # Determine cancer type display
        if cancer_prediction:
            predicted_cancer = cancer_prediction.get('predicted_cancer', 'Unknown')
            cancer_korean = cancer_prediction.get('predicted_cancer_korean', cancer_prediction.get('cancer_korean', ''))
            confidence = cancer_prediction.get('confidence', 0)
            agreement_ratio = cancer_prediction.get('agreement_ratio', 0)

            # Check if this was a user-specified validation case
            user_specified = cancer_prediction.get('user_specified_cancer')
            ml_predicted = cancer_prediction.get('ml_predicted_cancer')
            prediction_matches = cancer_prediction.get('prediction_matches_user')

            if user_specified:
                # User specified + ML validation case
                cancer_type_display = self.config.get('cancer_type_korean', user_specified)

                # Confidence badge based on ML match
                if prediction_matches:
                    confidence_badge = f'<span class="confidence-badge high">✅ ML 검증 일치 ({confidence:.1%})</span>'
                    prediction_method = "사용자 지정 + ML 검증"
                    prediction_note = f"<small>ML 예측: {ml_predicted} (샘플 일치율: {agreement_ratio:.1%})</small>"
                else:
                    confidence_badge = f'<span class="confidence-badge low">[!] ML 불일치</span>'
                    prediction_method = "사용자 지정 (ML 검증 불일치)"
                    ml_korean = cancer_prediction.get('predicted_cancer_korean', ml_predicted)
                    prediction_note = f"<small>[!] ML 예측: {ml_predicted} ({ml_korean}) - 신뢰도: {confidence:.1%}</small>"
            else:
                # ML prediction only (no user specification)
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

                prediction_method = "[AI] ML 예측 (Pan-Cancer Classifier)"
                prediction_note = f"<small>샘플 일치율: {agreement_ratio:.1%}</small>"
        else:
            # Fallback to config-specified cancer type (no ML prediction available)
            cancer_type_display = self.config.get('cancer_type_korean', self.config.get('cancer_type', 'Unknown'))
            if cancer_type_display.lower() == 'unknown':
                cancer_type_display = '암종 미확인'
                confidence_badge = '<span class="confidence-badge low">예측 불가</span>'
                prediction_method = "[!] ML 예측 실패"
                prediction_note = "<small>count matrix 확인 필요</small>"
            else:
                confidence_badge = '<span class="confidence-badge medium">사용자 지정</span>'
                prediction_method = "사용자 지정 (ML 검증 없음)"
                prediction_note = ""

        # Get sample info from config
        original_files = self.config.get('original_files', {})
        dataset_id = original_files.get('count_matrix', 'Unknown Dataset')

        # Get counts
        total_deg = len(deg_df) if deg_df is not None else 0
        up_count = len(deg_df[deg_df['log2FC'] > 0]) if deg_df is not None and 'log2FC' in deg_df.columns else 0
        down_count = len(deg_df[deg_df['log2FC'] < 0]) if deg_df is not None and 'log2FC' in deg_df.columns else 0

        contrast = self.config.get('contrast', ['tumor', 'normal'])

        # ★ ML 성능 지표 카드 생성 (v3)
        ml_performance_html = ""
        model_perf = cancer_prediction.get('model_performance', {}) if cancer_prediction else {}

        if model_perf:
            overall = model_perf.get('overall', {})
            per_class = model_perf.get('per_class', {})
            ci = model_perf.get('confidence_interval', {})

            # Build ML Performance Scorecard
            ml_performance_html = f'''
            <div class="ml-performance-card" style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;">
                <h4 style="margin: 0 0 12px 0; display: flex; align-items: center; gap: 8px;">
                    <span>[Chart]</span> Pan-Cancer Classifier 성능 지표
                </h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.15); border-radius: 8px;">
                        <div style="font-size: 1.3em; font-weight: bold;">{overall.get('accuracy', 0):.3f}</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">Accuracy</div>
                    </div>
                    <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.15); border-radius: 8px;">
                        <div style="font-size: 1.3em; font-weight: bold;">{overall.get('f1_macro', 0):.3f}</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">F1 (Macro)</div>
                    </div>
                    <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.15); border-radius: 8px;">
                        <div style="font-size: 1.3em; font-weight: bold;">{overall.get('mcc', 0):.3f}</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">MCC</div>
                    </div>
                    <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.15); border-radius: 8px;">
                        <div style="font-size: 1.3em; font-weight: bold;">{overall.get('pr_auc_macro', 0):.3f}</div>
                        <div style="font-size: 0.8em; opacity: 0.9;">PR-AUC</div>
                    </div>
                </div>'''

            # Per-class metrics if available
            if per_class:
                cancer_type = per_class.get('cancer_type', '')
                ml_performance_html += f'''
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.3);">
                    <div style="font-size: 0.9em; margin-bottom: 8px;">[Pin] <b>{cancer_type}</b> 분류 성능</div>
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; font-size: 0.85em;">
                        <div style="text-align: center;">
                            <div style="font-weight: bold;">{per_class.get('f1', 0):.3f}</div>
                            <div style="opacity: 0.8;">F1</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold;">{per_class.get('precision', 0):.3f}</div>
                            <div style="opacity: 0.8;">Precision</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold;">{per_class.get('recall', 0):.3f}</div>
                            <div style="opacity: 0.8;">Recall</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold;">{per_class.get('pr_auc', 0):.3f}</div>
                            <div style="opacity: 0.8;">PR-AUC</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold;">{per_class.get('roc_auc', 0):.3f}</div>
                            <div style="opacity: 0.8;">ROC-AUC</div>
                        </div>
                    </div>
                </div>'''

            # Confidence intervals if available
            if ci:
                acc_ci = ci.get('accuracy', {})
                f1_ci = ci.get('f1_macro', {})
                ml_performance_html += f'''
                <div style="margin-top: 10px; font-size: 0.75em; opacity: 0.85;">
                    95% CI: Accuracy [{acc_ci.get('lower', 0):.3f} - {acc_ci.get('upper', 0):.3f}],
                    F1 [{f1_ci.get('lower', 0):.3f} - {f1_ci.get('upper', 0):.3f}]
                </div>'''

            ml_performance_html += '''
            </div>'''

        return f'''
        <section class="study-overview-section" id="study-overview">
            <h2>1. 연구 개요</h2>

            <div class="overview-grid">
                <div class="overview-table">
                    <table class="info-table">
                        <tr><td><strong>데이터셋 ID</strong></td><td>{dataset_id}</td></tr>
                        <tr>
                            <td><strong>예측 암종</strong></td>
                            <td>
                                <span class="cancer-type-predicted">{cancer_type_display}</span> {confidence_badge}
                                <br/>{prediction_note}
                            </td>
                        </tr>
                        <tr><td><strong>예측 방법</strong></td><td>{prediction_method}</td></tr>
                        <tr><td><strong>비교 조건</strong></td><td>{contrast[0]} vs {contrast[1]}</td></tr>
                        <tr><td><strong>분석 파이프라인</strong></td><td>BioInsight AI v2.0</td></tr>
                        <tr><td><strong>분석 일자</strong></td><td>{datetime.now().strftime("%Y-%m-%d")}</td></tr>
                    </table>
                </div>

                <div class="deg-summary-box">
                    <h4>DEG 요약</h4>
                    <table class="info-table">
                        <tr><td>총 DEG 수</td><td><strong>{total_deg:,}</strong></td></tr>
                        <tr><td>상향 발현</td><td class="up-text">{up_count:,}</td></tr>
                        <tr><td>하향 발현</td><td class="down-text">{down_count:,}</td></tr>
                        <tr><td>기준값 (|log2FC|)</td><td>> 1.0</td></tr>
                        <tr><td>기준값 (padj)</td><td>< 0.05</td></tr>
                    </table>
                </div>
            </div>

            {ml_performance_html}
        </section>
        '''

    def _generate_brief_abstract_html(self, data: Dict) -> str:
        """Generate a comprehensive abstract section with driver and literature interpretations."""
        extended_abstract = data.get('abstract_extended', {})

        if not extended_abstract:
            return ''

        # Get key information
        title = extended_abstract.get('title', '')
        title_en = extended_abstract.get('title_en', '')
        key_findings = extended_abstract.get('key_findings', [])
        abstract_text = extended_abstract.get('abstract_extended', '')
        driver_interp = extended_abstract.get('driver_interpretation', '')
        rag_interp = extended_abstract.get('rag_interpretation', '')
        validation = extended_abstract.get('validation_priorities', {})

        # Use full abstract text (not just brief parts)
        formatted_abstract = ''
        if abstract_text:
            paragraphs = abstract_text.split('\n\n')
            formatted_abstract = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])

        # Format key findings as list with highlighted numbers
        findings_html = ''
        if key_findings:
            import re
            def highlight_numbers(text):
                """Highlight numbers and percentages in text."""
                # Highlight numbers with units (201개, 174개, 86.6% etc)
                text = re.sub(r'(\d+(?:,\d+)?(?:\.\d+)?)\s*(개|%|점|배)',
                             r'<strong class="num">\1</strong>\2', text)
                # Highlight standalone significant numbers
                text = re.sub(r'(\d+(?:\.\d+)?)\s*(log2FC|AUC|p값|p-value)',
                             r'<strong class="num">\1</strong> \2', text, flags=re.IGNORECASE)
                return text

            findings_items = ''.join([f'<li>{highlight_numbers(f)}</li>' for f in key_findings[:10]])
            findings_html = f'''
            <div class="key-findings-box">
                <h4>[Pin] 핵심 발견 <span class="findings-count">{len(key_findings)}건</span></h4>
                <ul>{findings_items}</ul>
            </div>
            '''

        # Driver Gene interpretation
        driver_html = ''
        if driver_interp:
            driver_html = f'''
            <div class="interpretation-box driver">
                <h4>[DNA] Driver Gene 연관성 분석</h4>
                <p>{driver_interp}</p>
            </div>
            '''

        # RAG Literature interpretation
        rag_html = ''
        if rag_interp:
            rag_html = f'''
            <div class="interpretation-box literature">
                <h4>[Ref] 문헌 기반 해석</h4>
                <p>{rag_interp}</p>
            </div>
            '''

        # Validation priorities
        validation_html = ''
        if validation:
            val_items = []
            if validation.get('qPCR'):
                val_items.append(f'<div class="val-item"><strong>qRT-PCR:</strong> {", ".join(validation["qPCR"][:5])}</div>')
            if validation.get('western_blot'):
                val_items.append(f'<div class="val-item"><strong>Western Blot:</strong> {", ".join(validation["western_blot"][:3])}</div>')
            if validation.get('functional_study'):
                val_items.append(f'<div class="val-item"><strong>Functional Study:</strong> {", ".join(validation["functional_study"][:3])}</div>')
            if validation.get('biomarker_candidates'):
                val_items.append(f'<div class="val-item"><strong>Biomarker 후보:</strong> {", ".join(validation["biomarker_candidates"][:5])}</div>')
            if val_items:
                validation_html = f'''
                <div class="validation-box">
                    <h4>[Lab] 실험적 검증 제안</h4>
                    <div class="val-grid">{''.join(val_items)}</div>
                </div>
                '''

        # Recommended papers section
        papers_html = ''
        recommended_papers = data.get('recommended_papers', {})
        if recommended_papers and recommended_papers.get('paper_count', 0) > 0:
            classic_papers = recommended_papers.get('classic_papers', [])
            breakthrough_papers = recommended_papers.get('breakthrough_papers', [])

            classic_items = []
            for p in classic_papers[:3]:
                paper_title = p.get('title', '')[:80]
                pmid = p.get('pmid', '')
                citations = p.get('citation_count', 0)
                year = p.get('year', '')
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                classic_items.append(f'<li><a href="{pubmed_url}" target="_blank">{paper_title}...</a> <span class="paper-meta">({year}, 인용 {citations}회)</span></li>')

            breakthrough_items = []
            for p in breakthrough_papers[:3]:
                paper_title = p.get('title', '')[:80]
                pmid = p.get('pmid', '')
                citations = p.get('citation_count', 0)
                year = p.get('year', '')
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                breakthrough_items.append(f'<li><a href="{pubmed_url}" target="_blank">{paper_title}...</a> <span class="paper-meta">({year}, 인용 {citations}회)</span></li>')

            papers_html = f'''
            <div class="papers-box">
                <h4>📄 추천 논문</h4>
                <div class="papers-grid">
                    <div class="paper-category classic">
                        <h5>[Ref] 교과서급 핵심 연구 (Classic)</h5>
                        <ul>{''.join(classic_items) if classic_items else '<li>없음</li>'}</ul>
                    </div>
                    <div class="paper-category emerging">
                        <h5>🚀 최신 주목 연구 (Emerging)</h5>
                        <ul>{''.join(breakthrough_items) if breakthrough_items else '<li>없음</li>'}</ul>
                    </div>
                </div>
            </div>
            '''

        # Title section
        title_html = ''
        if title:
            title_html = f'''
            <div class="abstract-title-box">
                <h3>{title}</h3>
                {f'<p class="title-en">{title_en}</p>' if title_en else ''}
            </div>
            '''

        return f'''
        <section class="brief-abstract-section" id="brief-abstract">
            <h2>📄 연구 요약 (Extended Abstract)</h2>
            <p class="section-subtitle">LLM 기반 종합 분석 요약</p>

            {title_html}

            <!-- 핵심 발견을 상단에 전체 너비로 배치 -->
            {findings_html}

            <!-- 본문 -->
            <div class="abstract-main-full">
                <div class="abstract-text">
                    {formatted_abstract}
                </div>
            </div>

            <div class="interpretation-section">
                {driver_html}
                {rag_html}
            </div>

            {validation_html}

            {papers_html}

            <div class="abstract-note">
                <span class="note-icon">[Info]</span>
                <span>본 요약은 Claude AI + RAG 문헌 검색을 통해 자동 생성되었습니다.</span>
            </div>

            <style>
                .brief-abstract-section {{
                    margin-bottom: var(--sp-6);
                }}
                .brief-abstract-section .section-subtitle {{
                    font-size: 14px;
                    color: var(--text-secondary);
                    margin-top: -8px;
                    margin-bottom: var(--sp-4);
                }}
                .abstract-title-box {{
                    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    padding: var(--sp-4) var(--sp-5);
                    border-radius: var(--radius-md);
                    border-left: 4px solid #0ea5e9;
                    margin-bottom: var(--sp-4);
                }}
                .abstract-title-box h3 {{
                    font-size: 17px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 var(--sp-2) 0;
                }}
                .abstract-title-box .title-en {{
                    font-size: 13px;
                    color: var(--text-secondary);
                    font-style: italic;
                    margin: 0;
                }}
                /* 핵심 발견 - 상단 전체 너비 흰색 배너 */
                .key-findings-box {{
                    background: white;
                    padding: 28px 32px;
                    border-radius: 12px;
                    border: none;
                    margin-bottom: 24px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                }}
                .key-findings-box h4 {{
                    font-size: 18px;
                    font-weight: 700;
                    color: #1e293b;
                    margin: 0 0 20px 0;
                    padding-bottom: 12px;
                    border-bottom: 3px solid #10b981;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .key-findings-box ul {{
                    margin: 0;
                    padding: 0;
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 12px;
                    list-style: none;
                    counter-reset: findings;
                }}
                .key-findings-box li {{
                    font-size: 14px;
                    line-height: 1.7;
                    color: #334155;
                    padding: 14px 16px 14px 50px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                    position: relative;
                    counter-increment: findings;
                    transition: all 0.2s ease;
                }}
                .key-findings-box li:hover {{
                    background: #f1f5f9;
                    border-color: #10b981;
                    transform: translateX(4px);
                }}
                .key-findings-box li::before {{
                    content: counter(findings);
                    position: absolute;
                    left: 14px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 26px;
                    height: 26px;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    font-size: 12px;
                    font-weight: 700;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .key-findings-box li strong.num {{
                    color: #059669;
                    font-weight: 700;
                    font-size: 15px;
                }}
                .key-findings-box .findings-count {{
                    font-size: 12px;
                    font-weight: 600;
                    color: white;
                    background: #10b981;
                    padding: 4px 10px;
                    border-radius: 12px;
                    margin-left: auto;
                }}
                /* 본문 - 전체 너비, 세련된 카드 스타일 */
                .abstract-main-full {{
                    background: white;
                    padding: 28px 32px;
                    border-radius: 12px;
                    border: none;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
                }}
                .abstract-main-full::before {{
                    content: "[List] 연구 요약";
                    display: block;
                    font-size: 16px;
                    font-weight: 700;
                    color: #1e293b;
                    margin-bottom: 16px;
                    padding-bottom: 12px;
                    border-bottom: 2px solid #3b82f6;
                }}
                .abstract-text {{
                    font-size: 15px;
                    line-height: 2;
                    color: #374151;
                    text-align: justify;
                    column-count: 2;
                    column-gap: 40px;
                    column-rule: 1px solid #e5e7eb;
                }}
                .abstract-text p {{
                    margin-bottom: 16px;
                    text-indent: 0;
                }}
                .abstract-text p:first-letter {{
                    font-size: 1.5em;
                    font-weight: 600;
                    color: #3b82f6;
                }}
                /* 해석 섹션 - 2열 카드 */
                .interpretation-section {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 16px;
                    margin-bottom: 20px;
                }}
                .interpretation-box {{
                    background: white;
                    padding: 24px;
                    border-radius: 12px;
                    border: none;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
                    position: relative;
                    overflow: hidden;
                }}
                .interpretation-box::before {{
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 4px;
                    height: 100%;
                }}
                .interpretation-box.driver {{
                    background: white;
                }}
                .interpretation-box.driver::before {{
                    background: linear-gradient(180deg, #8b5cf6 0%, #a78bfa 100%);
                }}
                .interpretation-box.literature {{
                    background: white;
                }}
                .interpretation-box.literature::before {{
                    background: linear-gradient(180deg, #f59e0b 0%, #fbbf24 100%);
                }}
                .interpretation-box h4 {{
                    font-size: 16px;
                    font-weight: 700;
                    margin: 0 0 16px 0;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .interpretation-box.driver h4 {{
                    color: #7c3aed;
                }}
                .interpretation-box.literature h4 {{
                    color: #d97706;
                }}
                .interpretation-box p {{
                    font-size: 14px;
                    line-height: 1.8;
                    color: #4b5563;
                    margin: 0;
                }}
                /* 검증 제안 - 4열 그리드 */
                .validation-box {{
                    background: white;
                    padding: 24px 28px;
                    border-radius: 12px;
                    border: none;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
                    position: relative;
                    overflow: hidden;
                }}
                .validation-box::before {{
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 4px;
                    background: linear-gradient(90deg, #ef4444 0%, #f87171 50%, #fca5a5 100%);
                }}
                .validation-box h4 {{
                    font-size: 16px;
                    font-weight: 700;
                    color: #dc2626;
                    margin: 0 0 16px 0;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .val-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 12px;
                }}
                .val-item {{
                    font-size: 13px;
                    color: #374151;
                    padding: 14px 16px;
                    background: #fef2f2;
                    border-radius: 8px;
                    border: 1px solid #fecaca;
                    transition: all 0.2s ease;
                }}
                .val-item:hover {{
                    background: #fee2e2;
                    transform: translateY(-2px);
                }}
                .val-item strong {{
                    display: block;
                    color: #dc2626;
                    font-size: 12px;
                    margin-bottom: 6px;
                    letter-spacing: 0.5px;
                }}
                /* 추천 논문 박스 */
                .papers-box {{
                    background: white;
                    padding: 24px 28px;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                    margin-top: 20px;
                    position: relative;
                    overflow: hidden;
                }}
                .papers-box::before {{
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
                }}
                .papers-box h4 {{
                    font-size: 16px;
                    font-weight: 700;
                    color: #1d4ed8;
                    margin: 0 0 16px 0;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .papers-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
                .paper-category {{
                    padding: 16px;
                    border-radius: 8px;
                }}
                .paper-category.classic {{
                    background: #fef3c7;
                    border: 1px solid #fcd34d;
                }}
                .paper-category.emerging {{
                    background: #dbeafe;
                    border: 1px solid #93c5fd;
                }}
                .paper-category h5 {{
                    font-size: 14px;
                    font-weight: 600;
                    margin: 0 0 12px 0;
                }}
                .paper-category.classic h5 {{
                    color: #b45309;
                }}
                .paper-category.emerging h5 {{
                    color: #1d4ed8;
                }}
                .paper-category ul {{
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }}
                .paper-category li {{
                    font-size: 13px;
                    padding: 8px 0;
                    border-bottom: 1px solid rgba(0,0,0,0.1);
                }}
                .paper-category li:last-child {{
                    border-bottom: none;
                }}
                .paper-category a {{
                    color: #374151;
                    text-decoration: none;
                }}
                .paper-category a:hover {{
                    color: #1d4ed8;
                    text-decoration: underline;
                }}
                .paper-meta {{
                    display: block;
                    font-size: 11px;
                    color: #6b7280;
                    margin-top: 4px;
                }}
                .abstract-note {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: var(--sp-3);
                    background: var(--bg-tertiary);
                    border-radius: var(--radius-sm);
                    font-size: 12px;
                    color: var(--text-secondary);
                }}
                .note-icon {{
                    font-size: 16px;
                }}
                @media (max-width: 768px) {{
                    .key-findings-box ul {{
                        grid-template-columns: 1fr;
                    }}
                    .abstract-text {{
                        column-count: 1;
                    }}
                    .interpretation-section {{
                        grid-template-columns: 1fr;
                    }}
                    .val-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </section>
        '''

    def _generate_qc_section_html(self, data: Dict) -> str:
        """Generate Data Quality Control section with PCA and correlation heatmap."""
        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})
        viz_interpretations = data.get('visualization_interpretations', {})

        # Look for PCA - prefer interactive over static
        pca_fig = figures.get('pca_plot', '')
        pca_interactive = interactive_figures.get('pca_interactive', '')

        pca_html = ''
        if pca_interactive:
            # Use interactive PCA with hover for sample names
            escaped_html = pca_interactive.replace('"', '&quot;')
            pca_html = f'''
            <div class="view-toggle">
                <button class="toggle-btn active" onclick="showPcaView('interactive')">Interactive</button>
                <button class="toggle-btn" onclick="showPcaView('static')">Static</button>
            </div>
            <div id="pca-interactive" class="pca-view active" style="display:flex; flex-direction:column; align-items:center;">
                <iframe id="pca-iframe" srcdoc="{escaped_html}" style="width:100%; max-width:720px; height:420px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스를 올리면 샘플 ID를 확인할 수 있습니다</p>
            </div>
            <div id="pca-static" class="pca-view" style="display:none; text-align:center;">
                <img src="{pca_fig}" alt="PCA Plot" class="figure-img" style="max-width:100%;">
            </div>
            '''
        elif pca_fig:
            pca_html = f'<div style="text-align:center;"><img src="{pca_fig}" alt="PCA Plot" class="figure-img" style="max-width:100%;"></div>'
        else:
            pca_html = '<p class="no-data">PCA plot not available</p>'

        # Get PCA interpretation from LLM - DETAILED VERSION
        pca_interp = viz_interpretations.get('pca_plot', {})
        pca_ai_section = ''
        if pca_interp:
            sample_quality = pca_interp.get('sample_quality', '')
            biological_meaning = pca_interp.get('biological_meaning', '')
            interpretation_guide = pca_interp.get('interpretation_guide', '')

            pca_ai_section = f'''
            <div class="ai-analysis-box detailed">
                <div class="ai-analysis-header">
                    <span class="ai-icon">[AI]</span>
                    <span class="ai-title">AI 상세 분석: PCA</span>
                </div>
                <div class="ai-analysis-content">
                    <div class="ai-section">
                        <h4>[Chart] 분석 요약</h4>
                        <p class="ai-summary-text">{pca_interp.get('summary', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>🔀 샘플 분리도 분석</h4>
                        <p>{pca_interp.get('separation_analysis', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Up] 분산 설명</h4>
                        <p>{pca_interp.get('variance_explanation', '')}</p>
                    </div>

                    {f'<div class="ai-section"><h4>✅ 샘플 품질 평가</h4><p>{sample_quality}</p></div>' if sample_quality else ''}

                    {f'<div class="ai-section"><h4>[DNA] 생물학적 의미</h4><p>{biological_meaning}</p></div>' if biological_meaning else ''}

                    {f'<div class="ai-section guide"><h4>[Book] 해석 가이드</h4><p class="guide-text">{interpretation_guide}</p></div>' if interpretation_guide else ''}
                </div>
            </div>
            '''

        return f'''
        <section class="qc-section" id="qc">
            <h2>2. 데이터 품질 관리</h2>
            <div class="figure-panel pca-container">
                <div class="figure-header">주성분 분석 (PCA)</div>
                <div class="figure-container">{pca_html}</div>
                <div class="figure-caption">조건별 샘플 클러스터링 (Tumor vs Normal) - <span style="color:#dc2626;">●</span> Tumor <span style="color:#2563eb;">●</span> Normal</div>
                {pca_ai_section}
            </div>
        </section>
        '''

    def _generate_deg_analysis_html(self, data: Dict) -> str:
        """Generate Differential Expression Analysis section."""
        # Prefer integrated_gene_table (has gene_symbol) over deg_significant (only has gene_id/Entrez)
        deg_df = data.get('integrated_gene_table_df')
        if deg_df is None or len(deg_df) == 0:
            deg_df = data.get('deg_significant_df')

        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})
        viz_interpretations = data.get('visualization_interpretations', {})

        # Get figures
        volcano_fig = figures.get('volcano_plot', '')
        volcano_interactive = interactive_figures.get('volcano_interactive', '')
        heatmap_fig = figures.get('heatmap_top50', figures.get('heatmap_key_genes', figures.get('top_genes_heatmap', '')))

        # Check for interactive heatmap (multiple possible keys)
        heatmap_interactive = (
            interactive_figures.get('heatmap_interactive') or
            interactive_figures.get('heatmap_top50_interactive') or
            interactive_figures.get('deg_heatmap_interactive') or
            ''
        )

        # Volcano plot with interactive toggle
        if volcano_interactive:
            escaped_html = volcano_interactive.replace('"', '&quot;')
            volcano_html = f'''
            <div class="view-toggle">
                <button class="toggle-btn active" onclick="showVolcanoView('interactive')">Interactive</button>
                <button class="toggle-btn" onclick="showVolcanoView('static')">Static</button>
            </div>
            <div id="volcano-interactive" class="volcano-view active" style="display:flex; flex-direction:column; align-items:center;">
                <iframe id="volcano-iframe" srcdoc="{escaped_html}" style="width:100%; max-width:900px; height:500px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스를 올리면 유전자 정보를 확인할 수 있습니다. 드래그하여 확대/축소 가능합니다.</p>
            </div>
            <div id="volcano-static" class="volcano-view" style="display:none; text-align:center;">
                <img src="{volcano_fig}" alt="Volcano Plot" class="figure-img" style="max-width:100%;">
            </div>
            '''
        elif volcano_fig:
            volcano_html = f'<div style="text-align:center;"><img src="{volcano_fig}" alt="Volcano Plot" class="figure-img" style="max-width:100%;"></div>'
        else:
            volcano_html = '<p class="no-data">Volcano plot not available</p>'

        # Heatmap with interactive toggle (similar to volcano)
        if heatmap_interactive:
            escaped_heatmap_html = heatmap_interactive.replace('"', '&quot;')
            heatmap_html = f'''
            <div class="view-toggle">
                <button class="toggle-btn active" onclick="showHeatmapView('interactive')">Interactive</button>
                <button class="toggle-btn" onclick="showHeatmapView('static')">Static</button>
            </div>
            <div id="heatmap-interactive" class="heatmap-view active" style="display:flex; flex-direction:column; align-items:center;">
                <iframe id="heatmap-iframe" srcdoc="{escaped_heatmap_html}" style="width:100%; max-width:1000px; height:600px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스를 올리면 유전자/샘플 정보를 확인할 수 있습니다.</p>
            </div>
            <div id="heatmap-static" class="heatmap-view" style="display:none; text-align:center;">
                <img src="{heatmap_fig}" alt="Heatmap" class="figure-img" style="max-width:100%;">
            </div>
            '''
        elif heatmap_fig:
            heatmap_html = f'<div style="text-align:center;"><img src="{heatmap_fig}" alt="Heatmap" class="figure-img" style="max-width:100%;"></div>'
        else:
            heatmap_html = '<p class="no-data">Heatmap not available</p>'

        # AI interpretation for volcano plot - DETAILED VERSION
        volcano_interp = viz_interpretations.get('volcano_plot', {})
        volcano_ai_section = ''
        if volcano_interp:
            observations = volcano_interp.get('key_observations', [])
            observations_html = ''.join([f'<li>{obs}</li>' for obs in observations]) if observations else ''

            clinical_relevance = volcano_interp.get('clinical_relevance', '')
            interpretation_guide = volcano_interp.get('interpretation_guide', '')

            volcano_ai_section = f'''
            <div class="ai-analysis-box detailed">
                <div class="ai-analysis-header">
                    <span class="ai-icon">[AI]</span>
                    <span class="ai-title">AI 상세 분석: Volcano Plot</span>
                </div>
                <div class="ai-analysis-content">
                    <div class="ai-section">
                        <h4>[Chart] 분석 요약</h4>
                        <p class="ai-summary-text">{volcano_interp.get('summary', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Search] 주요 관찰 사항</h4>
                        <ul class="ai-observations-list">{observations_html}</ul>
                    </div>

                    <div class="ai-section">
                        <h4>[DNA] 생물학적 의의</h4>
                        <p>{volcano_interp.get('biological_significance', '')}</p>
                    </div>

                    {f'<div class="ai-section"><h4>[Drug] 임상적 관련성</h4><p>{clinical_relevance}</p></div>' if clinical_relevance else ''}

                    {f'<div class="ai-section guide"><h4>[Book] 해석 가이드</h4><p class="guide-text">{interpretation_guide}</p></div>' if interpretation_guide else ''}
                </div>
            </div>
            '''

        # AI interpretation for heatmap - DETAILED VERSION
        heatmap_interp = viz_interpretations.get('heatmap', {})
        heatmap_ai_section = ''
        if heatmap_interp:
            observations = heatmap_interp.get('key_observations', [])
            observations_html = ''.join([f'<li>{obs}</li>' for obs in observations]) if observations else ''

            sample_clustering = heatmap_interp.get('sample_clustering', '')
            interpretation_guide = heatmap_interp.get('interpretation_guide', '')

            heatmap_ai_section = f'''
            <div class="ai-analysis-box detailed">
                <div class="ai-analysis-header">
                    <span class="ai-icon">[AI]</span>
                    <span class="ai-title">AI 상세 분석: 발현 히트맵</span>
                </div>
                <div class="ai-analysis-content">
                    <div class="ai-section">
                        <h4>[Chart] 분석 요약</h4>
                        <p class="ai-summary-text">{heatmap_interp.get('summary', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Search] 주요 관찰 사항</h4>
                        <ul class="ai-observations-list">{observations_html}</ul>
                    </div>

                    <div class="ai-section">
                        <h4>[DNA] 발현 패턴 분석</h4>
                        <p>{heatmap_interp.get('pattern_analysis', '')}</p>
                    </div>

                    {f'<div class="ai-section"><h4>[Up] 샘플 클러스터링</h4><p>{sample_clustering}</p></div>' if sample_clustering else ''}

                    {f'<div class="ai-section guide"><h4>[Book] 해석 가이드</h4><p class="guide-text">{interpretation_guide}</p></div>' if interpretation_guide else ''}
                </div>
            </div>
            '''

        # Top upregulated genes table
        up_table = ''
        down_table = ''

        if deg_df is not None and len(deg_df) > 0:
            # Sort by log2FC and get top 20 up/down
            if 'log2FC' in deg_df.columns:
                # Top 5 for concise display (Best Practice: 핵심 데이터만 간소화된 표로)
                up_genes = deg_df[deg_df['log2FC'] > 0].nlargest(5, 'log2FC')
                down_genes = deg_df[deg_df['log2FC'] < 0].nsmallest(5, 'log2FC')

                # Build upregulated table - prefer gene_symbol over gene_id
                up_rows = ''
                for _, row in up_genes.iterrows():
                    gene_symbol = row.get('gene_symbol', '')
                    gene_id = row.get('gene_id', 'N/A')
                    display_name = gene_symbol if gene_symbol and str(gene_symbol) != 'nan' else str(gene_id)
                    log2fc = row.get('log2FC', 0)
                    padj = row.get('padj', 1)
                    fold_change = 2 ** abs(log2fc)
                    up_rows += f'<tr><td><strong>{display_name}</strong></td><td class="up-text">+{log2fc:.2f}</td><td>{fold_change:.1f}x</td><td>{padj:.2e}</td></tr>'

                up_table = f'''
                <div class="table-wrapper compact">
                    <div class="table-header">
                        <span class="table-title">상향 발현 Top 5 (암에서 증가)</span>
                    </div>
                    <table class="data-table">
                        <thead><tr><th>유전자</th><th>log2FC</th><th>FC</th><th>p-adj</th></tr></thead>
                        <tbody>{up_rows}</tbody>
                    </table>
                </div>
                '''

                # Build downregulated table
                down_rows = ''
                for _, row in down_genes.iterrows():
                    gene_symbol = row.get('gene_symbol', '')
                    gene_id = row.get('gene_id', 'N/A')
                    display_name = gene_symbol if gene_symbol and str(gene_symbol) != 'nan' else str(gene_id)
                    log2fc = row.get('log2FC', 0)
                    padj = row.get('padj', 1)
                    fold_change = 2 ** abs(log2fc)
                    down_rows += f'<tr><td><strong>{display_name}</strong></td><td class="down-text">{log2fc:.2f}</td><td>{fold_change:.1f}x</td><td>{padj:.2e}</td></tr>'

                down_table = f'''
                <div class="table-wrapper compact">
                    <div class="table-header">
                        <span class="table-title">하향 발현 Top 5 (암에서 감소)</span>
                    </div>
                    <table class="data-table">
                        <thead><tr><th>유전자</th><th>log2FC</th><th>FC</th><th>p-adj</th></tr></thead>
                        <tbody>{down_rows}</tbody>
                    </table>
                </div>
                '''

        # Calculate summary stats
        n_total = len(deg_df) if deg_df is not None else 0
        n_up = len(deg_df[deg_df['log2FC'] > 0]) if deg_df is not None and 'log2FC' in deg_df.columns else 0
        n_down = n_total - n_up

        return f'''
        <section class="deg-section" id="deg-analysis">
            <h2>3. 차등발현 분석</h2>

            <!-- 1. Volcano Plot (전체 폭) -->
            <div class="figure-panel volcano-container" style="margin-bottom: 24px;">
                <div class="figure-header">Volcano Plot</div>
                <div class="figure-container">{volcano_html}</div>
                <div class="figure-caption">X축: log2FC | Y축: -log10(padj) | <span style="color:#dc2626;">●</span> 상향 | <span style="color:#2563eb;">●</span> 하향</div>
                {volcano_ai_section}
            </div>

            <!-- 2. Heatmap (전체 폭) -->
            <div class="figure-panel" style="margin-bottom: 24px;">
                <div class="figure-header">발현 히트맵</div>
                <div class="figure-container">{heatmap_html}</div>
                <div class="figure-caption">상위 DEG 발현 패턴. Red=High, Blue=Low</div>
                {heatmap_ai_section}
            </div>

            <!-- 2. 요약 (Summary): 전체적인 변화 양상을 한 문장으로 정리 -->
            <div class="deg-summary-statement">
                <p>총 <strong>{n_total:,}</strong>개의 유전자가 유의미하게 변화하였습니다
                (상향조절: <span class="up-text">{n_up:,}</span>개, 하향조절: <span class="down-text">{n_down:,}</span>개).</p>
            </div>

            <div class="metrics-row compact">
                <div class="metric-box primary">
                    <div class="metric-value">{n_total:,}</div>
                    <div class="metric-label">총 DEG</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value up">{n_up:,}</div>
                    <div class="metric-label">상향 발현</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value down">{n_down:,}</div>
                    <div class="metric-label">하향 발현</div>
                </div>
            </div>

            <!-- 3. 핵심 데이터 (Selected Table): 가장 많이 변한 Top 5만 깔끔하게 -->
            <div class="deg-tables-header">
                <p>그 중 가장 크게 변화한 상위 5개 유전자는 다음과 같습니다:</p>
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
        viz_interpretations = data.get('visualization_interpretations', {})

        pathway_fig = figures.get('pathway_barplot', figures.get('pathway_enrichment', figures.get('go_enrichment', '')))
        pathway_html = f'<div style="text-align:center;"><img src="{pathway_fig}" alt="Pathway Enrichment" class="figure-img" style="max-width:100%;"></div>' if pathway_fig else ''

        # AI interpretation for pathway - DETAILED VERSION
        pathway_interp = viz_interpretations.get('pathway_barplot', {})
        pathway_ai_section = ''
        if pathway_interp:
            top_pathways = pathway_interp.get('top_pathways', [])
            pathways_html = ''.join([f'<li>{pw}</li>' for pw in top_pathways]) if top_pathways else ''

            cross_pathway = pathway_interp.get('cross_pathway_interactions', '')
            interpretation_guide = pathway_interp.get('interpretation_guide', '')

            pathway_ai_section = f'''
            <div class="ai-analysis-box detailed green-theme">
                <div class="ai-analysis-header">
                    <span class="ai-icon">[AI]</span>
                    <span class="ai-title">AI 상세 분석: Pathway 분석</span>
                </div>
                <div class="ai-analysis-content">
                    <div class="ai-section">
                        <h4>[Chart] 분석 요약</h4>
                        <p class="ai-summary-text">{pathway_interp.get('summary', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Lab] 주요 Pathway 상세 설명</h4>
                        <ul class="ai-observations-list">{pathways_html}</ul>
                    </div>

                    <div class="ai-section">
                        <h4>[DNA] 기능적 테마</h4>
                        <p>{pathway_interp.get('functional_theme', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Drug] 치료적 함의</h4>
                        <p>{pathway_interp.get('therapeutic_implications', '')}</p>
                    </div>

                    {f'<div class="ai-section"><h4>[Link] Pathway 간 상호작용</h4><p>{cross_pathway}</p></div>' if cross_pathway else ''}

                    {f'<div class="ai-section guide"><h4>[Book] 해석 가이드</h4><p class="guide-text">{interpretation_guide}</p></div>' if interpretation_guide else ''}
                </div>
            </div>
            '''

        # Separate pathways by category
        go_bp_rows = ''
        go_mf_rows = ''
        go_cc_rows = ''
        kegg_rows = ''

        if pathway_df is not None and len(pathway_df) > 0:
            for _, row in pathway_df.head(50).iterrows():
                # Handle multiple column name formats
                term = row.get('term_name', row.get('term', row.get('Term', 'N/A')))
                gene_count = row.get('gene_count', row.get('Overlap', 'N/A'))
                pval = row.get('padj', row.get('pvalue', row.get('P-value', row.get('Adjusted P-value', 1))))
                db = row.get('database', row.get('Gene_set', ''))
                genes = row.get('genes', '')

                if isinstance(gene_count, str) and '/' in gene_count:
                    gene_count = gene_count.split('/')[0]

                # Format term name (remove GO ID for cleaner display)
                term_display = str(term)[:55] + "..." if len(str(term)) > 55 else str(term)

                # Show genes on hover
                genes_preview = str(genes)[:100] + "..." if len(str(genes)) > 100 else str(genes)
                row_html = f'<tr title="{genes_preview}"><td><strong>{term_display}</strong></td><td>{gene_count}</td><td>{pval:.2e}</td></tr>'

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
            <h2>4. 경로 및 기능 분석</h2>

            <div class="pathway-figure">
                {pathway_html}
                {pathway_ai_section}
            </div>

            <div class="pathway-subsections">
                <div class="pathway-panel">
                    <h4>4.1 GO 생물학적 과정 (BP)</h4>
                    <p class="panel-desc">세포의 생물학적 과정과 관련된 경로</p>
                    <table class="pathway-table">
                        <thead><tr><th>용어</th><th>유전자 수</th><th>adj.p-value</th></tr></thead>
                        <tbody>{go_bp_rows if go_bp_rows else "<tr><td colspan='3'>유의한 BP 용어 없음</td></tr>"}</tbody>
                    </table>
                </div>

                <div class="pathway-panel">
                    <h4>4.2 GO 분자 기능 (MF)</h4>
                    <p class="panel-desc">분자 수준의 기능 (효소 활성, 결합 등)</p>
                    <table class="pathway-table">
                        <thead><tr><th>용어</th><th>유전자 수</th><th>adj.p-value</th></tr></thead>
                        <tbody>{go_mf_rows if go_mf_rows else "<tr><td colspan='3'>유의한 MF 용어 없음</td></tr>"}</tbody>
                    </table>
                </div>

                <div class="pathway-panel">
                    <h4>4.3 GO 세포 구성요소 (CC)</h4>
                    <p class="panel-desc">세포 내 위치 (막, 세포질, 핵 등)</p>
                    <table class="pathway-table">
                        <thead><tr><th>용어</th><th>유전자 수</th><th>adj.p-value</th></tr></thead>
                        <tbody>{go_cc_rows if go_cc_rows else "<tr><td colspan='3'>유의한 CC 용어 없음</td></tr>"}</tbody>
                    </table>
                </div>

                <div class="pathway-panel">
                    <h4>4.4 KEGG 경로</h4>
                    <p class="panel-desc">대사/신호전달 경로 (KEGG 데이터베이스)</p>
                    <table class="pathway-table">
                        <thead><tr><th>경로</th><th>유전자 수</th><th>adj.p-value</th></tr></thead>
                        <tbody>{kegg_rows if kegg_rows else "<tr><td colspan='3'>유의한 KEGG 경로 없음</td></tr>"}</tbody>
                    </table>
                </div>
            </div>
        </section>
        '''

    def _generate_network_section_html(self, data: Dict) -> str:
        """Generate Network Analysis section."""
        hub_df = data.get('hub_genes_df')
        integrated_df = data.get('integrated_gene_table_df')
        figures = data.get('figures', {})
        interactive_figures = data.get('interactive_figures', {})
        viz_interpretations = data.get('visualization_interpretations', {})

        network_fig = figures.get('network_graph', figures.get('network_plot', figures.get('network_2d', '')))
        network_3d_interactive = interactive_figures.get('network_3d_interactive', '')
        network_2d_interactive = interactive_figures.get('network_2d_interactive', '')

        # Network with interactive toggle (2D default, 3D optional)
        if network_3d_interactive and network_2d_interactive:
            escaped_3d = network_3d_interactive.replace('"', '&quot;')
            escaped_2d = network_2d_interactive.replace('"', '&quot;')
            network_html = f'''
            <div class="network-toggle-container">
                <button class="network-toggle-btn" onclick="toggleNetworkView(this, '2d')">2D (Hover)</button>
                <button class="network-toggle-btn" onclick="toggleNetworkView(this, '3d')">3D (회전)</button>
            </div>
            <div id="network-2d-view" style="display:flex; flex-direction:column; align-items:center;">
                <iframe id="network-2d-iframe" srcdoc="{escaped_2d}" style="width:100%; max-width:800px; height:570px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스를 올리면 유전자 정보 확인</p>
            </div>
            <div id="network-3d-view" style="display:none; flex-direction:column; align-items:center;">
                <iframe id="network-3d-iframe" srcdoc="{escaped_3d}" style="width:100%; max-width:800px; height:600px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스로 회전, 확대/축소 가능</p>
            </div>
            '''
        elif network_3d_interactive:
            escaped_html = network_3d_interactive.replace('"', '&quot;')
            network_html = f'''
            <div style="display:flex; flex-direction:column; align-items:center;">
                <iframe srcdoc="{escaped_html}" style="width:100%; max-width:800px; height:600px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스로 회전, 확대/축소 가능</p>
            </div>
            '''
        elif network_2d_interactive:
            escaped_html = network_2d_interactive.replace('"', '&quot;')
            network_html = f'''
            <div style="display:flex; flex-direction:column; align-items:center;">
                <iframe srcdoc="{escaped_html}" style="width:100%; max-width:800px; height:570px; border:none; border-radius:8px;"></iframe>
                <p class="panel-note">[Tip] 마우스를 올리면 유전자 정보 확인</p>
            </div>
            '''
        elif network_fig:
            network_html = f'<div style="text-align:center;"><img src="{network_fig}" alt="Network" class="figure-img" style="max-width:100%;"></div>'
        else:
            network_html = ''

        # Build gene_id to gene_symbol mapping from integrated_gene_table
        id_to_symbol = {}
        if integrated_df is not None and len(integrated_df) > 0:
            for _, row in integrated_df.iterrows():
                gene_id = str(row.get('gene_id', ''))
                gene_symbol = row.get('gene_symbol', '')
                if gene_id and gene_symbol and str(gene_symbol) != 'nan':
                    id_to_symbol[gene_id] = gene_symbol

        # AI interpretation for network - DETAILED VERSION
        network_interp = viz_interpretations.get('network_graph', {})
        network_ai_section = ''
        if network_interp:
            therapeutic_potential = network_interp.get('therapeutic_potential', '')
            interpretation_guide = network_interp.get('interpretation_guide', '')

            network_ai_section = f'''
            <div class="ai-analysis-box detailed orange-theme">
                <div class="ai-analysis-header">
                    <span class="ai-icon">[AI]</span>
                    <span class="ai-title">AI 상세 분석: 유전자 네트워크</span>
                </div>
                <div class="ai-analysis-content">
                    <div class="ai-section">
                        <h4>[Chart] 분석 요약</h4>
                        <p class="ai-summary-text">{network_interp.get('summary', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Network] Hub 유전자 심층 분석</h4>
                        <p>{network_interp.get('hub_gene_analysis', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[Link] 네트워크 토폴로지</h4>
                        <p>{network_interp.get('network_topology', '')}</p>
                    </div>

                    <div class="ai-section">
                        <h4>[DNA] 생물학적 의미</h4>
                        <p>{network_interp.get('biological_implications', '')}</p>
                    </div>

                    {f'<div class="ai-section"><h4>[Drug] 치료적 잠재력</h4><p>{therapeutic_potential}</p></div>' if therapeutic_potential else ''}

                    {f'<div class="ai-section guide"><h4>[Book] 해석 가이드</h4><p class="guide-text">{interpretation_guide}</p></div>' if interpretation_guide else ''}
                </div>
            </div>
            '''

        # Hub genes table with improved display
        hub_table = ''
        if hub_df is not None and len(hub_df) > 0:
            hub_rows = ''
            for _, row in hub_df.head(15).iterrows():
                gene_id = str(row.get('gene_id', row.get('gene', 'N/A')))
                gene_symbol = id_to_symbol.get(gene_id, gene_id)  # Use symbol if available

                log2fc = row.get('log2FC', 0)
                direction = row.get('direction', 'up' if log2fc > 0 else 'down')
                hub_score = row.get('hub_score', row.get('enhanced_hub_score', 0))
                degree = row.get('degree', 0)
                targets = row.get('regulatory_targets', 0)

                # Format display
                fc_class = 'cell-up' if direction == 'up' else 'cell-down'
                fc_arrow = '↑' if direction == 'up' else '↓'
                fc_display = f'{fc_arrow} {abs(log2fc):.2f}'
                score_display = f'{hub_score:.2f}' if isinstance(hub_score, float) else str(hub_score)

                hub_rows += f'''<tr>
                    <td class="cell-gene">{gene_symbol}</td>
                    <td class="{fc_class}">{fc_display}</td>
                    <td>{degree}</td>
                    <td>{targets}</td>
                    <td>{score_display}</td>
                </tr>'''

            hub_table = f'''
            <div class="table-wrapper">
                <div class="table-header">
                    <span class="table-title">상위 Hub 유전자</span>
                </div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>유전자</th>
                            <th>log2FC</th>
                            <th>연결 수</th>
                            <th>조절 대상</th>
                            <th>Hub 점수</th>
                        </tr>
                    </thead>
                    <tbody>{hub_rows}</tbody>
                </table>
            </div>
            <div class="ai-box orange" style="margin-top: 16px;">
                <div class="ai-box-header">Hub 유전자 해석</div>
                <div class="ai-box-content">
                    <p><strong>Hub Gene이란?</strong> 네트워크에서 많은 유전자와 연결된 중심 유전자로, 주요 조절자 역할을 할 가능성이 높습니다.</p>
                    <p><strong>연결 수:</strong> 연결된 유전자 수 (높을수록 영향력 큼)</p>
                    <p><strong>조절 대상:</strong> 조절 대상 유전자 수</p>
                    <p><strong>Hub 점수:</strong> 네트워크 중심성 종합 점수 (0-1, 높을수록 중요)</p>
                </div>
            </div>
            '''

        return f'''
        <section class="network-section" id="network-analysis">
            <h2>6. 네트워크 분석</h2>

            <!-- 1. Hub 유전자 테이블 먼저 -->
            <div style="margin-bottom: 32px;">
                {hub_table if hub_table else '<p class="no-data">Hub 유전자가 확인되지 않음</p>'}
            </div>

            <!-- 2. 네트워크 시각화 -->
            <div class="figure-panel network-container" style="margin-bottom: 24px;">
                <div class="figure-header">유전자 공발현 네트워크</div>
                <div class="figure-container">
                    {network_html if network_html else '<p class="no-data">네트워크 시각화를 사용할 수 없음</p>'}
                </div>
                <div class="figure-caption">DEG 기반 공발현 네트워크. 연결선은 유전자 간 발현 상관관계를 나타냄.</div>
            </div>

            <!-- 3. AI 분석 마지막 -->
            {network_ai_section}
        </section>
        '''

    def _generate_clinical_implications_html(self, data: Dict) -> str:
        """Generate Clinical Implications section with detailed therapeutic targets and biomarkers."""
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        recommendations = data.get('research_recommendations', {})
        integrated_df = data.get('integrated_gene_table_df')

        # Build gene_id to symbol mapping
        id_to_symbol = {}
        if integrated_df is not None and len(integrated_df) > 0:
            for _, row in integrated_df.iterrows():
                gene_id = str(row.get('gene_id', ''))
                gene_symbol = row.get('gene_symbol', '')
                if gene_id and gene_symbol and str(gene_symbol) != 'nan':
                    id_to_symbol[gene_id] = gene_symbol

        # ============ 8.1 Biomarker Potential ============
        biomarker_data = recommendations.get('biomarker_development', {})
        diagnostic_candidates = biomarker_data.get('diagnostic_candidates', [])
        prognostic_candidates = biomarker_data.get('prognostic_candidates', [])

        biomarker_rows = ''
        # Use diagnostic candidates from research_recommendations
        for candidate in diagnostic_candidates[:4]:
            gene = candidate.get('gene', '')
            marker_type = candidate.get('marker_type', '진단')
            evidence = candidate.get('evidence_level', 'medium')
            rationale = candidate.get('rationale', '')

            evidence_badge = 'high' if evidence == 'high' else ('medium' if evidence == 'medium' else 'low')
            biomarker_rows += f'''
            <tr>
                <td class="cell-gene">{gene}</td>
                <td>{marker_type}</td>
                <td><span class="evidence-badge {evidence_badge}">{evidence.upper()}</span></td>
                <td>{rationale[:80]}{'...' if len(rationale) > 80 else ''}</td>
            </tr>'''

        # Add prognostic candidates
        for candidate in prognostic_candidates[:2]:
            gene = candidate.get('gene', '')
            association = candidate.get('association', '')
            validation = candidate.get('validation_needed', '')

            biomarker_rows += f'''
            <tr>
                <td class="cell-gene">{gene}</td>
                <td>예후</td>
                <td><span class="evidence-badge medium">MEDIUM</span></td>
                <td>{association} - {validation[:50]}</td>
            </tr>'''

        # Fallback to driver_known if no recommendations
        if not biomarker_rows:
            for d in driver_known[:5]:
                gene = d.get('gene_symbol', d.get('gene', 'Unknown'))
                # Convert if numeric
                if gene.isdigit():
                    gene = id_to_symbol.get(gene, gene)
                evidence = d.get('evidence_summary', '데이터베이스 검증 기반 바이오마커 후보')
                biomarker_rows += f'''
                <tr>
                    <td class="cell-gene">{gene}</td>
                    <td>진단/예후</td>
                    <td><span class="evidence-badge medium">MEDIUM</span></td>
                    <td>{evidence[:80]}{'...' if len(evidence) > 80 else ''}</td>
                </tr>'''

        # ============ 8.2 Therapeutic Targets ============
        therapeutic_data = recommendations.get('therapeutic_targets', {})
        high_priority = therapeutic_data.get('high_priority', [])
        medium_priority = therapeutic_data.get('medium_priority', [])

        therapeutic_rows = ''
        for target in high_priority[:4]:
            gene = target.get('gene', '')
            target_class = target.get('target_class', '')
            drugs = target.get('existing_drugs', [])
            rationale = target.get('rationale', '')

            drugs_display = ', '.join(drugs[:2]) if drugs else '연구 중'
            therapeutic_rows += f'''
            <tr>
                <td class="cell-gene">{gene}</td>
                <td>{target_class}</td>
                <td><span class="priority-badge high">HIGH</span></td>
                <td class="cell-drugs">{drugs_display}</td>
                <td>{rationale[:60]}{'...' if len(rationale) > 60 else ''}</td>
            </tr>'''

        for target in medium_priority[:2]:
            gene = target.get('gene', '')
            target_class = target.get('target_class', '')
            rationale = target.get('rationale', '')

            therapeutic_rows += f'''
            <tr>
                <td class="cell-gene">{gene}</td>
                <td>{target_class}</td>
                <td><span class="priority-badge medium">MEDIUM</span></td>
                <td class="cell-drugs">연구 필요</td>
                <td>{rationale[:60]}{'...' if len(rationale) > 60 else ''}</td>
            </tr>'''

        # Fallback to driver_novel
        if not therapeutic_rows:
            for d in driver_novel[:5]:
                gene = d.get('gene_symbol', d.get('gene', 'Unknown'))
                if gene.isdigit():
                    gene = id_to_symbol.get(gene, gene)
                evidence = d.get('regulatory_evidence', d.get('evidence_summary', ''))
                therapeutic_rows += f'''
                <tr>
                    <td class="cell-gene">{gene}</td>
                    <td>candidate</td>
                    <td><span class="priority-badge medium">MEDIUM</span></td>
                    <td class="cell-drugs">연구 필요</td>
                    <td>{evidence[:60] if evidence else "후보 조절 유전자"}...</td>
                </tr>'''

        # ============ 8.3 Drug Repurposing ============
        drug_repurposing = recommendations.get('drug_repurposing', {})
        repurposing_candidates = drug_repurposing.get('candidates', [])

        repurposing_rows = ''
        for candidate in repurposing_candidates[:3]:
            drug = candidate.get('drug', '')
            target_gene = candidate.get('target_gene', '')
            original = candidate.get('original_indication', '')
            rationale = candidate.get('repurposing_rationale', '')
            status = candidate.get('clinical_status', '')

            status_class = 'approved' if 'FDA' in status or '승인' in status else 'trial'
            repurposing_rows += f'''
            <tr>
                <td class="cell-drug-name">{drug}</td>
                <td class="cell-gene">{target_gene}</td>
                <td>{original}</td>
                <td><span class="status-badge {status_class}">{status}</span></td>
                <td>{rationale[:50]}{'...' if len(rationale) > 50 else ''}</td>
            </tr>'''

        return f'''
        <section class="clinical-section" id="clinical-implications">
            <h2>7. 임상적 시사점</h2>

            <div class="ai-box orange" style="margin-bottom: 20px;">
                <div class="ai-box-header">임상적 의미 요약</div>
                <div class="ai-box-content">
                    <p>본 분석에서 식별된 유전자들은 {data.get('cancer_prediction', {}).get('predicted_cancer', 'cancer')}의 진단, 예후 예측,
                    그리고 치료 표적으로서의 잠재력을 보여줍니다. 아래 표는 데이터베이스 검증 및 문헌 분석을 기반으로
                    우선순위가 높은 후보들을 정리한 것입니다.</p>
                </div>
            </div>

            <div class="table-wrapper">
                <div class="table-header">
                    <span class="table-title">8.1 바이오마커 후보</span>
                </div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>유전자</th>
                            <th>유형</th>
                            <th>근거</th>
                            <th>근거 설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        {biomarker_rows if biomarker_rows else '<tr><td colspan="4">바이오마커 후보가 없습니다.</td></tr>'}
                    </tbody>
                </table>
            </div>

            <div class="table-wrapper" style="margin-top: 24px;">
                <div class="table-header">
                    <span class="table-title">8.2 치료 표적</span>
                </div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>유전자</th>
                            <th>분류</th>
                            <th>우선순위</th>
                            <th>기존 약물</th>
                            <th>근거 설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        {therapeutic_rows if therapeutic_rows else '<tr><td colspan="5">치료 표적 후보가 없습니다.</td></tr>'}
                    </tbody>
                </table>
            </div>

            {f"""
            <div class="table-wrapper" style="margin-top: 24px;">
                <div class="table-header">
                    <span class="table-title">8.3 약물 재목적화 후보</span>
                </div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>약물</th>
                            <th>표적 유전자</th>
                            <th>기존 적응증</th>
                            <th>상태</th>
                            <th>근거 설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        {repurposing_rows}
                    </tbody>
                </table>
            </div>
            """ if repurposing_rows else ''}

            <div class="disclaimer-box" style="margin-top: 24px;">
                <strong>[!] 중요 안내:</strong> 모든 임상적 의미는 계산적 예측이며, 진단 또는 치료 적용 전에
                반드시 실험적·임상적 검증이 필요합니다. 본 분석은 연구 참고용이며 의학적 조언이 아닙니다.
            </div>
        </section>
        '''

    def _generate_recommended_papers_html(self, data: Dict) -> str:
        """Generate Recommended Papers section with Classic/Breakthrough classification."""
        papers_data = data.get('recommended_papers', {})

        if not papers_data or not papers_data.get('papers'):
            return '''
            <section class="recommended-papers-section" id="recommended-papers">
                <h2>8.4 추천 논문</h2>
                <p class="no-data">PubMed 검색을 통한 추천 논문이 없습니다.</p>
            </section>
            '''

        papers = papers_data.get('papers', [])
        cancer_type = papers_data.get('cancer_type', 'cancer')
        search_genes = papers_data.get('search_genes', [])

        # Check for enhanced format (classic_papers, breakthrough_papers)
        classic_papers = papers_data.get('classic_papers', [])
        breakthrough_papers = papers_data.get('breakthrough_papers', [])
        has_enhanced = bool(classic_papers or breakthrough_papers)

        def build_paper_card(paper: Dict, idx: int, paper_type: str = "") -> str:
            """Build HTML for a single paper card."""
            title = paper.get('title', 'No title')
            authors = paper.get('authors', 'Unknown')
            journal = paper.get('journal', '')
            year = paper.get('year', '')
            abstract = paper.get('abstract', '')[:300]
            if len(paper.get('abstract', '')) > 300:
                abstract += '...'
            pmid = paper.get('pmid', '')
            pubmed_url = paper.get('pubmed_url', f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/')
            doi = paper.get('doi', '')
            relevance = paper.get('relevance_reason', '')

            # Citation info (for enhanced papers)
            citation_count = paper.get('citation_count', 0)
            citation_velocity = paper.get('citation_velocity', 0)
            quality_score = paper.get('quality_score', 0)

            # Determine paper type badge
            p_type = paper.get('paper_type', paper_type)
            if 'classic' in p_type:
                type_badge = '<span class="paper-type-badge classic">[Ref] Classic Study</span>'
                type_class = "classic"
            elif 'breakthrough' in p_type:
                type_badge = '<span class="paper-type-badge breakthrough">🚀 Emerging Research</span>'
                type_class = "breakthrough"
            elif paper_type == "related" or p_type == "unknown":
                type_badge = '<span class="paper-type-badge related">📄 Related Paper</span>'
                type_class = "related"
            else:
                type_badge = ''
                type_class = ""

            # Citation metrics display
            citation_html = ''
            if citation_count > 0 or citation_velocity > 0:
                citation_html = f'''
                <div class="citation-metrics">
                    <span class="citation-count" title="Total citations">[Chart] 인용: {citation_count:,}회</span>
                    {f'<span class="citation-velocity" title="Citations per year">({citation_velocity:.1f}회/년)</span>' if citation_velocity > 0 else ''}
                </div>
                '''

            return f'''
            <div class="paper-card {type_class}">
                <div class="paper-number">{idx}</div>
                <div class="paper-content">
                    {type_badge}
                    <h4 class="paper-title">
                        <a href="{pubmed_url}" target="_blank" rel="noopener">{title}</a>
                    </h4>
                    <p class="paper-meta">
                        <span class="authors">{authors}</span>
                        <span class="journal">{journal}</span>
                        <span class="year">({year})</span>
                    </p>
                    {citation_html}
                    <p class="paper-abstract">{abstract}</p>
                    <div class="paper-footer">
                        <span class="relevance-tag">{relevance}</span>
                        <span class="pmid">PMID: <a href="{pubmed_url}" target="_blank">{pmid}</a></span>
                        {f'<span class="doi">DOI: {doi}</span>' if doi else ''}
                    </div>
                </div>
            </div>
            '''

        # Build paper cards
        if has_enhanced:
            # Build separate sections for classic and breakthrough
            classic_cards = ''
            for i, paper in enumerate(classic_papers[:3], 1):
                classic_cards += build_paper_card(paper, i, "classic")

            breakthrough_cards = ''
            for i, paper in enumerate(breakthrough_papers[:3], 1):
                breakthrough_cards += build_paper_card(paper, i, "breakthrough")

            # Other papers (unknown type - 인용 데이터 미확인)
            other_papers = papers_data.get('other_papers', [])
            other_cards = ''
            for i, paper in enumerate(other_papers[:3], 1):
                other_cards += build_paper_card(paper, i, "related")

            paper_sections = f'''
            <div class="paper-category">
                <h3 class="category-title">[Ref] 필수 참고 논문 (Classic Studies)</h3>
                <p class="category-desc">해당 분야의 기초가 되는 고인용 논문들입니다. (50회 이상 인용, 3년 이상 경과)</p>
                <div class="paper-list">
                    {classic_cards if classic_cards else '<p class="no-papers">인용 데이터 기준을 만족하는 Classic 논문이 없습니다.</p>'}
                </div>
            </div>

            <div class="paper-category">
                <h3 class="category-title">🚀 최신 주목 논문 (Emerging Research)</h3>
                <p class="category-desc">빠르게 인용되고 있는 최근 연구들입니다. (10회 이상 인용, 높은 인용 속도)</p>
                <div class="paper-list">
                    {breakthrough_cards if breakthrough_cards else '<p class="no-papers">인용 데이터 기준을 만족하는 Breakthrough 논문이 없습니다.</p>'}
                </div>
            </div>

            {f"""<div class="paper-category">
                <h3 class="category-title">📄 관련 논문 (Related Papers)</h3>
                <p class="category-desc">분석 유전자와 관련된 최신 논문들입니다. (인용 데이터 미확인 또는 기준 미달)</p>
                <div class="paper-list">
                    {other_cards}
                </div>
            </div>""" if other_cards else ''}
            '''
        else:
            # Legacy format - single list
            paper_cards = ''
            for i, paper in enumerate(papers[:6], 1):
                paper_cards += build_paper_card(paper, i)
            paper_sections = f'<div class="paper-list">{paper_cards}</div>'

        # Stats summary for enhanced format
        stats_html = ''
        if has_enhanced:
            classic_count = papers_data.get('classic_count', len(classic_papers))
            breakthrough_count = papers_data.get('breakthrough_count', len(breakthrough_papers))
            stats_html = f'''
            <div class="papers-stats">
                <span class="stat-item"><span class="stat-icon">[Ref]</span> Classic: {classic_count}편</span>
                <span class="stat-item"><span class="stat-icon">🚀</span> Emerging: {breakthrough_count}편</span>
            </div>
            '''

        return f'''
        <section class="recommended-papers-section" id="recommended-papers">
            <h2>8.4 추천 논문</h2>

            <div class="papers-intro">
                <p>아래 논문들은 <strong>{cancer_type}</strong> 및 분석에서 도출된 주요 유전자
                ({', '.join(search_genes[:5])})를 기반으로 PubMed/Semantic Scholar에서 검색 및 평가된 결과입니다.
                인용 지표와 학술적 영향력을 기준으로 선정되었습니다.</p>
                {stats_html}
            </div>

            {paper_sections}

            <div class="papers-disclaimer">
                <p><strong>참고:</strong> 논문 분류는 인용수와 출판연도를 기반으로 자동 산정되었습니다.
                Classic Study는 3년 이상 경과 및 100회 이상 인용된 논문,
                Emerging Research는 2년 이내 출판되어 빠르게 인용되는 논문입니다.</p>
            </div>
        </section>

        <style>
        .recommended-papers-section {{
            margin: 2rem 0;
            padding: 1.5rem;
            background: #fafafa;
            border-radius: 8px;
        }}
        .papers-intro {{
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: #e8f4fd;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
        .papers-stats {{
            display: flex;
            gap: 1.5rem;
            margin-top: 0.75rem;
            font-size: 0.9rem;
        }}
        .stat-item {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}
        .paper-category {{
            margin-bottom: 2rem;
        }}
        .category-title {{
            font-size: 1.1rem;
            color: #333;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }}
        .category-desc {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 1rem;
        }}
        .paper-list {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .no-papers {{
            color: #888;
            font-style: italic;
            padding: 1rem;
            background: #f5f5f5;
            border-radius: 4px;
        }}
        .paper-card {{
            display: flex;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            transition: box-shadow 0.2s;
        }}
        .paper-card.classic {{
            border-left: 4px solid #9c27b0;
        }}
        .paper-card.breakthrough {{
            border-left: 4px solid #ff9800;
        }}
        .paper-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .paper-type-badge {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .paper-type-badge.classic {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}
        .paper-type-badge.breakthrough {{
            background: #fff3e0;
            color: #e65100;
        }}
        .paper-type-badge.related {{
            background: #e3f2fd;
            color: #1565c0;
        }}
        .paper-card.related {{
            border-left: 4px solid #2196F3;
        }}
        .paper-number {{
            flex-shrink: 0;
            width: 32px;
            height: 32px;
            background: #2196F3;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-right: 1rem;
        }}
        .paper-content {{
            flex: 1;
        }}
        .paper-title {{
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            line-height: 1.4;
        }}
        .paper-title a {{
            color: #1a1a1a;
            text-decoration: none;
        }}
        .paper-title a:hover {{
            color: #2196F3;
            text-decoration: underline;
        }}
        .paper-meta {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 0.5rem;
        }}
        .paper-meta .authors {{
            font-style: italic;
        }}
        .paper-meta .journal {{
            color: #2196F3;
            margin-left: 0.5rem;
        }}
        .paper-meta .year {{
            color: #888;
            margin-left: 0.25rem;
        }}
        .citation-metrics {{
            display: flex;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: #555;
            margin-bottom: 0.5rem;
            padding: 0.3rem 0.5rem;
            background: #f5f5f5;
            border-radius: 4px;
            width: fit-content;
        }}
        .citation-count {{
            font-weight: 500;
        }}
        .citation-velocity {{
            color: #888;
        }}
        .paper-abstract {{
            font-size: 0.9rem;
            color: #444;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }}
        .paper-footer {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: center;
            font-size: 0.8rem;
        }}
        .relevance-tag {{
            background: #e3f2fd;
            color: #1565c0;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 500;
        }}
        .pmid, .doi {{
            color: #666;
        }}
        .pmid a {{
            color: #2196F3;
        }}
        .papers-disclaimer {{
            margin-top: 1rem;
            padding: 0.75rem;
            background: #fff3e0;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #666;
        }}
        </style>
        '''

    def _generate_research_recommendations_html(self, data: Dict) -> str:
        """Generate comprehensive Research Recommendations section."""
        recommendations = data.get('research_recommendations', {})

        if not recommendations:
            return '''
            <section class="research-recommendations-section" id="research-recommendations">
                <h2>8. 후속 연구 제안</h2>
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

        # Build HTML for cautions
        cautions_html = self._build_cautions_html(cautions)

        return f'''
        <section class="research-recommendations-section" id="research-recommendations">
            <h2>8. 후속 연구 제안</h2>

            <div class="rec-intro">
                <p>본 섹션은 RNA-seq 분석 결과를 바탕으로 AI가 생성한 후속 연구 추천입니다.
                치료 타겟 후보, 약물 재목적화 가능성, 실험 검증 전략, 바이오마커 개발 방향을 제시합니다.</p>
            </div>

            {therapeutic_html}
            {drug_html}
            {experimental_html}
            {biomarker_html}
            {future_html}
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
            <h3>[Target] 치료 타겟 후보 (Therapeutic Targets)</h3>
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
            <h3>[Drug] 약물 재목적화 후보 (Drug Repurposing)</h3>
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
            <h3>[Lab] 실험 검증 전략 (Experimental Validation)</h3>
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
            <h3>[DNA] 바이오마커 개발 (Biomarker Development)</h3>
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
            <h3>[!] 주의사항 및 한계점 (Cautions & Limitations)</h3>
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
            <h2>9. 분석 방법</h2>

            <div class="methods-grid">
                <div class="method-card">
                    <h4>[DNA] 차등발현 분석</h4>
                    <ul>
                        <li>도구: DESeq2</li>
                        <li>기준값: |log2FC| > 1, padj < 0.05</li>
                        <li>정규화: Median of ratios</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>[Web] 네트워크 분석</h4>
                    <ul>
                        <li>도구: NetworkX</li>
                        <li>상관계수: Spearman > 0.7</li>
                        <li>Hub: 중심성 기준 상위 20개</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>[Chart] 경로 농축 분석</h4>
                    <ul>
                        <li>도구: gseapy (Enrichr)</li>
                        <li>DB: GO (BP/MF/CC), KEGG</li>
                        <li>기준값: padj < 0.05</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>✅ DB 검증</h4>
                    <ul>
                        <li>COSMIC Tier 1 유전자</li>
                        <li>OncoKB 주석</li>
                        <li>암종 특이적</li>
                    </ul>
                </div>
            </div>

            <div class="confidence-explanation">
                <h4>신뢰도 점수 계산</h4>
                <table class="score-table">
                    <tr><td>DEG 통계 유의성 (padj < 0.05)</td><td>+1점</td></tr>
                    <tr><td>TCGA 패턴 일치</td><td>+1점</td></tr>
                    <tr><td>문헌 검증 (DB match)</td><td>+1점</td></tr>
                    <tr><td>Hub 유전자 여부</td><td>+1점</td></tr>
                    <tr><td>암종 특이적</td><td>+1점</td></tr>
                </table>

                <div class="confidence-legend">
                    <span>[G][G][G][G][G] 5/5 매우 높음</span>
                    <span>[G][G][G][G][O] 4/5 높음</span>
                    <span>[G][G][G][O][O] 3/5 중간</span>
                    <span>[G][G][O][O][O] 2/5 낮음</span>
                    <span>[G][O][O][O][O] 1/5 검증 필요</span>
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

            /* Extended Abstract Styles - Full Page */
            .extended-abstract-section {
                margin: 0 0 40px 0;
                page-break-after: always;
            }

            .section-header-large {
                background: linear-gradient(135deg, var(--npj-blue) 0%, #1e40af 100%);
                color: white;
                padding: 24px 32px;
                border-radius: 12px 12px 0 0;
                margin-bottom: 0;
            }

            .section-header-large h2 {
                color: white;
                font-size: 24px;
                margin: 0 0 8px 0;
                border: none;
                padding: 0;
            }

            .section-subtitle {
                color: rgba(255, 255, 255, 0.85);
                font-size: 14px;
                margin: 0;
            }

            .abstract-box.extended.full-page {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 2px solid var(--npj-blue);
                border-top: none;
                border-radius: 0 0 12px 12px;
                padding: 32px 40px;
                min-height: 600px;
            }

            .abstract-title {
                text-align: center;
                margin-bottom: 28px;
                padding-bottom: 20px;
                border-bottom: 2px solid var(--gray-200);
            }

            .abstract-title h3 {
                font-size: 20px;
                color: var(--gray-900);
                margin: 0 0 8px 0;
                line-height: 1.4;
            }

            .abstract-title .title-en {
                font-size: 14px;
                color: var(--gray-600);
                font-style: italic;
                margin: 0;
            }

            .abstract-main-content {
                margin-bottom: 28px;
            }

            .abstract-body p {
                font-size: 15px;
                line-height: 1.8;
                color: var(--gray-800);
                text-align: justify;
                margin-bottom: 16px;
            }

            .abstract-supplementary {
                display: grid;
                gap: 20px;
            }

            .abstract-interpretations {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
            }

            .driver-interpretation,
            .rag-interpretation,
            .ml-interpretation {
                background: white;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid var(--gray-200);
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }

            .driver-interpretation h4,
            .rag-interpretation h4,
            .ml-interpretation h4 {
                font-size: 14px;
                color: var(--npj-blue);
                margin: 0 0 12px 0;
            }

            .driver-interpretation p,
            .rag-interpretation p,
            .ml-interpretation p {
                font-size: 13px;
                color: var(--gray-700);
                line-height: 1.6;
                margin: 0;
            }

            .abstract-note {
                margin-top: 24px;
                padding: 14px 20px;
                background: #eff6ff;
                border-radius: 8px;
                font-size: 13px;
                color: var(--gray-600);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .abstract-note .note-icon {
                font-size: 16px;
            }

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

            /* Network Toggle Buttons */
            .network-toggle-container {
                display: flex;
                justify-content: center;
                gap: 8px;
                margin-bottom: 12px;
            }

            .network-toggle-btn {
                padding: 8px 16px;
                border: 1px solid var(--gray-300);
                border-radius: 4px;
                background: white;
                color: var(--gray-600);
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.15s;
            }

            .network-toggle-btn:hover {
                background: var(--gray-100);
            }

            .network-toggle-btn.active {
                background: var(--npj-blue);
                color: white;
                border-color: var(--npj-blue);
            }

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

            /* DEG Summary Box */
            .deg-summary-box {
                display: flex;
                justify-content: center;
                gap: var(--spacing-xl);
                margin-bottom: var(--spacing-xl);
                padding: var(--spacing-lg);
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px;
            }

            .deg-summary-box .summary-stat {
                text-align: center;
                background: white;
                padding: var(--spacing-lg) var(--spacing-xl);
                border-radius: 12px;
                min-width: 150px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }

            .deg-summary-box .stat-number {
                display: block;
                font-size: 36px;
                font-weight: 700;
                color: var(--gray-800);
            }

            .deg-summary-box .summary-stat.up .stat-number { color: #dc2626; }
            .deg-summary-box .summary-stat.down .stat-number { color: #2563eb; }

            .deg-summary-box .stat-label {
                display: block;
                font-size: 14px;
                color: var(--gray-600);
                margin-top: 4px;
            }

            /* AI Interpretation Panel */
            .ai-interpretation-panel {
                background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
                border-left: 4px solid var(--npj-blue);
                padding: var(--spacing-md);
                margin-top: var(--spacing-md);
                border-radius: 0 8px 8px 0;
            }

            .ai-interpretation-panel .ai-header {
                font-weight: 600;
                color: var(--npj-blue);
                margin-bottom: 8px;
            }

            .ai-interpretation-panel .ai-summary {
                font-size: 14px;
                line-height: 1.6;
                color: var(--gray-700);
            }

            .ai-interpretation-panel .ai-observations {
                font-size: 13px;
                color: var(--gray-600);
                margin: 8px 0;
                padding-left: 20px;
            }

            .ai-interpretation-panel .ai-observations li {
                margin-bottom: 4px;
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

            .gene-function {
                display: flex;
                align-items: flex-start;
                gap: 8px;
                padding: 10px 12px;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-radius: 8px;
                margin-bottom: 12px;
                border-left: 3px solid #0ea5e9;
            }

            .function-icon {
                font-size: 14px;
                flex-shrink: 0;
                margin-top: 1px;
            }

            .function-text {
                font-size: 12px;
                color: #0c4a6e;
                line-height: 1.5;
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

            /* DEG Summary Statement */
            .deg-summary-statement {
                text-align: center;
                font-size: 16px;
                color: var(--gray-700);
                margin: var(--spacing-lg) 0;
                padding: var(--spacing-md);
                background: var(--gray-50);
                border-radius: 8px;
                border-left: 4px solid var(--npj-blue);
            }

            .deg-summary-statement p {
                margin: 0;
            }

            .deg-tables-header {
                text-align: center;
                font-size: 14px;
                color: var(--gray-600);
                margin: var(--spacing-md) 0 var(--spacing-sm) 0;
            }

            .deg-tables-header p {
                margin: 0;
            }

            .metrics-row.compact {
                margin: var(--spacing-sm) 0;
            }

            .metrics-row.compact .metric-box {
                padding: var(--spacing-sm) var(--spacing-md);
            }

            .table-wrapper.compact {
                max-width: 100%;
            }

            .table-wrapper.compact .data-table {
                font-size: 13px;
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
                content: "[Target] ";
            }

            .clinical-card h4.therapeutic::before {
                content: "[Drug] ";
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
                content: "[Test]";
            }

            .followup-card.functional h4::before {
                content: "[Lab]";
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

            /* ========== AI ANALYSIS BOX (DETAILED) - Spacious & Clean ========== */
            .ai-analysis-box.detailed {
                background: white;
                border: none;
                border-radius: 16px;
                padding: 60px 72px;
                margin: 48px 0;
                box-shadow: 0 2px 16px rgba(14, 165, 233, 0.08);
                position: relative;
            }

            .ai-analysis-box.detailed::before {
                content: "";
                position: absolute;
                top: 0;
                left: 72px;
                right: 72px;
                height: 3px;
                background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 50%, #7dd3fc 100%);
                border-radius: 0 0 3px 3px;
            }

            .ai-analysis-box.detailed.green-theme {
                box-shadow: 0 2px 16px rgba(34, 197, 94, 0.08);
            }

            .ai-analysis-box.detailed.green-theme::before {
                background: linear-gradient(90deg, #22c55e 0%, #4ade80 50%, #86efac 100%);
            }

            .ai-analysis-box.detailed.orange-theme {
                box-shadow: 0 2px 16px rgba(249, 115, 22, 0.08);
            }

            .ai-analysis-box.detailed.orange-theme::before {
                background: linear-gradient(90deg, #f97316 0%, #fb923c 50%, #fdba74 100%);
            }

            .ai-analysis-header {
                display: flex;
                align-items: center;
                gap: 14px;
                margin-bottom: 40px;
                padding-bottom: 28px;
                border-bottom: 1px solid #e2e8f0;
            }

            .green-theme .ai-analysis-header {
                border-bottom-color: #ecfdf5;
            }

            .orange-theme .ai-analysis-header {
                border-bottom-color: #fff7ed;
            }

            .ai-icon {
                font-size: 24px;
            }

            .green-theme .ai-icon {
                color: #22c55e;
            }

            .orange-theme .ai-icon {
                color: #f97316;
            }

            .ai-title {
                font-size: 17px;
                font-weight: 600;
                color: #334155;
                letter-spacing: -0.2px;
            }

            .green-theme .ai-title {
                color: #166534;
            }

            .orange-theme .ai-title {
                color: #9a3412;
            }

            .ai-analysis-content {
                display: flex;
                flex-direction: column;
                gap: 36px;
            }

            .ai-section {
                background: transparent;
                border-radius: 0;
                padding: 0;
                border: none;
                border-bottom: 1px solid #f1f5f9;
                padding-bottom: 36px;
            }

            .ai-section:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }

            .ai-section:hover {
                background: transparent;
            }

            .green-theme .ai-section {
                background: transparent;
                border-color: #ecfdf5;
            }

            .green-theme .ai-section:hover {
                background: transparent;
            }

            .orange-theme .ai-section {
                background: transparent;
                border-color: #fff7ed;
            }

            .orange-theme .ai-section:hover {
                background: transparent;
            }

            .ai-section h4 {
                font-size: 13px;
                font-weight: 600;
                color: #94a3b8;
                margin: 0 0 20px 0;
                padding-top: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }

            .ai-section p {
                font-size: 15px;
                color: #475569;
                line-height: 2.2;
                margin: 0;
                max-width: 75ch;
                padding: 4px 0;
            }

            .ai-summary-text {
                font-size: 16px !important;
                font-weight: 400;
                color: #334155 !important;
                line-height: 2.3 !important;
                padding: 8px 0 !important;
            }

            .ai-observations-list {
                list-style: none;
                padding: 0;
                margin: 0;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .ai-observations-list li {
                font-size: 15px;
                color: #334155;
                padding: 0 0 0 28px;
                position: relative;
                line-height: 1.8;
            }

            .ai-observations-list li:hover {
                color: #1e293b;
            }

            .ai-observations-list li::before {
                content: "•";
                position: absolute;
                left: 8px;
                top: 0;
                color: #0ea5e9;
                font-size: 18px;
                font-weight: bold;
            }

            .green-theme .ai-observations-list li::before {
                color: #22c55e;
            }

            .orange-theme .ai-observations-list li::before {
                color: #f97316;
            }

            .ai-section.guide {
                background: #f8fafc;
                border: none;
                border-left: 3px solid #0ea5e9;
                padding: 20px 24px;
                margin-top: 8px;
                border-radius: 0 8px 8px 0;
            }

            .green-theme .ai-section.guide {
                background: #f0fdf4;
                border-left-color: #22c55e;
            }

            .orange-theme .ai-section.guide {
                background: #fff7ed;
                border-left-color: #f97316;
            }

            .guide-text {
                font-style: italic;
                color: #64748b !important;
                font-size: 14px !important;
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

            function toggleNetworkView(btn, view) {{
                const view2d = document.getElementById('network-2d-view');
                const view3d = document.getElementById('network-3d-view');
                const buttons = document.querySelectorAll('.network-toggle-btn');

                // Reset all buttons
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                if (view === '3d') {{
                    if (view3d) view3d.style.display = 'flex';
                    if (view2d) view2d.style.display = 'none';
                }} else {{
                    if (view2d) view2d.style.display = 'flex';
                    if (view3d) view3d.style.display = 'none';
                }}
            }}

            // Initialize network toggle (2D default)
            document.addEventListener('DOMContentLoaded', function() {{
                const firstBtn = document.querySelector('.network-toggle-btn');
                if (firstBtn) firstBtn.classList.add('active');
            }});

            // PCA view toggle
            function showPcaView(view) {{
                const interactiveView = document.getElementById('pca-interactive');
                const staticView = document.getElementById('pca-static');
                const buttons = document.querySelectorAll('.pca-container .view-toggle .toggle-btn');

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

            // Heatmap view toggle
            function showHeatmapView(view) {{
                const interactiveView = document.getElementById('heatmap-interactive');
                const staticView = document.getElementById('heatmap-static');
                const container = interactiveView ? interactiveView.closest('.figure-panel') : null;
                const buttons = container ? container.querySelectorAll('.view-toggle .toggle-btn') : [];

                if (view === 'interactive') {{
                    if (interactiveView) {{
                        interactiveView.classList.add('active');
                        interactiveView.style.display = 'flex';
                    }}
                    if (staticView) {{
                        staticView.classList.remove('active');
                        staticView.style.display = 'none';
                    }}
                    if (buttons.length >= 2) {{
                        buttons[0].classList.add('active');
                        buttons[1].classList.remove('active');
                    }}
                }} else {{
                    if (interactiveView) {{
                        interactiveView.classList.remove('active');
                        interactiveView.style.display = 'none';
                    }}
                    if (staticView) {{
                        staticView.classList.add('active');
                        staticView.style.display = 'block';
                    }}
                    if (buttons.length >= 2) {{
                        buttons[0].classList.remove('active');
                        buttons[1].classList.add('active');
                    }}
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
                <div class="cover-badge">RNA-seq 차등발현 분석</div>
                <h1 class="cover-title">{cancer_type_kr} 전사체 분석 보고서</h1>
                <p class="cover-subtitle">포괄적 전사체 프로파일링 및 경로 분석</p>

                <div class="cover-stats">
                    <div class="cover-stat">
                        <span class="stat-number">{deg_count:,}</span>
                        <span class="stat-label">차등발현 유전자</span>
                    </div>
                    <div class="cover-stat">
                        <span class="stat-number">{hub_count}</span>
                        <span class="stat-label">Hub 유전자</span>
                    </div>
                    <div class="cover-stat">
                        <span class="stat-number">{pathway_count}</span>
                        <span class="stat-label">농축 경로</span>
                    </div>
                </div>

                <div class="cover-meta">
                    <p><strong>분석 일자:</strong> {datetime.now().strftime("%Y년 %m월 %d일")}</p>
                    <p><strong>파이프라인:</strong> BioInsight AI RNA-seq Pipeline v2.0</p>
                    <p><strong>분석 방법:</strong> DESeq2, WGCNA 네트워크 분석, GO/KEGG 농축 분석</p>
                </div>
            </div>
            <div class="cover-footer">
                <p>본 보고서는 AI 지원 분석을 통해 생성되었습니다. 모든 발견은 실험적 검증이 필요합니다.</p>
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
                findings_html = '<div class="key-findings"><h4>[Pin] 주요 발견</h4><ul>'
                for finding in key_findings[:8]:
                    findings_html += f'<li>{finding}</li>'
                findings_html += '</ul></div>'

            # Driver Gene interpretation
            driver_html = ''
            if driver_interp:
                driver_html = f'<div class="driver-interpretation"><h4>[DNA] Driver Gene Analysis 해석</h4><p>{driver_interp}</p></div>'

            # RAG Literature interpretation
            rag_html = ''
            if rag_interp:
                rag_html = f'<div class="rag-interpretation"><h4>[Ref] 문헌 기반 해석</h4><p>{rag_interp}</p></div>'

            # Validation priorities
            validation_html = ''
            if validation:
                validation_html = '<div class="validation-priorities"><h4>[Lab] 실험적 검증 제안</h4><div class="validation-grid">'
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
                ml_html = f'<div class="ml-interpretation"><h4>[AI] ML 예측 해석</h4><p>{ml_interp}</p></div>'

            return f'''
        <section class="extended-abstract-section" id="abstract">
            <div class="section-header-large">
                <h2>📄 연구 요약 (Extended Abstract)</h2>
                <p class="section-subtitle">LLM 기반 종합 분석 요약 - 1페이지 요약본</p>
            </div>
            <div class="abstract-box extended full-page">
                {title_html}

                <div class="abstract-main-content">
                    <div class="abstract-body">
                        {formatted_paragraphs}
                    </div>
                </div>

                <div class="abstract-supplementary">
                    {findings_html}

                    <div class="abstract-interpretations">
                        {driver_html}
                        {rag_html}
                        {ml_html}
                    </div>

                    {validation_html}
                </div>

                <div class="abstract-note">
                    <span class="note-icon">[Info]</span>
                    <span>본 요약은 Claude AI + RAG 문헌 검색을 통해 자동 생성되었습니다. 상세 내용은 각 섹션을 참조하세요.</span>
                </div>
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
            findings_html = '<div class="key-findings"><h4>[Pin] 주요 발견</h4><ul>'
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
            driver_html = f'<div class="driver-interpretation"><h4>[DNA] Driver Gene Analysis 해석</h4><p>{driver_interp}</p></div>'

        # Validation suggestions
        validation_html = '<div class="validation-priorities"><h4>[Lab] 실험적 검증 제안</h4><div class="validation-grid">'
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
            <span class="nav-brand">BioInsight 보고서</span>
            <div class="nav-links">
                <a href="#study-overview">개요</a>
                <a href="#brief-abstract">요약</a>
                <a href="#qc">QC</a>
                <a href="#deg-analysis">DEG</a>
                <a href="#pathway-analysis">경로</a>
                <a href="#driver-analysis">Driver</a>
                <a href="#network-analysis">네트워크</a>
                <a href="#clinical-implications">임상</a>
                <a href="#recommended-papers">논문</a>
                <a href="#research-recommendations">연구</a>
                <a href="#methods">방법</a>
                <a href="#detailed-table">부록</a>
            </div>
        </div>
    </nav>

    <main class="paper-content">
        <!-- 1. Study Overview -->
        {self._generate_study_overview_html(data)}

        <!-- 1.5 Brief Abstract -->
        {self._generate_brief_abstract_html(data)}

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

        <!-- 7. Clinical Implications -->
        {self._generate_clinical_implications_html(data)}

        <!-- 8. Research Recommendations (통합) -->
        {self._generate_research_recommendations_html(data)}

        <!-- 8.4 Recommended Papers (내부 통합) -->
        {self._generate_recommended_papers_html(data)}

        <!-- 9. Methods Summary -->
        {self._generate_methods_html() if self.config["include_methods"] else ""}

        <!-- 10. Literature-Based Interpretation (RAG) -->
        {self._generate_rag_summary_html(data)}

        <!-- 12. Appendix / Supplementary Data -->
        <section class="data-section" id="detailed-table">
            <h2>11. 부록</h2>
            {self._generate_detailed_table_html(data)}
        </section>
    </main>

    <footer class="paper-footer">
        <div class="footer-content">
            <p><strong>면책조항:</strong> 본 보고서는 AI 지원 분석 파이프라인에 의해 생성되었습니다.
            모든 발견은 예비적이며, 임상 적용 전 실험적 검증이 필요합니다.</p>
            <p class="footer-credit">BioInsight AI RNA-seq Pipeline v2.0 생성 | {datetime.now().strftime("%Y-%m-%d")}</p>
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
            self.logger.info("Generating extended abstract with LLM API...")
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
            self.logger.info("Generating visualization interpretations with LLM API...")
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

        # Generate paper recommendations (PubMed real-time search)
        recommended_papers_path = run_dir / "recommended_papers.json"
        if recommended_papers_path.exists():
            try:
                with open(recommended_papers_path, 'r', encoding='utf-8') as f:
                    data['recommended_papers'] = json.load(f)
                self.logger.info("Loaded existing paper recommendations")
            except Exception as e:
                self.logger.warning(f"Error loading paper recommendations: {e}")
        else:
            self.logger.info("Fetching paper recommendations from PubMed...")
            recommended_papers = self._fetch_paper_recommendations(data)
            if recommended_papers:
                data['recommended_papers'] = recommended_papers

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
        """Generate extended abstract using LLM API.

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

        # Runtime check for OpenAI availability (more reliable than module-level check)
        openai_available = False
        try:
            from openai import OpenAI as OpenAIClient
            openai_available = True
        except ImportError:
            pass

        anthropic_available = False
        try:
            import anthropic as anthropic_module
            anthropic_available = True
        except ImportError:
            pass

        self.logger.info(f"LLM availability: OpenAI={openai_available}, Anthropic={anthropic_available}")
        self.logger.info(f"API keys set: OpenAI={bool(openai_key)}, Anthropic={bool(anthropic_key)}")

        use_openai = openai_available and openai_key
        use_anthropic = anthropic_available and anthropic_key and not use_openai

        if not use_openai and not use_anthropic:
            self.logger.warning("No LLM API available (need OPENAI_API_KEY or ANTHROPIC_API_KEY)")
            return self._generate_fallback_extended_abstract(data)

        llm_provider = "OpenAI (gpt-4o-mini)" if use_openai else "Anthropic (Claude)"
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

        # Paper recommendations info
        papers_info = ""
        recommended_papers = data.get('recommended_papers', {})
        if recommended_papers:
            classic_papers = recommended_papers.get('classic_papers', [])
            breakthrough_papers = recommended_papers.get('breakthrough_papers', [])

            classic_summary = []
            for p in classic_papers[:3]:
                title = p.get('title', '')[:80]
                citations = p.get('citation_count', 0)
                year = p.get('year', '')
                pmid = p.get('pmid', '')
                classic_summary.append(f"- {title}... ({year}, 인용 {citations}회, PMID: {pmid})")

            breakthrough_summary = []
            for p in breakthrough_papers[:3]:
                title = p.get('title', '')[:80]
                citations = p.get('citation_count', 0)
                year = p.get('year', '')
                pmid = p.get('pmid', '')
                velocity = p.get('citation_velocity', 0)
                breakthrough_summary.append(f"- {title}... ({year}, 인용 {citations}회, 연간 {velocity:.1f}회, PMID: {pmid})")

            papers_info = f"""
## 추천 논문 (참고 문헌)
총 {recommended_papers.get('paper_count', 0)}편의 논문이 분석 결과와 관련하여 추천되었습니다.

### 교과서적 논문 (Classic Studies) - {len(classic_papers)}편
해당 분야의 기반이 되는 필수 논문들 (100회 이상 인용):
{chr(10).join(classic_summary) if classic_summary else '없음'}

### 최신 주목 논문 (Emerging Research) - {len(breakthrough_papers)}편
빠르게 인용되고 있는 최근 연구:
{chr(10).join(breakthrough_summary) if breakthrough_summary else '없음'}
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

{papers_info}

## 요청 사항
위 분석 결과를 종합하여 학술 논문 수준의 Extended Abstract를 JSON 형식으로 작성해주세요.

반드시 아래 모든 섹션을 포함해야 합니다:
1. 배경 (Background) - 연구의 필요성과 목적
2. 방법 (Methods) - DESeq2, Network analysis, Pathway enrichment, Driver prediction 등
3. 결과 (Results) - DEG 수, Hub 유전자, 주요 Pathway, Driver 후보 등 핵심 수치 포함
4. Driver Gene Analysis - Known Driver와 Candidate Regulator 후보 구분하여 설명
5. 문헌 기반 해석 - RAG 분석 결과 요약
6. 추천 논문 - 교과서적 논문과 최신 연구 논문 간략히 언급 (PMID 포함)
7. 검증 제안 - 실험적 검증 방법 제안
8. 결론 (Conclusions) - 연구의 의의와 향후 방향

```json
{{
  "title": "한국어 제목 (암종, DEG 수, 주요 발견 포함)",
  "title_en": "English Title",
  "abstract_extended": "배경: ...\\n\\n방법: ...\\n\\n결과: ...\\n\\nDriver Gene Analysis: ...\\n\\n문헌 기반 해석: ...\\n\\n추천 논문: ...\\n\\n검증 제안: ...\\n\\n결론: ...",
  "key_findings": [
    "주요 발견 1 - 한국어로 작성 (유전자명, 경로명 등 학술 용어만 영어)",
    "주요 발견 2 - 예: FMO2는 N-Acetylornithine 조절을 통해 혈관신생을 촉진함",
    "주요 발견 3 - 예: 7,583개의 차등발현 유전자 중 10개의 Hub 유전자 식별",
    "주요 발견 4 - 예: PI3K-Akt 신호전달 경로가 유의하게 활성화됨",
    "주요 발견 5 - 예: 문헌 분석 결과 BRCA1이 DNA 손상 복구에 핵심 역할 수행",
    "주요 발견 6 - 예: qPCR 및 Western blot을 통한 발현 검증 권장"
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
6. abstract_extended는 최소 3000자 이상으로 매우 상세하게 작성 (A4 1페이지 이상)
7. key_findings는 10개 이상, 각 섹션에서 핵심 발견 포함
8. 각 섹션(배경, 방법, 결과, Driver Gene Analysis, 문헌 기반 해석, 추천 논문, 검증 제안, 결론)은 각각 4-6문장 이상으로 상세히 기술
9. 추천 논문 섹션에서는 위에 제공된 논문 목록을 반드시 활용하여 구체적인 제목과 PMID를 명시할 것 (예: "PMID 34976204의 연구는...")

key_findings 작성 지침 (매우 중요):
- 반드시 한국어로 작성하세요!
- 유전자명(예: FMO2, BRCA1), 경로명(예: PI3K-Akt), 분석 도구명(예: DESeq2)만 영어로 표기
- 영어 문장은 절대 금지: "Promotes Angiogenesis" (X) → "혈관신생을 촉진함" (O)
- 예시 형식:
  - "FMO2가 N-Acetylornithine 조절을 통해 혈관신생을 촉진함"
  - "{n_deg:,}개의 차등발현 유전자와 네트워크 기반 {len(hub_gene_names)}개 Hub 유전자 식별"
  - "ESR1, PGR 등 호르몬 수용체 유전자가 유의하게 발현 증가"

문체 지침 (매우 중요):
- 학술 논문이면서도 읽는 이를 사로잡는 매력적인 글쓰기를 해주세요
- 단순한 사실 나열이 아닌, 발견의 의미와 맥락을 이야기처럼 풀어가세요
- 각 발견이 왜 중요한지, 어떤 새로운 가능성을 열어주는지 설명하세요
- 데이터 뒤에 숨겨진 생물학적 스토리를 끌어내세요
- "~입니다", "~했습니다"의 단조로운 반복을 피하고, 문장 구조와 어미를 다양하게 사용하세요
- 마크다운 특수기호 사용 금지 (**, __, ##, [], () 등)
- 괄호 안의 영문 병기는 최소화하고 필요시 한글로 풀어 설명
- 독자가 "이 연구를 더 알고 싶다"는 마음이 들도록 흥미를 유발하세요
- 결론부에서는 이 연구가 환자 치료에 어떤 기여를 할 수 있는지 비전을 제시하세요
"""

        try:
            # Call LLM API with RAG context
            # Use Claude Opus 4 for Extended Abstract (highest quality writing)
            cancer_type = self.config.get('cancer_type', 'unknown')
            self.logger.info("Using Claude Opus 4 for Extended Abstract generation (premium quality)")
            response_text = call_llm_with_rag(
                prompt=prompt,
                cancer_type=cancer_type,
                key_genes=hub_gene_names[:10],
                max_tokens=8000,  # Increased for longer, more eloquent abstract
                logger=self.logger,
                use_opus=True  # Use Opus for best writing quality
            )

            if not response_text:
                self.logger.warning("LLM returned empty response")
                return self._generate_fallback_extended_abstract(data)

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

                self.logger.info(f"Extended abstract generated (RAG-based): {output_path}")
                return extended_abstract
            else:
                self.logger.warning("Could not extract JSON from LLM response")
                return self._generate_fallback_extended_abstract(data)

        except Exception as e:
            self.logger.error(f"Error generating extended abstract: {e}")
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

        # Runtime check for LLM availability
        openai_available = False
        try:
            from openai import OpenAI as OpenAIClient
            openai_available = True
        except ImportError:
            pass

        anthropic_available = False
        try:
            import anthropic as anthropic_module
            anthropic_available = True
        except ImportError:
            pass

        use_openai = openai_available and openai_key
        use_anthropic = anthropic_available and anthropic_key and not use_openai

        if not use_openai and not use_anthropic:
            self.logger.warning("No LLM API available for visualization interpretations")
            return self._generate_fallback_viz_interpretations(data)

        llm_provider = "OpenAI (gpt-4o-mini)" if use_openai else "Anthropic (Claude)"
        self.logger.info(f"Using {llm_provider} for visualization interpretations")

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

다음 JSON 형식으로 각 시각화에 대한 **매우 상세한** 해석을 제공해주세요. 각 항목은 3-5문장 이상으로 구체적으로 작성해야 합니다:

```json
{{
  "volcano_plot": {{
    "title": "Volcano Plot 해석",
    "summary": "3-4문장으로 전체적인 DEG 분포 특성과 의미 요약",
    "key_observations": [
      "DEG 분포 패턴에 대한 상세 관찰 (상향/하향 비율, 극단값 등)",
      "통계적 유의성 분포 특성 (-log10 p-value 분포)",
      "주요 상향조절 유전자 특성과 잠재적 역할",
      "주요 하향조절 유전자 특성과 잠재적 역할",
      "전체적인 발현 변화 양상이 시사하는 바"
    ],
    "biological_significance": "이러한 DEG 패턴이 암 생물학적 관점에서 의미하는 바를 3-4문장으로 상세히 설명. 종양 촉진/억제 경로, 대사 변화, 세포 주기 등과 연관지어 해석",
    "clinical_relevance": "DEG 결과의 임상적 의의 - 진단, 예후, 치료 타겟 관점에서 2-3문장",
    "interpretation_guide": "연구자가 Volcano Plot을 해석할 때 주의해야 할 점과 올바른 해석 방법을 3문장 이상으로 안내"
  }},
  "heatmap": {{
    "title": "발현 히트맵 해석",
    "summary": "3-4문장으로 전체적인 발현 패턴 특성 요약",
    "key_observations": [
      "샘플 간 클러스터링 패턴에 대한 상세 분석",
      "유전자 간 클러스터링 패턴과 공발현 그룹",
      "종양-정상 조직 간 발현 차이의 명확성",
      "특이적으로 높거나 낮은 발현을 보이는 유전자 그룹"
    ],
    "pattern_analysis": "발현 패턴의 생물학적 의미를 3-4문장으로 상세 분석. 공발현 유전자 그룹이 시사하는 기능적 모듈, 샘플 이질성 등 해석",
    "sample_clustering": "샘플 클러스터링 결과가 의미하는 바 - 종양 아형, 예후 그룹 등과의 연관성 2-3문장",
    "interpretation_guide": "히트맵 해석 시 색상 스케일, 정규화 방법, 클러스터링 알고리즘의 영향을 고려한 올바른 해석 방법 안내"
  }},
  "network_graph": {{
    "title": "유전자 상호작용 네트워크 해석",
    "summary": "3-4문장으로 네트워크의 전체적 구조와 특성 요약",
    "hub_gene_analysis": "각 Hub 유전자의 역할과 중요성을 4-5문장으로 상세 분석. 높은 연결성이 의미하는 생물학적 의미, 각 Hub 유전자의 알려진 기능과 암에서의 역할",
    "network_topology": "네트워크 구조 특성 (scale-free 특성, 모듈 구조, 연결 밀도 등)을 3문장으로 분석",
    "biological_implications": "네트워크 분석 결과가 시사하는 생물학적 의미 4-5문장. 핵심 조절 메커니즘, 취약점(druggable targets), 경로 간 crosstalk 등",
    "therapeutic_potential": "Hub 유전자를 표적으로 한 치료 전략 가능성 2-3문장",
    "interpretation_guide": "네트워크 그래프 해석 시 edge의 의미, node 크기/색상의 의미, 상관관계 기반 분석의 한계점 등 안내"
  }},
  "pca_plot": {{
    "title": "PCA 분석 해석",
    "summary": "3-4문장으로 샘플 분포와 분리도 요약",
    "separation_analysis": "종양-정상 조직 간 분리도를 4-5문장으로 상세 분석. 분리가 명확한지, 겹치는 샘플이 있는지, 이상치(outlier)가 있는지 등",
    "variance_explanation": "각 주성분(PC)이 설명하는 분산 비율의 의미, PC1/PC2가 반영하는 생물학적 변이 3문장",
    "sample_quality": "PCA 결과로부터 추론할 수 있는 샘플 품질 및 배치 효과 여부 2문장",
    "biological_meaning": "샘플 분포 패턴이 의미하는 생물학적 차이 (전사체 프로파일의 전반적 변화) 3문장",
    "interpretation_guide": "PCA 해석 시 분산 설명 비율, 샘플 레이블, 잠재적 교란 요인 고려 방법 안내"
  }},
  "pathway_barplot": {{
    "title": "Pathway 분석 해석",
    "summary": "3-4문장으로 전체적인 pathway 농축 결과 요약",
    "top_pathways": [
      "가장 유의한 pathway 1에 대한 상세 설명 - 해당 경로의 생물학적 기능, 암과의 관련성, 포함된 DEG 등",
      "가장 유의한 pathway 2에 대한 상세 설명",
      "가장 유의한 pathway 3에 대한 상세 설명",
      "전체 pathway 결과의 공통 주제/패턴"
    ],
    "functional_theme": "발굴된 pathway들의 전체적인 기능적 테마를 4-5문장으로 분석. 세포 증식, 면역, 대사, 신호전달 등 어떤 생물학적 과정이 주로 변화했는지",
    "therapeutic_implications": "pathway 분석 결과가 시사하는 치료적 함의 3-4문장. 표적 치료제 가능성, 약물 재목적화 후보, 병용 치료 전략 등",
    "cross_pathway_interactions": "주요 pathway 간의 상호작용과 crosstalk 2-3문장",
    "interpretation_guide": "Pathway 분석 해석 시 FDR 보정, 유전자 중복 계산, 데이터베이스 특성 등 고려사항 안내"
  }},
  "expression_boxplot": {{
    "title": "유전자 발현 분포 해석",
    "summary": "3-4문장으로 전체적인 발현 분포 특성 요약",
    "key_observations": [
      "종양-정상 간 발현 수준 차이의 정도와 일관성",
      "발현 분포의 변동성(분산) 차이",
      "이상치(outlier) 존재 여부와 의미",
      "전체적인 발현 변화 경향"
    ],
    "statistical_significance": "발현 차이의 통계적 유의성과 효과 크기(effect size) 해석 2-3문장",
    "biological_context": "발현 변화의 생물학적 맥락 - 해당 유전자(들)의 기능과 암에서의 역할 3문장",
    "interpretation_guide": "Boxplot 해석 시 정규화 방법, 샘플 수, 분포 가정 등 고려사항 안내"
  }}
}}
```

중요 지침:
1. 한국어로 작성하되, 학술적이고 전문적인 문체 사용
2. 각 시각화에 대해 **구체적인 숫자와 유전자명을 포함**하여 해석
3. 생물학적/의학적 맥락에서 깊이 있는 해석 제공
4. 연구자가 실제로 논문에 활용할 수 있는 수준의 상세한 설명
5. 각 항목은 최소 3문장 이상으로 작성
6. 임상적 관련성과 치료적 함의를 반드시 포함
"""

        try:
            # Call LLM API with RAG context (Claude preferred)
            cancer_type = self.config.get('cancer_type', 'unknown')
            hub_gene_names = []
            if hub_df is not None:
                for _, row in hub_df.head(10).iterrows():
                    gene_name = str(row.get('gene_id', row.get('gene_symbol', '')))
                    if gene_name and not gene_name.startswith('ENSG'):
                        hub_gene_names.append(gene_name)

            response_text = call_llm_with_rag(
                prompt=prompt,
                cancer_type=cancer_type,
                key_genes=hub_gene_names,
                max_tokens=8000,  # Increased for detailed AI analysis
                logger=self.logger
            )

            if not response_text:
                self.logger.warning("LLM returned empty response for viz interpretations")
                return self._generate_fallback_viz_interpretations(data)

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

                self.logger.info(f"Visualization interpretations generated (RAG-based): {output_path}")
                return viz_interpretations
            else:
                self.logger.warning("Could not extract JSON from LLM response")
                return self._generate_fallback_viz_interpretations(data)

        except Exception as e:
            self.logger.error(f"Error generating visualization interpretations: {e}")
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
        """Generate comprehensive research recommendations using LLM + DGIdb.

        Creates actionable next-step recommendations including:
        - Therapeutic target candidates (druggable genes) - DGIdb verified
        - Drug repurposing suggestions (DGIdb-based) - real drug-gene interactions
        - Experimental validation priorities
        - Future research directions
        - Biomarker development opportunities
        """
        # Try OpenAI first (cheaper), then Anthropic as fallback
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        # Runtime check for LLM availability
        openai_available = False
        try:
            from openai import OpenAI as OpenAIClient
            openai_available = True
        except ImportError:
            pass

        anthropic_available = False
        try:
            import anthropic as anthropic_module
            anthropic_available = True
        except ImportError:
            pass

        use_openai = openai_available and openai_key
        use_anthropic = anthropic_available and anthropic_key and not use_openai

        if not use_openai and not use_anthropic:
            self.logger.warning("No LLM API available for research recommendations")
            return self._generate_fallback_research_recommendations(data)

        llm_provider = "OpenAI (gpt-4o-mini)" if use_openai else "Anthropic (Claude)"
        self.logger.info(f"Generating research recommendations via {llm_provider}...")

        # Query DGIdb for verified drug-gene interactions
        dgidb_info = self._query_dgidb_for_recommendations(data)
        dgidb_section = dgidb_info.get("prompt_section", "")

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

## DGIdb 약물-유전자 상호작용 (검증된 데이터)
{dgidb_section if dgidb_section else '(DGIdb 조회 실패 또는 매칭 결과 없음)'}

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
4. **drug_repurposing 섹션은 반드시 DGIdb 데이터에서 제공된 약물만 사용** (위 DGIdb 섹션 참조)
5. DGIdb에서 검증된 약물-유전자 상호작용을 우선 추천
6. therapeutic_targets는 DGIdb에서 druggable로 분류된 유전자를 우선
7. 우선순위와 근거를 명확히 제시
8. 현실적인 timeline과 resource 제안
9. therapeutic_targets의 high_priority는 3-5개, medium_priority는 3-5개
10. drug_repurposing candidates는 DGIdb 매칭 기준 3-5개 (검증되지 않은 약물 추천 금지)
11. 각 섹션에 최소 2개 이상의 구체적 항목 포함
"""

        try:
            # Call LLM API with RAG context (Claude preferred)
            cancer_type = self.config.get('cancer_type', 'unknown')

            # Get key genes for RAG context
            hub_gene_names = []
            hub_df = data.get('hub_genes_df')
            if hub_df is not None:
                for _, row in hub_df.head(10).iterrows():
                    gene_name = str(row.get('gene_id', row.get('gene_symbol', '')))
                    if gene_name and not gene_name.startswith('ENSG'):
                        hub_gene_names.append(gene_name)

            response_text = call_llm_with_rag(
                prompt=prompt,
                cancer_type=cancer_type,
                key_genes=hub_gene_names,
                max_tokens=6000,
                logger=self.logger
            )

            if not response_text:
                self.logger.warning("LLM returned empty response for recommendations")
                return self._generate_fallback_research_recommendations(data)

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

                self.logger.info(f"Research recommendations generated (RAG-based): {output_path}")
                return recommendations
            else:
                self.logger.warning("Could not extract JSON from LLM response")
                return self._generate_fallback_research_recommendations(data)

        except Exception as e:
            self.logger.error(f"Error generating research recommendations: {e}")
            return self._generate_fallback_research_recommendations(data)

    def _query_dgidb_for_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query DGIdb for verified drug-gene interactions.

        Returns a dictionary with:
        - prompt_section: Formatted string for LLM prompt
        - interactions: Raw interaction data
        - druggable_genes: List of druggable gene symbols
        """
        try:
            from ..rag.dgidb_client import DGIdbClient, get_therapeutic_targets
        except ImportError:
            self.logger.warning("DGIdb client not available")
            return {"prompt_section": "", "interactions": {}, "druggable_genes": []}

        # Collect gene symbols from hub genes and drivers
        gene_symbols = set()

        hub_df = data.get('hub_genes_df')
        if hub_df is not None:
            gene_col = 'gene_symbol' if 'gene_symbol' in hub_df.columns else 'gene_id'
            for _, row in hub_df.head(20).iterrows():
                gene = str(row.get(gene_col, row.get('gene_id', '')))
                if gene and not gene.startswith('ENSG'):
                    gene_symbols.add(gene)

        driver_known = data.get('driver_known', [])
        for d in driver_known[:10]:
            gene = d.get('gene_symbol', '')
            if gene:
                gene_symbols.add(gene)

        driver_novel = data.get('driver_novel', [])
        for d in driver_novel[:10]:
            gene = d.get('gene_symbol', '')
            if gene:
                gene_symbols.add(gene)

        if not gene_symbols:
            return {"prompt_section": "", "interactions": {}, "druggable_genes": []}

        self.logger.info(f"Querying DGIdb for {len(gene_symbols)} genes...")

        try:
            client = DGIdbClient(timeout=30)
            interactions = client.get_drug_interactions(list(gene_symbols))
            categories = client.get_gene_categories(list(gene_symbols))

            # Build prompt section
            prompt_lines = []
            druggable_genes = []

            for gene, drugs in interactions.items():
                if drugs:
                    druggable_genes.append(gene)
                    drug_info = []
                    for drug in drugs[:3]:  # Top 3 drugs per gene
                        int_types = ", ".join(drug.interaction_types[:2]) if drug.interaction_types else "unknown"
                        sources = ", ".join(drug.sources[:2]) if drug.sources else ""
                        drug_info.append(f"{drug.drug_name} ({int_types}; {sources})")
                    prompt_lines.append(f"- {gene}: {'; '.join(drug_info)}")

            # Add category info for genes without drugs
            for gene, cat in categories.items():
                if gene not in druggable_genes and cat.is_druggable:
                    cat_str = ", ".join(cat.categories[:2]) if cat.categories else "druggable"
                    prompt_lines.append(f"- {gene}: (druggable - {cat_str}, 약물 조회 필요)")
                    druggable_genes.append(gene)

            self.logger.info(f"DGIdb: Found {len(druggable_genes)} druggable genes with {sum(len(v) for v in interactions.values())} drug interactions")

            return {
                "prompt_section": "\n".join(prompt_lines) if prompt_lines else "(매칭 결과 없음)",
                "interactions": {g: [d.to_dict() for d in drugs] for g, drugs in interactions.items()},
                "druggable_genes": druggable_genes
            }

        except Exception as e:
            self.logger.error(f"DGIdb query error: {e}")
            return {"prompt_section": f"(DGIdb 조회 실패: {e})", "interactions": {}, "druggable_genes": []}

    def _generate_fallback_research_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template-based research recommendations when LLM API is unavailable.

        Uses DGIdb for verified drug-gene interactions to avoid hallucination.
        """
        deg_df = data.get('deg_significant_df')
        hub_df = data.get('hub_genes_df')
        pathway_df = data.get('pathway_summary_df')
        driver_known = data.get('driver_known', [])
        driver_novel = data.get('driver_novel', [])
        cancer_type = self.config.get('cancer_type', 'cancer').replace('_', ' ').title()

        # Query DGIdb for verified drug-gene interactions
        dgidb_info = self._query_dgidb_for_recommendations(data)
        dgidb_interactions = dgidb_info.get("interactions", {})
        druggable_genes = dgidb_info.get("druggable_genes", [])

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
                "description": f"DGIdb (Drug-Gene Interaction Database)에서 검증된 약물-유전자 상호작용 정보입니다. {len(dgidb_interactions)}개 유전자에서 약물 상호작용이 확인되었습니다.",
                "candidates": self._build_dgidb_drug_candidates(dgidb_interactions, cancer_type)
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

    def _build_dgidb_drug_candidates(
        self,
        dgidb_interactions: Dict[str, List[Dict]],
        cancer_type: str
    ) -> List[Dict[str, Any]]:
        """Build drug repurposing candidates from DGIdb interactions.

        Returns verified drug-gene interactions only, avoiding hallucination.
        """
        candidates = []

        for gene, drugs in dgidb_interactions.items():
            if not drugs:
                continue

            for drug in drugs[:2]:  # Top 2 drugs per gene
                drug_name = drug.get("drug_name", "Unknown")
                int_types = drug.get("interaction_types", [])
                sources = drug.get("sources", [])
                pmids = drug.get("pmids", [])

                # Determine clinical status based on sources
                clinical_status = "연구 단계"
                if "DrugBank" in sources or "FDA" in " ".join(sources):
                    clinical_status = "FDA 승인 (타 적응증)"
                elif "CIViC" in sources or "OncoKB" in sources:
                    clinical_status = "임상시험 진행/완료"
                elif "PharmGKB" in sources:
                    clinical_status = "약물유전체학 근거 있음"

                # Build interaction type string
                int_type_str = ", ".join(int_types[:2]) if int_types else "상호작용"

                candidates.append({
                    "drug": drug_name,
                    "target_gene": gene,
                    "original_indication": f"DGIdb 출처: {', '.join(sources[:2])}" if sources else "조회 필요",
                    "repurposing_rationale": f"{gene} {int_type_str} - {cancer_type}에서 유의한 발현 변화 관찰",
                    "clinical_status": clinical_status,
                    "evidence": f"PMID: {', '.join(pmids[:2])}" if pmids else "DGIdb 데이터베이스"
                })

                if len(candidates) >= 5:
                    break

            if len(candidates) >= 5:
                break

        # If no DGIdb results, return placeholder
        if not candidates:
            candidates.append({
                "drug": "DGIdb 매칭 결과 없음",
                "target_gene": "-",
                "original_indication": "해당 유전자에 대한 약물 상호작용이 DGIdb에서 발견되지 않았습니다",
                "repurposing_rationale": "수동 문헌 검색 또는 다른 데이터베이스 조회를 권장합니다",
                "clinical_status": "-"
            })

        return candidates

    def _fetch_paper_recommendations(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch paper recommendations with citation-based quality filtering.

        Searches PubMed and enriches with Semantic Scholar citation data to provide:
        - Classic papers: High-citation foundational studies (100+ citations, 3+ years)
        - Breakthrough papers: Recent rapidly-cited research (1-2 years, high velocity)

        Returns:
            Dictionary containing recommended papers organized by type
        """
        try:
            from ..rag.paper_recommender import recommend_papers_enhanced_sync, recommend_papers_sync
            use_enhanced = True
        except ImportError:
            try:
                from ..rag.paper_recommender import recommend_papers_sync
                use_enhanced = False
            except ImportError:
                self.logger.warning("Paper recommender module not available")
                return None

        # Get cancer type
        cancer_type = self.config.get('cancer_type', 'unknown')
        if cancer_type == 'unknown':
            cancer_type = data.get('cancer_type', 'cancer')

        # Get hub genes - prefer integrated_gene_table which has proper gene symbols
        hub_genes = []

        # First try integrated_gene_table (has proper gene symbols)
        integrated_df = data.get('integrated_gene_table_df')
        if integrated_df is not None and len(integrated_df) > 0:
            # Filter for hub genes with proper symbols
            if 'is_hub' in integrated_df.columns and 'gene_symbol' in integrated_df.columns:
                hub_subset = integrated_df[integrated_df['is_hub'] == True]
                if len(hub_subset) > 0:
                    hub_genes = [str(g) for g in hub_subset['gene_symbol'].head(10).tolist()
                                if g and not str(g).startswith('ENSG')]
            # If no is_hub column, just get top genes with proper symbols
            if not hub_genes and 'gene_symbol' in integrated_df.columns:
                hub_genes = [str(g) for g in integrated_df['gene_symbol'].head(10).tolist()
                            if g and not str(g).startswith('ENSG')]

        # Fallback to hub_genes_df
        if not hub_genes:
            hub_df = data.get('hub_genes_df')
            if hub_df is not None and len(hub_df) > 0:
                gene_col = None
                for col in ['gene_symbol', 'gene_id', 'gene_name']:
                    if col in hub_df.columns:
                        gene_col = col
                        break
                if gene_col:
                    hub_genes = [str(g) for g in hub_df[gene_col].head(10).tolist()
                                if g and not str(g).startswith('ENSG')]

        # Get pathways
        pathways = []
        pathway_df = data.get('pathway_summary_df')
        if pathway_df is not None and len(pathway_df) > 0:
            term_col = None
            for col in ['Term', 'term', 'pathway', 'Pathway', 'name', 'Name']:
                if col in pathway_df.columns:
                    term_col = col
                    break
            if term_col:
                pathways = pathway_df[term_col].head(5).tolist()

        if not hub_genes:
            self.logger.warning("No hub genes found for paper recommendations")
            return None

        self.logger.info(f"Fetching papers for cancer_type={cancer_type}, genes={hub_genes[:5]}")

        try:
            run_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir

            if use_enhanced:
                # Use enhanced version with citation-based quality filtering
                self.logger.info("Using enhanced paper recommendations with citation filtering")
                result = recommend_papers_enhanced_sync(
                    cancer_type=cancer_type,
                    hub_genes=hub_genes,
                    pathways=pathways,
                    output_dir=run_dir,
                    max_papers=6,
                    quality_filter=True,
                    balance_classic_breakthrough=True
                )

                if result and result.get('papers'):
                    classic_count = result.get('classic_count', 0)
                    breakthrough_count = result.get('breakthrough_count', 0)
                    self.logger.info(f"Retrieved {result.get('paper_count', 0)} papers "
                                   f"(Classic: {classic_count}, Breakthrough: {breakthrough_count})")
                    return result
            else:
                # Fallback to basic version
                papers = recommend_papers_sync(
                    cancer_type=cancer_type,
                    hub_genes=hub_genes,
                    pathways=pathways,
                    output_dir=run_dir,
                    max_papers=5
                )

                if papers:
                    result = {
                        "cancer_type": cancer_type,
                        "search_genes": hub_genes[:5],
                        "search_pathways": pathways[:3] if pathways else [],
                        "paper_count": len(papers),
                        "papers": papers
                    }
                    self.logger.info(f"Retrieved {len(papers)} paper recommendations (basic mode)")
                    return result

            self.logger.warning("No papers found from search")
            return None

        except Exception as e:
            self.logger.error(f"Error fetching paper recommendations: {e}")
            import traceback
            traceback.print_exc()
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
