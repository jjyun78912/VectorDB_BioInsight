"""
Single-Cell RNA-seq Report Agent

Generates an interactive HTML report for single-cell analysis results.

Input:
- adata.h5ad: Annotated data from SingleCellAgent
- cluster_markers.csv: Marker genes per cluster
- cell_composition.csv: Cell type composition
- umap_coordinates.csv: UMAP embeddings
- figures/: Visualization outputs

Output:
- report.html: Interactive HTML report
- report_data.json: Report data for API consumption
"""

import json
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ..utils.base_agent import BaseAgent

# Optional imports
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


class SingleCellReportAgent(BaseAgent):
    """Agent for generating single-cell analysis HTML reports."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "report_title": "Single-Cell RNA-seq Analysis Report",
            "author": "BioInsight AI Pipeline",
            "cancer_type": "unknown",
            "tissue_type": "unknown",
            "embed_figures": True,
            "max_markers_per_cluster": 20,
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__("agent_singlecell_report", input_dir, output_dir, merged_config)

    def validate_inputs(self) -> bool:
        """Validate required input files."""
        # Check for key outputs from SingleCellAgent
        required = ["cluster_markers.csv", "cell_composition.csv", "umap_coordinates.csv"]
        for f in required:
            if not (self.input_dir / f).exists():
                self.logger.warning(f"Missing file: {f}")
        return True  # Allow partial data

    def _load_all_data(self) -> Dict[str, Any]:
        """Load all data from SingleCellAgent outputs."""
        data = {}

        # Load CSV files
        csv_files = [
            "cluster_markers.csv",
            "cell_composition.csv",
            "umap_coordinates.csv",
            "cell_metadata.csv",
            "top_markers_summary.csv",
            # NEW: Enhanced analysis outputs
            "driver_genes.csv",
            "cluster_pathways.csv",
            "trajectory_pseudotime.csv",
            "cell_interactions.csv",
            # Advanced analysis outputs
            "tme_composition.csv",
            "tme_signature_scores.csv",
            "ploidy_by_celltype.csv",
            "grn_edges.csv",
            "tf_activity_scores.csv",
            "master_regulators.csv",
        ]

        for csv_name in csv_files:
            csv_path = self.input_dir / csv_name
            if csv_path.exists():
                try:
                    data[csv_name.replace('.csv', '')] = pd.read_csv(csv_path)
                    self.logger.info(f"Loaded {csv_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {csv_name}: {e}")

        # Load h5ad summary if scanpy available
        h5ad_path = self.input_dir / "adata.h5ad"
        if h5ad_path.exists() and HAS_SCANPY:
            try:
                adata = sc.read_h5ad(h5ad_path)
                data['adata_summary'] = {
                    'n_cells': adata.n_obs,
                    'n_genes': adata.n_vars,
                    'n_clusters': len(adata.obs['cluster'].unique()) if 'cluster' in adata.obs else 0,
                    'n_celltypes': len(adata.obs['cell_type'].unique()) if 'cell_type' in adata.obs else 0,
                    'obs_columns': list(adata.obs.columns),
                    'var_columns': list(adata.var.columns),
                }
                self.logger.info(f"Loaded adata summary: {data['adata_summary']['n_cells']} cells")
            except Exception as e:
                self.logger.warning(f"Failed to load adata.h5ad: {e}")

        # Load figures
        data['figures'] = {}
        figures_dir = self.input_dir / "figures"
        if figures_dir.exists():
            for fig_path in figures_dir.glob("*.png"):
                if self.config.get("embed_figures", True):
                    with open(fig_path, 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                        data['figures'][fig_path.stem] = f"data:image/png;base64,{b64}"
                else:
                    data['figures'][fig_path.stem] = str(fig_path.relative_to(self.output_dir))
                self.logger.info(f"Loaded figure: {fig_path.name}")

        # NEW: Load cancer prediction JSON
        pred_path = self.input_dir / "pseudobulk_prediction.json"
        if pred_path.exists():
            try:
                with open(pred_path, 'r', encoding='utf-8') as f:
                    data['cancer_prediction'] = json.load(f)
                self.logger.info("Loaded cancer prediction")
            except Exception as e:
                self.logger.warning(f"Failed to load cancer prediction: {e}")

        return data

    def _generate_css(self) -> str:
        """Generate CSS styles for the report."""
        return """
:root {
    --bg-primary: #fafafa;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f5f5f5;
    --border-light: #e5e5e5;
    --text-primary: #171717;
    --text-secondary: #525252;
    --text-muted: #a3a3a3;
    --accent-blue: #2563eb;
    --accent-blue-light: #eff6ff;
    --accent-green: #059669;
    --accent-purple: #7c3aed;
    --accent-orange: #d97706;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
    background: var(--bg-primary);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
}

/* Header */
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 48px 24px;
    text-align: center;
    margin-bottom: 32px;
}

.header h1 {
    font-size: 28px;
    margin-bottom: 8px;
}

.header .subtitle {
    font-size: 16px;
    opacity: 0.9;
}

.header .meta {
    margin-top: 16px;
    font-size: 13px;
    opacity: 0.8;
}

/* Summary Cards */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}

.summary-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.summary-card .value {
    font-size: 36px;
    font-weight: 700;
    color: var(--accent-blue);
    line-height: 1;
}

.summary-card .label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 8px;
}

/* AI Interpretation Cards */
.interpretation-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 16px;
}

.interpretation-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 20px;
    transition: box-shadow 0.2s;
}

.interpretation-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.interpretation-card.full-width {
    grid-column: 1 / -1;
}

.interpretation-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.interpretation-icon {
    font-size: 20px;
}

.interpretation-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
}

.interpretation-content {
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-secondary);
}

.interpretation-note {
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
    margin-top: 16px;
    padding: 12px;
    background: #fef3c7;
    border-radius: 8px;
}

/* Sections */
.section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}

.section h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--accent-blue);
}

.section h3 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Figures */
.figure-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: stretch;
}

.figure-grid > .figure-panel {
    flex: 1 1 400px;
    max-width: calc(50% - 12px);
    display: flex;
    flex-direction: column;
}

.figure-panel {
    background: transparent;
}

.figure-title {
    text-align: center;
    padding: 0 0 12px 0;
    font-weight: 600;
    font-size: 14px;
    color: var(--text-primary);
}

.figure-panel .title {
    padding: 0 0 8px 0;
    font-weight: 600;
    font-size: 13px;
}

.figure-panel .figure-content {
    padding: 0;
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
}

.figure-panel img {
    width: 100%;
    max-height: 450px;
    object-fit: contain;
    object-position: top center;
}

/* Cell Type Analysis Section - Special Layout */
.celltype-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
    align-items: start;
}

.celltype-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}

.celltype-title {
    text-align: center;
    font-weight: 600;
    font-size: 13px;
    color: var(--text-primary);
    margin-bottom: 12px;
    padding: 6px 16px;
    background: var(--bg-tertiary);
    border-radius: 4px;
}

.celltype-figure {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: flex-start;
}

.celltype-figure img {
    max-width: 100%;
    height: auto;
    object-fit: contain;
}

/* Tables */
.table-wrapper {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border-light);
}

th {
    background: var(--bg-tertiary);
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
}

tr:hover {
    background: var(--bg-tertiary);
}

/* Cluster Colors */
.cluster-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

.cluster-0 { background: #fee2e2; color: #991b1b; }
.cluster-1 { background: #dbeafe; color: #1e40af; }
.cluster-2 { background: #dcfce7; color: #166534; }
.cluster-3 { background: #fef3c7; color: #92400e; }
.cluster-4 { background: #f3e8ff; color: #6b21a8; }
.cluster-5 { background: #ffedd5; color: #9a3412; }
.cluster-6 { background: #cffafe; color: #0e7490; }
.cluster-7 { background: #fce7f3; color: #9d174d; }
.cluster-8 { background: #e0e7ff; color: #3730a3; }
.cluster-9 { background: #fef9c3; color: #854d0e; }
.cluster-10 { background: #d1fae5; color: #065f46; }
.cluster-11 { background: #ffe4e6; color: #be123c; }
.cluster-12 { background: #e0f2fe; color: #0369a1; }
.cluster-13 { background: #fae8ff; color: #86198f; }
.cluster-14 { background: #ecfccb; color: #3f6212; }
.cluster-15 { background: #fecaca; color: #b91c1c; }

/* Accordion Styles */
.accordion-container {
    margin-top: 24px;
}

.accordion-item {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
    background: var(--bg-secondary);
}

.accordion-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    cursor: pointer;
    transition: background 0.2s;
    user-select: none;
}

.accordion-header:hover {
    background: var(--bg-tertiary);
}

.accordion-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.accordion-header-right {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 12px;
    color: var(--text-muted);
}

.accordion-chevron {
    transition: transform 0.2s;
    color: var(--text-muted);
}

.accordion-item.open .accordion-chevron {
    transform: rotate(180deg);
}

.accordion-content {
    display: none;
    padding: 0 16px 16px 16px;
}

.accordion-item.open .accordion-content {
    display: block;
}

.accordion-table {
    width: 100%;
    font-size: 13px;
}

.accordion-table th,
.accordion-table td {
    padding: 8px 12px;
    text-align: left;
}

.accordion-table th {
    background: var(--bg-tertiary);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-muted);
}

.accordion-table tr:hover {
    background: var(--bg-tertiary);
}

.show-more-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
    padding: 8px 16px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
}

.show-more-btn:hover {
    background: var(--primary-dark);
}

.hidden-rows {
    display: none;
}

.hidden-rows.visible {
    display: table-row-group;
}

.marker-count {
    font-weight: 600;
    color: var(--text-primary);
}

.top-gene {
    background: #fef3c7;
    color: #92400e;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}

/* Composition Chart */
.composition-chart {
    display: flex;
    height: 32px;
    border-radius: 8px;
    overflow: hidden;
    margin: 16px 0;
}

.composition-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 11px;
    font-weight: 600;
    min-width: 30px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 12px;
}

/* Navigation */
.nav {
    position: sticky;
    top: 0;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-light);
    padding: 12px 24px;
    z-index: 100;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}

.nav a {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    padding: 4px 8px;
    border-radius: 4px;
}

.nav a:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

/* Clinical & Research Section Styles */
.cell-gene {
    font-weight: 600;
    color: var(--accent-blue);
}

.evidence-badge, .priority-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
}

.evidence-badge.high, .priority-badge.high {
    background: #dcfce7;
    color: #166534;
}

.evidence-badge.medium, .priority-badge.medium {
    background: #fef3c7;
    color: #92400e;
}

.evidence-badge.low, .priority-badge.low {
    background: #fee2e2;
    color: #991b1b;
}

.tme-badge.hot {
    background: #fee2e2;
    color: #991b1b;
}

.tme-badge.altered {
    background: #fef3c7;
    color: #92400e;
}

.tme-badge.cold {
    background: #dbeafe;
    color: #1e40af;
}

.ai-box {
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.ai-box.orange {
    background: #fff7ed;
    border-left: 4px solid #d97706;
}

.ai-box.green {
    background: #f0fdf4;
    border-left: 4px solid #059669;
}

.ai-box.blue {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
}

.rec-intro {
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    margin-bottom: 24px;
}
"""

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate the HTML report."""
        # Extract data
        adata_summary = data.get('adata_summary', {})
        markers_df = data.get('cluster_markers')
        composition_df = data.get('cell_composition')
        umap_df = data.get('umap_coordinates')
        figures = data.get('figures', {})
        # NEW: Enhanced analysis data
        driver_genes_df = data.get('driver_genes')
        cluster_pathways_df = data.get('cluster_pathways')
        cancer_prediction = data.get('cancer_prediction', {})
        trajectory_df = data.get('trajectory_pseudotime')
        interactions_df = data.get('cell_interactions')

        # Generate LLM interpretation
        llm_interpretation = self._generate_llm_interpretation(data)

        # Calculate summary stats
        n_cells = adata_summary.get('n_cells', 0)
        n_genes = adata_summary.get('n_genes', 0)
        n_clusters = adata_summary.get('n_clusters', 0)
        n_celltypes = adata_summary.get('n_celltypes', 0)
        n_markers = len(markers_df) if markers_df is not None else 0
        n_drivers = len(driver_genes_df) if driver_genes_df is not None else 0

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Cancer prediction info
        predicted_cancer = cancer_prediction.get('predicted_cancer', 'N/A')
        predicted_cancer_kr = cancer_prediction.get('predicted_cancer_korean', '알 수 없음')
        confidence = cancer_prediction.get('confidence', 0)
        confidence_pct = f"{confidence:.1%}" if confidence else "N/A"

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.get('report_title', 'Single-Cell Report')}</title>
    <style>
{self._generate_css()}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.get('report_title', 'Single-Cell RNA-seq Analysis Report')}</h1>
        <p class="subtitle">Powered by Scanpy | BioInsight AI Pipeline</p>
        <p class="meta">생성일: {timestamp} | 예측 암종: {predicted_cancer_kr} ({predicted_cancer})</p>
    </div>

    <nav class="nav">
        <a href="#summary">요약</a>
        <a href="#interpretation">AI 해석</a>
        <a href="#prediction">암종 예측</a>
        <a href="#umap">UMAP</a>
        <a href="#composition">세포 구성</a>
        <a href="#markers">마커 유전자</a>
        <a href="#drivers">드라이버 유전자</a>
        <a href="#pathways">Pathway</a>
        <a href="#trajectory">Trajectory</a>
        <a href="#interactions">Cell-Cell</a>
        <a href="#tme">TME</a>
        <a href="#grn">GRN</a>
        <a href="#ploidy">Ploidy</a>
        <a href="#clinical">임상</a>
        <a href="#followup">검증실험</a>
        <a href="#methods">방법론</a>
        <a href="#research">연구추천</a>
        <a href="#qc">QC</a>
    </nav>

    <div class="container">
        <!-- Summary Section -->
        <section id="summary" class="section">
            <h2>분석 요약</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="value">{n_cells:,}</div>
                    <div class="label">Total Cells</div>
                </div>
                <div class="summary-card">
                    <div class="value">{n_genes:,}</div>
                    <div class="label">Total Genes</div>
                </div>
                <div class="summary-card">
                    <div class="value">{n_clusters}</div>
                    <div class="label">Clusters</div>
                </div>
                <div class="summary-card">
                    <div class="value">{n_celltypes}</div>
                    <div class="label">Cell Types</div>
                </div>
                <div class="summary-card">
                    <div class="value">{n_markers:,}</div>
                    <div class="label">Marker Genes</div>
                </div>
                <div class="summary-card">
                    <div class="value">{n_drivers}</div>
                    <div class="label">Driver Genes</div>
                </div>
            </div>
        </section>

        <!-- AI Interpretation Section -->
        <section id="interpretation" class="section">
            <h2>🤖 AI 분석 해석</h2>
            <div class="interpretation-grid">
                <div class="interpretation-card">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">📊</span>
                        <span class="interpretation-title">Executive Summary</span>
                    </div>
                    <div class="interpretation-content">{llm_interpretation.get('executive_summary', '해석을 생성할 수 없습니다.')}</div>
                </div>
                <div class="interpretation-card">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">🧬</span>
                        <span class="interpretation-title">세포 구성 해석</span>
                    </div>
                    <div class="interpretation-content">{llm_interpretation.get('cell_composition', '')}</div>
                </div>
                <div class="interpretation-card">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">🔬</span>
                        <span class="interpretation-title">마커 유전자 해석</span>
                    </div>
                    <div class="interpretation-content">{llm_interpretation.get('marker_interpretation', '')}</div>
                </div>
                <div class="interpretation-card">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">🛡️</span>
                        <span class="interpretation-title">TME/면역 해석</span>
                    </div>
                    <div class="interpretation-content">{llm_interpretation.get('tme_interpretation', '')}</div>
                </div>
                <div class="interpretation-card full-width">
                    <div class="interpretation-header">
                        <span class="interpretation-icon">💊</span>
                        <span class="interpretation-title">임상적 함의</span>
                    </div>
                    <div class="interpretation-content">{llm_interpretation.get('clinical_implications', '')}</div>
                </div>
            </div>
            <p class="interpretation-note">
                ⚠️ 본 해석은 AI 모델에 의해 자동 생성되었으며, 최종 판단은 전문가의 검토가 필요합니다.
            </p>
        </section>

        <!-- NEW: Cancer Prediction Section -->
        <section id="prediction" class="section">
            <h2>🔬 암종 예측 (Pseudo-bulk ML)</h2>
            {self._cancer_prediction_html(cancer_prediction)}
        </section>

        <!-- Cell Type Analysis Section (UMAP + Bar Chart) -->
        <section id="celltype" class="section">
            <h2>세포 유형 분석 (Cell Type Analysis)</h2>
            <div class="celltype-grid">
                <div class="celltype-panel">
                    <div class="celltype-title">Cell Types (UMAP)</div>
                    <div class="celltype-figure">
                        {f'<img src="{figures["umap_celltypes"]}" alt="Cell Types UMAP">' if 'umap_celltypes' in figures else '<p>Figure not available</p>'}
                    </div>
                </div>
                <div class="celltype-panel">
                    <div class="celltype-title">Cell Type Composition</div>
                    <div class="celltype-figure">
                        {f'<img src="{figures["celltype_barchart"]}" alt="Cell Type Composition">' if 'celltype_barchart' in figures else '<p>Figure not available</p>'}
                    </div>
                </div>
            </div>
        </section>

        <!-- Marker Genes Section -->
        <section id="markers" class="section">
            <h2>클러스터별 마커 유전자</h2>
            <div class="figure-grid">
                {self._figure_html(figures, 'heatmap_markers', 'Marker Heatmap')}
                {self._figure_html(figures, 'dotplot_markers', 'Marker Dotplot')}
            </div>
            {self._markers_table_html(markers_df)}
        </section>

        <!-- NEW: Driver Genes Section -->
        <section id="drivers" class="section">
            <h2>🎯 드라이버 유전자 (COSMIC/OncoKB)</h2>
            {self._driver_genes_html(driver_genes_df)}
        </section>

        <!-- NEW: Pathway Section -->
        <section id="pathways" class="section">
            <h2>🧬 클러스터별 Pathway 분석</h2>
            {self._cluster_pathways_html(cluster_pathways_df)}
        </section>

        <!-- NEW: Trajectory Section -->
        <section id="trajectory" class="section">
            <h2>🔄 Trajectory 분석 (Pseudotime)</h2>
            {self._trajectory_html(trajectory_df, figures)}
        </section>

        <!-- NEW: Cell-Cell Interaction Section -->
        <section id="interactions" class="section">
            <h2>🔗 Cell-Cell Interaction</h2>
            {self._interactions_html(interactions_df)}
        </section>

        <!-- NEW: TME Analysis Section -->
        <section id="tme" class="section">
            <h2>🏔️ 종양 미세환경 (TME) 분석</h2>
            {self._tme_analysis_html(data.get('tme_composition'), data.get('tme_signature_scores'))}
        </section>

        <!-- NEW: Gene Regulatory Network Section -->
        <section id="grn" class="section">
            <h2>🧬 유전자 조절 네트워크 (GRN)</h2>
            {self._grn_analysis_html(data.get('master_regulators'), data.get('tf_activity_scores'))}
        </section>

        <!-- NEW: Ploidy Inference Section -->
        <section id="ploidy" class="section">
            <h2>📊 Ploidy 추론 (악성세포 감별)</h2>
            {self._ploidy_analysis_html(data.get('ploidy_by_celltype', data.get('cnv_by_celltype')))}
        </section>

        <!-- QC Section -->
        <section id="qc" class="section">
            <h2>Quality Control</h2>
            <div class="figure-grid">
                {self._figure_html(figures, 'violin_qc', 'QC Metrics')}
            </div>
        </section>

        <!-- Clinical Implications Section -->
        <section id="clinical" class="section">
            <h2>💊 임상적 시사점 (Clinical Implications)</h2>
            {self._clinical_implications_html(data)}
        </section>

        <!-- Follow-up Experiments Section -->
        <section id="followup" class="section">
            <h2>🔬 검증 실험 제안 (Suggested Follow-up Experiments)</h2>
            {self._followup_experiments_html(data)}
        </section>

        <!-- Methods Section with Quality Scorecard -->
        <section id="methods" class="section">
            <h2>📊 분석 방법 및 품질 점수 (Methods & Quality Scorecard)</h2>
            {self._methods_scorecard_html(data)}
        </section>

        <!-- Research Recommendations Section -->
        <section id="research" class="section">
            <h2>🔭 후속 연구 추천 (Research Recommendations)</h2>
            {self._research_recommendations_html(data)}
        </section>
    </div>

    <footer class="footer">
        <p>Generated by BioInsight AI | Single-Cell RNA-seq Pipeline v2.0</p>
        <p>Report generated at {timestamp}</p>
        <p style="margin-top:8px;font-size:11px;color:#999;">⚠️ 암종 예측은 참고용이며, 진단 목적으로 사용할 수 없습니다.</p>
    </footer>
</body>
</html>
"""
        return html

    def _cancer_prediction_html(self, prediction: Optional[Dict]) -> str:
        """Generate HTML for cancer prediction section."""
        if not prediction:
            return '<p style="color:#999;">암종 예측 데이터가 없습니다. ML 모델이 설치되어 있는지 확인하세요.</p>'

        predicted = prediction.get('predicted_cancer', 'N/A')
        predicted_kr = prediction.get('predicted_cancer_korean', '알 수 없음')
        confidence = prediction.get('confidence', 0)
        conf_level = prediction.get('confidence_level', 'low')
        cluster_agreement = prediction.get('cluster_agreement', 0)

        # Confidence color
        if conf_level == 'high':
            conf_color = '#059669'
            conf_badge = '높음'
        elif conf_level == 'medium':
            conf_color = '#d97706'
            conf_badge = '중간'
        else:
            conf_color = '#dc2626'
            conf_badge = '낮음'

        # Cluster predictions table
        cluster_preds = prediction.get('cluster_predictions', [])
        cluster_rows = ''
        for cp in cluster_preds[:10]:  # Max 10 clusters
            cluster_rows += f'''<tr>
                <td>{cp.get('sample_id', 'N/A')}</td>
                <td>{cp.get('predicted_cancer', 'N/A')}</td>
                <td>{cp.get('predicted_cancer_korean', '')}</td>
                <td>{cp.get('confidence', 0):.1%}</td>
            </tr>'''

        return f'''
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
            <div style="background:var(--bg-tertiary);padding:24px;border-radius:8px;">
                <h3 style="margin-bottom:16px;">Pseudo-bulk 예측 결과</h3>
                <div style="font-size:32px;font-weight:700;color:var(--accent-blue);">{predicted}</div>
                <div style="font-size:18px;color:var(--text-secondary);margin-top:4px;">{predicted_kr}</div>
                <div style="margin-top:16px;">
                    <span style="background:{conf_color};color:white;padding:4px 12px;border-radius:12px;font-size:12px;">
                        신뢰도: {confidence:.1%} ({conf_badge})
                    </span>
                </div>
                <div style="margin-top:12px;font-size:13px;color:var(--text-muted);">
                    클러스터 일치율: {cluster_agreement:.1%}
                </div>
            </div>
            <div>
                <h3 style="margin-bottom:12px;">클러스터별 예측</h3>
                <div class="table-wrapper" style="max-height:200px;overflow-y:auto;">
                    <table>
                        <thead><tr><th>Cluster</th><th>Cancer</th><th>암종명</th><th>Confidence</th></tr></thead>
                        <tbody>{cluster_rows}</tbody>
                    </table>
                </div>
            </div>
        </div>
        <p style="margin-top:16px;font-size:12px;color:var(--text-muted);background:#fef3c7;padding:12px;border-radius:8px;">
            ⚠️ <strong>주의:</strong> 이 예측은 Pseudo-bulk 집계 후 ML 모델을 적용한 결과입니다.
            Single-cell의 이질성으로 인해 Bulk RNA-seq 예측보다 정확도가 낮을 수 있습니다.
            진단 목적으로 사용하지 마세요.
        </p>
        '''

    def _driver_genes_html(self, df: Optional[pd.DataFrame]) -> str:
        """Generate HTML for driver genes section."""
        if df is None or len(df) == 0:
            return '<p style="color:#999;">드라이버 유전자가 발견되지 않았습니다.</p>'

        # Summary stats
        n_cosmic = len(df[df['is_cosmic_tier1'] == True]) if 'is_cosmic_tier1' in df.columns else 0
        n_oncokb = len(df[df['is_oncokb_actionable'] == True]) if 'is_oncokb_actionable' in df.columns else 0
        n_tme = len(df[df['is_tme_marker'] == True]) if 'is_tme_marker' in df.columns else 0

        # Top drivers table
        top_drivers = df.head(20)
        rows_html = ''
        for _, row in top_drivers.iterrows():
            badges = []
            if row.get('is_cosmic_tier1'):
                badges.append('<span style="background:#fee2e2;color:#991b1b;padding:2px 6px;border-radius:4px;font-size:10px;">COSMIC</span>')
            if row.get('is_oncokb_actionable'):
                badges.append('<span style="background:#dbeafe;color:#1e40af;padding:2px 6px;border-radius:4px;font-size:10px;">OncoKB</span>')
            if row.get('is_tme_marker'):
                badges.append('<span style="background:#dcfce7;color:#166534;padding:2px 6px;border-radius:4px;font-size:10px;">TME</span>')

            lfc = row.get('logfoldchange', 0)
            lfc_color = '#dc2626' if lfc > 0 else '#2563eb'

            rows_html += f'''<tr>
                <td><strong>{row.get('gene', 'N/A')}</strong></td>
                <td>{row.get('cluster', 'N/A')}</td>
                <td style="color:{lfc_color}">{lfc:+.2f}</td>
                <td>{row.get('pval_adj', 1):.2e}</td>
                <td>{' '.join(badges)}</td>
                <td style="font-size:11px;color:var(--text-muted);">{row.get('driver_type', '')}</td>
            </tr>'''

        return f'''
        <div class="summary-grid" style="margin-bottom:24px;">
            <div class="summary-card">
                <div class="value" style="color:#991b1b;">{n_cosmic}</div>
                <div class="label">COSMIC Tier1</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#1e40af;">{n_oncokb}</div>
                <div class="label">OncoKB Actionable</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#166534;">{n_tme}</div>
                <div class="label">TME Markers</div>
            </div>
        </div>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>Gene</th><th>Cluster</th><th>Log2FC</th><th>Adj.P</th><th>Database</th><th>Type</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <p style="margin-top:12px;font-size:12px;color:var(--text-muted);">
            * COSMIC Tier1: Cancer Gene Census에 등록된 주요 암 유전자<br>
            * OncoKB Actionable: 치료 표적으로 활용 가능한 유전자<br>
            * TME: 종양 미세환경 관련 마커 (면역, 혈관신생, 기질)
        </p>
        '''

    def _cluster_pathways_html(self, df: Optional[pd.DataFrame]) -> str:
        """Generate HTML for cluster pathway analysis section."""
        if df is None or len(df) == 0:
            return '<p style="color:#999;">Pathway 분석 결과가 없습니다. gseapy가 설치되어 있는지 확인하세요.</p>'

        # Group by cluster
        clusters = df['cluster'].unique()
        accordion_html = ''

        for cluster in sorted(clusters):
            cluster_df = df[df['cluster'] == cluster]
            # Get top pathways per database
            rows_html = ''
            for _, row in cluster_df.head(10).iterrows():
                padj = row.get('padj', 1)
                padj_color = '#059669' if padj < 0.01 else ('#d97706' if padj < 0.05 else '#dc2626')
                genes = row.get('genes', '')
                gene_list = genes.split(';')[:5] if genes else []
                gene_preview = ', '.join(gene_list) + ('...' if len(gene_list) == 5 else '')

                rows_html += f'''<tr>
                    <td style="max-width:300px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{row.get('term', '')}">{row.get('term', 'N/A')[:50]}...</td>
                    <td style="font-size:11px;">{row.get('database', 'N/A').replace('_', ' ')[:15]}</td>
                    <td style="color:{padj_color}">{padj:.2e}</td>
                    <td>{row.get('gene_count', 0)}</td>
                    <td style="font-size:11px;color:var(--text-muted);">{gene_preview}</td>
                </tr>'''

            accordion_html += f'''
            <details style="margin-bottom:12px;border:1px solid var(--border-light);border-radius:8px;">
                <summary style="padding:12px;background:var(--bg-tertiary);cursor:pointer;font-weight:600;">
                    Cluster {cluster} ({len(cluster_df)} terms)
                </summary>
                <div style="padding:12px;">
                    <table style="font-size:12px;">
                        <thead><tr><th>Term</th><th>Database</th><th>Adj.P</th><th>Genes</th><th>Top Genes</th></tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            </details>
            '''

        return f'''
        <p style="margin-bottom:16px;">
            총 <strong>{len(clusters)}</strong>개 클러스터에서 <strong>{len(df)}</strong>개의 유의한 Pathway term이 발견되었습니다.
        </p>
        {accordion_html}
        '''

    def _trajectory_html(self, df: Optional[pd.DataFrame], figures: Dict) -> str:
        """Generate HTML for trajectory analysis section."""
        has_data = df is not None and len(df) > 0
        has_paga = 'paga_paga' in figures
        has_pseudotime = 'umap_pseudotime' in figures

        if not has_data and not has_paga and not has_pseudotime:
            return '<p style="color:#999;">Trajectory 분석 결과가 없습니다. 충분한 세포 수가 필요합니다.</p>'

        content = '<div class="figure-grid">'

        # PAGA graph
        if has_paga:
            content += self._figure_html(figures, 'paga_paga', 'PAGA Graph (Cluster Connectivity)')

        # Pseudotime UMAP
        if has_pseudotime:
            content += self._figure_html(figures, 'umap_pseudotime', 'Diffusion Pseudotime')

        content += '</div>'

        # Pseudotime statistics
        if has_data:
            # Group by cluster and calculate mean pseudotime
            cluster_pt = df.groupby('cluster')['dpt_pseudotime'].agg(['mean', 'std', 'count']).reset_index()
            cluster_pt = cluster_pt.sort_values('mean')

            rows_html = ''
            for _, row in cluster_pt.iterrows():
                pt_color = f'hsl({(1-row["mean"])*120}, 70%, 50%)'  # Green to red gradient
                rows_html += f'''<tr>
                    <td>{row['cluster']}</td>
                    <td style="color:{pt_color};font-weight:600;">{row['mean']:.3f}</td>
                    <td>±{row['std']:.3f}</td>
                    <td>{int(row['count']):,}</td>
                </tr>'''

            content += f'''
            <div style="margin-top:24px;">
                <h3 style="margin-bottom:12px;">클러스터별 Pseudotime</h3>
                <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px;">
                    Pseudotime은 세포의 분화 상태를 나타냅니다. 낮은 값(녹색)은 초기 상태, 높은 값(빨간색)은 분화된 상태를 의미합니다.
                </p>
                <div class="table-wrapper">
                    <table>
                        <thead><tr><th>Cluster</th><th>Mean PT</th><th>Std</th><th>Cells</th></tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            </div>
            '''

        return content

    def _interactions_html(self, df: Optional[pd.DataFrame]) -> str:
        """Generate HTML for cell-cell interaction section."""
        if df is None or len(df) == 0:
            return '<p style="color:#999;">Cell-Cell Interaction 분석 결과가 없습니다.</p>'

        # Check which columns are available (LIANA vs simple LR)
        is_liana = 'magnitude_rank' in df.columns or 'ligand_complex' in df.columns

        if is_liana:
            # LIANA results
            n_interactions = len(df)
            rows_html = ''
            for _, row in df.head(30).iterrows():
                rows_html += f'''<tr>
                    <td>{row.get('source', 'N/A')}</td>
                    <td>{row.get('target', 'N/A')}</td>
                    <td><strong>{row.get('ligand_complex', row.get('ligand', 'N/A'))}</strong></td>
                    <td><strong>{row.get('receptor_complex', row.get('receptor', 'N/A'))}</strong></td>
                    <td>{row.get('magnitude_rank', row.get('interaction_score', 0)):.3f}</td>
                </tr>'''

            return f'''
            <p style="margin-bottom:16px;">
                총 <strong>{n_interactions}</strong>개의 ligand-receptor 상호작용이 발견되었습니다.
            </p>
            <div class="table-wrapper" style="max-height:400px;overflow-y:auto;">
                <table>
                    <thead><tr><th>Source Cell</th><th>Target Cell</th><th>Ligand</th><th>Receptor</th><th>Score</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            <p style="margin-top:16px;font-size:12px;color:var(--text-muted);">
                * Score가 낮을수록 상호작용이 강합니다 (LIANA aggregate rank)
            </p>
            '''
        else:
            # Simple LR results
            n_interactions = len(df)

            # Group by ligand-receptor pair
            pair_summary = df.groupby(['ligand', 'receptor']).agg({
                'interaction_score': 'max',
                'source': lambda x: ', '.join(x.unique()[:3]),
                'target': lambda x: ', '.join(x.unique()[:3])
            }).reset_index()
            pair_summary = pair_summary.nlargest(20, 'interaction_score')

            rows_html = ''
            for _, row in pair_summary.iterrows():
                score = row['interaction_score']
                score_color = '#059669' if score > 1 else ('#d97706' if score > 0.5 else '#6b7280')
                rows_html += f'''<tr>
                    <td><strong>{row['ligand']}</strong></td>
                    <td><strong>{row['receptor']}</strong></td>
                    <td style="font-size:11px;">{row['source']}</td>
                    <td style="font-size:11px;">{row['target']}</td>
                    <td style="color:{score_color};font-weight:600;">{score:.3f}</td>
                </tr>'''

            # Interaction categories
            categories = {
                'Immune Checkpoint': ['CD274-PDCD1', 'CD80-CTLA4', 'LGALS9-HAVCR2'],
                'Growth Factor': ['VEGFA-KDR', 'EGF-EGFR', 'HGF-MET', 'TGFB1-TGFBR1'],
                'Chemokine': ['CXCL12-CXCR4', 'CCL2-CCR2', 'CXCL8-CXCR1'],
            }

            return f'''
            <p style="margin-bottom:16px;">
                총 <strong>{n_interactions}</strong>개의 ligand-receptor 상호작용이 발견되었습니다.
            </p>
            <div class="table-wrapper" style="max-height:400px;overflow-y:auto;">
                <table>
                    <thead><tr><th>Ligand</th><th>Receptor</th><th>Source Cells</th><th>Target Cells</th><th>Score</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            <p style="margin-top:16px;font-size:12px;color:var(--text-muted);">
                * Score = Ligand 발현 × Receptor 발현 (높을수록 강한 상호작용)<br>
                * 검사된 L-R 쌍: Immune checkpoint, Growth factor, Chemokine 등 주요 경로
            </p>
            '''

    def _figure_html(self, figures: Dict, key: str, title: str) -> str:
        """Generate HTML for a figure panel with centered title."""
        if key not in figures:
            return f'''<div class="figure-panel">
                <div class="figure-title">{title}</div>
                <div class="figure-content"><p style="padding:24px;color:#999;">Figure not available</p></div>
            </div>'''

        return f'''<div class="figure-panel">
            <div class="figure-title">{title}</div>
            <div class="figure-content">
                <img src="{figures[key]}" alt="{title}">
            </div>
        </div>'''

    def _composition_html(self, df: Optional[pd.DataFrame], figures: Dict[str, str] = None) -> str:
        """Generate cell composition visualization with bar chart image."""
        if df is None or len(df) == 0:
            return '<p>No composition data available</p>'

        # Use cell_type column if available, otherwise fall back to cluster
        type_col = 'cell_type' if 'cell_type' in df.columns else 'cluster'

        # Aggregate by cell type
        if type_col in df.columns and 'count' in df.columns:
            type_counts = df.groupby(type_col)['count'].sum().reset_index()
            type_counts = type_counts[type_counts['count'] > 0].sort_values('count', ascending=False)
            total = type_counts['count'].sum()

            # Bar chart image (if available)
            barchart_html = ''
            if figures and 'celltype_barchart' in figures:
                barchart_html = f'''
                <div style="margin-bottom:24px;">
                    <img src="{figures['celltype_barchart']}" alt="Cell Type Composition" style="max-width:100%;border-radius:8px;">
                </div>
                '''

            # Generate table rows with cell type colors
            colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#f97316', '#06b6d4', '#ec4899',
                      '#84cc16', '#a855f7', '#14b8a6', '#f43f5e', '#6366f1', '#eab308', '#0ea5e9', '#d946ef']

            rows_html = ''
            for i, (_, row) in enumerate(type_counts.iterrows()):
                cell_type = row[type_col]
                count = row['count']
                pct = count / total * 100
                color = colors[i % len(colors)]

                rows_html += f'''<tr>
                    <td><span style="display:inline-block;width:12px;height:12px;background:{color};border-radius:3px;margin-right:8px;"></span><strong>{cell_type}</strong></td>
                    <td style="text-align:right;">{count:,}</td>
                    <td style="text-align:right;">{pct:.1f}%</td>
                </tr>'''

            return f'''
            {barchart_html}
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>Cell Type</th><th style="text-align:right;">Cell Count</th><th style="text-align:right;">Percentage</th></tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            '''
        return '<p>Composition data format not recognized</p>'

    def _composition_table_html(self, df: Optional[pd.DataFrame]) -> str:
        """Generate cell composition table only (no images)."""
        if df is None or len(df) == 0:
            return ''

        type_col = 'cell_type' if 'cell_type' in df.columns else 'cluster'

        if type_col in df.columns and 'count' in df.columns:
            type_counts = df.groupby(type_col)['count'].sum().reset_index()
            type_counts = type_counts[type_counts['count'] > 0].sort_values('count', ascending=False)
            total = type_counts['count'].sum()

            rows_html = ''
            for _, row in type_counts.iterrows():
                cell_type = row[type_col]
                count = row['count']
                pct = count / total * 100
                rows_html += f'''<tr>
                    <td><strong>{cell_type}</strong></td>
                    <td style="text-align:right;">{count:,}</td>
                    <td style="text-align:right;">{pct:.1f}%</td>
                </tr>'''

            return f'''
            <div class="table-wrapper" style="margin-top:16px;">
                <table>
                    <thead>
                        <tr><th>Cell Type</th><th style="text-align:right;">Cell Count</th><th style="text-align:right;">Percentage</th></tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            '''
        return ''

    def _markers_table_html(self, df: Optional[pd.DataFrame]) -> str:
        """Generate accordion-style marker genes table grouped by cluster."""
        if df is None or len(df) == 0:
            return '<p>No marker data available</p>'

        if 'cluster' not in df.columns:
            return '<p>No cluster information in marker data</p>'

        # Sort by cluster and score
        df_sorted = df.sort_values(['cluster', 'score'], ascending=[True, False])
        clusters = sorted(df_sorted['cluster'].unique())

        initial_show = 10  # Show first 10 markers initially

        accordion_html = '<div class="accordion-container">'

        for cluster in clusters:
            cluster_df = df_sorted[df_sorted['cluster'] == cluster]
            n_markers = len(cluster_df)
            top_gene = cluster_df.iloc[0]['gene'] if n_markers > 0 else '-'

            # Build rows for initial display (first 10)
            visible_rows = ''
            hidden_rows = ''

            for idx, (_, row) in enumerate(cluster_df.iterrows()):
                gene = row.get('gene', '-')
                score = row.get('score', 0)
                lfc = row.get('logfoldchange', 0)
                pval = row.get('pval_adj', row.get('pval', 1))

                row_html = f'''<tr>
                    <td><strong>{gene}</strong></td>
                    <td>{score:.2f}</td>
                    <td style="color:{'#dc2626' if lfc > 0 else '#2563eb'}">{lfc:+.2f}</td>
                    <td>{pval:.2e}</td>
                </tr>'''

                if idx < initial_show:
                    visible_rows += row_html
                else:
                    hidden_rows += row_html

            # Show more button if there are hidden rows
            show_more_html = ''
            if n_markers > initial_show:
                remaining = n_markers - initial_show
                show_more_html = f'''
                <button class="show-more-btn" onclick="toggleMoreRows(this, 'cluster-{cluster}-hidden')">
                    <span class="btn-text">더보기 (+{remaining}개)</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                </button>'''

            accordion_html += f'''
            <div class="accordion-item" data-cluster="{cluster}">
                <div class="accordion-header" onclick="toggleAccordion(this)">
                    <div class="accordion-header-left">
                        <span class="cluster-badge cluster-{cluster}">Cluster {cluster}</span>
                        <span class="top-gene">Top: {top_gene}</span>
                    </div>
                    <div class="accordion-header-right">
                        <span class="marker-count">{n_markers} markers</span>
                        <svg class="accordion-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="accordion-content">
                    <table class="accordion-table">
                        <thead>
                            <tr><th>Gene</th><th>Score</th><th>Log2FC</th><th>Adj. P-value</th></tr>
                        </thead>
                        <tbody class="visible-rows">{visible_rows}</tbody>
                        <tbody class="hidden-rows" id="cluster-{cluster}-hidden">{hidden_rows}</tbody>
                    </table>
                    {show_more_html}
                </div>
            </div>
            '''

        accordion_html += '</div>'

        # Add JavaScript for accordion functionality
        accordion_html += '''
        <script>
        function toggleAccordion(header) {
            const item = header.parentElement;
            item.classList.toggle('open');
        }

        function toggleMoreRows(btn, hiddenId) {
            const hiddenRows = document.getElementById(hiddenId);
            const btnText = btn.querySelector('.btn-text');

            if (hiddenRows.classList.contains('visible')) {
                hiddenRows.classList.remove('visible');
                btnText.textContent = btnText.textContent.replace('접기', '더보기');
                btn.querySelector('svg').style.transform = 'rotate(0deg)';
            } else {
                hiddenRows.classList.add('visible');
                btnText.textContent = btnText.textContent.replace('더보기', '접기');
                btn.querySelector('svg').style.transform = 'rotate(180deg)';
            }
        }

        // Expand all / Collapse all functionality
        function expandAllAccordions() {
            document.querySelectorAll('.accordion-item').forEach(item => {
                item.classList.add('open');
            });
        }

        function collapseAllAccordions() {
            document.querySelectorAll('.accordion-item').forEach(item => {
                item.classList.remove('open');
            });
        }
        </script>
        '''

        return accordion_html

    def _generate_llm_interpretation(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate LLM-based interpretations for single-cell analysis.

        Returns:
            Dict with keys: executive_summary, cell_composition, marker_interpretation,
                           tme_interpretation, clinical_implications
        """
        # Check for LLM API availability
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

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
            self.logger.warning("No LLM API available for interpretation")
            return self._generate_fallback_interpretation(data)

        llm_provider = "OpenAI" if use_openai else "Anthropic"
        self.logger.info(f"Generating LLM interpretation using {llm_provider}...")

        # Prepare data summary
        adata_summary = data.get('adata_summary', {})
        n_cells = adata_summary.get('n_cells', 0)
        n_clusters = adata_summary.get('n_clusters', 0)
        n_celltypes = adata_summary.get('n_celltypes', 0)

        # Cell composition
        composition_df = data.get('cell_composition')
        composition_text = ""
        if composition_df is not None and len(composition_df) > 0:
            type_col = 'cell_type' if 'cell_type' in composition_df.columns else 'cluster'
            if type_col in composition_df.columns and 'count' in composition_df.columns:
                comp_data = composition_df.groupby(type_col)['count'].sum().sort_values(ascending=False)
                total = comp_data.sum()
                composition_text = ", ".join([f"{ct}: {cnt/total*100:.1f}%" for ct, cnt in comp_data.head(10).items()])

        # Top markers per cluster
        markers_df = data.get('cluster_markers')
        markers_text = ""
        if markers_df is not None and len(markers_df) > 0:
            top_markers = markers_df.groupby('cluster').head(3)[['cluster', 'gene']].values.tolist()
            cluster_markers = {}
            for cluster, gene in top_markers:
                if cluster not in cluster_markers:
                    cluster_markers[cluster] = []
                cluster_markers[cluster].append(gene)
            markers_text = "; ".join([f"Cluster {c}: {', '.join(genes)}" for c, genes in sorted(cluster_markers.items())[:8]])

        # TME composition
        tme_df = data.get('tme_composition')
        tme_text = ""
        if tme_df is not None and len(tme_df) > 0:
            tme_text = ", ".join([f"{row['category']}: {row['percentage']:.1f}%" for _, row in tme_df.iterrows()])

        # Cancer prediction
        cancer_pred = data.get('cancer_prediction', {})
        pred_text = ""
        if cancer_pred:
            pred_label = cancer_pred.get('predicted_label', 'Unknown')
            pred_prob = cancer_pred.get('probability', 0)
            pred_text = f"예측 암종: {pred_label} ({pred_prob*100:.1f}%)"

        # Master regulators (GRN)
        mr_df = data.get('master_regulators')
        mr_text = ""
        if mr_df is not None and len(mr_df) > 0:
            top_mrs = mr_df.head(5)['tf'].tolist() if 'tf' in mr_df.columns else []
            mr_text = ", ".join(top_mrs)

        # Build prompt
        prompt = f"""당신은 단일세포 RNA-seq 분석 전문가입니다. 다음 분석 결과를 바탕으로 연구자를 위한 해석을 제공해주세요.

## 분석 데이터 요약
- 총 세포 수: {n_cells:,}개
- 클러스터 수: {n_clusters}개
- 세포 유형 수: {n_celltypes}개

## 세포 구성
{composition_text}

## 클러스터별 마커 유전자
{markers_text}

## 종양 미세환경 (TME)
{tme_text}

## 암종 예측
{pred_text}

## 주요 전사인자 (Master Regulators)
{mr_text}

---

다음 항목들을 한국어로 작성해주세요. 각 항목은 2-4문장으로 간결하게 작성하세요:

1. **Executive Summary**: 전체 분석 결과의 핵심 발견을 요약
2. **세포 구성 해석**: 세포 유형 분포의 의미와 특징
3. **마커 유전자 해석**: 주요 마커 유전자들이 나타내는 생물학적 의미
4. **TME/면역 해석**: 종양 미세환경 구성이 시사하는 면역학적 특성
5. **임상적 함의**: 이 결과가 치료 전략에 어떤 시사점을 주는지

JSON 형식으로 응답해주세요:
{{"executive_summary": "...", "cell_composition": "...", "marker_interpretation": "...", "tme_interpretation": "...", "clinical_implications": "..."}}
"""

        try:
            if use_openai:
                client = OpenAIClient(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
            else:
                client = anthropic_module.Anthropic(api_key=anthropic_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt + "\n\nJSON 형식으로만 응답하세요."}]
                )
                result_text = response.content[0].text

            # Parse JSON
            result = json.loads(result_text)
            self.logger.info("LLM interpretation generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"LLM interpretation failed: {e}")
            return self._generate_fallback_interpretation(data)

    def _generate_fallback_interpretation(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate rule-based fallback interpretation when LLM is unavailable."""
        adata_summary = data.get('adata_summary', {})
        n_cells = adata_summary.get('n_cells', 0)
        n_clusters = adata_summary.get('n_clusters', 0)

        # TME phenotype detection
        tme_df = data.get('tme_composition')
        immune_pct = 0
        if tme_df is not None and len(tme_df) > 0:
            immune_rows = tme_df[tme_df['category'].str.contains('immune|T_cell|NK|B_cell', case=False, na=False)]
            if len(immune_rows) > 0:
                immune_pct = immune_rows['percentage'].sum()

        tme_phenotype = "Hot (Inflamed)" if immune_pct > 30 else "Altered" if immune_pct > 15 else "Cold (Desert)"

        return {
            "executive_summary": f"본 분석에서 총 {n_cells:,}개 세포가 {n_clusters}개 클러스터로 분류되었습니다. "
                                f"TME 표현형은 '{tme_phenotype}'으로 분류되었으며, 면역세포 비율은 {immune_pct:.1f}%입니다.",
            "cell_composition": "세포 유형별 구성 비율은 UMAP 시각화와 Bar chart에서 확인할 수 있습니다. "
                               "각 클러스터의 세포 유형은 마커 유전자 발현 패턴을 기반으로 자동 추론되었습니다.",
            "marker_interpretation": "클러스터별 마커 유전자는 Wilcoxon rank-sum test를 통해 식별되었습니다. "
                                    "상위 마커 유전자들은 각 클러스터의 세포 유형 특성을 반영합니다.",
            "tme_interpretation": f"면역 침윤 분석 결과, TME는 '{tme_phenotype}' 표현형을 보입니다. "
                                 f"면역세포 비율 {immune_pct:.1f}%는 {'높은' if immune_pct > 30 else '중간' if immune_pct > 15 else '낮은'} 면역 활성도를 시사합니다.",
            "clinical_implications": "TME 구성 및 세포 유형 분포는 면역치료 반응성 예측에 참고될 수 있습니다. "
                                    "자세한 치료 전략은 추가적인 임상 정보와 함께 종합적으로 평가되어야 합니다."
        }

    def run(self) -> Dict[str, Any]:
        """Generate the single-cell report."""
        self.logger.info("Generating Single-Cell Report...")

        # Load all data
        data = self._load_all_data()

        # Generate HTML
        html_content = self._generate_html(data)

        # Save report
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.logger.info(f"Report saved: {report_path}")

        # Save report data as JSON
        report_data = {
            'summary': data.get('adata_summary', {}),
            'n_clusters': data.get('adata_summary', {}).get('n_clusters', 0),
            'n_cells': data.get('adata_summary', {}).get('n_cells', 0),
            'n_markers': len(data.get('cluster_markers', [])) if data.get('cluster_markers') is not None else 0,
            'config': self.config,
            'generated_at': datetime.now().isoformat()
        }

        report_data_path = self.output_dir / "report_data.json"
        with open(report_data_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return {
            'report_path': str(report_path),
            'report_data_path': str(report_data_path),
            **report_data
        }

    def _tme_analysis_html(self, tme_composition: Optional[pd.DataFrame], signature_scores: Optional[pd.DataFrame]) -> str:
        """Generate HTML for TME analysis section."""
        if tme_composition is None or tme_composition.empty:
            return '<p style="color:#999;">TME 분석 데이터가 없습니다.</p>'

        # Build composition table
        html = '''
        <div class="content-box">
            <h3>종양 미세환경 구성</h3>
            <p style="font-size:12px;color:#666;margin-bottom:16px;">
                CIBERSORT 스타일의 면역 침윤 분석 결과입니다. 면역세포, 기질세포, 종양세포 비율을 기반으로
                TME 표현형(Hot/Altered/Cold)을 분류합니다.
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>세포 카테고리</th>
                        <th>세포 수</th>
                        <th>비율 (%)</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for _, row in tme_composition.iterrows():
            category = row.get('category', '')
            count = row.get('count', 0)
            pct = row.get('percentage', 0)

            # Color coding based on category
            if 'immune' in category.lower() or 'T_cells' in category or 'NK' in category or 'B_cells' in category:
                color = '#059669'  # Green for immune
            elif 'stromal' in category.lower() or 'fibro' in category.lower():
                color = '#d97706'  # Orange for stromal
            elif 'tumor' in category.lower() or 'malign' in category.lower():
                color = '#dc2626'  # Red for tumor
            else:
                color = '#6b7280'  # Gray for others

            html += f'''
                    <tr>
                        <td style="color:{color};font-weight:500;">{category}</td>
                        <td>{count:,}</td>
                        <td>{pct:.1f}%</td>
                    </tr>
            '''

        html += '''
                </tbody>
            </table>
        </div>
        '''

        # Add signature scores if available
        if signature_scores is not None and not signature_scores.empty:
            html += '''
            <div class="content-box" style="margin-top:16px;">
                <h3>면역 시그니처 점수</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>시그니처</th>
                            <th>평균 점수</th>
                            <th>표준 편차</th>
                        </tr>
                    </thead>
                    <tbody>
            '''

            for _, row in signature_scores.iterrows():
                sig_name = row.get('signature', row.get('index', ''))
                mean_score = row.get('mean', 0)
                std_score = row.get('std', 0)

                html += f'''
                        <tr>
                            <td>{sig_name}</td>
                            <td>{mean_score:.3f}</td>
                            <td>{std_score:.3f}</td>
                        </tr>
                '''

            html += '''
                    </tbody>
                </table>
            </div>
            '''

        return html

    def _grn_analysis_html(self, master_regulators: Optional[pd.DataFrame], tf_activity: Optional[pd.DataFrame]) -> str:
        """Generate HTML for GRN analysis section."""
        if master_regulators is None or master_regulators.empty:
            return '<p style="color:#999;">GRN 분석 데이터가 없습니다. 충분한 TF가 발현되지 않았을 수 있습니다.</p>'

        html = '''
        <div class="content-box">
            <h3>세포 유형별 Master Regulator</h3>
            <p style="font-size:12px;color:#666;margin-bottom:16px;">
                SCENIC 스타일의 유전자 조절 네트워크 분석 결과입니다.
                각 세포 유형에서 가장 많은 타겟 유전자를 조절하는 전사 인자(TF)를 식별합니다.
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>세포 유형</th>
                        <th>순위</th>
                        <th>전사인자 (TF)</th>
                        <th>타겟 유전자 수</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for _, row in master_regulators.iterrows():
            cell_type = row.get('cell_type', '')
            rank = row.get('rank', 0)
            tf = row.get('TF', '')
            n_targets = row.get('n_targets', 0)

            # Highlight top TFs
            tf_color = '#dc2626' if tf in ['TP53', 'MYC', 'STAT3', 'HIF1A', 'NFKB1'] else '#1e40af'

            html += f'''
                    <tr>
                        <td>{cell_type}</td>
                        <td>#{rank}</td>
                        <td style="color:{tf_color};font-weight:600;">{tf}</td>
                        <td>{n_targets}</td>
                    </tr>
            '''

        html += '''
                </tbody>
            </table>
        </div>
        '''

        # Add TF activity heatmap-style visualization (simplified as table)
        if tf_activity is not None and not tf_activity.empty:
            html += '''
            <div class="content-box" style="margin-top:16px;">
                <h3>전사인자 활성 점수 (세포 유형별)</h3>
                <p style="font-size:12px;color:#666;margin-bottom:8px;">
                    높은 값 = 해당 TF가 해당 세포 유형에서 활성화됨
                </p>
            '''

            # Convert to proper format if needed
            if 'cell_type' in tf_activity.columns:
                tf_activity = tf_activity.set_index('cell_type')

            # Get top TFs by variance
            top_tfs = tf_activity.var().nlargest(10).index.tolist()

            if top_tfs:
                html += '''
                <table class="data-table" style="font-size:11px;">
                    <thead>
                        <tr>
                            <th>세포 유형</th>
                '''
                for tf in top_tfs[:8]:
                    html += f'<th>{tf}</th>'
                html += '''
                        </tr>
                    </thead>
                    <tbody>
                '''

                for cell_type in tf_activity.index[:10]:
                    html += f'<tr><td style="font-weight:500;">{cell_type}</td>'
                    for tf in top_tfs[:8]:
                        val = tf_activity.loc[cell_type, tf] if tf in tf_activity.columns else 0
                        # Color based on value
                        if val > 1:
                            bg_color = '#fef3c7'
                        elif val > 0.5:
                            bg_color = '#d1fae5'
                        else:
                            bg_color = 'transparent'
                        html += f'<td style="background:{bg_color};">{val:.2f}</td>'
                    html += '</tr>'

                html += '''
                    </tbody>
                </table>
                '''

            html += '</div>'

        return html

    def _ploidy_analysis_html(self, ploidy_by_celltype: Optional[pd.DataFrame]) -> str:
        """Generate HTML for InferPloidy-based malignant cell detection section."""
        if ploidy_by_celltype is None or ploidy_by_celltype.empty:
            return '<p style="color:#999;">Ploidy 분석 데이터가 없습니다.</p>'

        html = '''
        <div class="content-box">
            <h3>세포 유형별 Ploidy 분석</h3>
            <p style="font-size:12px;color:#666;margin-bottom:16px;">
                InferPloidy 기반의 염색체 배수성(Ploidy) 추론 결과입니다.
                정상 세포는 2n(diploid)이지만, 암세포는 종종 이수성(aneuploidy)을 보입니다.
                Ploidy 점수가 높을수록 염색체 불안정성이 높아 악성세포일 가능성이 증가합니다.
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>세포 유형</th>
                        <th>Ploidy 점수 (평균)</th>
                        <th>악성 추정 세포 수</th>
                        <th>전체 세포 수</th>
                        <th>악성 비율 (%)</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for _, row in ploidy_by_celltype.iterrows():
            cell_type = row.get('cell_type', row.get('index', ''))
            # Support both old cnv_mean and new ploidy_score column names
            ploidy_score = row.get('ploidy_score', row.get('cnv_mean', 0))
            n_malignant = int(row.get('n_malignant', 0))
            n_total = int(row.get('n_total', 1))
            pct_malignant = row.get('pct_malignant', 0)

            # Color code based on malignancy percentage
            if pct_malignant > 50:
                row_color = 'background:#fef2f2;'  # Red tint
                badge = '<span style="background:#dc2626;color:white;padding:2px 6px;border-radius:4px;font-size:10px;">High</span>'
            elif pct_malignant > 25:
                row_color = 'background:#fffbeb;'  # Yellow tint
                badge = '<span style="background:#d97706;color:white;padding:2px 6px;border-radius:4px;font-size:10px;">Med</span>'
            else:
                row_color = ''
                badge = '<span style="background:#059669;color:white;padding:2px 6px;border-radius:4px;font-size:10px;">Low</span>'

            html += f'''
                    <tr style="{row_color}">
                        <td style="font-weight:500;">{cell_type}</td>
                        <td>{ploidy_score:.2f}</td>
                        <td>{n_malignant:,}</td>
                        <td>{n_total:,}</td>
                        <td>{pct_malignant:.1f}% {badge}</td>
                    </tr>
            '''

        html += '''
                </tbody>
            </table>
        </div>

        <div class="content-box" style="margin-top:16px;background:#fff7ed;border-left:4px solid #d97706;">
            <h4 style="color:#92400e;margin-bottom:8px;">⚠️ 해석 주의사항</h4>
            <ul style="font-size:12px;color:#78350f;margin-left:16px;">
                <li>Ploidy 점수가 높다고 반드시 악성세포를 의미하지 않습니다.</li>
                <li>정상 세포도 세포주기에 따라 일시적으로 배수성이 변할 수 있습니다.</li>
                <li>최종 판단은 조직학적 검사와 전문가 검토가 필요합니다.</li>
                <li>면역세포(T, NK, B cells)를 정상 참조군(diploid reference)으로 사용하였습니다.</li>
            </ul>
        </div>
        '''

        return html

    def _clinical_implications_html(self, data: Dict[str, Any]) -> str:
        """Generate Clinical Implications section for single-cell analysis."""
        driver_genes_df = data.get('driver_genes')
        cancer_prediction = data.get('cancer_prediction', {})
        tme_composition = data.get('tme_composition')
        cluster_markers = data.get('cluster_markers')

        predicted_cancer = cancer_prediction.get('predicted_cancer', 'Unknown')
        predicted_cancer_kr = cancer_prediction.get('predicted_cancer_korean', '알 수 없음')

        # Build biomarker candidates from driver genes
        biomarker_rows = ''
        if driver_genes_df is not None and len(driver_genes_df) > 0:
            for _, row in driver_genes_df.head(6).iterrows():
                gene = row.get('gene', 'N/A')
                lfc = row.get('logfoldchange', 0)
                is_cosmic = row.get('is_cosmic_tier1', False)
                is_oncokb = row.get('is_oncokb_actionable', False)
                driver_type = row.get('driver_type', '')

                evidence = 'HIGH' if is_cosmic or is_oncokb else 'MEDIUM'
                evidence_class = 'high' if evidence == 'HIGH' else 'medium'
                marker_type = '진단/예후' if is_cosmic else '치료 반응'

                rationale = f"{'COSMIC Tier1' if is_cosmic else ''} {'OncoKB' if is_oncokb else ''} {driver_type}"
                biomarker_rows += f'''
                <tr>
                    <td class="cell-gene">{gene}</td>
                    <td>{marker_type}</td>
                    <td><span class="evidence-badge {evidence_class}">{evidence}</span></td>
                    <td>{rationale.strip()}</td>
                </tr>'''

        # Build therapeutic targets
        therapeutic_rows = ''
        if driver_genes_df is not None:
            actionable = driver_genes_df[driver_genes_df.get('is_oncokb_actionable', False) == True] if 'is_oncokb_actionable' in driver_genes_df.columns else pd.DataFrame()
            for _, row in actionable.head(4).iterrows():
                gene = row.get('gene', 'N/A')
                lfc = row.get('logfoldchange', 0)
                direction = "↑" if lfc > 0 else "↓"

                therapeutic_rows += f'''
                <tr>
                    <td class="cell-gene">{gene}</td>
                    <td>OncoKB Actionable</td>
                    <td><span class="priority-badge high">HIGH</span></td>
                    <td>연구 필요</td>
                    <td>{direction} {abs(lfc):.2f} 발현 변화</td>
                </tr>'''

        # TME-based immunotherapy prediction
        tme_interpretation = ""
        if tme_composition is not None and len(tme_composition) > 0:
            immune_cells = tme_composition[tme_composition['category'].str.contains('immune|T_cell|NK|B_cell', case=False, na=False)]
            immune_pct = immune_cells['percentage'].sum() if len(immune_cells) > 0 else 0

            if immune_pct > 30:
                tme_class = "hot"
                tme_label = "Hot (Inflamed)"
                immunotherapy_pred = "면역관문억제제 반응 가능성 높음"
            elif immune_pct > 15:
                tme_class = "altered"
                tme_label = "Altered"
                immunotherapy_pred = "면역치료 + 병용요법 고려"
            else:
                tme_class = "cold"
                tme_label = "Cold (Desert)"
                immunotherapy_pred = "면역치료 저항 가능성, 병용요법 필요"

            tme_interpretation = f'''
            <div class="tme-immunotherapy" style="margin-top:24px;padding:20px;background:var(--bg-tertiary);border-radius:8px;">
                <h4 style="margin-bottom:12px;">🛡️ TME 기반 면역치료 반응 예측</h4>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                    <div>
                        <span style="font-weight:600;">TME 표현형:</span>
                        <span class="tme-badge {tme_class}" style="padding:4px 12px;border-radius:4px;margin-left:8px;">{tme_label}</span>
                    </div>
                    <div>
                        <span style="font-weight:600;">면역세포 비율:</span>
                        <span>{immune_pct:.1f}%</span>
                    </div>
                </div>
                <p style="margin-top:12px;font-size:13px;color:var(--text-secondary);">
                    <strong>예측:</strong> {immunotherapy_pred}
                </p>
            </div>
            '''

        return f'''
        <div class="ai-box orange" style="margin-bottom:20px;padding:16px;background:#fff7ed;border-left:4px solid #d97706;border-radius:4px;">
            <div style="font-weight:600;margin-bottom:8px;">임상적 의미 요약</div>
            <p style="font-size:13px;color:var(--text-secondary);">
                본 Single-cell 분석에서 식별된 세포 유형과 마커 유전자들은 {predicted_cancer_kr} ({predicted_cancer})의
                세포 이질성 이해, 치료 반응 예측, 그리고 정밀 의료 전략 수립에 활용될 수 있습니다.
            </p>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">바이오마커 후보</h3>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr><th>유전자</th><th>유형</th><th>근거</th><th>설명</th></tr>
                    </thead>
                    <tbody>
                        {biomarker_rows if biomarker_rows else '<tr><td colspan="4" style="color:#999;">바이오마커 후보가 없습니다.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">치료 표적 후보</h3>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr><th>유전자</th><th>분류</th><th>우선순위</th><th>기존 약물</th><th>근거</th></tr>
                    </thead>
                    <tbody>
                        {therapeutic_rows if therapeutic_rows else '<tr><td colspan="5" style="color:#999;">치료 표적 후보가 없습니다.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        {tme_interpretation}

        <div style="margin-top:24px;padding:16px;background:#fef3c7;border-radius:8px;font-size:12px;">
            <strong>⚠️ 중요 안내:</strong> Single-cell 분석 기반 임상적 시사점은 계산적 예측이며,
            진단 또는 치료 적용 전 반드시 실험적·임상적 검증이 필요합니다.
            세포 수준의 이질성 분석은 Bulk RNA-seq 대비 해상도가 높으나, 충분한 세포 수 확보가 중요합니다.
        </div>
        '''

    def _followup_experiments_html(self, data: Dict[str, Any]) -> str:
        """Generate Follow-up Experiments section for single-cell analysis."""
        driver_genes_df = data.get('driver_genes')
        cluster_markers = data.get('cluster_markers')
        tme_composition = data.get('tme_composition')

        # Get top genes for validation
        top_genes = []
        if driver_genes_df is not None and len(driver_genes_df) > 0:
            top_genes = driver_genes_df.head(5)['gene'].tolist()
        elif cluster_markers is not None and len(cluster_markers) > 0:
            top_genes = cluster_markers.groupby('cluster').head(1)['gene'].tolist()[:5]

        genes_str = ', '.join(top_genes) if top_genes else '분석된 후보 유전자들'

        # Cell type specific validation targets
        celltype_validation = ""
        if tme_composition is not None and len(tme_composition) > 0:
            major_celltypes = tme_composition.nlargest(3, 'count')['category'].tolist()
            celltype_validation = f'''
            <div style="margin-top:24px;">
                <h4 style="margin-bottom:12px;">세포 유형별 FACS 정렬 검증</h4>
                <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px;">
                    주요 세포 유형 ({', '.join(major_celltypes)})을 FACS로 분리 후 개별 검증
                </p>
            </div>
            '''

        return f'''
        <div class="ai-box green" style="margin-bottom:20px;padding:16px;background:#f0fdf4;border-left:4px solid #059669;border-radius:4px;">
            <div style="font-weight:600;margin-bottom:8px;">실험 검증 전략 요약</div>
            <p style="font-size:13px;">
                본 분석에서 식별된 <strong>{genes_str}</strong>에 대해 아래 단계적 검증 실험을 권장합니다.
                Single-cell 데이터의 특성상 세포 유형별 분리 검증이 핵심입니다.
            </p>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">9.1 발현 수준 검증</h3>
            <table>
                <thead>
                    <tr><th>Method</th><th>Target</th><th>Purpose</th><th>Sample Type</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>scRNA-seq 재현</strong></td>
                        <td>{genes_str[:30]}...</td>
                        <td>독립 샘플에서 클러스터 마커 재현성 확인</td>
                        <td>신선 조직 (10X Genomics)</td>
                    </tr>
                    <tr>
                        <td><strong>Spatial Transcriptomics</strong></td>
                        <td>주요 마커 유전자</td>
                        <td>조직 내 공간적 발현 패턴 검증</td>
                        <td>FFPE/신선 조직</td>
                    </tr>
                    <tr>
                        <td><strong>FACS + qRT-PCR</strong></td>
                        <td>세포 유형별 마커</td>
                        <td>세포 분리 후 마커 발현 확인</td>
                        <td>단일 세포 현탁액</td>
                    </tr>
                    <tr>
                        <td><strong>Multiplex IF/IHC</strong></td>
                        <td>단백질 마커 패널</td>
                        <td>단백질 수준 발현 및 공동 발현 확인</td>
                        <td>FFPE 조직</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">9.2 기능 연구</h3>
            <table>
                <thead>
                    <tr><th>Experiment</th><th>Target</th><th>Method</th><th>Readout</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Organoid Culture</strong></td>
                        <td>종양 세포 클러스터</td>
                        <td>3D organoid 배양</td>
                        <td>세포 유형 구성, 약물 반응</td>
                    </tr>
                    <tr>
                        <td><strong>CRISPR Perturbation</strong></td>
                        <td>{top_genes[0] if top_genes else 'target gene'}</td>
                        <td>Perturb-seq (scRNA-seq + CRISPR)</td>
                        <td>유전자 기능 및 하위 경로</td>
                    </tr>
                    <tr>
                        <td><strong>Co-culture Assay</strong></td>
                        <td>면역세포 + 종양세포</td>
                        <td>T cell cytotoxicity assay</td>
                        <td>면역 살상 능력</td>
                    </tr>
                    <tr>
                        <td><strong>Drug Screen</strong></td>
                        <td>주요 클러스터</td>
                        <td>세포 유형별 약물 감수성</td>
                        <td>IC50, 세포 사멸 비율</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">9.3 임상 검증</h3>
            <table>
                <thead>
                    <tr><th>Study Type</th><th>Description</th><th>Expected Outcome</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Independent Cohort</strong></td>
                        <td>독립 환자 코호트에서 세포 유형 구성 비교</td>
                        <td>세포 유형 비율과 예후 연관성</td>
                    </tr>
                    <tr>
                        <td><strong>Treatment Response</strong></td>
                        <td>치료 전후 샘플 비교 (paired scRNA-seq)</td>
                        <td>치료 반응 바이오마커 식별</td>
                    </tr>
                    <tr>
                        <td><strong>Multi-region Sampling</strong></td>
                        <td>종양 내 여러 부위에서 샘플링</td>
                        <td>종양 내 이질성 지도</td>
                    </tr>
                </tbody>
            </table>
        </div>

        {celltype_validation}

        <div style="padding:16px;background:var(--bg-tertiary);border-radius:8px;">
            <h4 style="margin-bottom:8px;">우선순위 권장사항</h4>
            <p style="font-size:13px;"><strong>1순위:</strong> Spatial Transcriptomics로 공간적 발현 패턴 검증</p>
            <p style="font-size:13px;"><strong>2순위:</strong> FACS 분리 + qRT-PCR로 세포 유형별 마커 검증</p>
            <p style="font-size:13px;"><strong>3순위:</strong> Organoid/Perturb-seq으로 기능 연구 수행</p>
        </div>
        '''

    def _methods_scorecard_html(self, data: Dict[str, Any]) -> str:
        """Generate Methods & Quality Scorecard section."""
        adata_summary = data.get('adata_summary', {})
        n_cells = adata_summary.get('n_cells', 0)
        n_genes = adata_summary.get('n_genes', 0)
        n_clusters = adata_summary.get('n_clusters', 0)
        cancer_prediction = data.get('cancer_prediction', {})
        confidence = cancer_prediction.get('confidence', 0)

        # Calculate quality scores
        cell_score = min(5, max(1, int(n_cells / 2000))) if n_cells > 0 else 1
        gene_score = min(5, max(1, int(n_genes / 4000))) if n_genes > 0 else 1
        cluster_score = 5 if 5 <= n_clusters <= 20 else (3 if n_clusters > 0 else 1)
        ml_score = min(5, max(1, int(confidence * 5))) if confidence > 0 else 1

        # Tool performance indicators
        def score_dots(score):
            return "🟢" * score + "⚪" * (5 - score)

        return f'''
        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">분석 도구 및 파라미터</h3>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;">
                <div style="background:var(--bg-tertiary);padding:16px;border-radius:8px;">
                    <h4 style="font-size:13px;margin-bottom:8px;">🧬 전처리 (Preprocessing)</h4>
                    <ul style="font-size:12px;margin-left:16px;">
                        <li>도구: Scanpy v1.9+</li>
                        <li>QC: min_genes={self.config.get('min_genes_per_cell', 200)}, max_mito={self.config.get('max_mito_percent', 20)}%</li>
                        <li>정규화: Total count + log1p</li>
                        <li>HVG: {self.config.get('n_top_genes', 2000)} genes ({self.config.get('hvg_flavor', 'seurat_v3')})</li>
                    </ul>
                </div>
                <div style="background:var(--bg-tertiary);padding:16px;border-radius:8px;">
                    <h4 style="font-size:13px;margin-bottom:8px;">📊 차원축소 & 클러스터링</h4>
                    <ul style="font-size:12px;margin-left:16px;">
                        <li>PCA: {self.config.get('n_pcs', 50)} components</li>
                        <li>UMAP: n_neighbors=15, min_dist=0.3</li>
                        <li>클러스터링: {self.config.get('clustering_method', 'leiden')}</li>
                        <li>Resolution: {self.config.get('clustering_resolution', 0.5)}</li>
                    </ul>
                </div>
                <div style="background:var(--bg-tertiary);padding:16px;border-radius:8px;">
                    <h4 style="font-size:13px;margin-bottom:8px;">🔬 DEG & Annotation</h4>
                    <ul style="font-size:12px;margin-left:16px;">
                        <li>DEG: {self.config.get('deg_method', 'wilcoxon')}</li>
                        <li>Cell type: CellTypist / Marker-based</li>
                        <li>Driver: COSMIC Tier1 + OncoKB</li>
                        <li>Pathway: Enrichr (GO, KEGG)</li>
                    </ul>
                </div>
                <div style="background:var(--bg-tertiary);padding:16px;border-radius:8px;">
                    <h4 style="font-size:13px;margin-bottom:8px;">🤖 ML & 고급 분석</h4>
                    <ul style="font-size:12px;margin-left:16px;">
                        <li>암종 예측: Pan-Cancer CatBoost</li>
                        <li>Trajectory: PAGA + DPT</li>
                        <li>GRN: Correlation-based TF analysis</li>
                        <li>Ploidy: InferPloidy (악성세포 감별)</li>
                    </ul>
                </div>
            </div>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">품질 점수표 (Quality Scorecard)</h3>
            <table>
                <thead>
                    <tr><th>평가 항목</th><th>현재 값</th><th>점수</th><th>평가</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>세포 수 (Cell Count)</td>
                        <td>{n_cells:,}개</td>
                        <td>{score_dots(cell_score)}</td>
                        <td>{'충분' if cell_score >= 4 else '적정' if cell_score >= 2 else '부족'}</td>
                    </tr>
                    <tr>
                        <td>유전자 수 (Gene Count)</td>
                        <td>{n_genes:,}개</td>
                        <td>{score_dots(gene_score)}</td>
                        <td>{'충분' if gene_score >= 4 else '적정' if gene_score >= 2 else '부족'}</td>
                    </tr>
                    <tr>
                        <td>클러스터 수 (Clusters)</td>
                        <td>{n_clusters}개</td>
                        <td>{score_dots(cluster_score)}</td>
                        <td>{'적정' if cluster_score >= 4 else '확인 필요'}</td>
                    </tr>
                    <tr>
                        <td>ML 예측 신뢰도</td>
                        <td>{confidence:.1%}</td>
                        <td>{score_dots(ml_score)}</td>
                        <td>{'높음' if ml_score >= 4 else '중간' if ml_score >= 2 else '낮음'}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div style="background:#f8fafc;padding:20px;border-radius:8px;border:1px solid var(--border-light);">
            <h4 style="margin-bottom:12px;">신뢰도 점수 계산 기준</h4>
            <table style="font-size:12px;">
                <tr><td>세포 수 ≥10,000</td><td>+5점</td><td>충분한 통계적 검증력</td></tr>
                <tr><td>유전자 수 ≥20,000</td><td>+5점</td><td>전체 전사체 커버리지</td></tr>
                <tr><td>클러스터 수 5-20개</td><td>+5점</td><td>적절한 해상도</td></tr>
                <tr><td>ML 신뢰도 ≥80%</td><td>+5점</td><td>높은 예측 정확도</td></tr>
                <tr><td>Driver 유전자 발견</td><td>+1점</td><td>임상적 관련성</td></tr>
            </table>
        </div>
        '''

    def _research_recommendations_html(self, data: Dict[str, Any]) -> str:
        """Generate Research Recommendations section for single-cell analysis."""
        driver_genes_df = data.get('driver_genes')
        tme_composition = data.get('tme_composition')
        cancer_prediction = data.get('cancer_prediction', {})
        cluster_markers = data.get('cluster_markers')

        predicted_cancer = cancer_prediction.get('predicted_cancer', 'Unknown')

        # Determine research focus based on TME
        tme_focus = ""
        if tme_composition is not None and len(tme_composition) > 0:
            immune_cells = tme_composition[tme_composition['category'].str.contains('immune|T_cell|NK', case=False, na=False)]
            immune_pct = immune_cells['percentage'].sum() if len(immune_cells) > 0 else 0

            if immune_pct > 30:
                tme_focus = "면역 침윤이 높아 면역치료 기전 연구 및 반응 바이오마커 탐색에 적합"
            elif immune_pct > 15:
                tme_focus = "중간 수준의 면역 침윤으로 면역 활성화 전략 및 병용요법 연구 추천"
            else:
                tme_focus = "면역 사막(cold TME) 상태로 면역세포 유입 유도 전략 연구 필요"

        # Top driver genes for research
        top_drivers = []
        if driver_genes_df is not None and len(driver_genes_df) > 0:
            top_drivers = driver_genes_df.head(5)['gene'].tolist()

        drivers_str = ', '.join(top_drivers) if top_drivers else '주요 마커 유전자'

        return f'''
        <div class="rec-intro" style="margin-bottom:24px;padding:16px;background:var(--bg-tertiary);border-radius:8px;">
            <p style="font-size:13px;">
                본 섹션은 Single-cell RNA-seq 분석 결과를 바탕으로 후속 연구 방향을 제안합니다.
                세포 유형별 이질성, 종양 미세환경 특성, 그리고 잠재적 치료 표적을 고려한 연구 전략을 포함합니다.
            </p>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">🎯 단기 연구 방향 (6개월 이내)</h3>
            <div style="background:white;padding:16px;border:1px solid var(--border-light);border-radius:8px;">
                <ul style="margin-left:16px;font-size:13px;">
                    <li><strong>마커 검증:</strong> {drivers_str}의 독립 코호트 발현 검증</li>
                    <li><strong>세포 분리:</strong> FACS/MACS를 이용한 주요 세포 유형 분리 및 기능 분석</li>
                    <li><strong>Spatial 검증:</strong> Visium/MERFISH로 세포 유형 공간적 분포 확인</li>
                    <li><strong>바이오마커 개발:</strong> 진단/예후 마커 패널 구성 및 임상 유용성 평가</li>
                </ul>
            </div>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">🔬 중기 연구 방향 (6개월-2년)</h3>
            <div style="background:white;padding:16px;border:1px solid var(--border-light);border-radius:8px;">
                <ul style="margin-left:16px;font-size:13px;">
                    <li><strong>기능 연구:</strong> Perturb-seq을 통한 주요 유전자 기능 규명</li>
                    <li><strong>TME 연구:</strong> {tme_focus if tme_focus else '종양 미세환경 특성 규명'}</li>
                    <li><strong>약물 스크리닝:</strong> 세포 유형별 약물 감수성 프로파일링</li>
                    <li><strong>Organoid 모델:</strong> 환자 유래 오가노이드에서 세포 구성 재현 확인</li>
                    <li><strong>다중오믹스:</strong> scATAC-seq, CITE-seq 통합 분석</li>
                </ul>
            </div>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">🚀 장기 연구 방향 (2년 이상)</h3>
            <div style="background:white;padding:16px;border:1px solid var(--border-light);border-radius:8px;">
                <ul style="margin-left:16px;font-size:13px;">
                    <li><strong>임상 시험:</strong> 세포 유형 기반 바이오마커 전향적 검증</li>
                    <li><strong>정밀 의료:</strong> 환자별 세포 구성에 따른 맞춤 치료 전략 개발</li>
                    <li><strong>내성 연구:</strong> 치료 전후 세포 구성 변화 및 내성 기전 규명</li>
                    <li><strong>Atlas 구축:</strong> {predicted_cancer} 세포 유형 레퍼런스 아틀라스 기여</li>
                </ul>
            </div>
        </div>

        <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;margin-bottom:12px;">🤝 협력 연구 제안</h3>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;">
                <div style="padding:12px;background:#eff6ff;border-radius:8px;">
                    <strong style="font-size:12px;">병리학 협력</strong>
                    <p style="font-size:11px;margin-top:4px;">조직학적 검증, Multiplex IHC/IF</p>
                </div>
                <div style="padding:12px;background:#f0fdf4;border-radius:8px;">
                    <strong style="font-size:12px;">임상 협력</strong>
                    <p style="font-size:11px;margin-top:4px;">전향적 코호트, 치료 반응 데이터</p>
                </div>
                <div style="padding:12px;background:#fef3c7;border-radius:8px;">
                    <strong style="font-size:12px;">생물정보학 협력</strong>
                    <p style="font-size:11px;margin-top:4px;">다중오믹스 통합, AI 모델 개발</p>
                </div>
                <div style="padding:12px;background:#fce7f3;border-radius:8px;">
                    <strong style="font-size:12px;">제약/바이오텍 협력</strong>
                    <p style="font-size:11px;margin-top:4px;">약물 스크리닝, 표적 치료제 개발</p>
                </div>
            </div>
        </div>

        <div style="padding:16px;background:#fff7ed;border-left:4px solid #d97706;border-radius:4px;">
            <h4 style="margin-bottom:8px;font-size:13px;">⚠️ 연구 수행 시 주의사항</h4>
            <ul style="font-size:12px;margin-left:16px;color:var(--text-secondary);">
                <li>Single-cell 데이터의 기술적 노이즈(dropout, batch effect)를 고려한 분석 필요</li>
                <li>세포 수 및 시퀀싱 깊이가 결과 해석에 영향을 미칠 수 있음</li>
                <li>in silico 예측은 반드시 실험적 검증을 통해 확인 필요</li>
                <li>환자 간 이질성을 고려한 충분한 샘플 수 확보 중요</li>
            </ul>
        </div>
        '''

    def validate_outputs(self) -> bool:
        """Validate that required output files were created."""
        required_files = ["report.html"]
        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing output file: {filename}")
                return False
        return True
