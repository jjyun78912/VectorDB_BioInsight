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
            "cnv_by_celltype.csv",
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
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 24px;
}

.figure-panel {
    background: var(--bg-tertiary);
    border-radius: 8px;
    overflow: hidden;
}

.figure-panel .title {
    padding: 12px 16px;
    font-weight: 600;
    font-size: 13px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-light);
}

.figure-panel img {
    width: 100%;
    height: auto;
    display: block;
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
        <a href="#cnv">CNV</a>
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

        <!-- NEW: Cancer Prediction Section -->
        <section id="prediction" class="section">
            <h2>🔬 암종 예측 (Pseudo-bulk ML)</h2>
            {self._cancer_prediction_html(cancer_prediction)}
        </section>

        <!-- Cell Type Analysis Section (UMAP + Bar Chart) -->
        <section id="celltype" class="section">
            <h2>세포 유형 분석 (Cell Type Analysis)</h2>
            <div class="figure-grid">
                {self._figure_html(figures, 'umap_celltypes', 'Cell Types (UMAP)')}
                {self._figure_html(figures, 'celltype_barchart', 'Cell Type Composition')}
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

        <!-- NEW: CNV Inference Section -->
        <section id="cnv" class="section">
            <h2>📊 CNV 추론 (악성세포 감별)</h2>
            {self._cnv_analysis_html(data.get('cnv_by_celltype'))}
        </section>

        <!-- QC Section -->
        <section id="qc" class="section">
            <h2>Quality Control</h2>
            <div class="figure-grid">
                {self._figure_html(figures, 'violin_qc', 'QC Metrics')}
            </div>
        </section>

        <!-- Methods Section -->
        <section class="section">
            <h2>분석 방법</h2>
            <p>본 분석은 Scanpy 기반 single-cell RNA-seq 파이프라인을 사용하였습니다.</p>
            <ul style="margin-top: 12px; margin-left: 24px;">
                <li>QC Filtering: min_genes={self.config.get('min_genes_per_cell', 200)}, max_mito={self.config.get('max_mito_percent', 20)}%</li>
                <li>Normalization: Total count normalization + log1p</li>
                <li>HVG Selection: Top {self.config.get('n_top_genes', 2000)} genes ({self.config.get('hvg_flavor', 'seurat_v3')})</li>
                <li>Dimensionality Reduction: PCA ({self.config.get('n_pcs', 50)} components)</li>
                <li>Clustering: {self.config.get('clustering_method', 'leiden')} (resolution={self.config.get('clustering_resolution', 0.5)})</li>
                <li>DEG Analysis: {self.config.get('deg_method', 'wilcoxon')}</li>
                <li>Cancer Prediction: Pan-Cancer CatBoost (Pseudo-bulk aggregation)</li>
                <li>Driver Matching: COSMIC Tier1 + OncoKB Actionable</li>
                <li>Pathway: Enrichr (GO, KEGG)</li>
                <li>Trajectory: PAGA + Diffusion Pseudotime</li>
                <li>Cell-Cell Interaction: Ligand-Receptor analysis</li>
            </ul>
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
        """Generate HTML for a figure panel."""
        if key not in figures:
            return f'<div class="figure-panel"><div class="title">{title}</div><p style="padding:24px;color:#999;">Figure not available</p></div>'

        return f'''<div class="figure-panel">
            <div class="title">{title}</div>
            <img src="{figures[key]}" alt="{title}">
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
        """Generate marker genes table."""
        if df is None or len(df) == 0:
            return '<p>No marker data available</p>'

        # Get top markers per cluster
        max_markers = self.config.get('max_markers_per_cluster', 10)
        if 'cluster' in df.columns:
            top_markers = df.groupby('cluster').head(max_markers)
        else:
            top_markers = df.head(100)

        rows_html = ''
        for _, row in top_markers.iterrows():
            cluster = row.get('cluster', '-')
            gene = row.get('gene', '-')
            score = row.get('score', 0)
            lfc = row.get('logfoldchange', 0)
            pval = row.get('pval_adj', row.get('pval', 1))

            rows_html += f'''<tr>
                <td><span class="cluster-badge cluster-{cluster}">{cluster}</span></td>
                <td><strong>{gene}</strong></td>
                <td>{score:.2f}</td>
                <td style="color:{'#dc2626' if lfc > 0 else '#2563eb'}">{lfc:+.2f}</td>
                <td>{pval:.2e}</td>
            </tr>'''

        return f'''
        <div class="table-wrapper" style="margin-top:24px;">
            <table>
                <thead>
                    <tr><th>Cluster</th><th>Gene</th><th>Score</th><th>Log2FC</th><th>Adj. P-value</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        '''

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

    def _cnv_analysis_html(self, cnv_by_celltype: Optional[pd.DataFrame]) -> str:
        """Generate HTML for CNV inference section."""
        if cnv_by_celltype is None or cnv_by_celltype.empty:
            return '<p style="color:#999;">CNV 분석 데이터가 없습니다.</p>'

        html = '''
        <div class="content-box">
            <h3>세포 유형별 CNV 분석</h3>
            <p style="font-size:12px;color:#666;margin-bottom:16px;">
                inferCNV 스타일의 복제수 변이(CNV) 추론 결과입니다.
                MYC, EGFR 등 종양 유전자 증폭과 TP53, CDKN2A 등 종양 억제 유전자 결실을 기반으로
                악성세포를 감별합니다.
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>세포 유형</th>
                        <th>CNV 점수 (평균)</th>
                        <th>악성 추정 세포 수</th>
                        <th>전체 세포 수</th>
                        <th>악성 비율 (%)</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for _, row in cnv_by_celltype.iterrows():
            cell_type = row.get('cell_type', row.get('index', ''))
            cnv_mean = row.get('cnv_mean', 0)
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
                        <td>{cnv_mean:.2f}</td>
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
                <li>CNV 점수가 높다고 반드시 악성세포를 의미하지 않습니다.</li>
                <li>정상 세포도 체세포 변이를 가질 수 있습니다.</li>
                <li>최종 판단은 조직학적 검사와 전문가 검토가 필요합니다.</li>
                <li>면역세포(T, NK, B cells)를 정상 참조군으로 사용하였습니다.</li>
            </ul>
        </div>
        '''

        return html

    def validate_outputs(self) -> bool:
        """Validate that required output files were created."""
        required_files = ["report.html"]
        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing output file: {filename}")
                return False
        return True
