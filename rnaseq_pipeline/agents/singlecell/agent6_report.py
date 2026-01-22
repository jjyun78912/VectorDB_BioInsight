"""
Single-Cell Agent 6: Visualization & Report Generation

Comprehensive HTML report with all visualizations and LLM interpretation.

Input:
- All outputs from Agents 1-5

Output:
- singlecell_report.html: Comprehensive HTML report
- figures/: All visualization plots
- report_data.json: Report data summary
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

from ...utils.base_agent import BaseAgent

# Import scanpy
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

# Import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class SingleCellReportAgent(BaseAgent):
    """Agent 6: Visualization & Report Generation."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Report settings
            "report_title": "Single-Cell RNA-seq Analysis Report",
            "cancer_type": "Unknown",
            "sample_id": "Sample",

            # Visualization settings
            "generate_interactive_plots": True,
            "top_markers_per_cluster": 5,
            "top_pathways_to_show": 10,

            # LLM interpretation
            "enable_llm_interpretation": True,

            # Language
            "language": "ko",  # ko or en
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent6_sc_report", input_dir, output_dir, merged_config)

        self.adata = None
        self.report_data = {}
        self.figures = {}

    def validate_inputs(self) -> bool:
        """Validate input files."""
        if not HAS_SCANPY:
            self.logger.error("Scanpy not installed")
            return False

        # Load final h5ad
        for name in ["adata_cnv.h5ad", "adata_trajectory.h5ad", "adata_clustered.h5ad"]:
            h5ad_file = self.input_dir / name
            if h5ad_file.exists():
                self.adata = sc.read_h5ad(h5ad_file)
                self.logger.info(f"Loaded: {h5ad_file.name}")
                break

        if self.adata is None:
            h5ad_files = list(self.input_dir.glob("*.h5ad"))
            if h5ad_files:
                self.adata = sc.read_h5ad(h5ad_files[0])
            else:
                self.logger.error("No h5ad file found")
                return False

        self.logger.info(f"Data: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        return True

    def run(self) -> Dict[str, Any]:
        """Execute report generation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Agent 6: Visualization & Report Generation")
        self.logger.info("=" * 60)

        # Step 1: Load all results from previous agents
        self._load_all_results()

        # Step 2: Generate all visualizations
        self._generate_all_visualizations()

        # Step 3: Generate HTML report
        self._generate_html_report()

        # Step 4: Save outputs
        self._save_outputs()

        return {
            "status": "success",
            "report_file": str(self.output_dir / "singlecell_report.html"),
            "n_figures": len(self.figures),
            "report_sections": list(self.report_data.keys())
        }

    def _load_all_results(self):
        """Load results from all previous agents."""
        self.logger.info("Loading results from previous agents...")

        # QC statistics
        qc_file = self.input_dir / "qc_statistics.json"
        if qc_file.exists():
            with open(qc_file) as f:
                self.report_data['qc'] = json.load(f)

        # Cell cycle info
        cc_file = self.input_dir / "cell_cycle_info.json"
        if cc_file.exists():
            with open(cc_file) as f:
                self.report_data['cell_cycle'] = json.load(f)

        # Cell type predictions
        ct_file = self.input_dir / "celltype_predictions.json"
        if ct_file.exists():
            with open(ct_file) as f:
                self.report_data['celltype'] = json.load(f)

        # Markers
        markers_file = self.input_dir / "cluster_markers.csv"
        if markers_file.exists():
            self.report_data['markers'] = pd.read_csv(markers_file)

        # Cell composition
        comp_file = self.input_dir / "cell_composition.csv"
        if comp_file.exists():
            self.report_data['composition'] = pd.read_csv(comp_file)

        # Pathway results
        pathway_file = self.input_dir / "cluster_pathways.csv"
        if pathway_file.exists():
            self.report_data['pathways'] = pd.read_csv(pathway_file)

        # Driver genes
        driver_file = self.input_dir / "driver_genes.csv"
        if driver_file.exists():
            self.report_data['drivers'] = pd.read_csv(driver_file)

        # TME scores
        tme_file = self.input_dir / "tme_scores.csv"
        if tme_file.exists():
            self.report_data['tme'] = pd.read_csv(tme_file)

        # Trajectory results
        traj_file = self.input_dir / "trajectory_results.json"
        if traj_file.exists():
            with open(traj_file) as f:
                self.report_data['trajectory'] = json.load(f)

        # Cancer prediction
        cancer_file = self.input_dir / "cancer_prediction.json"
        if cancer_file.exists():
            with open(cancer_file) as f:
                self.report_data['cancer_prediction'] = json.load(f)

        # Malignant results
        mal_file = self.input_dir / "malignant_results.json"
        if mal_file.exists():
            with open(mal_file) as f:
                self.report_data['malignant'] = json.load(f)

        self.logger.info(f"  Loaded {len(self.report_data)} result sets")

    def _generate_all_visualizations(self):
        """Generate all visualization plots.

        Order follows analysis flow:
        1. Marker Heatmap & Dotplot (annotation 근거)
        2. Cell Type UMAP & Bar Chart (최종 결과)
        3. Other plots (Trajectory, Malignant, etc.)
        """
        self.logger.info("Generating visualizations...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # 1. Marker plots (Heatmap & Dotplot) - annotation 근거
        if 'markers' in self.report_data:
            self._generate_marker_plots(figures_dir)

        # 2. Cell Type UMAP & Composition - 최종 결과
        if 'X_umap' in self.adata.obsm and 'cell_type' in self.adata.obs:
            self._generate_celltype_plots(figures_dir)

        # 3. Other UMAPs (Pseudotime, Malignant)
        if 'X_umap' in self.adata.obsm:
            self._generate_other_umaps(figures_dir)

        # 4. Interactive UMAP (Plotly)
        if HAS_PLOTLY and self.config.get("generate_interactive_plots", True):
            self._generate_interactive_umap(figures_dir)

        self.logger.info(f"  Generated {len(self.figures)} figures")

    def _generate_marker_plots(self, figures_dir: Path):
        """Generate Marker Heatmap & Dotplot (Step 2: annotation basis)."""
        if not HAS_MATPLOTLIB:
            return

        try:
            markers_df = self.report_data['markers']
            top_n = self.config.get("top_markers_per_cluster", 5)

            # Get top markers per cluster
            top_markers = markers_df.groupby('cluster').apply(
                lambda x: x.nlargest(top_n, 'score')
            ).reset_index(drop=True)

            genes = top_markers['gene'].unique().tolist()[:50]

            if not genes:
                return

            # 1. Marker Heatmap
            try:
                sc.pl.heatmap(
                    self.adata,
                    var_names=genes,
                    groupby='cluster',
                    show=False,
                    cmap='viridis',
                    figsize=(14, 10),
                    save='_markers.png'
                )
                # Move saved file from scanpy default location
                scanpy_path = Path("./figures/heatmap_markers.png")
                path = figures_dir / "marker_heatmap.png"
                if scanpy_path.exists():
                    import shutil
                    shutil.move(str(scanpy_path), str(path))
                    self.figures['marker_heatmap'] = str(path)
            except Exception as e:
                self.logger.warning(f"Marker heatmap failed: {e}")

            # 2. Marker Dotplot
            try:
                # Use fewer genes for dotplot readability
                dotplot_genes = genes[:30]
                sc.pl.dotplot(
                    self.adata,
                    var_names=dotplot_genes,
                    groupby='cluster',
                    show=False,
                    figsize=(16, 8),
                    save='_markers.png'
                )
                # Move saved file from scanpy default location
                scanpy_path = Path("./figures/dotplot_markers.png")
                path = figures_dir / "marker_dotplot.png"
                if scanpy_path.exists():
                    import shutil
                    shutil.move(str(scanpy_path), str(path))
                    self.figures['marker_dotplot'] = str(path)
            except Exception as e:
                self.logger.warning(f"Marker dotplot failed: {e}")

        except Exception as e:
            self.logger.warning(f"Marker plots failed: {e}")

    def _generate_celltype_plots(self, figures_dir: Path):
        """Generate Cell Type UMAP & Composition (Step 3: annotation results)."""
        if not HAS_MATPLOTLIB:
            return

        # Get unique cell types for consistent coloring
        cell_types = self.adata.obs['cell_type'].unique().tolist()
        n_types = len(cell_types)

        # Create consistent color palette
        if n_types <= 10:
            cmap = plt.cm.tab10
        elif n_types <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.gist_ncar

        colors = {ct: cmap(i / n_types) for i, ct in enumerate(cell_types)}

        try:
            # 1. Cell Type UMAP
            fig, ax = plt.subplots(figsize=(12, 8))
            sc.pl.umap(self.adata, color='cell_type', ax=ax, show=False,
                      palette=colors)
            plt.title('UMAP - Cell Types (After Annotation)')
            plt.tight_layout()
            path = figures_dir / "umap_celltypes.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures['umap_celltypes'] = str(path)

        except Exception as e:
            self.logger.warning(f"Cell type UMAP failed: {e}")

        try:
            # 2. Cell Type Composition Bar Chart (same colors as UMAP)
            if 'composition' in self.report_data:
                comp_df = self.report_data['composition']

                # Aggregate by cell type
                celltype_counts = comp_df.groupby('cell_type')['count'].sum().sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                bar_colors = [colors.get(ct, 'gray') for ct in celltype_counts.index]
                celltype_counts.plot(kind='barh', ax=ax, color=bar_colors)

                ax.set_xlabel('Number of Cells')
                ax.set_ylabel('Cell Type')
                ax.set_title('Cell Type Composition')

                # Add count labels
                for i, (ct, count) in enumerate(celltype_counts.items()):
                    ax.text(count + 10, i, str(int(count)), va='center')

                plt.tight_layout()
                path = figures_dir / "celltype_barchart.png"
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                self.figures['celltype_barchart'] = str(path)

        except Exception as e:
            self.logger.warning(f"Cell type bar chart failed: {e}")

    def _generate_other_umaps(self, figures_dir: Path):
        """Generate other UMAP plots (Pseudotime, Malignant, etc.)."""
        if not HAS_MATPLOTLIB:
            return

        try:
            # UMAP by pseudotime
            if 'dpt_pseudotime' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(self.adata, color='dpt_pseudotime', ax=ax, show=False, cmap='viridis')
                plt.title('UMAP - Pseudotime')
                plt.tight_layout()
                path = figures_dir / "umap_pseudotime.png"
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                self.figures['umap_pseudotime'] = str(path)

            # UMAP by malignant status
            if 'is_malignant' in self.adata.obs:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(self.adata, color='is_malignant', ax=ax, show=False)
                plt.title('UMAP - Malignant Status')
                plt.tight_layout()
                path = figures_dir / "umap_malignant.png"
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                self.figures['umap_malignant'] = str(path)

        except Exception as e:
            self.logger.warning(f"Other UMAP plots failed: {e}")

    def _generate_interactive_umap(self, figures_dir: Path):
        """Generate interactive UMAP with Plotly."""
        if not HAS_PLOTLY or 'X_umap' not in self.adata.obsm:
            return

        try:
            # Prepare data
            umap_df = pd.DataFrame(
                self.adata.obsm['X_umap'],
                columns=['UMAP1', 'UMAP2']
            )
            umap_df['cluster'] = self.adata.obs['cluster'].values

            if 'cell_type' in self.adata.obs:
                umap_df['cell_type'] = self.adata.obs['cell_type'].values

            if 'malignant_score' in self.adata.obs:
                umap_df['malignant_score'] = self.adata.obs['malignant_score'].values

            # Create interactive plot
            fig = px.scatter(
                umap_df,
                x='UMAP1',
                y='UMAP2',
                color='cluster',
                hover_data=['cell_type'] if 'cell_type' in umap_df else None,
                title='Interactive UMAP - Clusters'
            )

            fig.update_layout(
                width=900,
                height=700,
                template='plotly_white'
            )

            path = figures_dir / "umap_interactive.html"
            fig.write_html(str(path))
            self.figures['umap_interactive'] = str(path)

        except Exception as e:
            self.logger.warning(f"Interactive UMAP failed: {e}")

    def _generate_html_report(self):
        """Generate comprehensive HTML report."""
        self.logger.info("Generating HTML report...")

        # Collect all data for report
        report_content = self._build_report_content()

        # Generate HTML
        html = self._render_html_template(report_content)

        # Save
        report_path = self.output_dir / "singlecell_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        self.logger.info(f"  Report saved: {report_path}")

    def _build_report_content(self) -> Dict[str, Any]:
        """Build content dictionary for report."""
        content = {
            "title": self.config.get("report_title", "Single-Cell Analysis Report"),
            "sample_id": self.config.get("sample_id", "Sample"),
            "cancer_type": self.config.get("cancer_type", "Unknown"),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_cells": self.adata.n_obs,
            "n_genes": self.adata.n_vars,
            "n_clusters": len(self.adata.obs['cluster'].unique()) if 'cluster' in self.adata.obs else 0,
        }

        # QC summary
        if 'qc' in self.report_data:
            qc = self.report_data['qc']
            content['qc_summary'] = {
                "initial_cells": qc.get('initial_cells', 'N/A'),
                "final_cells": qc.get('final_cells', 'N/A'),
                "cells_removed": qc.get('cells_removed_total', 'N/A'),
                "filters": qc.get('filters_applied', [])
            }

        # Cell cycle
        if 'cell_cycle' in self.report_data:
            content['cell_cycle'] = self.report_data['cell_cycle'].get('phase_distribution', {})

        # Cell types
        if 'celltype' in self.report_data:
            content['celltype'] = self.report_data['celltype']

        # Top markers per cluster
        if 'markers' in self.report_data:
            markers_df = self.report_data['markers']
            top_markers = markers_df.groupby('cluster').head(5)[['cluster', 'gene', 'logfoldchange', 'pval_adj']]
            content['top_markers'] = top_markers.to_dict('records')

        # Top pathways
        if 'pathways' in self.report_data:
            pathways_df = self.report_data['pathways']
            top_pathways = pathways_df.nlargest(10, 'combined_score')[['cluster', 'term', 'pval_adj', 'combined_score']]
            content['top_pathways'] = top_pathways.to_dict('records')

        # Driver genes
        if 'drivers' in self.report_data:
            content['driver_genes'] = self.report_data['drivers'].to_dict('records')

        # Cancer prediction
        if 'cancer_prediction' in self.report_data:
            content['cancer_prediction'] = self.report_data['cancer_prediction']

        # Malignant cells
        if 'malignant' in self.report_data:
            content['malignant'] = self.report_data['malignant']

        # Trajectory
        if 'trajectory' in self.report_data:
            content['trajectory'] = self.report_data['trajectory']

        # Figures
        content['figures'] = self.figures

        return content

    def _render_html_template(self, content: Dict[str, Any]) -> str:
        """Render HTML template with content."""
        is_korean = self.config.get("language", "ko") == "ko"

        # Labels
        labels = {
            "title": content['title'],
            "overview": "분석 개요" if is_korean else "Analysis Overview",
            "qc": "품질 관리 (QC)" if is_korean else "Quality Control",
            "clustering": "클러스터링 & 세포 유형" if is_korean else "Clustering & Cell Types",
            "markers": "마커 유전자" if is_korean else "Marker Genes",
            "pathways": "경로 분석" if is_korean else "Pathway Analysis",
            "drivers": "드라이버 유전자" if is_korean else "Driver Genes",
            "trajectory": "궤적 분석" if is_korean else "Trajectory Analysis",
            "ml_prediction": "ML 예측" if is_korean else "ML Predictions",
            "cancer_pred": "암종 예측" if is_korean else "Cancer Type Prediction",
            "malignant": "악성 세포 검출" if is_korean else "Malignant Cell Detection",
            "disclaimer": "⚠️ 본 분석 결과는 연구 참고용이며, 임상 진단에 사용할 수 없습니다." if is_korean else "⚠️ These results are for research purposes only and should not be used for clinical diagnosis."
        }

        html = f'''<!DOCTYPE html>
<html lang="{"ko" if is_korean else "en"}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{labels["title"]}</title>
    <style>
        :root {{
            --primary: #3498db;
            --secondary: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: #f5f7fa;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header .meta {{ opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: var(--primary);
            border-bottom: 2px solid var(--light);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-box {{
            background: var(--light);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--primary);
        }}
        .stat-box .label {{ color: #666; margin-top: 5px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--light);
        }}
        th {{ background: var(--light); font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .figure-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid var(--warning);
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .success-box {{
            background: #d4edda;
            border-left: 4px solid var(--secondary);
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .ml-prediction {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .ml-prediction h3 {{ margin-bottom: 15px; }}
        .prediction-result {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .confidence {{ opacity: 0.9; margin-top: 10px; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧬 {labels["title"]}</h1>
            <div class="meta">
                <p>Sample: {content['sample_id']} | Cancer Type: {content['cancer_type']}</p>
                <p>Generated: {content['date']}</p>
            </div>
        </header>

        <!-- Overview -->
        <div class="card">
            <h2>📊 {labels["overview"]}</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{content['n_cells']:,}</div>
                    <div class="label">{"세포 수" if is_korean else "Cells"}</div>
                </div>
                <div class="stat-box">
                    <div class="value">{content['n_genes']:,}</div>
                    <div class="label">{"유전자 수" if is_korean else "Genes"}</div>
                </div>
                <div class="stat-box">
                    <div class="value">{content['n_clusters']}</div>
                    <div class="label">{"클러스터" if is_korean else "Clusters"}</div>
                </div>
            </div>
        </div>
'''

        # QC Section
        if 'qc_summary' in content:
            qc = content['qc_summary']
            html += f'''
        <div class="card">
            <h2>🔬 {labels["qc"]}</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{qc['initial_cells']:,}</div>
                    <div class="label">{"초기 세포" if is_korean else "Initial Cells"}</div>
                </div>
                <div class="stat-box">
                    <div class="value">{qc['final_cells']:,}</div>
                    <div class="label">{"최종 세포" if is_korean else "Final Cells"}</div>
                </div>
                <div class="stat-box">
                    <div class="value">{qc['cells_removed']:,}</div>
                    <div class="label">{"제거된 세포" if is_korean else "Removed"}</div>
                </div>
            </div>
            <h3 style="margin-top: 20px;">{"적용된 필터" if is_korean else "Applied Filters"}</h3>
            <table>
                <tr><th>Filter</th><th>Details</th><th>Cells Removed</th></tr>
'''
            for f in qc.get('filters', []):
                html += f'                <tr><td>{f.get("filter", "N/A")}</td><td>{str(f)[:100]}</td><td>{f.get("cells_removed", "N/A")}</td></tr>\n'
            html += '''            </table>
        </div>
'''

        # Cell Cycle
        if 'cell_cycle' in content:
            cc = content['cell_cycle']
            html += f'''
        <div class="card">
            <h2>🔄 {"세포 주기" if is_korean else "Cell Cycle"}</h2>
            <div class="stats-grid">
'''
            for phase, count in cc.items():
                html += f'''                <div class="stat-box">
                    <div class="value">{count:,}</div>
                    <div class="label">{phase}</div>
                </div>
'''
            html += '''            </div>
        </div>
'''

        # Section 1: Marker Gene Analysis (Heatmap + Dotplot - annotation 근거)
        has_marker_plots = 'marker_heatmap' in self.figures or 'marker_dotplot' in self.figures
        if has_marker_plots or ('top_markers' in content and content['top_markers']):
            html += f'''
        <div class="card">
            <h2>🧬 {"1. 마커 유전자 분석 (Annotation 근거)" if is_korean else "1. Marker Gene Analysis (Annotation Basis)"}</h2>
            <p style="color: #666; margin-bottom: 15px;">{"각 클러스터의 특이적 마커 유전자를 기반으로 Cell Type을 결정합니다" if is_korean else "Cell types are determined based on cluster-specific marker genes"}</p>
'''
            # Marker Heatmap
            if 'marker_heatmap' in self.figures:
                html += '''            <div class="figure-container">
                <img src="figures/marker_heatmap.png" alt="Marker Heatmap">
                <p>Marker Gene Heatmap by Cluster</p>
            </div>
'''
            # Marker Dotplot
            if 'marker_dotplot' in self.figures:
                html += '''            <div class="figure-container">
                <img src="figures/marker_dotplot.png" alt="Marker Dotplot">
                <p>Marker Gene Dotplot by Cluster</p>
            </div>
'''
            # Marker Table
            if 'top_markers' in content and content['top_markers']:
                html += f'''            <h3 style="margin-top: 25px;">{"클러스터별 상위 마커 유전자" if is_korean else "Top Marker Genes per Cluster"}</h3>
            <table>
                <tr><th>Cluster</th><th>Gene</th><th>Log2FC</th><th>Adj. P-value</th></tr>
'''
                for m in content['top_markers'][:30]:
                    html += f'                <tr><td>{m["cluster"]}</td><td><b>{m["gene"]}</b></td><td>{m["logfoldchange"]:.2f}</td><td>{m["pval_adj"]:.2e}</td></tr>\n'
                html += '''            </table>
'''
            html += '''        </div>
'''

        # Section 2: Cell Type Annotation Results (UMAP + Bar Chart - 최종 결과)
        if 'umap_celltypes' in self.figures or 'celltype_barchart' in self.figures:
            html += f'''
        <div class="card">
            <h2>🏷️ {"2. Cell Type 분류 결과" if is_korean else "2. Cell Type Classification Results"}</h2>
            <p style="color: #666; margin-bottom: 15px;">{"마커 기반 annotation 최종 결과 (동일한 색상 팔레트 사용)" if is_korean else "Final marker-based annotation results (consistent color palette)"}</p>
            <div class="figure-grid">
'''
            if 'umap_celltypes' in self.figures:
                html += f'''                <div class="figure-container">
                    <img src="figures/umap_celltypes.png" alt="UMAP Cell Types">
                    <p>UMAP - Cell Types</p>
                </div>
'''
            if 'celltype_barchart' in self.figures:
                html += f'''                <div class="figure-container">
                    <img src="figures/celltype_barchart.png" alt="Cell Type Composition">
                    <p>{"Cell Type 구성 비율" if is_korean else "Cell Type Composition"}</p>
                </div>
'''
            html += '''            </div>
        </div>
'''

        # Pathway Analysis
        if 'top_pathways' in content and content['top_pathways']:
            html += f'''
        <div class="card">
            <h2>🛤️ {labels["pathways"]}</h2>
            <table>
                <tr><th>Cluster</th><th>Pathway</th><th>Adj. P-value</th><th>Score</th></tr>
'''
            for p in content['top_pathways']:
                html += f'                <tr><td>{p["cluster"]}</td><td>{p["term"][:60]}</td><td>{p["pval_adj"]:.2e}</td><td>{p["combined_score"]:.1f}</td></tr>\n'
            html += '''            </table>
        </div>
'''

        # Driver Genes
        if 'driver_genes' in content and content['driver_genes']:
            html += f'''
        <div class="card">
            <h2>🎯 {labels["drivers"]}</h2>
            <table>
                <tr><th>Cluster</th><th>Gene</th><th>Log2FC</th><th>COSMIC</th><th>OncoKB</th></tr>
'''
            for d in content['driver_genes'][:20]:
                cosmic = "✅" if d.get("in_cosmic") else "❌"
                oncokb = "✅" if d.get("in_oncokb") else "❌"
                html += f'                <tr><td>{d["cluster"]}</td><td><b>{d["gene"]}</b></td><td>{d["logfoldchange"]:.2f}</td><td>{cosmic}</td><td>{oncokb}</td></tr>\n'
            html += '''            </table>
        </div>
'''

        # ML Predictions
        html += f'''
        <div class="card">
            <h2>🤖 {labels["ml_prediction"]}</h2>
'''

        # Cancer Type Prediction (ML #2)
        if 'cancer_prediction' in content:
            pred = content['cancer_prediction']
            model_perf = pred.get('model_performance', {})

            # Build model performance scorecard HTML
            perf_html = ""
            if model_perf:
                overall = model_perf.get('overall', {})
                per_class = model_perf.get('per_class', {})
                ci = model_perf.get('confidence_interval', {})
                training = model_perf.get('training_info', {})

                perf_html = f'''
                <div class="model-performance" style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.15); border-radius: 8px; font-size: 0.85em;">
                    <div style="font-weight: bold; margin-bottom: 8px;">📊 {"모델 성능 지표" if is_korean else "Model Performance"}</div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                        <div style="text-align: center; padding: 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                            <div style="font-size: 1.1em; font-weight: bold;">{overall.get('accuracy', 0)*100:.1f}%</div>
                            <div style="font-size: 0.8em; opacity: 0.9;">Accuracy</div>
                        </div>
                        <div style="text-align: center; padding: 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                            <div style="font-size: 1.1em; font-weight: bold;">{overall.get('f1_macro', 0)*100:.1f}%</div>
                            <div style="font-size: 0.8em; opacity: 0.9;">F1 (Macro)</div>
                        </div>
                        <div style="text-align: center; padding: 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                            <div style="font-size: 1.1em; font-weight: bold;">{overall.get('mcc', 0):.3f}</div>
                            <div style="font-size: 0.8em; opacity: 0.9;">MCC</div>
                        </div>
                        <div style="text-align: center; padding: 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                            <div style="font-size: 1.1em; font-weight: bold;">{overall.get('pr_auc_macro', 0)*100:.1f}%</div>
                            <div style="font-size: 0.8em; opacity: 0.9;">PR-AUC</div>
                        </div>
                    </div>'''

                # Per-class metrics for predicted cancer type
                if per_class:
                    cancer_type = per_class.get('cancer_type', pred.get('predicted_type', ''))
                    perf_html += f'''
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                        <div style="font-size: 0.9em; margin-bottom: 6px;">📌 <b>{cancer_type}</b> {"분류 성능" if is_korean else "Class Performance"}</div>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; font-size: 0.8em;">
                            <div style="text-align: center;">
                                <div style="font-weight: bold;">{per_class.get('f1', 0)*100:.1f}%</div>
                                <div style="opacity: 0.8;">F1</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-weight: bold;">{per_class.get('precision', 0)*100:.1f}%</div>
                                <div style="opacity: 0.8;">Precision</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-weight: bold;">{per_class.get('recall', 0)*100:.1f}%</div>
                                <div style="opacity: 0.8;">Recall</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-weight: bold;">{per_class.get('pr_auc', 0)*100:.1f}%</div>
                                <div style="opacity: 0.8;">PR-AUC</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-weight: bold;">{per_class.get('roc_auc', 0)*100:.1f}%</div>
                                <div style="opacity: 0.8;">ROC-AUC</div>
                            </div>
                        </div>
                    </div>'''

                perf_html += '''
                </div>'''

            html += f'''
            <div class="ml-prediction">
                <h3>ML #2: {labels["cancer_pred"]}</h3>
                <div class="prediction-result">{pred.get('predicted_type', 'Unknown')}</div>
                <div class="confidence">{"신뢰도" if is_korean else "Confidence"}: {pred.get('confidence', 0)*100:.1f}%</div>
                {perf_html}
                <div style="margin-top: 10px; font-size: 0.9em;">{pred.get('warning', '')}</div>
            </div>
'''

        # Malignant Cell Detection (ML #3)
        if 'malignant' in content:
            mal = content['malignant']
            html += f'''
            <div class="ml-prediction" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <h3>ML #3: {labels["malignant"]}</h3>
                <div class="stats-grid" style="margin-top: 15px;">
                    <div class="stat-box" style="background: rgba(255,255,255,0.2); color: white;">
                        <div class="value" style="color: white;">{mal.get('n_malignant', 0):,}</div>
                        <div class="label" style="color: rgba(255,255,255,0.9);">{"악성 세포" if is_korean else "Malignant"}</div>
                    </div>
                    <div class="stat-box" style="background: rgba(255,255,255,0.2); color: white;">
                        <div class="value" style="color: white;">{mal.get('n_normal', 0):,}</div>
                        <div class="label" style="color: rgba(255,255,255,0.9);">{"정상 세포" if is_korean else "Normal"}</div>
                    </div>
                    <div class="stat-box" style="background: rgba(255,255,255,0.2); color: white;">
                        <div class="value" style="color: white;">{mal.get('pct_malignant', 0):.1f}%</div>
                        <div class="label" style="color: rgba(255,255,255,0.9);">{"악성 비율" if is_korean else "Malignant %"}</div>
                    </div>
                </div>
            </div>
'''

        html += '''        </div>
'''

        # Trajectory
        if 'trajectory' in content:
            traj = content['trajectory']
            html += f'''
        <div class="card">
            <h2>🔀 {labels["trajectory"]}</h2>
'''
            if 'umap_pseudotime' in self.figures:
                html += '''            <div class="figure-container"><img src="figures/umap_pseudotime.png" alt="Pseudotime"></div>
'''
            if 'pseudotime' in traj:
                pt = traj['pseudotime']
                html += f'''            <div class="success-box">
                {"분석 완료" if is_korean else "Analysis Complete"}: Root cluster = {pt.get('root_cluster', 'auto')}, Method = {pt.get('method', 'dpt')}
            </div>
'''
            html += '''        </div>
'''

        # Malignant UMAP
        if 'umap_malignant' in self.figures:
            html += f'''
        <div class="card">
            <h2>🔍 {"악성 세포 분포" if is_korean else "Malignant Cell Distribution"}</h2>
            <div class="figure-container"><img src="figures/umap_malignant.png" alt="Malignant Cells"></div>
        </div>
'''

        # Disclaimer and Footer
        html += f'''
        <div class="warning-box">
            {labels["disclaimer"]}
        </div>

        <footer>
            <p>Generated by BioInsight AI Single-Cell Pipeline v2.0</p>
            <p>🧬 6-Agent Architecture: QC → Cluster → Pathway → Trajectory → CNV/ML → Report</p>
        </footer>
    </div>
</body>
</html>
'''

        return html

    def _save_outputs(self):
        """Save output files."""
        self.logger.info("Saving outputs...")

        # Save report data JSON
        data_json = self.output_dir / "report_data.json"
        with open(data_json, 'w') as f:
            # Convert DataFrames to dicts
            json_data = {}
            for k, v in self.report_data.items():
                if isinstance(v, pd.DataFrame):
                    json_data[k] = v.to_dict('records')
                else:
                    json_data[k] = v
            json.dump(json_data, f, indent=2, default=str)

        self.logger.info(f"  Saved: {data_json}")

        # Copy h5ad forward
        for name in ["adata_cnv.h5ad", "adata_trajectory.h5ad", "adata_clustered.h5ad"]:
            src = self.input_dir / name
            if src.exists():
                import shutil
                shutil.copy(src, self.output_dir / "adata_final.h5ad")
                break

        self.logger.info("Report generation complete!")

    def validate_outputs(self) -> bool:
        """Validate output files were generated correctly."""
        required_files = [
            self.output_dir / "singlecell_report.html"
        ]
        for f in required_files:
            if not f.exists():
                self.logger.error(f"Required output missing: {f}")
                return False
        return True
