"""
Base Report Generator for RNA-seq Pipeline

Provides common functionality for both Bulk and Single-cell reports:
- Unified styling and CSS
- Common section templates
- Data loading utilities
- LLM interpretation helpers
"""

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import pandas as pd


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_title: str = "RNA-seq Analysis Report"
    author: str = "BioInsight AI Pipeline"
    data_type: Literal["bulk", "singlecell", "multiomic"] = "bulk"
    cancer_type: str = "unknown"
    tissue_type: str = "unknown"
    language: Literal["ko", "en"] = "ko"
    embed_figures: bool = True
    include_methods: bool = True
    include_downloads: bool = True
    max_table_rows: int = 100
    theme: Literal["light", "dark", "auto"] = "light"


@dataclass
class ReportData:
    """Container for all report data."""
    # Common data
    figures: Dict[str, str] = field(default_factory=dict)
    interactive_figures: Dict[str, str] = field(default_factory=dict)

    # Bulk RNA-seq specific
    deg_significant: Optional[pd.DataFrame] = None
    deg_all: Optional[pd.DataFrame] = None
    hub_genes: Optional[pd.DataFrame] = None
    pathway_summary: Optional[pd.DataFrame] = None
    integrated_gene_table: Optional[pd.DataFrame] = None
    db_matched_genes: Optional[pd.DataFrame] = None
    interpretation_report: Optional[Dict] = None

    # Single-cell specific
    cluster_markers: Optional[pd.DataFrame] = None
    cell_composition: Optional[pd.DataFrame] = None
    umap_coordinates: Optional[pd.DataFrame] = None
    cell_metadata: Optional[pd.DataFrame] = None
    driver_genes: Optional[pd.DataFrame] = None
    cluster_pathways: Optional[pd.DataFrame] = None
    trajectory_pseudotime: Optional[pd.DataFrame] = None
    cell_interactions: Optional[pd.DataFrame] = None
    tme_composition: Optional[pd.DataFrame] = None
    ploidy_by_celltype: Optional[pd.DataFrame] = None
    grn_edges: Optional[pd.DataFrame] = None
    adata_summary: Optional[Dict] = None

    # Multi-omic specific
    integrated_drivers: Optional[pd.DataFrame] = None
    confirmed_drivers: Optional[pd.DataFrame] = None
    actionable_targets: Optional[pd.DataFrame] = None
    driver_mutations: Optional[pd.DataFrame] = None

    # LLM-generated content
    abstract_extended: Optional[Dict] = None
    visualization_interpretations: Optional[Dict] = None
    research_recommendations: Optional[Dict] = None
    recommended_papers: Optional[List[Dict]] = None
    cancer_prediction: Optional[Dict] = None

    # Driver analysis
    driver_known: Optional[List[Dict]] = None
    driver_novel: Optional[List[Dict]] = None
    driver_summary: Optional[Dict] = None

    # RAG + External API interpretation results
    rag_interpretations: Optional[Dict] = None  # Multi-source gene interpretations

    # Metadata
    meta_agents: Dict[str, Dict] = field(default_factory=dict)


class BaseReportGenerator(ABC):
    """Base class for report generation."""

    VERSION = "3.0.0"

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[ReportConfig] = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or ReportConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"report_{self.config.data_type}")
        self.logger.setLevel(logging.DEBUG)

        # Add handlers if not present
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def load_data(self) -> ReportData:
        """Load all available data from input directory."""
        data = ReportData()

        search_paths = [
            self.input_dir,
            self.input_dir / "agent1_deg",
            self.input_dir / "agent2_network",
            self.input_dir / "agent3_pathway",
            self.input_dir / "agent4_validation",
            self.input_dir / "agent5_visualization",
            self.input_dir / "accumulated",
        ]

        # CSV files mapping
        csv_mappings = {
            # Bulk RNA-seq
            "deg_significant.csv": "deg_significant",
            "deg_all_results.csv": "deg_all",
            "hub_genes.csv": "hub_genes",
            "pathway_summary.csv": "pathway_summary",
            "integrated_gene_table.csv": "integrated_gene_table",
            "db_matched_genes.csv": "db_matched_genes",
            # Single-cell
            "cluster_markers.csv": "cluster_markers",
            "cell_composition.csv": "cell_composition",
            "umap_coordinates.csv": "umap_coordinates",
            "cell_metadata.csv": "cell_metadata",
            "driver_genes.csv": "driver_genes",
            "cluster_pathways.csv": "cluster_pathways",
            "trajectory_pseudotime.csv": "trajectory_pseudotime",
            "cell_interactions.csv": "cell_interactions",
            "tme_composition.csv": "tme_composition",
            "ploidy_by_celltype.csv": "ploidy_by_celltype",
            "grn_edges.csv": "grn_edges",
            # Multi-omic
            "integrated_drivers.csv": "integrated_drivers",
            "confirmed_drivers.csv": "confirmed_drivers",
            "actionable_targets.csv": "actionable_targets",
            "driver_mutations.csv": "driver_mutations",
        }

        # Load CSVs
        for filename, attr_name in csv_mappings.items():
            for path in search_paths:
                filepath = path / filename
                if filepath.exists():
                    try:
                        df = pd.read_csv(filepath)
                        setattr(data, attr_name, df)
                        self.logger.info(f"Loaded {filename}: {len(df)} rows")
                    except Exception as e:
                        self.logger.warning(f"Error loading {filename}: {e}")
                    break

        # JSON files
        json_mappings = {
            "interpretation_report.json": "interpretation_report",
            "rag_interpretations.json": "rag_interpretations",  # Multi-source RAG + External API
            "abstract_extended.json": "abstract_extended",
            "visualization_interpretations.json": "visualization_interpretations",
            "research_recommendations.json": "research_recommendations",
            "recommended_papers.json": "recommended_papers",
            "cancer_prediction.json": "cancer_prediction",
        }

        for filename, attr_name in json_mappings.items():
            for path in search_paths:
                filepath = path / filename
                if filepath.exists():
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            setattr(data, attr_name, json.load(f))
                        self.logger.info(f"Loaded {filename}")
                    except Exception as e:
                        self.logger.warning(f"Error loading {filename}: {e}")
                    break

        # Also check parent directory for JSON files
        parent_dir = self.input_dir.parent if self.input_dir.name == 'accumulated' else self.input_dir
        for filename, attr_name in json_mappings.items():
            if getattr(data, attr_name) is None:
                filepath = parent_dir / filename
                if filepath.exists():
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            setattr(data, attr_name, json.load(f))
                        self.logger.info(f"Loaded {filename} from parent")
                    except Exception as e:
                        self.logger.warning(f"Error loading {filename}: {e}")

        # Load figures
        figures_dirs = [
            self.input_dir / "figures",
            self.input_dir / "agent5_visualization" / "figures",
            self.input_dir / "accumulated" / "figures",
        ]

        for figures_dir in figures_dirs:
            if figures_dir.exists():
                # PNG/JPG images
                for img_path in list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.jpg")):
                    if self.config.embed_figures:
                        with open(img_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                            ext = img_path.suffix[1:]
                            data.figures[img_path.stem] = f"data:image/{ext};base64,{img_data}"
                    else:
                        data.figures[img_path.stem] = str(img_path)
                    self.logger.info(f"Loaded figure: {img_path.name}")

                # Interactive HTML (Plotly)
                for html_path in figures_dir.glob("*.html"):
                    try:
                        with open(html_path, 'r', encoding='utf-8') as f:
                            data.interactive_figures[html_path.stem] = f.read()
                        self.logger.info(f"Loaded interactive: {html_path.name}")
                    except Exception as e:
                        self.logger.warning(f"Error loading {html_path.name}: {e}")
                break  # Use first found figures dir

        # Load agent metadata
        for i in range(1, 7):
            for path in search_paths:
                meta_path = path / f"meta_agent{i}.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            data.meta_agents[f"agent{i}"] = json.load(f)
                    except:
                        pass
                    break

        return data

    def generate_css(self) -> str:
        """Generate unified CSS for the report."""
        return '''
/* ============================================
   BioInsight RNA-seq Report - Unified Styles v3.0
   ============================================ */

:root {
    /* Color Palette */
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary: #64748b;
    --accent: #0891b2;
    --success: #059669;
    --warning: #d97706;
    --error: #dc2626;

    /* Background Colors */
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #f1f5f9;
    --bg-card: #ffffff;

    /* Text Colors */
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #94a3b8;

    /* Borders */
    --border-light: #e2e8f0;
    --border-medium: #cbd5e1;

    /* Shadows */
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1);

    /* Typography */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-serif: 'Crimson Pro', Georgia, serif;
    --font-mono: 'Fira Code', 'Monaco', monospace;

    /* Spacing */
    --section-gap: 3rem;
    --content-max-width: 1200px;
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    :root.theme-auto {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --border-light: #334155;
        --border-medium: #475569;
    }
}

.theme-dark {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --border-light: #334155;
    --border-medium: #475569;
}

/* Base Reset */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    scroll-behavior: smooth;
    font-size: 16px;
}

body {
    font-family: var(--font-sans);
    background: var(--bg-secondary);
    color: var(--text-primary);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-serif);
    font-weight: 600;
    line-height: 1.3;
    color: var(--text-primary);
}

h1 { font-size: 2.25rem; }
h2 { font-size: 1.75rem; margin-bottom: 1rem; }
h3 { font-size: 1.25rem; }
h4 { font-size: 1.1rem; }

p {
    margin-bottom: 1rem;
    color: var(--text-secondary);
}

a {
    color: var(--primary);
    text-decoration: none;
    transition: color 0.2s;
}

a:hover {
    color: var(--primary-dark);
    text-decoration: underline;
}

/* Cover Page */
.cover-page {
    min-height: 60vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    color: white;
}

.cover-page h1 {
    font-size: 2.5rem;
    color: white;
    margin-bottom: 1rem;
}

.cover-page .subtitle {
    font-size: 1.25rem;
    opacity: 0.9;
    margin-bottom: 2rem;
}

.cover-page .meta-info {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    justify-content: center;
}

.cover-page .meta-item {
    background: rgba(255,255,255,0.15);
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    backdrop-filter: blur(10px);
}

.cover-page .meta-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.8;
}

.cover-page .meta-value {
    font-size: 1.1rem;
    font-weight: 600;
}

/* Data Type Badge */
.data-type-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.data-type-badge.bulk {
    background: #dbeafe;
    color: #1d4ed8;
}

.data-type-badge.singlecell {
    background: #d1fae5;
    color: #059669;
}

.data-type-badge.multiomic {
    background: #fef3c7;
    color: #d97706;
}

/* Navigation */
.nav-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-light);
    box-shadow: var(--shadow-sm);
}

.nav-container {
    max-width: var(--content-max-width);
    margin: 0 auto;
    padding: 0 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
}

.nav-brand {
    font-family: var(--font-serif);
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--primary);
}

.nav-links {
    display: flex;
    gap: 0.25rem;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}

.nav-links::-webkit-scrollbar {
    display: none;
}

.nav-links a {
    padding: 0.5rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    border-radius: 6px;
    white-space: nowrap;
    transition: all 0.2s;
}

.nav-links a:hover {
    background: var(--bg-tertiary);
    color: var(--primary);
    text-decoration: none;
}

.nav-links a.active {
    background: var(--primary);
    color: white;
}

/* Main Content */
.main-content {
    max-width: var(--content-max-width);
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

/* Sections - Page-based Layout */
.section {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: var(--section-gap);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
    /* Page break for paper-like layout */
    page-break-after: always;
    break-after: page;
    min-height: calc(100vh - 80px);
    position: relative;
}

.section:last-of-type {
    page-break-after: avoid;
    break-after: avoid;
}

/* Page indicator */
.section::after {
    content: attr(data-page);
    position: absolute;
    bottom: 1rem;
    right: 1.5rem;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-light);
}

.section-icon {
    font-size: 1.5rem;
}

.section-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--primary);
    color: white;
    border-radius: 50%;
    font-size: 0.875rem;
    font-weight: 600;
}

.section-title {
    flex: 1;
    margin: 0;
}

.section-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.25rem;
}

/* Summary Cards Grid */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.summary-card {
    background: var(--bg-tertiary);
    padding: 1.25rem;
    border-radius: 10px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}

.summary-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.summary-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary);
    font-family: var(--font-mono);
}

.summary-card .label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 0.25rem;
}

.summary-card.highlight {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
}

.summary-card.highlight .value,
.summary-card.highlight .label {
    color: white;
}

/* Figure Grid */
.figure-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
}

.figure-panel {
    background: var(--bg-tertiary);
    border-radius: 10px;
    overflow: hidden;
}

.figure-title {
    padding: 0.75rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-light);
}

.figure-content {
    padding: 1rem;
}

.figure-content img {
    width: 100%;
    height: auto;
    border-radius: 6px;
}

.figure-caption {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.75rem;
    font-style: italic;
}

/* Interactive Figure Container */
.interactive-container {
    width: 100%;
    min-height: 500px;
    border-radius: 8px;
    overflow: hidden;
    background: var(--bg-primary);
}

.interactive-container iframe {
    width: 100%;
    height: 100%;
    border: none;
}

/* Tables */
.table-wrapper {
    overflow-x: auto;
    margin: 1rem 0;
    border-radius: 8px;
    border: 1px solid var(--border-light);
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

thead {
    background: var(--bg-tertiary);
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-light);
}

th {
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
}

tbody tr:hover {
    background: var(--bg-secondary);
}

tbody tr:last-child td {
    border-bottom: none;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-up {
    background: #fee2e2;
    color: #dc2626;
}

.badge-down {
    background: #dbeafe;
    color: #2563eb;
}

.badge-high {
    background: #d1fae5;
    color: #059669;
}

.badge-medium {
    background: #fef3c7;
    color: #d97706;
}

.badge-low {
    background: #f1f5f9;
    color: #64748b;
}

/* Confidence Indicator */
.confidence-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.confidence-track {
    flex: 1;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.confidence-fill.high { background: var(--success); }
.confidence-fill.medium { background: var(--warning); }
.confidence-fill.low { background: var(--error); }

/* Interpretation Cards */
.interpretation-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}

.interpretation-card {
    background: var(--bg-tertiary);
    border-radius: 10px;
    padding: 1.25rem;
    border-left: 4px solid var(--primary);
}

.interpretation-card.full-width {
    grid-column: 1 / -1;
}

.interpretation-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.interpretation-icon {
    font-size: 1.25rem;
}

.interpretation-title {
    font-weight: 600;
    color: var(--text-primary);
}

.interpretation-content {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.7;
}

/* Warning/Note Box */
.note-box {
    padding: 1rem 1.25rem;
    border-radius: 8px;
    margin: 1rem 0;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}

.note-box.warning {
    background: #fef3c7;
    border: 1px solid #fbbf24;
    color: #92400e;
}

.note-box.info {
    background: #dbeafe;
    border: 1px solid #60a5fa;
    color: #1e40af;
}

.note-box.success {
    background: #d1fae5;
    border: 1px solid #34d399;
    color: #065f46;
}

.note-box-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
}

.note-box-content {
    flex: 1;
    font-size: 0.9rem;
}

/* Gene Status Cards */
.gene-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
}

.gene-card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 10px;
    padding: 1rem;
    transition: box-shadow 0.2s;
}

.gene-card:hover {
    box-shadow: var(--shadow-md);
}

.gene-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.gene-symbol {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--primary);
}

.gene-symbol-large {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
}

.gene-card-body {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* RAG Interpretations Panel */
.rag-interpretations-panel {
    margin-top: 2rem;
    padding: 1.5rem;
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border-light);
}

.rag-interpretations-panel h3 {
    margin-bottom: 1rem;
    color: var(--primary);
}

.interpretation-summary {
    padding: 1rem;
    background: var(--bg-tertiary);
    border-radius: 8px;
    margin-bottom: 1.5rem;
}

.gene-cards-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.gene-interpretation-card {
    background: white;
    border-radius: 10px;
    border: 1px solid var(--border-light);
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.gene-interpretation-card .card-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-bottom: 1px solid var(--border-light);
    flex-wrap: wrap;
}

.direction-badge {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 600;
}

.direction-badge.up {
    background: #fef2f2;
    color: #dc2626;
}

.direction-badge.down {
    background: #eff6ff;
    color: #2563eb;
}

.badge {
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 12px;
    font-weight: 500;
}

.badge-success {
    background: #dcfce7;
    color: #16a34a;
}

.badge-warning {
    background: #fef3c7;
    color: #d97706;
}

.badge-danger {
    background: #fee2e2;
    color: #dc2626;
}

.gene-interpretation-card .card-body {
    padding: 1.25rem;
}

.interpretation-text {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

.api-annotations {
    margin: 1rem 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: flex-start;
}

.api-tag {
    display: inline-block;
    font-size: 0.8rem;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-weight: 500;
}

.api-tag.oncokb {
    background: #fef3c7;
    color: #92400e;
}

.api-tag.oncogene {
    background: #fee2e2;
    color: #991b1b;
}

.api-tag.tsg {
    background: #dbeafe;
    color: #1e40af;
}

.api-tag.actionable {
    background: #dcfce7;
    color: #166534;
}

.protein-func, .pathways {
    width: 100%;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

.citations {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px dashed var(--border-light);
}

.citation-list {
    margin: 0.5rem 0 0 0;
    padding-left: 1.25rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.citation-list li {
    margin-bottom: 0.3rem;
}

.sources-used {
    margin-top: 0.75rem;
    font-size: 0.75rem;
    color: var(--text-muted);
}

.gene-stat {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem 0;
    border-bottom: 1px dashed var(--border-light);
}

.gene-stat:last-child {
    border-bottom: none;
}

/* Footer */
.report-footer {
    background: var(--bg-tertiary);
    padding: 2rem;
    text-align: center;
    border-top: 1px solid var(--border-light);
    margin-top: 3rem;
}

.footer-disclaimer {
    max-width: 800px;
    margin: 0 auto 1rem;
    padding: 1rem;
    background: #fef3c7;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #92400e;
}

.footer-credit {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* Pagination Controls */
.pagination-controls {
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 1rem;
    background: var(--bg-card);
    padding: 0.75rem 1.5rem;
    border-radius: 30px;
    box-shadow: var(--shadow-lg);
    z-index: 1000;
    border: 1px solid var(--border-light);
}

.pagination-controls button {
    background: var(--primary);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.pagination-controls button:hover {
    background: var(--primary-dark);
    transform: translateY(-2px);
}

.pagination-controls button:disabled {
    background: var(--text-muted);
    cursor: not-allowed;
    transform: none;
}

.page-indicator {
    display: flex;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--text-secondary);
    gap: 0.5rem;
}

.page-indicator .current {
    font-weight: 700;
    color: var(--primary);
}

/* Section Page Numbers */
.section-page-number {
    position: absolute;
    bottom: 1rem;
    right: 1.5rem;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    background: var(--bg-tertiary);
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
}

/* Page View Mode */
body.page-view-mode .main-content {
    scroll-snap-type: y mandatory;
    overflow-y: scroll;
    height: calc(100vh - 60px);
}

body.page-view-mode .section {
    scroll-snap-align: start;
    scroll-snap-stop: always;
    min-height: calc(100vh - 80px);
    margin-bottom: 0;
    overflow-y: auto;
    max-height: calc(100vh - 80px);
}

/* View Mode Toggle */
.view-mode-toggle {
    display: flex;
    gap: 0.25rem;
    background: var(--bg-tertiary);
    padding: 0.25rem;
    border-radius: 8px;
}

.view-mode-toggle button {
    background: transparent;
    border: none;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.8rem;
    color: var(--text-secondary);
    transition: all 0.2s;
}

.view-mode-toggle button.active {
    background: var(--bg-card);
    color: var(--primary);
    box-shadow: var(--shadow-sm);
}

.view-mode-toggle button:hover:not(.active) {
    background: var(--bg-secondary);
}

/* Print Styles */
@media print {
    .nav-bar, .pagination-controls, .view-mode-toggle {
        display: none !important;
    }

    .section {
        break-inside: avoid;
        break-after: page;
        box-shadow: none;
        border: 1px solid #ddd;
        min-height: auto;
        max-height: none;
    }

    body.page-view-mode .main-content {
        overflow: visible;
        height: auto;
        scroll-snap-type: none;
    }

    .cover-page {
        min-height: auto;
        padding: 2rem;
    }
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .cover-page h1 {
        font-size: 1.75rem;
    }

    .cover-page .meta-info {
        flex-direction: column;
        gap: 0.75rem;
    }

    .nav-container {
        flex-direction: column;
        height: auto;
        padding: 0.75rem;
        gap: 0.5rem;
    }

    .section {
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .figure-grid {
        grid-template-columns: 1fr;
    }

    .summary-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
'''

    def generate_javascript(self, data: ReportData) -> str:
        """Generate JavaScript for interactivity."""
        return '''
<script>
// Page-based Navigation System
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('.section[id]');
    const navLinks = document.querySelectorAll('.nav-links a');
    const mainContent = document.querySelector('.main-content');
    let currentPage = 0;
    let isPageViewMode = true;

    // Initialize page numbers on sections
    sections.forEach((section, index) => {
        const pageNum = document.createElement('span');
        pageNum.className = 'section-page-number';
        pageNum.textContent = `Page ${index + 1} / ${sections.length}`;
        section.appendChild(pageNum);
        section.setAttribute('data-page-index', index);
    });

    // Set initial page view mode
    document.body.classList.add('page-view-mode');

    // Create pagination controls
    const paginationHTML = `
        <div class="pagination-controls">
            <button id="prevPage" onclick="navigatePage(-1)">
                <span>◀</span> 이전
            </button>
            <div class="page-indicator">
                <span class="current" id="currentPageNum">1</span>
                <span>/</span>
                <span id="totalPages">${sections.length}</span>
            </div>
            <button id="nextPage" onclick="navigatePage(1)">
                다음 <span>▶</span>
            </button>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', paginationHTML);

    // Add view mode toggle to nav
    const navContainer = document.querySelector('.nav-container');
    if (navContainer) {
        const toggleHTML = `
            <div class="view-mode-toggle">
                <button class="active" id="pageViewBtn" onclick="setViewMode('page')">📄 페이지</button>
                <button id="scrollViewBtn" onclick="setViewMode('scroll')">📜 스크롤</button>
            </div>
        `;
        navContainer.insertAdjacentHTML('beforeend', toggleHTML);
    }

    // Navigate to specific page
    window.navigateToPage = function(index) {
        if (index < 0 || index >= sections.length) return;
        currentPage = index;

        if (isPageViewMode) {
            sections[currentPage].scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            sections[currentPage].scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        updatePaginationUI();
        updateActiveNavLink();
    };

    // Navigate by delta
    window.navigatePage = function(delta) {
        const newPage = currentPage + delta;
        if (newPage >= 0 && newPage < sections.length) {
            navigateToPage(newPage);
        }
    };

    // Update pagination UI
    function updatePaginationUI() {
        document.getElementById('currentPageNum').textContent = currentPage + 1;
        document.getElementById('prevPage').disabled = currentPage === 0;
        document.getElementById('nextPage').disabled = currentPage === sections.length - 1;
    }

    // Update active nav link
    function updateActiveNavLink() {
        navLinks.forEach(link => link.classList.remove('active'));
        const currentSectionId = sections[currentPage]?.getAttribute('id');
        navLinks.forEach(link => {
            if (link.getAttribute('href') === '#' + currentSectionId) {
                link.classList.add('active');
            }
        });
    }

    // Set view mode
    window.setViewMode = function(mode) {
        isPageViewMode = mode === 'page';
        document.body.classList.toggle('page-view-mode', isPageViewMode);
        document.querySelector('.pagination-controls').style.display = isPageViewMode ? 'flex' : 'none';

        document.getElementById('pageViewBtn').classList.toggle('active', isPageViewMode);
        document.getElementById('scrollViewBtn').classList.toggle('active', !isPageViewMode);
    };

    // Detect scroll position in page view mode
    if (mainContent) {
        mainContent.addEventListener('scroll', function() {
            if (!isPageViewMode) {
                // In scroll mode, update based on scroll position
                const scrollPos = mainContent.scrollTop + 100;
                sections.forEach((section, index) => {
                    if (scrollPos >= section.offsetTop && scrollPos < section.offsetTop + section.offsetHeight) {
                        if (currentPage !== index) {
                            currentPage = index;
                            updatePaginationUI();
                            updateActiveNavLink();
                        }
                    }
                });
            }
        });

        // Detect scroll snap end in page view mode
        let scrollTimeout;
        mainContent.addEventListener('scroll', function() {
            if (isPageViewMode) {
                clearTimeout(scrollTimeout);
                scrollTimeout = setTimeout(function() {
                    // Find the section that is most visible
                    const viewportCenter = mainContent.scrollTop + mainContent.clientHeight / 2;
                    sections.forEach((section, index) => {
                        const sectionTop = section.offsetTop;
                        const sectionBottom = sectionTop + section.offsetHeight;
                        if (viewportCenter >= sectionTop && viewportCenter < sectionBottom) {
                            if (currentPage !== index) {
                                currentPage = index;
                                updatePaginationUI();
                                updateActiveNavLink();
                            }
                        }
                    });
                }, 100);
            }
        });
    }

    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'PageDown') {
            if (isPageViewMode) {
                e.preventDefault();
                navigatePage(1);
            }
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp' || e.key === 'PageUp') {
            if (isPageViewMode) {
                e.preventDefault();
                navigatePage(-1);
            }
        } else if (e.key === 'Home') {
            if (isPageViewMode) {
                e.preventDefault();
                navigateToPage(0);
            }
        } else if (e.key === 'End') {
            if (isPageViewMode) {
                e.preventDefault();
                navigateToPage(sections.length - 1);
            }
        }
    });

    // Nav links click handler
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').slice(1);
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                const targetIndex = Array.from(sections).findIndex(s => s.id === targetId);
                if (targetIndex !== -1) {
                    navigateToPage(targetIndex);
                }
            }
        });
    });

    // Initial UI update
    updatePaginationUI();
    updateActiveNavLink();
});

// Table sorting (basic implementation)
function sortTable(table, column, asc = true) {
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll('tr'));

    rows.sort((a, b) => {
        const aVal = a.cells[column].textContent.trim();
        const bVal = b.cells[column].textContent.trim();

        const aNum = parseFloat(aVal);
        const bNum = parseFloat(bVal);

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return asc ? aNum - bNum : bNum - aNum;
        }

        return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });

    rows.forEach(row => tbody.appendChild(row));
}

// Add click handlers to table headers
document.querySelectorAll('th[data-sortable]').forEach(th => {
    let asc = true;
    th.style.cursor = 'pointer';
    th.addEventListener('click', function() {
        const table = this.closest('table');
        const column = Array.from(this.parentNode.cells).indexOf(this);
        sortTable(table, column, asc);
        asc = !asc;
    });
});
</script>
'''

    def _get_llm_client(self):
        """Get available LLM client (Claude preferred for RAG-based interpretation)."""
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        # Prefer Claude for accurate, hallucination-free responses
        if anthropic_key:
            try:
                import anthropic
                return ("anthropic", anthropic.Anthropic())
            except ImportError:
                pass

        # Fallback to OpenAI
        if openai_key:
            try:
                from openai import OpenAI
                return ("openai", OpenAI())
            except ImportError:
                pass

        return (None, None)

    def _get_rag_context(self, cancer_type: str, genes: List[str] = None) -> str:
        """Get RAG context from VectorDB for grounded interpretation."""
        try:
            from rnaseq_pipeline.rag.gene_interpreter import GeneInterpreter
            interpreter = GeneInterpreter()

            context_parts = []

            # Get cancer-specific context
            if cancer_type and cancer_type != "unknown":
                cancer_query = f"{cancer_type} RNA-seq transcriptomics gene expression"
                cancer_results = interpreter.search_papers(cancer_query, top_k=3)
                if cancer_results:
                    context_parts.append(f"## {cancer_type} 관련 문헌 근거:")
                    for r in cancer_results[:3]:
                        title = r.get('title', 'Unknown')
                        pmid = r.get('pmid', '')
                        abstract = r.get('abstract', r.get('content', ''))[:500]
                        context_parts.append(f"- {title} [PMID: {pmid}]\n  {abstract}...")

            # Get gene-specific context
            if genes and len(genes) > 0:
                for gene in genes[:5]:  # Top 5 genes
                    gene_results = interpreter.search_papers(f"{gene} cancer function role", top_k=2)
                    if gene_results:
                        context_parts.append(f"\n## {gene} 관련 근거:")
                        for r in gene_results[:2]:
                            title = r.get('title', 'Unknown')
                            pmid = r.get('pmid', '')
                            context_parts.append(f"- {title} [PMID: {pmid}]")

            return "\n".join(context_parts) if context_parts else ""

        except Exception as e:
            self.logger.warning(f"RAG context retrieval failed: {e}")
            return ""

    def generate_llm_content(self, prompt: str, max_tokens: int = 2000,
                             use_rag: bool = True, cancer_type: str = None,
                             key_genes: List[str] = None) -> Optional[str]:
        """Generate content using Claude with RAG-based grounding.

        Args:
            prompt: The main prompt
            max_tokens: Maximum tokens for response
            use_rag: Whether to include RAG context for grounding
            cancer_type: Cancer type for context retrieval
            key_genes: Key genes for context retrieval
        """
        provider, client = self._get_llm_client()

        # Get RAG context for grounding (anti-hallucination)
        rag_context = ""
        if use_rag and (cancer_type or key_genes):
            rag_context = self._get_rag_context(
                cancer_type or self.config.cancer_type,
                key_genes
            )

        # Build system prompt for accurate, grounded responses
        system_prompt = """당신은 RNA-seq 분석 결과를 해석하는 전문 바이오인포매틱스 연구자입니다.

중요한 지침:
1. 반드시 제공된 데이터와 문헌 근거만을 기반으로 해석하세요.
2. 확실하지 않은 내용은 "~일 가능성이 있다", "추가 검증이 필요하다"로 표현하세요.
3. 가능한 경우 PMID 인용을 포함하세요.
4. 절대로 데이터에 없는 정보를 추측하지 마세요.
5. 임상적 결론이나 진단적 판단은 피하세요.
6. 모든 해석은 한국어로 작성하세요."""

        # Combine RAG context with prompt
        full_prompt = prompt
        if rag_context:
            full_prompt = f"""다음은 VectorDB에서 검색된 관련 문헌 정보입니다. 이 정보를 참고하여 해석하세요:

{rag_context}

---

{prompt}"""

        if provider == "anthropic":
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.content[0].text
            except Exception as e:
                self.logger.warning(f"Claude API error: {e}")

        elif provider == "openai":
            try:
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
                self.logger.warning(f"OpenAI error: {e}")

        return None

    def format_number(self, n: float, decimals: int = 2) -> str:
        """Format number with commas and decimals."""
        if pd.isna(n):
            return "N/A"
        if isinstance(n, int) or n == int(n):
            return f"{int(n):,}"
        return f"{n:,.{decimals}f}"

    def get_direction_badge(self, log2fc: float) -> str:
        """Get HTML badge for expression direction."""
        if log2fc > 0:
            return '<span class="badge badge-up">↑ Up</span>'
        else:
            return '<span class="badge badge-down">↓ Down</span>'

    def get_confidence_badge(self, score: float) -> str:
        """Get HTML badge for confidence level."""
        if score >= 0.8:
            return '<span class="badge badge-high">High</span>'
        elif score >= 0.5:
            return '<span class="badge badge-medium">Medium</span>'
        else:
            return '<span class="badge badge-low">Low</span>'

    @abstractmethod
    def generate(self) -> Path:
        """Generate the report. Returns path to generated HTML."""
        pass
