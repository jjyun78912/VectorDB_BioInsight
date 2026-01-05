#!/usr/bin/env python3
"""
HTML Report Generator for RNA-seq Analysis

Generates interactive HTML reports with:
- Summary statistics
- Interactive plots (Plotly)
- Gene tables
- Disease associations
- Downloadable data

Author: BioInsight AI
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for RNA-seq analysis"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        deg_results: pd.DataFrame,
        hub_genes: pd.DataFrame,
        gene_cards: Optional[List] = None,
        pathway_results: Optional[pd.DataFrame] = None,
        qc_report: Optional[Dict] = None,
        figures_dir: Optional[str] = None,
        title: str = "RNA-seq Analysis Report",
        sample_info: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive HTML report

        Args:
            deg_results: DEG analysis results
            hub_genes: Hub gene analysis results
            gene_cards: List of GeneCard objects
            pathway_results: Pathway enrichment results
            qc_report: QC report dictionary
            figures_dir: Directory containing figure images
            title: Report title
            sample_info: Sample/experiment information

        Returns:
            Path to generated HTML file
        """
        logger.info("Generating HTML report...")

        # Prepare data
        sig_degs = deg_results[
            (deg_results['padj'] < 0.05) &
            (abs(deg_results['log2FoldChange']) > 1)
        ]

        n_up = len(sig_degs[sig_degs['log2FoldChange'] > 0])
        n_down = len(sig_degs[sig_degs['log2FoldChange'] < 0])

        # Build HTML sections
        html_content = self._build_html(
            title=title,
            summary_stats={
                'total_genes': len(deg_results),
                'significant_degs': len(sig_degs),
                'upregulated': n_up,
                'downregulated': n_down,
                'hub_genes': len(hub_genes)
            },
            deg_results=deg_results,
            hub_genes=hub_genes,
            gene_cards=gene_cards,
            pathway_results=pathway_results,
            qc_report=qc_report,
            figures_dir=figures_dir,
            sample_info=sample_info
        )

        # Save report
        report_path = self.output_dir / 'analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✓ HTML report saved: {report_path}")

        return str(report_path)

    def _build_html(
        self,
        title: str,
        summary_stats: Dict,
        deg_results: pd.DataFrame,
        hub_genes: pd.DataFrame,
        gene_cards: Optional[List],
        pathway_results: Optional[pd.DataFrame],
        qc_report: Optional[Dict],
        figures_dir: Optional[str],
        sample_info: Optional[Dict]
    ) -> str:
        """Build complete HTML document"""

        # CSS styles
        css = self._get_css()

        # Sections
        header = self._build_header(title)
        summary = self._build_summary_section(summary_stats, sample_info)
        deg_section = self._build_deg_section(deg_results)
        hub_section = self._build_hub_section(hub_genes)

        gene_card_section = ""
        if gene_cards:
            gene_card_section = self._build_gene_card_section(gene_cards)

        pathway_section = ""
        if pathway_results is not None and len(pathway_results) > 0:
            pathway_section = self._build_pathway_section(pathway_results)

        qc_section = ""
        if qc_report:
            qc_section = self._build_qc_section(qc_report)

        figures_section = ""
        if figures_dir:
            figures_section = self._build_figures_section(figures_dir)

        # Build complete HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>{css}</style>
</head>
<body>
    {header}

    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <nav class="col-md-2 d-none d-md-block sidebar">
                <div class="sidebar-sticky">
                    <h6 class="sidebar-heading">Navigation</h6>
                    <ul class="nav flex-column">
                        <li class="nav-item"><a class="nav-link" href="#summary"><i class="fas fa-chart-pie"></i> Summary</a></li>
                        <li class="nav-item"><a class="nav-link" href="#deg"><i class="fas fa-dna"></i> DEG Analysis</a></li>
                        <li class="nav-item"><a class="nav-link" href="#hub"><i class="fas fa-project-diagram"></i> Hub Genes</a></li>
                        {"<li class='nav-item'><a class='nav-link' href='#genecards'><i class='fas fa-id-card'></i> Gene Cards</a></li>" if gene_cards else ""}
                        {"<li class='nav-item'><a class='nav-link' href='#pathways'><i class='fas fa-route'></i> Pathways</a></li>" if pathway_results is not None else ""}
                        {"<li class='nav-item'><a class='nav-link' href='#qc'><i class='fas fa-clipboard-check'></i> QC Report</a></li>" if qc_report else ""}
                        {"<li class='nav-item'><a class='nav-link' href='#figures'><i class='fas fa-image'></i> Figures</a></li>" if figures_dir else ""}
                    </ul>
                </div>
            </nav>

            <!-- Main content -->
            <main class="col-md-10 ml-sm-auto px-4">
                {summary}
                {deg_section}
                {hub_section}
                {gene_card_section}
                {pathway_section}
                {qc_section}
                {figures_section}

                <!-- Footer -->
                <footer class="footer mt-5 py-3">
                    <div class="text-center text-muted">
                        <p>Generated by BioInsight AI RNA-seq Pipeline</p>
                        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </footer>
            </main>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize interactive plots
        document.addEventListener('DOMContentLoaded', function() {{
            // Add table sorting
            const tables = document.querySelectorAll('.sortable');
            tables.forEach(table => {{
                const headers = table.querySelectorAll('th');
                headers.forEach((header, index) => {{
                    header.style.cursor = 'pointer';
                    header.addEventListener('click', () => sortTable(table, index));
                }});
            }});
        }});

        function sortTable(table, column) {{
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            rows.sort((a, b) => {{
                const aVal = a.cells[column].textContent;
                const bVal = b.cells[column].textContent;
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return bNum - aNum;
                }}
                return aVal.localeCompare(bVal);
            }});

            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</body>
</html>"""

        return html

    def _get_css(self) -> str:
        """Get CSS styles for the report"""
        return """
        :root {
            --primary-color: #3498db;
            --success-color: #27ae60;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --dark-color: #2c3e50;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }

        .header {
            background: linear-gradient(135deg, var(--dark-color), var(--primary-color));
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }

        .sidebar {
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 100;
            padding: 80px 0 0;
            background-color: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }

        .sidebar-heading {
            font-size: 0.75rem;
            text-transform: uppercase;
            padding: 1rem;
            color: #6c757d;
        }

        .sidebar .nav-link {
            font-weight: 500;
            color: #333;
            padding: 0.5rem 1rem;
        }

        .sidebar .nav-link:hover {
            color: var(--primary-color);
            background-color: #e9ecef;
        }

        .sidebar .nav-link i {
            margin-right: 0.5rem;
            width: 20px;
        }

        main {
            padding-top: 80px;
        }

        .card {
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: none;
            margin-bottom: 1.5rem;
        }

        .card-header {
            background-color: white;
            border-bottom: 2px solid var(--primary-color);
            font-weight: 600;
        }

        .stat-card {
            text-align: center;
            padding: 1.5rem;
        }

        .stat-card .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
        }

        .stat-card .stat-label {
            color: #6c757d;
            font-size: 0.9rem;
        }

        .stat-card.up {
            border-left: 4px solid var(--danger-color);
        }

        .stat-card.up .stat-value {
            color: var(--danger-color);
        }

        .stat-card.down {
            border-left: 4px solid var(--primary-color);
        }

        .stat-card.down .stat-value {
            color: var(--primary-color);
        }

        .table-responsive {
            max-height: 500px;
            overflow-y: auto;
        }

        table.sortable th {
            cursor: pointer;
        }

        table.sortable th:hover {
            background-color: #e9ecef;
        }

        .badge-up {
            background-color: var(--danger-color);
        }

        .badge-down {
            background-color: var(--primary-color);
        }

        .gene-card {
            border-left: 4px solid var(--primary-color);
            margin-bottom: 1rem;
        }

        .gene-card.upregulated {
            border-left-color: var(--danger-color);
        }

        .gene-card.downregulated {
            border-left-color: var(--primary-color);
        }

        .figure-container {
            margin-bottom: 2rem;
        }

        .figure-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
        }

        .section-title {
            color: var(--dark-color);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .progress-bar-up {
            background-color: var(--danger-color);
        }

        .progress-bar-down {
            background-color: var(--primary-color);
        }

        .therapeutic-badge {
            background-color: var(--success-color);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
            display: inline-block;
        }

        .disease-tag {
            background-color: var(--warning-color);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            margin-right: 0.25rem;
        }

        @media print {
            .sidebar {
                display: none;
            }
            main {
                padding-top: 0;
            }
        }
        """

    def _build_header(self, title: str) -> str:
        """Build header section"""
        return f"""
        <header class="header">
            <div class="container">
                <h1><i class="fas fa-dna"></i> {title}</h1>
                <p class="lead mb-0">Comprehensive RNA-seq Differential Expression Analysis</p>
            </div>
        </header>
        """

    def _build_summary_section(self, stats: Dict, sample_info: Optional[Dict]) -> str:
        """Build summary statistics section"""
        return f"""
        <section id="summary" class="mb-5">
            <h2 class="section-title"><i class="fas fa-chart-pie"></i> Analysis Summary</h2>

            <div class="row">
                <div class="col-md-3">
                    <div class="card stat-card">
                        <div class="stat-value">{stats['total_genes']:,}</div>
                        <div class="stat-label">Total Genes Analyzed</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stat-card">
                        <div class="stat-value">{stats['significant_degs']}</div>
                        <div class="stat-label">Significant DEGs</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stat-card up">
                        <div class="stat-value">{stats['upregulated']}</div>
                        <div class="stat-label"><i class="fas fa-arrow-up"></i> Upregulated</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stat-card down">
                        <div class="stat-value">{stats['downregulated']}</div>
                        <div class="stat-label"><i class="fas fa-arrow-down"></i> Downregulated</div>
                    </div>
                </div>
            </div>

            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            <h5>DEG Distribution</h5>
                            <div class="progress" style="height: 30px;">
                                <div class="progress-bar progress-bar-up" role="progressbar"
                                     style="width: {stats['upregulated']/max(stats['significant_degs'],1)*100:.1f}%"
                                     aria-valuenow="{stats['upregulated']}" aria-valuemin="0" aria-valuemax="{stats['significant_degs']}">
                                    Up ({stats['upregulated']})
                                </div>
                                <div class="progress-bar progress-bar-down" role="progressbar"
                                     style="width: {stats['downregulated']/max(stats['significant_degs'],1)*100:.1f}%"
                                     aria-valuenow="{stats['downregulated']}" aria-valuemin="0" aria-valuemax="{stats['significant_degs']}">
                                    Down ({stats['downregulated']})
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _build_deg_section(self, deg_results: pd.DataFrame) -> str:
        """Build DEG results section"""
        sig = deg_results[
            (deg_results['padj'] < 0.05) &
            (abs(deg_results['log2FoldChange']) > 1)
        ].head(50)

        rows = ""
        for _, row in sig.iterrows():
            direction = "up" if row['log2FoldChange'] > 0 else "down"
            badge_class = "badge-up" if direction == "up" else "badge-down"
            arrow = "↑" if direction == "up" else "↓"

            rows += f"""
            <tr>
                <td><strong>{row['gene']}</strong></td>
                <td><span class="badge {badge_class}">{arrow} {row['log2FoldChange']:.2f}</span></td>
                <td>{row['padj']:.2e}</td>
                <td>{row.get('baseMean', 0):.1f}</td>
            </tr>
            """

        return f"""
        <section id="deg" class="mb-5">
            <h2 class="section-title"><i class="fas fa-dna"></i> Differential Expression Results</h2>

            <div class="card">
                <div class="card-header">
                    <i class="fas fa-table"></i> Top 50 Significant DEGs
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover sortable">
                            <thead class="table-dark">
                                <tr>
                                    <th>Gene</th>
                                    <th>log₂ Fold Change</th>
                                    <th>Adjusted P-value</th>
                                    <th>Base Mean</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
        """

    def _build_hub_section(self, hub_genes: pd.DataFrame) -> str:
        """Build hub genes section"""
        top_hubs = hub_genes.head(20)

        rows = ""
        for _, row in top_hubs.iterrows():
            direction = "up" if row['log2FoldChange'] > 0 else "down"
            badge_class = "badge-up" if direction == "up" else "badge-down"

            rows += f"""
            <tr>
                <td><strong>{row['gene']}</strong></td>
                <td>{row['composite_score']:.4f}</td>
                <td>{row['degree']:.4f}</td>
                <td>{row['betweenness']:.4f}</td>
                <td><span class="badge {badge_class}">{row['log2FoldChange']:.2f}</span></td>
            </tr>
            """

        return f"""
        <section id="hub" class="mb-5">
            <h2 class="section-title"><i class="fas fa-project-diagram"></i> Hub Gene Analysis</h2>

            <div class="card">
                <div class="card-header">
                    <i class="fas fa-star"></i> Top 20 Hub Genes
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover sortable">
                            <thead class="table-dark">
                                <tr>
                                    <th>Gene</th>
                                    <th>Composite Score</th>
                                    <th>Degree Centrality</th>
                                    <th>Betweenness</th>
                                    <th>log₂FC</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
        """

    def _build_gene_card_section(self, gene_cards: List) -> str:
        """Build gene cards section"""
        cards_html = ""

        for card in gene_cards[:15]:
            regulation_class = "upregulated" if card.regulation == "Upregulated" else "downregulated"
            therapeutics = "".join([
                f'<span class="therapeutic-badge">{t}</span>'
                for t in card.therapeutics[:5]
            ]) if card.therapeutics else '<span class="text-muted">None known</span>'

            diseases = "".join([
                f'<span class="disease-tag">{d.disease_name}</span>'
                for d in card.diseases[:3]
            ]) if card.diseases else '<span class="text-muted">None found</span>'

            cards_html += f"""
            <div class="col-md-4">
                <div class="card gene-card {regulation_class}">
                    <div class="card-body">
                        <h5 class="card-title">{card.gene_symbol}</h5>
                        <p class="card-text">
                            <strong>{card.regulation}</strong> ({card.fold_change:.1f}x)<br>
                            <small class="text-muted">padj: {card.adjusted_p_value:.2e}</small>
                        </p>
                        <p class="mb-1"><strong>Diseases:</strong></p>
                        <p>{diseases}</p>
                        <p class="mb-1"><strong>Therapeutics:</strong></p>
                        <p>{therapeutics}</p>
                    </div>
                </div>
            </div>
            """

        return f"""
        <section id="genecards" class="mb-5">
            <h2 class="section-title"><i class="fas fa-id-card"></i> Gene Status Cards</h2>
            <div class="row">
                {cards_html}
            </div>
        </section>
        """

    def _build_pathway_section(self, pathway_results: pd.DataFrame) -> str:
        """Build pathway enrichment section"""
        top_pathways = pathway_results.head(20)

        rows = ""
        for _, row in top_pathways.iterrows():
            rows += f"""
            <tr>
                <td>{row['Term']}</td>
                <td>{row['Adjusted P-value']:.2e}</td>
                <td>{row.get('Overlap', 'N/A')}</td>
            </tr>
            """

        return f"""
        <section id="pathways" class="mb-5">
            <h2 class="section-title"><i class="fas fa-route"></i> Pathway Enrichment</h2>

            <div class="card">
                <div class="card-header">
                    <i class="fas fa-sitemap"></i> Top Enriched Pathways (GO/KEGG)
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover sortable">
                            <thead class="table-dark">
                                <tr>
                                    <th>Pathway Term</th>
                                    <th>Adjusted P-value</th>
                                    <th>Overlap</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
        """

    def _build_qc_section(self, qc_report: Dict) -> str:
        """Build QC report section"""
        return f"""
        <section id="qc" class="mb-5">
            <h2 class="section-title"><i class="fas fa-clipboard-check"></i> Quality Control</h2>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Sample QC</div>
                        <div class="card-body">
                            <ul class="list-group list-group-flush">
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Total Samples</span>
                                    <strong>{qc_report.get('total_samples', 'N/A')}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Samples Passed</span>
                                    <strong class="text-success">{qc_report.get('samples_passed', 'N/A')}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Samples Failed</span>
                                    <strong class="text-danger">{qc_report.get('samples_failed', 'N/A')}</strong>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Gene Filtering</div>
                        <div class="card-body">
                            <ul class="list-group list-group-flush">
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Genes Before Filter</span>
                                    <strong>{qc_report.get('genes_before_filter', 'N/A'):,}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Genes After Filter</span>
                                    <strong>{qc_report.get('genes_after_filter', 'N/A'):,}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Median Library Size</span>
                                    <strong>{qc_report.get('median_library_size', 0):,.0f}</strong>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _build_figures_section(self, figures_dir: str) -> str:
        """Build figures section with embedded images"""
        figures_path = Path(figures_dir)

        if not figures_path.exists():
            return ""

        figure_files = list(figures_path.glob("*.png"))

        if not figure_files:
            return ""

        figures_html = ""
        for fig_file in sorted(figure_files):
            # Read and encode image
            with open(fig_file, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

            fig_name = fig_file.stem.replace('_', ' ').title()

            figures_html += f"""
            <div class="col-md-6 figure-container">
                <div class="card">
                    <div class="card-header">{fig_name}</div>
                    <div class="card-body text-center">
                        <img src="data:image/png;base64,{img_data}" alt="{fig_name}" class="img-fluid">
                    </div>
                </div>
            </div>
            """

        return f"""
        <section id="figures" class="mb-5">
            <h2 class="section-title"><i class="fas fa-image"></i> Visualizations</h2>
            <div class="row">
                {figures_html}
            </div>
        </section>
        """


def generate_html_report(
    deg_results: pd.DataFrame,
    hub_genes: pd.DataFrame,
    gene_cards: Optional[List] = None,
    pathway_results: Optional[pd.DataFrame] = None,
    qc_report: Optional[Dict] = None,
    figures_dir: Optional[str] = None,
    output_dir: str = "reports",
    title: str = "RNA-seq Analysis Report"
) -> str:
    """
    Convenience function to generate HTML report

    Returns:
        Path to generated HTML file
    """
    generator = HTMLReportGenerator(output_dir=output_dir)
    return generator.generate_report(
        deg_results=deg_results,
        hub_genes=hub_genes,
        gene_cards=gene_cards,
        pathway_results=pathway_results,
        qc_report=qc_report,
        figures_dir=figures_dir,
        title=title
    )


if __name__ == "__main__":
    # Test with existing data
    results_dir = Path('/Users/admin/VectorDB_BioInsight/rnaseq_test_results/test_run')

    deg_results = pd.read_csv(results_dir / 'deseq2_all_results.csv')
    hub_genes = pd.read_csv(results_dir / 'hub_genes.csv')
    pathway_results = pd.read_csv(results_dir / 'pathway_enrichment.csv')

    report_path = generate_html_report(
        deg_results=deg_results,
        hub_genes=hub_genes,
        pathway_results=pathway_results,
        figures_dir=str(results_dir / 'figures'),
        output_dir=str(results_dir / 'reports'),
        title="Test RNA-seq Analysis Report"
    )

    print(f"Report generated: {report_path}")
