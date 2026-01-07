"""
Agent 6: Interactive HTML Report Generation

Generates a comprehensive, interactive HTML report combining all analysis results.

Input:
- All outputs from Agents 1-5
- config.json: Report configuration

Output:
- report.html: Interactive HTML report
- report_data.json: Data used by the report
- meta_agent6.json: Execution metadata
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ..utils.base_agent import BaseAgent


class ReportAgent(BaseAgent):
    """Agent for generating interactive HTML reports."""

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
            "embed_figures": True
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent6_report", input_dir, output_dir, merged_config)

    def validate_inputs(self) -> bool:
        """Validate that required data files exist."""
        # Check for essential files
        essential_files = ["deg_significant.csv", "integrated_gene_table.csv"]
        for f in essential_files:
            if not (self.input_dir / f).exists():
                # Try parent directories (pipeline output structure)
                found = False
                for subdir in ["agent1_deg", "agent4_validation"]:
                    if (self.input_dir.parent / subdir / f).exists():
                        found = True
                        break
                if not found:
                    self.logger.warning(f"Missing file: {f}")

        return True  # Allow report generation with partial data

    def _load_all_data(self) -> Dict[str, Any]:
        """Load all available data from previous agents."""
        data = {}

        # Try loading from input_dir and subdirectories
        search_paths = [
            self.input_dir,
            self.input_dir / "agent1_deg",
            self.input_dir / "agent2_network",
            self.input_dir / "agent3_pathway",
            self.input_dir / "agent4_validation",
            self.input_dir / "agent5_visualization"
        ]

        # Files to load
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

        return data

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate interactive HTML report."""

        # Extract summary statistics
        deg_count = len(data.get('deg_significant', []))
        hub_count = len(data.get('hub_genes', []))
        pathway_count = len(data.get('pathway_summary', []))

        interpretation = data.get('interpretation_report', {})
        summary = interpretation.get('summary', {})
        db_matched = summary.get('db_matched_count', 0)
        high_confidence = summary.get('high_confidence_count', 0)
        novel_candidates = summary.get('novel_candidates_count', 0)

        # Generate table HTML for integrated results
        integrated_table_html = self._generate_table_html(
            data.get('integrated_gene_table', []),
            ['gene_id', 'log2FC', 'padj', 'is_hub', 'db_matched', 'confidence', 'tags'],
            'integrated-table'
        )

        # Generate table HTML for pathways
        pathway_table_html = self._generate_table_html(
            data.get('pathway_summary', []),
            ['database', 'term_name', 'padj', 'gene_count'],
            'pathway-table'
        )

        # Generate figures HTML
        figures_html = ""
        for name, src in data.get('figures', {}).items():
            title = name.replace('_', ' ').title()
            figures_html += f'''
            <div class="figure-card">
                <h4>{title}</h4>
                <img src="{src}" alt="{title}" />
            </div>
            '''

        # Interpretation narratives
        matched_narratives = ""
        for gene_info in interpretation.get('matched_genes', [])[:10]:
            matched_narratives += f'''
            <div class="gene-card matched">
                <h4>{gene_info['gene']}</h4>
                <p class="confidence">Confidence: <span class="badge {gene_info['checklist']['confidence']}">{gene_info['checklist']['confidence']}</span></p>
                <p>{gene_info['narrative']}</p>
                <p class="tags">Tags: {', '.join(gene_info['checklist'].get('tags', []))}</p>
            </div>
            '''

        unmatched_narratives = ""
        for gene_info in interpretation.get('unmatched_genes', [])[:10]:
            unmatched_narratives += f'''
            <div class="gene-card novel">
                <h4>{gene_info['gene']}</h4>
                <p class="hypothesis">Hypothesis: {gene_info.get('hypothesis', 'N/A')}</p>
                <p>{gene_info['narrative']}</p>
            </div>
            '''

        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config["report_title"]}</title>
    <style>
        :root {{
            --primary: #4C1D95;
            --secondary: #7C3AED;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --gray: #6B7280;
            --light: #F3F4F6;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--light);
            color: #1F2937;
            line-height: 1.6;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
        }}

        header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        header .meta {{ opacity: 0.9; font-size: 0.9rem; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .summary-card .number {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .summary-card .label {{
            color: var(--gray);
            font-size: 0.9rem;
            margin-top: 5px;
        }}

        section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        section h2 {{
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--light);
        }}

        th {{
            background: var(--light);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}

        tr:hover {{ background: #F9FAFB; }}

        .table-container {{
            max-height: 500px;
            overflow-y: auto;
            margin-top: 15px;
        }}

        .search-box {{
            width: 100%;
            padding: 12px;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            font-size: 1rem;
            margin-bottom: 15px;
        }}

        .figures-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}

        .figure-card {{
            background: var(--light);
            border-radius: 8px;
            padding: 15px;
        }}

        .figure-card h4 {{
            margin-bottom: 10px;
            color: var(--gray);
        }}

        .figure-card img {{
            width: 100%;
            border-radius: 4px;
        }}

        .gene-card {{
            background: var(--light);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid var(--gray);
        }}

        .gene-card.matched {{ border-left-color: var(--success); }}
        .gene-card.novel {{ border-left-color: var(--secondary); }}

        .gene-card h4 {{
            color: var(--primary);
            margin-bottom: 10px;
        }}

        .gene-card .tags {{
            font-size: 0.85rem;
            color: var(--gray);
            margin-top: 10px;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .badge.high {{ background: #D1FAE5; color: #065F46; }}
        .badge.medium {{ background: #FEF3C7; color: #92400E; }}
        .badge.low {{ background: #FEE2E2; color: #991B1B; }}
        .badge.novel_candidate {{ background: #EDE9FE; color: #5B21B6; }}
        .badge.requires_validation {{ background: #F3F4F6; color: #4B5563; }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}

        @media (max-width: 768px) {{
            .two-column {{ grid-template-columns: 1fr; }}
            .figures-grid {{ grid-template-columns: 1fr; }}
        }}

        .methods {{
            background: #F9FAFB;
            border-radius: 8px;
            padding: 20px;
            font-size: 0.9rem;
        }}

        .methods h4 {{ margin-bottom: 10px; }}
        .methods ul {{ margin-left: 20px; }}

        footer {{
            text-align: center;
            padding: 30px;
            color: var(--gray);
            font-size: 0.9rem;
        }}

        .guidelines {{
            background: linear-gradient(135deg, #FEF3C7, #FDE68A);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }}

        .guidelines h4 {{
            color: #92400E;
            margin-bottom: 10px;
        }}

        .guidelines ul {{
            margin-left: 20px;
            color: #78350F;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{self.config["report_title"]}</h1>
            <p class="meta">
                Generated by {self.config["author"]} |
                {datetime.now().strftime("%Y-%m-%d %H:%M")} |
                Cancer Type: {interpretation.get('cancer_type', 'N/A')}
            </p>
        </div>
    </header>

    <div class="container">
        <!-- Executive Summary -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="number">{deg_count}</div>
                <div class="label">Significant DEGs</div>
            </div>
            <div class="summary-card">
                <div class="number">{hub_count}</div>
                <div class="label">Hub Genes</div>
            </div>
            <div class="summary-card">
                <div class="number">{db_matched}</div>
                <div class="label">DB Matched</div>
            </div>
            <div class="summary-card">
                <div class="number">{high_confidence}</div>
                <div class="label">High Confidence</div>
            </div>
            <div class="summary-card">
                <div class="number">{novel_candidates}</div>
                <div class="label">Novel Candidates</div>
            </div>
            <div class="summary-card">
                <div class="number">{pathway_count}</div>
                <div class="label">Enriched Pathways</div>
            </div>
        </div>

        <!-- Integrated Results -->
        <section>
            <h2>Integrated Gene Analysis</h2>
            <input type="text" class="search-box" id="gene-search"
                   placeholder="Search genes..." onkeyup="filterTable('integrated-table', this.value)">
            <div class="table-container">
                {integrated_table_html}
            </div>
        </section>

        <!-- Interpretation -->
        <section>
            <h2>Interpretation Results</h2>

            <div class="guidelines">
                <h4>Interpretation Guidelines Applied</h4>
                <ul>
                    <li>DEG alone does not equal biological importance</li>
                    <li>DB match provides context, not proof</li>
                    <li>DB mismatch means unknown, not unimportant</li>
                    <li>Hub status increases confidence for novel candidates</li>
                    <li>Pathway context strengthens interpretation</li>
                </ul>
            </div>

            <div class="two-column" style="margin-top: 20px;">
                <div>
                    <h3 style="color: var(--success); margin-bottom: 15px;">DB-Matched Genes</h3>
                    {matched_narratives if matched_narratives else '<p class="text-gray">No DB-matched genes found</p>'}
                </div>
                <div>
                    <h3 style="color: var(--secondary); margin-bottom: 15px;">Novel Candidates</h3>
                    {unmatched_narratives if unmatched_narratives else '<p class="text-gray">No novel candidates identified</p>'}
                </div>
            </div>
        </section>

        <!-- Pathway Results -->
        <section>
            <h2>Pathway Enrichment</h2>
            <div class="table-container">
                {pathway_table_html}
            </div>
        </section>

        <!-- Visualizations -->
        <section>
            <h2>Visualizations</h2>
            <div class="figures-grid">
                {figures_html if figures_html else '<p>No figures available</p>'}
            </div>
        </section>

        <!-- Methods -->
        {self._generate_methods_html() if self.config["include_methods"] else ""}

    </div>

    <footer>
        <p>Generated by BioInsight AI RNA-seq Pipeline v1.0</p>
        <p>This report is for research purposes only.</p>
    </footer>

    <script>
        function filterTable(tableId, searchText) {{
            const table = document.getElementById(tableId);
            const rows = table.getElementsByTagName('tr');
            const search = searchText.toLowerCase();

            for (let i = 1; i < rows.length; i++) {{
                const cells = rows[i].getElementsByTagName('td');
                let found = false;
                for (let j = 0; j < cells.length; j++) {{
                    if (cells[j].textContent.toLowerCase().includes(search)) {{
                        found = true;
                        break;
                    }}
                }}
                rows[i].style.display = found ? '' : 'none';
            }}
        }}
    </script>
</body>
</html>
'''
        return html

    def _generate_table_html(self, data: List[Dict], columns: List[str], table_id: str) -> str:
        """Generate HTML table from data."""
        if not data:
            return '<p>No data available</p>'

        # Limit rows
        data = data[:self.config["max_table_rows"]]

        # Filter to available columns
        available_cols = [c for c in columns if c in data[0]]

        html = f'<table id="{table_id}">\n<thead><tr>'
        for col in available_cols:
            html += f'<th>{col.replace("_", " ").title()}</th>'
        html += '</tr></thead>\n<tbody>'

        for row in data:
            html += '<tr>'
            for col in available_cols:
                val = row.get(col, '')
                if isinstance(val, float):
                    val = f'{val:.4f}' if abs(val) < 0.01 else f'{val:.2f}'
                elif isinstance(val, bool):
                    val = 'Yes' if val else 'No'
                html += f'<td>{val}</td>'
            html += '</tr>\n'

        html += '</tbody></table>'
        return html

    def _generate_methods_html(self) -> str:
        """Generate methods section HTML."""
        return '''
        <section>
            <h2>Methods</h2>
            <div class="methods">
                <h4>Analysis Pipeline</h4>
                <ul>
                    <li><strong>DEG Analysis:</strong> DESeq2 with |log2FC| > 1, padj < 0.05</li>
                    <li><strong>Network Analysis:</strong> NetworkX co-expression network, Spearman correlation > 0.7</li>
                    <li><strong>Pathway Enrichment:</strong> gseapy with GO (BP, MF, CC) and KEGG databases</li>
                    <li><strong>DB Validation:</strong> COSMIC Tier 1, OncoKB gene matching</li>
                    <li><strong>Interpretation:</strong> Systematic checklist-based scoring</li>
                </ul>

                <h4 style="margin-top: 15px;">Confidence Scoring</h4>
                <ul>
                    <li>Hub gene: +2 points</li>
                    <li>DB matched: +2 points</li>
                    <li>Cancer type specific: +1.5 points</li>
                    <li>High pathway involvement: +0.5 points</li>
                </ul>

                <h4 style="margin-top: 15px;">Confidence Thresholds</h4>
                <ul>
                    <li><strong>High:</strong> Score ≥ 5 and DB matched</li>
                    <li><strong>Medium:</strong> Score ≥ 3 and DB matched</li>
                    <li><strong>Novel Candidate:</strong> Not in DB but is Hub gene</li>
                    <li><strong>Requires Validation:</strong> Score < 1.5</li>
                </ul>
            </div>
        </section>
        '''

    def run(self) -> Dict[str, Any]:
        """Generate the HTML report."""
        # Load all data
        data = self._load_all_data()

        # Save data as JSON for potential reuse
        self.save_json(data, "report_data.json")

        # Generate HTML
        html_content = self._generate_html(data)

        # Save HTML
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

        # Check file size (should be non-trivial)
        if report_path.stat().st_size < 1000:
            self.logger.error("Report HTML seems too small")
            return False

        return True
