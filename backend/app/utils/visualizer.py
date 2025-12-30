"""
Visualization Module for Research Paper Analysis.

Features:
- Paper similarity heatmap/network
- Keyword frequency charts
- Year-over-year trend visualization
- Topic distribution charts
- Export to HTML/PNG
"""
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional
import json
import math

from .vector_store import BioVectorStore, create_vector_store
from .embeddings import PubMedBertEmbedder, get_embedder


@dataclass
class SimilarityMatrix:
    """Paper similarity matrix data."""
    paper_titles: list[str]
    similarity_scores: list[list[float]]
    clusters: list[int] = field(default_factory=list)

    def get_most_similar(self, paper_idx: int, top_n: int = 3) -> list[tuple[int, float]]:
        """Get most similar papers to a given paper."""
        scores = self.similarity_scores[paper_idx]
        indexed = [(i, s) for i, s in enumerate(scores) if i != paper_idx]
        return sorted(indexed, key=lambda x: -x[1])[:top_n]


@dataclass
class VisualizationData:
    """Container for all visualization data."""
    disease_domain: str

    # Paper similarity
    similarity_matrix: Optional[SimilarityMatrix] = None

    # Keyword data
    keyword_freq: list[tuple[str, int]] = field(default_factory=list)
    keyword_by_year: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    # Year distribution
    papers_by_year: dict[int, int] = field(default_factory=dict)

    # Topic clusters
    topic_clusters: dict[str, list[str]] = field(default_factory=dict)

    # Network edges (for graph visualization)
    paper_edges: list[tuple[str, str, float]] = field(default_factory=list)


class PaperVisualizer:
    """
    Generate visualizations for research paper analysis.

    Creates interactive HTML charts and exports using pure Python
    (no matplotlib dependency required).
    """

    def __init__(self, disease_domain: str):
        """Initialize visualizer."""
        self.disease_domain = disease_domain
        self.vector_store = create_vector_store(disease_domain=disease_domain)
        self.embeddings = get_embedder()

    def generate_all(self, output_dir: str = "./output/viz") -> VisualizationData:
        """Generate all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect data
        viz_data = self._collect_data()

        # Generate HTML visualizations
        self._generate_similarity_html(viz_data, output_path)
        self._generate_keyword_html(viz_data, output_path)
        self._generate_trend_html(viz_data, output_path)
        self._generate_network_html(viz_data, output_path)

        # Generate index page
        self._generate_index_html(viz_data, output_path)

        print(f"\n📊 Visualizations generated at: {output_path}")
        print(f"   Open {output_path}/index.html in a browser")

        return viz_data

    def _collect_data(self) -> VisualizationData:
        """Collect all data needed for visualizations."""
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        if not all_data["ids"]:
            return VisualizationData(disease_domain=self.disease_domain)

        # Organize by paper
        papers = {}
        for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
            title = meta.get("paper_title", "Unknown")
            if title not in papers:
                papers[title] = {
                    "year": self._extract_year(meta.get("year", "")),
                    "docs": [],
                    "embeddings": []
                }
            papers[title]["docs"].append(doc)
            if all_data["embeddings"] is not None and len(all_data["embeddings"]) > i:
                papers[title]["embeddings"].append(all_data["embeddings"][i])

        # Calculate paper-level embeddings (average of chunks)
        paper_titles = list(papers.keys())
        paper_embeddings = []

        for title in paper_titles:
            if papers[title]["embeddings"]:
                embs = papers[title]["embeddings"]
                avg_emb = [sum(e[i] for e in embs) / len(embs) for i in range(len(embs[0]))]
                paper_embeddings.append(avg_emb)
            else:
                paper_embeddings.append(None)

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(paper_titles, paper_embeddings)

        # Extract keywords
        all_text = " ".join(all_data["documents"])
        keyword_freq = self._extract_keywords(all_text)

        # Keywords by year
        docs_by_year = defaultdict(list)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            year = self._extract_year(meta.get("year", ""))
            if year:
                docs_by_year[year].append(doc)

        keyword_by_year = {}
        for year, docs in docs_by_year.items():
            keyword_by_year[year] = self._extract_keywords(" ".join(docs))[:10]

        # Papers by year
        papers_by_year = Counter(p["year"] for p in papers.values() if p["year"])

        # Paper edges for network
        paper_edges = []
        if similarity_matrix:
            for i, title1 in enumerate(paper_titles):
                for j, title2 in enumerate(paper_titles):
                    if i < j and similarity_matrix.similarity_scores[i][j] > 0.5:
                        paper_edges.append((
                            title1[:30],
                            title2[:30],
                            similarity_matrix.similarity_scores[i][j]
                        ))

        return VisualizationData(
            disease_domain=self.disease_domain,
            similarity_matrix=similarity_matrix,
            keyword_freq=keyword_freq,
            keyword_by_year=keyword_by_year,
            papers_by_year=dict(papers_by_year),
            paper_edges=paper_edges
        )

    def _calculate_similarity_matrix(
        self,
        titles: list[str],
        embeddings: list
    ) -> Optional[SimilarityMatrix]:
        """Calculate cosine similarity between all papers."""
        if not embeddings or all(e is None for e in embeddings):
            return None

        n = len(titles)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif embeddings[i] and embeddings[j]:
                    matrix[i][j] = self._cosine_similarity(embeddings[i], embeddings[j])

        return SimilarityMatrix(
            paper_titles=titles,
            similarity_scores=matrix
        )

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _extract_year(self, year_str: str) -> int:
        """Extract year from string."""
        import re
        if not year_str:
            return 0
        try:
            year = int(year_str)
            if 1900 <= year <= 2030:
                return year
        except (ValueError, TypeError):
            pass
        match = re.search(r'(19|20)\d{2}', str(year_str))
        if match:
            return int(match.group())
        return 0

    def _extract_keywords(self, text: str, top_n: int = 20) -> list[tuple[str, int]]:
        """Extract top keywords from text."""
        import re
        stopwords = {
            'study', 'patient', 'patients', 'result', 'results', 'method',
            'methods', 'conclusion', 'background', 'data', 'analysis',
            'group', 'level', 'value', 'significant', 'however', 'using',
            'based', 'showed', 'found', 'figure', 'table', 'abstract'
        }
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        return Counter(filtered).most_common(top_n)

    def _generate_similarity_html(self, data: VisualizationData, output_path: Path):
        """Generate paper similarity heatmap HTML."""
        if not data.similarity_matrix:
            return

        titles = [t[:25] + "..." if len(t) > 25 else t
                  for t in data.similarity_matrix.paper_titles]
        scores = data.similarity_matrix.similarity_scores

        # Build similarity table rows
        sim_rows = []
        for i in range(len(titles)):
            similar = data.similarity_matrix.get_most_similar(i, 1)
            if similar:
                sim_title = titles[similar[0][0]]
                sim_score = f"{similar[0][1]:.3f}"
            else:
                sim_title = "N/A"
                sim_score = "0"
            sim_rows.append(f"<tr><td>{titles[i]}</td><td>{sim_title}</td><td>{sim_score}</td></tr>")

        # Build matrix rows
        matrix_rows = []
        for i in range(len(titles)):
            cells = "".join(
                f'<td style="background: rgba(76, 175, 80, {scores[i][j]:.1f})">{scores[i][j]:.2f}</td>'
                for j in range(len(titles))
            )
            matrix_rows.append(f"<tr><td><strong>{i+1}. {titles[i][:15]}...</strong></td>{cells}</tr>")

        header_cells = "".join(f'<th title="{t}">{i+1}</th>' for i, t in enumerate(titles))

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Similarity - {data.disease_domain}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Paper Similarity Matrix</h1>
        <p>Domain: {data.disease_domain} | Papers: {len(titles)}</p>

        <h2>Similarity Scores</h2>
        <table>
            <tr><th>Paper</th><th>Most Similar To</th><th>Score</th></tr>
            {"".join(sim_rows)}
        </table>

        <h2>Raw Similarity Matrix</h2>
        <div style="overflow-x: auto;">
            <table style="font-size: 12px;">
                <tr><th></th>{header_cells}</tr>
                {"".join(matrix_rows)}
            </table>
        </div>
    </div>
</body>
</html>"""

        (output_path / "similarity.html").write_text(html)

    def _generate_keyword_html(self, data: VisualizationData, output_path: Path):
        """Generate keyword frequency chart HTML."""
        labels = [kw for kw, _ in data.keyword_freq[:15]]
        values = [count for _, count in data.keyword_freq[:15]]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Keyword Analysis - {data.disease_domain}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .chart-container {{ height: 400px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔑 Keyword Frequency Analysis</h1>
        <p>Domain: {data.disease_domain}</p>

        <div class="chart-container">
            <canvas id="keywordChart"></canvas>
        </div>

        <h2>Keywords by Year</h2>
        {"".join(f'''<h3>{year}</h3><p>{", ".join(f"{kw} ({c})" for kw, c in kws[:10])}</p>'''
                  for year, kws in sorted(data.keyword_by_year.items(), reverse=True))}
    </div>

    <script>
        new Chart(document.getElementById('keywordChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Frequency',
                    data: {json.dumps(values)},
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
    </script>
</body>
</html>"""

        (output_path / "keywords.html").write_text(html)

    def _generate_trend_html(self, data: VisualizationData, output_path: Path):
        """Generate year trend chart HTML."""
        years = sorted(data.papers_by_year.keys())
        counts = [data.papers_by_year[y] for y in years]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Trends - {data.disease_domain}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .chart-container {{ height: 300px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Research Publication Trends</h1>
        <p>Domain: {data.disease_domain}</p>

        <div class="chart-container">
            <canvas id="trendChart"></canvas>
        </div>

        <h2>Summary</h2>
        <ul>
            <li>Total Years Covered: {len(years)}</li>
            <li>Peak Year: {max(data.papers_by_year.items(), key=lambda x: x[1])[0] if data.papers_by_year else 'N/A'}</li>
            <li>Total Papers: {sum(counts)}</li>
        </ul>
    </div>

    <script>
        new Chart(document.getElementById('trendChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(years)},
                datasets: [{{
                    label: 'Papers Published',
                    data: {json.dumps(counts)},
                    fill: true,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
    </script>
</body>
</html>"""

        (output_path / "trends.html").write_text(html)

    def _generate_network_html(self, data: VisualizationData, output_path: Path):
        """Generate paper relationship network HTML."""
        if not data.paper_edges:
            return

        # Create nodes and edges for vis.js
        nodes = list(set([e[0] for e in data.paper_edges] + [e[1] for e in data.paper_edges]))
        node_data = [{"id": i, "label": n, "title": n} for i, n in enumerate(nodes)]
        edge_data = [{"from": nodes.index(e[0]), "to": nodes.index(e[1]), "value": e[2]}
                     for e in data.paper_edges]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Network - {data.disease_domain}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        #network {{ width: 100%; height: 600px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🕸️ Paper Relationship Network</h1>
        <p>Domain: {data.disease_domain} | Connections shown: similarity > 0.5</p>
        <div id="network"></div>
    </div>

    <script>
        var nodes = new vis.DataSet({json.dumps(node_data)});
        var edges = new vis.DataSet({json.dumps(edge_data)});

        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                color: {{ inherit: 'both' }}
            }},
            physics: {{
                stabilization: {{ iterations: 100 }}
            }}
        }};

        new vis.Network(container, data, options);
    </script>
</body>
</html>"""

        (output_path / "network.html").write_text(html)

    def _generate_index_html(self, data: VisualizationData, output_path: Path):
        """Generate index page linking all visualizations."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BioInsight Visualizations - {data.disease_domain}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ color: white; text-align: center; margin-bottom: 40px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 30px; text-decoration: none; color: #333; transition: transform 0.2s, box-shadow 0.2s; }}
        .card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .card h2 {{ margin: 0 0 10px 0; color: #667eea; }}
        .card p {{ margin: 0; color: #666; }}
        .card .icon {{ font-size: 40px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 BioInsight Visualizations</h1>
        <p style="color: white; text-align: center; margin-bottom: 30px;">Domain: {data.disease_domain}</p>

        <div class="card-grid">
            <a href="similarity.html" class="card">
                <div class="icon">🔗</div>
                <h2>Paper Similarity</h2>
                <p>View similarity matrix between all indexed papers</p>
            </a>

            <a href="keywords.html" class="card">
                <div class="icon">🔑</div>
                <h2>Keyword Analysis</h2>
                <p>Top keywords and their frequency across papers</p>
            </a>

            <a href="trends.html" class="card">
                <div class="icon">📈</div>
                <h2>Publication Trends</h2>
                <p>Year-over-year research publication trends</p>
            </a>

            <a href="network.html" class="card">
                <div class="icon">🕸️</div>
                <h2>Paper Network</h2>
                <p>Interactive network of related papers</p>
            </a>
        </div>
    </div>
</body>
</html>"""

        (output_path / "index.html").write_text(html)


def create_visualizer(disease_domain: str) -> PaperVisualizer:
    """Create a visualizer instance."""
    return PaperVisualizer(disease_domain=disease_domain)
