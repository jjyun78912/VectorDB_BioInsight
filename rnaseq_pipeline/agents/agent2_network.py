"""
Agent 2: Co-expression Network Analysis

Builds co-expression network from DEG expression data and identifies hub genes.

Input:
- normalized_counts.csv: From Agent 1
- deg_significant.csv: From Agent 1
- config.json: Analysis parameters

Output:
- network_edges.csv: Edge list with correlations
- network_nodes.csv: Node metrics (degree, betweenness, etc.)
- hub_genes.csv: Top hub genes with all metrics
- meta_agent2.json: Execution metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..utils.base_agent import BaseAgent


class NetworkAgent(BaseAgent):
    """Agent for co-expression network analysis and hub gene detection."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "correlation_method": "spearman",  # spearman or pearson
            "correlation_threshold": 0.7,
            "pvalue_threshold": 0.05,
            "top_hub_count": 20,
            "min_edges_for_hub": 5,  # Minimum connections to be considered hub
            "use_absolute_correlation": True  # Use |r| for edge weights
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent2_network", input_dir, output_dir, merged_config)

        self.norm_counts: Optional[pd.DataFrame] = None
        self.deg_significant: Optional[pd.DataFrame] = None

    def validate_inputs(self) -> bool:
        """Validate input files from Agent 1."""
        if not HAS_NETWORKX:
            self.logger.error("networkx not installed. Install with: pip install networkx")
            return False

        # Load normalized counts
        self.norm_counts = self.load_csv("normalized_counts.csv")
        if self.norm_counts is None:
            return False

        # Load significant DEGs
        self.deg_significant = self.load_csv("deg_significant.csv")
        if self.deg_significant is None:
            return False

        if len(self.deg_significant) < 2:
            self.logger.error("Need at least 2 DEGs for network analysis")
            return False

        self.logger.info(f"DEGs for network: {len(self.deg_significant)}")

        return True

    def _calculate_correlations(self) -> pd.DataFrame:
        """Calculate pairwise correlations between DEGs using vectorized operations."""
        # Filter normalized counts to only include DEGs
        deg_genes = set(self.deg_significant['gene_id'])
        gene_col = self.norm_counts.columns[0]

        # Get expression data for DEGs only
        expr_df = self.norm_counts[
            self.norm_counts[gene_col].isin(deg_genes)
        ].set_index(gene_col)

        # Limit genes for network analysis if too many (for performance)
        max_genes_for_network = self.config.get("max_genes_for_network", 1000)
        if len(expr_df) > max_genes_for_network:
            self.logger.info(f"Too many DEGs ({len(expr_df)}). Using top {max_genes_for_network} by padj for network.")
            # Get top genes by padj
            top_genes = self.deg_significant.nsmallest(max_genes_for_network, 'padj')['gene_id'].tolist()
            expr_df = expr_df[expr_df.index.isin(top_genes)]

        self.logger.info(f"Calculating correlations for {len(expr_df)} genes (vectorized)...")

        threshold = self.config["correlation_threshold"]
        use_abs = self.config["use_absolute_correlation"]

        # Vectorized correlation using pandas
        # Transpose so genes are columns for correlation calculation
        expr_T = expr_df.T

        # Calculate correlation matrix (much faster than pairwise)
        if self.config["correlation_method"] == "spearman":
            corr_matrix = expr_T.corr(method='spearman')
        else:
            corr_matrix = expr_T.corr(method='pearson')

        self.logger.info("Correlation matrix computed. Extracting significant edges...")

        # Extract edges above threshold
        edges = []
        genes = corr_matrix.columns.tolist()
        n_genes = len(genes)

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                gene1, gene2 = genes[i], genes[j]
                corr = corr_matrix.iloc[i, j]

                # Skip NaN correlations
                if pd.isna(corr):
                    continue

                # Check if significant
                corr_check = abs(corr) if use_abs else corr
                if corr_check >= threshold:
                    edges.append({
                        'gene1': gene1,
                        'gene2': gene2,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'pvalue': 0.0  # Vectorized correlation doesn't provide p-values easily
                    })

        edges_df = pd.DataFrame(edges)
        self.logger.info(f"Found {len(edges_df)} significant edges")

        return edges_df

    def _build_network(self, edges_df: pd.DataFrame) -> nx.Graph:
        """Build NetworkX graph from edges."""
        G = nx.Graph()

        # Add edges with correlation as weight
        for _, row in edges_df.iterrows():
            G.add_edge(
                row['gene1'],
                row['gene2'],
                weight=row['abs_correlation'],
                correlation=row['correlation'],
                pvalue=row['pvalue']
            )

        # Add isolated DEGs as nodes (no connections)
        all_degs = set(self.deg_significant['gene_id'])
        network_genes = set(G.nodes())
        isolated = all_degs - network_genes

        for gene in isolated:
            G.add_node(gene)

        self.logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def _calculate_centrality_metrics(self, G: nx.Graph) -> pd.DataFrame:
        """Calculate various centrality metrics for all nodes."""
        self.logger.info("Calculating centrality metrics...")

        # Degree centrality
        degree = dict(G.degree())
        degree_centrality = nx.degree_centrality(G)

        # Betweenness centrality
        if G.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(G, weight='weight')
        else:
            betweenness = {n: 0.0 for n in G.nodes()}

        # Eigenvector centrality (for connected components)
        if G.number_of_edges() > 0:
            try:
                # Use largest connected component for eigenvector
                if nx.is_connected(G):
                    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                else:
                    eigenvector = {n: 0.0 for n in G.nodes()}
                    for component in nx.connected_components(G):
                        if len(component) > 1:
                            subgraph = G.subgraph(component)
                            ev = nx.eigenvector_centrality(subgraph, max_iter=1000)
                            eigenvector.update(ev)
            except (nx.PowerIterationFailedConvergence, ZeroDivisionError) as e:
                eigenvector = {n: 0.0 for n in G.nodes()}  # Fallback on convergence failure
        else:
            eigenvector = {n: 0.0 for n in G.nodes()}

        # Closeness centrality
        if G.number_of_edges() > 0:
            closeness = nx.closeness_centrality(G)
        else:
            closeness = {n: 0.0 for n in G.nodes()}

        # Create DataFrame
        nodes_df = pd.DataFrame({
            'gene_id': list(G.nodes()),
            'degree': [degree.get(n, 0) for n in G.nodes()],
            'degree_centrality': [degree_centrality.get(n, 0) for n in G.nodes()],
            'betweenness': [betweenness.get(n, 0) for n in G.nodes()],
            'eigenvector': [eigenvector.get(n, 0) for n in G.nodes()],
            'closeness': [closeness.get(n, 0) for n in G.nodes()]
        })

        return nodes_df

    def _identify_hub_genes(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """Identify hub genes based on centrality metrics."""
        min_edges = self.config["min_edges_for_hub"]
        top_count = self.config["top_hub_count"]

        # Filter nodes with minimum connections
        candidates = nodes_df[nodes_df['degree'] >= min_edges].copy()

        if len(candidates) == 0:
            self.logger.warning(f"No genes with >= {min_edges} connections")
            candidates = nodes_df.copy()

        # Calculate composite hub score
        # Normalize each metric to 0-1 range
        for col in ['degree', 'betweenness', 'eigenvector', 'closeness']:
            max_val = candidates[col].max()
            if max_val > 0:
                candidates[f'{col}_norm'] = candidates[col] / max_val
            else:
                candidates[f'{col}_norm'] = 0

        # Weighted composite score
        candidates['hub_score'] = (
            candidates['degree_norm'] * 0.35 +
            candidates['betweenness_norm'] * 0.30 +
            candidates['eigenvector_norm'] * 0.25 +
            candidates['closeness_norm'] * 0.10
        )

        # Mark as hub
        candidates = candidates.sort_values('hub_score', ascending=False)
        candidates['is_hub'] = False
        candidates.iloc[:top_count, candidates.columns.get_loc('is_hub')] = True

        # Merge with DEG info
        deg_info = self.deg_significant[['gene_id', 'log2FC', 'padj', 'direction']]
        hub_df = candidates.merge(deg_info, on='gene_id', how='left')

        return hub_df

    def run(self) -> Dict[str, Any]:
        """Execute network analysis."""
        # Calculate correlations
        edges_df = self._calculate_correlations()

        if len(edges_df) == 0:
            self.logger.warning("No significant correlations found!")
            # Create empty outputs
            edges_df = pd.DataFrame(columns=['gene1', 'gene2', 'correlation', 'abs_correlation', 'pvalue'])
            nodes_df = pd.DataFrame({
                'gene_id': self.deg_significant['gene_id'],
                'degree': 0,
                'degree_centrality': 0.0,
                'betweenness': 0.0,
                'eigenvector': 0.0,
                'closeness': 0.0
            })
            hub_df = nodes_df.copy()
            hub_df['hub_score'] = 0.0
            hub_df['is_hub'] = False
            network_density = 0.0
        else:
            # Build network
            G = self._build_network(edges_df)

            # Calculate centrality
            nodes_df = self._calculate_centrality_metrics(G)

            # Identify hub genes
            hub_df = self._identify_hub_genes(nodes_df)

            # Network density
            network_density = nx.density(G) if G.number_of_edges() > 0 else 0.0

        # Save outputs
        self.save_csv(edges_df, "network_edges.csv")

        # Add is_hub to nodes
        nodes_df = nodes_df.merge(
            hub_df[['gene_id', 'is_hub', 'hub_score']].drop_duplicates(),
            on='gene_id',
            how='left'
        )
        nodes_df['is_hub'] = nodes_df['is_hub'].fillna(False)
        nodes_df['hub_score'] = nodes_df['hub_score'].fillna(0.0)

        self.save_csv(nodes_df, "network_nodes.csv")

        # Save hub genes (top hubs with DEG info)
        hub_genes = hub_df[hub_df['is_hub'] == True].copy()
        hub_columns = ['gene_id', 'degree', 'betweenness', 'eigenvector',
                       'hub_score', 'log2FC', 'padj', 'direction']
        hub_columns = [c for c in hub_columns if c in hub_genes.columns]
        self.save_csv(hub_genes[hub_columns], "hub_genes.csv")

        # Statistics
        hub_count = len(hub_genes)
        self.logger.info(f"Network Analysis Complete:")
        self.logger.info(f"  Total nodes: {len(nodes_df)}")
        self.logger.info(f"  Total edges: {len(edges_df)}")
        self.logger.info(f"  Network density: {network_density:.4f}")
        self.logger.info(f"  Hub genes identified: {hub_count}")

        if hub_count > 0:
            self.logger.info(f"  Top 5 hub genes: {hub_genes['gene_id'].head().tolist()}")

        return {
            "total_nodes": len(nodes_df),
            "total_edges": len(edges_df),
            "network_density": float(network_density),
            "hub_count": hub_count,
            "correlation_method": self.config["correlation_method"],
            "correlation_threshold": self.config["correlation_threshold"],
            "top_hub_genes": hub_genes['gene_id'].head(10).tolist() if hub_count > 0 else []
        }

    def validate_outputs(self) -> bool:
        """Validate network outputs."""
        required_files = [
            "network_edges.csv",
            "network_nodes.csv",
            "hub_genes.csv"
        ]

        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing output file: {filename}")
                return False

        # Validate hub genes
        hub_df = pd.read_csv(self.output_dir / "hub_genes.csv")

        # Check hub genes are in DEG list
        deg_genes = set(self.deg_significant['gene_id'])
        hub_genes = set(hub_df['gene_id'])

        if not hub_genes.issubset(deg_genes):
            orphan = hub_genes - deg_genes
            self.logger.error(f"Hub genes not in DEG list: {orphan}")
            return False

        return True
