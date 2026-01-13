"""
Network Enhancement Module

Based on concepts from:
1. GCNN (Graph Convolutional Neural Networks) - Co-expression graph with significance testing
2. KG4SL (Knowledge Graph) - Entity/relationship integration for gene networks
3. inferCSN - L0+L2 sparse regression for Gene Regulatory Network inference

This module provides enhanced network analysis capabilities for the RNA-seq pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats
from scipy.sparse import csr_matrix
import warnings

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

import logging

logger = logging.getLogger(__name__)


class NetworkEnhancer:
    """
    Enhanced network analysis module incorporating methods from recent papers.

    Features:
    1. Significance-tested co-expression graph (GCNN paper)
    2. Graph Laplacian and spectral features
    3. L0+L2 sparse regression for GRN inference (inferCSN)
    4. Knowledge graph entity integration concepts
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            # Co-expression graph parameters (GCNN-style)
            "correlation_method": "spearman",
            "correlation_threshold": 0.6,  # GCNN uses 0.6
            "pvalue_threshold": 0.05,
            "use_absolute_correlation": True,

            # Sparse regression parameters (inferCSN-style)
            "l0_penalty": 0.01,  # γ₁ for L0 norm (sparsity)
            "l2_penalty": 0.001,  # γ₂ for L2 norm (regularization)
            "sparse_max_iter": 100,

            # Graph Laplacian parameters
            "laplacian_type": "normalized",  # 'normalized' or 'combinatorial'

            # Hub detection
            "top_hub_count": 20,
            "min_edges_for_hub": 5,
        }
        self.config = {**default_config, **(config or {})}

    def build_coexpression_graph_with_significance(
        self,
        expression_df: pd.DataFrame,
        gene_ids: List[str]
    ) -> Tuple[pd.DataFrame, nx.Graph]:
        """
        Build co-expression graph with significance testing (GCNN-style).

        From GCNN paper:
        - Spearman correlation > 0.6 with p-value < 0.05
        - Creates adjacency matrix for graph convolution

        Uses vectorized correlation calculation for efficiency.

        Args:
            expression_df: Expression matrix (genes x samples)
            gene_ids: List of gene IDs to include

        Returns:
            edges_df: Edge list with correlation and p-values
            G: NetworkX graph
        """
        logger.info(f"Building co-expression graph for {len(gene_ids)} genes (GCNN-style)")

        # Filter expression data
        expr_data = expression_df.loc[expression_df.index.isin(gene_ids)]

        threshold = self.config["correlation_threshold"]
        pvalue_thresh = self.config["pvalue_threshold"]
        use_abs = self.config["use_absolute_correlation"]
        method = self.config["correlation_method"]

        genes = expr_data.index.tolist()
        n_genes = len(genes)
        n_samples = expr_data.shape[1]

        logger.info(f"Calculating pairwise correlations (vectorized, {n_genes} genes, {n_samples} samples)...")

        # Vectorized correlation calculation
        if method == "spearman":
            # Convert to ranks for Spearman
            expr_matrix = expr_data.T.rank().values  # samples x genes
        else:
            expr_matrix = expr_data.T.values  # samples x genes

        # Standardize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expr_mean = np.nanmean(expr_matrix, axis=0, keepdims=True)
            expr_std = np.nanstd(expr_matrix, axis=0, keepdims=True)
            expr_std[expr_std == 0] = 1  # Avoid division by zero
            expr_standardized = (expr_matrix - expr_mean) / expr_std

        # Replace NaN with 0 for matrix multiplication
        expr_standardized = np.nan_to_num(expr_standardized, nan=0.0)

        # Compute correlation matrix: corr = (X.T @ X) / (n - 1)
        corr_matrix = np.dot(expr_standardized.T, expr_standardized) / (n_samples - 1)

        logger.info(f"Correlation matrix computed ({n_genes}x{n_genes})")

        # Calculate approximate p-values using t-distribution
        # t = r * sqrt((n-2)/(1-r^2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat = corr_matrix * np.sqrt((n_samples - 2) / (1 - corr_matrix**2 + 1e-10))
            pval_matrix = 2 * stats.t.sf(np.abs(t_stat), df=n_samples - 2)

        # Extract significant edges (upper triangle only)
        logger.info(f"Extracting significant edges (threshold={threshold}, pvalue<{pvalue_thresh})...")

        edges = []
        # Use numpy operations for speed
        if use_abs:
            mask = (np.abs(corr_matrix) >= threshold) & (pval_matrix < pvalue_thresh)
        else:
            mask = (corr_matrix >= threshold) & (pval_matrix < pvalue_thresh)

        # Only upper triangle (avoid duplicates and self-loops)
        upper_mask = np.triu(mask, k=1)
        edge_indices = np.where(upper_mask)

        for i, j in zip(edge_indices[0], edge_indices[1]):
            corr = corr_matrix[i, j]
            pval = pval_matrix[i, j]
            edges.append({
                'gene1': genes[i],
                'gene2': genes[j],
                'correlation': float(corr),
                'abs_correlation': float(abs(corr)),
                'pvalue': float(pval),
                'significant': True
            })

        edges_df = pd.DataFrame(edges)
        logger.info(f"Found {len(edges_df)} significant edges (corr >= {threshold}, p < {pvalue_thresh})")

        # Build NetworkX graph using from_pandas_edgelist (vectorized, much faster)
        logger.info("Building NetworkX graph (vectorized)...")

        if len(edges_df) > 0:
            # Rename columns for edge attributes
            edges_df['weight'] = edges_df['abs_correlation']
            G = nx.from_pandas_edgelist(
                edges_df,
                source='gene1',
                target='gene2',
                edge_attr=['weight', 'correlation', 'pvalue']
            )
            # Add any isolated nodes (genes with no significant edges)
            isolated_genes = set(genes) - set(G.nodes())
            G.add_nodes_from(isolated_genes)
        else:
            G = nx.Graph()
            G.add_nodes_from(genes)

        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return edges_df, G

    def calculate_graph_laplacian(
        self,
        G: nx.Graph,
        laplacian_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Graph Laplacian matrix (GCNN paper).

        From GCNN paper:
        L = I_n - D^(-1/2) W D^(-1/2)  (normalized Laplacian)

        Args:
            G: NetworkX graph
            laplacian_type: 'normalized' or 'combinatorial'

        Returns:
            L: Laplacian matrix
            eigenvalues: Eigenvalues for spectral analysis
        """
        laplacian_type = laplacian_type or self.config["laplacian_type"]

        if G.number_of_nodes() == 0:
            return np.array([]), np.array([])

        # Get adjacency matrix
        nodes = list(G.nodes())
        n = len(nodes)

        A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

        if laplacian_type == "normalized":
            # L = I - D^(-1/2) W D^(-1/2)
            D = np.diag(A.sum(axis=1))
            D_inv_sqrt = np.zeros_like(D)
            nonzero = np.diag(D) > 0
            D_inv_sqrt[nonzero, nonzero] = 1.0 / np.sqrt(np.diag(D)[nonzero])
            L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            # Combinatorial Laplacian: L = D - A
            D = np.diag(A.sum(axis=1))
            L = D - A

        # Calculate eigenvalues for spectral analysis
        try:
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = np.sort(eigenvalues)
        except np.linalg.LinAlgError:
            eigenvalues = np.array([])

        logger.info(f"Computed {laplacian_type} Laplacian ({n}x{n})")

        return L, eigenvalues

    def sparse_regression_grn(
        self,
        expression_df: pd.DataFrame,
        target_genes: List[str],
        regulator_genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Infer Gene Regulatory Network using L0+L2 sparse regression (inferCSN-style).

        From inferCSN paper:
        argmin (1/2)||y - Xβ||² + γ₁||β||₀ + γ₂||β||²

        This simplified implementation uses coordinate descent with L1 approximation
        of L0 norm (Lasso) combined with L2 (Elastic Net).

        Args:
            expression_df: Expression matrix (genes x samples)
            target_genes: Genes to predict (targets)
            regulator_genes: Potential regulators (if None, use all genes)

        Returns:
            grn_df: Regulatory relationships with weights
        """
        logger.info("Inferring GRN using sparse regression (inferCSN-style)")

        try:
            from sklearn.linear_model import ElasticNet
        except ImportError:
            logger.warning("sklearn not available, skipping sparse regression")
            return pd.DataFrame()

        l1_ratio = self.config["l0_penalty"] / (self.config["l0_penalty"] + self.config["l2_penalty"])
        alpha = self.config["l0_penalty"] + self.config["l2_penalty"]

        if regulator_genes is None:
            regulator_genes = expression_df.index.tolist()

        # Filter to available genes
        target_genes = [g for g in target_genes if g in expression_df.index]
        regulator_genes = [g for g in regulator_genes if g in expression_df.index]

        grn_edges = []

        for target in target_genes:
            # Exclude self-regulation
            predictors = [g for g in regulator_genes if g != target]

            if len(predictors) == 0:
                continue

            y = expression_df.loc[target].values
            X = expression_df.loc[predictors].T.values

            # Remove samples with NaN
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue

            y_clean = y[mask]
            X_clean = X[mask, :]

            # Fit Elastic Net (approximation of L0+L2)
            try:
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=self.config["sparse_max_iter"],
                    random_state=42
                )
                model.fit(X_clean, y_clean)

                # Extract non-zero coefficients
                for idx, coef in enumerate(model.coef_):
                    if abs(coef) > 1e-6:
                        grn_edges.append({
                            'regulator': predictors[idx],
                            'target': target,
                            'weight': coef,
                            'abs_weight': abs(coef),
                            'direction': 'activation' if coef > 0 else 'repression'
                        })
            except Exception as e:
                logger.debug(f"Failed to fit model for {target}: {e}")
                continue

        grn_df = pd.DataFrame(grn_edges)
        if len(grn_df) > 0:
            grn_df = grn_df.sort_values('abs_weight', ascending=False)

        logger.info(f"Inferred {len(grn_df)} regulatory relationships")

        return grn_df

    def integrate_knowledge_graph_features(
        self,
        nodes_df: pd.DataFrame,
        disease_associations: Optional[pd.DataFrame] = None,
        pathway_annotations: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Integrate knowledge graph-style features (KG4SL concept).

        From KG4SL paper:
        - Entity types: Gene, Disease, Pathway, Biological Process, etc.
        - Relationship types: participates, interacts, associated_with, etc.

        This adds semantic features to nodes based on external annotations.

        Args:
            nodes_df: Node dataframe with gene_id
            disease_associations: DisGeNET/OMIM associations
            pathway_annotations: GO/KEGG pathway annotations

        Returns:
            Enhanced nodes_df with KG features
        """
        logger.info("Integrating knowledge graph-style features (KG4SL concept)")

        enhanced_df = nodes_df.copy()

        # Add disease association count (entity: Disease)
        if disease_associations is not None and 'gene_id' in disease_associations.columns:
            disease_counts = disease_associations.groupby('gene_id').size()
            enhanced_df['disease_association_count'] = enhanced_df['gene_id'].map(
                disease_counts
            ).fillna(0).astype(int)

            # Max disease score if available
            if 'score' in disease_associations.columns:
                max_scores = disease_associations.groupby('gene_id')['score'].max()
                enhanced_df['max_disease_score'] = enhanced_df['gene_id'].map(
                    max_scores
                ).fillna(0)
        else:
            enhanced_df['disease_association_count'] = 0
            enhanced_df['max_disease_score'] = 0

        # Add pathway participation count (entity: Pathway)
        if pathway_annotations is not None and 'gene_id' in pathway_annotations.columns:
            pathway_counts = pathway_annotations.groupby('gene_id').size()
            enhanced_df['pathway_count'] = enhanced_df['gene_id'].map(
                pathway_counts
            ).fillna(0).astype(int)
        else:
            enhanced_df['pathway_count'] = 0

        # Calculate KG-enhanced hub score
        # Incorporate biological relevance from KG features
        if 'hub_score' in enhanced_df.columns:
            # Normalize KG features
            for col in ['disease_association_count', 'pathway_count']:
                max_val = enhanced_df[col].max()
                if max_val > 0:
                    enhanced_df[f'{col}_norm'] = enhanced_df[col] / max_val
                else:
                    enhanced_df[f'{col}_norm'] = 0

            # KG-enhanced hub score (adds biological relevance)
            enhanced_df['kg_enhanced_hub_score'] = (
                enhanced_df['hub_score'] * 0.6 +
                enhanced_df['disease_association_count_norm'] * 0.25 +
                enhanced_df['pathway_count_norm'] * 0.15
            )

        logger.info(f"Added KG features to {len(enhanced_df)} nodes")

        return enhanced_df

    def calculate_spectral_features(
        self,
        G: nx.Graph,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate spectral features for network analysis (GCNN paper).

        Features based on Graph Laplacian eigenvalues.

        Args:
            G: NetworkX graph
            k: Number of top eigenvalues to use

        Returns:
            Dictionary of spectral features
        """
        L, eigenvalues = self.calculate_graph_laplacian(G)

        if len(eigenvalues) == 0:
            return {
                'spectral_gap': 0,
                'algebraic_connectivity': 0,
                'num_components': 0,
                'spectral_radius': 0
            }

        # Spectral gap (λ₂ - λ₁)
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0

        # Algebraic connectivity (second smallest eigenvalue of Laplacian)
        algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0

        # Number of connected components (count of zero eigenvalues)
        num_components = np.sum(np.abs(eigenvalues) < 1e-10)

        # Spectral radius
        spectral_radius = eigenvalues[-1] if len(eigenvalues) > 0 else 0

        return {
            'spectral_gap': float(spectral_gap),
            'algebraic_connectivity': float(algebraic_connectivity),
            'num_components': int(num_components),
            'spectral_radius': float(spectral_radius),
            'top_eigenvalues': eigenvalues[:k].tolist() if len(eigenvalues) >= k else eigenvalues.tolist()
        }

    def identify_enhanced_hub_genes(
        self,
        G: nx.Graph,
        deg_df: pd.DataFrame,
        grn_df: Optional[pd.DataFrame] = None,
        disease_df: Optional[pd.DataFrame] = None,
        pathway_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Identify hub genes using enhanced scoring with all available features.

        Combines:
        - Network centrality metrics (degree, betweenness, eigenvector, closeness)
        - GRN regulatory importance (from sparse regression)
        - KG features (disease associations, pathway participation)

        Args:
            G: NetworkX graph
            deg_df: DEG dataframe with log2FC, padj
            grn_df: GRN dataframe from sparse regression
            disease_df: Disease association data
            pathway_df: Pathway annotation data

        Returns:
            Hub genes dataframe with enhanced scores
        """
        logger.info("Identifying hub genes with enhanced scoring")

        # Calculate basic centrality metrics
        nodes = list(G.nodes())

        degree = dict(G.degree())
        degree_centrality = nx.degree_centrality(G)

        if G.number_of_edges() > 0:
            # Use approximate betweenness centrality with k-sampling for large graphs
            # For large graphs (>1000 nodes), exact calculation is O(V*E) which is too slow
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            # For very dense graphs (>1M edges), skip betweenness entirely
            # Use degree as primary hub indicator (highly correlated with betweenness)
            if n_edges > 1_000_000:
                logger.info(f"Graph too dense ({n_edges:,} edges), using degree-only hub scoring")
                # Use weighted degree (sum of edge weights) as betweenness proxy
                betweenness = {n: sum(G[n][neighbor]['weight'] for neighbor in G[n]) for n in G.nodes()}
                # Normalize to 0-1 range
                max_b = max(betweenness.values()) if betweenness else 1
                betweenness = {n: v/max_b for n, v in betweenness.items()}
            elif n_nodes > 1000:
                # Sample k nodes for approximate betweenness (much faster)
                k = min(100, n_nodes // 50)  # Use at most 100 sample nodes
                logger.info(f"Using approximate betweenness centrality (k={k} samples for {n_nodes} nodes)")
                betweenness = nx.betweenness_centrality(G, k=k, weight='weight')
            else:
                betweenness = nx.betweenness_centrality(G, weight='weight')

            # Skip closeness for large graphs (very expensive)
            if n_nodes > 1000:
                # Use degree as a proxy for closeness (highly connected nodes are often central)
                logger.info("Using degree-based proxy for closeness centrality")
                closeness = {n: G.degree(n) / (n_nodes - 1) for n in G.nodes()}
            else:
                closeness = nx.closeness_centrality(G)
            try:
                if nx.is_connected(G):
                    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                else:
                    eigenvector = {n: 0.0 for n in nodes}
                    for component in nx.connected_components(G):
                        if len(component) > 1:
                            subgraph = G.subgraph(component)
                            try:
                                ev = nx.eigenvector_centrality(subgraph, max_iter=1000)
                                eigenvector.update(ev)
                            except:
                                pass
            except:
                eigenvector = {n: 0.0 for n in nodes}
        else:
            betweenness = {n: 0.0 for n in nodes}
            closeness = {n: 0.0 for n in nodes}
            eigenvector = {n: 0.0 for n in nodes}

        # Build nodes dataframe
        nodes_df = pd.DataFrame({
            'gene_id': nodes,
            'degree': [degree.get(n, 0) for n in nodes],
            'degree_centrality': [degree_centrality.get(n, 0) for n in nodes],
            'betweenness': [betweenness.get(n, 0) for n in nodes],
            'eigenvector': [eigenvector.get(n, 0) for n in nodes],
            'closeness': [closeness.get(n, 0) for n in nodes]
        })

        # Add GRN regulatory importance
        if grn_df is not None and len(grn_df) > 0:
            # Count outgoing regulations (as regulator)
            reg_out = grn_df.groupby('regulator').agg({
                'target': 'count',
                'abs_weight': 'sum'
            }).rename(columns={'target': 'regulatory_targets', 'abs_weight': 'regulatory_strength'})

            nodes_df = nodes_df.merge(
                reg_out.reset_index().rename(columns={'regulator': 'gene_id'}),
                on='gene_id',
                how='left'
            )
            nodes_df['regulatory_targets'] = nodes_df['regulatory_targets'].fillna(0).astype(int)
            nodes_df['regulatory_strength'] = nodes_df['regulatory_strength'].fillna(0)
        else:
            nodes_df['regulatory_targets'] = 0
            nodes_df['regulatory_strength'] = 0

        # Add KG features
        nodes_df = self.integrate_knowledge_graph_features(
            nodes_df,
            disease_associations=disease_df,
            pathway_annotations=pathway_df
        )

        # Merge with DEG info
        if 'gene_id' in deg_df.columns:
            deg_info = deg_df[['gene_id', 'log2FC', 'padj']].copy()
            if 'direction' in deg_df.columns:
                deg_info['direction'] = deg_df['direction']
            nodes_df = nodes_df.merge(deg_info, on='gene_id', how='left')

        # Calculate enhanced hub score
        # Normalize all features to 0-1
        feature_cols = ['degree', 'betweenness', 'eigenvector', 'closeness',
                       'regulatory_targets', 'regulatory_strength',
                       'disease_association_count', 'pathway_count']

        for col in feature_cols:
            if col in nodes_df.columns:
                max_val = nodes_df[col].max()
                if max_val > 0:
                    nodes_df[f'{col}_norm'] = nodes_df[col] / max_val
                else:
                    nodes_df[f'{col}_norm'] = 0
            else:
                nodes_df[f'{col}_norm'] = 0

        # Enhanced hub score with weights
        nodes_df['enhanced_hub_score'] = (
            nodes_df['degree_norm'] * 0.20 +
            nodes_df['betweenness_norm'] * 0.20 +
            nodes_df['eigenvector_norm'] * 0.15 +
            nodes_df['closeness_norm'] * 0.05 +
            nodes_df['regulatory_targets_norm'] * 0.15 +
            nodes_df['regulatory_strength_norm'] * 0.10 +
            nodes_df['disease_association_count_norm'] * 0.10 +
            nodes_df['pathway_count_norm'] * 0.05
        )

        # Sort and mark top hubs
        nodes_df = nodes_df.sort_values('enhanced_hub_score', ascending=False)
        top_count = self.config['top_hub_count']
        min_edges = self.config['min_edges_for_hub']

        # Filter by minimum edges
        candidates = nodes_df[nodes_df['degree'] >= min_edges].copy()
        if len(candidates) == 0:
            candidates = nodes_df.copy()

        nodes_df['is_enhanced_hub'] = False
        top_hub_ids = candidates.head(top_count)['gene_id'].tolist()
        nodes_df.loc[nodes_df['gene_id'].isin(top_hub_ids), 'is_enhanced_hub'] = True

        logger.info(f"Identified {len(top_hub_ids)} enhanced hub genes")

        return nodes_df


def enhance_network_analysis(
    expression_df: pd.DataFrame,
    deg_df: pd.DataFrame,
    disease_df: Optional[pd.DataFrame] = None,
    pathway_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main function to run enhanced network analysis.

    This combines all methods from the three papers:
    1. GCNN-style co-expression graph with significance testing
    2. inferCSN-style sparse regression for GRN
    3. KG4SL-style knowledge graph feature integration

    Args:
        expression_df: Expression matrix (genes x samples)
        deg_df: DEG dataframe with gene_id, log2FC, padj
        disease_df: Optional disease association data
        pathway_df: Optional pathway annotation data
        config: Optional configuration overrides

    Returns:
        Dictionary with all analysis results
    """
    enhancer = NetworkEnhancer(config)

    # Get gene list from DEGs
    gene_ids = deg_df['gene_id'].tolist()

    # 1. Build co-expression graph with significance testing (GCNN)
    edges_df, G = enhancer.build_coexpression_graph_with_significance(
        expression_df, gene_ids
    )

    # 2. Calculate spectral features (GCNN)
    spectral_features = enhancer.calculate_spectral_features(G)

    # 3. Infer GRN using sparse regression (inferCSN)
    # Use top DEGs as targets for computational efficiency
    top_degs = deg_df.nsmallest(100, 'padj')['gene_id'].tolist()
    grn_df = enhancer.sparse_regression_grn(expression_df, top_degs, gene_ids)

    # 4. Identify enhanced hub genes with KG features
    hub_df = enhancer.identify_enhanced_hub_genes(
        G, deg_df, grn_df, disease_df, pathway_df
    )

    return {
        'edges_df': edges_df,
        'graph': G,
        'grn_df': grn_df,
        'hub_df': hub_df,
        'spectral_features': spectral_features,
        'network_stats': {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_edges() > 0 else 0,
            'grn_edges': len(grn_df) if grn_df is not None else 0,
            'enhanced_hubs': int(hub_df['is_enhanced_hub'].sum())
        }
    }
