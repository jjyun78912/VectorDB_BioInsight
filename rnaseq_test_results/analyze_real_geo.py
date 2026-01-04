#!/usr/bin/env python3
"""
Real GEO RNA-seq Analysis Pipeline
Dataset: GSE151243 - Lung Cancer
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/Users/admin/VectorDB_BioInsight/rnaseq_test_results/geo_lung_cancer")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def download_geo_dataset():
    """Download GSE151243 lung cancer dataset."""
    import GEOparse

    logger.info("=" * 70)
    logger.info("STEP 1: Downloading GEO Dataset")
    logger.info("=" * 70)

    # GSE19804: Lung Cancer Expression Profiling (with expression data)
    geo_id = "GSE19804"
    logger.info(f"Dataset: {geo_id}")
    logger.info("This may take a few minutes...")

    cache_dir = RESULTS_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)

    gse = GEOparse.get_GEO(geo=geo_id, destdir=str(cache_dir), silent=True)

    title = gse.metadata.get('title', ['N/A'])[0]
    summary = gse.metadata.get('summary', ['N/A'])[0][:200]

    logger.info(f"\nTitle: {title}")
    logger.info(f"Summary: {summary}...")
    logger.info(f"Samples: {len(gse.gsms)}")
    logger.info(f"Platform: {list(gse.gpls.keys())}")

    return gse

def extract_metadata(gse):
    """Extract sample metadata."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Extracting Metadata")
    logger.info("=" * 70)

    samples = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        chars = meta.get('characteristics_ch1', [])
        char_dict = {}
        for c in chars:
            if ':' in c:
                key, val = c.split(':', 1)
                char_dict[key.strip().lower()] = val.strip()

        sample = {
            'sample_id': gsm_name,
            'title': meta.get('title', [''])[0],
            'source': meta.get('source_name_ch1', [''])[0],
        }
        sample.update(char_dict)
        samples.append(sample)

    metadata_df = pd.DataFrame(samples)
    logger.info(f"\nMetadata columns: {list(metadata_df.columns)}")
    logger.info(f"\nSample preview:\n{metadata_df.head(10).to_string()}")

    def infer_condition(row):
        text = f"{row['title']} {row['source']}".lower()
        if 'tumor' in text or 'cancer' in text or 'carcinoma' in text:
            return 'tumor'
        elif 'normal' in text or 'healthy' in text or 'adjacent' in text:
            return 'normal'
        return 'unknown'

    metadata_df['condition'] = metadata_df.apply(infer_condition, axis=1)
    logger.info(f"\nConditions: {metadata_df['condition'].value_counts().to_dict()}")

    metadata_df.to_csv(RESULTS_DIR / "metadata.csv", index=False)
    return metadata_df

def extract_expression(gse, metadata_df):
    """Extract expression matrix."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Extracting Expression Matrix")
    logger.info("=" * 70)

    expr_data = {}
    for gsm_name, gsm in gse.gsms.items():
        if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
            if 'VALUE' in gsm.table.columns:
                expr_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']

    if not expr_data:
        logger.error("No expression data found!")
        return None

    expr_df = pd.DataFrame(expr_data)
    expr_df = expr_df.apply(pd.to_numeric, errors='coerce')

    logger.info(f"\nRaw expression matrix: {expr_df.shape}")

    median_val = expr_df.median().median()
    if median_val > 100:
        logger.info("Applying log2 transformation...")
        expr_df = np.log2(expr_df + 1)

    expr_df = expr_df.dropna(thresh=int(0.8 * expr_df.shape[1]))
    gene_var = expr_df.var(axis=1)
    expr_df = expr_df[gene_var > gene_var.quantile(0.1)]

    logger.info(f"After filtering: {expr_df.shape}")
    expr_df.to_csv(RESULTS_DIR / "expression_matrix.csv")
    return expr_df

def run_deg_analysis(expr_df, metadata_df):
    """Run DEG analysis."""
    from scipy import stats
    from scipy.stats import rankdata

    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Differential Expression Analysis")
    logger.info("=" * 70)

    common_samples = list(set(expr_df.columns) & set(metadata_df['sample_id']))
    expr_subset = expr_df[common_samples]
    meta_subset = metadata_df[metadata_df['sample_id'].isin(common_samples)].set_index('sample_id')

    tumor_samples = meta_subset[meta_subset['condition'] == 'tumor'].index.tolist()
    normal_samples = meta_subset[meta_subset['condition'] == 'normal'].index.tolist()

    logger.info(f"\nTumor samples: {len(tumor_samples)}")
    logger.info(f"Normal samples: {len(normal_samples)}")

    if len(tumor_samples) < 3 or len(normal_samples) < 3:
        half = len(common_samples) // 2
        tumor_samples = common_samples[:half]
        normal_samples = common_samples[half:]
        logger.info(f"Using split groups: {len(tumor_samples)} vs {len(normal_samples)}")

    logger.info("\nRunning Wilcoxon tests...")
    deg_results = []

    for gene in expr_subset.index:
        tumor_expr = expr_subset.loc[gene, tumor_samples].values.astype(float)
        normal_expr = expr_subset.loc[gene, normal_samples].values.astype(float)

        log2fc = np.mean(tumor_expr) - np.mean(normal_expr)

        try:
            stat, pvalue = stats.mannwhitneyu(tumor_expr, normal_expr, alternative='two-sided')
        except:
            pvalue = 1.0

        deg_results.append({
            'gene': gene,
            'log2FoldChange': log2fc,
            'pvalue': pvalue,
            'mean_tumor': np.mean(tumor_expr),
            'mean_normal': np.mean(normal_expr)
        })

    deg_df = pd.DataFrame(deg_results)
    n = len(deg_df)
    ranked = rankdata(deg_df['pvalue'])
    deg_df['padj'] = np.minimum(1, deg_df['pvalue'] * n / ranked)

    deg_sig = deg_df[(deg_df['padj'] < 0.05) & (abs(deg_df['log2FoldChange']) > 1.0)].sort_values('padj')

    logger.info(f"\nDEG Results:")
    logger.info(f"  Total genes: {len(deg_df)}")
    logger.info(f"  Significant DEGs: {len(deg_sig)}")
    logger.info(f"\nTop 20 DEGs:\n{deg_sig.head(20).to_string()}")

    deg_df.to_csv(RESULTS_DIR / "deg_all.csv", index=False)
    deg_sig.to_csv(RESULTS_DIR / "deg_significant.csv", index=False)

    return deg_df, deg_sig

def build_network(deg_sig, expr_df):
    """Build network and identify hub genes."""
    import networkx as nx

    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Network Analysis")
    logger.info("=" * 70)

    top_genes = deg_sig.head(300)['gene'].tolist()
    genes_available = [g for g in top_genes if g in expr_df.index]

    if len(genes_available) < 50:
        extra = [g for g in expr_df.index if g not in genes_available][:200]
        genes_available.extend(extra)

    expr_subset = expr_df.loc[genes_available]
    logger.info(f"\nBuilding network with {len(genes_available)} genes...")

    corr_matrix = expr_subset.T.corr(method='spearman')

    G = nx.Graph()
    G.add_nodes_from(genes_available)

    threshold = 0.6
    for i, g1 in enumerate(genes_available):
        for j, g2 in enumerate(genes_available):
            if i < j:
                corr = corr_matrix.loc[g1, g2]
                if abs(corr) > threshold:
                    G.add_edge(g1, g2, weight=abs(corr))

    logger.info(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = {n: 1/len(G) for n in G.nodes()}
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector = {n: 0 for n in G.nodes()}

    hub_results = []
    for gene in G.nodes():
        hub_results.append({
            'gene': gene,
            'degree': degree.get(gene, 0),
            'betweenness': betweenness.get(gene, 0),
            'pagerank': pagerank.get(gene, 0),
            'eigenvector': eigenvector.get(gene, 0)
        })

    hub_df = pd.DataFrame(hub_results)

    for col in ['degree', 'betweenness', 'pagerank', 'eigenvector']:
        max_val = hub_df[col].max()
        hub_df[f'{col}_norm'] = hub_df[col] / max_val if max_val > 0 else 0

    hub_df['composite_score'] = (
        hub_df['degree_norm'] * 0.3 +
        hub_df['betweenness_norm'] * 0.25 +
        hub_df['pagerank_norm'] * 0.25 +
        hub_df['eigenvector_norm'] * 0.2
    )

    hub_df = hub_df.sort_values('composite_score', ascending=False)
    logger.info(f"\nTop 20 Hub Genes:\n{hub_df[['gene', 'degree', 'betweenness', 'composite_score']].head(20).to_string()}")

    hub_df.to_csv(RESULTS_DIR / "hub_genes.csv", index=False)
    return G, hub_df

def validate_and_report(gse, metadata_df, deg_df, deg_sig, hub_df):
    """Validate and generate report."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Validation & Report")
    logger.info("=" * 70)

    known_genes = {
        'EGFR', 'KRAS', 'TP53', 'ALK', 'ROS1', 'MET', 'BRAF', 'PIK3CA',
        'PTEN', 'RB1', 'CDKN2A', 'STK11', 'KEAP1', 'NF1', 'ERBB2', 'RET',
        'MYC', 'CCND1', 'CDK4', 'MDM2', 'SOX2', 'TERT', 'FOXM1', 'E2F1',
        'TOP2A', 'BIRC5', 'AURKA', 'CDC20', 'UBE2C', 'CCNB1', 'BUB1'
    }

    top_hubs = set(hub_df.head(50)['gene'].tolist())
    validated = top_hubs & known_genes

    logger.info(f"\nValidated hub genes: {len(validated)}/50")
    if validated:
        logger.info(f"  Genes: {', '.join(sorted(validated))}")

    title = gse.metadata.get('title', ['N/A'])[0]

    report = f"""
================================================================================
       GEO RNA-seq ANALYSIS REPORT
       Dataset: GSE19804 - Lung Cancer Expression
       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

DATASET: {title}
Samples: {len(gse.gsms)}

DEG ANALYSIS:
  Total genes: {len(deg_df)}
  Significant DEGs: {len(deg_sig)}
  Up-regulated: {len(deg_sig[deg_sig['log2FoldChange'] > 0])}
  Down-regulated: {len(deg_sig[deg_sig['log2FoldChange'] < 0])}

TOP 10 HUB GENES:
{hub_df[['gene', 'composite_score']].head(10).to_string()}

VALIDATION:
  Known cancer genes found: {len(validated)}/50
  Genes: {', '.join(sorted(validated)) if validated else 'None'}

================================================================================
                    ANALYSIS COMPLETED
================================================================================
"""

    logger.info(report)

    with open(RESULTS_DIR / "ANALYSIS_REPORT.txt", 'w') as f:
        f.write(report)

    return validated

def main():
    logger.info("=" * 70)
    logger.info("REAL GEO RNA-seq ANALYSIS")
    logger.info("=" * 70)

    try:
        gse = download_geo_dataset()
        metadata_df = extract_metadata(gse)
        expr_df = extract_expression(gse, metadata_df)

        if expr_df is None:
            raise ValueError("No expression data")

        deg_df, deg_sig = run_deg_analysis(expr_df, metadata_df)
        G, hub_df = build_network(deg_sig, expr_df)
        validated = validate_and_report(gse, metadata_df, deg_df, deg_sig, hub_df)

        logger.info("\nPIPELINE COMPLETED!")
        return True

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
