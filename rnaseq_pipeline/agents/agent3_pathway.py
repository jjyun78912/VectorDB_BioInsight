"""
Agent 3: Pathway Enrichment Analysis

Performs GO (Gene Ontology) and KEGG pathway enrichment analysis on DEGs.

Input:
- deg_significant.csv: From Agent 1
- config.json: Analysis parameters

Output:
- pathway_go_bp.csv: GO Biological Process results
- pathway_go_mf.csv: GO Molecular Function results
- pathway_go_cc.csv: GO Cellular Component results
- pathway_kegg.csv: KEGG pathway results
- pathway_summary.csv: Top pathways from all databases
- gene_to_pathway.csv: Gene to pathway mapping
- meta_agent3.json: Execution metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

from ..utils.base_agent import BaseAgent


class PathwayAgent(BaseAgent):
    """Agent for GO/KEGG pathway enrichment analysis."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "organism": "human",  # human or mouse
            "databases": ["GO_Biological_Process_2023", "GO_Molecular_Function_2023",
                         "GO_Cellular_Component_2023", "KEGG_2021_Human"],
            "pvalue_cutoff": 0.05,
            "min_genes": 3,
            "top_terms": 20,  # Top terms to report per database
            "background": None  # Custom background gene list
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent3_pathway", input_dir, output_dir, merged_config)

        self.deg_significant: Optional[pd.DataFrame] = None

    def validate_inputs(self) -> bool:
        """Validate DEG input."""
        if not HAS_GSEAPY:
            self.logger.error("gseapy not installed. Install with: pip install gseapy")
            return False

        self.deg_significant = self.load_csv("deg_significant.csv")
        if self.deg_significant is None:
            return False

        if len(self.deg_significant) < self.config["min_genes"]:
            self.logger.error(f"Need at least {self.config['min_genes']} DEGs for pathway analysis")
            return False

        self.logger.info(f"DEGs for pathway analysis: {len(self.deg_significant)}")

        return True

    def _convert_gene_ids(self, gene_list: List) -> List[str]:
        """Convert gene IDs to symbols if needed."""
        # Convert all to strings first
        gene_list = [str(g) for g in gene_list]

        # Check if these are Entrez IDs (all numeric)
        if all(g.isdigit() for g in gene_list):
            self.logger.info("Detected Entrez IDs - converting to Gene Symbols via mygene...")
            try:
                import mygene
                mg = mygene.MyGeneInfo()
                # Query in batches
                results = mg.querymany(gene_list, scopes='entrezgene', fields='symbol', species='human', verbose=False)

                # Build mapping
                entrez_to_symbol = {}
                for r in results:
                    if 'symbol' in r:
                        entrez_to_symbol[str(r['query'])] = r['symbol']

                # Convert
                converted = [entrez_to_symbol.get(g, g) for g in gene_list]
                # Filter out numeric ones (failed conversions)
                converted = [g for g in converted if not g.isdigit()]

                self.logger.info(f"Converted {len(converted)}/{len(gene_list)} Entrez IDs to Gene Symbols")
                return converted
            except Exception as e:
                self.logger.warning(f"mygene conversion failed: {e}. Using Entrez IDs directly.")
                return gene_list

        # If already symbols (no dots, not ENSG), return as is
        if all(not g.startswith('ENSG') for g in gene_list):
            return gene_list

        # Would need conversion via biomart or similar
        # For now, assume they're symbols or return original
        self.logger.warning("Gene ID conversion not implemented - using original IDs")
        return gene_list

    def _run_enrichr(self, gene_list: List[str], database: str) -> Optional[pd.DataFrame]:
        """Run Enrichr enrichment analysis for a single database."""
        self.logger.info(f"Running enrichment for {database}...")

        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=database,
                organism=self.config["organism"],
                outdir=None,  # Don't save files
                cutoff=self.config["pvalue_cutoff"],
                no_plot=True
            )

            results = enr.results

            if len(results) == 0:
                self.logger.warning(f"No significant results for {database}")
                return None

            # Filter by adjusted p-value
            results = results[results['Adjusted P-value'] < self.config["pvalue_cutoff"]]

            # Filter by minimum genes
            results = results[results['Overlap'].apply(
                lambda x: int(x.split('/')[0]) >= self.config["min_genes"]
            )]

            # Standardize column names
            results = results.rename(columns={
                'Term': 'term_name',
                'Adjusted P-value': 'padj',
                'P-value': 'pvalue',
                'Odds Ratio': 'odds_ratio',
                'Combined Score': 'combined_score',
                'Overlap': 'overlap',
                'Genes': 'genes'
            })

            # Extract gene count
            results['gene_count'] = results['overlap'].apply(
                lambda x: int(x.split('/')[0])
            )

            # Add database column
            results['database'] = database

            # Sort by padj
            results = results.sort_values('padj')

            return results

        except Exception as e:
            self.logger.error(f"Enrichr failed for {database}: {e}")
            return None

    def _create_gene_to_pathway_mapping(self, all_results: pd.DataFrame) -> pd.DataFrame:
        """Create reverse mapping from genes to pathways."""
        gene_pathways = {}

        for _, row in all_results.iterrows():
            genes = row['genes'].split(';')
            term = row['term_name']
            db = row['database']

            for gene in genes:
                gene = gene.strip()
                if gene not in gene_pathways:
                    gene_pathways[gene] = {
                        'pathway_ids': [],
                        'pathway_names': [],
                        'databases': []
                    }
                gene_pathways[gene]['pathway_ids'].append(f"{db}:{term}")
                gene_pathways[gene]['pathway_names'].append(term)
                gene_pathways[gene]['databases'].append(db)

        # Convert to DataFrame
        rows = []
        for gene, info in gene_pathways.items():
            rows.append({
                'gene_id': gene,
                'pathway_count': len(info['pathway_names']),
                'pathway_ids': ';'.join(info['pathway_ids']),
                'pathway_names': ';'.join(info['pathway_names']),
                'databases': ';'.join(set(info['databases']))
            })

        return pd.DataFrame(rows)

    def run(self) -> Dict[str, Any]:
        """Execute pathway enrichment analysis."""
        # Get gene list
        gene_list = self.deg_significant['gene_id'].tolist()
        gene_list = self._convert_gene_ids(gene_list)

        self.logger.info(f"Running pathway enrichment for {len(gene_list)} genes")

        # Run enrichment for each database
        all_results = []
        db_counts = {}

        databases = self.config["databases"]

        # Map database names to output files
        db_file_map = {
            "GO_Biological_Process_2023": "pathway_go_bp.csv",
            "GO_Molecular_Function_2023": "pathway_go_mf.csv",
            "GO_Cellular_Component_2023": "pathway_go_cc.csv",
            "KEGG_2021_Human": "pathway_kegg.csv",
            "KEGG_2019_Human": "pathway_kegg.csv",
            "Reactome_2022": "pathway_reactome.csv"
        }

        for db in databases:
            results = self._run_enrichr(gene_list, db)

            if results is not None and len(results) > 0:
                all_results.append(results)
                db_counts[db] = len(results)

                # Save individual database results
                output_file = db_file_map.get(db, f"pathway_{db.lower()}.csv")
                top_results = results.head(self.config["top_terms"])
                self.save_csv(top_results, output_file)
            else:
                db_counts[db] = 0
                # Create empty file
                output_file = db_file_map.get(db, f"pathway_{db.lower()}.csv")
                empty_df = pd.DataFrame(columns=[
                    'term_name', 'pvalue', 'padj', 'odds_ratio',
                    'combined_score', 'overlap', 'genes', 'gene_count', 'database'
                ])
                self.save_csv(empty_df, output_file)

        # Combine all results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)

            # Create summary (top from each database)
            summary_rows = []
            for db in databases:
                db_results = combined[combined['database'] == db]
                if len(db_results) > 0:
                    top = db_results.head(5)
                    summary_rows.append(top)

            if summary_rows:
                summary = pd.concat(summary_rows, ignore_index=True)
                summary = summary.sort_values('padj')
                self.save_csv(
                    summary[['database', 'term_name', 'padj', 'gene_count', 'genes']],
                    "pathway_summary.csv"
                )
            else:
                self.save_csv(pd.DataFrame(), "pathway_summary.csv")

            # Create gene to pathway mapping
            gene_pathway_map = self._create_gene_to_pathway_mapping(combined)
            self.save_csv(gene_pathway_map, "gene_to_pathway.csv")

        else:
            self.logger.warning("No significant pathways found in any database")
            self.save_csv(pd.DataFrame(), "pathway_summary.csv")
            self.save_csv(pd.DataFrame(columns=['gene_id', 'pathway_count', 'pathway_ids', 'pathway_names', 'databases']),
                         "gene_to_pathway.csv")

        # Calculate statistics
        total_significant = sum(db_counts.values())

        self.logger.info(f"Pathway Analysis Complete:")
        self.logger.info(f"  Databases analyzed: {len(databases)}")
        for db, count in db_counts.items():
            self.logger.info(f"    {db}: {count} significant terms")
        self.logger.info(f"  Total significant terms: {total_significant}")

        return {
            "databases_analyzed": databases,
            "significant_terms_per_db": db_counts,
            "total_significant_terms": total_significant,
            "pvalue_cutoff": self.config["pvalue_cutoff"],
            "min_genes": self.config["min_genes"]
        }

    def validate_outputs(self) -> bool:
        """Validate pathway outputs."""
        # Check summary file exists
        summary_file = self.output_dir / "pathway_summary.csv"
        if not summary_file.exists():
            self.logger.error("Missing pathway_summary.csv")
            return False

        # Check gene mapping file exists
        mapping_file = self.output_dir / "gene_to_pathway.csv"
        if not mapping_file.exists():
            self.logger.error("Missing gene_to_pathway.csv")
            return False

        # At least one database should have results (warning if not)
        summary = pd.read_csv(summary_file)
        if len(summary) == 0:
            self.logger.warning("No significant pathways found - this may be expected for some datasets")

        return True
