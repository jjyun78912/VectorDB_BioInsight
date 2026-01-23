"""
Unified External Data Fetcher.

Aggregates data from multiple external APIs to provide comprehensive
gene context for RNA-seq analysis.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    ExternalDataFetcher                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: ["TP53", "BRCA1", "KRAS"]                              │
│              │                                                  │
│      ┌───────┴────────┬────────────┬────────────┐              │
│      ▼                ▼            ▼            ▼              │
│  ┌────────┐    ┌──────────┐  ┌────────┐  ┌──────────┐         │
│  │OncoKB  │    │  CIViC   │  │ STRING │  │ UniProt  │         │
│  │ (암)   │    │(임상근거)│  │ (PPI)  │  │(단백질)  │         │
│  └────┬───┘    └────┬─────┘  └───┬────┘  └────┬─────┘         │
│       │             │            │            │                │
│  ┌────┴───┐    ┌────┴─────┐                                   │
│  │  KEGG  │    │Reactome  │                                   │
│  │(pathway)│    │(pathway) │                                   │
│  └────┬───┘    └────┬─────┘                                   │
│       │             │                                          │
│       └──────┬──────┘                                          │
│              ▼                                                  │
│       Unified Gene Context                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from .oncokb_client import OncoKBClient
from .civic_client import CIViCClient
from .string_client import STRINGClient
from .uniprot_client import UniProtClient
from .kegg_client import KEGGClient
from .reactome_client import ReactomeClient

logger = logging.getLogger(__name__)


@dataclass
class GeneContext:
    """Comprehensive gene context from all external sources."""
    gene_symbol: str

    # Cancer annotations (OncoKB + CIViC)
    is_oncogene: bool = False
    is_tsg: bool = False
    cancer_role: str = "Unknown"
    clinical_evidence_count: int = 0
    actionable: bool = False
    therapeutic_targets: List[str] = field(default_factory=list)

    # Protein function (UniProt)
    protein_function: str = ""
    subcellular_location: List[str] = field(default_factory=list)
    disease_associations: List[str] = field(default_factory=list)

    # Interactions (STRING)
    interaction_partners: List[str] = field(default_factory=list)
    interaction_score: int = 0

    # Pathways (KEGG + Reactome)
    kegg_pathways: List[Dict[str, str]] = field(default_factory=list)
    reactome_pathways: List[Dict[str, str]] = field(default_factory=list)

    # Source tracking
    sources_queried: List[str] = field(default_factory=list)
    sources_success: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_summary(self) -> str:
        """Generate a text summary for RAG."""
        parts = [f"Gene: {self.gene_symbol}"]

        if self.cancer_role != "Unknown":
            parts.append(f"Cancer role: {self.cancer_role}")

        if self.protein_function:
            parts.append(f"Function: {self.protein_function[:200]}")

        if self.interaction_partners:
            parts.append(f"Key interactors: {', '.join(self.interaction_partners[:5])}")

        if self.kegg_pathways:
            pathway_names = [p.get('name', '') for p in self.kegg_pathways[:3]]
            parts.append(f"KEGG pathways: {', '.join(pathway_names)}")

        if self.disease_associations:
            parts.append(f"Disease associations: {', '.join(self.disease_associations[:3])}")

        return ". ".join(parts)


class ExternalDataFetcher:
    """
    Unified fetcher for external biological databases.

    Aggregates data from:
    - OncoKB: Cancer gene annotations
    - CIViC: Clinical evidence
    - STRING: Protein interactions
    - UniProt: Protein function
    - KEGG: Pathway information
    - Reactome: Pathway analysis
    """

    def __init__(
        self,
        cancer_type: Optional[str] = None,
        enable_oncokb: bool = True,
        enable_civic: bool = True,
        enable_string: bool = True,
        enable_uniprot: bool = True,
        enable_kegg: bool = True,
        enable_reactome: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the unified fetcher.

        Args:
            cancer_type: Cancer type for context-specific queries
            enable_*: Enable/disable specific APIs
            cache_dir: Cache directory for API responses
        """
        self.cancer_type = cancer_type
        self.cache_dir = cache_dir

        # Initialize clients
        self.clients = {}

        if enable_oncokb:
            self.clients['oncokb'] = OncoKBClient()
        if enable_civic:
            self.clients['civic'] = CIViCClient()
        if enable_string:
            self.clients['string'] = STRINGClient()
        if enable_uniprot:
            self.clients['uniprot'] = UniProtClient()
        if enable_kegg:
            self.clients['kegg'] = KEGGClient()
        if enable_reactome:
            self.clients['reactome'] = ReactomeClient()

        logger.info(f"ExternalDataFetcher initialized with {len(self.clients)} APIs")

    async def close(self):
        """Close all client sessions."""
        for client in self.clients.values():
            await client.close()

    async def get_gene_context(
        self,
        gene_symbol: str,
        include_interactions: bool = True,
        include_pathways: bool = True
    ) -> GeneContext:
        """
        Get comprehensive context for a single gene.

        Args:
            gene_symbol: Gene symbol (e.g., 'TP53')
            include_interactions: Include STRING interactions
            include_pathways: Include pathway information

        Returns:
            GeneContext with aggregated data
        """
        context = GeneContext(gene_symbol=gene_symbol)
        tasks = []

        # Cancer annotations
        if 'oncokb' in self.clients:
            tasks.append(('oncokb', self.clients['oncokb'].search_gene(gene_symbol)))
            context.sources_queried.append('OncoKB')

        if 'civic' in self.clients:
            tasks.append(('civic', self.clients['civic'].search_gene(gene_symbol)))
            context.sources_queried.append('CIViC')

        # Protein function
        if 'uniprot' in self.clients:
            tasks.append(('uniprot', self.clients['uniprot'].search_gene(gene_symbol)))
            context.sources_queried.append('UniProt')

        # Interactions
        if include_interactions and 'string' in self.clients:
            tasks.append(('string', self.clients['string'].get_interaction_partners(gene_symbol, limit=10)))
            context.sources_queried.append('STRING')

        # Pathways
        if include_pathways:
            if 'kegg' in self.clients:
                tasks.append(('kegg', self.clients['kegg'].search_gene(gene_symbol)))
                context.sources_queried.append('KEGG')

            if 'reactome' in self.clients:
                tasks.append(('reactome', self.clients['reactome'].search_gene(gene_symbol)))
                context.sources_queried.append('Reactome')

        # Execute all tasks concurrently
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True
            )

            for (name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    logger.warning(f"{name} error for {gene_symbol}: {result}")
                else:
                    results[name] = result

        # Aggregate results
        self._aggregate_results(context, results)

        return context

    def _aggregate_results(self, context: GeneContext, results: Dict[str, Any]):
        """Aggregate results from all APIs into GeneContext."""

        # OncoKB
        if 'oncokb' in results and results['oncokb'].success:
            data = results['oncokb'].data
            context.is_oncogene = data.get('oncogene', False)
            context.is_tsg = data.get('tsg', False)

            if context.is_oncogene and context.is_tsg:
                context.cancer_role = "Oncogene/TSG (context-dependent)"
            elif context.is_oncogene:
                context.cancer_role = "Oncogene"
            elif context.is_tsg:
                context.cancer_role = "Tumor Suppressor"

            context.actionable = data.get('has_actionable_alterations', False)
            context.sources_success.append('OncoKB')

        # CIViC
        if 'civic' in results and results['civic'].success:
            data = results['civic'].data
            context.clinical_evidence_count = data.get('variants_count', 0)
            if data.get('description'):
                context.protein_function = context.protein_function or data['description'][:300]
            context.sources_success.append('CIViC')

        # UniProt
        if 'uniprot' in results and results['uniprot'].success:
            data = results['uniprot'].data
            if data.get('function'):
                context.protein_function = data['function']
            context.subcellular_location = data.get('subcellular_location', [])
            context.disease_associations = data.get('disease_associations', [])
            context.sources_success.append('UniProt')

        # STRING
        if 'string' in results and results['string'].success:
            data = results['string'].data
            if isinstance(data, list):
                context.interaction_partners = [p.get('partner', '') for p in data[:10]]
                if data:
                    context.interaction_score = data[0].get('score', 0)
            context.sources_success.append('STRING')

        # KEGG
        if 'kegg' in results and results['kegg'].success:
            data = results['kegg'].data
            context.kegg_pathways = data.get('pathways', [])[:10]
            context.sources_success.append('KEGG')

        # Reactome
        if 'reactome' in results and results['reactome'].success:
            data = results['reactome'].data
            context.reactome_pathways = data.get('pathways', [])[:10]
            context.sources_success.append('Reactome')

    async def get_batch_context(
        self,
        gene_symbols: List[str],
        include_interactions: bool = True,
        include_pathways: bool = True,
        max_concurrent: int = 5
    ) -> Dict[str, GeneContext]:
        """
        Get context for multiple genes with concurrency control.

        Args:
            gene_symbols: List of gene symbols
            include_interactions: Include STRING interactions
            include_pathways: Include pathway information
            max_concurrent: Maximum concurrent requests

        Returns:
            Dict mapping gene symbol to GeneContext
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(gene: str) -> tuple:
            async with semaphore:
                context = await self.get_gene_context(
                    gene, include_interactions, include_pathways
                )
                return gene, context

        tasks = [fetch_with_limit(gene) for gene in gene_symbols]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Batch fetch error: {result}")
            else:
                gene, context = result
                results[gene] = context

        return results

    async def get_ppi_network(
        self,
        gene_symbols: List[str],
        score_threshold: int = 700
    ) -> Dict[str, Any]:
        """
        Get protein-protein interaction network for gene list.

        Args:
            gene_symbols: List of gene symbols
            score_threshold: Minimum STRING score

        Returns:
            Dict with nodes and edges
        """
        if 'string' not in self.clients:
            return {"nodes": [], "edges": []}

        response = await self.clients['string'].get_interactions(
            gene_symbols, score_threshold
        )

        if not response.success:
            return {"nodes": [], "edges": []}

        # Build network structure
        nodes = set()
        edges = []

        for interaction in response.data:
            protein_a = interaction.get('protein_a', '')
            protein_b = interaction.get('protein_b', '')
            score = interaction.get('combined_score', 0)

            nodes.add(protein_a)
            nodes.add(protein_b)
            edges.append({
                "source": protein_a,
                "target": protein_b,
                "weight": score / 1000.0
            })

        return {
            "nodes": list(nodes),
            "edges": edges,
            "source": "STRING"
        }

    async def get_pathway_enrichment(
        self,
        gene_symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Get pathway enrichment from Reactome.

        Args:
            gene_symbols: List of gene symbols

        Returns:
            Dict with enrichment results
        """
        if 'reactome' not in self.clients:
            return {"pathways": []}

        response = await self.clients['reactome'].analyze_gene_list(gene_symbols)

        if response.success:
            return response.data

        return {"pathways": []}

    def generate_rag_context(
        self,
        gene_contexts: Dict[str, GeneContext]
    ) -> str:
        """
        Generate text context for RAG from gene contexts.

        Args:
            gene_contexts: Dict of gene symbols to GeneContext

        Returns:
            Formatted text for RAG augmentation
        """
        sections = []

        for gene, context in gene_contexts.items():
            sections.append(f"### {gene}")
            sections.append(context.get_summary())
            sections.append("")

        return "\n".join(sections)

    def save_context(
        self,
        gene_contexts: Dict[str, GeneContext],
        output_path: Path
    ):
        """Save gene contexts to JSON file."""
        data = {
            gene: context.to_dict()
            for gene, context in gene_contexts.items()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(data)} gene contexts to {output_path}")


def get_external_fetcher(
    cancer_type: Optional[str] = None,
    **kwargs
) -> ExternalDataFetcher:
    """Factory function to create ExternalDataFetcher."""
    return ExternalDataFetcher(cancer_type=cancer_type, **kwargs)


# Synchronous wrapper for non-async contexts
def fetch_gene_context_sync(
    gene_symbols: List[str],
    cancer_type: Optional[str] = None,
    **kwargs
) -> Dict[str, GeneContext]:
    """
    Synchronous wrapper for fetching gene context.

    Args:
        gene_symbols: List of gene symbols
        cancer_type: Cancer type for context

    Returns:
        Dict mapping gene symbol to GeneContext
    """
    async def _fetch():
        fetcher = ExternalDataFetcher(cancer_type=cancer_type, **kwargs)
        try:
            return await fetcher.get_batch_context(gene_symbols)
        finally:
            await fetcher.close()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(_fetch())
