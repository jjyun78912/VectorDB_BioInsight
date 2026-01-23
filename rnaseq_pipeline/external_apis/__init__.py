"""
External API Clients for RNA-seq Analysis Pipeline.

Provides real-time access to biological databases:

1. Cancer-specific:
   - OncoKB: Actionable mutations, therapeutic implications
   - CIViC: Clinical interpretations of variants
   - cBioPortal: Cancer genomics data

2. Gene Information:
   - STRING: Protein-protein interactions
   - UniProt: Protein function and annotation
   - NCBI Gene: Gene information and references

3. Pathway Information:
   - KEGG: Pathway maps and gene associations
   - Reactome: Pathway analysis and visualization

Usage:
    from rnaseq_pipeline.external_apis import ExternalDataFetcher

    fetcher = ExternalDataFetcher(cancer_type="BRCA")
    gene_info = await fetcher.get_gene_context(["TP53", "BRCA1", "ERBB2"])
"""

from .oncokb_client import OncoKBClient
from .civic_client import CIViCClient
from .string_client import STRINGClient
from .uniprot_client import UniProtClient
from .kegg_client import KEGGClient
from .reactome_client import ReactomeClient
from .unified_fetcher import (
    ExternalDataFetcher,
    GeneContext,
    get_external_fetcher
)

__all__ = [
    # Individual clients
    "OncoKBClient",
    "CIViCClient",
    "STRINGClient",
    "UniProtClient",
    "KEGGClient",
    "ReactomeClient",
    # Unified fetcher
    "ExternalDataFetcher",
    "GeneContext",
    "get_external_fetcher",
]
