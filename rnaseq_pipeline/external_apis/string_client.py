"""
STRING API Client.

STRING (Search Tool for the Retrieval of Interacting Genes/Proteins)
provides protein-protein interaction networks.

Features:
- Protein-protein interactions
- Functional enrichment
- Network analysis
- Interaction scores

API Documentation: https://string-db.org/help/api/
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


@dataclass
class StringInteraction:
    """STRING protein-protein interaction."""
    protein_a: str
    protein_b: str
    combined_score: int  # 0-1000
    experimental_score: int = 0
    database_score: int = 0
    textmining_score: int = 0
    coexpression_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StringProteinInfo:
    """STRING protein information."""
    protein_id: str
    gene_symbol: str
    annotation: str = ""
    preferred_name: str = ""
    taxon_id: int = 9606  # Human

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class STRINGClient(BaseAPIClient):
    """
    Client for STRING database API.

    Provides access to protein-protein interaction networks.
    No API key required for basic usage.
    """

    BASE_URL = "https://string-db.org/api"
    API_NAME = "STRING"
    RATE_LIMIT_DELAY = 0.5

    # Default species
    SPECIES_HUMAN = 9606

    # Score thresholds
    SCORE_HIGHEST = 900
    SCORE_HIGH = 700
    SCORE_MEDIUM = 400
    SCORE_LOW = 150

    def __init__(self, species: int = 9606, **kwargs):
        """
        Initialize STRING client.

        Args:
            species: NCBI taxonomy ID (default: 9606 for human)
        """
        super().__init__(**kwargs)
        self.species = species

    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """
        Get protein information for a gene.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with protein information
        """
        cache_key = self._get_cache_key("get_string_ids", {"identifier": gene_symbol})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=gene_symbol,
                cached=True
            )

        response = await self.get(
            f"json/get_string_ids",
            params={
                "identifiers": gene_symbol,
                "species": self.species,
                "limit": 1
            }
        )

        if response.success and response.data:
            protein_data = response.data[0] if isinstance(response.data, list) and response.data else None

            if protein_data:
                info = StringProteinInfo(
                    protein_id=protein_data.get("stringId", ""),
                    gene_symbol=gene_symbol,
                    annotation=protein_data.get("annotation", ""),
                    preferred_name=protein_data.get("preferredName", gene_symbol),
                    taxon_id=protein_data.get("ncbiTaxonId", self.species)
                )
                result = info.to_dict()
                self._write_cache(cache_key, result)

                return APIResponse(
                    success=True,
                    data=result,
                    source=self.API_NAME,
                    query=gene_symbol
                )

        return APIResponse(
            success=False,
            data=None,
            source=self.API_NAME,
            query=gene_symbol,
            error="Protein not found in STRING"
        )

    async def batch_search(self, gene_symbols: List[str]) -> Dict[str, APIResponse]:
        """Batch search for multiple genes."""
        results = {}
        for gene in gene_symbols:
            results[gene] = await self.search_gene(gene)
        return results

    async def get_interactions(
        self,
        gene_symbols: List[str],
        score_threshold: int = 400
    ) -> APIResponse:
        """
        Get protein-protein interactions for a list of genes.

        Args:
            gene_symbols: List of gene symbols
            score_threshold: Minimum combined score (0-1000)

        Returns:
            APIResponse with list of interactions
        """
        if not gene_symbols:
            return APIResponse(
                success=False,
                data=[],
                source=self.API_NAME,
                query="interactions",
                error="No genes provided"
            )

        identifiers = "%0d".join(gene_symbols)  # STRING uses %0d for newline

        cache_key = self._get_cache_key(
            "network",
            {"identifiers": identifiers, "score": score_threshold}
        )
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=f"interactions:{len(gene_symbols)} genes",
                cached=True
            )

        response = await self.get(
            "json/network",
            params={
                "identifiers": identifiers,
                "species": self.species,
                "required_score": score_threshold
            }
        )

        if response.success and response.data:
            interactions = []
            for item in response.data:
                interaction = StringInteraction(
                    protein_a=item.get("preferredName_A", item.get("stringId_A", "")),
                    protein_b=item.get("preferredName_B", item.get("stringId_B", "")),
                    combined_score=int(item.get("score", 0) * 1000),
                    experimental_score=int(item.get("escore", 0) * 1000),
                    database_score=int(item.get("dscore", 0) * 1000),
                    textmining_score=int(item.get("tscore", 0) * 1000),
                    coexpression_score=int(item.get("ascore", 0) * 1000)
                )
                interactions.append(interaction.to_dict())

            self._write_cache(cache_key, interactions)

            return APIResponse(
                success=True,
                data=interactions,
                source=self.API_NAME,
                query=f"interactions:{len(gene_symbols)} genes"
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=f"interactions:{len(gene_symbols)} genes",
            error="Failed to get interactions"
        )

    async def get_enrichment(
        self,
        gene_symbols: List[str],
        category: str = "Process"
    ) -> APIResponse:
        """
        Get functional enrichment for a list of genes.

        Args:
            gene_symbols: List of gene symbols
            category: Enrichment category (Process, Component, Function, KEGG, etc.)

        Returns:
            APIResponse with enrichment results
        """
        if not gene_symbols:
            return APIResponse(
                success=False,
                data=[],
                source=self.API_NAME,
                query="enrichment",
                error="No genes provided"
            )

        identifiers = "%0d".join(gene_symbols)

        response = await self.get(
            "json/enrichment",
            params={
                "identifiers": identifiers,
                "species": self.species
            }
        )

        if response.success and response.data:
            # Filter by category if specified
            enrichment = []
            for item in response.data:
                if category.lower() in item.get("category", "").lower() or category == "all":
                    enrichment.append({
                        "term": item.get("term", ""),
                        "description": item.get("description", ""),
                        "category": item.get("category", ""),
                        "fdr": item.get("fdr", 1.0),
                        "p_value": item.get("p_value", 1.0),
                        "gene_count": item.get("number_of_genes", 0),
                        "genes": item.get("inputGenes", "").split(",")
                    })

            # Sort by FDR
            enrichment.sort(key=lambda x: x["fdr"])

            return APIResponse(
                success=True,
                data=enrichment[:50],  # Top 50 terms
                source=self.API_NAME,
                query=f"enrichment:{len(gene_symbols)} genes"
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=f"enrichment:{len(gene_symbols)} genes",
            error="Failed to get enrichment"
        )

    async def get_interaction_partners(
        self,
        gene_symbol: str,
        limit: int = 20,
        score_threshold: int = 700
    ) -> APIResponse:
        """
        Get interaction partners for a single gene.

        Args:
            gene_symbol: Gene symbol
            limit: Maximum number of partners
            score_threshold: Minimum interaction score

        Returns:
            APIResponse with list of interacting proteins
        """
        response = await self.get(
            "json/interaction_partners",
            params={
                "identifiers": gene_symbol,
                "species": self.species,
                "limit": limit,
                "required_score": score_threshold
            }
        )

        if response.success and response.data:
            partners = []
            for item in response.data:
                # Get the partner (not the query gene)
                partner = item.get("preferredName_B")
                if partner == gene_symbol:
                    partner = item.get("preferredName_A")

                partners.append({
                    "partner": partner,
                    "score": int(item.get("score", 0) * 1000),
                    "experimental": int(item.get("escore", 0) * 1000),
                    "database": int(item.get("dscore", 0) * 1000)
                })

            return APIResponse(
                success=True,
                data=partners,
                source=self.API_NAME,
                query=gene_symbol
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=gene_symbol,
            error="No interaction partners found"
        )

    def get_network_url(self, gene_symbols: List[str]) -> str:
        """Get URL for STRING network visualization."""
        identifiers = "%0d".join(gene_symbols)
        return f"https://string-db.org/api/svg/network?identifiers={identifiers}&species={self.species}"


def get_string_client() -> STRINGClient:
    """Factory function to create STRING client."""
    return STRINGClient()
