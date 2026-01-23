"""
Reactome API Client.

Reactome is a free, open-source, curated and peer-reviewed
pathway database.

Features:
- Pathway analysis
- Pathway hierarchy
- Reaction information
- Interactor data

API Documentation: https://reactome.org/ContentService/
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


@dataclass
class ReactomePathway:
    """Reactome pathway information."""
    stable_id: str
    name: str
    species: str = "Homo sapiens"
    summary: str = ""
    diagram_available: bool = False
    has_ehld: bool = False  # Enhanced High Level Diagram
    sub_pathways: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReactomeGeneInfo:
    """Reactome gene/entity information."""
    stable_id: str
    gene_symbol: str
    name: str = ""
    pathways: List[Dict[str, str]] = field(default_factory=list)
    reactions: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReactomeClient(BaseAPIClient):
    """
    Client for Reactome Content Service API.

    Provides access to curated pathway data.
    No API key required.
    """

    BASE_URL = "https://reactome.org/ContentService"
    API_NAME = "Reactome"
    RATE_LIMIT_DELAY = 0.3

    # Reactome species ID for human
    SPECIES_HUMAN = 48887

    def __init__(self, **kwargs):
        """Initialize Reactome client. No API key required."""
        super().__init__(**kwargs)

    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """
        Search for gene in Reactome.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with ReactomeGeneInfo
        """
        cache_key = self._get_cache_key("search", {"gene": gene_symbol})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=gene_symbol,
                cached=True
            )

        # Search for the gene
        response = await self.get(
            "search/query",
            params={
                "query": gene_symbol,
                "species": "Homo sapiens",
                "types": "Protein",
                "cluster": "true"
            }
        )

        if response.success and response.data:
            results = response.data.get("results", [])
            if results:
                # Find best match
                best_match = None
                for result in results:
                    entries = result.get("entries", [])
                    for entry in entries:
                        if gene_symbol.upper() in entry.get("name", "").upper():
                            best_match = entry
                            break
                    if best_match:
                        break

                if not best_match and results:
                    best_match = results[0].get("entries", [{}])[0]

                if best_match:
                    stable_id = best_match.get("stId", "")

                    # Get pathways for this entity
                    pathways_response = await self.get_pathways_for_entity(stable_id)
                    pathways = pathways_response.data if pathways_response.success else []

                    gene_info = ReactomeGeneInfo(
                        stable_id=stable_id,
                        gene_symbol=gene_symbol,
                        name=best_match.get("name", ""),
                        pathways=[{
                            "id": p.get("stId", ""),
                            "name": p.get("displayName", "")
                        } for p in pathways[:20]]
                    )

                    result = gene_info.to_dict()
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
            error="Gene not found in Reactome"
        )

    async def batch_search(self, gene_symbols: List[str]) -> Dict[str, APIResponse]:
        """Batch search for multiple genes."""
        results = {}
        for gene in gene_symbols:
            results[gene] = await self.search_gene(gene)
        return results

    async def get_pathway(self, pathway_id: str) -> APIResponse:
        """
        Get pathway information.

        Args:
            pathway_id: Reactome pathway stable ID (e.g., 'R-HSA-109582')

        Returns:
            APIResponse with ReactomePathway
        """
        cache_key = self._get_cache_key("pathway", {"id": pathway_id})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=pathway_id,
                cached=True
            )

        response = await self.get(f"data/pathway/{pathway_id}")

        if response.success and response.data:
            data = response.data

            pathway = ReactomePathway(
                stable_id=data.get("stId", pathway_id),
                name=data.get("displayName", ""),
                species=data.get("speciesName", "Homo sapiens"),
                summary=data.get("summation", [{}])[0].get("text", "")[:500] if data.get("summation") else "",
                diagram_available=data.get("hasDiagram", False),
                has_ehld=data.get("hasEHLD", False),
                sub_pathways=[
                    p.get("displayName", "") for p in data.get("hasEvent", [])[:10]
                ]
            )

            result = pathway.to_dict()
            self._write_cache(cache_key, result)

            return APIResponse(
                success=True,
                data=result,
                source=self.API_NAME,
                query=pathway_id
            )

        return APIResponse(
            success=False,
            data=None,
            source=self.API_NAME,
            query=pathway_id,
            error="Pathway not found"
        )

    async def get_pathways_for_entity(self, entity_id: str) -> APIResponse:
        """
        Get pathways containing an entity.

        Args:
            entity_id: Reactome entity stable ID

        Returns:
            APIResponse with list of pathways
        """
        response = await self.get(
            f"data/pathways/low/entity/{entity_id}",
            params={"species": self.SPECIES_HUMAN}
        )

        if response.success and response.data:
            return APIResponse(
                success=True,
                data=response.data,
                source=self.API_NAME,
                query=entity_id
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=entity_id,
            error="No pathways found"
        )

    async def analyze_gene_list(
        self,
        gene_symbols: List[str],
        include_interactors: bool = False
    ) -> APIResponse:
        """
        Perform pathway enrichment analysis.

        Args:
            gene_symbols: List of gene symbols
            include_interactors: Include protein interactors

        Returns:
            APIResponse with enrichment results
        """
        if not gene_symbols:
            return APIResponse(
                success=False,
                data=None,
                source=self.API_NAME,
                query="analysis",
                error="No genes provided"
            )

        session = await self._get_session()
        await self._rate_limit()

        # Reactome analysis expects gene list as text
        gene_list = "\n".join(gene_symbols)

        try:
            async with session.post(
                f"{self.BASE_URL}/identifiers/projection",
                data=gene_list,
                headers={"Content-Type": "text/plain"},
                params={
                    "interactors": str(include_interactors).lower(),
                    "pageSize": 50,
                    "page": 1,
                    "sortBy": "ENTITIES_PVALUE",
                    "order": "ASC"
                }
            ) as resp:
                if resp.status != 200:
                    return APIResponse(
                        success=False,
                        data=None,
                        source=self.API_NAME,
                        query="analysis",
                        error=f"Analysis failed: {resp.status}"
                    )

                data = await resp.json()

                # Extract pathway results
                pathways = []
                for pathway in data.get("pathways", [])[:30]:
                    entities = pathway.get("entities", {})
                    pathways.append({
                        "pathway_id": pathway.get("stId", ""),
                        "name": pathway.get("name", ""),
                        "p_value": entities.get("pValue", 1.0),
                        "fdr": entities.get("fdr", 1.0),
                        "found": entities.get("found", 0),
                        "total": entities.get("total", 0),
                        "ratio": entities.get("ratio", 0)
                    })

                return APIResponse(
                    success=True,
                    data={
                        "summary": data.get("summary", {}),
                        "pathways": pathways
                    },
                    source=self.API_NAME,
                    query=f"analysis:{len(gene_symbols)} genes"
                )

        except Exception as e:
            logger.error(f"Reactome analysis error: {e}")
            return APIResponse(
                success=False,
                data=None,
                source=self.API_NAME,
                query="analysis",
                error=str(e)
            )

    async def get_top_level_pathways(self) -> APIResponse:
        """Get top-level Reactome pathways."""
        response = await self.get(
            "data/pathways/top/9606"  # Human
        )

        if response.success and response.data:
            pathways = [
                {
                    "id": p.get("stId", ""),
                    "name": p.get("displayName", "")
                }
                for p in response.data
            ]
            return APIResponse(
                success=True,
                data=pathways,
                source=self.API_NAME,
                query="top_level_pathways"
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query="top_level_pathways",
            error="Failed to get pathways"
        )

    def get_pathway_diagram_url(self, pathway_id: str) -> str:
        """Get URL for pathway diagram."""
        return f"https://reactome.org/PathwayBrowser/#/{pathway_id}"


def get_reactome_client() -> ReactomeClient:
    """Factory function to create Reactome client."""
    return ReactomeClient()
