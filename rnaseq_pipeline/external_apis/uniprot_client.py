"""
UniProt API Client.

UniProt is a comprehensive resource for protein sequence and
functional information.

Features:
- Protein function and annotation
- GO terms
- Subcellular location
- Disease associations
- Protein families

API Documentation: https://www.uniprot.org/help/api
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


@dataclass
class UniProtEntry:
    """UniProt protein entry."""
    accession: str
    gene_symbol: str
    protein_name: str = ""
    organism: str = "Homo sapiens"
    function: str = ""
    subcellular_location: List[str] = field(default_factory=list)
    go_terms: List[Dict[str, str]] = field(default_factory=list)
    disease_associations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sequence_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UniProtClient(BaseAPIClient):
    """
    Client for UniProt REST API.

    Provides access to protein annotations including:
    - Function descriptions
    - GO terms
    - Disease associations
    - Subcellular localization

    No API key required.
    """

    BASE_URL = "https://rest.uniprot.org"
    API_NAME = "UniProt"
    RATE_LIMIT_DELAY = 0.3

    # UniProt query fields for human proteins
    DEFAULT_FIELDS = [
        "accession", "gene_names", "protein_name", "organism_name",
        "cc_function", "cc_subcellular_location", "go", "cc_disease",
        "keyword", "length"
    ]

    def __init__(self, **kwargs):
        """Initialize UniProt client. No API key required."""
        super().__init__(**kwargs)

    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """
        Search for protein by gene symbol.

        Args:
            gene_symbol: Gene symbol (e.g., 'TP53')

        Returns:
            APIResponse with UniProtEntry data
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

        # Search for human protein with exact gene match
        query = f"(gene_exact:{gene_symbol}) AND (organism_id:9606) AND (reviewed:true)"
        fields = ",".join(self.DEFAULT_FIELDS)

        response = await self.get(
            "uniprotkb/search",
            params={
                "query": query,
                "fields": fields,
                "format": "json",
                "size": 1
            }
        )

        if response.success and response.data:
            results = response.data.get("results", [])
            if results:
                entry_data = results[0]

                # Parse GO terms
                go_terms = []
                for go in entry_data.get("uniProtKBCrossReferences", []):
                    if go.get("database") == "GO":
                        go_terms.append({
                            "id": go.get("id", ""),
                            "term": go.get("properties", [{}])[0].get("value", "") if go.get("properties") else ""
                        })

                # Parse function comment
                function_text = ""
                for comment in entry_data.get("comments", []):
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            function_text = texts[0].get("value", "")
                            break

                # Parse subcellular location
                locations = []
                for comment in entry_data.get("comments", []):
                    if comment.get("commentType") == "SUBCELLULAR LOCATION":
                        for loc in comment.get("subcellularLocations", []):
                            location = loc.get("location", {}).get("value", "")
                            if location:
                                locations.append(location)

                # Parse disease associations
                diseases = []
                for comment in entry_data.get("comments", []):
                    if comment.get("commentType") == "DISEASE":
                        disease = comment.get("disease", {}).get("diseaseId", "")
                        if disease:
                            diseases.append(disease)

                # Parse keywords
                keywords = [kw.get("name", "") for kw in entry_data.get("keywords", [])]

                entry = UniProtEntry(
                    accession=entry_data.get("primaryAccession", ""),
                    gene_symbol=gene_symbol,
                    protein_name=entry_data.get("proteinDescription", {}).get(
                        "recommendedName", {}
                    ).get("fullName", {}).get("value", ""),
                    organism=entry_data.get("organism", {}).get("scientificName", ""),
                    function=function_text[:500] if function_text else "",
                    subcellular_location=locations[:5],
                    go_terms=go_terms[:20],
                    disease_associations=diseases[:10],
                    keywords=keywords[:15],
                    sequence_length=entry_data.get("sequence", {}).get("length", 0)
                )

                result = entry.to_dict()
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
            error="Protein not found in UniProt"
        )

    async def batch_search(self, gene_symbols: List[str]) -> Dict[str, APIResponse]:
        """Batch search for multiple genes."""
        results = {}
        for gene in gene_symbols:
            results[gene] = await self.search_gene(gene)
        return results

    async def get_protein_function(self, gene_symbol: str) -> Optional[str]:
        """
        Get protein function description.

        Args:
            gene_symbol: Gene symbol

        Returns:
            Function description string or None
        """
        response = await self.search_gene(gene_symbol)
        if response.success and response.data:
            return response.data.get("function", "")
        return None

    async def get_go_terms(
        self,
        gene_symbol: str,
        category: Optional[str] = None
    ) -> APIResponse:
        """
        Get GO terms for a gene.

        Args:
            gene_symbol: Gene symbol
            category: Filter by GO category ('P' for Process, 'F' for Function, 'C' for Component)

        Returns:
            APIResponse with list of GO terms
        """
        response = await self.search_gene(gene_symbol)

        if response.success and response.data:
            go_terms = response.data.get("go_terms", [])

            if category:
                # Filter by category (GO:00xxxxx format)
                # P = biological process, F = molecular function, C = cellular component
                category_map = {"P": "GO:0008", "F": "GO:0003", "C": "GO:0005"}
                prefix = category_map.get(category.upper(), "")
                if prefix:
                    go_terms = [g for g in go_terms if g.get("id", "").startswith(prefix)]

            return APIResponse(
                success=True,
                data=go_terms,
                source=self.API_NAME,
                query=gene_symbol
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=gene_symbol,
            error="Failed to get GO terms"
        )

    async def get_disease_associations(self, gene_symbol: str) -> APIResponse:
        """
        Get disease associations for a gene.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with list of diseases
        """
        response = await self.search_gene(gene_symbol)

        if response.success and response.data:
            diseases = response.data.get("disease_associations", [])
            return APIResponse(
                success=True,
                data=diseases,
                source=self.API_NAME,
                query=gene_symbol
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=gene_symbol,
            error="Failed to get disease associations"
        )

    async def get_subcellular_location(self, gene_symbol: str) -> APIResponse:
        """
        Get subcellular location for a protein.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with subcellular locations
        """
        response = await self.search_gene(gene_symbol)

        if response.success and response.data:
            locations = response.data.get("subcellular_location", [])
            return APIResponse(
                success=True,
                data=locations,
                source=self.API_NAME,
                query=gene_symbol
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=gene_symbol,
            error="Failed to get subcellular location"
        )


def get_uniprot_client() -> UniProtClient:
    """Factory function to create UniProt client."""
    return UniProtClient()
