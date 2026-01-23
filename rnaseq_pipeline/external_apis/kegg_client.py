"""
KEGG REST API Client.

KEGG (Kyoto Encyclopedia of Genes and Genomes) provides
comprehensive pathway information.

Features:
- Pathway information
- Gene-pathway associations
- Pathway visualization
- Disease pathways

API Documentation: https://www.kegg.jp/kegg/rest/keggapi.html
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


@dataclass
class KEGGPathway:
    """KEGG pathway information."""
    pathway_id: str
    name: str
    description: str = ""
    gene_count: int = 0
    category: str = ""
    genes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KEGGGeneInfo:
    """KEGG gene information."""
    kegg_id: str
    gene_symbol: str
    name: str = ""
    definition: str = ""
    orthology: str = ""
    pathways: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KEGGClient(BaseAPIClient):
    """
    Client for KEGG REST API.

    Provides access to pathway information and gene annotations.
    No API key required.
    """

    BASE_URL = "https://rest.kegg.jp"
    API_NAME = "KEGG"
    RATE_LIMIT_DELAY = 0.5

    # Human organism prefix
    ORGANISM = "hsa"

    # Common cancer-related pathways
    CANCER_PATHWAYS = {
        "hsa05200": "Pathways in cancer",
        "hsa05210": "Colorectal cancer",
        "hsa05212": "Pancreatic cancer",
        "hsa05214": "Glioma",
        "hsa05215": "Prostate cancer",
        "hsa05216": "Thyroid cancer",
        "hsa05220": "Chronic myeloid leukemia",
        "hsa05221": "Acute myeloid leukemia",
        "hsa05222": "Small cell lung cancer",
        "hsa05223": "Non-small cell lung cancer",
        "hsa05224": "Breast cancer",
        "hsa05225": "Hepatocellular carcinoma",
        "hsa05226": "Gastric cancer",
        "hsa05230": "Central carbon metabolism in cancer",
        "hsa05231": "Choline metabolism in cancer"
    }

    def __init__(self, organism: str = "hsa", **kwargs):
        """
        Initialize KEGG client.

        Args:
            organism: KEGG organism code (default: 'hsa' for human)
        """
        super().__init__(**kwargs)
        self.organism = organism

    async def _parse_kegg_response(self, text: str) -> Dict[str, str]:
        """Parse KEGG flat file format."""
        result = {}
        current_key = None
        current_value = []

        for line in text.split("\n"):
            if not line.strip():
                continue

            if line.startswith(" ") or line.startswith("\t"):
                # Continuation of previous field
                if current_key:
                    current_value.append(line.strip())
            else:
                # New field
                if current_key:
                    result[current_key] = "\n".join(current_value)

                parts = line.split(None, 1)
                if len(parts) >= 2:
                    current_key = parts[0]
                    current_value = [parts[1]]
                elif parts:
                    current_key = parts[0]
                    current_value = []

        if current_key:
            result[current_key] = "\n".join(current_value)

        return result

    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """
        Search for gene in KEGG.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with KEGGGeneInfo
        """
        cache_key = self._get_cache_key("find", {"gene": gene_symbol})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=gene_symbol,
                cached=True
            )

        # First find the KEGG gene ID
        session = await self._get_session()
        await self._rate_limit()

        try:
            async with session.get(
                f"{self.BASE_URL}/find/{self.organism}/{gene_symbol}"
            ) as resp:
                if resp.status != 200:
                    return APIResponse(
                        success=False, data=None,
                        source=self.API_NAME, query=gene_symbol,
                        error="Gene not found"
                    )

                text = await resp.text()
                lines = text.strip().split("\n")

                if not lines or not lines[0]:
                    return APIResponse(
                        success=False, data=None,
                        source=self.API_NAME, query=gene_symbol,
                        error="Gene not found in KEGG"
                    )

                # Parse first result
                parts = lines[0].split("\t")
                kegg_id = parts[0] if parts else ""

            # Get detailed gene info
            await self._rate_limit()
            async with session.get(f"{self.BASE_URL}/get/{kegg_id}") as resp:
                if resp.status != 200:
                    return APIResponse(
                        success=False, data=None,
                        source=self.API_NAME, query=gene_symbol,
                        error="Failed to get gene details"
                    )

                text = await resp.text()
                parsed = await self._parse_kegg_response(text)

                # Parse pathways
                pathways = []
                pathway_text = parsed.get("PATHWAY", "")
                for line in pathway_text.split("\n"):
                    parts = line.strip().split(None, 1)
                    if len(parts) >= 2:
                        pathways.append({
                            "id": parts[0],
                            "name": parts[1]
                        })

                gene_info = KEGGGeneInfo(
                    kegg_id=kegg_id,
                    gene_symbol=gene_symbol,
                    name=parsed.get("NAME", "").split(",")[0].strip(),
                    definition=parsed.get("DEFINITION", "")[:300],
                    orthology=parsed.get("ORTHOLOGY", "").split()[0] if parsed.get("ORTHOLOGY") else "",
                    pathways=pathways[:20]
                )

                result = gene_info.to_dict()
                self._write_cache(cache_key, result)

                return APIResponse(
                    success=True,
                    data=result,
                    source=self.API_NAME,
                    query=gene_symbol
                )

        except Exception as e:
            logger.error(f"KEGG error for {gene_symbol}: {e}")
            return APIResponse(
                success=False, data=None,
                source=self.API_NAME, query=gene_symbol,
                error=str(e)
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
            pathway_id: KEGG pathway ID (e.g., 'hsa05200')

        Returns:
            APIResponse with KEGGPathway
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

        session = await self._get_session()
        await self._rate_limit()

        try:
            async with session.get(f"{self.BASE_URL}/get/{pathway_id}") as resp:
                if resp.status != 200:
                    return APIResponse(
                        success=False, data=None,
                        source=self.API_NAME, query=pathway_id,
                        error="Pathway not found"
                    )

                text = await resp.text()
                parsed = await self._parse_kegg_response(text)

                # Parse genes in pathway
                genes = []
                gene_text = parsed.get("GENE", "")
                for line in gene_text.split("\n"):
                    parts = line.strip().split(None, 1)
                    if parts:
                        # Extract gene symbol from description
                        if len(parts) >= 2:
                            gene_desc = parts[1].split(";")[0].strip()
                            genes.append(gene_desc)

                pathway = KEGGPathway(
                    pathway_id=pathway_id,
                    name=parsed.get("NAME", "").split(" - ")[0].strip(),
                    description=parsed.get("DESCRIPTION", "")[:500],
                    gene_count=len(genes),
                    category=parsed.get("CLASS", ""),
                    genes=genes[:100]
                )

                result = pathway.to_dict()
                self._write_cache(cache_key, result)

                return APIResponse(
                    success=True,
                    data=result,
                    source=self.API_NAME,
                    query=pathway_id
                )

        except Exception as e:
            logger.error(f"KEGG pathway error: {e}")
            return APIResponse(
                success=False, data=None,
                source=self.API_NAME, query=pathway_id,
                error=str(e)
            )

    async def get_gene_pathways(self, gene_symbol: str) -> APIResponse:
        """
        Get all pathways containing a gene.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with list of pathways
        """
        response = await self.search_gene(gene_symbol)

        if response.success and response.data:
            pathways = response.data.get("pathways", [])
            return APIResponse(
                success=True,
                data=pathways,
                source=self.API_NAME,
                query=gene_symbol
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=gene_symbol,
            error="Failed to get pathways"
        )

    async def list_pathways(self, category: Optional[str] = None) -> APIResponse:
        """
        List all human pathways.

        Args:
            category: Filter by category (e.g., 'Disease')

        Returns:
            APIResponse with list of pathways
        """
        session = await self._get_session()
        await self._rate_limit()

        try:
            async with session.get(f"{self.BASE_URL}/list/pathway/{self.organism}") as resp:
                if resp.status != 200:
                    return APIResponse(
                        success=False, data=[],
                        source=self.API_NAME, query="list_pathways",
                        error="Failed to list pathways"
                    )

                text = await resp.text()
                pathways = []

                for line in text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        pathway_id = parts[0].replace("path:", "")
                        name = parts[1].split(" - ")[0].strip()
                        pathways.append({
                            "id": pathway_id,
                            "name": name
                        })

                return APIResponse(
                    success=True,
                    data=pathways,
                    source=self.API_NAME,
                    query="list_pathways"
                )

        except Exception as e:
            logger.error(f"KEGG list pathways error: {e}")
            return APIResponse(
                success=False, data=[],
                source=self.API_NAME, query="list_pathways",
                error=str(e)
            )

    def get_pathway_image_url(self, pathway_id: str) -> str:
        """Get URL for pathway image."""
        return f"https://www.kegg.jp/kegg/pathway/{self.organism}/{pathway_id}.png"


def get_kegg_client() -> KEGGClient:
    """Factory function to create KEGG client."""
    return KEGGClient()
