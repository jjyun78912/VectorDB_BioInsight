"""
CIViC (Clinical Interpretation of Variants in Cancer) API Client.

CIViC is an open-access, community-driven knowledge base for
clinical interpretation of variants in cancer.

Features:
- Variant-level clinical evidence
- Gene summaries
- Evidence items with clinical significance
- Assertions and therapies

API Documentation: https://civicdb.org/api/graphql
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


@dataclass
class CIViCGeneInfo:
    """CIViC gene information."""
    gene_symbol: str
    civic_id: Optional[int] = None
    description: str = ""
    variants_count: int = 0
    assertions_count: int = 0
    evidence_items_count: int = 0
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CIViCVariantInfo:
    """CIViC variant information."""
    variant_name: str
    gene_symbol: str
    civic_id: Optional[int] = None
    clinical_significance: str = ""
    variant_types: List[str] = None
    evidence_count: int = 0
    therapies: List[str] = None

    def __post_init__(self):
        if self.variant_types is None:
            self.variant_types = []
        if self.therapies is None:
            self.therapies = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CIViCClient(BaseAPIClient):
    """
    Client for CIViC GraphQL API.

    CIViC provides clinical interpretations of variants in cancer,
    with community-curated evidence.

    Note: CIViC API is free and does not require authentication.
    """

    BASE_URL = "https://civicdb.org/api/graphql"
    API_NAME = "CIViC"
    RATE_LIMIT_DELAY = 0.3

    # CIViC evidence levels
    EVIDENCE_LEVELS = {
        "A": "Validated association",
        "B": "Clinical evidence",
        "C": "Case study",
        "D": "Preclinical evidence",
        "E": "Inferential association"
    }

    # Clinical significance types
    CLINICAL_SIGNIFICANCE = {
        "Sensitivity/Response": "Positive therapeutic response",
        "Resistance": "Therapeutic resistance",
        "Adverse Response": "Negative therapeutic response",
        "Reduced Sensitivity": "Reduced response",
        "Pathogenic": "Disease causing",
        "Likely Pathogenic": "Likely disease causing",
        "Uncertain Significance": "Unknown significance",
        "Positive": "Positive prognostic/diagnostic",
        "Negative": "Negative prognostic/diagnostic"
    }

    def __init__(self, **kwargs):
        """Initialize CIViC client. No API key required."""
        super().__init__(api_key=None, **kwargs)

    async def _graphql_query(self, query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
        """Execute GraphQL query."""
        session = await self._get_session()
        await self._rate_limit()

        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables

            async with session.post(
                self.BASE_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("data")
                else:
                    logger.warning(f"CIViC query failed: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"CIViC error: {e}")
            return None

    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """
        Search for gene in CIViC.

        Args:
            gene_symbol: HUGO gene symbol

        Returns:
            APIResponse with CIViCGeneInfo
        """
        # Check cache
        cache_key = self._get_cache_key("gene", {"symbol": gene_symbol})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=gene_symbol,
                cached=True
            )

        query = """
        query GeneSearch($name: String!) {
            genes(name: $name) {
                nodes {
                    id
                    name
                    description
                    variants {
                        totalCount
                    }
                    geneAliases
                }
            }
        }
        """

        data = await self._graphql_query(query, {"name": gene_symbol})

        if data and data.get("genes", {}).get("nodes"):
            gene_data = data["genes"]["nodes"][0]

            gene_info = CIViCGeneInfo(
                gene_symbol=gene_data.get("name", gene_symbol),
                civic_id=gene_data.get("id"),
                description=gene_data.get("description", ""),
                variants_count=gene_data.get("variants", {}).get("totalCount", 0),
                aliases=gene_data.get("geneAliases", [])
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
            error="Gene not found in CIViC"
        )

    async def batch_search(self, gene_symbols: List[str]) -> Dict[str, APIResponse]:
        """
        Batch search for multiple genes.

        Args:
            gene_symbols: List of gene symbols

        Returns:
            Dict mapping gene symbol to APIResponse
        """
        results = {}
        for gene in gene_symbols:
            results[gene] = await self.search_gene(gene)
        return results

    async def get_variants(self, gene_symbol: str) -> APIResponse:
        """
        Get variants for a gene from CIViC.

        Args:
            gene_symbol: Gene symbol

        Returns:
            APIResponse with list of variants
        """
        cache_key = self._get_cache_key("variants", {"gene": gene_symbol})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=gene_symbol,
                cached=True
            )

        query = """
        query GeneVariants($name: String!) {
            genes(name: $name) {
                nodes {
                    variants {
                        nodes {
                            id
                            name
                            variantTypes {
                                name
                            }
                            molecularProfiles {
                                nodes {
                                    evidenceItems {
                                        totalCount
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        data = await self._graphql_query(query, {"name": gene_symbol})

        if data and data.get("genes", {}).get("nodes"):
            variants_data = data["genes"]["nodes"][0].get("variants", {}).get("nodes", [])

            variants = []
            for v in variants_data[:20]:  # Limit to top 20 variants
                variant_types = [vt.get("name") for vt in v.get("variantTypes", [])]
                evidence_count = sum(
                    mp.get("evidenceItems", {}).get("totalCount", 0)
                    for mp in v.get("molecularProfiles", {}).get("nodes", [])
                )

                variants.append(CIViCVariantInfo(
                    variant_name=v.get("name", ""),
                    gene_symbol=gene_symbol,
                    civic_id=v.get("id"),
                    variant_types=variant_types,
                    evidence_count=evidence_count
                ).to_dict())

            self._write_cache(cache_key, variants)

            return APIResponse(
                success=True,
                data=variants,
                source=self.API_NAME,
                query=gene_symbol
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query=gene_symbol,
            error="No variants found"
        )

    async def get_evidence(self, gene_symbol: str, limit: int = 10) -> APIResponse:
        """
        Get clinical evidence for a gene.

        Args:
            gene_symbol: Gene symbol
            limit: Maximum number of evidence items

        Returns:
            APIResponse with evidence items
        """
        query = """
        query GeneEvidence($name: String!, $first: Int) {
            genes(name: $name) {
                nodes {
                    variants {
                        nodes {
                            molecularProfiles {
                                nodes {
                                    evidenceItems(first: $first) {
                                        nodes {
                                            id
                                            status
                                            evidenceLevel
                                            evidenceType
                                            significance
                                            description
                                            therapies {
                                                name
                                            }
                                            disease {
                                                name
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        data = await self._graphql_query(query, {"name": gene_symbol, "first": limit})

        evidence_items = []
        if data and data.get("genes", {}).get("nodes"):
            for variant in data["genes"]["nodes"][0].get("variants", {}).get("nodes", []):
                for mp in variant.get("molecularProfiles", {}).get("nodes", []):
                    for item in mp.get("evidenceItems", {}).get("nodes", []):
                        if item.get("status") == "accepted":
                            therapies = [t.get("name") for t in item.get("therapies", [])]
                            evidence_items.append({
                                "evidence_id": item.get("id"),
                                "level": item.get("evidenceLevel"),
                                "type": item.get("evidenceType"),
                                "significance": item.get("significance"),
                                "description": item.get("description", "")[:300],
                                "therapies": therapies,
                                "disease": item.get("disease", {}).get("name", "")
                            })

        return APIResponse(
            success=bool(evidence_items),
            data=evidence_items[:limit],
            source=self.API_NAME,
            query=gene_symbol
        )

    async def get_actionable_genes(self) -> APIResponse:
        """Get list of genes with clinical evidence in CIViC."""
        query = """
        query ActionableGenes {
            genes(first: 500) {
                nodes {
                    name
                    variants {
                        totalCount
                    }
                }
            }
        }
        """

        data = await self._graphql_query(query)

        if data and data.get("genes", {}).get("nodes"):
            genes = [
                g.get("name") for g in data["genes"]["nodes"]
                if g.get("variants", {}).get("totalCount", 0) > 0
            ]
            return APIResponse(
                success=True,
                data=genes,
                source=self.API_NAME,
                query="actionable_genes"
            )

        return APIResponse(
            success=False,
            data=[],
            source=self.API_NAME,
            query="actionable_genes",
            error="Failed to fetch genes"
        )


def get_civic_client() -> CIViCClient:
    """Factory function to create CIViC client."""
    return CIViCClient()
