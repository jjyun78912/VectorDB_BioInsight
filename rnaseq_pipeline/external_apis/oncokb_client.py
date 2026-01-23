"""
OncoKB API Client.

OncoKB is a precision oncology knowledge base that contains
information about the effects and treatment implications of
specific cancer gene alterations.

Features:
- Gene annotations (oncogene/TSG status)
- Actionable alterations
- FDA-approved therapies
- Clinical evidence levels

API Documentation: https://www.oncokb.org/api/v1/swagger-ui.html
"""

import os
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


@dataclass
class OncoKBGeneInfo:
    """OncoKB gene annotation."""
    gene_symbol: str
    entrez_gene_id: Optional[int] = None
    oncogene: bool = False
    tsg: bool = False  # Tumor Suppressor Gene
    highest_level: Optional[str] = None  # Highest therapeutic level
    has_actionable_alterations: bool = False
    summary: str = ""
    background: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OncoKBDrugInfo:
    """OncoKB drug/therapy information."""
    drug_name: str
    gene: str
    alteration: str
    level: str  # e.g., "LEVEL_1", "LEVEL_2"
    indication: str
    fda_approved: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OncoKBClient(BaseAPIClient):
    """
    Client for OncoKB API.

    Provides access to:
    - Gene annotations (oncogene/TSG classification)
    - Therapeutic levels and actionability
    - Drug associations

    Note: OncoKB API requires authentication for full access.
    Limited public access is available without API key.
    """

    BASE_URL = "https://www.oncokb.org/api/v1"
    API_NAME = "OncoKB"
    RATE_LIMIT_DELAY = 0.5

    # OncoKB therapeutic evidence levels
    EVIDENCE_LEVELS = {
        "LEVEL_1": "FDA-recognized biomarker",
        "LEVEL_2": "Standard care biomarker",
        "LEVEL_3A": "Compelling clinical evidence",
        "LEVEL_3B": "Standard care in another indication",
        "LEVEL_4": "Compelling biological evidence",
        "LEVEL_R1": "Resistance - standard care",
        "LEVEL_R2": "Resistance - investigational"
    }

    # Curated oncogene/TSG list (fallback when API unavailable)
    CURATED_ONCOGENES = {
        'KRAS', 'NRAS', 'HRAS', 'BRAF', 'PIK3CA', 'AKT1', 'EGFR', 'ERBB2',
        'MET', 'ALK', 'ROS1', 'RET', 'FGFR1', 'FGFR2', 'FGFR3', 'KIT',
        'PDGFRA', 'ABL1', 'JAK2', 'MYC', 'MYCN', 'BCL2', 'MDM2', 'CDK4',
        'CDK6', 'CCND1', 'CCNE1', 'IDH1', 'IDH2', 'CTNNB1', 'NOTCH1'
    }

    CURATED_TSG = {
        'TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'NF1', 'NF2',
        'VHL', 'WT1', 'CDKN2A', 'CDKN2B', 'SMAD4', 'STK11', 'ATM', 'ATR',
        'CHEK2', 'PALB2', 'BAP1', 'ARID1A', 'ARID2', 'SMARCA4', 'KEAP1',
        'FBXW7', 'CREBBP', 'EP300', 'KMT2A', 'KMT2D', 'SETD2'
    }

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OncoKB client.

        Args:
            api_key: OncoKB API token. If not provided, will try
                    ONCOKB_API_KEY environment variable.
        """
        api_key = api_key or os.environ.get("ONCOKB_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """
        Get gene annotation from OncoKB.

        Args:
            gene_symbol: HUGO gene symbol (e.g., 'TP53')

        Returns:
            APIResponse with OncoKBGeneInfo data
        """
        # Check cache first
        cache_key = self._get_cache_key("genes", {"hugoSymbol": gene_symbol})
        cached = self._read_cache(cache_key)
        if cached:
            return APIResponse(
                success=True,
                data=cached,
                source=self.API_NAME,
                query=gene_symbol,
                cached=True
            )

        # Try API if key available
        if self.api_key:
            try:
                response = await self.get(
                    f"genes/{gene_symbol}",
                    use_cache=False
                )

                if response.success and response.data:
                    gene_info = OncoKBGeneInfo(
                        gene_symbol=gene_symbol,
                        entrez_gene_id=response.data.get("entrezGeneId"),
                        oncogene=response.data.get("oncogene", False),
                        tsg=response.data.get("tsg", False),
                        highest_level=response.data.get("highestSensitiveLevel"),
                        summary=response.data.get("summary", ""),
                        background=response.data.get("background", "")
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
                logger.warning(f"OncoKB API error for {gene_symbol}: {e}")

        # Fallback to curated list
        gene_info = OncoKBGeneInfo(
            gene_symbol=gene_symbol,
            oncogene=gene_symbol in self.CURATED_ONCOGENES,
            tsg=gene_symbol in self.CURATED_TSG,
            summary="Curated annotation (API unavailable)"
        )

        return APIResponse(
            success=True,
            data=gene_info.to_dict(),
            source=f"{self.API_NAME} (curated)",
            query=gene_symbol
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

    async def get_actionable_genes(self) -> APIResponse:
        """Get list of all actionable genes in OncoKB."""
        if not self.api_key:
            # Return curated list as fallback
            return APIResponse(
                success=True,
                data=list(self.CURATED_ONCOGENES | self.CURATED_TSG),
                source=f"{self.API_NAME} (curated)",
                query="actionable_genes"
            )

        return await self.get("genes")

    async def get_therapeutic_implications(
        self,
        gene_symbol: str,
        alteration: Optional[str] = None,
        cancer_type: Optional[str] = None
    ) -> APIResponse:
        """
        Get therapeutic implications for gene/alteration.

        Args:
            gene_symbol: Gene symbol
            alteration: Specific alteration (e.g., 'V600E')
            cancer_type: OncoTree cancer type

        Returns:
            APIResponse with therapeutic information
        """
        if not self.api_key:
            return APIResponse(
                success=False,
                data=None,
                source=self.API_NAME,
                query=gene_symbol,
                error="API key required for therapeutic implications"
            )

        params = {"hugoSymbol": gene_symbol}
        if alteration:
            params["alteration"] = alteration
        if cancer_type:
            params["tumorType"] = cancer_type

        return await self.get("annotate/mutations/byProteinChange", params=params)

    def get_gene_role(self, gene_symbol: str) -> str:
        """
        Get gene role (oncogene/TSG) from curated list.

        Quick synchronous lookup without API call.
        """
        if gene_symbol in self.CURATED_ONCOGENES:
            return "Oncogene"
        elif gene_symbol in self.CURATED_TSG:
            return "Tumor Suppressor"
        elif gene_symbol in (self.CURATED_ONCOGENES & self.CURATED_TSG):
            return "Oncogene/TSG (context-dependent)"
        return "Unknown"

    def classify_genes(self, gene_symbols: List[str]) -> Dict[str, Dict]:
        """
        Classify genes as oncogene/TSG synchronously.

        Args:
            gene_symbols: List of gene symbols

        Returns:
            Dict with gene classifications
        """
        results = {}
        for gene in gene_symbols:
            results[gene] = {
                "gene_symbol": gene,
                "oncogene": gene in self.CURATED_ONCOGENES,
                "tsg": gene in self.CURATED_TSG,
                "role": self.get_gene_role(gene),
                "actionable": gene in (self.CURATED_ONCOGENES | self.CURATED_TSG)
            }
        return results


def get_oncokb_client() -> OncoKBClient:
    """Factory function to create OncoKB client."""
    return OncoKBClient()
