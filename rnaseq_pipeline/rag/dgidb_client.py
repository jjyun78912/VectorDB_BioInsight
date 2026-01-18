"""
DGIdb (Drug-Gene Interaction Database) Client Module.

Uses DGIdb v5 GraphQL API for verified drug-gene interactions.

API Documentation: https://dgidb.org/api
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DrugGeneInteraction:
    """Represents a drug-gene interaction from DGIdb."""
    gene_name: str
    drug_name: str
    interaction_types: List[str]
    interaction_score: float = 0.0
    pmids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_name": self.gene_name,
            "drug_name": self.drug_name,
            "interaction_types": self.interaction_types,
            "interaction_score": self.interaction_score,
            "pmids": self.pmids,
            "sources": self.sources
        }


@dataclass
class GeneCategory:
    """Gene category/druggability information from DGIdb."""
    gene_name: str
    categories: List[str]
    is_druggable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_name": self.gene_name,
            "categories": self.categories,
            "is_druggable": self.is_druggable
        }


class DGIdbClient:
    """
    Client for DGIdb v5 GraphQL API.

    DGIdb provides curated drug-gene interactions from multiple sources
    including DrugBank, PharmGKB, ChEMBL, OncoKB, and others.

    Usage:
        client = DGIdbClient()

        # Get drug interactions for genes
        interactions = client.get_drug_interactions(["EGFR", "KRAS", "TP53"])

        # Check if genes are druggable
        categories = client.get_gene_categories(["EGFR", "BRAF"])
    """

    GRAPHQL_URL = "https://dgidb.org/api/graphql"

    def __init__(self, timeout: int = 30):
        """
        Initialize DGIdb client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

    def _execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute a GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = self.session.post(
                self.GRAPHQL_URL,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("DGIdb API timeout")
            return {"errors": [{"message": "Timeout"}]}
        except requests.exceptions.RequestException as e:
            logger.error(f"DGIdb API error: {e}")
            return {"errors": [{"message": str(e)}]}
        except Exception as e:
            logger.error(f"Error parsing DGIdb response: {e}")
            return {"errors": [{"message": str(e)}]}

    def get_drug_interactions(
        self,
        genes: List[str],
        min_score: float = 0.0
    ) -> Dict[str, List[DrugGeneInteraction]]:
        """
        Get drug-gene interactions for a list of genes.

        Args:
            genes: List of gene symbols (e.g., ["EGFR", "KRAS"])
            min_score: Minimum interaction score to include

        Returns:
            Dictionary mapping gene symbols to their drug interactions
        """
        if not genes:
            return {}

        # GraphQL query for interactions
        query = """
        query GetInteractions($names: [String!]) {
          genes(names: $names) {
            nodes {
              name
              interactions {
                drug {
                  name
                  approved
                }
                interactionScore
                interactionTypes {
                  type
                  directionality
                }
                publications {
                  pmid
                }
                sources {
                  sourceDbName
                }
              }
            }
          }
        }
        """

        logger.info(f"Querying DGIdb for {len(genes)} genes...")

        result = self._execute_query(query, {"names": genes})

        if "errors" in result:
            logger.error(f"DGIdb query error: {result['errors']}")
            return {}

        return self._parse_interactions(result, min_score)

    def _parse_interactions(self, data: Dict, min_score: float = 0.0) -> Dict[str, List[DrugGeneInteraction]]:
        """Parse GraphQL response into DrugGeneInteraction objects."""
        result = {}

        try:
            nodes = data.get("data", {}).get("genes", {}).get("nodes", [])

            for node in nodes:
                gene_name = node.get("name", "")
                if not gene_name:
                    continue

                interactions = []
                for interaction in node.get("interactions", []):
                    drug_info = interaction.get("drug", {})
                    drug_name = drug_info.get("name", "")
                    if not drug_name:
                        continue

                    score = interaction.get("interactionScore", 0) or 0
                    if score < min_score:
                        continue

                    # Extract interaction types
                    int_types = []
                    for it in interaction.get("interactionTypes", []):
                        if it.get("type"):
                            int_types.append(it["type"])

                    # Extract PMIDs
                    pmids = []
                    for pub in interaction.get("publications", []):
                        if pub.get("pmid"):
                            pmids.append(str(pub["pmid"]))

                    # Extract sources
                    sources = []
                    for src in interaction.get("sources", []):
                        if src.get("sourceDbName"):
                            sources.append(src["sourceDbName"])

                    # Check if FDA approved
                    if drug_info.get("approved"):
                        sources.append("FDA Approved")

                    interactions.append(DrugGeneInteraction(
                        gene_name=gene_name,
                        drug_name=drug_name,
                        interaction_types=int_types,
                        interaction_score=score,
                        pmids=pmids[:5],  # Limit PMIDs
                        sources=list(set(sources))[:5]  # Dedupe and limit
                    ))

                if interactions:
                    # Sort by score (higher = more reliable)
                    interactions.sort(key=lambda x: x.interaction_score, reverse=True)
                    result[gene_name] = interactions

            logger.info(f"Found interactions for {len(result)} genes")

        except Exception as e:
            logger.error(f"Error parsing DGIdb interactions: {e}")

        return result

    def get_gene_categories(self, genes: List[str]) -> Dict[str, GeneCategory]:
        """
        Get gene categories (druggability) for a list of genes.

        Args:
            genes: List of gene symbols

        Returns:
            Dictionary mapping gene symbols to their categories
        """
        if not genes:
            return {}

        query = """
        query GetCategories($names: [String!]) {
          genes(names: $names) {
            nodes {
              name
              geneCategories {
                name
              }
            }
          }
        }
        """

        result = self._execute_query(query, {"names": genes})

        if "errors" in result:
            logger.error(f"DGIdb category query error: {result['errors']}")
            return {}

        return self._parse_categories(result)

    def _parse_categories(self, data: Dict) -> Dict[str, GeneCategory]:
        """Parse gene categories response."""
        result = {}

        try:
            nodes = data.get("data", {}).get("genes", {}).get("nodes", [])

            for node in nodes:
                gene_name = node.get("name", "")
                if not gene_name:
                    continue

                categories = []
                for cat in node.get("geneCategories", []):
                    if cat.get("name"):
                        categories.append(cat["name"])

                # Determine if druggable based on categories
                druggable_keywords = [
                    "DRUGGABLE", "CLINICALLY ACTIONABLE", "KINASE",
                    "RECEPTOR", "TRANSPORTER", "ENZYME", "FDA"
                ]
                is_druggable = any(
                    any(kw in cat.upper() for kw in druggable_keywords)
                    for cat in categories
                ) or len(categories) > 0

                result[gene_name] = GeneCategory(
                    gene_name=gene_name,
                    categories=categories,
                    is_druggable=is_druggable
                )

        except Exception as e:
            logger.error(f"Error parsing DGIdb categories: {e}")

        return result

    def get_druggable_genes(
        self,
        genes: List[str],
        include_interactions: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive druggability info for genes.

        Combines gene categories and drug interactions into a single result.

        Args:
            genes: List of gene symbols
            include_interactions: Whether to also fetch drug interactions

        Returns:
            Dictionary with druggability info for each gene
        """
        result = {}

        # Get categories
        categories = self.get_gene_categories(genes)

        # Get interactions if requested
        interactions = {}
        if include_interactions:
            interactions = self.get_drug_interactions(genes)

        # Combine results
        for gene in genes:
            gene_upper = gene.upper()

            cat_info = categories.get(gene_upper) or categories.get(gene)
            int_info = interactions.get(gene_upper) or interactions.get(gene)

            result[gene] = {
                "gene": gene,
                "is_druggable": (cat_info.is_druggable if cat_info else False) or bool(int_info),
                "categories": cat_info.categories if cat_info else [],
                "interactions": [i.to_dict() for i in (int_info or [])[:5]],  # Top 5
                "drug_count": len(int_info) if int_info else 0,
                "has_fda_approved_drugs": any(
                    "FDA Approved" in i.sources
                    for i in (int_info or [])
                )
            }

        return result


def get_therapeutic_targets(
    hub_genes: List[Dict[str, Any]],
    deg_genes: List[Dict[str, Any]],
    max_targets: int = 10
) -> Dict[str, Any]:
    """
    Identify therapeutic targets from hub genes and DEGs using DGIdb.

    This function queries DGIdb to find:
    1. Which hub genes are druggable
    2. What drugs target these genes
    3. Evidence from literature (PMIDs)

    Args:
        hub_genes: List of hub genes with gene_symbol, log2FC, etc.
        deg_genes: List of DEGs for additional context
        max_targets: Maximum number of targets to return

    Returns:
        Dictionary with therapeutic target recommendations
    """
    client = DGIdbClient()

    # Extract gene symbols
    gene_symbols = []
    gene_info = {}

    for gene in hub_genes:
        symbol = gene.get("gene_symbol") or gene.get("gene")
        if symbol:
            gene_symbols.append(symbol)
            gene_info[symbol] = {
                "log2FC": gene.get("log2FC") or gene.get("log2fc", 0),
                "padj": gene.get("padj", 0),
                "hub_score": gene.get("hub_score", 0)
            }

    if not gene_symbols:
        return {"error": "No gene symbols provided", "targets": []}

    logger.info(f"Checking {len(gene_symbols)} genes for druggability...")

    # Query DGIdb
    druggability = client.get_druggable_genes(gene_symbols[:20])  # Limit API calls

    # Build therapeutic targets list
    targets = []
    for gene, info in druggability.items():
        if info["is_druggable"] or info["drug_count"] > 0:
            gene_data = gene_info.get(gene, {})

            # Get top drugs
            top_drugs = []
            for interaction in info["interactions"][:3]:
                drug_entry = {
                    "name": interaction["drug_name"],
                    "interaction_type": interaction["interaction_types"][0] if interaction["interaction_types"] else "unknown",
                    "sources": interaction["sources"][:3],
                    "pmids": interaction["pmids"][:2]
                }
                top_drugs.append(drug_entry)

            targets.append({
                "gene": gene,
                "log2FC": gene_data.get("log2FC", 0),
                "direction": "upregulated" if gene_data.get("log2FC", 0) > 0 else "downregulated",
                "categories": info["categories"],
                "is_clinically_actionable": "CLINICALLY ACTIONABLE" in " ".join(info["categories"]).upper(),
                "drug_count": info["drug_count"],
                "has_fda_approved": info["has_fda_approved_drugs"],
                "top_drugs": top_drugs,
                "therapeutic_rationale": _generate_rationale(gene, gene_data, info)
            })

    # Sort by actionability and drug count
    targets.sort(key=lambda x: (
        x["is_clinically_actionable"],
        x["has_fda_approved"],
        x["drug_count"]
    ), reverse=True)

    return {
        "total_genes_checked": len(gene_symbols),
        "druggable_genes": len(targets),
        "targets": targets[:max_targets],
        "source": "DGIdb (Drug-Gene Interaction Database)",
        "disclaimer": "약물-유전자 상호작용은 DGIdb 데이터베이스에서 검증된 정보입니다. 임상 적용 전 전문가 검토가 필요합니다."
    }


def _generate_rationale(gene: str, gene_data: Dict, druggability: Dict) -> str:
    """Generate therapeutic rationale for a gene."""
    log2fc = gene_data.get("log2FC", 0)
    direction = "상향조절" if log2fc > 0 else "하향조절"

    categories = druggability.get("categories", [])
    drug_count = druggability.get("drug_count", 0)

    rationale_parts = []

    # Expression change
    rationale_parts.append(f"{gene}은(는) {direction}됨 (log2FC={log2fc:.2f})")

    # Druggability category
    if "CLINICALLY ACTIONABLE" in " ".join(categories).upper():
        rationale_parts.append("임상적으로 표적 가능한 유전자")
    elif "KINASE" in " ".join(categories).upper():
        rationale_parts.append("키나제 억제제 개발 가능")
    elif drug_count > 0:
        rationale_parts.append(f"{drug_count}개 약물이 이 유전자를 표적")

    # FDA approved
    if druggability.get("has_fda_approved_drugs"):
        rationale_parts.append("FDA 승인 약물 존재")

    return ". ".join(rationale_parts) + "."


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = DGIdbClient()

    # Test with common cancer genes
    test_genes = ["EGFR", "KRAS", "TP53", "BRAF", "PIK3CA"]

    print("=== Drug-Gene Interactions ===")
    interactions = client.get_drug_interactions(test_genes)
    for gene, drugs in interactions.items():
        print(f"\n{gene}: {len(drugs)} interactions")
        for drug in drugs[:3]:
            print(f"  - {drug.drug_name} ({', '.join(drug.interaction_types)}) score={drug.interaction_score:.3f}")
            if drug.pmids:
                print(f"    PMIDs: {', '.join(drug.pmids[:3])}")

    print("\n=== Gene Categories ===")
    categories = client.get_gene_categories(test_genes)
    for gene, cat in categories.items():
        print(f"{gene}: {cat.categories} (druggable={cat.is_druggable})")

    print("\n=== Therapeutic Targets ===")
    hub_genes = [
        {"gene_symbol": "EGFR", "log2FC": 2.5, "padj": 0.001},
        {"gene_symbol": "KRAS", "log2FC": 1.8, "padj": 0.005},
        {"gene_symbol": "TP53", "log2FC": -1.2, "padj": 0.01},
    ]
    targets = get_therapeutic_targets(hub_genes, [])
    print(f"Found {targets['druggable_genes']} druggable genes")
    for t in targets["targets"]:
        print(f"  {t['gene']}: {t['drug_count']} drugs, FDA approved: {t['has_fda_approved']}")
