"""
Enhanced RAG Gene Interpreter with External API Integration.

Combines internal vector search (ChromaDB) with external biological databases
to provide comprehensive, literature-backed gene interpretations.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Enhanced Gene Interpreter                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Gene Symbol + Cancer Type + DEG Data                                │
│              │                                                              │
│      ┌───────┴───────┐                                                      │
│      │               │                                                      │
│      ▼               ▼                                                      │
│  ┌────────────┐  ┌────────────────┐                                         │
│  │ Internal   │  │   External     │                                         │
│  │ Vector DB  │  │   APIs         │                                         │
│  │ (ChromaDB) │  │                │                                         │
│  │            │  │ • OncoKB       │                                         │
│  │ Hybrid:    │  │ • CIViC        │                                         │
│  │ • Dense    │  │ • STRING       │                                         │
│  │ • Sparse   │  │ • UniProt      │                                         │
│  │            │  │ • KEGG         │                                         │
│  │            │  │ • Reactome     │                                         │
│  └─────┬──────┘  └───────┬────────┘                                         │
│        │                 │                                                  │
│        └────────┬────────┘                                                  │
│                 ▼                                                           │
│        ┌───────────────┐                                                    │
│        │  Context      │                                                    │
│        │  Fusion       │                                                    │
│        └───────┬───────┘                                                    │
│                │                                                            │
│                ▼                                                            │
│        ┌───────────────┐                                                    │
│        │  Claude LLM   │                                                    │
│        │ Interpretation│                                                    │
│        └───────┬───────┘                                                    │
│                │                                                            │
│                ▼                                                            │
│        Enhanced GeneInterpretation                                          │
│        (with external annotations)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.app.core.vector_store import create_vector_store
from rnaseq_pipeline.external_apis import (
    ExternalDataFetcher,
    GeneContext,
    get_external_fetcher
)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedGeneInterpretation:
    """Enhanced interpretation with external database annotations."""
    gene_symbol: str
    gene_id: str
    log2fc: float
    direction: str
    interpretation: str

    # Internal RAG results
    citations: List[Dict[str, Any]] = field(default_factory=list)
    pmids: List[str] = field(default_factory=list)

    # External API annotations
    cancer_role: str = "Unknown"
    is_oncogene: bool = False
    is_tsg: bool = False
    actionable: bool = False
    clinical_evidence_count: int = 0

    # Protein info (UniProt)
    protein_function: str = ""
    subcellular_location: List[str] = field(default_factory=list)
    disease_associations: List[str] = field(default_factory=list)

    # Interactions (STRING)
    interaction_partners: List[str] = field(default_factory=list)

    # Pathways (KEGG + Reactome)
    pathways: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    confidence: str = "medium"
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_symbol": self.gene_symbol,
            "gene_id": self.gene_id,
            "log2fc": self.log2fc,
            "direction": self.direction,
            "interpretation": self.interpretation,
            "citations": self.citations,
            "pmids": self.pmids,
            "cancer_role": self.cancer_role,
            "is_oncogene": self.is_oncogene,
            "is_tsg": self.is_tsg,
            "actionable": self.actionable,
            "clinical_evidence_count": self.clinical_evidence_count,
            "protein_function": self.protein_function,
            "subcellular_location": self.subcellular_location,
            "disease_associations": self.disease_associations,
            "interaction_partners": self.interaction_partners,
            "pathways": self.pathways,
            "confidence": self.confidence,
            "sources_used": self.sources_used
        }


class EnhancedGeneInterpreter:
    """
    Enhanced RAG interpreter with external API integration.

    Combines:
    - Internal VectorDB (Hybrid search: Dense + Sparse)
    - External APIs (OncoKB, CIViC, STRING, UniProt, KEGG, Reactome)
    - Claude LLM for interpretation synthesis
    """

    def __init__(
        self,
        cancer_type: str = "breast_cancer",
        top_k: int = 5,
        use_llm: bool = True,
        use_external_apis: bool = True,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the enhanced interpreter.

        Args:
            cancer_type: Cancer type for context
            top_k: Number of papers to retrieve
            use_llm: Use Claude for interpretation
            use_external_apis: Fetch data from external APIs
            model: Claude model to use
        """
        self.cancer_type = cancer_type
        self.top_k = top_k
        self.use_llm = use_llm
        self.use_external_apis = use_external_apis
        self.model = model
        self.logger = logging.getLogger("enhanced_interpreter")

        # Domain mapping for ChromaDB
        self.domain_map = {
            "LUAD": "lung_cancer", "LUSC": "lung_cancer",
            "BRCA": "breast_cancer", "COAD": "colorectal_cancer",
            "PAAD": "pancreatic_cancer", "LIHC": "liver_cancer",
            "GBM": "glioblastoma", "LGG": "low_grade_glioma",
            "KIRC": "kidney_cancer", "BLCA": "bladder_cancer",
            "HNSC": "head_neck_cancer", "THCA": "thyroid_cancer",
            "PRAD": "prostate_cancer", "STAD": "stomach_cancer",
            "SKCM": "melanoma", "OV": "ovarian_cancer",
            "UCEC": "uterine_cancer",
            # Full names
            "breast_cancer": "breast_cancer",
            "lung_cancer": "lung_cancer",
            "pancreatic_cancer": "pancreatic_cancer",
            "colorectal_cancer": "colorectal_cancer",
            "liver_cancer": "liver_cancer",
            "glioblastoma": "glioblastoma",
            "kidney_cancer": "kidney_cancer",
            "bladder_cancer": "bladder_cancer",
            "head_neck_cancer": "head_neck_cancer",
            "thyroid_cancer": "thyroid_cancer",
            "prostate_cancer": "prostate_cancer",
            "stomach_cancer": "stomach_cancer",
            "melanoma": "melanoma",
            "ovarian_cancer": "ovarian_cancer",
            "uterine_cancer": "uterine_cancer",
            "low_grade_glioma": "low_grade_glioma",
            "blood_cancer": "blood_cancer",
        }

        # Initialize vector store
        domain = self.domain_map.get(cancer_type, cancer_type)
        try:
            self.vector_store = create_vector_store(disease_domain=domain)
            self.logger.info(f"Vector store initialized for domain: {domain}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize vector store: {e}")
            self.vector_store = None

        # Initialize Claude client
        self.client = None
        if use_llm and HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.logger.info("Claude API initialized")
            else:
                self.logger.warning("ANTHROPIC_API_KEY not found - using lite mode")
                self.use_llm = False
        elif use_llm:
            self.logger.warning("anthropic not installed - using lite mode")
            self.use_llm = False

        # Initialize external data fetcher
        self.external_fetcher = None
        if use_external_apis:
            try:
                self.external_fetcher = get_external_fetcher(cancer_type=cancer_type)
                self.logger.info("External API fetcher initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize external fetcher: {e}")

    def _search_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search internal vector database."""
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.search(query, top_k=self.top_k)
            papers = []
            for r in results:
                papers.append({
                    "paper_title": r.metadata.get("paper_title", "Unknown"),
                    "section": r.metadata.get("section", "Unknown"),
                    "content": r.content,
                    "doi": r.metadata.get("doi", ""),
                    "year": r.metadata.get("year", ""),
                    "relevance_score": r.relevance_score,
                    "pmid": r.metadata.get("pmid", "")
                })
            return papers
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    async def _fetch_external_context(self, gene_symbol: str) -> Optional[GeneContext]:
        """Fetch external API data for a gene."""
        if not self.external_fetcher:
            return None

        try:
            context = await self.external_fetcher.get_gene_context(
                gene_symbol,
                include_interactions=True,
                include_pathways=True
            )
            return context
        except Exception as e:
            self.logger.warning(f"External API error for {gene_symbol}: {e}")
            return None

    def _build_enhanced_prompt(
        self,
        gene_symbol: str,
        direction: str,
        log2fc: float,
        papers: List[Dict[str, Any]],
        external_context: Optional[GeneContext]
    ) -> str:
        """Build prompt with both internal and external context."""
        dir_text = "upregulated" if direction == "up" else "downregulated"
        cancer_name = self.cancer_type.replace("_", " ")

        # Literature context
        lit_context = ""
        if papers:
            parts = []
            for i, paper in enumerate(papers[:5], 1):
                parts.append(
                    f"[{i}] {paper['paper_title']} ({paper.get('year', 'N/A')})\n"
                    f"Content: {paper['content'][:400]}..."
                )
            lit_context = "\n\n".join(parts)
        else:
            lit_context = "(No internal literature found)"

        # External database context
        ext_context = ""
        if external_context:
            ext_parts = []

            if external_context.cancer_role != "Unknown":
                ext_parts.append(f"• Cancer Role: {external_context.cancer_role}")

            if external_context.protein_function:
                ext_parts.append(f"• Protein Function: {external_context.protein_function[:300]}")

            if external_context.disease_associations:
                ext_parts.append(f"• Disease Associations: {', '.join(external_context.disease_associations[:5])}")

            if external_context.interaction_partners:
                ext_parts.append(f"• Key Interactors: {', '.join(external_context.interaction_partners[:5])}")

            if external_context.kegg_pathways:
                pathways = [p.get('name', '') for p in external_context.kegg_pathways[:3]]
                ext_parts.append(f"• KEGG Pathways: {', '.join(pathways)}")

            if external_context.reactome_pathways:
                pathways = [p.get('name', '') for p in external_context.reactome_pathways[:3]]
                ext_parts.append(f"• Reactome Pathways: {', '.join(pathways)}")

            if external_context.actionable:
                ext_parts.append("• Actionable Target: Yes (therapies available)")

            if external_context.clinical_evidence_count > 0:
                ext_parts.append(f"• Clinical Evidence: {external_context.clinical_evidence_count} variants documented")

            ext_context = "\n".join(ext_parts) if ext_parts else "(No external annotations)"
        else:
            ext_context = "(External APIs not queried)"

        prompt = f"""You are analyzing RNA-seq differential expression data.

Gene: {gene_symbol}
Expression: {dir_text} (log2FC = {log2fc:.2f})
Cancer Type: {cancer_name}

=== LITERATURE EVIDENCE (Internal Database) ===
{lit_context}

=== EXTERNAL DATABASE ANNOTATIONS ===
{ext_context}

=== TASK ===
Provide a concise interpretation (3-4 sentences) that:
1. Explains the biological significance of this gene's expression change
2. Connects to the cancer context using both literature and database evidence
3. Uses non-causal language ("associated with", "may indicate", "linked to")
4. References sources with [1], [2] for literature and database names (OncoKB, KEGG, etc.)
5. Notes any therapeutic implications if this is an actionable target

Write in Korean (한국어로 작성).

해석:"""

        return prompt

    async def interpret_gene_async(
        self,
        gene_symbol: str,
        gene_id: str,
        log2fc: float,
        direction: str,
        padj: float = 0.05
    ) -> EnhancedGeneInterpretation:
        """
        Generate enhanced interpretation for a gene (async version).
        """
        # Build search query
        cancer_name = self.cancer_type.replace("_", " ")
        query = f"{gene_symbol} {cancer_name} expression role function"

        # Parallel fetch: internal + external
        papers = self._search_papers(query)
        external_context = await self._fetch_external_context(gene_symbol)

        # Track sources used
        sources_used = []
        if papers:
            sources_used.append("Internal VectorDB")
        if external_context and external_context.sources_success:
            sources_used.extend(external_context.sources_success)

        # Generate interpretation
        interpretation = ""
        if self.client:
            prompt = self._build_enhanced_prompt(
                gene_symbol, direction, log2fc, papers, external_context
            )
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                interpretation = response.content[0].text.strip()
            except Exception as e:
                self.logger.error(f"Claude API error: {e}")
                interpretation = self._generate_lite_interpretation(
                    gene_symbol, direction, log2fc, papers, external_context
                )
        else:
            interpretation = self._generate_lite_interpretation(
                gene_symbol, direction, log2fc, papers, external_context
            )

        # Build citations
        citations = [
            {
                "paper_title": p['paper_title'],
                "section": p['section'],
                "doi": p.get('doi', ''),
                "year": p.get('year', ''),
                "relevance_score": p.get('relevance_score', 0),
                "content_preview": p['content'][:150] + "..."
            }
            for p in papers
        ]

        # Extract PMIDs
        pmids = self._extract_pmids(interpretation, papers)

        # Determine confidence
        confidence = self._calculate_confidence(papers, external_context)

        # Build pathways list
        pathways = []
        if external_context:
            for p in external_context.kegg_pathways[:5]:
                pathways.append({"source": "KEGG", "id": p.get("id", ""), "name": p.get("name", "")})
            for p in external_context.reactome_pathways[:5]:
                pathways.append({"source": "Reactome", "id": p.get("id", ""), "name": p.get("name", "")})

        return EnhancedGeneInterpretation(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            log2fc=log2fc,
            direction=direction,
            interpretation=interpretation,
            citations=citations,
            pmids=pmids,
            cancer_role=external_context.cancer_role if external_context else "Unknown",
            is_oncogene=external_context.is_oncogene if external_context else False,
            is_tsg=external_context.is_tsg if external_context else False,
            actionable=external_context.actionable if external_context else False,
            clinical_evidence_count=external_context.clinical_evidence_count if external_context else 0,
            protein_function=external_context.protein_function if external_context else "",
            subcellular_location=external_context.subcellular_location if external_context else [],
            disease_associations=external_context.disease_associations if external_context else [],
            interaction_partners=external_context.interaction_partners if external_context else [],
            pathways=pathways,
            confidence=confidence,
            sources_used=sources_used
        )

    def interpret_gene(
        self,
        gene_symbol: str,
        gene_id: str,
        log2fc: float,
        direction: str,
        padj: float = 0.05
    ) -> EnhancedGeneInterpretation:
        """Synchronous wrapper for interpret_gene_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.interpret_gene_async(gene_symbol, gene_id, log2fc, direction, padj)
        )

    async def interpret_genes_async(
        self,
        genes: List[Dict[str, Any]],
        max_genes: int = 20
    ) -> List[EnhancedGeneInterpretation]:
        """Interpret multiple genes asynchronously."""
        results = []
        genes_to_process = genes[:max_genes]

        self.logger.info(f"Interpreting {len(genes_to_process)} genes with enhanced context...")

        for i, gene in enumerate(genes_to_process):
            gene_symbol = gene.get("gene_symbol", "Unknown")
            self.logger.info(f"  [{i+1}/{len(genes_to_process)}] {gene_symbol}")

            interpretation = await self.interpret_gene_async(
                gene_symbol=gene_symbol,
                gene_id=gene.get("gene_id", ""),
                log2fc=gene.get("log2fc", gene.get("log2FC", 0)),
                direction=gene.get("direction", "up"),
                padj=gene.get("padj", 0.05)
            )
            results.append(interpretation)

        return results

    def interpret_genes(
        self,
        genes: List[Dict[str, Any]],
        max_genes: int = 20
    ) -> List[EnhancedGeneInterpretation]:
        """Synchronous wrapper for interpret_genes_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.interpret_genes_async(genes, max_genes))

    def _generate_lite_interpretation(
        self,
        gene_symbol: str,
        direction: str,
        log2fc: float,
        papers: List[Dict[str, Any]],
        external_context: Optional[GeneContext]
    ) -> str:
        """Generate interpretation without LLM."""
        dir_text = "상향조절" if direction == "up" else "하향조절"
        cancer_name = self.cancer_type.replace("_", " ")

        parts = [f"{gene_symbol}은(는) {cancer_name}에서 {dir_text}됨 (log2FC={log2fc:.2f})."]

        if external_context:
            if external_context.cancer_role != "Unknown":
                parts.append(f"암에서의 역할: {external_context.cancer_role}.")

            if external_context.protein_function:
                func = external_context.protein_function[:150]
                parts.append(f"기능: {func}...")

            if external_context.actionable:
                parts.append("치료 표적으로 활용 가능.")

        if papers:
            parts.append(f"관련 문헌 {len(papers)}편 발견.")
        else:
            parts.append("관련 문헌을 찾지 못함.")

        return " ".join(parts)

    def _extract_pmids(self, text: str, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract PMID references."""
        pmids = []
        pmids.extend(re.findall(r'PMID[:\s]*(\d+)', text, re.IGNORECASE))
        for p in papers:
            if p.get('pmid'):
                pmids.append(str(p['pmid']))
        return list(set(pmids))

    def _calculate_confidence(
        self,
        papers: List[Dict[str, Any]],
        external_context: Optional[GeneContext]
    ) -> str:
        """Calculate interpretation confidence."""
        score = 0

        # Literature evidence
        if len(papers) >= 3:
            score += 2
        elif len(papers) >= 1:
            score += 1

        if any(p.get('relevance_score', 0) > 0.7 for p in papers):
            score += 1

        # External database evidence
        if external_context:
            if external_context.cancer_role != "Unknown":
                score += 2
            if external_context.protein_function:
                score += 1
            if len(external_context.sources_success) >= 3:
                score += 1

        if score >= 5:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"

    async def close(self):
        """Close external fetcher connections."""
        if self.external_fetcher:
            await self.external_fetcher.close()

    def generate_summary(
        self,
        interpretations: List[EnhancedGeneInterpretation],
        include_external: bool = True
    ) -> str:
        """Generate summary report."""
        if not interpretations:
            return "No gene interpretations available."

        lines = []
        lines.append("=" * 80)
        lines.append("  Enhanced RAG Gene Interpretation Summary (Internal + External)")
        lines.append("=" * 80)
        lines.append("")

        # Stats
        high_conf = [i for i in interpretations if i.confidence == "high"]
        med_conf = [i for i in interpretations if i.confidence == "medium"]
        oncogenes = [i for i in interpretations if i.is_oncogene]
        tsgs = [i for i in interpretations if i.is_tsg]
        actionable = [i for i in interpretations if i.actionable]

        lines.append(f"Total genes: {len(interpretations)}")
        lines.append(f"  High confidence: {len(high_conf)}")
        lines.append(f"  Medium confidence: {len(med_conf)}")
        lines.append(f"  Oncogenes: {len(oncogenes)}")
        lines.append(f"  Tumor Suppressors: {len(tsgs)}")
        lines.append(f"  Actionable targets: {len(actionable)}")
        lines.append("")

        for interp in interpretations:
            arrow = "↑" if interp.direction == "up" else "↓"
            lines.append("-" * 80)
            lines.append(f"Gene: {interp.gene_symbol} ({arrow} {abs(interp.log2fc):.2f}x)")
            lines.append(f"Confidence: {interp.confidence.upper()} | Role: {interp.cancer_role}")

            if include_external:
                if interp.actionable:
                    lines.append("⭐ ACTIONABLE TARGET")
                if interp.sources_used:
                    lines.append(f"Sources: {', '.join(interp.sources_used)}")

            lines.append("")
            lines.append("Interpretation:")
            lines.append(interp.interpretation)

            if interp.pathways:
                pathway_names = [p['name'] for p in interp.pathways[:3]]
                lines.append(f"\nPathways: {', '.join(pathway_names)}")

            if interp.pmids:
                lines.append(f"PMIDs: {', '.join(interp.pmids)}")

            lines.append("")

        lines.append("=" * 80)
        lines.append("⚠️ Note: Interpretations combine AI analysis with multiple databases.")
        lines.append("   Always verify with original sources before clinical decisions.")
        lines.append("=" * 80)

        return "\n".join(lines)


def create_enhanced_interpreter(
    cancer_type: str = "breast_cancer",
    use_llm: bool = True,
    use_external_apis: bool = True
) -> EnhancedGeneInterpreter:
    """Factory function to create enhanced interpreter."""
    return EnhancedGeneInterpreter(
        cancer_type=cancer_type,
        use_llm=use_llm,
        use_external_apis=use_external_apis
    )


# Test
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        interpreter = create_enhanced_interpreter("breast_cancer")

        test_genes = [
            {"gene_symbol": "BRCA1", "gene_id": "ENSG00000012048", "log2fc": -1.5, "direction": "down"},
            {"gene_symbol": "TP53", "gene_id": "ENSG00000141510", "log2fc": -0.8, "direction": "down"},
        ]

        results = await interpreter.interpret_genes_async(test_genes)
        print(interpreter.generate_summary(results))

        await interpreter.close()

    asyncio.run(test())
