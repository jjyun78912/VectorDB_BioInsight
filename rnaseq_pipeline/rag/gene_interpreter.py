"""
RAG-based Gene Interpretation Module for RNA-seq Pipeline.

Uses vector search + Claude API to provide literature-backed
interpretations of DEG findings.

Supports:
- Claude API (Anthropic) - default
- Search-only mode (no LLM) - lite mode
"""
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.app.core.vector_store import create_vector_store

# Try to import anthropic
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class GeneInterpretation:
    """Interpretation result for a single gene."""
    gene_symbol: str
    gene_id: str
    log2fc: float
    direction: str
    interpretation: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"
    pmids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_symbol": self.gene_symbol,
            "gene_id": self.gene_id,
            "log2fc": self.log2fc,
            "direction": self.direction,
            "interpretation": self.interpretation,
            "citations": self.citations,
            "confidence": self.confidence,
            "pmids": self.pmids
        }


class GeneRAGInterpreter:
    """
    RAG-based interpreter for DEG results.

    Uses vector search to retrieve relevant papers, then uses Claude
    to generate literature-backed interpretations.
    """

    def __init__(
        self,
        cancer_type: str = "breast_cancer",
        top_k: int = 5,
        use_llm: bool = True,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the interpreter.

        Args:
            cancer_type: Cancer type for domain-specific search
            top_k: Number of papers to retrieve per query
            use_llm: Whether to use LLM for interpretation (False = lite mode)
            model: Claude model to use
        """
        self.cancer_type = cancer_type
        self.top_k = top_k
        self.use_llm = use_llm
        self.model = model
        self.logger = logging.getLogger("gene_interpreter")

        # Map cancer types to disease domains (ChromaDB collection names)
        # Includes TCGA codes, full names, and collection prefixes
        self.domain_map = {
            # TCGA codes → ChromaDB collection domains
            "LUAD": "lung_cancer",
            "LUSC": "lung_cancer",
            "BRCA": "breast_cancer",
            "COAD": "colorectal_cancer",
            "READ": "colorectal_cancer",
            "PAAD": "pancreatic_cancer",
            "LIHC": "liver_cancer",
            "GBM": "glioblastoma",
            "LGG": "low_grade_glioma",
            "KIRC": "kidney_cancer",
            "KIRP": "kidney_cancer",
            "KICH": "kidney_cancer",
            "BLCA": "bladder_cancer",
            "HNSC": "head_neck_cancer",
            "THCA": "thyroid_cancer",
            "PRAD": "prostate_cancer",
            "STAD": "stomach_cancer",
            "SKCM": "melanoma",
            "OV": "ovarian_cancer",
            "UCEC": "uterine_cancer",
            # Full names
            "breast_cancer": "breast_cancer",
            "lung_cancer": "lung_cancer",
            "pancreatic_cancer": "pancreatic_cancer",
            "colorectal_cancer": "colorectal_cancer",
            "liver_cancer": "liver_cancer",
            "glioblastoma": "glioblastoma",
            "blood_cancer": "blood_cancer",
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
                self.logger.info("Claude API client initialized")
            else:
                self.logger.warning("ANTHROPIC_API_KEY not found - using lite mode")
                self.use_llm = False
        elif use_llm:
            self.logger.warning("anthropic package not installed - using lite mode")
            self.use_llm = False

    def _build_query(self, gene_symbol: str, direction: str, log2fc: float) -> str:
        """Build a search query for a gene."""
        dir_text = "upregulated" if direction == "up" else "downregulated"
        cancer_name = self.cancer_type.replace("_", " ")

        return (
            f"{gene_symbol} {cancer_name} expression role function"
        )

    def _search_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant papers."""
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

    def _generate_interpretation(
        self,
        gene_symbol: str,
        direction: str,
        log2fc: float,
        papers: List[Dict[str, Any]],
        cancer_type: str
    ) -> str:
        """Generate interpretation using Claude API."""
        if not self.client or not papers:
            return self._generate_lite_interpretation(gene_symbol, direction, log2fc, papers)

        # Build context from papers
        context_parts = []
        for i, paper in enumerate(papers[:5], 1):
            context_parts.append(
                f"[{i}] {paper['paper_title']} ({paper.get('year', 'N/A')})\n"
                f"Section: {paper['section']}\n"
                f"Content: {paper['content'][:500]}..."
            )
        context = "\n\n".join(context_parts)

        dir_text = "upregulated" if direction == "up" else "downregulated"
        cancer_name = cancer_type.replace("_", " ")

        prompt = f"""Based on the following scientific literature excerpts, provide a concise interpretation
of {gene_symbol} being {dir_text} (log2FC={log2fc:.2f}) in {cancer_name}.

Literature Context:
{context}

Guidelines:
1. Be concise (2-3 sentences max)
2. Focus on the functional significance in {cancer_name}
3. Use non-causal language ("is associated with", "may indicate", "has been linked to")
4. Reference the sources using [1], [2], etc.
5. If the literature doesn't directly address this finding, say so clearly
6. Include any relevant PMIDs if mentioned in the sources

Interpretation:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            return self._generate_lite_interpretation(gene_symbol, direction, log2fc, papers)

    def _generate_lite_interpretation(
        self,
        gene_symbol: str,
        direction: str,
        log2fc: float,
        papers: List[Dict[str, Any]]
    ) -> str:
        """Generate interpretation without LLM (lite mode)."""
        dir_text = "상향조절" if direction == "up" else "하향조절"
        cancer_name = self.cancer_type.replace("_", " ")

        if not papers:
            return f"{gene_symbol}은(는) {dir_text}됨 (log2FC={log2fc:.2f}). 관련 문헌을 찾지 못했습니다."

        paper_titles = [p['paper_title'][:50] for p in papers[:3]]
        return (
            f"{gene_symbol}은(는) {cancer_name}에서 {dir_text}됨 (log2FC={log2fc:.2f}). "
            f"관련 문헌 {len(papers)}편 발견: {'; '.join(paper_titles)}..."
        )

    def _extract_pmids(self, text: str, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract PMID references."""
        pmids = []
        # From text
        pmids.extend(re.findall(r'PMID[:\s]*(\d+)', text, re.IGNORECASE))
        # From papers
        for p in papers:
            if p.get('pmid'):
                pmids.append(str(p['pmid']))
        return list(set(pmids))

    def interpret_gene(
        self,
        gene_symbol: str,
        gene_id: str,
        log2fc: float,
        direction: str,
        padj: float = 0.05
    ) -> GeneInterpretation:
        """
        Generate interpretation for a single gene.

        Args:
            gene_symbol: Gene symbol (e.g., BRCA2)
            gene_id: Ensembl gene ID
            log2fc: Log2 fold change
            direction: "up" or "down"
            padj: Adjusted p-value

        Returns:
            GeneInterpretation object
        """
        # Search for relevant papers
        query = self._build_query(gene_symbol, direction, log2fc)
        papers = self._search_papers(query)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            gene_symbol, direction, log2fc, papers, self.cancer_type
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
        if len(papers) >= 3 and any(p['relevance_score'] > 0.7 for p in papers):
            confidence = "high"
        elif len(papers) >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        return GeneInterpretation(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            log2fc=log2fc,
            direction=direction,
            interpretation=interpretation,
            citations=citations,
            confidence=confidence,
            pmids=pmids
        )

    def interpret_genes(
        self,
        genes: List[Dict[str, Any]],
        max_genes: int = 20
    ) -> List[GeneInterpretation]:
        """
        Generate interpretations for multiple genes.

        Args:
            genes: List of gene dictionaries with keys:
                   gene_symbol, gene_id, log2fc, direction, padj
            max_genes: Maximum number of genes to interpret

        Returns:
            List of GeneInterpretation objects
        """
        results = []
        genes_to_process = genes[:max_genes]

        self.logger.info(f"Interpreting {len(genes_to_process)} genes...")

        for i, gene in enumerate(genes_to_process):
            self.logger.info(f"  [{i+1}/{len(genes_to_process)}] {gene.get('gene_symbol', 'Unknown')}")

            interpretation = self.interpret_gene(
                gene_symbol=gene.get("gene_symbol", "Unknown"),
                gene_id=gene.get("gene_id", ""),
                log2fc=gene.get("log2fc", gene.get("log2FC", 0)),
                direction=gene.get("direction", "up"),
                padj=gene.get("padj", 0.05)
            )
            results.append(interpretation)

        return results

    def generate_summary(
        self,
        interpretations: List[GeneInterpretation],
        include_citations: bool = True
    ) -> str:
        """
        Generate a summary report of all interpretations.

        Args:
            interpretations: List of GeneInterpretation objects
            include_citations: Whether to include citation details

        Returns:
            Formatted summary string
        """
        if not interpretations:
            return "No gene interpretations available."

        lines = []
        lines.append("=" * 70)
        lines.append("  RAG-Based Gene Interpretation Summary")
        lines.append("=" * 70)
        lines.append("")

        # Group by confidence
        high_conf = [i for i in interpretations if i.confidence == "high"]
        med_conf = [i for i in interpretations if i.confidence == "medium"]
        low_conf = [i for i in interpretations if i.confidence == "low"]

        lines.append(f"Total genes interpreted: {len(interpretations)}")
        lines.append(f"  High confidence: {len(high_conf)}")
        lines.append(f"  Medium confidence: {len(med_conf)}")
        lines.append(f"  Low confidence: {len(low_conf)}")
        lines.append("")

        # Detailed interpretations
        for interp in interpretations:
            arrow = "↑" if interp.direction == "up" else "↓"
            lines.append("-" * 70)
            lines.append(f"Gene: {interp.gene_symbol} ({arrow} {abs(interp.log2fc):.2f}x)")
            lines.append(f"Confidence: {interp.confidence.upper()}")
            lines.append("")
            lines.append("Interpretation:")
            lines.append(interp.interpretation)

            if include_citations and interp.citations:
                lines.append("")
                lines.append("References:")
                for j, cit in enumerate(interp.citations[:3], 1):
                    year = f" ({cit['year']})" if cit.get('year') else ""
                    lines.append(f"  [{j}] {cit['paper_title']}{year}")

            if interp.pmids:
                lines.append(f"PMIDs: {', '.join(interp.pmids)}")

            lines.append("")

        lines.append("=" * 70)
        lines.append("⚠️ Note: Interpretations are AI-generated based on indexed literature.")
        lines.append("   Always verify findings with original sources.")
        lines.append("=" * 70)

        return "\n".join(lines)


def create_interpreter(cancer_type: str = "breast_cancer", use_llm: bool = True) -> GeneRAGInterpreter:
    """Factory function to create a gene interpreter."""
    return GeneRAGInterpreter(cancer_type=cancer_type, use_llm=use_llm)


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test interpretation
    interpreter = create_interpreter("breast_cancer")

    test_genes = [
        {"gene_symbol": "BRCA2", "gene_id": "ENSG00000139618", "log2fc": 1.3, "direction": "up"},
    ]

    results = interpreter.interpret_genes(test_genes)
    print(interpreter.generate_summary(results))
